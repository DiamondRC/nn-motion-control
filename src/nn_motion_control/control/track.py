"""
Closed-loop tracking evaluation for a trained controller.

Loads the controller checkpoint and its frozen plant, seeds from quiescent
relaxation holds (the operating regime), rolls the config's reference
trajectory through the loop, logs per-axis position + velocity RMS over the
trained horizon and an error-vs-step ladder (which exposes compounding past
that horizon), then draws a position-vs-step plot, a 3D animated trajectory
trace, and a native interactive 3D window (rotatable).
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import numpy as np
import torch

from nn_motion_control.control.closed_loop import (
    tracking_percentiles,
    zero_policy,
)
from nn_motion_control.control.config import Controller, ControllerConfig
from nn_motion_control.core.config import RunConfiguration
from nn_motion_control.core.progress import spinner
from nn_motion_control.eval.horizon import position_resolution
from nn_motion_control.eval.tables import (
    PERCENTILE_LABELS,
    PERCENTILES,
    log_axis_percentiles,
    log_table,
    render_table,
)
from nn_motion_control.plant.plant import (
    Plant,
    RolloutLayout,
    rollout_splits_from_config,
)
from nn_motion_control.training.control_run import (
    make_reference_gen,
    shape_spec,
)

logger = logging.getLogger(os.path.basename(__file__))

# Error-plot hardware floor: sensor/DAC noise below this is not controller
# error, so +/- this band is shaded rather than chased.
HARDWARE_FLOOR_NM = 20.0

# Re-anchor error ladder: segment counts sampled (pre-anchor peak and
# post-anchor point) so stationarity is visible across the whole setpoint.
REANCHOR_SAMPLE_MULTIPLES = (2, 4, 6, 8, 10)

# Animation GIF cosmetics: frame budget, playback rate, inter-frame delay.
GIF_TARGET_FRAMES = 150
GIF_FPS = 12
GIF_FRAME_INTERVAL_MS = 70


def _to_nm_np(t: torch.Tensor, to_nm: np.ndarray) -> np.ndarray:
    """
    Move a position/error tensor to a host array, scaled counts to nm
    per axis.
    """

    return t.cpu().numpy() * to_nm


def _agg_pyplot():
    """Matplotlib pyplot on the Agg backend, for file output only."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _log_metrics(
    name: str,
    predicted: torch.Tensor,
    reference: torch.Tensor,
    axes: list[str],
    to_nm: np.ndarray,
    unit: str,
) -> None:
    """
    Log a per-axis five-number (P50/P95/P99/P99.9/max) RMS tracking
    summary in unit.
    """

    pc = tracking_percentiles(
        predicted, reference, PERCENTILES
    )  # {pct: [A]} in counts
    values = [
        [float(pc[p][a]) * to_nm[a] for p in PERCENTILES]
        for a in range(len(axes))
    ]
    log_axis_percentiles(logger, name, unit, axes, values)


def _log_error_ladder(
    name: str,
    positions: torch.Tensor,
    reference: torch.Tensor,
    axes: list[str],
    to_nm: np.ndarray,
    steps: list[int],
) -> None:
    """
    Log the full five-number |error| (P50/P95/P99/P99.9/max, nm) per axis
    at a ladder of steps -- one block per axis, so the tail is never
    hidden behind a weaker summary.
    """

    scale = torch.as_tensor(
        to_nm, device=positions.device, dtype=positions.dtype
    )
    err = (positions - reference).abs() * scale  # [B, H, A] nm
    q = torch.tensor(
        [p / 100.0 for p in PERCENTILES], device=err.device, dtype=err.dtype
    )
    logger.info("")
    logger.info("%s |error|:", name)

    for a, axis in enumerate(axes):
        rows = [
            [float(v) for v in torch.quantile(err[:, k, a], q)] for k in steps
        ]
        logger.info("")
        logger.info("  [%s]", axis)
        log_table(
            logger,
            render_table(
                [str(k) for k in steps],
                list(PERCENTILE_LABELS),
                rows,
                corner_label="step",
                unit="nm",
                value_fmt="{:.1f}",
            ),
        )


def _show_trajectory(pos, ref, axes, to_nm, config):
    """
    Open a native, rotatable 3D trajectory window (no web/HTML).

    Switches Matplotlib to an interactive GUI backend and blocks on
    plt.show so the path can be inspected from any angle, the controller
    line is coloured by step. It is skipped with a note if no display /
    GUI backend is available.
    """

    import matplotlib
    import matplotlib.pyplot as plt

    for backend in ("QtAgg", "TkAgg"):
        try:
            matplotlib.use(backend, force=True)
            plt.switch_backend(backend)
            break
        except (
            Exception
        ):  # pragma: no cover - depends on the host's GUI toolkit
            continue
    else:
        logger.info(
            "Interactive 3D skipped: no GUI backend (set DISPLAY / install Qt)."
        )

        return

    p = _to_nm_np(pos, to_nm)  # [H, A] nm
    r = _to_nm_np(ref, to_nm)
    step = np.arange(p.shape[0])
    fig = plt.figure(figsize=(8, 7))
    ax: Any  # 3D axes take z coordinates the 2D matplotlib stub does not type
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(
        r[:, 0], r[:, 1], r[:, 2], "--", color="0.6", lw=1.3, label="reference"
    )
    pts = ax.scatter(p[:, 0], p[:, 1], p[:, 2], c=step, cmap="viridis", s=6)
    ax.plot(p[:, 0], p[:, 1], p[:, 2], "-", color="C3", lw=1.0, alpha=0.5)
    fig.colorbar(pts, ax=ax, label="step", shrink=0.6)
    ax.set_xlabel(f"{axes[0]} (nm)")
    ax.set_ylabel(f"{axes[1]} (nm)")
    ax.set_zlabel(f"{axes[2]} (nm)")
    ax.set_title(f"Tracking: {config.model_name} (free-run {p.shape[0]} steps)")
    ax.legend(loc="upper right", fontsize=9)
    logger.info("Opening interactive 3D window (close it to continue) ...")
    plt.show()


def run_track(
    controller_cfg_pth: str,
    device: str = "cpu",
    samples: int = 4096,
    animate: bool = True,
    viz_steps: int = 512,
    show: bool = True,
    reanchor: int = 0,
    shape: str = "config",
    seed: int = 42,
) -> float:
    """
    Evaluate a trained controller closed-loop: log metrics and draw motion
    plots.

    The RMS summary and position-vs-step plot use the trained horizon, the
    error-vs-step ladder, 3D animation and (with show) an interactive 3D
    window free-run viz_steps long so compounding and the full spatial
    arc/helix are visible. The signed-residual plot (_error.svg) is the
    error-magnified view: it plots only the tracking error per axis in nm,
    resolved at any trajectory scale.

    reanchor > 0 corrects the believed position onto the reference every
    that-many steps -- the position update real per-step feedback provides
    -- so the plant's own autoregressive drift cannot accumulate. This is
    the honest long-setpoint metric: a bounded, stationary trend means the
    controller tracks arbitrarily long trajectories, whereas the raw
    (reanchor=0) run conflates control error with simulator drift.

    shape selects the reference trajectory ('config' keeps the controller
    config's own reference; the other named shapes come from
    'control_run.shape_spec'); seed makes the randomised shapes
    reproducible.
    """

    start = time.perf_counter()
    config = ControllerConfig(controller_cfg_pth)
    logger.info(
        "Loading controller '%s' and its plant (device=%s) ...",
        config.model_name,
        device,
    )
    plant_cfg = RunConfiguration(config.plant_config_path)
    plant = Plant.from_checkpoint(plant_cfg, config.plant_checkpoint, device)
    policy = Controller.from_checkpoint(config.save_path, device).policy
    layout = RolloutLayout.from_config(plant_cfg)
    train = config.training
    horizon = int(train.get("horizon", 32))
    to_nm = position_resolution(plant_cfg) * 1e9  # counts -> nm, per axis
    axes = config.axes

    logger.info("Loading dataset and sampling %d quiescent seeds ...", samples)
    loaders = rollout_splits_from_config(
        plant_cfg,
        layout,
        max_horizon=horizon,
        batch_size=samples,
        seed=1,
        device=device,
        val_start_stride=8,
        quiescent_seed=train.get("seed_quiescence"),
    )
    # A --shape override selects a named trajectory; 'config' keeps the
    # controller config's own reference block (backward compatible). The
    # seeded generator makes the randomised shapes reproducible.
    ref_spec = (
        train.get("reference", {}) if shape == "config" else shape_spec(shape)
    )
    ref_gen = make_reference_gen(ref_spec, device)
    ref_generator = torch.Generator(device=device).manual_seed(seed)
    viz_h = max(int(viz_steps), horizon)
    warmup = next(iter(loaders.val_loader))[0].to(device)
    origin, _ = plant.seed_state(warmup)
    reference, ref_v = ref_gen(origin, viz_h, ref_generator)

    rollout_msg = (
        f"Rolling out {warmup.shape[0]} seeds x {viz_h} steps "
        f"(reanchor={reanchor})"
    )
    with spinner(rollout_msg), torch.no_grad():
        pos_ctrl, _ = plant.closed_loop_rollout(
            warmup,
            reference,
            policy,
            viz_h,
            reference_velocity=ref_v,
            reanchor_every=reanchor,
        )
        pos_zero, _ = plant.closed_loop_rollout(
            warmup, reference, zero_policy, viz_h, reference_velocity=ref_v
        )

    anchor_note = (
        f"; re-anchored every {reanchor} (feedback-corrected)"
        if reanchor
        else ""
    )
    logger.info(
        "Closed-loop tracking: %s (%d quiescent seeds, %.0f Hz; "
        "trained horizon %d%s)",
        config.model_name,
        warmup.shape[0],
        config.servo_rate_hz,
        horizon,
        anchor_note,
    )
    # In-horizon RMS (the training regime) plus its velocity, for the summary.
    hz = slice(0, horizon)
    _log_metrics(
        f"trained controller (RMS over trained horizon {horizon})",
        pos_ctrl[:, hz],
        reference[:, hz],
        axes,
        to_nm,
        "nm",
    )
    # Per-step realised velocity vs demanded, over the in-horizon window
    # (stay in bounds when viz_h == horizon: diffs span steps 1..horizon-1).
    vel_ctrl = pos_ctrl[:, 1:horizon] - pos_ctrl[:, : horizon - 1]
    _log_metrics(
        f"velocity (RMS over trained horizon {horizon})",
        vel_ctrl,
        ref_v[:, 1:horizon],
        axes,
        to_nm,
        "nm/step",
    )
    # Error vs step -- a horizon-averaged RMS hides that error compounds
    # past the trained horizon. When re-anchoring, sample the pre-anchor
    # peak (deepest drift in a segment) and post-anchor point (freshly
    # corrected) across the whole setpoint, so stationarity -- same
    # excursion early vs late -- is directly visible.
    if reanchor:
        cand = [1, reanchor - 1, reanchor]

        for m in REANCHOR_SAMPLE_MULTIPLES:
            cand += [reanchor * m - 1, reanchor * m]
        cand.append(viz_h - 1)
    else:
        cand = [
            1,
            max(1, horizon // 2),
            horizon,
            horizon * 2,
            horizon * 4,
            viz_h - 1,
        ]
    steps = sorted({s for s in cand if 1 <= s < viz_h})
    _log_error_ladder(
        "trained controller error vs step",
        pos_ctrl,
        reference,
        axes,
        to_nm,
        steps,
    )
    _log_error_ladder(
        "zero-policy baseline error vs step",
        pos_zero,
        reference,
        axes,
        to_nm,
        steps,
    )

    # A representative rollout for the plots: the one at the median
    # in-horizon RMS.
    per_sample = (
        (pos_ctrl[:, hz] - reference[:, hz]).pow(2).mean(dim=(1, 2)).sqrt()
    )
    idx = int(per_sample.argsort()[per_sample.numel() // 2])

    os.makedirs(config.eval_dir, exist_ok=True)
    base = os.path.join(config.eval_dir, config.model_name)
    render_msg = "Rendering plots" + (" and animation" if animate else "")
    with spinner(render_msg):
        # The 2D panel stays at the trained horizon (clean tracking detail).
        _plot_motion(
            pos_ctrl[:, hz],
            reference[:, hz],
            idx,
            axes,
            to_nm,
            config,
            f"{base}_track.svg",
        )
        # The error-magnified view: signed residual (controller -
        # reference) per axis over the full setpoint, in nm -- resolves
        # single-digit-nm tracking regardless of the trajectory's physical
        # scale, with re-anchor points marked.
        _plot_error(
            pos_ctrl[idx],
            reference[idx],
            axes,
            to_nm,
            config,
            reanchor,
            f"{base}_error.svg",
        )
        if animate:
            # Spans the full viz horizon so the spatial arc/helix develops.
            _animate_trajectory(
                pos_ctrl,
                reference,
                idx,
                axes,
                to_nm,
                config,
                f"{base}_track.gif",
            )

    elapsed = time.perf_counter() - start
    logger.info("Track complete in %.1fs", elapsed)
    # The interactive window blocks (user time), so open it last -- after
    # the timer and after every file is written.
    if show and len(axes) >= 3:
        _show_trajectory(pos_ctrl[idx], reference[idx], axes, to_nm, config)

    return elapsed


def _plot_motion(pos, ref, idx, axes, to_nm, config, path):
    """
    Clean stacked position-vs-step panels, one per axis, controller vs
    reference (nm).
    """

    plt = _agg_pyplot()

    p = _to_nm_np(pos[idx], to_nm)  # [H, A] nm
    r = _to_nm_np(ref[idx], to_nm)
    n_steps, n_ax = p.shape
    step = np.arange(n_steps)
    dt_ms = 1000.0 / config.servo_rate_hz

    fig, axs = plt.subplots(
        n_ax, 1, sharex=True, figsize=(10, 2.2 * n_ax + 1.0), squeeze=False
    )

    for a, axis in enumerate(axes):
        ax = axs[a, 0]
        ax.plot(step, r[:, a], "--", color="0.55", lw=1.5, label="reference")
        ax.plot(step, p[:, a], "-", color=f"C{a}", lw=1.7, label="controller")
        ax.set_ylabel(f"{axis} (nm)")
        ax.grid(alpha=0.3)
        if a == 0:
            ax.legend(loc="upper right", fontsize=9)
            top = ax.secondary_xaxis(
                "top", functions=(lambda s: s * dt_ms, lambda t: t / dt_ms)
            )
            top.set_xlabel("time (ms)")
    axs[-1, 0].set_xlabel("step")
    fig.suptitle(
        f"Closed-loop position tracking: {config.model_name} "
        f"(sample {idx}, deterministic plant)"
    )
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    logger.info("")
    logger.info("Saved motion plot: %s", path)


def _plot_error(pos, ref, axes, to_nm, config, reanchor, path):
    """
    Error-magnified view: signed residual (controller - reference) per
    axis vs step, nm.

    Plots only the tracking error, so single-digit-nm deviation is
    resolved at any trajectory scale. Re-anchor points (where feedback
    resets the state) are marked, so the per-segment drift-and-snap shows,
    a shaded +/-20 nm band marks the hardware floor. pos/ref are a single
    rollout [H, A].
    """

    plt = _agg_pyplot()

    err = _to_nm_np(pos - ref, to_nm)  # [H, A] nm, signed
    n_steps, n_ax = err.shape
    step = np.arange(n_steps)
    dt_ms = 1000.0 / config.servo_rate_hz
    floor_label = f"+/-{HARDWARE_FLOOR_NM:.0f} nm"

    fig, axs = plt.subplots(
        n_ax, 1, sharex=True, figsize=(11, 2.1 * n_ax + 1.0), squeeze=False
    )

    for a, axis in enumerate(axes):
        ax = axs[a, 0]
        ax.axhline(0.0, color="0.6", lw=0.8)
        ax.axhspan(
            -HARDWARE_FLOOR_NM,
            HARDWARE_FLOOR_NM,
            color="0.85",
            alpha=0.5,
            zorder=0,
            label=floor_label,
        )
        ax.plot(step, err[:, a], "-", color=f"C{a}", lw=1.2)
        if reanchor:
            for x in range(reanchor, n_steps, reanchor):
                ax.axvline(x, color="k", ls=":", lw=0.6, alpha=0.5)
        peak = np.abs(err[:, a]).max()
        ax.set_ylabel(f"{axis} error (nm)")
        ax.set_title(
            f"{axis}: peak |err| {peak:.1f} nm", fontsize=9, loc="left"
        )
        ax.grid(alpha=0.3)
        if a == 0:
            ax.legend(loc="upper right", fontsize=8)
            top = ax.secondary_xaxis(
                "top", functions=(lambda s: s * dt_ms, lambda t: t / dt_ms)
            )
            top.set_xlabel("time (ms)")
    axs[-1, 0].set_xlabel("step")
    anchor_note = f", re-anchor every {reanchor}" if reanchor else ", free-run"
    fig.suptitle(
        f"Tracking error (signed residual): {config.model_name}{anchor_note}"
    )
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    logger.info("Saved error plot: %s", path)


def _animate_trajectory(pos, ref, idx, axes, to_nm, config, path):
    """
    Animate the tracked trajectory filling in step by step, 3D when there
    are >=3 axes (the natural helix: xy circles while z ramps), else a 2D
    trace. Saved as a GIF.
    """

    plt = _agg_pyplot()
    from matplotlib.animation import FuncAnimation, PillowWriter

    p = _to_nm_np(pos[idx], to_nm)  # [H, A] nm
    r = _to_nm_np(ref[idx], to_nm)
    n_steps, n_ax = p.shape
    dims = [0, 1, 2] if n_ax >= 3 else [0, 1] if n_ax >= 2 else [0]
    is_3d = len(dims) == 3

    fig = plt.figure(figsize=(7, 6.2))
    # 3D axes/lines expose set_zlim / set_3d_properties, untyped in matplotlib.
    ax: Any
    if is_3d:
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(
            r[:, 0],
            r[:, 1],
            r[:, 2],
            "--",
            color="0.6",
            lw=1.3,
            label="reference",
        )
        ax.set_zlabel(f"{axes[2]} (nm)")
        (trail,) = ax.plot(
            [], [], [], "-", color="C3", lw=1.8, label="controller"
        )
        (head,) = ax.plot([], [], [], "o", color="C3", ms=7)
    else:
        ax = fig.add_subplot(111)
        xi, yi = dims[0], dims[1] if len(dims) > 1 else dims[0]
        ax.plot(
            r[:, xi], r[:, yi], "--", color="0.6", lw=1.3, label="reference"
        )
        ax.set_aspect("equal", "box")
        (trail,) = ax.plot([], [], "-", color="C3", lw=1.8, label="controller")
        (head,) = ax.plot([], [], "o", color="C3", ms=7)

    def lim(col):
        lo, hi = (
            min(r[:, col].min(), p[:, col].min()),
            max(r[:, col].max(), p[:, col].max()),
        )
        pad = 0.08 * (hi - lo) + 1e-6

        return lo - pad, hi + pad

    ax.set_xlim(*lim(dims[0]))
    ax.set_ylim(*lim(dims[1] if len(dims) > 1 else dims[0]))
    ax.set_xlabel(f"{axes[dims[0]]} (nm)")
    ax.set_ylabel(f"{axes[dims[1]]} (nm)" if len(dims) > 1 else "step")
    if is_3d:
        ax.set_zlim(*lim(2))
        # A balanced cube (not the raw data ranges, which collapse a
        # small-motion axis to a near-flat line): each axis fills its own
        # limits so the path reads as 3D.
        ax.set_box_aspect((1.0, 1.0, 0.85))
    ax.set_title(f"Tracking trajectory: {config.model_name}")
    ax.legend(loc="upper right", fontsize=9)

    # Subsample frames so the GIF stays light regardless of the viz horizon.
    stride = max(1, n_steps // GIF_TARGET_FRAMES)
    frames = list(range(0, n_steps, stride)) + [n_steps - 1]

    def update(k):
        j = k + 1
        if is_3d:
            trail.set_data(p[:j, 0], p[:j, 1])
            trail.set_3d_properties(p[:j, 2])
            head.set_data([p[k, 0]], [p[k, 1]])
            head.set_3d_properties([p[k, 2]])
        else:
            trail.set_data(p[:j, dims[0]], p[:j, dims[-1]])
            head.set_data([p[k, dims[0]]], [p[k, dims[-1]]])

        return trail, head

    anim = FuncAnimation(
        fig, update, frames=frames, interval=GIF_FRAME_INTERVAL_MS, blit=False
    )
    try:
        anim.save(path, writer=PillowWriter(fps=GIF_FPS))
        logger.info("Saved motion animation: %s", path)
    except Exception as exc:  # pragma: no cover - writer availability varies
        logger.info("Animation skipped (%s)", repr(exc)[:120])
    plt.close(fig)
