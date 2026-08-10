"""
Error-vs-horizon evaluation for a forward-dynamics plant.

Free-run a frozen plant from held-out seeds and measure how the position prediction
drifts from the true trajectory as the rollout grows. The curve (worst-case error vs
horizon) is the rollout gate: later milestones read off the horizon they can trust. It
works on any delta-position checkpoint, so it also baselines the one-step plant.
"""

from __future__ import annotations

import logging
import os

import numpy as np
import torch

from nn_motion_control.core.config import RunConfiguration
from nn_motion_control.data import build_rollout_splits
from nn_motion_control.eval.metrics import DEFAULT_GATE, ChannelMetrics, channel_metrics
from nn_motion_control.plant.plant import Plant, RolloutLayout

logger = logging.getLogger(os.path.basename(__file__))


def _sampled_batches(loader, max_batches: int | None):
    """
    Yield up to ``max_batches`` batches spread evenly across the whole loader.

    The rollout loaders iterate window starts in timeline order, so the first N batches
    are one contiguous segment that can be entirely quiet (motion at the noise floor).
    Striding across the full loader instead keeps the sample representative of the whole
    motion range. ``max_batches is None`` yields every batch.
    """

    if max_batches is None:
        yield from loader
        return
    stride = max(1, len(loader) // max_batches)
    kept = 0
    for i, batch in enumerate(loader):
        if i % stride != 0:
            continue
        if kept >= max_batches:
            break
        yield batch
        kept += 1


@torch.no_grad()
def collect_horizon_errors(
    plant: Plant,
    loader,
    horizon: int,
    max_batches: int | None = None,
) -> torch.Tensor:
    """
    Free-run the plant and return absolute position error per step, shape [N, H, A].

    Error is returned in physical units (the position channel's own units): the z-score
    mean cancels in the difference, so it is the normalised error scaled by the position
    std. At most max_batches batches are read, sampled across the whole test timeline.
    """

    plant.model.eval()
    pos_std = plant.pos_std  # [A]
    chunks: list[torch.Tensor] = []
    for warmup, dac_future, gt_pos in _sampled_batches(loader, max_batches):
        preds = plant.roll_forward(warmup, dac_future[:, :horizon], horizon)
        gt = gt_pos[:, :horizon].float()
        chunks.append(((preds - gt) * pos_std).abs().cpu())
    if not chunks:
        raise ValueError("Loader produced no batches for horizon evaluation")
    return torch.cat(chunks)  # [N, H, A]


@torch.no_grad()
def collect_horizon_trajectories(
    plant: Plant,
    loader,
    horizon: int,
    max_batches: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Free-run the plant and return position trajectories for the per-channel table.

    Returns ``(pred, truth, anchor)``: predicted and true positions ``[N, H, A]`` and
    the shared last-warmup position ``[N, A]`` both trajectories start from. All are in
    the position channel's physical units with the (cancelling) z-score mean removed, so
    a per-step difference is a clean physical velocity. At most ``max_batches`` are
    read, keeping the eval to a representative sample.
    """

    plant.model.eval()
    pos_std = plant.pos_std  # [A]
    pos_cols = plant.layout.pos_cols
    preds_all: list[torch.Tensor] = []
    truth_all: list[torch.Tensor] = []
    anchor_all: list[torch.Tensor] = []
    for warmup, dac_future, gt_pos in _sampled_batches(loader, max_batches):
        preds = plant.roll_forward(warmup, dac_future[:, :horizon], horizon)
        gt = gt_pos[:, :horizon].float()
        anchor = warmup.float()[:, pos_cols, -1]  # [B, A], normalised last position
        preds_all.append((preds * pos_std).cpu())
        truth_all.append((gt * pos_std).cpu())
        anchor_all.append((anchor * pos_std).cpu())
    if not preds_all:
        raise ValueError("Loader produced no batches for horizon evaluation")
    return torch.cat(preds_all), torch.cat(truth_all), torch.cat(anchor_all)


def horizon_channel_table(
    preds: torch.Tensor,
    truth: torch.Tensor,
    anchor: torch.Tensor,
    axes: list[str],
    steps: list[int],
    to_nm: np.ndarray,
) -> dict[int, list[ChannelMetrics]]:
    """
    Per-channel percentile metrics for position and derived velocity, per horizon step.

    ``preds``/``truth`` are ``[N, H, A]`` physical positions sharing the ``[N, A]``
    ``anchor`` start; velocity is their per-step difference (the plant's own derived
    velocity). Returns ``{step: [ChannelMetrics, ...]}`` with two rows per axis (pos
    then vel), in nm and nm/step. At step 1 the free-run has not diverged, so the
    velocity row mirrors the position row (the one-step floor); the two separate as the
    horizon grows.
    """

    prev_pred = torch.cat([anchor.unsqueeze(1), preds[:, :-1, :]], dim=1)
    prev_true = torch.cat([anchor.unsqueeze(1), truth[:, :-1, :]], dim=1)
    scale = torch.as_tensor(to_nm, dtype=preds.dtype)  # [A], units/count -> nm
    p_nm, t_nm = preds * scale, truth * scale
    vp_nm, vt_nm = (preds - prev_pred) * scale, (truth - prev_true) * scale

    table: dict[int, list[ChannelMetrics]] = {}
    for step in steps:
        k = step - 1
        rows: list[ChannelMetrics] = []
        for a, axis in enumerate(axes):
            rows.append(
                channel_metrics(
                    f"{axis} pos", p_nm[:, k, a].numpy(), t_nm[:, k, a].numpy()
                )
            )
            rows.append(
                channel_metrics(
                    f"{axis} vel", vp_nm[:, k, a].numpy(), vt_nm[:, k, a].numpy()
                )
            )
        table[step] = rows
    return table


def _log_examples(
    preds: torch.Tensor,
    truth: torch.Tensor,
    anchor: torch.Tensor,
    axes: list[str],
    to_nm: np.ndarray,
    step: int,
    data_rate_hz: float | None = None,
    n_examples: int = 4,
    descriptor: str = "",
) -> None:
    """
    Log a few concrete predicted-vs-true rows so the model's scale is legible by eye.

    Shows, at ``step``, each axis's position (displacement from the run's start) and its
    derived velocity for the first ``n_examples`` samples. Velocity is a real rate
    (micrometres/second) when ``data_rate_hz`` is known, else the per-step change.
    ``descriptor`` annotates the regime (e.g. noise floor vs compounded) in the header.
    """

    k = step - 1
    scale = torch.as_tensor(to_nm, dtype=preds.dtype)  # counts -> nm
    prev_pred = anchor if k == 0 else preds[:, k - 1, :]
    prev_true = anchor if k == 0 else truth[:, k - 1, :]
    n = min(n_examples, preds.shape[0])

    pos_pred = (preds[:, k, :] - anchor) * scale  # displacement from start, nm
    pos_true = (truth[:, k, :] - anchor) * scale
    vel_pred = (preds[:, k, :] - prev_pred) * scale  # per-step change, nm
    vel_true = (truth[:, k, :] - prev_true) * scale

    # Pick samples spanning the true-motion range (quiet -> active), nearest distinct
    # sample to each quantile, so the table shows tracking rather than only the noise
    # floor (where diff% is tiny-over-tiny and uninformative).
    mag = pos_true.norm(dim=1).numpy()
    sel: list[int] = []
    for q in np.linspace(0.2, 0.999, n):
        target = float(np.quantile(mag, q))
        for idx in np.argsort(np.abs(mag - target)):
            if int(idx) not in sel:
                sel.append(int(idx))
                break
    if data_rate_hz:
        vel_pred = vel_pred * (data_rate_hz / 1000.0)  # nm/step -> um/s
        vel_true = vel_true * (data_rate_hz / 1000.0)
        vel_unit = "um/s"
    else:
        vel_unit = "nm/step"

    suffix = f" -- {descriptor}" if descriptor else ""
    logger.info("")
    if data_rate_hz:
        logger.info(
            "Example predictions vs truth at step %d (%.2f ms into the rollout)%s:",
            step,
            step * 1000.0 / data_rate_hz,
            suffix,
        )
    else:
        logger.info("Example predictions vs truth at step %d%s:", step, suffix)

    def block(title: str, pv: torch.Tensor, tv: torch.Tensor, unit: str) -> None:
        cols = (
            f"{'sample':>6} {'axis':>4} | {f'predicted ({unit})':>17} | "
            f"{f'actual ({unit})':>17} | {f'diff ({unit})':>15} | {'diff (%)':>9}"
        )
        rule = "-" * len(cols)
        logger.info("")
        logger.info("%s:", title)
        logger.info(rule)
        logger.info(cols)
        logger.info(rule)
        for s in sel:
            for a, axis in enumerate(axes):
                p, t = pv[s, a].item(), tv[s, a].item()
                d = p - t
                pct = f"{100 * d / t:9.1f}" if abs(t) > 1e-9 else f"{'n/a':>9}"
                logger.info(
                    f"{s:>6} {axis:>4} | {p:>17.3f} | {t:>17.3f} | {d:>15.3f} | {pct}"
                )
            logger.info(rule)  # per-sample separator for readability

    block("Position (displacement from start)", pos_pred, pos_true, "nm")
    block("Velocity", vel_pred, vel_true, vel_unit)


def _log_channel_table(table: dict[int, list[ChannelMetrics]]) -> None:
    """
    Log the slim per-channel accuracy table across horizon steps.

    Errors are shown as a percentage of that step's true-value std, so they are
    scale-free: the P95 column is the acceptance gate (typical worst case) and the P99
    column flags heavier tails / discontinuities. FIT is the system-ID goodness-of-fit.
    """

    gate_pct = DEFAULT_GATE * 100
    header = (
        f"{'channel':>7} | {'P95 (%)':>9} | {'P99 (%)':>9} | {'FIT (%)':>9} | verdict"
    )
    logger.info("")
    logger.info(
        "Per-channel free-run accuracy (P95/P99 |error| as %% of true-value std; "
        "gate: P95 <= %.0f%%, P99 flags discontinuities):",
        gate_pct,
    )
    for step, rows in table.items():
        logger.info("-" * len(header))
        logger.info("horizon step %d", step)
        logger.info(header)
        logger.info("-" * len(header))
        for m in rows:
            # Velocity is noise-floor-limited by construction, so the position gate does
            # not apply -- flag it as such rather than a misleading PASS/FAIL.
            if m.name.endswith("vel"):
                verdict = "noise-fl"
            else:
                verdict = "PASS" if m.passes else "FAIL"
            logger.info(
                f"{m.name:>7} | {m.p95_frac * 100:9.2f} | {m.p99_frac * 100:9.2f} | "
                f"{m.fit:9.2f} | {verdict}"
            )


def horizon_curves(errors: torch.Tensor) -> dict[str, np.ndarray]:
    """
    Per-horizon-step error curves across samples: mean, median and P99, each [H, A].
    """

    e = errors.numpy()
    return {
        "mean": e.mean(axis=0),
        "p50": np.percentile(e, 50, axis=0),
        "p99": np.percentile(e, 99, axis=0),
    }


def position_resolution(config: RunConfiguration) -> np.ndarray:
    """
    Per-axis units-per-count of the predicted position channel (1.0 if unknown).
    """

    channel = config.system.channel(config.target_channels[0])
    axes = config.system.axes
    if channel.resolution is None:
        return np.ones(len(axes))
    return np.array([channel.resolution[a] for a in axes])


def run_horizon_eval(
    config_path: str,
    ckpt_path: str,
    horizon: int = 256,
    device: str = "cpu",
    batch_size: int = 4096,
    max_batches: int = 8,
    seed: int = 42,
    plot_path: str | None = None,
    example_step: int | None = None,
    early_step: int = 1,
) -> dict[str, np.ndarray]:
    """
    Build a plant from a checkpoint, free-run it to horizon on the test split, log the
    error-vs-horizon curve (converted to nm) and optionally save a plot.

    Worked predicted-vs-truth examples are logged at two horizon steps so the regimes
    can be compared side by side: ``early_step`` (the per-step noise floor) and
    ``example_step`` (the compounded free-run; defaults to a mid-rollout step where the
    drift is visible but not saturating).
    """

    config = RunConfiguration(config_path)
    layout = RolloutLayout.from_config(config)
    data = build_rollout_splits(
        h5_path=config.datafile_dir,
        allowed_inputs=config.input_params,
        allowed_targets=config.target_params,
        window_size=config.window_size,
        max_horizon=horizon,
        pos_cols=layout.pos_cols,
        dac_cols=layout.dac_cols,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        training_dtype=torch.float32,
        batch_size=batch_size,
        seed=seed,
        device=device,
    )
    plant = Plant.from_checkpoint(config, ckpt_path, device)

    preds, truth, anchor = collect_horizon_trajectories(
        plant, data.tst_loader, horizon, max_batches
    )
    errors = (preds - truth).abs()  # [N, H, A], physical (z-score mean cancels)
    curves = horizon_curves(errors)

    axes = config.system.axes
    to_nm = (
        position_resolution(config) * 1e9
    )  # units-per-count (m) -> nm; 1.0 if unknown
    p99_nm = curves["p99"] * to_nm[None, :]  # [H, A]

    steps = _reference_steps(horizon)
    logger.info(
        "\nError-vs-horizon (P99 |position error|, nm) over %d seeds:", len(errors)
    )
    logger.info("%6s | %s", "step", " ".join(f"{a:>10}" for a in axes))
    for step in steps:
        row = " ".join(f"{p99_nm[step - 1, a]:10.2f}" for a in range(len(axes)))
        logger.info("%6d | %s", step, row)

    ex_step = example_step if example_step is not None else min(64, horizon)
    early = max(1, min(early_step, horizon))
    # Show the noise-floor regime and the compounded regime together, skipping the
    # duplicate if they coincide.
    example_plan = [(early, "near the per-step noise floor")]
    if ex_step != early:
        example_plan.append((ex_step, "compounded free-run"))
    for s, descriptor in example_plan:
        _log_examples(
            preds,
            truth,
            anchor,
            axes,
            to_nm,
            s,
            config.system.data_rate_hz,
            descriptor=descriptor,
        )

    table = horizon_channel_table(preds, truth, anchor, axes, steps, to_nm)
    _log_channel_table(table)

    if plot_path is not None:
        _plot_curves(curves, axes, to_nm, plot_path)
        logger.info("Saved error-vs-horizon plot to %s", plot_path)

    return curves


def _reference_steps(horizon: int) -> list[int]:
    """
    A few representative horizon steps to tabulate (1, 8, 32, ..., horizon).
    """

    candidates = [1, 8, 16, 32, 64, 128, 256, 512]
    steps = [s for s in candidates if s <= horizon]
    if horizon not in steps:
        steps.append(horizon)
    return steps


def _plot_curves(
    curves: dict[str, np.ndarray], axes: list[str], to_nm: np.ndarray, path: str
) -> None:
    import matplotlib.pyplot as plt

    horizon = curves["p99"].shape[0]
    steps = np.arange(1, horizon + 1)
    fig, ax = plt.subplots(figsize=(7, 4))
    for a, axis in enumerate(axes):
        ax.plot(steps, curves["p99"][:, a] * to_nm[a], label=f"{axis} P99")
        ax.plot(
            steps,
            curves["mean"][:, a] * to_nm[a],
            "--",
            alpha=0.5,
            label=f"{axis} mean",
        )
    ax.set_xlabel("rollout horizon (steps)")
    ax.set_ylabel("position error (nm)")
    ax.set_title("Error vs horizon (free-running plant)")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, format="svg")
    plt.close(fig)
