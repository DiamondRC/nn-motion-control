"""
Compare the characterisation battery on data versus a plant.

Drives the plant with the recorded command over a long window and runs the
same battery on three trajectories: the real data, the plant's one-step
(teacher-forced) predictions, and its free-run generation. Teacher-forced
tests whether the plant reproduces the data's dynamical structure; free-run
tests whether it is a stable generator. Reuses the battery unchanged (its
measures take plain arrays).
"""

from __future__ import annotations

import logging
import os

import numpy as np
import torch

from nn_motion_control.core.config import RunConfiguration
from nn_motion_control.data_analysis.characterise import _axis_measures
from nn_motion_control.data_analysis.plotting import (
    COMPACT_PANEL_W,
    COMPACT_ROW_H,
    style_axis,
)
from nn_motion_control.data_analysis.report import write_report
from nn_motion_control.data_analysis.spectral import psd, resonance_peaks
from nn_motion_control.eval.sampling import sampled_batches
from nn_motion_control.eval.tables import log_table, render_table
from nn_motion_control.plant.plant import (
    Plant,
    RolloutLayout,
    rollout_splits_from_config,
)

DEFAULT_ROOT = "data_analysis"
COMPARE_HORIZON = 20000
COMPARE_BATCH = 8
COMPARE_SEED = 42
# Closed-loop controller comparison: the two re-anchor regimes evaluated.
CTRL_REGIMES = (("r0", 0), ("re-anch", None))
# Headline measures shown in the data-vs-plant table. All are scale-invariant
# structural/complexity measures on the position trajectory; the physics and
# input-output axes (which need the physical command) live in the characterise
# report, not this data-vs-plant structural comparison.
COMPARE_KEYS = [
    ("d2", "D2"),
    ("higuchi", "higuchi"),
    ("bandwidth", "bw(Hz)"),
    ("spectral_entropy", "spec-ent"),
    ("one_over_f", "1/f"),
    ("perm_entropy", "perm-e"),
    ("sample_entropy", "samp-e"),
    ("lempel_ziv", "LZ"),
    ("lyapunov", "lyap"),
    ("dfa_hurst", "DFA-H"),
    ("rqa_det", "RQA-det"),
    ("kurtosis", "kurt"),
]


def _physical(norm: torch.Tensor, std, mean) -> np.ndarray:
    return (norm * std + mean).cpu().numpy()


def plant_sources(
    plant: Plant,
    warmup: torch.Tensor,
    dac_future: torch.Tensor,
    gt: torch.Tensor,
    horizon: int,
) -> dict[str, np.ndarray]:
    """
    The three [H, A] physical trajectories for batch element 0: the recorded
    data, the plant's teacher-forced one-step predictions, and its free-run.
    """

    std, mean = plant.pos_std, plant.pos_mean

    with torch.no_grad():
        free_run = plant.roll_forward(warmup, dac_future, horizon)
        teacher = plant.roll_forward(
            warmup, dac_future, horizon, teacher_pos=gt, ss_prob=1.0
        )

    return {
        "data": _physical(gt[0], std, mean),
        "plant (teacher-forced)": _physical(teacher[0], std, mean),
        "plant (free-run)": _physical(free_run[0], std, mean),
    }


def _derived(pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vel = np.concatenate([[0.0], np.diff(pos)])
    acc = np.concatenate([[0.0], np.diff(vel)])

    return vel, acc


def _plot_psd_overlay(axes, sources, sample_rate, path):
    from nn_motion_control.control.track import _agg_pyplot

    plt = _agg_pyplot()
    n = len(axes)
    fig, axs = plt.subplots(
        1, n, figsize=(COMPACT_PANEL_W * n, COMPACT_ROW_H), squeeze=False
    )
    styles = {
        "data": "-",
        "plant (teacher-forced)": "--",
        "plant (free-run)": ":",
    }

    for a, axis in enumerate(axes):
        ax = axs[0, a]

        for c, (name, pos) in enumerate(sources.items()):
            freqs, power = psd(pos[:, a], sample_rate)
            ax.loglog(
                freqs[1:],
                power[1:],
                styles.get(name, "-"),
                color=f"C{c}",
                lw=1.0,
                label=name,
            )
        ax.set_title(axis)
        ax.set_xlabel("frequency (Hz)")
        ax.set_ylabel("PSD (nm^2/Hz)" if a == 0 else "")
        style_axis(ax)
        if a == 0:
            ax.legend(fontsize=7)
    fig.suptitle("Power spectral density: data vs plant")
    fig.tight_layout()
    fig.savefig(path, format="svg")
    plt.close(fig)


def run_comparison(
    config_path: str,
    out_dir: str | None = None,
    horizon: int = COMPARE_HORIZON,
    device: str | None = None,
) -> str:
    """
    Characterise data vs a plant (teacher-forced and free-run) and report.
    """

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    config = RunConfiguration(config_path)
    out_dir = out_dir or os.path.join(
        DEFAULT_ROOT, config.model_name, "comparison"
    )
    os.makedirs(out_dir, exist_ok=True)

    layout = RolloutLayout.from_config(config)
    data = rollout_splits_from_config(
        config,
        layout,
        max_horizon=horizon,
        batch_size=COMPARE_BATCH,
        seed=COMPARE_SEED,
        device=device,
    )
    plant = Plant.from_checkpoint(config, config.m_save_dir, device)
    tensors = list(next(iter(sampled_batches(data.tst_loader, 1))))
    warmup = tensors[0].to(device)
    dac_future = tensors[1].to(device)[:, :horizon]
    gt = tensors[2].to(device)[:, :horizon]

    logger = logging.getLogger(os.path.basename(__file__))
    sample_rate = config.system.data_rate_hz or 1.0
    axes = config.system.axes

    sources = plant_sources(plant, warmup, dac_future, gt, horizon)
    logger.info(
        "Comparing %s: %d-step window, %.0f Hz, axes %s",
        config.model_name,
        horizon,
        sample_rate,
        axes,
    )
    dac = dac_future[0].cpu().numpy()

    measures = {
        name: [
            _axis_measures(
                pos[:, a], dac[:, a], *_derived(pos[:, a]), sample_rate
            )
            for a in range(len(axes))
        ]
        for name, pos in sources.items()
    }

    sections = []
    for a, axis in enumerate(axes):
        rows = [
            [measures[name][a][key] for name in sources]
            for key, _ in COMPARE_KEYS
        ]
        table = render_table(
            [label for _, label in COMPARE_KEYS],
            list(sources),
            rows,
            corner_label="measure",
            value_fmt="{:.3f}",
        )
        logger.info("")
        logger.info("axis %s: data vs plant", axis)
        log_table(logger, table)
        sections.append((axis, table))

    peaks = {
        name: resonance_peaks(pos[:, 0], sample_rate)
        for name, pos in sources.items()
    }
    data_peak = peaks["data"][0][0] if peaks["data"] else None
    peak_note = (
        f"Data's top x-resonance {data_peak:.1f} Hz; "
        + "; ".join(
            f"{name}: "
            + (
                "reproduced"
                if data_peak
                and any(abs(f - data_peak) < 5 for f, _ in peaks[name])
                else "absent"
            )
            for name in sources
            if name != "data"
        )
        if data_peak
        else "No dominant data resonance found on x."
    )

    psd_path = os.path.join(out_dir, "psd_compare.svg")
    _plot_psd_overlay(axes, sources, sample_rate, psd_path)

    report_sections = [
        {
            "title": f"Axis {axis}",
            "table": table,
            "plots": [],
            "notes": [],
        }
        for axis, table in sections
    ]
    report_sections.append(
        {
            "title": "Spectrum",
            "plots": [psd_path],
            "notes": [
                peak_note,
                "Teacher-forced = plant one-step predictions along the real "
                "path (structural fidelity); free-run = plant generating from "
                "a seed on the recorded command (generative stability).",
            ],
        }
    )
    header = [
        f"Plant: `{config.model_name}` vs data",
        "",
        f"- {horizon}-step window at {sample_rate:g} Hz, axes "
        f"{', '.join(axes)}",
    ]
    path = write_report(
        f"Data vs plant: {config.model_name}", header, report_sections, out_dir
    )
    logger.info("")
    logger.info("%s", peak_note)
    logger.info("Report saved to %s", path)

    return path


def _controller_closed_loop(
    cfg_path: str, ref_spec: dict, horizon: int, seed: int, device: str
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """
    Roll a controller closed-loop on a reference, returning per-regime
    position trajectories [H, A] (physical) and the reference [H, A].
    """

    from nn_motion_control.control.config import Controller, ControllerConfig
    from nn_motion_control.training.control_run import make_reference_gen

    cfg = ControllerConfig(cfg_path)
    plant_cfg = RunConfiguration(cfg.plant_config_path)
    plant = Plant.from_checkpoint(plant_cfg, cfg.plant_checkpoint, device)
    policy = Controller.from_checkpoint(cfg.save_path, device).policy
    layout = RolloutLayout.from_config(plant_cfg)
    train = cfg.training
    ctrl_h = int(train.get("horizon", 96))
    loaders = rollout_splits_from_config(
        plant_cfg,
        layout,
        max_horizon=horizon,
        batch_size=4,
        seed=1,
        device=device,
        quiescent_seed=train.get("seed_quiescence"),
    )
    ref_gen = make_reference_gen(ref_spec, device)
    gen = torch.Generator(device=device).manual_seed(seed)
    warmup = next(iter(loaders.val_loader))[0].to(device)
    origin, _ = plant.seed_state(warmup)
    reference, ref_v = ref_gen(origin, horizon, gen)

    regimes: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for tag, reanchor in CTRL_REGIMES:
            every = ctrl_h if reanchor is None else reanchor
            pos, _ = plant.closed_loop_rollout(
                warmup,
                reference,
                policy,
                horizon,
                reference_velocity=ref_v,
                reanchor_every=every,
            )
            regimes[tag] = pos[0].cpu().numpy()

    return regimes, reference[0].cpu().numpy()


def _ctrl_measures(pos: np.ndarray, axes: list[str], sample_rate: float):
    zero = np.zeros_like(pos[:, 0])

    return [
        _axis_measures(pos[:, a], zero, *_derived(pos[:, a]), sample_rate)
        for a in range(len(axes))
    ]


def run_controller_comparison(
    ssm_cfg: str,
    tcn_cfg: str,
    out_dir: str | None = None,
    horizon: int = COMPARE_HORIZON,
    device: str | None = None,
) -> str:
    """
    Compare the battery on the SSM and TCN controllers, closed-loop, for the
    controlled position (vs data and reference) and the tracking error, in
    both re-anchor regimes.
    """

    from nn_motion_control.control.config import ControllerConfig
    from nn_motion_control.data_analysis.loader import load_channels
    from nn_motion_control.training.control_run import shape_spec

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = out_dir or os.path.join(DEFAULT_ROOT, "controller_comparison")
    os.makedirs(out_dir, exist_ok=True)
    logger = logging.getLogger(os.path.basename(__file__))

    ref_spec = shape_spec("mixed")
    ssm, ref = _controller_closed_loop(ssm_cfg, ref_spec, horizon, 42, device)
    tcn, ref_t = _controller_closed_loop(tcn_cfg, ref_spec, horizon, 42, device)

    plant_cfg = RunConfiguration(ControllerConfig(ssm_cfg).plant_config_path)
    system = plant_cfg.system
    sample_rate = system.data_rate_hz or 1.0
    axes = system.axes
    channels = load_channels(system, plant_cfg.datafile_dir)
    data_pos = channels.values["position"][:horizon]
    logger.info(
        "Controller comparison: %d-step closed loop, %.0f Hz, axes %s",
        horizon,
        sample_rate,
        axes,
    )

    position = {
        "data": data_pos,
        "reference": ref,
        "SSM r0": ssm["r0"],
        "SSM re-anch": ssm["re-anch"],
        "TCN r0": tcn["r0"],
        "TCN re-anch": tcn["re-anch"],
    }
    error = {
        "SSM r0": ssm["r0"] - ref,
        "SSM re-anch": ssm["re-anch"] - ref,
        "TCN r0": tcn["r0"] - ref_t,
        "TCN re-anch": tcn["re-anch"] - ref_t,
    }
    pos_m = {
        n: _ctrl_measures(p, axes, sample_rate) for n, p in position.items()
    }
    err_m = {n: _ctrl_measures(p, axes, sample_rate) for n, p in error.items()}

    sections = []
    for a, axis in enumerate(axes):
        pos_tbl = render_table(
            [label for _, label in COMPARE_KEYS],
            list(position),
            [[pos_m[n][a][k] for n in position] for k, _ in COMPARE_KEYS],
            corner_label=f"pos {axis}",
            value_fmt="{:.3f}",
        )
        err_tbl = render_table(
            [label for _, label in COMPARE_KEYS],
            list(error),
            [[err_m[n][a][k] for n in error] for k, _ in COMPARE_KEYS],
            corner_label=f"err {axis}",
            value_fmt="{:.3f}",
        )
        logger.info("")
        logger.info("axis %s: controlled position", axis)
        log_table(logger, pos_tbl)
        logger.info("")
        logger.info("axis %s: tracking error", axis)
        log_table(logger, err_tbl)
        sections.append(
            {
                "title": f"Axis {axis}",
                "tables": [
                    ("controlled position", pos_tbl),
                    ("tracking error", err_tbl),
                ],
                "plots": [],
                "notes": [],
            }
        )

    psd_path = os.path.join(out_dir, "psd_controllers.svg")
    _plot_psd_overlay(axes, position, sample_rate, psd_path)
    sections.append(
        {
            "title": "Spectrum (controlled position)",
            "plots": [psd_path],
            "notes": [
                "r0 = faithful closed loop (reanchor=0); re-anch isolates the "
                "controller from plant free-run drift. Lower dimension / "
                "narrower spectrum than data is the regulation signature.",
            ],
        }
    )
    header = [
        "SSM vs TCN controllers, closed-loop, vs data",
        "",
        f"- {horizon}-step mixed-reference rollout at {sample_rate:g} Hz, "
        f"axes {', '.join(axes)}",
    ]
    path = write_report("Controller comparison", header, sections, out_dir)
    logger.info("")
    logger.info("Report saved to %s", path)

    return path
