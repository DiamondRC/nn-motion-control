"""
A fixed evaluation suite for closed-loop controllers on the plant model.

Every controller, learned or the PandA PID baseline, is scored from a saved
checkpoint on the same cases: holds near and far from rest, each workspace
reference family, long endurance runs, and the hardware PID's own recorded
trajectory started from its recorded state. Each case reports tracking
error, divergence and command activity, so a controller that tracks by
thrashing cannot look good, and the suite's numbers are the ones runs are
compared by.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import h5py
import numpy as np
import torch

from nn_motion_control.control.config import (
    Controller,
    ControllerConfig,
    build_actuator,
    build_sensor,
)
from nn_motion_control.control.pid import PandaPID, PidGains
from nn_motion_control.control.workspace import (
    FAMILIES,
    WorkspaceLimits,
    WorkspaceReference,
)
from nn_motion_control.core.config import RunConfiguration
from nn_motion_control.data.device_loader import RolloutWindowLoader
from nn_motion_control.data_analysis.report import write_report
from nn_motion_control.eval.replay import build_replay_batch
from nn_motion_control.eval.tables import render_table
from nn_motion_control.plant.plant import (
    ControlPolicy,
    Plant,
    RolloutLayout,
    rollout_splits_from_config,
)
from nn_motion_control.training.control import command_activity
from nn_motion_control.training.control_run import (
    reference_limits,
    strict_float32,
)

logger = logging.getLogger(__name__)

# Bump when a case, its seeds or a metric changes, so reports say which
# suite produced them.
SUITE_VERSION = 3
SEED = 2026
ROWS = 64
ENDURANCE_ROWS = 32
CASE_SECONDS = 1.0
ENDURANCE_SECONDS = 30.0
# Seeds further than this from rest (position unit) count as far.
ORIGIN_RADIUS = 20_000.0
# Error is read over this final share of a case. A row is lost when it goes
# non-finite or its tail error passes DIVERGED (position unit) and
# LOST_SHARE of its reference's own excursion: it is no longer tracking,
# whether it ran away or could not follow, while a large reference tracked
# loosely is not counted.
TAIL = 0.1
DIVERGED = 1000.0
LOST_SHARE = 0.5
# Below this command std (counts) the command is still, and its spectral
# split is rounding noise, not reported.
MIN_ACTIVITY = 1.0
# Command energy above this frequency (Hz) marks a limit cycle or thrash.
THRASH_HZ = 200.0
# A command within this fraction of a non-zero limit is on the rail.
RAIL_FRACTION = 0.99
PERCENTILE = 95.0
# PID-scale cases: axis-aligned minimum-jerk moves on the first axis of
# these sizes (position unit) over MOVE_SECONDS, then held, the kind of
# motion the hardware PID runs.
PID_MOVES = (200_000.0, 20_000.0)
MOVE_SECONDS = 0.6
# The hardware trajectory case: the PID capture's window (seconds) and the
# span before it the rest offset is read over.
DRIVE_WINDOW_S = (0.6, 12.6)
REST_SPAN_S = 0.4
CAPTURE_STRIDE = 4


@dataclass
class Case:
    """
    One closed-loop test: seeds [B, F, W] and a reference [B, H, A].
    """

    name: str
    warmup: torch.Tensor
    reference: torch.Tensor
    reference_velocity: torch.Tensor


@dataclass
class Contender:
    """
    A controller under test: a policy factory plus its hardware models.
    """

    name: str
    make_policy: Callable[[], ControlPolicy]
    sensor: object | None = None
    actuator: object | None = None


@dataclass
class SuiteResult:
    cases: list[str] = field(default_factory=list)
    scores: dict[str, dict[str, dict[str, float]]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def score_rows(
    position: torch.Tensor,
    command: torch.Tensor,
    reference: torch.Tensor,
    safe_range: torch.Tensor,
    rate_hz: float,
) -> dict[str, np.ndarray]:
    """
    Per-row metrics over a case's tail: error RMS (largest axis), whether
    the row diverged, command std, the share of command energy above
    THRASH_HZ and the share of time on a rail.
    """

    tail = max(1, int(round(TAIL * position.shape[1])))
    err = (position - reference)[:, -tail:]
    rms = err.pow(2).mean(dim=1).sqrt().amax(dim=-1)
    offset = err.mean(dim=1).abs().amax(dim=-1)
    fluctuation = err.std(dim=1).amax(dim=-1)
    peak = err.abs().amax(dim=(1, 2))
    excursion = (reference - reference[:, :1]).abs().amax(dim=(1, 2))
    limit = torch.clamp(LOST_SHARE * excursion, min=DIVERGED)
    lost = ~torch.isfinite(peak) | (peak > limit)
    cmd = command[:, -tail:]
    std, thrash = command_activity(cmd, rate_hz, THRASH_HZ)
    lo = safe_range[:, 0].to(cmd.device)
    hi = safe_range[:, 1].to(cmd.device)
    railed = ((hi > 0) & (cmd >= RAIL_FRACTION * hi)) | (
        (lo < 0) & (cmd <= RAIL_FRACTION * lo)
    )

    thrash = torch.where(std >= MIN_ACTIVITY, thrash, torch.nan)

    return {
        "rms": rms.cpu().numpy(),
        "offset": offset.cpu().numpy(),
        "fluctuation": fluctuation.cpu().numpy(),
        "lost": lost.cpu().numpy(),
        "command_std": std.cpu().numpy(),
        "thrash": thrash.cpu().numpy(),
        "rail": railed.float().mean(dim=(1, 2)).cpu().numpy(),
    }


def summarise(rows: dict[str, np.ndarray]) -> dict[str, float]:
    finite = rows["rms"][np.isfinite(rows["rms"])]
    has = finite.size > 0

    thrash = rows["thrash"][np.isfinite(rows["thrash"])]

    return {
        "median_nm": float(np.median(finite)) if has else float("nan"),
        "p95_nm": float(np.percentile(finite, PERCENTILE)) if has else np.nan,
        "offset_nm": float(np.nanmedian(rows["offset"])),
        "fluctuation_nm": float(np.nanmedian(rows["fluctuation"])),
        "lost": int(rows["lost"].sum()),
        "rows": int(rows["rms"].size),
        "command_std": float(np.nanmedian(rows["command_std"])),
        "thrash": float(np.median(thrash)) if thrash.size else float("nan"),
        "rail": float(np.nanmean(rows["rail"])),
    }


def _seeds(
    cfg: ControllerConfig, plant_cfg: RunConfiguration, plant: Plant, device
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Validation seeds near rest and far from it, ROWS of each, spread evenly
    across the validation starts and kept inside the workspace box.
    """

    quiescence = dict(cfg.training.get("seed_quiescence") or {})
    limits = _limits(cfg, plant_cfg, _workspace_spec(cfg))
    box = torch.stack([limits.pos_lo, limits.pos_hi], 1)
    quiescence["bounds"] = box.tolist()
    # The balance is the suite's own: near and far are drawn separately.
    for key in ("off_origin_fraction", "origin_radius"):
        quiescence.pop(key, None)
    loaders = rollout_splits_from_config(
        plant_cfg,
        RolloutLayout.from_config(plant_cfg),
        max_horizon=1,
        batch_size=ROWS,
        seed=SEED,
        device=device,
        quiescent_seed=quiescence,
    )
    val = loaders.val_loader
    if not isinstance(val, RolloutWindowLoader):
        raise ValueError("The suite needs rollout validation seeds")
    # Classify every validation start, since far holds are a few percent
    # of them, then spread each side's picks across its own starts.
    origin, _ = plant.seed_state(val.last_frames()[:, :, None])
    far = origin.abs().amax(dim=1) > ORIGIN_RADIUS
    picked = []

    for mask in (~far, far):
        if int(mask.sum()) < ROWS:
            raise ValueError(
                f"Only {int(mask.sum())} validation seeds on one side of "
                f"rest, the suite needs {ROWS}"
            )
        val.restrict(mask)
        picked.append(val.spread_warmups(ROWS))
    val.restrict(None)

    return picked[0], picked[1]


def _limits(
    cfg: ControllerConfig, plant_cfg: RunConfiguration, spec: dict
) -> WorkspaceLimits:
    limits = reference_limits(cfg, plant_cfg, spec)
    if limits is None:
        raise ValueError("The suite needs workspace limits")

    return limits


def _workspace_spec(cfg: ControllerConfig) -> dict:
    spec = dict(cfg.training.get("reference", {}))
    if spec.get("kind") != "workspace":
        spec = {"kind": "workspace"}
    # The suite always draws the full range, never a curriculum stage.
    spec.pop("amplitude_curriculum", None)

    return spec


def _hold_case(name: str, plant: Plant, warmup: torch.Tensor, steps: int):
    origin, _ = plant.seed_state(warmup)
    ref = origin[:, None, :].expand(-1, steps, -1).contiguous()

    return Case(name, warmup, ref, torch.zeros_like(ref))


def _pid_move_case(
    name: str,
    plant: Plant,
    warmup: torch.Tensor,
    distance: float,
    steps: int,
    rate_hz: float,
) -> Case:
    # Each seed moves towards rest on the first axis, so the move stays in
    # the travel whichever side of rest it starts.
    origin, _ = plant.seed_state(warmup)
    t = torch.arange(steps, device=origin.device, dtype=origin.dtype) / rate_hz
    s = (t / MOVE_SECONDS).clamp(0.0, 1.0)
    prog = s**3 * (10.0 - 15.0 * s + 6.0 * s * s)
    toward = torch.where(origin[:, 0] > 0, -1.0, 1.0)
    ref = origin[:, None, :].expand(-1, steps, -1).clone()
    ref[:, :, 0] = origin[:, None, 0] + toward[:, None] * distance * prog
    ref_v = torch.zeros_like(ref)
    ref_v[:, 1:] = ref[:, 1:] - ref[:, :-1]

    return Case(name, warmup, ref, ref_v)


def suite_cases(
    cfg: ControllerConfig,
    plant_cfg: RunConfiguration,
    plant: Plant,
    device: str,
) -> list[Case]:
    rate = cfg.servo_rate_hz
    steps = int(round(CASE_SECONDS * rate))
    long_steps = int(round(ENDURANCE_SECONDS * rate))
    near, far = _seeds(cfg, plant_cfg, plant, device)
    mixed = torch.cat([near[: ROWS // 2], far[: ROWS // 2]])
    cases = [
        _hold_case("hold near rest", plant, near, steps),
        _hold_case("hold far", plant, far, steps),
    ]

    for distance in PID_MOVES:
        cases.append(
            _pid_move_case(
                f"PID-scale move {distance / 1e3:g} um",
                plant,
                mixed,
                distance,
                steps,
                rate,
            )
        )
    spec = _workspace_spec(cfg)
    limits = _limits(cfg, plant_cfg, spec)
    origin, _ = plant.seed_state(mixed)
    hold = plant.seed_command(mixed)

    for i, family in enumerate(FAMILIES):
        if family == "hold":
            continue
        weights = {name: float(name == family) for name in FAMILIES}
        gen = WorkspaceReference({**spec, "weights": weights}, limits)
        rng = torch.Generator(device=device).manual_seed(SEED + i)
        ref, ref_v = gen(origin, steps, rng, hold=hold)
        cases.append(Case(family, mixed, ref, ref_v))

    endurance = torch.cat(
        [near[: ENDURANCE_ROWS // 2], far[: ENDURANCE_ROWS // 2]]
    )
    cases.append(
        _hold_case(
            f"endurance hold {ENDURANCE_SECONDS:g} s",
            plant,
            endurance,
            long_steps,
        )
    )
    origin, _ = plant.seed_state(endurance)
    gen = WorkspaceReference(spec, limits)
    rng = torch.Generator(device=device).manual_seed(SEED)
    ref, ref_v = gen(
        origin, long_steps, rng, hold=plant.seed_command(endurance)
    )
    cases.append(
        Case(f"endurance mix {ENDURANCE_SECONDS:g} s", endurance, ref, ref_v)
    )

    return cases


def drive_case(
    capture_dir: str,
    plant: Plant,
    plant_cfg: RunConfiguration,
    axes: list[str],
    rate_hz: float,
) -> tuple[Case, dict[str, float]]:
    """
    The hardware PID's recorded trajectory as a case, seeded from its
    recorded state, plus the hardware's own tracking over the same window.

    The capture is read at the PID's ticks (every CAPTURE_STRIDE samples,
    just after each setpoint update) in the encoder frame, and placed in
    the plant's rest frame through the static gain over the rest span
    before the window, where the stage is held near rest.
    """

    run = Path(capture_dir)
    capture = next(run.glob("*_trajectory.h5"))
    with h5py.File(capture, "r") as f:
        keys = ["PGEN1.OUT.Value"] + [
            f"INENC{i + 1}.VAL.Value" for i in range(len(axes))
        ]
        raw = [np.asarray(cast(h5py.Dataset, f[k])[()]) for k in keys]
        out = np.asarray(cast(h5py.Dataset, f["PID.REAL_OUTPUT.Value"])[()])
        gain = np.asarray(
            h5py.File(plant_cfg.datafile_dir, "r").attrs[
                "static_gain_nm_per_count"
            ],
            float,
        )
    updates = np.nonzero(np.diff(raw[0]) != 0)[0] + 1
    phase = int(np.bincount(updates % CAPTURE_STRIDE).argmax())
    idx = np.arange(phase, len(raw[0]), CAPTURE_STRIDE)
    setpoint = raw[0][idx]
    fb = np.stack([r[idx] for r in raw[1:]])
    command = out[idx]
    k0, k1 = (int(round(t * rate_hz)) for t in DRIVE_WINDOW_S)
    rest = slice(k0 - int(round(REST_SPAN_S * rate_hz)), k0)
    offset = fb[:, rest].mean(axis=1)
    offset[0] = np.mean(fb[0, rest] - gain[0] * command[rest])
    pos = fb - offset[:, None]
    dac = np.zeros_like(pos)
    dac[0, 1:] = command[:-1]  # rows pair each position with the last command
    w = plant_cfg.window_size
    warmup, _, _ = build_replay_batch(
        plant,
        pos[:, k0 - w + 1 : k0 + 2],
        dac[:, k0 - w + 1 : k0 + 2],
        0,
        w,
        1,
        1,
    )
    steps = k1 - k0
    ref = torch.zeros(1, steps, len(axes), device=plant.device)
    ref[0, :, 0] = torch.as_tensor(
        setpoint[k0 + 1 : k1 + 1] - offset[0], device=plant.device
    )
    ref_v = torch.zeros_like(ref)
    ref_v[:, 1:] = ref[:, 1:] - ref[:, :-1]
    hw_err = setpoint[k0 + 1 : k1 + 1] - fb[0, k0 + 1 : k1 + 1]
    tail = max(1, int(round(TAIL * steps)))
    hardware = {
        "median_nm": float(np.sqrt(np.mean(hw_err[-tail:] ** 2))),
        "full_rms_nm": float(np.sqrt(np.mean(hw_err**2))),
        "command_std": float(np.std(command[k1 - tail : k1])),
    }

    return Case("hardware trajectory", warmup[:1], ref, ref_v), hardware


def pid_contender(
    capture_dir: str, plant_cfg: RunConfiguration, n_axes: int
) -> Contender:
    """
    The PandA PID with the capture's gains on every axis, its position
    feedforward scaled by each axis's static gain against the tuned (first)
    axis.
    """

    gains = PidGains.from_manifest(Path(capture_dir) / "manifest.json")
    with h5py.File(plant_cfg.datafile_dir, "r") as f:
        gain = np.asarray(f.attrs["static_gain_nm_per_count"], float)
    scale = torch.tensor(gain[0] / gain[:n_axes], dtype=torch.float32)

    return Contender(
        "panda PID (capture gains)", lambda: PandaPID(gains, scale)
    )


def checkpoint_contender(path: str, device: str) -> Contender:
    """
    A learned controller from a config (its save path) or a checkpoint.
    """

    if path.endswith(".json"):
        cfg = ControllerConfig(path)
        controller = Controller.from_checkpoint(cfg.save_path, device)

        return Contender(
            cfg.model_name,
            lambda: controller.policy,
            build_sensor(cfg),
            build_actuator(cfg),
        )
    controller = Controller.from_checkpoint(path, device)

    return Contender(Path(path).stem, lambda: controller.policy)


def run_suite(
    controllers: list[str],
    suite_config: str,
    pid_capture: str,
    out_dir: str | None = None,
) -> str:
    """
    Score every controller and the PID baseline on the suite, write
    report.md and results.json, and return the report path.
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Rollouts are float32: keep TF32 from coarsening the nm-level states.
    strict_float32()
    cfg = ControllerConfig(suite_config)
    plant_cfg = RunConfiguration(cfg.plant_config_path)
    plant = Plant.from_checkpoint(plant_cfg, cfg.plant_checkpoint, device)
    cases = suite_cases(cfg, plant_cfg, plant, device)
    drive, hardware = drive_case(
        pid_capture, plant, plant_cfg, list(cfg.axes), cfg.servo_rate_hz
    )
    cases.append(drive)
    contenders = [pid_contender(pid_capture, plant_cfg, len(cfg.axes))]
    contenders += [checkpoint_contender(p, device) for p in controllers]
    result = SuiteResult(cases=[c.name for c in cases])

    for who in contenders:
        result.scores[who.name] = {}

        for case in cases:
            with torch.no_grad():
                pos, dac = plant.closed_loop_rollout(
                    case.warmup,
                    case.reference,
                    who.make_policy(),
                    case.reference.shape[1],
                    reference_velocity=case.reference_velocity,
                    sensor=who.sensor,
                    actuator=who.actuator,
                )
            rows = score_rows(
                pos, dac, case.reference, cfg.safe_range, cfg.servo_rate_hz
            )
            result.scores[who.name][case.name] = summarise(rows)
            logger.info(
                "%s / %s: median %.3g nm, %d lost",
                who.name,
                case.name,
                result.scores[who.name][case.name]["median_nm"],
                result.scores[who.name][case.name]["lost"],
            )
    result.notes.append(
        "Hardware PID on the real stage, hardware trajectory: tail RMS "
        f"{hardware['median_nm']:.1f} nm, whole window "
        f"{hardware['full_rms_nm']:.1f} nm, command std "
        f"{hardware['command_std']:.2f} counts"
    )

    return _write(
        result,
        cfg,
        out_dir or os.path.join("data_analysis", "controller_suite"),
    )


def _write(result: SuiteResult, cfg: ControllerConfig, out: str) -> str:
    os.makedirs(out, exist_ok=True)
    names = list(result.scores)
    cols = [
        "median nm",
        "P95 nm",
        "offset nm",
        "fluct nm",
        "lost",
        "cmd std",
        "hf share",
        "rail %",
    ]
    tables = []

    for case in result.cases:
        rows = []

        for name in names:
            s = result.scores[name][case]
            rows.append(
                [
                    f"{s['median_nm']:.3g}",
                    f"{s['p95_nm']:.3g}",
                    f"{s['offset_nm']:.3g}",
                    f"{s['fluctuation_nm']:.3g}",
                    f"{s['lost']}/{s['rows']}",
                    f"{s['command_std']:.3g}",
                    "-" if np.isnan(s["thrash"]) else f"{s['thrash']:.2f}",
                    f"{100.0 * s['rail']:.1f}",
                ]
            )
        tables.append(
            (case, render_table(names, cols, rows, corner_label="controller"))
        )
    header = [
        f"- Suite v{SUITE_VERSION}, seed {SEED}, plant {cfg.plant_checkpoint}",
        f"- {ROWS} seeds per case ({ROWS} near rest and {ROWS} far for the "
        f"holds, half of each elsewhere), {CASE_SECONDS:g} s cases, "
        f"{ENDURANCE_ROWS} seeds over {ENDURANCE_SECONDS:g} s for endurance",
        f"- Error is the tail RMS over the last {100 * TAIL:g} % (largest "
        "axis), split into its steady offset and its fluctuation; a row is "
        f"lost past {DIVERGED:g} nm and {100 * LOST_SHARE:g} % of its "
        "reference's excursion (ran away or could not follow); hf share is "
        "the command energy above "
        f"{THRASH_HZ:g} Hz, shown only when the command moves (std >= "
        f"{MIN_ACTIVITY:g} count)",
        "- The PID runs the capture's x gains on every axis (y and z are an "
        "approximation); no sensor noise or disturbance is modelled",
        *[f"- {note}" for note in result.notes],
    ]
    sections = [{"title": "Cases", "tables": tables}]
    path = write_report("Controller suite", header, sections, out)
    with open(os.path.join(out, "results.json"), "w") as f:
        json.dump(
            {
                "suite_version": SUITE_VERSION,
                "scores": result.scores,
                "notes": result.notes,
            },
            f,
            indent=2,
        )

    return path
