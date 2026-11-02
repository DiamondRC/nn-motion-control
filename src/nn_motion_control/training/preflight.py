"""
Checks a controller run must pass before it trains.

Each check targets a mistake that cost a full run before it was caught:
references the stage cannot follow, seeds it cannot hold, a loss that ranks
a degenerate controller (silent, frozen, railed) as well as a working one,
one scale or one term swamping the gradient, a heavy tail of exploding
rows, and fixed-point ranges that clip. All of them read a warm-start (or
fresh) batch, so the whole set takes minutes rather than an epoch.
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, field

import numpy as np
import torch

from nn_motion_control.control.config import ControllerConfig
from nn_motion_control.control.pid import PandaPID, PidGains
from nn_motion_control.control.workspace import (
    COMMAND_FRACTION,
    FAMILIES,
    REFERENCE_MAX_HZ,
    MeasuredStage,
    WorkspaceReference,
)
from nn_motion_control.data.device_loader import RolloutWindowLoader
from nn_motion_control.plant.rollout import _BoundRowGradient
from nn_motion_control.training.control import ControlTrainer, control_loss

logger = logging.getLogger(__name__)

PASS, WARN, FAIL = "pass", "warn", "fail"
# References judged per row over this many steps, drawn twice as long so
# the judged part never ends mid-motion.
CHECK_STEPS = 2000
CHECK_ROWS = 256
# The gradient check backpropagates the full horizon, so it takes fewer.
GRADIENT_ROWS = 64
# Shares of references whose measured-stage command may pass the safe range
# (fail) or COMMAND_FRACTION of it (warn).
INFEASIBLE_FAIL = 0.01
OVER_BUDGET_WARN = 0.05
# Content counted in the command check, past the shape rate cap for the
# minimum-jerk ramps and triangle harmonics.
CHECK_BAND_HZ = 2.0 * REFERENCE_MAX_HZ
# A degenerate controller must score at least this many times the baseline
# controller's loss; closer means the loss barely tells them apart.
DEGENERATE_MARGIN = 2.0
# An amplitude decade carrying this share of the tracking loss while
# holding under DOMINANT_ROWS of the rows swamps the rest.
DOMINANT_LOSS = 0.8
DOMINANT_ROWS = 0.3
# A gradient tail past this multiple of the median row is explosive.
TAIL_RATIO = 1e3
# Share of activations a fixed-point range may clip.
SATURATION_WARN = 1e-3
# Seeds further than this from rest (position unit) count as far.
ORIGIN_RADIUS = 20_000.0


@dataclass
class Check:
    name: str
    status: str
    lines: list[str] = field(default_factory=list)


class PreflightError(RuntimeError):
    pass


class _Constant:
    """
    A degenerate policy: a fixed command, or the seed's own when primed.
    """

    def __init__(self, value: torch.Tensor | None):
        self.value = value
        self.held: torch.Tensor | None = None

    def reset(self, batch: int, device) -> None:
        self.held = None

    def prime(self, command: torch.Tensor) -> None:
        self.held = command

    def __call__(self, position, velocity, reference, reference_velocity):
        if self.value is None:
            held = self.held if self.held is not None else position * 0.0

            return held

        return self.value.to(position.device).expand_as(position)


def _sample(trainer: ControlTrainer) -> tuple[torch.Tensor, ...]:
    loader = trainer.val_loader
    if not isinstance(loader, RolloutWindowLoader):
        raise PreflightError("Preflight needs rollout validation seeds")
    warmup = loader.spread_warmups(CHECK_ROWS).to(trainer.device)
    origin, _ = trainer.plant.seed_state(warmup)

    return warmup, origin, trainer.plant.seed_command(warmup)


def check_settings(trainer: ControlTrainer, config: ControllerConfig) -> Check:
    _, std = trainer.plant.feature_stats(["position"])
    pos_std = std[:, 0].tolist()
    gen = trainer.reference_gen
    lines = [
        f"position std in the data (per axis): "
        f"{', '.join(f'{v:.3g}' for v in pos_std)}",
        f"error_scale {config.error_scale}, error_clip {config.error_clip}, "
        f"tracking_scale {trainer.tracking_scale}, huber_delta "
        f"{trainer.huber_delta}, row cap {trainer.row_error_cap}",
        f"weights: tracking {trainer.tracking_weight}, effort "
        f"{trainer.effort_weight}, rate {trainer.rate_weight}, growth "
        f"{trainer.growth_weight}",
        f"safe range: {config.safe_range.tolist()}",
    ]
    if isinstance(gen, WorkspaceReference):
        lim = gen.limits
        lines.append(
            f"reference amplitude {gen.amplitude}, frequency {gen.frequency} "
            f"Hz, box {lim.pos_lo.tolist()} .. {lim.pos_hi.tolist()}"
        )

    return Check("settings vs data", PASS, lines)


def check_realisability(
    trainer: ControlTrainer, config: ControllerConfig, measured: MeasuredStage
) -> Check:
    gen = trainer.reference_gen
    if not isinstance(gen, WorkspaceReference):
        return Check("realisability", PASS, ["Not a workspace reference"])
    _, origin, hold = _sample(trainer)
    gen.set_epoch(None)
    rng = torch.Generator(device=trainer.device).manual_seed(config.seed)
    ref, _ = gen(origin, 2 * CHECK_STEPS, rng, hold=hold)
    disp = (ref - origin[:, None, :]).double().cpu().numpy()
    need = measured.command(disp, config.servo_rate_hz, CHECK_BAND_HZ)
    need = need[:, :CHECK_STEPS] + hold.double().cpu().numpy()[:, None, :]
    safe = config.safe_range.double().numpy()
    use = np.maximum(need / safe[:, 1], need / safe[:, 0]).max(axis=(1, 2))
    infeasible = float((use > 1.0).mean())
    over = float((use > COMMAND_FRACTION).mean())
    fam = gen.last_families
    lines = [
        f"rows needing more than the safe range: {100 * infeasible:.2f} %, "
        f"more than {COMMAND_FRACTION:g} of it: {100 * over:.2f} %, worst "
        f"{use.max():.2f} of the range",
    ]
    if fam is not None:
        fam_np = fam.cpu().numpy()
        lines.append(
            "worst use by family: "
            + ", ".join(
                f"{name} {use[fam_np == i].max():.2f}"
                for i, name in enumerate(FAMILIES)
                if (fam_np == i).any()
            )
        )
    status = (
        FAIL
        if infeasible > INFEASIBLE_FAIL
        else WARN
        if over > OVER_BUDGET_WARN
        else PASS
    )

    return Check("realisability (measured FRF)", status, lines)


def check_seeds(trainer: ControlTrainer, config: ControllerConfig) -> Check:
    lines, status = [], PASS

    for name, loader in (
        ("train", trainer.train_loader),
        ("val", trainer.val_loader),
    ):
        if not isinstance(loader, RolloutWindowLoader):
            continue
        frames = loader.last_frames()[:, :, None]
        origin, _ = trainer.plant.seed_state(frames)
        command = trainer.plant.seed_command(frames)
        lo, hi = config.safe_range[:, 0], config.safe_range[:, 1]
        outside = (command < lo.to(command.device)) | (
            command > hi.to(command.device)
        )
        far = (origin.abs().amax(dim=1) > ORIGIN_RADIUS).float().mean()
        bad = float(outside.any(dim=1).float().mean())
        lines.append(
            f"{name}: {len(frames)} seeds, {100 * float(far):.1f} % beyond "
            f"{ORIGIN_RADIUS:g} of rest, {100 * bad:.2f} % held outside the "
            "safe range"
        )
        if bad > 0:
            status = FAIL

    return Check("seeds", status, lines)


def _loss(trainer: ControlTrainer, pos, ref, dac, ref_v) -> float:
    return float(
        control_loss(
            pos,
            ref,
            dac,
            tracking_weight=trainer.tracking_weight,
            effort_weight=trainer.effort_weight,
            rate_weight=trainer.rate_weight,
            velocity_weight=trainer.velocity_weight,
            huber_delta=trainer.huber_delta,
            reference_velocity=ref_v,
            horizon_weights=None,
            axis_weights=trainer.axis_weights,
            tracking_scale=trainer.tracking_scale,
        )
    )


def check_degenerate(
    trainer: ControlTrainer,
    config: ControllerConfig,
    pid: PidGains | None,
    measured: MeasuredStage,
) -> Check:
    warmup, origin, hold = _sample(trainer)
    gen = trainer.reference_gen
    if isinstance(gen, WorkspaceReference):
        gen.set_epoch(None)
        rng = torch.Generator(device=trainer.device).manual_seed(config.seed)
        ref, ref_v = gen(origin, CHECK_STEPS, rng, hold=hold)
    else:
        ref, ref_v = gen(origin, CHECK_STEPS, None)
    hi = config.safe_range[:, 1]
    contenders = {
        "this controller": trainer.policy,
        "zero output": _Constant(torch.zeros(len(config.axes))),
        "frozen seed command": _Constant(None),
        "positive rail": _Constant(hi),
    }
    if pid is not None:
        # Its position feedforward was tuned on the first axis.
        scale = torch.tensor(
            measured.gain[0] / measured.gain, dtype=torch.float32
        )
        contenders["PandA PID"] = PandaPID(pid, scale)
    scores = {}
    trainer.model.eval()

    for name, policy in contenders.items():
        with torch.no_grad():
            pos, dac = trainer.plant.closed_loop_rollout(
                warmup,
                ref,
                policy,
                CHECK_STEPS,
                reference_velocity=ref_v,
                sensor=trainer.sensor,
                actuator=trainer.actuator,
            )
        scores[name] = _loss(trainer, pos, ref, dac, ref_v)
    baseline = min(scores.get("PandA PID", math.inf), scores["this controller"])
    degenerate = {
        k: v
        for k, v in scores.items()
        if k not in ("this controller", "PandA PID")
    }
    worst_case = min(degenerate.values())
    lines = [f"{k}: {v:.4g}" for k, v in scores.items()]
    lines.append(
        f"best degenerate / best working = {worst_case / baseline:.2f} "
        f"(needs >= {DEGENERATE_MARGIN:g})"
    )
    status = FAIL if worst_case < DEGENERATE_MARGIN * baseline else PASS

    return Check("degenerate controllers rank worse", status, lines)


def check_loss_balance(
    trainer: ControlTrainer, config: ControllerConfig
) -> Check:
    gen = trainer.reference_gen
    if not isinstance(gen, WorkspaceReference):
        return Check(
            "loss balance by scale", PASS, ["Not a workspace reference"]
        )
    warmup, origin, hold = _sample(trainer)
    gen.set_epoch(None)
    rng = torch.Generator(device=trainer.device).manual_seed(config.seed + 1)
    ref, ref_v = gen(origin, CHECK_STEPS, rng, hold=hold)
    trainer.model.eval()
    with torch.no_grad():
        pos, dac = trainer.plant.closed_loop_rollout(
            warmup, ref, trainer.policy, CHECK_STEPS, reference_velocity=ref_v
        )
    amp = (ref - origin[:, None, :]).abs().amax(dim=(1, 2)).cpu().numpy()
    per_row = np.array(
        [
            _loss(
                trainer,
                pos[i : i + 1],
                ref[i : i + 1],
                dac[i : i + 1],
                ref_v[i : i + 1],
            )
            for i in range(pos.shape[0])
        ]
    )
    total = per_row.sum()
    lines, status = [], PASS
    decades = np.floor(np.log10(np.maximum(amp, 1.0)))

    for d in np.unique(decades):
        sel = decades == d
        share_loss = per_row[sel].sum() / max(total, 1e-30)
        share_rows = sel.mean()
        lines.append(
            f"amplitude 1e{int(d)} nm: {100 * share_rows:.0f} % of rows carry "
            f"{100 * share_loss:.0f} % of the loss"
        )
        if share_loss > DOMINANT_LOSS and share_rows < DOMINANT_ROWS:
            status = WARN

    return Check("loss balance by scale", status, lines)


def check_gradient_tail(
    trainer: ControlTrainer, config: ControllerConfig
) -> Check:
    warmup, origin, hold = (t[:GRADIENT_ROWS] for t in _sample(trainer))
    gen = trainer.reference_gen
    horizon = trainer.max_horizon
    if isinstance(gen, WorkspaceReference):
        gen.set_epoch(None)
        rng = torch.Generator(device=trainer.device).manual_seed(
            config.seed + 2
        )
        ref, ref_v = gen(origin, horizon, rng, hold=hold)
    else:
        ref, ref_v = gen(origin, horizon, None)
    rows: list[torch.Tensor] = []
    original = _BoundRowGradient.backward

    def spy(ctx, *grads):
        if grads[0].dim() == 3:
            rows.append(grads[0].flatten(1).norm(dim=1).detach())

        return original(ctx, *grads)

    _BoundRowGradient.backward = staticmethod(spy)
    trainer.model.train()
    try:
        pos, dac = trainer.plant.closed_loop_rollout(
            warmup,
            ref,
            trainer.policy,
            horizon,
            reference_velocity=ref_v,
            row_grad_bound=math.inf,
        )
        trainer.model.zero_grad()
        control_loss(
            pos,
            ref,
            dac,
            tracking_weight=trainer.tracking_weight,
            effort_weight=trainer.effort_weight,
            rate_weight=trainer.rate_weight,
            huber_delta=trainer.huber_delta,
            reference_velocity=ref_v,
            tracking_scale=trainer.tracking_scale,
        ).backward()
    finally:
        _BoundRowGradient.backward = original
        trainer.model.zero_grad()
        # Peaks seen here are not training batches.
        discard = getattr(trainer.model, "discard_act_peaks", None)
        if discard is not None:
            discard()
    peak = torch.stack(rows).amax(dim=0).cpu().numpy()
    median = float(np.median(peak))
    ratio = float(peak.max() / max(median, 1e-30))
    heavy = float((peak > TAIL_RATIO * median).mean())
    lines = [
        f"at horizon {horizon}: per-row peak state gradient median "
        f"{median:.3g}, max {peak.max():.3g} ({ratio:.3g}x), "
        f"{100 * heavy:.1f} % of rows past {TAIL_RATIO:g}x the median",
    ]
    status = WARN if ratio > TAIL_RATIO else PASS

    return Check("gradient tail", status, lines)


def check_saturation(
    trainer: ControlTrainer, config: ControllerConfig
) -> Check:
    net = trainer.model
    if not hasattr(net, "track_saturation"):
        return Check("activation saturation", PASS, ["No fixed ranges"])
    warmup, origin, hold = _sample(trainer)
    ref = origin[:, None, :].expand(-1, CHECK_STEPS, -1).contiguous()
    net.track_saturation = True
    net.saturated = torch.zeros(())
    net.observed = torch.zeros(())
    net.eval()
    try:
        with torch.no_grad():
            trainer.plant.closed_loop_rollout(
                warmup, ref, trainer.policy, CHECK_STEPS
            )
    finally:
        net.track_saturation = False
    observed = float(net.observed)
    if observed == 0:
        return Check("activation saturation", PASS, ["Ranges not yet observed"])
    share = float(net.saturated) / observed
    status = WARN if share > SATURATION_WARN else PASS

    return Check(
        "activation saturation",
        status,
        [f"{100 * share:.3f} % of activations clipped by the fixed range"],
    )


def run_preflight(
    trainer: ControlTrainer,
    config: ControllerConfig,
    dataset_path: str,
    pid: PidGains | None,
    out_dir: str | None = None,
) -> list[Check]:
    """
    Run every check, log and write them (preflight.md in 'out_dir'), and
    return them; the caller decides what a failure stops.
    """

    measured = MeasuredStage.from_dataset(dataset_path, tuple(config.axes))
    checks = [
        check_settings(trainer, config),
        check_realisability(trainer, config, measured),
        check_seeds(trainer, config),
        check_degenerate(trainer, config, pid, measured),
        check_loss_balance(trainer, config),
        check_gradient_tail(trainer, config),
        check_saturation(trainer, config),
    ]
    trainer.model.train()
    lines = ["# Preflight", ""]

    for c in checks:
        logger.info("Preflight %s: %s", c.status.upper(), c.name)
        lines.append(f"## {c.status.upper()}: {c.name}")

        for line in c.lines:
            logger.info("    %s", line)
            lines.append(f"- {line}")
        lines.append("")
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "preflight.md"), "w") as f:
            f.write("\n".join(lines))

    return checks
