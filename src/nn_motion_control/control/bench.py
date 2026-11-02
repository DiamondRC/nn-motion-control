"""
Head-to-head closed-loop benchmark of trained controllers.

Every controller runs through its own plant and hardware models from the
same quiescent seeds and the same reference draws, so differences are the
controllers', not the stimulus. The suite covers a long still hold
(stationarity, actuation realism and whether the hold is a limit cycle),
trajectory tracking, a step response, rejection of a constant DAC bias
switched on mid-hold (the defining test of integral action; an evaluation
stimulus only, never trained on) and FPGA cost. Recurrent controllers
also report their learned poles.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field

import numpy as np
import torch

from nn_motion_control.control.closed_loop import (
    step_reference,
    tracking_percentiles,
)
from nn_motion_control.control.config import (
    Controller,
    ControllerConfig,
    build_actuator,
    build_sensor,
)
from nn_motion_control.control.recurrent import RecurrentControllerNet
from nn_motion_control.control.resource import score_controller
from nn_motion_control.control.workspace import WorkspaceReference
from nn_motion_control.core.config import RunConfiguration, position_to_nm
from nn_motion_control.data.device_loader import RolloutWindowLoader
from nn_motion_control.data_analysis import spectral
from nn_motion_control.data_analysis.report import write_report
from nn_motion_control.eval.tables import render_table
from nn_motion_control.plant.plant import (
    Plant,
    RolloutLayout,
    rollout_splits_from_config,
)
from nn_motion_control.training.control_run import (
    make_reference_gen,
    reference_limits,
    shape_spec,
    strict_float32,
)

logger = logging.getLogger(__name__)

BATCH = 8
SEED_SELECT = 1  # quiescent-seed draw, shared by every arm
REFERENCE_SEED = 0  # randomised reference draw, shared by every arm
# Test durations in seconds, so arms at different servo rates see the same
# physical test.
HOLD_S = 0.8
TRACK_S = 0.2
STEP_AT_S = 0.01
BIAS_S = 0.6
BIAS_ONSET_S = 0.1
# DC-response metrics (step and bias) smooth the error over this long so a
# high-frequency hold limit cycle does not mask the response.
SMOOTH_S = 0.01
# Energy above this frequency (Hz) marks a high-frequency limit cycle in
# the position and thrash in the command.
HF_CUTOFF_HZ = 200.0
STEP_AMPLITUDE_NM = 500.0
SETTLE_TOL_FRAC = 0.02
# Reference-spec parameters given per servo step, rescaled between rates.
PER_STEP_RATES = ("angular", "angular_step", "z_rate")
PER_STEP_INDICES = ("step_at",)
# Per-step parameters a kind falls back to a default for when omitted, so a
# retimed spec must state them. Families that can draw a spiral or helix
# also take a z rate.
IMPLICIT_PER_STEP = {
    "spiral": ("angular_step",),
    "mixed": ("angular", "z_rate"),
    "helix": ("angular", "z_rate"),
    "morph": ("angular", "z_rate"),
    "sequence": ("angular", "z_rate"),
    "line": ("angular",),
    "smooth": ("angular",),
}
# The servo rate the named track-shape presets (control_run.shape_spec)
# were written for.
SHAPE_PRESET_RATE_HZ = 10_000.0
# A bias offset counts as recovered once it stays inside this tolerance.
BIAS_RECOVERY_TOL_NM = 2.0
BIAS_TAIL_FRAC = 0.2
DEFAULT_BIAS = 10.0
MS = 1e3
PERCENT = 100.0


@dataclass(frozen=True)
class BenchTiming:
    """
    The benchmark's durations in steps at one servo rate.
    """

    hold: int
    track: int
    step_at: int
    bias: int
    bias_onset: int
    smooth: int

    @classmethod
    def at(cls, rate_hz: float) -> BenchTiming:
        def steps(seconds: float) -> int:
            return max(1, int(round(seconds * rate_hz)))

        return cls(
            hold=steps(HOLD_S),
            track=steps(TRACK_S),
            step_at=steps(STEP_AT_S),
            bias=steps(BIAS_S),
            bias_onset=steps(BIAS_ONSET_S),
            smooth=steps(SMOOTH_S),
        )


def retime_spec(spec: dict, from_hz: float, to_hz: float) -> dict:
    """
    A reference spec written for 'from_hz' restated for 'to_hz', so it
    draws the same physical trajectory: per-step rates shrink and step
    indices grow as the rate rises.
    """

    if from_hz == to_hz:
        return dict(spec)
    k = from_hz / to_hz
    kind = spec.get("kind", "step")
    missing = [p for p in IMPLICIT_PER_STEP.get(kind, ()) if p not in spec]
    if missing:
        raise ValueError(
            f"Retiming a '{kind}' reference needs explicit {missing}, "
            "its defaults are per step"
        )
    out = dict(spec)

    for key in PER_STEP_RATES:
        if key in out:
            v = out[key]
            out[key] = [x * k for x in v] if isinstance(v, list) else v * k

    for key in PER_STEP_INDICES:
        if key in out:
            out[key] = int(round(out[key] / k))
    if "durations" in out:
        # Segment lengths in steps, each at least one, kept summing to the
        # retimed horizon (the longest segment absorbs the rounding).
        steps = [max(1, int(round(d / k))) for d in out["durations"]]
        total = max(len(steps), int(round(sum(out["durations"]) / k)))
        steps[int(np.argmax(steps))] += total - sum(steps)
        out["durations"] = steps

    return out


@dataclass
class ArmResult:
    """
    One controller's benchmark: per-axis metrics plus cost and poles.
    """

    name: str
    architecture: str
    axes: list[str]
    metrics: dict[str, list[float]] = field(default_factory=dict)
    cost: dict = field(default_factory=dict)
    poles: dict | None = None


class BiasStep:
    """
    Actuator wrapper adding a constant DAC offset from step 'onset' on,
    after the arm's own actuator model (if any).
    """

    def __init__(self, offset: float, onset: int, inner=None):
        self.offset = offset
        self.onset = onset
        self.inner = inner
        self._k = 0

    def reset(self, command: torch.Tensor) -> None:
        self._k = 0
        if self.inner is not None:
            self.inner.reset(command)

    def apply(self, command: torch.Tensor) -> torch.Tensor:
        if self.inner is not None:
            command = self.inner.apply(command)
        self._k += 1

        return command + self.offset if self._k > self.onset else command


def hf_share(signal: np.ndarray, sample_rate: float) -> float:
    """
    Fraction of a 1-D signal's variance above HF_CUTOFF_HZ.
    """

    centred = signal - signal.mean()
    freqs, power = spectral.psd(centred, sample_rate)
    total = float(power.sum())

    return float(power[freqs >= HF_CUTOFF_HZ].sum()) / total if total else 0.0


def moving_average(x: np.ndarray, width: int) -> np.ndarray:
    """
    Trailing mean along axis 1 of [B, H, A]; the first 'width - 1' steps
    average what is available.
    """

    cs = np.cumsum(x, axis=1)
    out = cs.copy()
    out[:, width:] = cs[:, width:] - cs[:, :-width]
    counts = np.minimum(np.arange(1, x.shape[1] + 1), width)

    return out / counts[None, :, None]


def recovery_steps(offset: np.ndarray, tol: float) -> np.ndarray:
    """
    Per-rollout, per-axis steps until |offset| [B, H, A] stays within tol
    for the rest of the run; inf if it never does.
    """

    inside = np.abs(offset) <= tol
    # sustained[b, k, a]: inside from step k to the end.
    sustained = np.flip(np.cumprod(np.flip(inside, axis=1), axis=1), axis=1)
    horizon = offset.shape[1]
    first = np.where(sustained.any(axis=1), sustained.argmax(axis=1), horizon)

    return np.where(first < horizon, first, np.inf).astype(float)


def hold_metrics(
    err: np.ndarray, dac: np.ndarray, pos: np.ndarray, sample_rate: float
) -> dict[str, np.ndarray]:
    """
    Still-hold metrics from [B, H, A] error (nm), command and position;
    the first quarter is treated as settling transient.
    """

    q = err.shape[1] // 4
    steady = np.abs(err[:, q:])
    early = np.abs(err[:, q : 2 * q])
    late = np.abs(err[:, 3 * q :])
    n_axes = err.shape[2]

    def axis_pct(x: np.ndarray, pct: float) -> np.ndarray:
        return np.percentile(x.reshape(-1, n_axes), pct, axis=0)

    def per_rollout(fn, x: np.ndarray) -> np.ndarray:
        return np.median(
            [[fn(x[b, q:, a]) for a in range(n_axes)] for b in range(len(x))],
            axis=0,
        )

    return {
        "hold_p50_nm": axis_pct(steady, 50),
        "hold_p95_nm": axis_pct(steady, 95),
        "hold_stationarity": axis_pct(late, 95) / axis_pct(early, 95),
        "hold_pos_hf_pct": PERCENT
        * per_rollout(lambda s: hf_share(s, sample_rate), pos),
        "dac_std": per_rollout(np.std, dac),
        # Per millisecond, so arms at different rates compare.
        "dac_slew_rms": per_rollout(
            lambda s: float(np.sqrt(np.mean(np.diff(s) ** 2))), dac
        )
        * sample_rate
        / MS,
        "dac_hf_pct": PERCENT
        * per_rollout(lambda s: hf_share(s, sample_rate), dac),
    }


def step_metrics(
    err: np.ndarray, amplitude: float, dt: float, smooth: int
) -> dict[str, np.ndarray]:
    """
    Step response from the [B, H, A] error (nm) from the step on, read on
    the smoothed error: settle time within SETTLE_TOL_FRAC of the
    amplitude, and overshoot past the target as a percentage of it.
    """

    smoothed = moving_average(err, smooth)
    tol = abs(amplitude) * SETTLE_TOL_FRAC
    settle = recovery_steps(smoothed, tol) * dt * MS
    beyond = np.clip(smoothed * np.sign(amplitude), 0.0, None).max(axis=1)

    return {
        "step_settle_ms": np.median(settle, axis=0),
        "step_overshoot_pct": PERCENT
        * np.median(beyond, axis=0)
        / abs(amplitude),
    }


def bias_metrics(
    err: np.ndarray, dt: float, onset: int, smooth: int
) -> dict[str, np.ndarray]:
    """
    DC-offset rejection from a bias applied at step 'onset': the peak
    smoothed excursion, the residual steady offset and the recovery time,
    each relative to the pre-bias mean error.
    """

    pre = err[:, onset // 2 : onset].mean(axis=1, keepdims=True)
    offset = moving_average(err, smooth) - pre
    after = offset[:, onset:]
    tail = int(err.shape[1] * BIAS_TAIL_FRAC)
    residual = np.abs(err[:, -tail:].mean(axis=1) - pre[:, 0])
    recovery = recovery_steps(after, BIAS_RECOVERY_TOL_NM) * dt * MS

    return {
        "bias_peak_nm": np.median(np.abs(after).max(axis=1), axis=0),
        "bias_residual_nm": np.median(residual, axis=0),
        "bias_recovery_ms": np.median(recovery, axis=0),
    }


def shared_seeds(first: ControllerConfig) -> dict | None:
    """
    The first config's seed spec, for every arm, so all controllers start
    from the same states; with a workspace reference the seeds are kept
    inside its travel box, as in training.
    """

    quiescence = first.training.get("seed_quiescence")
    spec = first.training.get("reference", {})
    plant_cfg = RunConfiguration(first.plant_config_path)
    limits = reference_limits(first, plant_cfg, spec)
    if quiescence is None or limits is None:
        return quiescence
    box = torch.stack([limits.pos_lo, limits.pos_hi], 1)

    return {**quiescence, "bounds": box.tolist()}


class _Arm:
    """A loaded controller plus the plant, seeds and hardware it runs on."""

    def __init__(
        self,
        cfg_path: str,
        device: str,
        cache: dict,
        quiescence: dict | None = None,
    ):
        self.cfg = ControllerConfig(cfg_path)
        plant_cfg = RunConfiguration(self.cfg.plant_config_path)
        self.plant_cfg = plant_cfg
        if quiescence is None:
            quiescence = self.cfg.training.get("seed_quiescence")
        key = (
            self.cfg.plant_config_path,
            self.cfg.plant_checkpoint,
            json.dumps(quiescence, sort_keys=True),
        )
        if key not in cache:
            plant = Plant.from_checkpoint(
                plant_cfg, self.cfg.plant_checkpoint, device
            )
            loaders = rollout_splits_from_config(
                plant_cfg,
                RolloutLayout.from_config(plant_cfg),
                max_horizon=BenchTiming.at(self.cfg.servo_rate_hz).hold,
                batch_size=BATCH,
                seed=SEED_SELECT,
                device=device,
                quiescent_seed=quiescence,
            )
            # Seeds spread across the whole validation set: neighbouring
            # starts are near-identical states.
            val_loader = loaders.val_loader
            assert isinstance(val_loader, RolloutWindowLoader)
            warmup = val_loader.spread_warmups(BATCH).to(device)
            cache[key] = (plant, warmup)
        self.plant, self.warmup = cache[key]
        self.to_nm = torch.as_tensor(
            position_to_nm(plant_cfg), dtype=torch.float32, device=device
        )
        self.controller = Controller.from_checkpoint(self.cfg.save_path, device)
        self.sensor = build_sensor(self.cfg)
        self.actuator = build_actuator(self.cfg)
        self.origin, _ = self.plant.seed_state(self.warmup)
        self.device = device
        self.sample_rate = self.cfg.servo_rate_hz

    def roll(self, reference, ref_v, horizon, actuator=None):
        with torch.no_grad():
            pos, dac = self.plant.closed_loop_rollout(
                self.warmup,
                reference,
                self.controller.policy,
                horizon,
                reference_velocity=ref_v,
                sensor=self.sensor,
                actuator=actuator if actuator is not None else self.actuator,
            )

        return pos, dac

    def reference(
        self, spec: dict, horizon: int, generator: torch.Generator
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Draw 'spec' from this arm's seeds; a workspace reference also gets
        the arm's travel and command limits and its seeds' held command.
        """

        limits = reference_limits(self.cfg, self.plant_cfg, spec)
        gen = make_reference_gen(spec, self.device, limits)
        if isinstance(gen, WorkspaceReference):
            hold = self.plant.seed_command(self.warmup)

            return gen(self.origin, horizon, generator, hold=hold)

        return gen(self.origin, horizon, generator)

    def hold_reference(self, horizon: int) -> torch.Tensor:
        b, a = self.origin.shape

        return self.origin[:, None, :].expand(b, horizon, a).contiguous()

    def err_nm(self, pos, reference) -> np.ndarray:
        return ((pos - reference) * self.to_nm).cpu().numpy()


def benchmark_arm(
    cfg_path: str,
    device: str,
    cache: dict,
    bias: float,
    mixed_spec: dict,
    spec_rate_hz: float,
    quiescence: dict | None = None,
) -> ArmResult:
    """
    Run the suite on one controller. 'mixed_spec' is written for
    'spec_rate_hz' and retimed to the arm's own servo rate, as are the
    durations, so every arm sees the same physical test.
    """

    arm = _Arm(cfg_path, device, cache, quiescence)
    cfg = arm.cfg
    rate = arm.sample_rate
    dt = 1.0 / rate
    t = BenchTiming.at(rate)
    result = ArmResult(cfg.model_name, cfg.architecture, list(cfg.axes))
    metrics: dict[str, np.ndarray] = {}
    logger.info("Benchmarking %s (%s)", cfg.model_name, cfg.architecture)

    ref = arm.hold_reference(t.hold)
    pos, dac = arm.roll(ref, torch.zeros_like(ref), t.hold)
    metrics |= hold_metrics(
        arm.err_nm(pos, ref),
        dac.cpu().numpy(),
        (pos * arm.to_nm).cpu().numpy(),
        rate,
    )
    specs = (
        ("spiral", shape_spec("spiral"), SHAPE_PRESET_RATE_HZ),
        ("mixed", mixed_spec, spec_rate_hz),
    )

    for tag, spec, written_hz in specs:
        gen = torch.Generator(device=device).manual_seed(REFERENCE_SEED)
        retimed = retime_spec(spec, written_hz, rate)
        ref, ref_v = arm.reference(retimed, t.track, gen)
        pos, _ = arm.roll(ref, ref_v, t.track)
        rms = tracking_percentiles(pos * arm.to_nm, ref * arm.to_nm)[50]
        metrics[f"{tag}_rms_nm"] = rms.cpu().numpy()

    amp = STEP_AMPLITUDE_NM / arm.to_nm
    ref = step_reference(arm.origin, amp, t.track, t.step_at)
    pos, _ = arm.roll(ref, torch.zeros_like(ref), t.track)
    err = arm.err_nm(pos[:, t.step_at :], ref[:, t.step_at :])
    metrics |= step_metrics(err, STEP_AMPLITUDE_NM, dt, t.smooth)

    ref = arm.hold_reference(t.bias)
    step = BiasStep(bias, t.bias_onset, inner=arm.actuator)
    pos, _ = arm.roll(ref, torch.zeros_like(ref), t.bias, actuator=step)
    metrics |= bias_metrics(arm.err_nm(pos, ref), dt, t.bias_onset, t.smooth)

    result.metrics = {
        k: [float(v) for v in vals] for k, vals in metrics.items()
    }
    net = arm.controller.net
    report = score_controller(net.resource_layers(), cfg.servo_rate_hz)
    result.cost = {
        "params": report.params,
        "bram_bits": report.bram_bits,
        "dsp_cycles": report.dsp_cycles,
        "cycles_needed": report.cycles_needed,
        "fits": report.fits,
    }
    if isinstance(net, RecurrentControllerNet):
        result.poles = net.modes.describe()

    return result


# Report order and headings for the per-axis metrics.
METRIC_TITLES = {
    "hold_p50_nm": "Hold |error| P50 (nm)",
    "hold_p95_nm": "Hold |error| P95 (nm)",
    "hold_stationarity": "Hold stationarity (late / early P95, 1 = flat)",
    "hold_pos_hf_pct": f"Hold position energy above {HF_CUTOFF_HZ:g} Hz (%)",
    "dac_std": "Hold DAC std",
    "dac_slew_rms": "Hold DAC slew rms (counts per ms)",
    "dac_hf_pct": f"Hold DAC energy above {HF_CUTOFF_HZ:g} Hz (%)",
    "spiral_rms_nm": "Spiral tracking RMS P50 (nm)",
    "mixed_rms_nm": "Mixed-trajectory tracking RMS P50 (nm)",
    "step_settle_ms": "Step settle time, smoothed (ms, inf = never)",
    "step_overshoot_pct": "Step overshoot, smoothed (%)",
    "bias_peak_nm": "Bias step: peak DC excursion (nm)",
    "bias_residual_nm": "Bias step: residual DC offset (nm, 0 = rejected)",
    "bias_recovery_ms": "Bias step: recovery time (ms, inf = never)",
}


def _metric_tables(results: list[ArmResult]) -> list[tuple[str, list[str]]]:
    names = [r.name for r in results]
    axes = results[0].axes

    return [
        (
            title,
            render_table(
                names,
                axes,
                [r.metrics[key] for r in results],
                corner_label="controller",
            ),
        )
        for key, title in METRIC_TITLES.items()
    ]


def _cost_table(results: list[ArmResult]) -> list[str]:
    cols = ["params", "bram_bits", "dsp_cycles", "cycles_needed", "fits"]
    rows = [[str(r.cost[c]) for c in cols] for r in results]

    return render_table(
        [r.name for r in results], cols, rows, corner_label="controller"
    )


def run_controller_bench(
    cfg_paths: list[str],
    out_dir: str | None = None,
    bias: float = DEFAULT_BIAS,
) -> str:
    """
    Benchmark each controller config and write report.md plus
    results.json; returns the report path.
    """

    if not cfg_paths:
        raise ValueError("Pass at least one controller config")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Rollouts are float32: keep TF32 from coarsening the nm-level states.
    strict_float32()
    out = out_dir or os.path.join("data_analysis", "controller_bench")
    os.makedirs(out, exist_ok=True)
    # Every arm tracks the first arm's training trajectory distribution,
    # retimed to its own servo rate.
    first = ControllerConfig(cfg_paths[0])
    mixed_spec = first.training.get("reference", {})
    seeds = shared_seeds(first)
    cache: dict = {}
    results = [
        benchmark_arm(
            p, device, cache, bias, mixed_spec, first.servo_rate_hz, seeds
        )
        for p in cfg_paths
    ]

    tables = _metric_tables(results)
    sections = [
        {"title": "Per-axis metrics", "tables": tables},
        {"title": "FPGA cost", "table": _cost_table(results)},
    ]
    poles = [f"- {r.name}: {json.dumps(r.poles)}" for r in results if r.poles]
    if poles:
        sections.append({"title": "Learned poles (quantised)", "notes": poles})
    header = [
        f"- {len(results)} controllers, {BATCH} quiescent seeds each "
        "(shared by arms on the same plant and seed settings), "
        f"{HOLD_S:g} s holds at reanchor 0, durations in "
        "seconds so arms at different servo rates see the same test",
        f"- Bias step {bias:g} DAC units at {BIAS_ONSET_S:g} s "
        "(evaluation stimulus only)",
        f"- Mixed trajectories from {os.path.basename(cfg_paths[0])}'s "
        "training reference",
    ]
    path = write_report("Controller benchmark", header, sections, out)
    with open(os.path.join(out, "results.json"), "w") as f:
        json.dump([r.__dict__ for r in results], f, indent=2)

    for title, table in tables:
        logger.info("%s", title)

        for line in table:
            logger.info("%s", line)
    logger.info("Report saved to %s", path)

    return path


# Endurance: how long a controller stays put, far past any training horizon.
ENDURANCE_S = 30.0
ENDURANCE_MARKS_S = (0.5, 1.0, 5.0, 10.0, 30.0)
# Error RMS is read over this window ending at each mark.
ENDURANCE_WINDOW_S = 0.1
# A command within this fraction of a rail counts as railed.
RAIL_FRACTION = 0.99
# A rollout whose windowed error passes this at any mark has diverged.
DIVERGED_NM = 1000.0


def endurance_marks(err: np.ndarray, rate_hz: float) -> np.ndarray:
    """
    Per-axis-max windowed RMS |error| [B, H, A] at each ENDURANCE_MARKS_S
    within the horizon, as [B, marks].
    """

    width = max(1, int(round(ENDURANCE_WINDOW_S * rate_hz)))
    out = []

    for mark in ENDURANCE_MARKS_S:
        end = min(err.shape[1], int(round(mark * rate_hz)))
        win = err[:, max(0, end - width) : end]
        out.append(np.sqrt(np.mean(win**2, axis=1)).max(axis=1))

    return np.stack(out, axis=1)


def rail_share(dac: np.ndarray, safe_range: np.ndarray) -> float:
    """
    Share of command samples [B, H, A] within RAIL_FRACTION of a rail.
    """

    lo, hi = safe_range[:, 0], safe_range[:, 1]
    # A limit at zero is the stage at rest, not a rail.
    railed = ((hi > 0) & (dac >= RAIL_FRACTION * hi)) | (
        (lo < 0) & (dac <= RAIL_FRACTION * lo)
    )

    return float(railed.mean())


def endurance_arm(
    cfg_path: str,
    device: str,
    cache: dict,
    bias: float,
    spec: dict,
    spec_rate_hz: float,
    quiescence: dict | None = None,
) -> dict:
    """
    One controller held for ENDURANCE_S on a biased hold, a step and a
    bounded mix of trajectories from 'spec' (authored at 'spec_rate_hz'),
    with the z ramp zeroed so a long run stays inside the travel.
    """

    arm = _Arm(cfg_path, device, cache, quiescence)
    rate = arm.sample_rate
    h = int(round(ENDURANCE_S * rate))
    t = BenchTiming.at(rate)
    safe = arm.cfg.safe_range.cpu().numpy()
    result: dict = {"name": arm.cfg.model_name, "tests": {}}
    logger.info("Endurance %s: %d steps per test", arm.cfg.model_name, h)

    ref = arm.hold_reference(h)
    step = BiasStep(bias, t.bias_onset, inner=arm.actuator)
    pos, dac = arm.roll(ref, torch.zeros_like(ref), h, actuator=step)
    tests = {"bias hold": (arm.err_nm(pos, ref), dac)}

    amp = STEP_AMPLITUDE_NM / arm.to_nm
    ref = step_reference(arm.origin, amp, h, t.step_at)
    pos, dac = arm.roll(ref, torch.zeros_like(ref), h)
    tests["step"] = (arm.err_nm(pos, ref), dac)

    spec = {**spec, "z_rate": [0.0, 0.0]}
    gen = torch.Generator(device=device).manual_seed(REFERENCE_SEED)
    ref, ref_v = arm.reference(retime_spec(spec, spec_rate_hz, rate), h, gen)
    pos, dac = arm.roll(ref, ref_v, h)
    tests["bounded mix"] = (arm.err_nm(pos, ref), dac)

    for name, (err, dac) in tests.items():
        cmd = dac.cpu().numpy()
        rms = endurance_marks(err, rate)
        result["tests"][name] = {
            "rms_nm_at": np.median(rms, axis=0).tolist(),
            "worst_nm_at": rms.max(axis=0).tolist(),
            "diverged": int(
                (~np.isfinite(rms) | (rms > DIVERGED_NM)).any(axis=1).sum()
            ),
            "rail_share": rail_share(cmd, safe),
        }

    return result


def run_controller_endurance(
    cfg_paths: list[str],
    out_dir: str | None = None,
    bias: float = DEFAULT_BIAS,
) -> str:
    """
    Endurance-test each controller config and write report.md plus
    results.json; returns the report path.
    """

    if not cfg_paths:
        raise ValueError("Pass at least one controller config")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Rollouts are float32: keep TF32 from coarsening the nm-level states.
    strict_float32()
    out = out_dir or os.path.join("data_analysis", "controller_endurance")
    os.makedirs(out, exist_ok=True)
    # Every arm sees the first config's references, retimed to its rate.
    first = ControllerConfig(cfg_paths[0])
    spec = first.training.get("reference", {})
    seeds = shared_seeds(first)
    cache: dict = {}
    results = [
        endurance_arm(p, device, cache, bias, spec, first.servo_rate_hz, seeds)
        for p in cfg_paths
    ]
    marks = [f"{m:g} s" for m in ENDURANCE_MARKS_S]
    tables = []

    for test in ("bias hold", "step", "bounded mix"):
        rows = [
            [
                *(
                    f"{m:.3g} / {w:.3g}"
                    for m, w in zip(
                        r["tests"][test]["rms_nm_at"],
                        r["tests"][test]["worst_nm_at"],
                        strict=True,
                    )
                ),
                f"{r['tests'][test]['diverged']}/{BATCH}",
                f"{PERCENT * r['tests'][test]['rail_share']:.1f}",
            ]
            for r in results
        ]
        tables.append(
            (
                f"{test}: worst-axis RMS |error| (nm, median / worst seed) "
                f"over the last {ENDURANCE_WINDOW_S:g} s before each mark, "
                f"seeds past {DIVERGED_NM:g} nm, and % of the command on "
                "a rail",
                render_table(
                    [r["name"] for r in results],
                    [*marks, "diverged", "railed %"],
                    rows,
                    corner_label="controller",
                ),
            )
        )
    header = [
        f"- {len(results)} controllers, {BATCH} quiescent seeds each, "
        "spread across the validation set, "
        f"{ENDURANCE_S:g} s per test at reanchor 0",
        f"- Bias {bias:g} DAC units from {BIAS_ONSET_S:g} s; step "
        f"{STEP_AMPLITUDE_NM:g} nm at {STEP_AT_S:g} s; mixed references with "
        "the z ramp zeroed so they stay inside the travel",
    ]
    sections = [{"title": "Endurance", "tables": tables}]
    path = write_report("Controller endurance", header, sections, out)
    with open(os.path.join(out, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    for title, table in tables:
        logger.info("%s", title)

        for line in table:
            logger.info("%s", line)
    logger.info("Report saved to %s", path)

    return path
