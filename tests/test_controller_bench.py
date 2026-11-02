"""
control.bench: the benchmark's stimulus wrapper and metric helpers.
"""

import math

import numpy as np
import pytest
import torch

from nn_motion_control.control.bench import (
    BenchTiming,
    BiasStep,
    bias_metrics,
    hf_share,
    hold_metrics,
    moving_average,
    recovery_steps,
    retime_spec,
    step_metrics,
)
from nn_motion_control.control.hardware import ActuatorModel

RATE = 10000.0
TIMING = BenchTiming.at(RATE)
BIAS_ONSET = TIMING.bias_onset


def test_bias_step_applies_only_after_onset():
    step = BiasStep(offset=3.0, onset=2)
    step.reset(torch.zeros(1, 1))
    outs = [float(step.apply(torch.zeros(1, 1))) for _ in range(4)]
    assert outs == [0.0, 0.0, 3.0, 3.0]
    step.reset(torch.zeros(1, 1))  # a reset restarts the onset count
    assert float(step.apply(torch.zeros(1, 1))) == 0.0


def test_bias_step_chains_the_arm_actuator():
    inner = ActuatorModel(cutoff_hz=25.0, servo_rate_hz=RATE)
    step = BiasStep(offset=1.0, onset=0, inner=inner)
    step.reset(torch.zeros(1, 1))
    out = float(step.apply(torch.ones(1, 1)))
    # The inner low-pass moves a fraction of the way, then the bias adds.
    assert 1.0 < out < 2.0


def test_hf_share_separates_noise_from_slow_motion():
    rng = np.random.default_rng(0)
    t = np.arange(8000) / RATE
    assert hf_share(rng.standard_normal(8000), RATE) > 0.9
    assert hf_share(np.sin(2 * math.pi * 5.0 * t), RATE) < 0.01


def test_moving_average_is_a_trailing_mean():
    x = np.arange(6, dtype=float).reshape(1, 6, 1)
    out = moving_average(x, 3)[0, :, 0]
    assert np.allclose(out, [0.0, 0.5, 1.0, 2.0, 3.0, 4.0])


def test_recovery_steps_waits_for_a_sustained_return():
    offset = np.array([5, 1, 5, 1, 1, 1], dtype=float).reshape(1, 6, 1)
    assert recovery_steps(offset, 2.0)[0, 0] == 3  # the dip at 1 is not it
    never = np.full((1, 6, 1), 5.0)
    assert math.isinf(recovery_steps(never, 2.0)[0, 0])


def _bias_err(steps, decay):
    # A zero-mean hold, then a unit-scaled offset from the onset that
    # either decays away (rejected) or persists (proportional at DC).
    k = np.arange(steps, dtype=float)
    shift = np.where(k >= BIAS_ONSET, 10.0, 0.0)
    if decay:
        shift *= np.exp(-np.clip(k - BIAS_ONSET, 0, None) / 200.0)

    return np.tile(shift[None, :, None], (2, 1, 1))


def test_bias_metrics_separate_rejection_from_offset():
    def metrics(err):
        return bias_metrics(err, 1.0 / RATE, BIAS_ONSET, TIMING.smooth)

    rejected = metrics(_bias_err(6000, decay=True))
    stuck = metrics(_bias_err(6000, decay=False))
    assert rejected["bias_residual_nm"][0] < 1e-3
    assert math.isclose(stuck["bias_residual_nm"][0], 10.0, rel_tol=1e-6)
    assert np.isfinite(rejected["bias_recovery_ms"][0])
    assert math.isinf(stuck["bias_recovery_ms"][0])


def test_hold_metrics_flat_noise_is_stationary():
    rng = np.random.default_rng(1)
    err = rng.standard_normal((4, 4000, 2))
    dac = rng.standard_normal((4, 4000, 2))
    m = hold_metrics(err, dac, err, RATE)
    assert m["hold_p95_nm"].shape == (2,)
    assert np.allclose(m["hold_stationarity"], 1.0, atol=0.1)
    assert np.all(m["dac_hf_pct"] > 90.0)  # white command is all thrash


def test_step_metrics_read_through_a_limit_cycle():
    # A 500 nm step that overshoots by about 30% and settles, under a 15 nm
    # 1 kHz dither far wider than the 2% (10 nm) settle band.
    k = np.arange(4000, dtype=float)
    response = -500.0 * np.exp(-k / 150.0) * np.cos(k / 60.0)
    dither = 15.0 * np.sin(2 * math.pi * 1000.0 * k / RATE)
    err = np.tile((response + dither)[None, :, None], (2, 1, 1))
    m = step_metrics(err, 500.0, 1.0 / RATE, TIMING.smooth)
    assert np.isfinite(m["step_settle_ms"][0])  # the raw error never settles
    assert 5.0 < m["step_overshoot_pct"][0] < 60.0


def test_timing_is_the_same_physical_test_at_any_rate():
    # At 10 kHz the durations are the original step counts.
    assert TIMING == BenchTiming(
        hold=8000,
        track=2000,
        step_at=100,
        bias=6000,
        bias_onset=1000,
        smooth=100,
    )
    fast = BenchTiming.at(2 * RATE)
    assert fast.hold == 2 * TIMING.hold and fast.smooth == 2 * TIMING.smooth


def test_retime_spec_keeps_the_physical_trajectory():
    spec = {
        "kind": "mixed",
        "angular": [0.002, 0.02],
        "z_rate": [-20.0, 20.0],
        "radius": [100.0, 1200.0],
        "step_at": 100,
    }
    assert retime_spec(spec, RATE, RATE) == spec
    fast = retime_spec(spec, RATE, 2 * RATE)
    # Per-step rates halve, step indices double, physical sizes stay.
    assert fast["angular"] == [0.001, 0.01]
    assert fast["z_rate"] == [-10.0, 10.0]
    assert fast["step_at"] == 200
    assert fast["radius"] == spec["radius"]


def test_retime_spec_refuses_implicit_per_step_defaults():
    with pytest.raises(ValueError, match="explicit"):
        retime_spec({"kind": "mixed", "radius": [1.0, 2.0]}, RATE, 2 * RATE)


def test_dac_slew_is_per_millisecond():
    # The same ramp in counts per second gives the same slew at any rate.
    noise = np.random.default_rng(3).standard_normal((2, 4000, 1))

    def slew(rate):
        k = np.arange(4000)
        dac = np.tile((k * 1000.0 / rate)[None, :, None], (2, 1, 1))
        return hold_metrics(noise, dac, noise, rate)["dac_slew_rms"][0]

    assert math.isclose(slew(RATE), slew(2 * RATE), rel_tol=1e-9)
    assert math.isclose(slew(RATE), 1.0, rel_tol=1e-9)  # 1000 counts/s


def test_retime_spec_rebalances_sequence_durations():
    spec = {"kind": "sequence", "angular": [0.01, 0.02], "z_rate": [0.0, 1.0]}
    spec["durations"] = [333, 333, 334]
    fast = retime_spec(spec, RATE, 2 * RATE)
    assert fast["durations"] == [666, 666, 668]
    assert sum(fast["durations"]) == 2 * sum(spec["durations"])


def test_retime_spec_asks_only_for_the_rates_a_kind_uses():
    # A line never draws a helix, so it needs no z rate.
    line = retime_spec(
        {"kind": "line", "angular": [0.01, 0.02]}, RATE, 2 * RATE
    )
    assert line["angular"] == [0.005, 0.01]
    with pytest.raises(ValueError, match="angular_step"):
        retime_spec({"kind": "spiral", "radius": 10.0}, RATE, 2 * RATE)


def test_spiral_preset_is_retimed_from_its_own_rate():
    # The preset has a fixed authoring rate, so its physical speed does not
    # depend on which arm happens to be benchmarked first.
    from nn_motion_control.control.bench import SHAPE_PRESET_RATE_HZ
    from nn_motion_control.training.control_run import shape_spec

    preset = shape_spec("spiral")
    at_20k = retime_spec(preset, SHAPE_PRESET_RATE_HZ, 20_000.0)
    k = SHAPE_PRESET_RATE_HZ / 20_000.0
    assert at_20k["angular_step"] == pytest.approx(preset["angular_step"] * k)


def test_retime_spec_never_rounds_a_segment_away():
    spec = {"kind": "sequence", "angular": [1, 2], "z_rate": [0, 1]}
    spec["durations"] = [1, 1, 1]
    slow = retime_spec(spec, 2 * RATE, RATE)
    assert all(d >= 1 for d in slow["durations"])


def test_endurance_marks_read_the_window_before_each_mark():
    from nn_motion_control.control.bench import (
        ENDURANCE_MARKS_S,
        endurance_marks,
    )

    rate = 1000.0
    n = int(ENDURANCE_MARKS_S[-1] * rate)
    err = np.zeros((2, n, 2))
    err[:, n // 2 :, 1] = 3.0  # the second axis drifts off halfway
    err[1, n // 2 :, 0] = 5.0  # one seed is worse
    marks = endurance_marks(err, rate)
    assert marks.shape == (2, len(ENDURANCE_MARKS_S))
    assert (marks[:, 0] == 0.0).all()  # 0.5 s, before the drift
    assert math.isclose(marks[0, -1], 3.0)  # worst axis at the end
    assert math.isclose(marks[1, -1], 5.0)


def test_rail_share_counts_commands_at_either_limit():
    from nn_motion_control.control.bench import rail_share

    safe = np.array([[-10.0, 20.0]])
    dac = np.array([[[0.0], [19.9], [-9.95], [5.0]]])
    assert rail_share(dac, safe) == 0.5


def test_rail_share_ignores_a_limit_at_zero():
    from nn_motion_control.control.bench import rail_share

    safe = np.array([[0.0, 20.0]])
    dac = np.array([[[0.0], [0.0], [19.9], [5.0]]])
    assert rail_share(dac, safe) == 0.25
