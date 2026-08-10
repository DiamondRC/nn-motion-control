"""
eval.metrics: absolute/normalised error metrics and the acceptance (P95/std) gate,
with P99 retained as a tail / discontinuity indicator.
"""

import numpy as np
import torch

from nn_motion_control.eval.evaluate import to_flat_numpy
from nn_motion_control.eval.metrics import DEFAULT_GATE, channel_metrics


def test_to_flat_numpy_upcasts_bf16_and_half():
    # numpy has no bf16/half; the eval loop must upcast before .numpy() (regression:
    # bf16 configs previously crashed on the targets conversion).
    for dt in (torch.bfloat16, torch.float16, torch.float32):
        out = to_flat_numpy(torch.ones(2, 3, dtype=dt))
        assert out.shape == (6,)
        assert out.dtype == np.float32


def test_perfect_prediction_passes_and_fits_100():
    tgt = np.linspace(-5, 5, 1000)
    m = channel_metrics("x", tgt.copy(), tgt.copy())
    assert m.mae == 0.0 and m.rmse == 0.0
    assert m.p95_abs == 0.0 and m.p99_abs == 0.0
    assert m.p99_frac == 0.0 and m.passes
    assert m.fit == 100.0


def test_absolute_errors_are_physical_units_not_percent():
    rng = np.random.default_rng(0)
    tgt = rng.normal(0, 100, size=100_000)  # std ~100 (e.g. counts)
    pred = tgt + 3.0  # constant 3-unit error
    m = channel_metrics("x", pred, tgt)
    # MAE / percentiles are in the target's units (~3), not a percentage.
    assert abs(m.mae - 3.0) < 0.05
    assert abs(m.p99_abs - 3.0) < 0.1
    # Normalised: 3 / ~100 ~ 3% -> within the 5% gate.
    assert abs(m.p99_frac - 0.03) < 0.005
    assert m.passes


def test_gate_fails_when_p95_exceeds_5pct():
    rng = np.random.default_rng(1)
    tgt = rng.normal(0, 10, size=100_000)
    pred = tgt + rng.normal(0, 1.0, size=tgt.size)  # noise std ~1 = 10% of target std
    m = channel_metrics("x", pred, tgt)
    assert m.p95_frac > DEFAULT_GATE  # the gate quantity itself is blown
    assert not m.passes


def test_heavy_p99_tail_alone_does_not_fail_gate():
    # P95 is the gate; P99 only flags the tail. Errors are tiny for 96% of samples but
    # a 3% burst is large: P95 stays within the gate, P99 blows past it -> still PASS.
    tgt = np.random.default_rng(3).normal(0, 10, size=100_000)
    err = np.full(tgt.size, 0.05)  # 0.5% of std for the bulk
    err[:3000] = 2.0  # top 3% ~ 20% of std -> lifts P99, not P95
    m = channel_metrics("x", tgt + err, tgt)
    assert m.p95_frac <= DEFAULT_GATE < m.p99_frac  # gate ok, tail flagged
    assert m.passes


def test_tracks_both_p95_and_p99():
    rng = np.random.default_rng(2)
    tgt = rng.normal(0, 10, size=200_000)
    pred = tgt + rng.normal(0, 0.5, size=tgt.size)
    m = channel_metrics("x", pred, tgt)
    # P99 tail error is strictly larger than P95.
    assert m.p99_abs > m.p95_abs
    assert m.p99_frac > m.p95_frac


def test_constant_target_has_no_scale_and_fails_gate():
    tgt = np.full(1000, 7.0)  # std == 0
    m = channel_metrics("x", tgt.copy(), tgt.copy())
    assert m.std == 0.0
    assert np.isnan(m.p99_frac)  # undefined normalisation
    assert not m.passes  # a NaN ratio never passes


def test_shape_mismatch_raises():
    import pytest

    with pytest.raises(ValueError, match="shape mismatch"):
        channel_metrics("x", np.zeros(10), np.zeros(11))
