"""Trainer early-stopping: the improvement threshold is relative (scale-invariant)."""

from nn_motion_control.training.trainer import Trainer


def _trainer(min_delta: float) -> Trainer:
    # Bypass the heavy __init__; _is_improvement only reads self.min_delta.
    t = Trainer.__new__(Trainer)
    t.min_delta = min_delta
    return t


def test_first_epoch_beats_inf():
    assert _trainer(0.01)._is_improvement(1.0, float("inf"))


def test_relative_threshold_is_scale_invariant():
    t = _trainer(0.01)  # require a 1% relative improvement
    # 2% better counts; 0.5% better does not — identically at any loss magnitude.
    assert t._is_improvement(0.98, 1.0)
    assert not t._is_improvement(0.995, 1.0)
    assert t._is_improvement(0.98e-5, 1.0e-5)
    assert not t._is_improvement(0.995e-5, 1.0e-5)


def test_zero_min_delta_saves_any_strict_improvement():
    t = _trainer(0.0)
    assert t._is_improvement(0.999999e-5, 1.0e-5)
    assert not t._is_improvement(1.0e-5, 1.0e-5)  # equal is not an improvement
