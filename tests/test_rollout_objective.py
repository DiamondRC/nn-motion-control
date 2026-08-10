"""
Rollout objective + schedules: anneal, curriculum, horizon weights, joint loss.
"""

import pytest
import torch

from nn_motion_control.training.rollout import (
    RolloutTrainer,
    curriculum_horizon,
    horizon_weights,
    linear_schedule,
    rollout_loss,
)


def test_linear_schedule_endpoints_and_hold():
    assert linear_schedule(0, 1.0, 0.0, 10) == 1.0
    assert linear_schedule(5, 1.0, 0.0, 10) == pytest.approx(0.5)
    assert linear_schedule(10, 1.0, 0.0, 10) == 0.0
    assert linear_schedule(50, 1.0, 0.0, 10) == 0.0  # held past the end
    assert linear_schedule(3, 1.0, 0.0, 0) == 0.0  # degenerate -> end


def test_curriculum_grows_then_holds():
    assert curriculum_horizon(0, 4, 32, 20) == 4
    assert curriculum_horizon(20, 4, 32, 20) == 32
    assert curriculum_horizon(100, 4, 32, 20) == 32  # clamped
    assert curriculum_horizon(10, 4, 32, 20) == 18  # halfway: 4 + 0.5*28
    assert curriculum_horizon(0, 4, 32, 0) == 32  # no ramp -> max


def test_horizon_weights_normalised_and_shaped():
    for mode in ("uniform", "discount", "increasing"):
        w = horizon_weights(8, mode)
        assert torch.allclose(w.sum(), torch.tensor(1.0))
        assert w.shape == (8,)
    assert torch.allclose(horizon_weights(4, "uniform"), torch.full((4,), 0.25))
    assert (horizon_weights(5, "discount").diff() < 0).all()  # decreasing
    assert (horizon_weights(5, "increasing").diff() > 0).all()  # increasing
    with pytest.raises(ValueError, match="weighting mode"):
        horizon_weights(4, "bogus")


def _loss_inputs(b=3, h=5, a=2):
    gt = torch.randn(b, h, a)
    seed = torch.randn(b, a)
    w = horizon_weights(h, "uniform")
    return gt, seed, w


def test_perfect_prediction_is_zero_loss():
    gt, seed, w = _loss_inputs()
    assert rollout_loss(
        gt.clone(), gt, seed, w, step_weight=0.5
    ).item() == pytest.approx(0.0, abs=1e-6)


def test_lacc_is_weighted_position_mse():
    gt, seed, w = _loss_inputs()
    preds = gt + 2.0  # constant offset
    # uniform weights, offset 2 -> per-step MSE = 4, weighted sum = 4.
    assert rollout_loss(preds, gt, seed, w, step_weight=0.0).item() == pytest.approx(
        4.0
    )


def test_axis_weights_rebalance_per_axis():
    # Axis 0 has error 1 (sq 1), axis 1 has error 3 (sq 9); mean over axes = 5.
    b, h, a = 4, 3, 2
    gt = torch.zeros(b, h, a)
    preds = torch.zeros(b, h, a)
    preds[..., 0] = 1.0
    preds[..., 1] = 3.0
    seed = torch.zeros(b, a)
    w = horizon_weights(h, "uniform")
    assert rollout_loss(preds, gt, seed, w, 0.0).item() == pytest.approx(5.0)
    # Uniform axis weights (mean 1) reproduce the unweighted loss.
    assert rollout_loss(
        preds, gt, seed, w, 0.0, axis_weights=torch.ones(a)
    ).item() == pytest.approx(5.0)
    # Favour the low-error axis: (1*1.5 + 9*0.5)/2 = 3.
    aw = torch.tensor([1.5, 0.5])
    assert rollout_loss(
        preds, gt, seed, w, 0.0, axis_weights=aw
    ).item() == pytest.approx(3.0, abs=1e-6)


def test_step_term_adds_only_with_positive_weight():
    gt, seed, w = _loss_inputs()
    preds = gt + torch.randn_like(gt) * 0.1
    base = rollout_loss(preds, gt, seed, w, step_weight=0.0)
    joint = rollout_loss(preds, gt, seed, w, step_weight=1.0)
    assert joint.item() > base.item()  # the increment term is non-negative and active


class _RecordingPlant:
    """Stub plant capturing the horizon and scheduled-sampling prob it is rolled at."""

    class _Layout:
        pos_cols = [0]

    def __init__(self):
        self.layout = self._Layout()
        self.model = torch.nn.Identity()
        self.calls = []

    def roll_forward(
        self, warmup, dac, horizon, teacher_pos=None, ss_prob=0.0, generator=None
    ):
        del dac, teacher_pos, generator  # signature-matching stub
        self.calls.append((horizon, ss_prob))
        return torch.zeros(warmup.shape[0], horizon, len(self.layout.pos_cols))


def _bare_rollout_trainer(plant, max_horizon=16, cur_h=4, cur_ss=1.0):
    # Bypass the heavy base Trainer __init__; set only what _forward_loss touches.
    rt = RolloutTrainer.__new__(RolloutTrainer)
    rt.plant = plant
    rt.max_horizon = max_horizon
    rt._cur_h, rt._cur_ss = cur_h, cur_ss
    rt._eval_mode = False
    rt.hw_mode, rt.step_weight, rt.device, rt._ss_gen = "uniform", 0.0, "cpu", None
    rt._weight_cache = {}
    rt.axis_weights = None
    return rt


def test_horizon_weights_are_cached_per_horizon():
    rt = _bare_rollout_trainer(_RecordingPlant())
    first = rt._weights_for(8)
    assert torch.equal(first, horizon_weights(8, "uniform"))
    assert rt._weights_for(8) is first  # same object -> not rebuilt
    assert rt._weights_for(4) is not first  # distinct horizon -> distinct vector


def test_validation_grades_free_run_at_full_horizon():
    # Training uses the curriculum H and current scheduled-sampling prob; validation
    # must instead free-run (ss=0) at the full horizon so early stopping tracks the
    # deployment drift rather than favouring the earliest teacher-forced short-H epoch.
    plant = _RecordingPlant()
    rt = _bare_rollout_trainer(plant, max_horizon=16, cur_h=4, cur_ss=1.0)
    warmup = torch.zeros(2, 3, 5)
    batch = (warmup, torch.zeros(2, 16, 1), torch.zeros(2, 16, 1))

    rt._forward_loss(batch)  # training path
    assert plant.calls[-1] == (4, 1.0)

    rt._eval_mode = True  # what _validate_epoch sets
    rt._forward_loss(batch)
    assert plant.calls[-1] == (16, 0.0)
