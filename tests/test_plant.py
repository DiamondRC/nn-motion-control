"""
Plant rollout: layout derivation, trajectory reconstruction and truth-anchoring.
"""

import pytest
import torch

from nn_motion_control.core.config import RunConfiguration
from nn_motion_control.data.normalize import NormStats
from nn_motion_control.plant.plant import Plant, RolloutLayout


class _ConstDelta(torch.nn.Module):
    """A stand-in model that always predicts a fixed (normalised) delta."""

    def __init__(self, value: float, n_out: int = 1):
        super().__init__()
        self.value, self.n_out = value, n_out

    def forward(self, window):
        return torch.full((window.shape[0], self.n_out), self.value)


def _identity_plant(const=0.5):
    # 1 axis, F=3 inputs [pos, vel, dac], A=1. Identity norm so physical == normalised.
    layout = RolloutLayout(pos_cols=[0], vel_cols=[1], dac_cols=[2], n_features=3)
    ones, zeros = torch.ones(3), torch.zeros(3)
    in_stats = NormStats(mean=zeros, std=ones, normalizable=ones.bool())
    t_stats = NormStats(
        mean=torch.zeros(1), std=torch.ones(1), normalizable=torch.ones(1).bool()
    )
    return Plant(_ConstDelta(const), in_stats, t_stats, layout, device="cpu")


def test_shapes():
    plant = _identity_plant()
    warmup = torch.zeros(4, 3, 5)  # [B, F, W]
    dac = torch.zeros(4, 6, 1)  # [B, H, A]
    out = plant.roll_forward(warmup, dac, horizon=6)
    assert out.shape == (4, 6, 1)


def test_constant_delta_accumulates_linearly():
    # With identity norm and ΔP=c, free-running position at step k is (k+1)*c.
    c = 0.3
    plant = _identity_plant(c)
    warmup = torch.zeros(2, 3, 4)  # seed position 0
    dac = torch.zeros(2, 5, 1)
    out = plant.roll_forward(warmup, dac, horizon=5)[0, :, 0]  # [H]
    expected = torch.tensor([(k + 1) * c for k in range(5)])
    assert torch.allclose(out, expected, atol=1e-6)


def test_scheduled_sampling_feeds_truth_forward():
    # ss_prob=1 anchors every fed position to truth, so the prediction at step k is
    # teacher[k-1] + c (and c at step 0). Validates truth is fed, not the prediction.
    c = 0.2
    plant = _identity_plant(c)
    warmup = torch.zeros(1, 3, 4)
    dac = torch.zeros(1, 4, 1)
    teacher = torch.tensor([[10.0], [20.0], [30.0], [40.0]]).unsqueeze(0)  # [1, H, 1]
    out = plant.roll_forward(warmup, dac, 4, teacher_pos=teacher, ss_prob=1.0)[0, :, 0]
    expected = torch.tensor([c, 10.0 + c, 20.0 + c, 30.0 + c])
    assert torch.allclose(out, expected, atol=1e-6)


def test_free_running_ignores_teacher():
    # ss_prob=0 must reproduce the plain free-running trajectory even if teacher given.
    c = 0.25
    plant = _identity_plant(c)
    warmup, dac = torch.zeros(1, 3, 4), torch.zeros(1, 5, 1)
    teacher = torch.full((1, 5, 1), 99.0)
    out = plant.roll_forward(warmup, dac, 5, teacher_pos=teacher, ss_prob=0.0)[0, :, 0]
    assert torch.allclose(out, torch.tensor([(k + 1) * c for k in range(5)]), atol=1e-6)


def test_layout_from_deltabot_config():
    rc = RunConfiguration("examples/deltabot/configs/plant_tcn.json")
    layout = RolloutLayout.from_config(rc)
    # axis-major [pos, vel, dac] -> pos {0,3,6}, vel {1,4,7}, dac {2,5,8}.
    assert layout.pos_cols == [0, 3, 6]
    assert layout.vel_cols == [1, 4, 7]
    assert layout.dac_cols == [2, 5, 8]
    assert layout.n_features == 9


def test_layout_rejects_multiple_predicted_channels(config_factory):
    # The default factory predicts P,V,A,J -> more than one state channel.
    rc = RunConfiguration(config_factory())
    with pytest.raises(NotImplementedError, match="one state channel"):
        RolloutLayout.from_config(rc)
