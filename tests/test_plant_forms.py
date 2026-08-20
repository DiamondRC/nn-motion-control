"""Rollout reconstruction: delta integrates, absolute feeds prediction."""

import torch
import torch.nn as nn

from nn_motion_control.data.normalize import NormStats
from nn_motion_control.plant.plant import Plant, RolloutLayout


class _Const(nn.Module):
    """A stub one-step model that ignores its input and returns a constant."""

    def __init__(self, val: float):
        super().__init__()
        self.val = val

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.full((x.shape[0], 3), self.val)


def _plant(val: float, predicts_delta: bool) -> Plant:
    layout = RolloutLayout([0, 3, 6], [1, 4, 7], [2, 5, 8], 9)
    ones9 = torch.ones(9)
    ident = NormStats(mean=torch.zeros(9), std=ones9, normalizable=ones9.bool())
    tgt = NormStats(
        mean=torch.zeros(3),
        std=torch.ones(3),
        normalizable=torch.ones(3).bool(),
    )

    return Plant(_Const(val), ident, tgt, layout, predicts_delta=predicts_delta)


def test_reconstruction_forms_differ():
    # Identity stats and a zero warmup, so the model output 'val' is the
    # physical prediction: delta accumulates it, absolute holds it.
    warmup = torch.zeros(2, 9, 4)
    dac = torch.zeros(2, 5, 3)
    val = 7.0
    absolute = _plant(val, predicts_delta=False).roll_forward(warmup, dac, 5)
    delta = _plant(val, predicts_delta=True).roll_forward(warmup, dac, 5)
    steps = torch.arange(1, 6, dtype=torch.float32).view(1, 5, 1)
    assert torch.allclose(absolute, torch.full_like(absolute, val))
    assert torch.allclose(delta, steps * val)  # cur + val each step
