"""Rollout steppers.

WindowedStepper reproduces the manual sliding-window forward.
"""

import torch
import torch.nn as nn

from nn_motion_control.plant.rollout import RolloutStepper, WindowedStepper


class _SumLastFrame(nn.Module):
    """Deterministic window model.

    Sum of the last frame's features, output shape [B, 1].
    """

    def forward(self, window: torch.Tensor) -> torch.Tensor:
        return window[:, :, -1].sum(dim=1, keepdim=True)


def test_windowed_stepper_matches_manual_slide():
    torch.manual_seed(0)
    b, f, w = 3, 4, 5
    model = _SumLastFrame()
    warmup = torch.randn(b, f, w)
    frames = [torch.randn(b, f) for _ in range(3)]

    stepper = WindowedStepper(model)
    got = [stepper.reset(warmup)]

    for fr in frames:
        got.append(stepper.step(fr))

    # Manual reference: slide the window and re-run the model each
    # step.
    window = warmup
    want = [model(window)]

    for fr in frames:
        window = torch.cat([window[:, :, 1:], fr.unsqueeze(-1)], dim=2)
        want.append(model(window))

    for g, w_ in zip(got, want, strict=True):
        assert torch.equal(g, w_)


def test_windowed_stepper_satisfies_protocol():
    assert isinstance(WindowedStepper(_SumLastFrame()), RolloutStepper)
