"""
control reference generators: step trajectory and the spiral kind.
"""

import pytest
import torch

from nn_motion_control.control.closed_loop import step_reference
from nn_motion_control.training.control_run import make_reference_gen


def test_step_reference_holds_then_steps():
    origin = torch.tensor([[1.0, 2.0]])  # [B=1, A=2]
    ref = step_reference(
        origin, amplitude=torch.tensor([0.5, -0.5]), horizon=5, step_at=2
    )
    assert ref.shape == (1, 5, 2)
    assert torch.equal(ref[0, 0], torch.tensor([1.0, 2.0]))  # before the step
    assert torch.equal(ref[0, 1], torch.tensor([1.0, 2.0]))
    assert torch.equal(ref[0, 2], torch.tensor([1.5, 1.5]))  # stepped
    assert torch.equal(ref[0, 4], torch.tensor([1.5, 1.5]))


def test_step_reference_needs_batch_for_1d():
    with pytest.raises(ValueError, match="Pass batch"):
        step_reference(torch.zeros(2), 1.0, 3)


def test_step_reference_broadcasts_batch():
    ref = step_reference(
        torch.zeros(2), amplitude=1.0, horizon=3, step_at=1, batch=4
    )
    assert ref.shape == (4, 3, 2)


def test_spiral_gen_anchors_on_origin_and_ramps_z():
    spec = {
        "kind": "spiral",
        "radius": 1000.0,
        "angular_step": 0.0031416,
        "xy": [0, 1],
        "z_rate": 30.0,
    }
    gen = make_reference_gen(spec, "cpu")
    origin = torch.tensor([[100.0, -50.0, 5.0]])
    horizon = 32
    pos, vel = gen(origin, horizon)

    assert pos.shape == (1, horizon, 3)
    assert vel.shape == (1, horizon, 3)
    # The spiral starts on the seed position, so step 0 is no reach,
    # just tracking.
    assert torch.allclose(pos[:, 0, :], origin, atol=1e-3)
    # z ramps linearly at z_rate on the third axis.
    expected_z = origin[0, 2] + 30.0 * torch.arange(
        horizon, dtype=torch.float32
    )
    assert torch.allclose(pos[0, :, 2], expected_z, atol=1e-3)
    # xy stays on a circle of the requested radius about its centre.
    xr = pos[0, :, 0] - (origin[0, 0] - 1000.0)
    yr = pos[0, :, 1] - origin[0, 1]
    r2 = torch.full((horizon,), 1000.0**2)
    assert torch.allclose(xr * xr + yr * yr, r2, atol=1.0)


def test_unknown_reference_kind_raises():
    with pytest.raises(ValueError, match="Unknown reference kind"):
        make_reference_gen({"kind": "helix"}, "cpu")
