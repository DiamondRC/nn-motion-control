"""
control reference generators: step trajectory and the spiral kind.
"""

import pytest
import torch

from nn_motion_control.control.closed_loop import step_reference
from nn_motion_control.training.control_run import (
    TRACK_SHAPES,
    make_reference_gen,
    shape_spec,
)


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
        make_reference_gen({"kind": "bogus"}, "cpu")


def test_helix_kind_anchors_and_climbs():
    gen = make_reference_gen({"kind": "helix"}, "cpu")
    origin = torch.tensor([[100.0, -50.0, 5.0]])
    pos, vel = gen(origin, 32, torch.Generator().manual_seed(3))
    assert pos.shape == (1, 32, 3)
    assert vel.shape == (1, 32, 3)
    assert torch.allclose(pos[:, 0, :], origin, atol=1e-3)  # anchored
    assert (pos[0, -1, 2] - origin[0, 2]).abs() > 1.0  # z climbs


@pytest.mark.parametrize("kind", ["line", "smooth", "helix", "mixed"])
def test_randomised_kinds_are_seed_reproducible(kind):
    gen = make_reference_gen({"kind": kind}, "cpu")
    origin = torch.randn(4, 3) * 100.0
    a = gen(origin, 24, torch.Generator().manual_seed(5))[0]
    b = gen(origin, 24, torch.Generator().manual_seed(5))[0]
    c = gen(origin, 24, torch.Generator().manual_seed(6))[0]
    assert torch.equal(a, b)  # same seed -> identical
    assert not torch.equal(a, c)  # different seed -> different
    assert torch.allclose(a[:, 0, :], origin, atol=1e-3)  # anchored


def test_spiral_shape_is_tilted_and_seed_varies():
    gen = make_reference_gen(shape_spec("spiral"), "cpu")
    origin = torch.zeros(3, 3)
    a, _ = gen(origin, 96, torch.Generator().manual_seed(1))
    b, _ = gen(origin, 96, torch.Generator().manual_seed(1))
    c, _ = gen(origin, 96, torch.Generator().manual_seed(2))
    assert torch.allclose(a[:, 0, :], origin, atol=1e-3)  # anchored
    assert a[:, :, 2].abs().max() > 10.0  # sweeps z, not a flat x-y circle
    assert torch.equal(a, b)  # reproducible per seed
    assert not torch.equal(a, c)  # orientation varies with the seed


def test_morph_kind_anchored_and_reproducible():
    gen = make_reference_gen(
        {"kind": "morph", "from": "spiral", "to": "step"}, "cpu"
    )
    origin = torch.zeros(3, 3)
    a, _ = gen(origin, 48, torch.Generator().manual_seed(2))
    b, _ = gen(origin, 48, torch.Generator().manual_seed(2))
    assert a.shape == (3, 48, 3)
    assert torch.allclose(a[:, 0, :], origin, atol=1e-3)  # both anchored
    assert torch.equal(a, b)


def test_sequence_kind_length_and_seam_continuity():
    gen = make_reference_gen(
        {"kind": "sequence", "segments": ["spiral", "line", "step"]}, "cpu"
    )
    origin = torch.zeros(2, 3)
    pos, _ = gen(origin, 30, torch.Generator().manual_seed(4))
    assert pos.shape == (2, 30, 3)
    assert torch.allclose(pos[:, 0, :], origin, atol=1e-3)

    for seam in (10, 20):  # even split of 30 into three segments
        assert torch.allclose(pos[:, seam, :], pos[:, seam - 1, :], atol=1e-3)


def test_shape_spec_builds_every_track_shape():
    for shape in TRACK_SHAPES:
        if shape == "config":
            continue
        spec = shape_spec(shape)
        assert "kind" in spec
        make_reference_gen(spec, "cpu")  # every preset builds a generator


def test_shape_spec_unknown_raises():
    with pytest.raises(ValueError, match="Unknown track shape"):
        shape_spec("bogus")
