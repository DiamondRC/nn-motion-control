"""Randomised trajectory families: anchoring, shapes, PVT, reproducibility."""

import torch

from nn_motion_control.control.trajectories import (
    line_family,
    sample_mixed_reference,
    spiral_family,
    step_family,
)

B, H, A = 6, 40, 3


def _origin():
    return torch.randn(B, A) * 100.0


def test_families_start_on_origin_and_have_pvt_shapes():
    origin = _origin()
    k = torch.arange(H, dtype=torch.float32)
    fams = [
        spiral_family(
            origin,
            k,
            torch.full((B,), 500.0),
            torch.full((B,), 0.01),
            torch.full((B,), 5.0),
            (0, 1),
        ),
        line_family(
            origin,
            k,
            torch.eye(A)[torch.zeros(B, dtype=torch.long)],
            torch.full((B,), 300.0),
            torch.full((B,), 0.01),
        ),
        step_family(
            origin,
            k,
            torch.randn(B, A) * 50,
            torch.full((B,), 4.0),
            torch.full((B,), 12.0),
        ),
    ]

    for pos, vel in fams:
        assert pos.shape == (B, H, A)
        assert vel.shape == (B, H, A)
        assert torch.allclose(pos[:, 0, :], origin, atol=1e-3)


def test_velocity_matches_finite_difference_at_low_rate():
    """Analytic PVT velocity approximates the per-step position change."""
    origin = _origin()
    k = torch.arange(H, dtype=torch.float32)
    pos, vel = spiral_family(
        origin,
        k,
        torch.full((B,), 400.0),
        torch.full((B,), 0.004),
        torch.full((B,), 2.0),
        (0, 1),
    )
    fd = pos[:, 1:, :] - pos[:, :-1, :]
    assert torch.allclose(vel[:, :-1, :], fd, atol=0.5)


def test_step_family_reaches_target_and_holds():
    origin = torch.zeros(B, A)
    k = torch.arange(H, dtype=torch.float32)
    delta = torch.full((B, A), 200.0)
    pos, vel = step_family(
        origin, k, delta, torch.full((B,), 5.0), torch.full((B,), 10.0)
    )
    assert torch.allclose(pos[:, -1, :], delta, atol=1e-3)  # settled at target
    assert torch.allclose(
        vel[:, -1, :], torch.zeros(B, A), atol=1e-4
    )  # and held


def test_mixed_reference_is_anchored_and_reproducible():
    origin = _origin()
    g1 = torch.Generator().manual_seed(7)
    g2 = torch.Generator().manual_seed(7)
    g3 = torch.Generator().manual_seed(99)
    pos1, _ = sample_mixed_reference(origin, H, {}, g1)
    pos2, _ = sample_mixed_reference(origin, H, {}, g2)
    pos3, _ = sample_mixed_reference(origin, H, {}, g3)
    assert pos1.shape == (B, H, A)
    assert torch.allclose(
        pos1[:, 0, :], origin, atol=1e-3
    )  # anchored per family
    assert torch.equal(
        pos1, pos2
    )  # same seed -> identical draw (val reproducibility)
    assert not torch.equal(
        pos1, pos3
    )  # different seed -> different trajectories


def test_mixed_weights_can_select_a_single_family():
    origin = torch.zeros(B, A)
    g = torch.Generator().manual_seed(1)
    # Weight only the step family: every sample settles and holds
    # (zero end velocity).
    _, vel = sample_mixed_reference(origin, H, {"weights": [0, 0, 1, 0]}, g)
    assert torch.allclose(vel[:, -1, :], torch.zeros(B, A), atol=1e-4)
