"""Randomised trajectory families: anchoring, shapes, PVT, reproducibility."""

import pytest
import torch

from nn_motion_control.control.trajectories import (
    build_family,
    line_family,
    morph_family,
    sample_mixed_reference,
    sequence_reference,
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


def test_build_family_anchors_and_is_reproducible():
    origin = _origin()
    k = torch.arange(H, dtype=torch.float32)

    for name in ("spiral", "helix", "line", "step", "smooth"):
        a, _ = build_family(
            name, origin, k, {}, torch.Generator().manual_seed(3)
        )
        b, _ = build_family(
            name, origin, k, {}, torch.Generator().manual_seed(3)
        )
        assert a.shape == (B, H, A)
        assert torch.allclose(a[:, 0, :], origin, atol=1e-3)  # anchored
        assert torch.equal(a, b)  # same seed -> identical draw


def test_build_family_unknown_name_raises():
    origin = torch.zeros(B, A)
    k = torch.arange(H, dtype=torch.float32)
    with pytest.raises(ValueError, match="Unknown trajectory family"):
        build_family("bogus", origin, k, {}, None)


def test_helix_ramps_z_where_spiral_default_may_not():
    origin = torch.zeros(B, A)
    k = torch.arange(H, dtype=torch.float32)
    pos, _ = build_family(
        "helix", origin, k, {}, torch.Generator().manual_seed(2)
    )
    # A helix always climbs (forced non-zero z ramp on the third axis).
    assert (pos[:, -1, 2] - origin[:, 2]).abs().min() > 1.0


def test_morph_family_anchored_and_velocity_matches_fd():
    origin = _origin()
    g = torch.Generator().manual_seed(8)
    # Pin params low so the analytic velocity (with the blend cross term)
    # matches a finite difference of the blended position.
    spec = {
        "radius": [300.0, 300.0],
        "angular": [0.003, 0.003],
        "z_rate": [0.0, 0.0],
        "smooth_amp": [50.0, 50.0],
    }
    pos, vel = morph_family(origin, H, "spiral", "smooth", spec, g)
    assert pos.shape == (B, H, A)
    assert torch.allclose(pos[:, 0, :], origin, atol=1e-3)  # both anchored
    fd = pos[:, 1:, :] - pos[:, :-1, :]
    assert torch.allclose(vel[:, :-1, :], fd, atol=1.5)


def test_sequence_reference_length_and_seam_continuity():
    origin = _origin()
    g = torch.Generator().manual_seed(9)
    pos, _ = sequence_reference(origin, 30, ["spiral", "line", "step"], {}, g)
    assert pos.shape == (B, 30, A)
    assert torch.allclose(pos[:, 0, :], origin, atol=1e-3)
    # An even split gives seams at 10 and 20; position is continuous there.
    for seam in (10, 20):
        assert torch.allclose(pos[:, seam, :], pos[:, seam - 1, :], atol=1e-3)


def test_sequence_reference_rejects_mismatched_durations():
    origin = torch.zeros(B, A)
    g = torch.Generator().manual_seed(1)
    with pytest.raises(ValueError, match="must sum to the horizon"):
        sequence_reference(
            origin, 30, ["spiral", "line"], {}, g, durations=[10, 5]
        )


def test_sequence_reference_empty_segments_raises():
    origin = torch.zeros(B, A)
    with pytest.raises(ValueError, match="at least one segment"):
        sequence_reference(origin, 30, [], {}, None)
