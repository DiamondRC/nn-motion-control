"""
Normalisation primitives: identity on masked columns + normalise/denormalise round-trip.
"""

import numpy as np
import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import array_shapes, arrays

from nn_motion_control.data.normalize import fit_stats


def test_non_normalizable_columns_get_identity():
    data = torch.randn(20, 3)
    mask = torch.tensor([True, False, True])
    stats = fit_stats(data, mask, torch.float32)
    # Masked column carries identity params so the transform is a no-op there.
    assert stats.mean[1] == 0.0
    assert stats.std[1] == 1.0


@settings(max_examples=50, deadline=None)
@given(
    arrays(
        np.float64,
        array_shapes(min_dims=2, max_dims=2, min_side=2, max_side=32),
        elements=st.floats(-1e3, 1e3, allow_nan=False, allow_infinity=False),
    )
)
def test_normalize_denormalize_roundtrip(arr):
    data = torch.from_numpy(arr)
    mask = torch.ones(data.shape[1], dtype=torch.bool)
    stats = fit_stats(data, mask, torch.float64)

    normed = (data - stats.mean) / stats.std
    restored = normed * stats.std + stats.mean
    assert torch.allclose(restored, data, atol=1e-4)
