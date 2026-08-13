"""Chunked dense forward.

Shape, chunk-start equivalence, and intended divergence.
"""

import torch

from nn_motion_control.core.config import RunConfiguration
from nn_motion_control.models.builder import JsonModel
from nn_motion_control.models.chunked import chunked_dense_forward

TCN_CFG = "examples/deltabot/configs/plant_tcn_rollout.json"


def _tcn():
    return JsonModel(config=RunConfiguration(TCN_CFG))


def test_chunked_dense_shape():
    m = _tcn().eval()
    chunk = torch.randn(4, 9, 100)
    with torch.no_grad():
        out = chunked_dense_forward(m.network, chunk, window=64)
    assert out.shape == (4, 100 - 64 + 1, 3)


def test_first_position_matches_windowed_forward():
    # Position 0's pooling window starts at the chunk's absolute start,
    # where the chunked conv zero-pads identically to a standalone
    # window, exact match.
    m = _tcn().eval()
    torch.manual_seed(0)
    chunk = torch.randn(4, 9, 100)
    with torch.no_grad():
        dense = chunked_dense_forward(m.network, chunk, window=64)
        windowed = m.network(chunk[:, :, :64])
    assert torch.allclose(dense[:, 0], windowed, atol=1e-4, rtol=1e-3)


def test_deeper_positions_diverge_from_windowed():
    # A position deep in the chunk pools real-history conv features,
    # unlike the windowed model's zero-padded early positions, genuine
    # divergence, not a bug. This is the modelling change the chunked
    # path introduces (streaming history).
    m = _tcn().eval()
    torch.manual_seed(0)
    chunk = torch.randn(4, 9, 200)
    p = 150  # end-position deep in the chunk; dense index is p - (window - 1)
    with torch.no_grad():
        dense = chunked_dense_forward(m.network, chunk, window=64)
        windowed_p = m.network(chunk[:, :, p - 64 + 1 : p + 1])
    assert (dense[:, p - 63] - windowed_p).abs().max() > 1e-4
