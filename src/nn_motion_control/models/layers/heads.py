"""
Temporal pooling heads for windowed sequence models.
"""

import torch
import torch.nn as nn


class LastFrame(nn.Module):
    """
    Select the last time step, keeping a singleton time dim:
    [B, C, T] to [B, C, 1].

    A streaming-friendly alternative to AdaptiveAvgPool1d(1): it reads
    only the last frame, so a model with receptive field <= window
    computes the same last-frame output whether run over the full
    window or advanced one frame at a time. The singleton time dim
    matches the pool's output shape, so a following Flatten head is
    unchanged.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, -1:]


class AvgPoolLastK(nn.Module):
    """
    Average the last k time steps, keeping a singleton dim:
    [B, C, T] to [B, C, 1].

    A streamable middle ground between LastFrame (k=1, maximally
    error-sensitive in a free-run) and full AdaptiveAvgPool1d (whole
    window, not streamable). Averaging the last k frames cuts the
    readout's sensitivity to any single corrupted frame ~1/k, which
    damps rollout drift, while a k-frame ring buffer keeps each stream
    step O(1). With receptive field <= window - k the pooled frames are
    all in-window, so the windowed forward and the streamed rollout
    still agree.
    """

    def __init__(self, k: int):
        super().__init__()
        if k < 1:
            raise ValueError(f"AvgPoolLastK needs k >= 1, got {k}")
        self.k = k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, -self.k :].mean(dim=2, keepdim=True)
