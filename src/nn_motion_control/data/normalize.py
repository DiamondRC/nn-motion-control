"""
Z-score normalisation primitives, fitted on training rows only.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

_STD_EPS = 1e-8

# Rows reduced per float64 chunk, bounding the upcast copy.
_CHUNK_ROWS = 1 << 20


@dataclass(frozen=True)
class NormStats:
    """
    Fitted z-score parameters for a column set.

    'mean'/'std' are length [n_cols] and applied as (x - mean) / std.
    Non-normalisable columns (e.g. 'timestep') carry identity params (0 / 1)
    so the same transform applies uniformly. 'std' already has an epsilon
    added.
    """

    mean: torch.Tensor
    std: torch.Tensor
    normalizable: torch.Tensor  # bool mask [n_cols]


def fit_stats(
    data: torch.Tensor, normalizable: torch.Tensor, dtype: torch.dtype
) -> NormStats:
    """
    Fit z-score params in float64; identity (0 / 1) on non-normalisable columns.

    Two chunked passes (mean, then squared deviations) keep the float64
    copy to one chunk however many rows there are.
    """

    n = data.shape[0]
    chunks = range(0, n, _CHUNK_ROWS)
    total = sum(
        data[i : i + _CHUNK_ROWS].to(torch.float64).sum(dim=0) for i in chunks
    )
    mean = torch.as_tensor(total) / n
    sq = sum(
        ((data[i : i + _CHUNK_ROWS].to(torch.float64) - mean) ** 2).sum(dim=0)
        for i in chunks
    )
    std = (torch.as_tensor(sq) / n).sqrt() + _STD_EPS

    mean = torch.where(normalizable, mean, torch.zeros_like(mean))
    std = torch.where(normalizable, std, torch.ones_like(std))

    return NormStats(
        mean=mean.to(dtype), std=std.to(dtype), normalizable=normalizable
    )
