"""
Z-score normalisation primitives, fitted on training rows only.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

_STD_EPS = 1e-8


@dataclass(frozen=True)
class NormStats:
    """
    Fitted z-score parameters for a column set.

    ``mean``/``std`` are length ``[n_cols]`` and applied as ``(x - mean) / std``.
    Non-normalisable columns (e.g. ``timestep``) carry identity params (0 / 1) so the
    same transform applies uniformly. ``std`` already has an epsilon added.
    """

    mean: torch.Tensor
    std: torch.Tensor
    normalizable: torch.Tensor  # bool mask [n_cols]


def fit_stats(
    data: torch.Tensor, normalizable: torch.Tensor, dtype: torch.dtype
) -> NormStats:
    """
    Fit z-score params in float64; identity (0 / 1) on non-normalisable columns.
    """

    data64 = data.to(torch.float64)
    mean = data64.mean(dim=0)
    std = data64.std(dim=0, unbiased=False) + _STD_EPS

    mean = torch.where(normalizable, mean, torch.zeros_like(mean))
    std = torch.where(normalizable, std, torch.ones_like(std))

    return NormStats(mean=mean.to(dtype), std=std.to(dtype), normalizable=normalizable)
