"""
Regression metrics for plant evaluation.

Errors are reported both in physical units (the channel's own scale) and normalised by
the target's standard deviation, so a scale-free correctness gate can be applied. The
acceptance gate is ``P95(|error|) / std(target) <= 5%`` — robust to the near-zero delta
targets that make per-sample relative error meaningless. ``P99`` is retained alongside
as a tail / discontinuity indicator (a large P99-vs-P95 gap flags rare large errors).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

GATE_PCTILE = 95  # the acceptance gate percentile (typical worst case)
TAIL_PCTILE = 99  # retained as a tail / discontinuity indicator
DEFAULT_GATE = 0.05  # 5% gate heuristic (fraction of target std)


@dataclass(frozen=True)
class ChannelMetrics:
    """
    Per-channel error metrics, absolute (physical units) and std-normalised.
    """

    name: str
    mae: float
    rmse: float
    p95_abs: float  # 95th percentile |error|, physical units
    p99_abs: float  # 99th percentile |error|, physical units
    std: float  # std of the target values (the normalising scale)
    p95_frac: float  # p95_abs / std  -- the acceptance gate quantity
    p99_frac: float  # p99_abs / std  -- tail / discontinuity indicator
    fit: float  # 100 * (1 - rmse/std): system-ID goodness-of-fit (%)

    @property
    def passes(self) -> bool:
        """
        Whether the typical worst-case (P95) error is within the 5% gate.
        """

        return np.isfinite(self.p95_frac) and self.p95_frac <= DEFAULT_GATE


def channel_metrics(name: str, pred: np.ndarray, target: np.ndarray) -> ChannelMetrics:
    """
    Compute absolute and std-normalised error metrics for one channel.
    """

    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if pred.shape != target.shape:
        raise ValueError(f"pred/target shape mismatch: {pred.shape} vs {target.shape}")

    err = np.abs(pred - target)
    mae = float(err.mean())
    rmse = float(np.sqrt(np.mean((pred - target) ** 2)))
    p95 = float(np.percentile(err, GATE_PCTILE))
    p99 = float(np.percentile(err, TAIL_PCTILE))
    std = float(target.std())

    # A constant target has no scale to normalise against; leave the ratios undefined
    # (NaN) rather than dividing by zero. `passes` then reads False.
    scale = std if std > 0 else np.nan
    return ChannelMetrics(
        name=name,
        mae=mae,
        rmse=rmse,
        p95_abs=p95,
        p99_abs=p99,
        std=std,
        p95_frac=p95 / scale,
        p99_frac=p99 / scale,
        fit=100.0 * (1.0 - rmse / scale),
    )
