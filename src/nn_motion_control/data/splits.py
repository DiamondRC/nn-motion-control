"""
Label validation and leakage-aware train/val/test splitting of window
starts.

A window start is a row that can begin a full window inside a single
recording. Splitting operates on these start rows, inserting a
'window_size - 1' gap at each seam so a training window's target row cannot
reach into the validation range.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence

import numpy as np

logger = logging.getLogger(os.path.basename(__file__))


def validate_labels(
    requested: Sequence[str], available: Sequence[str], kind: str
) -> list[int]:
    """
    Map requested labels to their column indices; raise if any are missing.
    """

    mapping = {label: idx for idx, label in enumerate(available)}
    missing = [x for x in requested if x not in mapping]
    if missing:
        raise ValueError(
            f"Invalid {kind} labels: {missing}. "
            f"Available {kind} labels: {list(available)}"
        )

    return [mapping[x] for x in requested]


def build_valid_window_starts(
    segment_offsets: np.ndarray, window_size: int, horizon: int = 0
) -> np.ndarray:
    """
    Return every row that can start a full window within a single recording.

    Segment 'k' spans rows [segment_offsets[k], segment_offsets[k+1]); a
    window starting at 's' occupies [s, s + window_size) and must stay
    inside that span, so no window ever straddles two recordings. For a
    rollout of 'horizon' future steps the sample also reads rows
    [s + window_size, s + window_size + horizon), so 'horizon' extra rows
    are reserved at the tail of each recording.
    """

    reserve = window_size + horizon
    starts: list[np.ndarray] = []

    for k in range(len(segment_offsets) - 1):
        s = int(segment_offsets[k])
        e = int(segment_offsets[k + 1])
        last_start = e - reserve  # inclusive
        if last_start >= s:
            starts.append(np.arange(s, last_start + 1, dtype=np.int64))
        else:
            logger.warning(
                "Segment %d (length %d) is shorter than "
                "window+horizon=%d; skipped",
                k,
                e - s,
                reserve,
            )

    if not starts:
        raise ValueError(
            f"No recording is long enough for window+horizon={reserve}"
        )

    return np.concatenate(starts)


def _validate_ratios(train_ratio: float, val_ratio: float) -> None:
    if not 0 < train_ratio < 1:
        raise ValueError(f"{train_ratio=} must be in (0, 1)")
    if not 0 <= val_ratio < 1:
        raise ValueError(f"{val_ratio=} must be in [0, 1)")
    if train_ratio + val_ratio >= 1:
        raise ValueError(f"{train_ratio=} + {val_ratio=} must be < 1")


def split_window_starts_contiguous(
    valid_starts: np.ndarray,
    train_ratio: float,
    val_ratio: float,
    window_size: int,
    horizon: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Contiguous split of ordered window starts with a leakage gap at each
    seam.

    After the ratio split, the last 'window_size + horizon - 1' starts of
    train and of val are dropped so a training sample's furthest-reached
    row (target, or the last rollout step) cannot fall into the validation
    range (and val cannot reach test).
    """

    _validate_ratios(train_ratio, val_ratio)

    n = len(valid_starts)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    if train_end <= 0 or val_end <= train_end or val_end >= n:
        raise ValueError(f"Invalid split sizes for {n=}")

    train_idx = valid_starts[:train_end]
    val_idx = valid_starts[train_end:val_end]
    test_idx = valid_starts[val_end:]

    gap = window_size + horizon - 1
    if gap > 0:
        train_idx = train_idx[:-gap] if len(train_idx) > gap else train_idx[:0]
        val_idx = val_idx[:-gap] if len(val_idx) > gap else val_idx[:0]

    if len(train_idx) == 0 or len(val_idx) == 0 or len(test_idx) == 0:
        raise ValueError(
            "A split is empty after applying the window gap; "
            "reduce window_size or adjust the split ratios"
        )

    return train_idx, val_idx, test_idx


def split_window_starts_random(
    valid_starts: np.ndarray,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Randomised split for non-windowed data ('window_size == 1').

    Each sample is a single independent row, so a random split introduces
    no window overlap and needs no gap.
    """

    _validate_ratios(train_ratio, val_ratio)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(valid_starts)
    n = len(perm)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    if train_end <= 0 or val_end <= train_end or val_end >= n:
        raise ValueError(f"Invalid split sizes for {n=}")

    return perm[:train_end], perm[train_end:val_end], perm[val_end:]
