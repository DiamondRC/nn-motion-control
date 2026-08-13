"""
Representative sampling from timeline-ordered rollout loaders.

The rollout loaders iterate window starts in timeline order, so the first
N batches are one contiguous segment that can be entirely quiet (motion
at the noise floor). Striding across the whole loader keeps any sample
representative of the full motion range.
"""

from __future__ import annotations

import torch


def sampled_batches(loader, max_batches: int | None):
    """
    Yield up to 'max_batches' batches spread evenly across the loader.

    max_batches is None yields every batch.
    """

    if max_batches is None:
        yield from loader
        return
    stride = max(1, len(loader) // max_batches)
    kept = 0

    for i, batch in enumerate(loader):
        if i % stride != 0:
            continue
        if kept >= max_batches:
            break
        yield batch
        kept += 1


def representative_windows(loader, max_batches: int = 8) -> torch.Tensor:
    """
    Concatenate the warmup windows from batches strided across the loader.

    Returns the input-window tensor [N, F, W] (batch element 0) gathered
    from up to max_batches evenly-spread batches, so a single-batch probe
    sees the whole motion range rather than the quiet leading slice
    next(iter(loader)) would give.
    """

    windows = [batch[0] for batch in sampled_batches(loader, max_batches)]
    if not windows:
        raise ValueError("Loader yielded no batches to sample")

    return torch.cat(windows, dim=0)
