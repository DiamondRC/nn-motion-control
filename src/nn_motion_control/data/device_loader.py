"""
Device-resident, vectorised window loaders.

A fast alternative to a per-item DataLoader when the selected columns fit in
memory: the normalised inputs and targets live on the training device, and each
batch's windows are gathered in one indexing op, with no per-batch host copy.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import torch


class _WindowLoaderBase:
    """
    Shared batching for the window loaders: per-epoch shuffle and batching.

    Subclasses gather each batch of start rows into their tensors in __iter__.
    """

    def __init__(
        self,
        starts: torch.Tensor,
        batch_size: int,
        *,
        shuffle: bool,
        seed: int,
        device: str,
        drop_last: bool,
    ):
        if batch_size < 1:
            raise ValueError(f"{batch_size=} must be >= 1")

        self._starts = starts
        self.batch_size = int(batch_size)
        self.shuffle = shuffle
        self.seed = seed
        self.device = device
        self.drop_last = drop_last
        self._epoch = 0
        # Mirror DataLoader.dataset so callers can read a split's window count.
        self.dataset = starts

    def __len__(self) -> int:
        n = len(self._starts)
        if self.drop_last:
            return n // self.batch_size

        return (n + self.batch_size - 1) // self.batch_size

    def _start_batches(self) -> Iterator[torch.Tensor]:
        order = self._starts
        if self.shuffle:
            # Reproducible, distinct shuffles per epoch.
            gen = torch.Generator(device=self.device).manual_seed(
                self.seed + self._epoch
            )
            perm = torch.randperm(len(order), generator=gen, device=self.device)
            order = order[perm]
            self._epoch += 1

        b, n = self.batch_size, len(order)
        last = (n // b) * b if self.drop_last else n

        for i in range(0, last, b):
            yield order[i : i + b]


class DeviceWindowLoader(_WindowLoaderBase):
    """
    Yield windowed (x, y) batches from device-resident normalised tensors.
    """

    def __init__(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        starts: torch.Tensor,
        window_size: int,
        batch_size: int,
        *,
        shuffle: bool = False,
        seed: int = 42,
        device: str = "cpu",
        drop_last: bool = False,
    ):
        super().__init__(
            starts,
            batch_size,
            shuffle=shuffle,
            seed=seed,
            device=device,
            drop_last=drop_last,
        )
        self._inputs = inputs
        self._targets = targets
        self.window_size = int(window_size)
        self._offsets = torch.arange(self.window_size, device=device)

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        w = self.window_size

        for s in self._start_batches():
            if w == 1:
                x = self._inputs[s]
            else:
                rows = s[:, None] + self._offsets[None, :]
                x = self._inputs[rows].permute(0, 2, 1)
            y = self._targets[s + (w - 1)]

            yield x, y

    @staticmethod
    def starts_to_device(
        starts: np.ndarray | torch.Tensor, device: str
    ) -> torch.Tensor:
        """
        Move a 1-D array of window-start rows onto device as int64.
        """

        return torch.as_tensor(
            np.asarray(starts, dtype=np.int64), device=device
        )


class RolloutWindowLoader(_WindowLoaderBase):
    """
    Yield rollout batches (warmup, dac_future, gt_pos) from device tensors.

    For each start row 's' with window 'W' and horizon 'H': warmup is the seed
    window over rows [s, s+W), shaped [B, F_in, W], dac_future and gt_pos are
    command and position columns over rows [s+W, s+W+H), each [B, H, A].
    pos_cols and dac_cols index those channels in the axis-major F_in layout.
    """

    def __init__(
        self,
        inputs: torch.Tensor,
        starts: torch.Tensor,
        window_size: int,
        horizon: int,
        batch_size: int,
        pos_cols: list[int],
        dac_cols: list[int],
        *,
        shuffle: bool = False,
        seed: int = 42,
        device: str = "cpu",
        drop_last: bool = False,
    ):
        if horizon < 1:
            raise ValueError(f"{horizon=} must be >= 1")

        super().__init__(
            starts,
            batch_size,
            shuffle=shuffle,
            seed=seed,
            device=device,
            drop_last=drop_last,
        )
        self._inputs = inputs
        self.window_size = int(window_size)
        self.horizon = int(horizon)
        self.pos_cols = torch.as_tensor(pos_cols, device=device)
        self.dac_cols = torch.as_tensor(dac_cols, device=device)
        self._win_off = torch.arange(self.window_size, device=device)
        self._hor_off = torch.arange(self.horizon, device=device)
        # Drop the ragged final batch so batches share a shape and a compiled
        # rollout step never recompiles on a short last batch.
        self.drop_last = drop_last

    def __iter__(
        self,
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        w = self.window_size

        for s in self._start_batches():
            warmup = self._inputs[s[:, None] + self._win_off[None, :]]
            warmup = warmup.permute(0, 2, 1)
            future = self._inputs[(s + w)[:, None] + self._hor_off[None, :]]

            yield (
                warmup,
                future[:, :, self.dac_cols],
                future[:, :, self.pos_cols],
            )
