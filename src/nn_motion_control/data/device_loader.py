"""
Device-resident, vectorised window loader.

A fast alternative to a per-item DataLoader for the common case where the
selected columns fit in memory. The (already normalised) inputs and targets live
on the training device; each batch's windows are gathered in a single indexing op
rather than one __getitem__ call per sample so there is no per-batch host->device copy.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import torch


class DeviceWindowLoader:
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
        if batch_size < 1:
            raise ValueError(f"{batch_size=} must be >= 1")

        self._inputs = inputs
        self._targets = targets
        self._starts = starts
        self.window_size = int(window_size)
        self.batch_size = int(batch_size)
        self.shuffle = shuffle
        self.seed = seed
        self.device = device
        self.drop_last = drop_last
        self._offsets = torch.arange(self.window_size, device=device)
        self._epoch = 0
        # Mirror DataLoader.dataset so callers can read the window count per split.
        self.dataset = starts

    def __len__(self) -> int:
        n = len(self._starts)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        order = self._starts
        if self.shuffle:
            # Reproducible, distinct shuffles per epoch.
            gen = torch.Generator(device=self.device).manual_seed(
                self.seed + self._epoch
            )
            order = order[torch.randperm(len(order), generator=gen, device=self.device)]
            self._epoch += 1

        w, b = self.window_size, self.batch_size
        n = len(order)
        last = (n // b) * b if self.drop_last else n
        for i in range(0, last, b):
            s = order[i : i + b]
            if w == 1:
                x = self._inputs[s]  # [b, F_in]
            else:
                rows = s[:, None] + self._offsets[None, :]  # [b, W]
                x = self._inputs[rows].permute(0, 2, 1)  # [b, F_in, W]
            y = self._targets[s + (w - 1)]  # [b, F_tgt]
            yield x, y

    @staticmethod
    def starts_to_device(
        starts: np.ndarray | torch.Tensor, device: str
    ) -> torch.Tensor:
        """
        Move a 1-D array of window-start rows onto device as int64.
        """

        return torch.as_tensor(np.asarray(starts, dtype=np.int64), device=device)


class RolloutWindowLoader:
    """
    Yield rollout batches (warmup, dac_future, gt_pos) from device-resident tensors.

    For a start row s (window W, horizon H):
      * warmup     [B, F_in, W] — the seed window, rows [s, s+W).
      * dac_future [B, H, A]    — recorded command cols, rows [s+W, s+W+H).
      * gt_pos     [B, H, A]    — ground-truth position cols, same rows.

    All normalised and already on device. Commands are fed forward and positions are
    the rollout targets; pos_cols / dac_cols index those channels within the
    F_in input layout (axis-major).
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
        if batch_size < 1:
            raise ValueError(f"{batch_size=} must be >= 1")
        if horizon < 1:
            raise ValueError(f"{horizon=} must be >= 1")
        self._inputs = inputs
        self._starts = starts
        self.window_size = int(window_size)
        self.horizon = int(horizon)
        self.batch_size = int(batch_size)
        self.pos_cols = torch.as_tensor(pos_cols, device=device)
        self.dac_cols = torch.as_tensor(dac_cols, device=device)
        self.shuffle = shuffle
        self.seed = seed
        self.device = device
        # Drop the ragged final batch so every batch has the same shape; a
        # torch.compile'd rollout step (or CUDA graphs) then never recompiles on a
        # short last batch.
        self.drop_last = drop_last
        self._win_off = torch.arange(self.window_size, device=device)
        self._hor_off = torch.arange(self.horizon, device=device)
        self._epoch = 0
        self.dataset = starts

    def __len__(self) -> int:
        n = len(self._starts)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        order = self._starts
        if self.shuffle:
            gen = torch.Generator(device=self.device).manual_seed(
                self.seed + self._epoch
            )
            order = order[torch.randperm(len(order), generator=gen, device=self.device)]
            self._epoch += 1

        w, b = self.window_size, self.batch_size
        n = len(order)
        last = (n // b) * b if self.drop_last else n
        for i in range(0, last, b):
            s = order[i : i + b]
            warmup = self._inputs[s[:, None] + self._win_off[None, :]]  # [b, W, F_in]
            warmup = warmup.permute(0, 2, 1)  # [b, F_in, W]
            fut_rows = (s + w)[:, None] + self._hor_off[None, :]  # [b, H]
            future = self._inputs[fut_rows]  # [b, H, F_in]
            dac_future = future[:, :, self.dac_cols]  # [b, H, A]
            gt_pos = future[:, :, self.pos_cols]  # [b, H, A]
            yield warmup, dac_future, gt_pos
