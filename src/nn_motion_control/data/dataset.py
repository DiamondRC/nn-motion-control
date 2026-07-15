"""
Windowed HDF5 time-series dataset with train-only z-score normalisation.
"""

from __future__ import annotations

import gc
import logging
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from nn_motion_control.data._hdf5 import as_dataset
from nn_motion_control.data.normalize import NormStats, fit_stats
from nn_motion_control.data.splits import validate_labels

logger = logging.getLogger(os.path.basename(__file__))


@dataclass(frozen=True)
class DatasetMetadata:
    """
    Everything inference needs to interpret the model's normalised I/O.
    """

    input_labels: np.ndarray
    target_labels: np.ndarray
    input_denorm_params: dict[str, dict[str, float]]
    target_denorm_params: dict[str, dict[str, float]]
    loss_weights: torch.Tensor
    input_stats: NormStats
    target_stats: NormStats


class H5TimeSeriesDataset(Dataset):
    """
    Windowed view over an HDF5 dataset, indexed by window-start row.
    """

    DATA_KEY = "inputs"
    TARGET_KEY = "targets"

    def __init__(
        self,
        h5_path: str,
        allowed_inputs: Sequence[str],
        allowed_targets: Sequence[Mapping[str, float]],
        window_size: int = 1,
        load_into_ram: bool = False,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()

        self._file: h5py.File | None = None
        self._inputs: torch.Tensor | None = None
        self._targets: torch.Tensor | None = None

        self.window_size = window_size
        self.h5_path = h5_path
        self.dtype = dtype

        self.allowed_inputs = list(allowed_inputs)
        self.allowed_targets = [next(iter(d)) for d in allowed_targets]
        self.loss_weighting = [next(iter(d.values())) for d in allowed_targets]

        self._load_metadata()

        if self.window_size > self.num_samples:
            raise ValueError(f"{self.window_size=} cannot exceed {self.num_samples=}")

        if load_into_ram:
            self._load_all_to_ram()

        # Identity normalisation until fit_normalization() is called.
        self._in_mean = torch.zeros(len(self.allowed_inputs), dtype=dtype)
        self._in_std = torch.ones(len(self.allowed_inputs), dtype=dtype)
        self._tgt_mean = torch.zeros(len(self.allowed_targets), dtype=dtype)
        self._tgt_std = torch.ones(len(self.allowed_targets), dtype=dtype)
        self.meta: DatasetMetadata | None = None

    def _load_metadata(self) -> None:
        logger.debug("Gathering dataset metadata...")

        with h5py.File(self.h5_path, "r") as f:
            if f.attrs.get("schema_version", 1) < 2 or "segment_offsets" not in f:
                raise ValueError(
                    f"{self.h5_path} predates schema v2 (no segment_offsets). "
                    "Rebuild it with nn_motion_control.data.ingest."
                )
            self.input_labels = np.array(
                [x.decode("utf-8") for x in as_dataset(f, "input_labels")]
            )
            self.target_labels = np.array(
                [x.decode("utf-8") for x in as_dataset(f, "target_labels")]
            )
            self.segment_offsets = np.asarray(
                as_dataset(f, "segment_offsets"), dtype=np.int64
            )
            self.num_samples = int(as_dataset(f, self.DATA_KEY).shape[0])

        self._input_idx = validate_labels(
            self.allowed_inputs, list(self.input_labels), "input"
        )
        self._target_idx = validate_labels(
            self.allowed_targets, list(self.target_labels), "target"
        )

        # A column is normalisable unless it is a timestep index/counter.
        self._input_norm_mask = torch.tensor(
            [not lbl.startswith("timestep") for lbl in self.allowed_inputs]
        )
        self._target_norm_mask = torch.tensor(
            [not lbl.startswith("timestep") for lbl in self.allowed_targets]
        )

    def _load_all_to_ram(self) -> None:
        # Read full columns into numpy first, then select: the requested column order
        # (channel-name expansion) need not be increasing, but h5py's multi-axis fancy
        # indexing requires increasing indices — numpy indexing does not.
        with h5py.File(self.h5_path, "r") as f:
            x = np.asarray(as_dataset(f, self.DATA_KEY))[:, self._input_idx]
            y = np.asarray(as_dataset(f, self.TARGET_KEY))[:, self._target_idx]

        self._inputs = torch.from_numpy(x).to(self.dtype)
        self._targets = torch.from_numpy(y).to(self.dtype)

    def _read_input_rows(self, row_sel) -> torch.Tensor:
        if self._inputs is not None:
            if isinstance(row_sel, np.ndarray):
                return self._inputs[torch.from_numpy(row_sel)]
            return self._inputs[row_sel]
        f = self._open_file()
        # h5py fancy indexing requires increasing indices; order is irrelevant to stats.
        sel = np.sort(row_sel) if isinstance(row_sel, np.ndarray) else row_sel
        return torch.as_tensor(
            as_dataset(f, self.DATA_KEY)[sel][:, self._input_idx], dtype=self.dtype
        )

    def _read_target_rows(self, row_sel) -> torch.Tensor:
        if self._targets is not None:
            if isinstance(row_sel, np.ndarray):
                return self._targets[torch.from_numpy(row_sel)]
            return self._targets[row_sel]
        f = self._open_file()
        sel = np.sort(row_sel) if isinstance(row_sel, np.ndarray) else row_sel
        return torch.as_tensor(
            as_dataset(f, self.TARGET_KEY)[sel][:, self._target_idx], dtype=self.dtype
        )

    def fit_normalization(self, train_starts: np.ndarray, split_mode: str) -> None:
        """
        Fit z-score params from the TRAIN rows only and build final metadata.
        """

        w = self.window_size
        if split_mode == "contiguous":
            lo, hi = int(train_starts.min()), int(train_starts.max())
            in_sel: object = slice(lo, hi + w)  # input rows across all train windows
            tgt_sel: object = slice(lo + w - 1, hi + w)  # their target rows
        else:  # random, window_size == 1 -> target row == start row
            sel = np.asarray(train_starts, dtype=np.int64)
            in_sel = sel
            tgt_sel = sel

        in_data = self._read_input_rows(in_sel)
        tgt_data = self._read_target_rows(tgt_sel)

        input_stats = fit_stats(in_data, self._input_norm_mask, self.dtype)
        target_stats = fit_stats(tgt_data, self._target_norm_mask, self.dtype)

        self._in_mean, self._in_std = input_stats.mean, input_stats.std
        self._tgt_mean, self._tgt_std = target_stats.mean, target_stats.std

        self.meta = self._build_meta(input_stats, target_stats)

    def _build_meta(
        self, input_stats: NormStats, target_stats: NormStats
    ) -> DatasetMetadata:
        def denorm(labels, stats):
            return {
                "mean": {lbl: float(stats.mean[i]) for i, lbl in enumerate(labels)},
                "std": {lbl: float(stats.std[i]) for i, lbl in enumerate(labels)},
            }

        return DatasetMetadata(
            input_labels=self.input_labels,
            target_labels=self.target_labels,
            input_denorm_params=denorm(self.allowed_inputs, input_stats),
            target_denorm_params=denorm(self.allowed_targets, target_stats),
            # Plain per-target weight vector (targets are already unit-variance).
            loss_weights=torch.tensor(self.loss_weighting, dtype=self.dtype),
            input_stats=input_stats,
            target_stats=target_stats,
        )

    def _open_file(self) -> h5py.File:
        if self._file is None:
            self._file = h5py.File(self.h5_path, "r")
        return self._file

    def __len__(self) -> int:
        return self.num_samples - self.window_size + 1

    def __getitem__(self, idx: int):
        """
        ``idx`` is a window-START row; returns the normalised (window, target).
        """

        w = self.window_size
        if idx < 0 or idx + w > self.num_samples:
            raise IndexError(idx)

        if self._inputs is not None and self._targets is not None:
            if w == 1:
                x = self._inputs[idx]
                y = self._targets[idx]
            else:
                x = self._inputs[idx : idx + w].T  # [features, window]
                y = self._targets[idx + w - 1]
        else:
            f = self._open_file()
            inputs = as_dataset(f, self.DATA_KEY)
            targets = as_dataset(f, self.TARGET_KEY)
            if w == 1:
                # Index the row first (-> numpy), then columns, so a non-increasing
                # column selection is allowed (h5py multi-axis fancy indexing is not).
                x = torch.as_tensor(inputs[idx][self._input_idx], dtype=self.dtype)
                y = torch.as_tensor(targets[idx][self._target_idx], dtype=self.dtype)
            else:
                x = torch.as_tensor(
                    inputs[idx : idx + w][:, self._input_idx].T, dtype=self.dtype
                )
                y = torch.as_tensor(
                    targets[idx + w - 1, self._target_idx], dtype=self.dtype
                )

        if w == 1:
            x = (x - self._in_mean) / self._in_std
        else:
            x = (x - self._in_mean.unsqueeze(1)) / self._in_std.unsqueeze(1)
        y = (y - self._tgt_mean) / self._tgt_std
        return x, y

    def close_file(self) -> None:
        """
        Close the opened datafile.
        """

        if self._file is not None:
            try:
                self._file.close()
            finally:
                self._file = None

    def cleanup(self) -> None:
        """
        Clear file handles and caches to avoid I/O and memory issues.
        """

        self.close_file()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class SubsetDataset(Dataset):
    """
    A view of ``dataset`` restricted to ``indices`` (which window starts to serve).
    """

    def __init__(self, dataset: Dataset, indices: Sequence[int] | np.ndarray):
        self.dataset = dataset
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        return self.dataset[int(self.indices[idx])]
