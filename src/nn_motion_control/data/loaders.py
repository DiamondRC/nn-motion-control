"""
DataLoader construction and the top-level split/normalise/load orchestrator.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import h5py
import numpy as np
import psutil
import torch
from torch.utils.data import DataLoader, Dataset

from nn_motion_control.data._hdf5 import as_dataset
from nn_motion_control.data.dataset import (
    DatasetMetadata,
    H5TimeSeriesDataset,
    SubsetDataset,
)
from nn_motion_control.data.device_loader import (
    DeviceWindowLoader,
    RolloutWindowLoader,
)
from nn_motion_control.data.splits import (
    build_valid_window_starts,
    split_window_starts_contiguous,
    split_window_starts_random,
)

logger = logging.getLogger(os.path.basename(__file__))

# Loader types that satisfy the trainer/evaluator contract (iterate batches + len +
# a dataset with the window count). RolloutWindowLoader yields 3-tuples consumed by
# RolloutTrainer; the others yield (x, y).
BatchLoader = DataLoader | DeviceWindowLoader | RolloutWindowLoader


@dataclass(frozen=True)
class AllDataInfo:
    """
    The three dataloaders plus the metadata needed for inference/denormalisation.
    """

    trn_loader: BatchLoader
    val_loader: BatchLoader
    tst_loader: BatchLoader
    node_info: DatasetMetadata


def build_time_series_splits(
    h5_path: str,
    allowed_inputs: Sequence[str],
    allowed_targets: Sequence[Mapping[str, float]],
    window_size: int,
    train_ratio: float,
    val_ratio: float,
    training_dtype: torch.dtype,
    seed: int = 42,
    load_into_ram: bool = False,
    batch_size: int = 32,
    num_workers: int = 0,
    cpu_core_util: int = 50,
    prefetch_factor: int | None = None,
    persistent_workers: bool = False,
    pin_memory: bool = True,
    auto_tune_workers: bool = True,
    enable_logging: bool = False,
    device: str = "cpu",
) -> AllDataInfo:
    """
    Create boundary-aware, leakage-free train/val/test splits.

    Order of operations:
      1. Load metadata + (optionally) selected columns into RAM.
      2. Build valid window-start rows per recording (no window crosses a boundary).
      3. Split the ordered starts (contiguous for windowed data, random for
         ``window_size == 1``), inserting a ``window_size - 1`` gap at split seams.
      4. Fit z-score normalisation on the TRAIN rows only, then expose it everywhere.
      5. Build the DataLoaders over the window-start rows.
    """

    split_mode = "contiguous" if window_size > 1 else "random"

    if load_into_ram:
        ensure_enough_ram(h5_path)

    dataset = H5TimeSeriesDataset(
        h5_path=h5_path,
        allowed_inputs=allowed_inputs,
        allowed_targets=allowed_targets,
        window_size=window_size,
        load_into_ram=load_into_ram,
        dtype=training_dtype,
    )

    if auto_tune_workers:
        num_workers, prefetch_factor, persistent_workers = setup_workers(
            num_workers=num_workers,
            cpu_core_util=cpu_core_util,
            load_into_ram=load_into_ram,
            enable_logging=enable_logging,
        )

    valid_starts = build_valid_window_starts(dataset.segment_offsets, window_size)

    if split_mode == "contiguous":
        train_idx, val_idx, test_idx = split_window_starts_contiguous(
            valid_starts, train_ratio, val_ratio, window_size
        )
    else:
        train_idx, val_idx, test_idx = split_window_starts_random(
            valid_starts, train_ratio, val_ratio, seed
        )

    # Fit normalisation on the train rows only, then apply to every split.
    dataset.fit_normalization(train_idx, split_mode)
    assert dataset.meta is not None  # set by fit_normalization

    if enable_logging:
        logger.debug(
            "Split (%s): train=%d val=%d test=%d window=%d",
            split_mode,
            len(train_idx),
            len(val_idx),
            len(test_idx),
            window_size,
        )

    if load_into_ram:
        # Fast path: hold the normalised tensors on ``device`` and gather each
        # batch's windows in one vectorised op (no per-item Python, no per-batch copy).
        trn_loader, val_loader, tst_loader = make_device_loaders(
            dataset=dataset,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            window_size=window_size,
            batch_size=batch_size,
            seed=seed,
            device=device,
        )
    else:
        # Fallback: stream windows from disk one item at a time.
        trn_loader, val_loader, tst_loader = make_dataloaders(
            dataset=dataset,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            batch_size=batch_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            pin_memory=pin_memory,
            shuffle_train=True,
            seed=seed,
            enable_logging=enable_logging,
        )
    return AllDataInfo(trn_loader, val_loader, tst_loader, node_info=dataset.meta)


def make_device_loaders(
    dataset: H5TimeSeriesDataset,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    window_size: int,
    batch_size: int,
    seed: int,
    device: str,
) -> tuple[DeviceWindowLoader, DeviceWindowLoader, DeviceWindowLoader]:
    """
    Build device-resident, vectorised window loaders sharing one normalised tensor.
    """

    inputs, targets = dataset.normalized_arrays(device)

    def _make(idx: np.ndarray, shuffle: bool) -> DeviceWindowLoader:
        return DeviceWindowLoader(
            inputs,
            targets,
            DeviceWindowLoader.starts_to_device(idx, device),
            window_size=window_size,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            device=device,
        )

    return _make(train_idx, True), _make(val_idx, False), _make(test_idx, False)


def build_rollout_splits(
    h5_path: str,
    allowed_inputs: Sequence[str],
    allowed_targets: Sequence[Mapping[str, float]],
    window_size: int,
    max_horizon: int,
    pos_cols: list[int],
    dac_cols: list[int],
    train_ratio: float,
    val_ratio: float,
    training_dtype: torch.dtype,
    batch_size: int,
    seed: int = 42,
    device: str = "cpu",
    train_start_stride: int = 1,
    val_start_stride: int = 1,
    drop_last: bool = False,
) -> AllDataInfo:
    """
    Build leakage-aware rollout splits as RolloutWindowLoaders on the device.

    All three loaders share one normalised input tensor; W + max_horizon rows are
    reserved at each recording tail and split seam so no rollout crosses a boundary or
    leaks across splits. Normalisation is fit on the train rows only.

    ``train_start_stride`` / ``val_start_stride`` subsample the window starts (adjacent
    rollout windows overlap in all but one warmup and one horizon row, so they are
    highly redundant); normalisation is still fit on the full train rows. ``drop_last``
    drops the ragged final batch so a compiled rollout step sees a constant batch shape.
    """

    ensure_enough_ram(h5_path)
    dataset = H5TimeSeriesDataset(
        h5_path=h5_path,
        allowed_inputs=allowed_inputs,
        allowed_targets=allowed_targets,
        window_size=window_size,
        load_into_ram=True,
        dtype=training_dtype,
    )
    valid = build_valid_window_starts(
        dataset.segment_offsets, window_size, horizon=max_horizon
    )
    train_idx, val_idx, test_idx = split_window_starts_contiguous(
        valid, train_ratio, val_ratio, window_size, horizon=max_horizon
    )
    # Fit normalisation on the full (unstrided) train rows for the best stats, then
    # subsample the starts each loader actually iterates.
    dataset.fit_normalization(train_idx, "contiguous")
    assert dataset.meta is not None
    inputs, _ = dataset.normalized_arrays(device)  # rollout reads state/dac from inputs

    def mk(idx: np.ndarray, shuffle: bool, stride: int = 1) -> RolloutWindowLoader:
        return RolloutWindowLoader(
            inputs,
            DeviceWindowLoader.starts_to_device(idx[::stride], device),
            window_size,
            max_horizon,
            batch_size,
            pos_cols,
            dac_cols,
            shuffle=shuffle,
            seed=seed,
            device=device,
            drop_last=drop_last,
        )

    return AllDataInfo(
        mk(train_idx, True, train_start_stride),
        mk(val_idx, False, val_start_stride),
        mk(test_idx, False),
        node_info=dataset.meta,
    )


def setup_workers(
    num_workers: int,
    cpu_core_util: int,
    load_into_ram: bool,
    enable_logging: bool = False,
) -> tuple[int, int | None, bool]:
    """
    Auto-tune DataLoader workers to the system's resources.
    """

    cpu_count = os.cpu_count() or 1
    if not load_into_ram:
        return num_workers, None, False

    target_workers = max(int(cpu_count * (cpu_core_util / 100.0)), 0)
    target_workers = min(target_workers, cpu_count)

    prefetch_factor = max(2, target_workers) if target_workers > 0 else None
    persistent_workers = target_workers > 0

    if enable_logging:
        logger.debug(
            f"Auto-tuned workers: {cpu_count=}, {target_workers=}, "
            f"{prefetch_factor=}, {persistent_workers=}"
        )

    return target_workers, prefetch_factor, persistent_workers


def make_dataloaders(
    dataset: Dataset,
    train_idx: Sequence[int] | np.ndarray,
    val_idx: Sequence[int] | np.ndarray,
    test_idx: Sequence[int] | np.ndarray,
    batch_size: int,
    num_workers: int = 0,
    prefetch_factor: int | None = None,
    persistent_workers: bool = False,
    pin_memory: bool = True,
    shuffle_train: bool = True,
    seed: int = 42,
    enable_logging: bool = False,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build the training, validation and testing dataloaders.
    """

    generator = torch.Generator().manual_seed(seed)

    train_ds = SubsetDataset(dataset, train_idx)
    val_ds = SubsetDataset(dataset, val_idx)
    test_ds = SubsetDataset(dataset, test_idx)

    # Force multiprocessing to use temporary files for inter-process communication.
    # Avoids descriptor exhaustion and cleanup deadlocks on Linux.
    if num_workers > 0:
        try:
            torch.multiprocessing.set_sharing_strategy("file_system")
        except RuntimeError:
            pass

    common_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory and torch.cuda.is_available(),
        "persistent_workers": persistent_workers and num_workers > 0,
    }

    if num_workers > 0 and prefetch_factor is not None:
        common_kwargs["prefetch_factor"] = prefetch_factor

    train_loader = DataLoader(
        train_ds,
        shuffle=shuffle_train,
        generator=generator if shuffle_train else None,
        **common_kwargs,
    )
    val_loader = DataLoader(val_ds, shuffle=False, **common_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **common_kwargs)

    if enable_logging:
        logger.debug(
            f"Created dataloaders with {batch_size=}, {num_workers=}, "
            f"pin_memory={common_kwargs['pin_memory']}"
        )

    return train_loader, val_loader, test_loader


def ensure_enough_ram(h5_path: str) -> None:
    """
    Estimate the dataset size and raise if it will not fit in available RAM.
    """

    with h5py.File(h5_path, "r") as f:
        nbytes = as_dataset(f, "inputs").nbytes + as_dataset(f, "targets").nbytes
    required = nbytes / (1024**3) * 1.2  # +20% headroom
    available = psutil.virtual_memory().available / (1024**3)

    if available < required:
        raise MemoryError(
            f"Not enough available RAM. required={required:.2f} GB, "
            f"available={available:.2f} GB"
        )
