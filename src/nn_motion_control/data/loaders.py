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

# Loader types that satisfy the trainer/evaluator contract (iterate
# batches + len + a dataset with the window count). RolloutWindowLoader
# yields 3-tuples consumed by RolloutTrainer; the others yield (x, y).
BatchLoader = DataLoader | DeviceWindowLoader | RolloutWindowLoader

# Fallback quiescent-seed filter thresholds when quiescent_seed omits
# 'max_dac' / 'max_speed'.
_DEFAULT_QUIESCENT_MAX_DAC = 20.0
_DEFAULT_QUIESCENT_MAX_SPEED = 2.0

# RAM headroom over the raw inputs+targets byte count required before
# loading a dataset fully into memory.
_RAM_HEADROOM_FACTOR = 1.2

# Floor DataLoader.prefetch_factor is auto-tuned to, once workers > 0.
_MIN_PREFETCH_FACTOR = 2


@dataclass(frozen=True)
class AllDataInfo:
    """
    The three dataloaders plus the metadata needed for
    inference/denormalisation.
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
      2. Build valid window-start rows per recording (no window crosses a
         boundary).
      3. Split the ordered starts (contiguous for windowed data, random
         for 'window_size == 1'), inserting a 'window_size - 1' gap at
         split seams.
      4. Fit z-score normalisation on the train rows only, then expose it
         everywhere.
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

    valid_starts = build_valid_window_starts(
        dataset.segment_offsets, window_size
    )

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
        # Fast path: hold the normalised tensors on 'device' and gather
        # each batch's windows in one vectorised op (no per-item Python,
        # no per-batch copy).
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

    return AllDataInfo(
        trn_loader, val_loader, tst_loader, node_info=dataset.meta
    )


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
    Build device-resident, vectorised window loaders sharing one
    normalised tensor.
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

    return (
        _make(train_idx, True),
        _make(val_idx, False),
        _make(test_idx, False),
    )


def select_quiescent_starts(
    inputs: torch.Tensor,
    starts: np.ndarray,
    window_size: int,
    dac_cols: Sequence[int],
    vel_cols: Sequence[int],
    pos_cols: Sequence[int],
    in_mean: torch.Tensor,
    in_std: torch.Tensor,
    max_dac: float,
    max_speed: float,
) -> np.ndarray:
    """
    Keep only window starts whose warmup is a quiescent relaxation hold.

    A controller starts from an operating condition, the stage held near
    rest at its relaxation point, not the mid-excitation transient a
    plant-identification window usually captures. A start is kept when
    the physical command magnitude stays below 'max_dac' across the whole
    window and the physical speed at the last window frame (the
    rollout's initial state) is below 'max_speed'. Velocity comes from
    'vel_cols' when present, else differenced from 'pos_cols'. 'inputs'
    is the normalised [T, F] tensor; 'in_mean'/'in_std' restore physical
    units.
    """

    if len(starts) == 0:
        return starts

    dev = inputs.device
    mean, std = in_mean.to(dev), in_std.to(dev)
    dac = torch.as_tensor(list(dac_cols), device=dev)
    # Physical |dac|, max over axes per row, then a windowed max over W
    # rows: element j of win_max holds the peak command anywhere in rows
    # [j, j + W).
    dac_abs = (inputs[:, dac] * std[dac] + mean[dac]).abs().amax(dim=1)
    win_max = dac_abs.unfold(0, window_size, 1).amax(dim=1)
    if len(vel_cols):
        vel = torch.as_tensor(list(vel_cols), device=dev)
        speed = (inputs[:, vel] * std[vel] + mean[vel]).norm(dim=1)
    else:
        pos = torch.as_tensor(list(pos_cols), device=dev)
        p = inputs[:, pos] * std[pos] + mean[pos]
        speed = torch.zeros(inputs.shape[0], device=dev)
        speed[1:] = (p[1:] - p[:-1]).norm(dim=1)
    st = torch.as_tensor(starts, device=dev, dtype=torch.long)
    keep = (win_max[st] < max_dac) & (speed[st + window_size - 1] < max_speed)

    return starts[keep.cpu().numpy()]


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
    vel_cols: list[int] | None = None,
    quiescent_seed: Mapping[str, float] | None = None,
) -> AllDataInfo:
    """
    Build leakage-aware rollout splits as RolloutWindowLoaders on the
    device.

    All three loaders share one normalised input tensor; W + max_horizon
    rows are reserved at each recording tail and split seam so no
    rollout crosses a boundary or leaks across splits. Normalisation is
    fit on the train rows only.

    'train_start_stride' / 'val_start_stride' subsample the window starts
    (adjacent rollout windows overlap in all but one warmup and one
    horizon row, so they are highly redundant); normalisation is still
    fit on the full train rows. 'drop_last' drops the ragged final batch
    so a compiled rollout step sees a constant batch shape.

    'quiescent_seed' ({"max_dac", "max_speed"}) restricts the train/val
    window starts to quiescent relaxation holds, the operating condition
    a controller starts from, via 'select_quiescent_starts'; 'vel_cols'
    supplies the velocity channels it reads. The plant path leaves both
    unset and sees every window.
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
    # Fit normalisation on the full (unstrided) train rows for the best
    # stats, then subsample the starts each loader actually iterates.
    dataset.fit_normalization(train_idx, "contiguous")
    assert dataset.meta is not None
    inputs, _ = dataset.normalized_arrays(
        device
    )  # rollout reads state/dac from inputs

    if quiescent_seed is not None:

        def quiescent(idx: np.ndarray) -> np.ndarray:
            return select_quiescent_starts(
                inputs,
                idx,
                window_size,
                dac_cols,
                vel_cols or [],
                pos_cols,
                *dataset.input_norm,
                max_dac=float(
                    quiescent_seed.get("max_dac", _DEFAULT_QUIESCENT_MAX_DAC)
                ),
                max_speed=float(
                    quiescent_seed.get(
                        "max_speed", _DEFAULT_QUIESCENT_MAX_SPEED
                    )
                ),
            )

        train_idx, val_idx = quiescent(train_idx), quiescent(val_idx)
        if len(train_idx) == 0 or len(val_idx) == 0:
            raise ValueError(
                "Quiescent-seed filter left no windows; relax "
                "max_dac / max_speed"
            )

    def mk(
        idx: np.ndarray, shuffle: bool, stride: int = 1
    ) -> RolloutWindowLoader:
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

    prefetch_factor = (
        max(_MIN_PREFETCH_FACTOR, target_workers)
        if target_workers > 0
        else None
    )
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

    # Force multiprocessing to use temporary files for inter-process
    # communication. Avoids descriptor exhaustion and cleanup deadlocks
    # on Linux.
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
        nbytes = (
            as_dataset(f, "inputs").nbytes + as_dataset(f, "targets").nbytes
        )
    required = nbytes / (1024**3) * _RAM_HEADROOM_FACTOR
    available = psutil.virtual_memory().available / (1024**3)

    if available < required:
        raise MemoryError(
            f"Not enough available RAM. required={required:.2f} GB, "
            f"available={available:.2f} GB"
        )
