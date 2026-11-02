"""
DataLoader construction and the top-level split/normalise/load orchestrator.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import h5py
import numpy as np
import psutil
import torch
from torch.utils.data import DataLoader, Dataset

from nn_motion_control.data._hdf5 import as_dataset
from nn_motion_control.data.augment import LinearityAugment
from nn_motion_control.data.capture_seeds import (
    DEFAULT_SEED_STRIDE,
    append_capture_seeds,
)
from nn_motion_control.data.dataset import (
    DatasetMetadata,
    H5TimeSeriesDataset,
    SubsetDataset,
)
from nn_motion_control.data.device_loader import (
    DeviceWindowLoader,
    RolloutWindowLoader,
)
from nn_motion_control.data.panda import DatasetFrame
from nn_motion_control.data.splits import (
    build_valid_window_starts,
    split_window_starts_by_segment,
    split_window_starts_contiguous,
    split_window_starts_random,
)

logger = logging.getLogger(os.path.basename(__file__))

# Loader types that satisfy the trainer/evaluator contract (iterate
# batches + len + a dataset with the window count). RolloutWindowLoader
# yields 3-tuples consumed by RolloutTrainer; the others yield (x, y).
BatchLoader = (
    DataLoader | DeviceWindowLoader | RolloutWindowLoader | LinearityAugment
)

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
    drop_last_train: bool = False,
    small_signal: Mapping[str, Any] | None = None,
    train_start_stride: int = 1,
) -> AllDataInfo:
    """
    Create boundary-aware, leakage-free train/val/test splits.

    'drop_last_train' drops the ragged final training batch, so a
    CUDA-graph-compiled step only ever sees one batch shape.

    Order of operations:
      1. Load metadata + (optionally) selected columns into RAM.
      2. Build valid window-start rows per recording (no window crosses a
         boundary).
      3. Split by each recording's assigned split when the dataset has
         one (schema v3, the ratios are then unused), else split the
         ordered starts (contiguous for windowed data, random for
         'window_size == 1'), inserting a 'window_size - 1' gap at split
         seams.
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

    if dataset.segment_split is not None:
        split_mode = "segment"
        _log_segment_split(h5_path)
        train_idx, val_idx, test_idx = split_window_starts_by_segment(
            dataset.segment_offsets, dataset.segment_split, window_size
        )
    else:
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
            drop_last_train=drop_last_train,
            small_signal=small_signal,
            train_start_stride=train_start_stride,
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


def _log_segment_split(h5_path: str) -> None:
    logger.info(
        "%s assigns each recording a split, train/validation ratios unused",
        h5_path,
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
    drop_last_train: bool = False,
    small_signal: Mapping[str, Any] | None = None,
    train_start_stride: int = 1,
) -> tuple[DeviceWindowLoader, DeviceWindowLoader, DeviceWindowLoader]:
    """
    Build device-resident, vectorised window loaders sharing one
    normalised tensor, the training starts optionally oversampled in the
    small-signal regime (see 'oversample_small_signal').
    """

    inputs, targets = dataset.normalized_arrays(device)
    # Neighbouring windows overlap in all but one row, so a stride keeps an
    # epoch representative at a fraction of the cost.
    train_idx = train_idx[::train_start_stride]
    if small_signal is not None:
        assert dataset.meta is not None
        labels = list(dataset.meta.input_denorm_params["mean"].keys())
        dac_cols = [i for i, label in enumerate(labels) if "DAC" in label]
        train_idx = oversample_small_signal(
            inputs,
            train_idx,
            window_size,
            dac_cols,
            *dataset.input_norm,
            small_signal,
            dataset.segment_offsets,
        )

    def _make(
        idx: np.ndarray, shuffle: bool, drop_last: bool = False
    ) -> DeviceWindowLoader:
        return DeviceWindowLoader(
            inputs,
            targets,
            DeviceWindowLoader.starts_to_device(idx, device),
            window_size=window_size,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            device=device,
            drop_last=drop_last,
        )

    return (
        _make(train_idx, True, drop_last_train),
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
    speed_window: int = 1,
    segment_offsets: np.ndarray | None = None,
    max_dac_ripple: float | None = None,
    bounds: np.ndarray | None = None,
    max_pos_ripple: float | None = None,
) -> np.ndarray:
    """
    Keep only window starts whose warmup is a quiescent relaxation hold.

    A controller starts from an operating condition, the stage held near
    rest at its relaxation point, not the mid-excitation transient a
    plant-identification window usually captures. A start is kept when
    the physical command magnitude stays below 'max_dac' across the whole
    window and the physical speed at the last window frame (the
    rollout's initial state) is below 'max_speed'. Velocity comes from
    'vel_cols' when present, else differenced from 'pos_cols'. A
    'speed_window' above one instead takes the mean speed over that many
    steps (displacement over the span), so sensor noise on the per-step
    velocity of a stage at rest does not decide the test. That span may
    reach back before the window, so with 'segment_offsets' a start whose
    span would cross into the previous recording is dropped. 'inputs' is
    the normalised [T, F] tensor, 'in_mean'/'in_std' restore physical
    units.

    With 'max_dac_ripple' a hold is a steady command at any level instead,
    so the stage may be held anywhere in its travel: the command's
    peak-to-peak on every axis stays below the ripple, and with
    'max_pos_ripple' the position's too, over the longer of the window and
    'speed_window' up to the last frame. Make that span outlast the slowest
    resonance period, so a stage still ringing is not taken for settled.
    'bounds' [A, 2] (physical position) additionally keeps only seeds
    inside that box.
    """

    if len(starts) == 0:
        return starts

    dev = inputs.device
    mean, std = in_mean.to(dev), in_std.to(dev)
    dac = torch.as_tensor(list(dac_cols), device=dev)
    # Physical |dac|, max over axes per row, then a windowed max over W
    # rows: element j of win_max holds the peak command anywhere in rows
    # [j, j + W).
    dac_phys = inputs[:, dac] * std[dac] + mean[dac]
    settle = max(window_size, speed_window)
    if max_dac_ripple is None:
        win_max = dac_phys.abs().amax(dim=1).unfold(0, window_size, 1)
        win_max = win_max.amax(dim=1)
        dac_limit = max_dac
        first = torch.as_tensor(starts, device=dev, dtype=torch.long)
    else:
        win_max = _sliding_extent(dac_phys, settle)
        dac_limit = max_dac_ripple
        # The span ends on the warmup's last frame.
        first = torch.as_tensor(
            starts + window_size - settle, device=dev, dtype=torch.long
        )
    if speed_window > 1:
        pos = torch.as_tensor(list(pos_cols), device=dev)
        p = inputs[:, pos] * std[pos] + mean[pos]
        speed = torch.full((inputs.shape[0],), float("inf"), device=dev)
        span = p[speed_window:] - p[:-speed_window]
        speed[speed_window:] = span.norm(dim=1) / speed_window
    elif len(vel_cols):
        vel = torch.as_tensor(list(vel_cols), device=dev)
        speed = (inputs[:, vel] * std[vel] + mean[vel]).norm(dim=1)
    else:
        pos = torch.as_tensor(list(pos_cols), device=dev)
        p = inputs[:, pos] * std[pos] + mean[pos]
        speed = torch.zeros(inputs.shape[0], device=dev)
        speed[1:] = (p[1:] - p[:-1]).norm(dim=1)
    st = torch.as_tensor(starts, device=dev, dtype=torch.long)
    inside = first >= 0
    at = first.clamp_min(0)
    keep = (
        inside
        & (win_max[at] < dac_limit)
        & (speed[st + window_size - 1] < max_speed)
    )
    if max_dac_ripple is not None and max_pos_ripple is not None:
        pos = torch.as_tensor(list(pos_cols), device=dev)
        p = inputs[:, pos] * std[pos] + mean[pos]
        keep &= _sliding_extent(p, settle)[at] < max_pos_ripple
    if bounds is not None:
        pos = torch.as_tensor(list(pos_cols), device=dev)
        p = inputs[st + window_size - 1][:, pos] * std[pos] + mean[pos]
        box = torch.as_tensor(bounds, device=dev, dtype=p.dtype)
        keep &= ((p >= box[:, 0]) & (p <= box[:, 1])).all(dim=1)
    keep = keep.cpu().numpy()
    if speed_window > 1 and segment_offsets is not None:
        last = starts + window_size - 1
        seg = np.searchsorted(segment_offsets, last, side="right") - 1
        keep &= last - settle >= segment_offsets[seg]

    return starts[keep]


def _sliding_extent(x: torch.Tensor, span: int) -> torch.Tensor:
    """
    Peak-to-peak over every run of 'span' rows of 'x' [T, A], the largest
    across columns: element j covers rows [j, j + span).
    """

    def slide(op):
        out, width = x, 1

        # Doubling: after each pass out[j] spans rows [j, j + 2 * width).
        while 2 * width <= span:
            out = op(out[:-width], out[width:])
            width *= 2
        rest = span - width
        if rest:
            out = op(out[:-rest], out[rest:])

        return out

    return (slide(torch.maximum) - slide(torch.minimum)).amax(dim=1)


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _optional_array(value: Any) -> np.ndarray | None:
    return None if value is None else np.asarray(value, dtype=float)


def balance_seeds(
    inputs: torch.Tensor,
    starts: np.ndarray,
    window_size: int,
    pos_cols: Sequence[int],
    in_mean: torch.Tensor,
    in_std: torch.Tensor,
    origin_radius: float,
    off_origin_fraction: float,
    seed: int,
) -> np.ndarray:
    """
    Resample seed starts so 'off_origin_fraction' of them sit further than
    'origin_radius' (physical position, any axis) from rest.

    The more common class is subsampled, seeded, the rarer one kept whole,
    so off-origin seeds keep the data's own spread across the travel.
    """

    if not 0.0 < off_origin_fraction < 1.0:
        raise ValueError("Off_origin_fraction must be in (0, 1)")
    if len(starts) == 0:
        return starts
    dev = inputs.device
    pos = torch.as_tensor(list(pos_cols), device=dev)
    st = torch.as_tensor(starts, device=dev, dtype=torch.long)
    mean, std = in_mean.to(dev), in_std.to(dev)
    p = inputs[st + window_size - 1][:, pos] * std[pos] + mean[pos]
    off = (p.abs().amax(dim=1) > origin_radius).cpu().numpy()
    near, far = starts[~off], starts[off]
    if len(near) == 0 or len(far) == 0:
        logger.warning(
            "Seed balance: %d near-origin and %d off-origin seeds, kept as is",
            len(near),
            len(far),
        )
        return starts

    ratio = off_origin_fraction / (1.0 - off_origin_fraction)
    rng = np.random.default_rng(seed)
    if len(far) < ratio * len(near):
        near = rng.choice(near, int(round(len(far) / ratio)), replace=False)
    else:
        far = rng.choice(far, int(round(len(near) * ratio)), replace=False)
    logger.info(
        "Seed balance: %d near-origin and %d off-origin seeds",
        len(near),
        len(far),
    )

    return np.sort(np.concatenate([near, far]))


def oversample_small_signal(
    inputs: torch.Tensor,
    starts: np.ndarray,
    window_size: int,
    dac_cols: Sequence[int],
    in_mean: torch.Tensor,
    in_std: torch.Tensor,
    small_signal: Mapping[str, Any],
    segment_offsets: np.ndarray | None = None,
) -> np.ndarray:
    """
    Repeat 'factor' times each start whose command activity falls in
    'band' (physical command std, largest axis, over the 'span' samples
    ending at the window's last frame).

    The regime a precision stage holds in (a count or so of command
    dithering) is a sliver of identification data, so a model fit to it
    barely learns the stage's small-signal response. Spans that would reach
    into the previous recording are left at their weight.
    """

    lo, hi = (float(v) for v in small_signal["band"])
    factor = int(small_signal["factor"])
    span = int(small_signal["span"])
    if factor < 1 or span < 2 or not 0 <= lo < hi:
        raise ValueError(
            "Small_signal needs factor >= 1, span >= 2 and 0 <= band lo < hi"
        )
    if factor == 1 or len(starts) == 0:
        return starts
    dev = inputs.device
    dac = torch.as_tensor(list(dac_cols), device=dev)
    mean, std = in_mean.to(dev), in_std.to(dev)
    x = (inputs[:, dac] * std[dac] + mean[dac]).double()
    zero = torch.zeros(1, x.shape[1], device=dev, dtype=x.dtype)
    c1 = torch.cat([zero, x.cumsum(0)])
    c2 = torch.cat([zero, (x * x).cumsum(0)])
    last = torch.as_tensor(starts + window_size - 1, device=dev)
    first = last - span + 1
    ok = first >= 0
    if segment_offsets is not None:
        seg = np.searchsorted(
            segment_offsets, starts + window_size - 1, "right"
        )
        seg_start = torch.as_tensor(segment_offsets[seg - 1], device=dev)
        ok &= first >= seg_start
    a, b = first.clamp_min(0), last + 1
    s1, s2 = c1[b] - c1[a], c2[b] - c2[a]
    var = ((s2 - s1 * s1 / span) / span).clamp_min(0.0)
    activity = var.sqrt().amax(dim=1)
    pick = (ok & (activity >= lo) & (activity < hi)).cpu().numpy()
    extra = np.repeat(starts[pick], factor - 1)
    logger.info(
        "Small-signal oversampling: %d of %d starts in %s counts, x%d",
        int(pick.sum()),
        len(starts),
        (lo, hi),
        factor,
    )

    return np.sort(np.concatenate([starts, extra]))


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
    quiescent_seed: Mapping[str, Any] | None = None,
    capture_seed: Mapping[str, Any] | None = None,
    small_signal: Mapping[str, Any] | None = None,
) -> AllDataInfo:
    """
    Build leakage-aware rollout splits as RolloutWindowLoaders on the
    device.

    All three loaders share one normalised input tensor; W + max_horizon
    rows are reserved at each recording tail and split seam so no
    rollout crosses a boundary or leaks across splits. A dataset that
    assigns each recording a split (schema v3) is split by it and the
    ratios are unused. Normalisation is fit on the train rows only.

    'train_start_stride' / 'val_start_stride' subsample the window starts
    (adjacent rollout windows overlap in all but one warmup and one
    horizon row, so they are highly redundant); normalisation is still
    fit on the full train rows. 'drop_last' drops the ragged final batch
    so a compiled rollout step sees a constant batch shape.

    'quiescent_seed' ({"max_dac", "max_speed", "speed_window",
    "max_dac_ripple", "max_pos_ripple", "bounds", "origin_radius",
    "off_origin_fraction"})
    restricts the train/val window starts to holds, the operating condition
    a controller starts from, via 'select_quiescent_starts', and with
    'off_origin_fraction' rebalances them between rest and the rest of the
    travel via 'balance_seeds'. 'vel_cols' supplies the velocity channels
    it reads. The plant path leaves both unset and sees every window.

    'capture_seed' ({"paths", "axes", "resolution_nm", "rate_hz", "axis",
    "stride", "val_fraction"})
    additionally seeds train/val rollouts from recorded closed-loop
    captures, covering the mid-scan operating states the open-loop
    identification set never visits (see 'data/capture_seeds.py').
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
    if dataset.segment_split is not None:
        split_mode = "segment"
        _log_segment_split(h5_path)
        train_idx, val_idx, test_idx = split_window_starts_by_segment(
            dataset.segment_offsets,
            dataset.segment_split,
            window_size,
            horizon=max_horizon,
        )
    else:
        split_mode = "contiguous"
        valid = build_valid_window_starts(
            dataset.segment_offsets, window_size, horizon=max_horizon
        )
        train_idx, val_idx, test_idx = split_window_starts_contiguous(
            valid, train_ratio, val_ratio, window_size, horizon=max_horizon
        )
    # Fit normalisation on the full (unstrided) train rows for the best
    # stats, then subsample the starts each loader actually iterates.
    dataset.fit_normalization(train_idx, split_mode)
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
                speed_window=int(quiescent_seed.get("speed_window", 1)),
                segment_offsets=dataset.segment_offsets,
                max_dac_ripple=_optional_float(
                    quiescent_seed.get("max_dac_ripple")
                ),
                max_pos_ripple=_optional_float(
                    quiescent_seed.get("max_pos_ripple")
                ),
                bounds=_optional_array(quiescent_seed.get("bounds")),
            )

        train_idx, val_idx = quiescent(train_idx), quiescent(val_idx)
        if "off_origin_fraction" in quiescent_seed:

            def balanced(idx: np.ndarray) -> np.ndarray:
                return balance_seeds(
                    inputs,
                    idx,
                    window_size,
                    pos_cols,
                    *dataset.input_norm,
                    origin_radius=float(quiescent_seed["origin_radius"]),
                    off_origin_fraction=float(
                        quiescent_seed["off_origin_fraction"]
                    ),
                    seed=seed,
                )

            train_idx, val_idx = balanced(train_idx), balanced(val_idx)
        if len(train_idx) == 0 or len(val_idx) == 0:
            raise ValueError(
                "Quiescent-seed filter left no windows, "
                "relax max_dac / max_speed"
            )

    if capture_seed is not None:
        inputs, seeds = append_capture_seeds(
            inputs,
            *dataset.input_norm,
            capture_paths=list(capture_seed["paths"]),
            frame=DatasetFrame.of(h5_path, capture_seed.get("rate_hz")),
            axes=list(capture_seed["axes"]),
            resolution_nm=float(capture_seed["resolution_nm"]),
            axis=int(capture_seed.get("axis", 0)),
            window_size=window_size,
            max_horizon=max_horizon,
            pos_cols=pos_cols,
            vel_cols=vel_cols or [],
            dac_cols=dac_cols,
            stride=int(capture_seed.get("stride", DEFAULT_SEED_STRIDE)),
        )
        # Hold out a tail slice of the capture seeds for validation, so
        # val measures the same operating states without training on them.
        val_frac = float(capture_seed.get("val_fraction", 0.1))
        cut = max(1, int(len(seeds) * (1.0 - val_frac)))
        train_idx = np.concatenate([train_idx, seeds[:cut]])
        val_idx = np.concatenate([val_idx, seeds[cut:]])

    if small_signal is not None:
        train_idx = oversample_small_signal(
            inputs,
            train_idx,
            window_size,
            dac_cols,
            *dataset.input_norm,
            small_signal,
            dataset.segment_offsets,
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
