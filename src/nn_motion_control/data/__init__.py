"""
Data pipeline: windowed HDF5 dataset, leakage-aware splitting, z-score normalisation,
and DataLoader construction.
"""

from nn_motion_control.data.dataset import (
    DatasetMetadata,
    H5TimeSeriesDataset,
    SubsetDataset,
)
from nn_motion_control.data.loaders import (
    AllDataInfo,
    build_time_series_splits,
    ensure_enough_ram,
    make_dataloaders,
    setup_workers,
)
from nn_motion_control.data.normalize import NormStats, fit_stats
from nn_motion_control.data.splits import (
    build_valid_window_starts,
    split_window_starts_contiguous,
    split_window_starts_random,
    validate_labels,
)

__all__ = [
    "AllDataInfo",
    "DatasetMetadata",
    "H5TimeSeriesDataset",
    "NormStats",
    "SubsetDataset",
    "build_time_series_splits",
    "build_valid_window_starts",
    "ensure_enough_ram",
    "fit_stats",
    "make_dataloaders",
    "setup_workers",
    "split_window_starts_contiguous",
    "split_window_starts_random",
    "validate_labels",
]
