"""
Small typed helpers over h5py.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import h5py
import numpy as np

# Rows read per block when loading selected columns.
_READ_BLOCK_ROWS = 1 << 20


def as_dataset(f: h5py.File, key: str) -> h5py.Dataset:
    """
    Return f[key] typed as a Dataset.

    h5py's __getitem__ is typed as a broad 'Group | Dataset | Datatype'
    union, our keys always address datasets, so this cast keeps call sites
    type-clean.
    """

    return cast(h5py.Dataset, f[key])


def read_columns(dset: h5py.Dataset, cols: Sequence[int]) -> np.ndarray:
    """
    Rows of 'dset' restricted to 'cols' (any order), read in row blocks.

    Only the selected columns are ever held in full, so a wide file costs
    the memory of the columns used, not of every column.
    """

    out = np.empty((dset.shape[0], len(cols)), dtype=dset.dtype)
    step = _READ_BLOCK_ROWS

    for lo in range(0, dset.shape[0], step):
        out[lo : lo + step] = dset[lo : lo + step][:, list(cols)]

    return out
