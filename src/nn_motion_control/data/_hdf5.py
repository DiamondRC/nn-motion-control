"""
Small typed helpers over h5py.
"""

from __future__ import annotations

from typing import cast

import h5py


def as_dataset(f: h5py.File, key: str) -> h5py.Dataset:
    """
    Return f[key] typed as a Dataset.

    h5py's __getitem__ is typed as a broad ``Group | Dataset | Datatype`` union,
    our keys always address datasets => this cast keeps call sites type-clean.
    """

    return cast(h5py.Dataset, f[key])
