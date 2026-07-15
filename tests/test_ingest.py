"""
Ingest (raw log -> HDF5) tests: alignment/row accounting, schema v2, input rejection.
"""

import h5py
import numpy as np
import pytest

from nn_motion_control.data import ingest as m
from nn_motion_control.data._hdf5 import as_dataset


def _write_raw(path, n=30, seed=0):
    """Write a synthetic raw log: 10 whitespace columns, increasing timestep."""
    rng = np.random.default_rng(seed)
    cols = [np.arange(n, dtype=float)]  # timestep
    for _ in range(9):  # x_input, x_DAC, ..., x_pos, y_pos, z_pos
        cols.append(rng.normal(size=n))
    np.savetxt(path, np.column_stack(cols))


def test_build_dataset_alignment_and_schema(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    _write_raw(raw / "a.txt", n=30, seed=0)
    _write_raw(raw / "b.txt", n=25, seed=1)
    out = tmp_path / "out.h5"

    m.build_dataset(raw, out, storage_dtype=np.dtype("float32"))

    with h5py.File(out, "r") as h:
        n_expected = (30 - m.ROWS_LOST_PER_FILE) + (25 - m.ROWS_LOST_PER_FILE)
        n = as_dataset(h, "inputs").shape[0]
        assert h.attrs["schema_version"] == 2
        assert n == n_expected
        assert int(as_dataset(h, "file_row_counts")[:].sum()) == n_expected
        segoff = as_dataset(h, "segment_offsets")[:]
        assert segoff[0] == 0 and segoff[-1] == n
        assert len(as_dataset(h, "input_labels")) == 16
        assert len(as_dataset(h, "target_labels")) == 13
        assert "input_norm_params" not in h  # normalisation left the build step
        assert "target_norm_params" not in h
        assert np.isfinite(as_dataset(h, "inputs")[:]).all()
        assert np.isfinite(as_dataset(h, "targets")[:]).all()
        assert as_dataset(h, "file_position_offsets").shape == (2, 3)


def test_header_row_is_rejected(tmp_path):
    bad = tmp_path / "bad.txt"
    bad.write_text("t x xd y yd z zd xp yp zp\n0 1 2 3 4 5 6 7 8 9\n")
    with pytest.raises(ValueError, match="not numeric"):
        m.read_raw_frame(bad)


def test_wrong_column_count_is_rejected(tmp_path):
    bad = tmp_path / "short.txt"
    bad.write_text("0 1 2 3 4\n1 2 3 4 5\n")
    with pytest.raises(ValueError, match="columns"):
        m.read_raw_frame(bad)


def test_no_matching_files_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        m.discover_files(tmp_path, "*.txt")


def test_overwrite_guard(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    _write_raw(raw / "a.txt", n=20)
    out = tmp_path / "out.h5"
    m.build_dataset(raw, out)
    with pytest.raises(FileExistsError, match="already exists"):
        m.build_dataset(raw, out, overwrite=False)
