"""Dataloader correctness tests: boundaries, split gaps, train-only
normalization and correctly-aligned denorm params.
"""

import h5py
import numpy as np
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from nn_motion_control import data as dl
from nn_motion_control.data.ingest import (
    INPUT_LABELS,
    TARGET_LABELS,
)

INPUTS_15 = [lbl for lbl in INPUT_LABELS if lbl != "timestep"]
TARGETS_12 = [
    lbl
    for lbl in TARGET_LABELS
    if lbl.endswith("_nxt") and lbl != "timestep_nxt"
]

WINDOW = 5


def _targets_spec():
    return [{lbl: (50 if "vel" in lbl else 1)} for lbl in TARGETS_12]


def test_no_window_crosses_a_boundary(synth_h5):
    offsets = synth_h5["offsets"]
    starts = dl.build_valid_window_starts(offsets, WINDOW)

    for s in starts:
        seg = int(np.searchsorted(offsets, s, side="right") - 1)
        assert s + WINDOW <= offsets[seg + 1]
    # Per segment: (len - window + 1) valid starts.
    expected = sum(
        int(e - s - WINDOW + 1)
        for s, e in zip(offsets[:-1], offsets[1:], strict=True)
    )
    assert len(starts) == expected


def test_contiguous_split_leaves_a_gap(synth_h5):
    starts = dl.build_valid_window_starts(synth_h5["offsets"], WINDOW)
    train, val, test = dl.split_window_starts_contiguous(
        starts, 0.7, 0.15, WINDOW
    )
    assert len(train) and len(val) and len(test)
    # Last train target/input row must sit strictly before the first
    # val input row.
    assert val.min() - (train.max() + WINDOW - 1) >= 1


def test_short_segment_is_skipped_or_errors():
    # window longer than any segment -> clear error
    offsets = np.array([0, 3, 6], dtype=np.int64)
    with pytest.raises(ValueError):
        dl.build_valid_window_starts(offsets, window_size=10)


def test_train_only_normalization(synth_h5, tmp_path):
    starts = dl.build_valid_window_starts(synth_h5["offsets"], WINDOW)
    train, val, _ = dl.split_window_starts_contiguous(starts, 0.7, 0.15, WINDOW)

    ds1 = dl.H5TimeSeriesDataset(
        synth_h5["path"],
        INPUTS_15,
        _targets_spec(),
        window_size=WINDOW,
        load_into_ram=True,
        dtype=torch.float32,
    )
    ds1.fit_normalization(train, "contiguous")
    mean_before = ds1._in_mean.clone()

    # Perturb only the val/test region and refit; train stats must not move.
    inputs2 = synth_h5["inputs"].copy()
    inputs2[int(val.min()) :] += 1e4
    path2 = tmp_path / "synth2.h5"
    with h5py.File(path2, "w") as f:
        f.create_dataset("inputs", data=inputs2)
        f.create_dataset("targets", data=synth_h5["targets"])
        f.create_dataset("segment_offsets", data=synth_h5["offsets"])
        f.create_dataset("input_labels", data=list(INPUT_LABELS))
        f.create_dataset("target_labels", data=list(TARGET_LABELS))
        f.attrs["schema_version"] = 2

    ds2 = dl.H5TimeSeriesDataset(
        str(path2),
        INPUTS_15,
        _targets_spec(),
        window_size=WINDOW,
        load_into_ram=True,
        dtype=torch.float32,
    )
    ds2.fit_normalization(train, "contiguous")
    assert torch.allclose(mean_before, ds2._in_mean, atol=1e-4)


def test_loss_weights_is_plain_vector(synth_h5):
    ds = dl.H5TimeSeriesDataset(
        synth_h5["path"],
        INPUTS_15,
        _targets_spec(),
        window_size=WINDOW,
        load_into_ram=True,
        dtype=torch.float32,
    )
    starts = dl.build_valid_window_starts(synth_h5["offsets"], WINDOW)
    train, _, _ = dl.split_window_starts_contiguous(starts, 0.7, 0.15, WINDOW)
    ds.fit_normalization(train, "contiguous")
    assert ds.meta is not None

    lw = ds.meta.loss_weights
    assert lw.shape == (len(TARGETS_12),)  # a vector, not an [N, N] matrix
    expected = [50 if "vel" in lbl else 1 for lbl in TARGETS_12]
    assert lw.tolist() == [float(x) for x in expected]


def test_denorm_params_keyed_by_selected_label(synth_h5):
    ds = dl.H5TimeSeriesDataset(
        synth_h5["path"],
        INPUTS_15,
        _targets_spec(),
        window_size=WINDOW,
        load_into_ram=True,
        dtype=torch.float32,
    )
    starts = dl.build_valid_window_starts(synth_h5["offsets"], WINDOW)
    train, _, _ = dl.split_window_starts_contiguous(starts, 0.7, 0.15, WINDOW)
    ds.fit_normalization(train, "contiguous")
    assert ds.meta is not None
    # Off-by-one fix: keys are exactly the selected labels, in order.
    assert list(ds.meta.target_denorm_params["mean"].keys()) == TARGETS_12
    assert list(ds.meta.input_denorm_params["mean"].keys()) == INPUTS_15


def test_ram_and_file_paths_agree(synth_h5):
    # The in-RAM and stream-from-file read paths must return identical windows.
    ram = dl.H5TimeSeriesDataset(
        synth_h5["path"],
        INPUTS_15,
        _targets_spec(),
        window_size=WINDOW,
        load_into_ram=True,
        dtype=torch.float32,
    )
    disk = dl.H5TimeSeriesDataset(
        synth_h5["path"],
        INPUTS_15,
        _targets_spec(),
        window_size=WINDOW,
        load_into_ram=False,
        dtype=torch.float32,
    )
    starts = dl.build_valid_window_starts(synth_h5["offsets"], WINDOW)
    train, _, _ = dl.split_window_starts_contiguous(starts, 0.7, 0.15, WINDOW)
    ram.fit_normalization(train, "contiguous")
    disk.fit_normalization(train, "contiguous")

    xr, yr = ram[int(starts[0])]
    xd, yd = disk[int(starts[0])]
    assert torch.allclose(xr, xd) and torch.allclose(yr, yd)
    disk.close_file()


def test_non_monotonic_column_order(synth_h5):
    # A channel-name expansion can map to non-increasing HDF5 column
    # indices (e.g. a DAC column, stored at the end of the file,
    # requested before a position column). Both read paths must
    # honour the requested order, not the file's column order.
    inputs = ["x_DAC_real", "x_pos"]  # HDF5 indices 13, 1 -> non-increasing
    targets = [{"x_pos_nxt": 1}]
    raw = synth_h5["inputs"][0]  # full row (all 16 columns)

    for ram in (True, False):
        ds = dl.H5TimeSeriesDataset(
            synth_h5["path"],
            inputs,
            targets,
            window_size=1,
            load_into_ram=ram,
            dtype=torch.float32,
        )
        x, _ = ds[0]  # identity normalisation until fit_normalization is called
        assert np.isclose(x[0].item(), raw[13])  # column 0 == x_DAC_real
        assert np.isclose(x[1].item(), raw[1])  # column 1 == x_pos
        ds.close_file()


@settings(max_examples=40, deadline=None)
@given(
    seg=st.lists(
        st.integers(min_value=3, max_value=20), min_size=1, max_size=5
    ),
    w=st.integers(min_value=1, max_value=6),
)
def test_no_window_crosses_boundary_property(seg, w):
    offsets = np.concatenate([[0], np.cumsum(seg)]).astype(np.int64)
    if max(seg) < w:  # no segment can hold a full window
        with pytest.raises(ValueError):
            dl.build_valid_window_starts(offsets, w)
        return
    starts = dl.build_valid_window_starts(offsets, w)

    for s in starts:
        k = int(np.searchsorted(offsets, s, side="right") - 1)
        assert s + w <= offsets[k + 1]  # window stays inside its segment


def test_pre_v2_schema_is_rejected(tmp_path):
    path = tmp_path / "old.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("inputs", data=np.zeros((10, 16), "float32"))
        f.create_dataset("targets", data=np.zeros((10, 13), "float32"))
        f.create_dataset("input_labels", data=list(INPUTS_15))
        f.create_dataset("target_labels", data=list(TARGETS_12))
        # no segment_offsets, no schema_version
    with pytest.raises(ValueError, match="schema v2"):
        dl.H5TimeSeriesDataset(
            str(path), INPUTS_15[:1], [{TARGETS_12[0]: 1}], window_size=1
        )


def test_select_quiescent_starts_keeps_only_holds():
    """Windows over a quiescent hold pass, excitation windows are
    filtered out.
    """
    w = 4
    # F=3: col0 position, col1 velocity, col2 dac. Identity normalisation.
    n_rows = 40
    inputs = torch.zeros(n_rows, 3)
    # First half: a hold (dac ~ 0, velocity ~ 0). Second half: excitation.
    inputs[20:, 2] = 100.0  # large dac
    inputs[20:, 1] = 50.0  # large velocity
    starts = np.arange(0, n_rows - w + 1, dtype=np.int64)
    kept = dl.select_quiescent_starts(
        inputs,
        starts,
        window_size=w,
        dac_cols=[2],
        vel_cols=[1],
        pos_cols=[0],
        in_mean=torch.zeros(3),
        in_std=torch.ones(3),
        max_dac=10.0,
        max_speed=5.0,
    )
    # A window is quiescent only if every row in [s, s+w) is a hold
    # and the last frame is slow: starts whose window reaches into
    # the excitation region are dropped.
    assert kept.max() <= 16
    assert set(kept.tolist()) == set(range(17))


def test_select_quiescent_starts_denormalises_thresholds():
    """Thresholds are physical: normalisation is inverted before comparing."""
    inputs = torch.zeros(6, 3)
    inputs[:, 2] = (
        1.0  # normalised dac = 1 -> physical = mean + std = 5 + 3 = 8
    )
    starts = np.arange(0, 5, dtype=np.int64)

    def keep(max_dac: float) -> int:
        return len(
            dl.select_quiescent_starts(
                inputs,
                starts,
                window_size=2,
                dac_cols=[2],
                vel_cols=[1],
                pos_cols=[0],
                in_mean=torch.tensor([0.0, 0.0, 5.0]),
                in_std=torch.tensor([1.0, 1.0, 3.0]),
                max_dac=max_dac,
                max_speed=1.0,
            )
        )

    assert keep(7.0) == 0  # physical dac 8 > 7 -> filtered out
    assert keep(9.0) == len(starts)  # physical dac 8 < 9 -> kept
