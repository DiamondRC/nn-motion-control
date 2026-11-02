"""Dataloader correctness tests: boundaries, split gaps, train-only
normalization and correctly-aligned denorm params.
"""

import math

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


def test_steady_holds_count_anywhere_in_the_travel():
    # A steady large command holding the stage far from rest is a hold
    # under the ripple test, a changing small one is not.
    w, n_rows = 4, 40
    inputs = torch.zeros(n_rows, 3)
    inputs[:20, 0] = 900.0  # held off origin
    inputs[:20, 2] = 900.0  # by a large, steady command
    inputs[20:, 2] = torch.arange(20.0) % 2  # small but ripping command
    starts = np.arange(0, n_rows - w + 1, dtype=np.int64)
    common = {
        "window_size": w,
        "dac_cols": [2],
        "vel_cols": [1],
        "pos_cols": [0],
        "in_mean": torch.zeros(3),
        "in_std": torch.ones(3),
        "max_dac": 10.0,
        "max_speed": 5.0,
    }
    kept = dl.select_quiescent_starts(
        inputs, starts, **common, max_dac_ripple=0.5
    )
    assert set(kept.tolist()) == set(range(17))
    # The old magnitude test sees only the small command.
    old = dl.select_quiescent_starts(inputs, starts, **common)
    assert old.min() >= 20
    # A box excludes the far holds.
    boxed = dl.select_quiescent_starts(
        inputs,
        starts,
        **common,
        max_dac_ripple=0.5,
        bounds=np.array([[-100.0, 100.0]]),
    )
    assert len(boxed) == 0


def test_sliding_extent_matches_brute_force():
    from nn_motion_control.data.loaders import _sliding_extent

    x = torch.randn(200, 2)

    for span in (1, 2, 7, 64, 100):
        got = _sliding_extent(x, span)
        want = torch.stack(
            [
                (x[j : j + span].amax(0) - x[j : j + span].amin(0)).amax()
                for j in range(len(x) - span + 1)
            ]
        )
        assert torch.allclose(got, want)


def test_a_seed_still_ringing_is_not_settled():
    # A steady command but a position still swinging through its rest over
    # the settle span: the slow ring hides inside a short warmup window.
    w, n_rows, settle = 4, 400, 200
    inputs = torch.zeros(n_rows, 3)
    inputs[:, 2] = 500.0
    t = torch.arange(n_rows, dtype=torch.float32)
    inputs[:200, 0] = 3000.0 * torch.sin(2 * math.pi * t[:200] / 150.0)
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
        max_speed=1e9,
        speed_window=settle,
        max_dac_ripple=1.0,
        max_pos_ripple=100.0,
    )
    # Only windows whose whole settle span lies in the quiet half pass.
    assert kept.min() + w - settle >= 200


def test_balance_seeds_sets_the_off_origin_share():
    inputs = torch.zeros(100, 1)
    inputs[80:, 0] = 5000.0  # 20 off-origin rows, 80 near rest
    starts = np.arange(100, dtype=np.int64)

    def balance(fraction):
        out = dl.balance_seeds(
            inputs,
            starts,
            1,
            [0],
            torch.zeros(1),
            torch.ones(1),
            origin_radius=100.0,
            off_origin_fraction=fraction,
            seed=0,
        )

        return out, float((out >= 80).mean())

    out, share = balance(0.5)
    assert len(out) == 40 and share == 0.5  # all 20 far kept, 20 near drawn
    out, share = balance(0.1)
    assert share == pytest.approx(0.1, abs=0.01)  # far rows subsampled
    with pytest.raises(ValueError, match="Off_origin_fraction"):
        balance(1.0)


def _with_segment_split(tmp_path, synth_h5, splits, seg=(30, 30, 20, 20)):
    """Copy the synthetic rows into a v3-style file with per-segment splits."""
    offsets = np.concatenate([[0], np.cumsum(seg)]).astype("int64")
    path = tmp_path / "split.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("inputs", data=synth_h5["inputs"])
        f.create_dataset("targets", data=synth_h5["targets"])
        f.create_dataset("segment_offsets", data=offsets)
        f.create_dataset("segment_split", data=list(splits))
        f.create_dataset("input_labels", data=list(INPUT_LABELS))
        f.create_dataset("target_labels", data=list(TARGET_LABELS))
        f.attrs["schema_version"] = 3

    return str(path), offsets


def test_segment_split_keeps_each_recording_whole():
    offsets = np.array([0, 30, 60, 80, 100])
    split = np.array(["train", "val", "train", "test"])
    train, val, test = dl.split_window_starts_by_segment(offsets, split, WINDOW)

    def segment(idx):
        return set(np.searchsorted(offsets, idx, side="right") - 1)

    assert segment(train) == {0, 2}
    assert segment(val) == {1}
    assert segment(test) == {3}

    # Every window stays inside its recording.
    for idx in (train, val, test):
        seg = np.searchsorted(offsets, idx, side="right") - 1
        assert np.all(idx + WINDOW <= offsets[seg + 1])


def test_segment_split_needs_every_split():
    offsets = np.array([0, 30, 60])
    with pytest.raises(ValueError, match="test"):
        dl.split_window_starts_by_segment(
            offsets, np.array(["train", "val"]), WINDOW
        )


def test_segment_split_rows_and_train_only_stats(synth_h5, tmp_path):
    path, offsets = _with_segment_split(
        tmp_path, synth_h5, ["train", "val", "train", "test"]
    )
    ds = dl.H5TimeSeriesDataset(
        path,
        INPUTS_15,
        _targets_spec(),
        window_size=WINDOW,
        load_into_ram=True,
        dtype=torch.float32,
    )
    rows = ds.split_rows("train")
    assert isinstance(rows, np.ndarray)  # train is not adjacent here
    expected = np.r_[0:30, 60:80]
    assert np.array_equal(rows, expected)
    train, _, _ = dl.split_window_starts_by_segment(
        offsets, ds.segment_split, WINDOW
    )
    ds.fit_normalization(train, "segment")
    cols = [INPUT_LABELS.index(c) for c in INPUTS_15]
    want = synth_h5["inputs"][expected][:, cols].astype("float64").mean(0)
    assert np.allclose(ds._in_mean.numpy(), want, atol=1e-5)


def test_adjacent_train_split_is_a_slice(synth_h5, tmp_path):
    path, _ = _with_segment_split(
        tmp_path, synth_h5, ["train", "train", "val", "test"]
    )
    ds = dl.H5TimeSeriesDataset(path, INPUTS_15, _targets_spec(), WINDOW)
    assert ds.split_rows("train") == slice(0, 60)


def test_time_series_splits_follow_segment_split(synth_h5, tmp_path):
    path, offsets = _with_segment_split(
        tmp_path, synth_h5, ["train", "val", "train", "test"]
    )
    info = dl.build_time_series_splits(
        path,
        INPUTS_15,
        _targets_spec(),
        window_size=WINDOW,
        # The ratios are ignored once the dataset assigns splits.
        train_ratio=0.5,
        val_ratio=0.25,
        training_dtype=torch.float32,
        auto_tune_workers=False,
        pin_memory=False,
    )
    val_idx = info.val_loader.dataset.indices
    seg = np.searchsorted(offsets, val_idx, side="right") - 1
    assert set(seg) == {1}
    # Train is recordings 0 (30 rows) and 2 (20 rows), no seam gaps.
    assert len(info.trn_loader.dataset) == (30 - WINDOW + 1) + (20 - WINDOW + 1)


def test_windowed_speed_sees_through_noise_and_respects_recordings():
    w, span = 4, 10
    n_rows = 60
    inputs = torch.zeros(n_rows, 3)  # position, velocity, dac
    # A stage at rest with a +-3 nm per-step jitter: per-step speed is
    # noise, the span average is not.
    inputs[:, 0] = torch.tensor([3.0 if i % 2 else -3.0 for i in range(n_rows)])
    inputs[:, 1] = 6.0  # per-step velocity reads the jitter
    starts = np.arange(0, n_rows - w + 1, dtype=np.int64)
    common = {
        "window_size": w,
        "dac_cols": [2],
        "vel_cols": [1],
        "pos_cols": [0],
        "in_mean": torch.zeros(3),
        "in_std": torch.ones(3),
        "max_dac": 10.0,
        "max_speed": 2.0,
    }
    assert len(dl.select_quiescent_starts(inputs, starts, **common)) == 0
    kept = dl.select_quiescent_starts(
        inputs,
        starts,
        speed_window=span,
        segment_offsets=np.array([0, 30, n_rows]),
        **common,
    )
    # Every kept start's span stays inside its own recording.
    last = kept + w - 1
    assert len(kept) and np.all((last < 30) | (last - span >= 30))
    assert np.all(last - span >= 0)
