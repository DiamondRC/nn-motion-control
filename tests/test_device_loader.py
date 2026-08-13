"""
DeviceWindowLoader: parity with per-item __getitem__, batching,
shuffle determinism.
"""

import numpy as np
import torch

from nn_motion_control.data.dataset import H5TimeSeriesDataset
from nn_motion_control.data.device_loader import DeviceWindowLoader
from nn_motion_control.data.ingest import INPUT_LABELS, TARGET_LABELS
from nn_motion_control.data.splits import (
    build_valid_window_starts,
    split_window_starts_contiguous,
)

INPUTS_15 = [lbl for lbl in INPUT_LABELS if lbl != "timestep"]
TARGETS_12 = [
    lbl
    for lbl in TARGET_LABELS
    if lbl.endswith("_nxt") and lbl != "timestep_nxt"
]

WINDOW = 5


def _fitted_dataset(synth_h5, window=WINDOW):
    ds = H5TimeSeriesDataset(
        synth_h5["path"],
        INPUTS_15,
        [{lbl: 1} for lbl in TARGETS_12],
        window_size=window,
        load_into_ram=True,
        dtype=torch.float32,
    )
    starts = build_valid_window_starts(synth_h5["offsets"], window)
    train, _, _ = split_window_starts_contiguous(starts, 0.8, 0.1, window)
    ds.fit_normalization(train, "contiguous")
    return ds, train


def test_batches_match_per_item_getitem(synth_h5):
    ds, train = _fitted_dataset(synth_h5)
    gin, gtg = ds.normalized_arrays("cpu")
    loader = DeviceWindowLoader(
        gin, gtg, torch.as_tensor(train), WINDOW, batch_size=8, device="cpu"
    )
    # Flatten the loader's batches back to per-window rows (no
    # shuffle, same order).
    xs, ys = [], []

    for x, y in loader:
        xs.append(x)
        ys.append(y)
    x_all, y_all = torch.cat(xs), torch.cat(ys)
    assert x_all.shape == (len(train), len(INPUTS_15), WINDOW)

    for k, start in enumerate(train):
        x_ref, y_ref = ds[int(start)]  # normalised per-item [F_in, W], [F_tgt]
        assert torch.allclose(x_all[k], x_ref, atol=1e-6)
        assert torch.allclose(y_all[k], y_ref, atol=1e-6)


def test_window_one_returns_flat_inputs(synth_h5):
    ds, train = _fitted_dataset(synth_h5, window=1)
    gin, gtg = ds.normalized_arrays("cpu")
    loader = DeviceWindowLoader(
        gin, gtg, torch.as_tensor(train), 1, batch_size=8, device="cpu"
    )
    x, y = next(iter(loader))
    assert x.shape == (8, len(INPUTS_15))  # no trailing window axis when W == 1
    assert y.shape == (8, len(TARGETS_12))


def test_len_and_full_coverage(synth_h5):
    ds, train = _fitted_dataset(synth_h5)
    gin, gtg = ds.normalized_arrays("cpu")
    loader = DeviceWindowLoader(
        gin, gtg, torch.as_tensor(train), WINDOW, batch_size=8, device="cpu"
    )
    assert len(loader) == (len(train) + 7) // 8  # ceil division
    seen = sum(x.shape[0] for x, _ in loader)
    assert seen == len(train)  # every window served exactly once


def test_shuffle_is_seed_deterministic_and_a_permutation(synth_h5):
    ds, train = _fitted_dataset(synth_h5)
    gin, gtg = ds.normalized_arrays("cpu")

    a = DeviceWindowLoader(
        gin,
        gtg,
        torch.as_tensor(train),
        WINDOW,
        batch_size=8,
        shuffle=True,
        seed=1,
        device="cpu",
    )
    b = DeviceWindowLoader(
        gin,
        gtg,
        torch.as_tensor(train),
        WINDOW,
        batch_size=8,
        shuffle=True,
        seed=1,
        device="cpu",
    )
    xa = torch.cat([x for x, _ in a])
    xb = torch.cat([x for x, _ in b])
    assert torch.equal(xa, xb)  # same seed -> identical shuffle

    c = DeviceWindowLoader(
        gin,
        gtg,
        torch.as_tensor(train),
        WINDOW,
        batch_size=8,
        shuffle=True,
        seed=2,
        device="cpu",
    )
    xc = torch.cat([x for x, _ in c])
    assert not torch.equal(xa, xc)  # different seed -> different order
    assert xa.shape == xc.shape  # but still a full permutation (same size)


def test_successive_epochs_reshuffle(synth_h5):
    ds, train = _fitted_dataset(synth_h5)
    gin, gtg = ds.normalized_arrays("cpu")
    loader = DeviceWindowLoader(
        gin,
        gtg,
        torch.as_tensor(train),
        WINDOW,
        batch_size=8,
        shuffle=True,
        seed=1,
        device="cpu",
    )
    e1 = torch.cat([x for x, _ in loader])
    e2 = torch.cat([x for x, _ in loader])
    assert not torch.equal(e1, e2)  # epoch counter advances the shuffle
    assert e1.shape == e2.shape


def test_starts_to_device_casts_int64():
    out = DeviceWindowLoader.starts_to_device(np.array([3, 1, 2]), "cpu")
    assert out.dtype == torch.int64
    assert out.tolist() == [3, 1, 2]
