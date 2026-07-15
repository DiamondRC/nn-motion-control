"""Metrics/denorm tests (Workstream C): the reshape must map state k -> row k, and
normalize/denormalize must round-trip.
"""

import numpy as np
import torch

from nn_motion_control import data as dl
from nn_motion_control.data.ingest import (
    INPUT_LABELS,
    TARGET_LABELS,
)

INPUTS_15 = [lbl for lbl in INPUT_LABELS if lbl != "timestep"]
TARGETS_12 = [lbl for lbl in TARGET_LABELS if lbl != "timestep_nxt"]


def test_reshape_maps_state_to_row():
    # 3 samples x 2 states, flattened sample-major (as collected in the test loop).
    per_sample = np.array([[10, 20], [11, 21], [12, 22]])
    flat = per_sample.flatten()

    n_points, n_states = 3, 2
    reshaped = flat.reshape(n_points, n_states).T  # the fix in test_saved_model

    assert reshaped.shape == (n_states, n_points)
    assert list(reshaped[0]) == [10, 11, 12]  # state 0 across all samples
    assert list(reshaped[1]) == [20, 21, 22]  # state 1 across all samples


def test_normalize_denormalize_roundtrip(synth_h5):
    ds = dl.H5TimeSeriesDataset(
        synth_h5["path"],
        INPUTS_15,
        [{lbl: 1} for lbl in TARGETS_12],
        window_size=1,
        load_into_ram=True,
        dtype=torch.float32,
    )
    starts = dl.build_valid_window_starts(synth_h5["offsets"], 1)
    train, _, _ = dl.split_window_starts_random(starts, 0.7, 0.15, 0)
    ds.fit_normalization(train, "random")
    assert ds.meta is not None

    tmean = np.array(list(ds.meta.target_denorm_params["mean"].values()))
    tstd = np.array(list(ds.meta.target_denorm_params["std"].values()))

    row = int(train[0])
    _, y = ds[row]  # normalized target
    denorm = y.numpy() * tstd + tmean
    raw = synth_h5["targets"][row, 1:]  # skip the timestep_nxt column
    assert np.allclose(denorm, raw, atol=1e-2)
