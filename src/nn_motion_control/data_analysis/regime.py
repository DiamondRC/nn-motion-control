"""
Operating-regime structure.

Stationarity (do the statistics drift over time?) and spatial homogeneity (do
the dynamics depend on absolute position - flexure stiffening near the travel
limits?). The homogeneity read bears on whether a plant is valid across the
workspace. Pure numpy.
"""

from __future__ import annotations

import numpy as np

STATIONARITY_CHUNKS = 10
HOMOGENEITY_BINS = 12
HOMOGENEITY_MIN_COUNT = 50


def stationarity(signal: np.ndarray, segment_offsets: np.ndarray) -> dict:
    """
    Within-recording drift index, normalised by the signal std.

    Each recording (from 'segment_offsets') is split into STATIONARITY_CHUNKS
    time windows; 'mean_drift' is the median dispersion of the window means and
    'std_drift' that of the window stds, both over the overall std. Near 0 is
    stationary.
    """

    x = np.asarray(signal, dtype=np.float64)
    overall = x.std()
    if overall == 0:
        return {"mean_drift": 0.0, "std_drift": 0.0}
    mean_disp, std_disp = [], []

    for k in range(len(segment_offsets) - 1):
        seg = x[segment_offsets[k] : segment_offsets[k + 1]]
        if len(seg) < STATIONARITY_CHUNKS:
            continue
        size = len(seg) // STATIONARITY_CHUNKS
        chunks = seg[: size * STATIONARITY_CHUNKS].reshape(
            STATIONARITY_CHUNKS, size
        )
        mean_disp.append(chunks.mean(axis=1).std())
        std_disp.append(chunks.std(axis=1).std())
    if not mean_disp:
        return {"mean_drift": 0.0, "std_drift": 0.0}

    return {
        "mean_drift": float(np.median(mean_disp) / overall),
        "std_drift": float(np.median(std_disp) / overall),
    }


def spatial_homogeneity(
    pos: np.ndarray,
    activity: np.ndarray,
    n_bins: int = HOMOGENEITY_BINS,
) -> dict:
    """
    Whether a dynamics proxy varies with absolute position.

    Bins samples by position and takes the std of 'activity' (e.g.
    acceleration) in each well-populated bin; 'cv' is the coefficient of
    variation of those per-bin values. Near 0 means position-independent
    (homogeneous) dynamics; large means position-dependent (e.g. end-of-travel
    stiffening).
    """

    p = np.asarray(pos, dtype=np.float64)
    a = np.asarray(activity, dtype=np.float64)
    edges = np.quantile(p, np.linspace(0.0, 1.0, n_bins + 1))
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    per_bin = []

    for b in range(n_bins):
        mask = idx == b
        if int(mask.sum()) >= HOMOGENEITY_MIN_COUNT:
            per_bin.append(float(np.std(a[mask])))
    values = np.array(per_bin)
    mean = values.mean() if values.size else 0.0
    cv = float(values.std() / mean) if mean > 0 else 0.0

    return {"cv": cv, "per_bin": values}
