"""
Dynamical-systems measures of a sequence.

Largest Lyapunov exponent (predictability horizon / chaos, Rosenstein),
detrended-fluctuation Hurst exponent (long-range memory), recurrence
quantification (determinism, laminarity, trapping time), and increment
non-Gaussianity (kurtosis/skew + structure-function scaling). Pure numpy.
"""

from __future__ import annotations

import numpy as np

from nn_motion_control.data_analysis.dimension import delay_embed

# Lyapunov: a contiguous window (divergence tracking needs temporal order),
# the divergence horizon, and the Theiler window excluding temporal neighbours.
LYAP_MAX_POINTS = 3000
LYAP_HORIZON = 40
# Detrended fluctuation analysis: window sizes span this range of the series.
DFA_MIN_SCALE = 8
DFA_SCALES = 16
# Recurrence quantification: contiguous points, target recurrence rate (sets
# the threshold), and the minimum line length counted.
RQA_MAX_POINTS = 1500
RQA_RECURRENCE_RATE = 0.05
RQA_MIN_LINE = 2
# Increment lags for the structure-function scaling exponent.
INCREMENT_LAGS = (1, 2, 4, 8, 16, 32)


def largest_lyapunov(
    signal: np.ndarray,
    tau: int,
    m: int,
    sample_rate: float = 1.0,
    max_points: int = LYAP_MAX_POINTS,
    horizon: int = LYAP_HORIZON,
    theiler: int | None = None,
) -> float:
    """
    Rosenstein largest Lyapunov exponent (per second).

    Embeds the signal, tracks the mean log distance between initially-near
    trajectory pairs over 'horizon' steps, and returns the divergence slope
    times the sample rate. Positive means sensitive dependence (chaos).
    """

    emb = delay_embed(signal, tau, m)[:max_points]
    n = len(emb)
    if n < horizon + 2:
        return 0.0
    theiler = theiler if theiler is not None else max(tau, 1)
    gram = emb @ emb.T
    sq = np.diag(gram)
    dist = np.sqrt(np.clip(sq[:, None] + sq[None, :] - 2.0 * gram, 0.0, None))
    band = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :]) <= theiler
    dist[band] = np.inf
    neighbour = np.argmin(dist, axis=1)
    curve = []

    for k in range(horizon):
        i = np.arange(n - k)
        j = neighbour[i]
        keep = j + k < n
        i, j = i[keep], j[keep]
        step = np.linalg.norm(emb[i + k] - emb[j + k], axis=1)
        step = step[step > 0]
        if step.size == 0:
            break
        curve.append(float(np.mean(np.log(step))))
    if len(curve) < 2:
        return 0.0
    # Fit the initial divergence up to the plateau (nearby pairs saturate at
    # the attractor size); the rising slope is the exponent.
    values = np.array(curve)
    end = min(len(values), max(2, int(np.argmax(values)) + 1))
    slope = np.polyfit(np.arange(end), values[:end], 1)[0]

    return float(slope * sample_rate)


def dfa_hurst(
    signal: np.ndarray,
    min_scale: int = DFA_MIN_SCALE,
    n_scales: int = DFA_SCALES,
) -> float:
    """
    Detrended-fluctuation Hurst exponent.

    Fluctuation of the linearly-detrended integrated signal vs window size, on
    a log-log fit. 0.5 is uncorrelated (white); >0.5 persistent, <0.5
    anti-persistent.
    """

    x = np.asarray(signal, dtype=np.float64)
    profile = np.cumsum(x - x.mean())
    max_scale = len(x) // 4
    if max_scale <= min_scale:
        return 0.5
    scales = np.unique(
        np.logspace(np.log10(min_scale), np.log10(max_scale), n_scales).astype(
            np.int64
        )
    )
    fluct = []

    for s in scales:
        n_win = len(profile) // s
        block = profile[: n_win * s].reshape(n_win, s).T  # [s, n_win]
        t = np.arange(s)
        coeffs = np.polyfit(t, block, 1)
        trend = coeffs[0] * t[:, None] + coeffs[1]
        fluct.append(np.sqrt(np.mean((block - trend) ** 2)))

    return float(np.polyfit(np.log(scales), np.log(fluct), 1)[0])


def _run_lengths(mask: np.ndarray) -> np.ndarray:
    """
    Lengths of consecutive True runs in a 1-D boolean array (vectorised).
    """

    if not mask.any():
        return np.empty(0, dtype=np.int64)
    padded = np.concatenate([[False], mask, [False]])
    edges = np.diff(padded.astype(np.int8))

    return np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)


def _lines(recurrent: np.ndarray, diagonals: bool) -> list[np.ndarray]:
    if diagonals:
        return [np.diag(recurrent, k) for k in range(1, recurrent.shape[0])]

    return list(recurrent.T)


def _line_fraction(recurrent: np.ndarray, diagonals: bool, l_min: int) -> float:
    """
    Fraction of recurrence points on diagonal (or vertical) lines at least
    l_min long. Diagonals are counted one side of the symmetric plot only.
    """

    total = int(recurrent.sum())
    if total == 0:
        return 0.0
    on_lines = 0

    for line in _lines(recurrent, diagonals):
        runs = _run_lengths(np.asarray(line))
        on_lines += int(runs[runs >= l_min].sum())
    scale = 2 if diagonals else 1

    return float(scale * on_lines / total)


def rqa(
    points: np.ndarray,
    max_points: int = RQA_MAX_POINTS,
    target_rate: float = RQA_RECURRENCE_RATE,
    l_min: int = RQA_MIN_LINE,
) -> dict:
    """
    Recurrence quantification of a trajectory (contiguous points).

    Recurrence rate, determinism (diagonal-line fraction), laminarity
    (vertical-line fraction) and mean trapping time (mean vertical run). The
    threshold is set so about 'target_rate' of pairs recur.
    """

    x = np.asarray(points, dtype=np.float64)[:max_points]
    n = len(x)
    if n < l_min + 1:
        return {
            "recurrence_rate": 0.0,
            "determinism": 0.0,
            "laminarity": 0.0,
            "trapping_time": 0.0,
        }
    gram = x @ x.T
    sq = np.diag(gram)
    dist = np.sqrt(np.clip(sq[:, None] + sq[None, :] - 2.0 * gram, 0.0, None))
    iu = np.triu_indices(n, k=1)
    threshold = np.quantile(dist[iu], target_rate)
    recurrent = dist <= threshold
    np.fill_diagonal(recurrent, False)

    vertical = [_run_lengths(np.asarray(col)) for col in recurrent.T]
    trapped = np.concatenate(vertical) if vertical else np.empty(0)
    trapped = trapped[trapped >= l_min]
    trapping = float(trapped.mean()) if trapped.size else 0.0

    return {
        "recurrence_rate": float(recurrent.mean()),
        "determinism": _line_fraction(recurrent, True, l_min),
        "laminarity": _line_fraction(recurrent, False, l_min),
        "trapping_time": trapping,
    }


def increment_stats(
    signal: np.ndarray, lags: tuple[int, ...] = INCREMENT_LAGS
) -> dict:
    """
    Non-Gaussianity of the increments and their structure-function scaling.

    Excess kurtosis and skew of the one-step increment (0 is Gaussian), plus
    the exponent of the second-order structure function E|dx(lag)|^2 vs lag.
    """

    x = np.asarray(signal, dtype=np.float64)
    d = np.diff(x)
    std = d.std()
    if std == 0:
        return {"kurtosis": 0.0, "skew": 0.0, "structure_exponent": 0.0}
    centred = d - d.mean()
    kurtosis = float(np.mean(centred**4) / std**4 - 3.0)
    skew = float(np.mean(centred**3) / std**3)
    valid = [lag for lag in lags if lag < len(x)]
    s2 = [np.mean((x[lag:] - x[:-lag]) ** 2) for lag in valid]
    exponent = float(np.polyfit(np.log(valid), np.log(s2), 1)[0])

    return {"kurtosis": kurtosis, "skew": skew, "structure_exponent": exponent}
