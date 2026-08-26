"""
Dimensionality of a sequence.

Linear degrees of freedom (PCA participation ratio), nonlinear attractor
dimension (Grassberger-Procaccia correlation dimension on a delay embedding,
with the D2-vs-m saturation), and Higuchi fractal dimension of the raw
waveform. Pure numpy: pairwise distances via the Gram-matrix identity.
"""

from __future__ import annotations

import numpy as np

VARIANCE_FRACTIONS = (0.9, 0.95, 0.99)
# Delay-embedding: autocorrelation drop used to pick tau, and the embedding
# dimensions swept for the D2 saturation curve.
TAU_MAX_LAG = 2000
TAU_THRESHOLD = 1.0 / np.e
EMBED_DIMS = (1, 2, 3, 4, 5, 6, 8, 10)
# Correlation integral: points kept (it is O(N^2)), radius grid, and the
# log C(r) quantile window fit for the scaling slope.
CORR_SUBSAMPLE = 3000
CORR_RADII = 40
CORR_QUANTILES = (0.02, 0.98)
SCALING_WINDOW = (0.1, 0.7)
HIGUCHI_KMAX = 16


def pca_dimension(states: np.ndarray) -> dict:
    """
    Linear effective dimension of a state cloud.

    Returns the participation ratio (sum lambda)^2 / sum(lambda^2), the
    explained-variance ratios (descending), and the component count reaching
    each of VARIANCE_FRACTIONS.
    """

    x = np.asarray(states, dtype=np.float64)
    x = x - x.mean(axis=0)
    cov = x.T @ x / len(x)
    ev = np.clip(np.linalg.eigvalsh(cov)[::-1], 0.0, None)
    total = ev.sum()
    if total <= 0:
        return {
            "participation_ratio": 0.0,
            "explained": ev,
            "components": dict.fromkeys(VARIANCE_FRACTIONS, 0),
        }
    explained = ev / total
    cumulative = np.cumsum(explained)
    components = {
        frac: int(np.searchsorted(cumulative, frac) + 1)
        for frac in VARIANCE_FRACTIONS
    }

    return {
        "participation_ratio": float(total**2 / np.sum(ev**2)),
        "explained": explained,
        "components": components,
    }


def _autocorr(x: np.ndarray, max_lag: int) -> np.ndarray:
    centred = x - x.mean()
    spec = np.fft.rfft(centred, 2 * len(centred))
    ac = np.fft.irfft(spec * np.conj(spec))[: max_lag + 1]

    return ac / ac[0] if ac[0] > 0 else ac


def estimate_tau(signal: np.ndarray, max_lag: int = TAU_MAX_LAG) -> int:
    """
    Delay for embedding: first lag where the autocorrelation falls below 1/e.
    """

    x = np.asarray(signal, dtype=np.float64)
    ac = _autocorr(x, min(max_lag, len(x) - 1))
    below = np.where(ac < TAU_THRESHOLD)[0]

    return int(below[0]) if below.size else 1


def delay_embed(signal: np.ndarray, tau: int, m: int) -> np.ndarray:
    """
    Takens delay embedding: [x(t), x(t-tau), ..., x(t-(m-1)tau)], shape [N, m].
    """

    x = np.asarray(signal, dtype=np.float64)
    count = len(x) - (m - 1) * tau
    if count <= 0:
        raise ValueError("Signal too short for the requested embedding")

    return np.stack([x[i * tau : i * tau + count] for i in range(m)], axis=1)


def _pairwise_upper(points: np.ndarray) -> np.ndarray:
    gram = points @ points.T
    sq = np.diag(gram)
    d2 = np.clip(sq[:, None] + sq[None, :] - 2.0 * gram, 0.0, None)
    iu = np.triu_indices(len(points), k=1)

    return np.sqrt(d2[iu])


def correlation_dimension(
    points: np.ndarray, subsample: int = CORR_SUBSAMPLE
) -> dict:
    """
    Grassberger-Procaccia correlation dimension of a point cloud.

    C(r) is the fraction of point pairs closer than r; D2 is the slope of
    log C(r) vs log r in a central scaling window. Returns D2, the fit R2 and
    the log-log curve (for the scaling plot). Points are strided to
    'subsample' first (the pairwise step is O(N^2)).
    """

    x = np.asarray(points, dtype=np.float64)
    if len(x) > subsample:
        x = x[np.linspace(0, len(x) - 1, subsample).astype(np.int64)]
    dists = _pairwise_upper(x)
    dists = dists[dists > 0]
    if dists.size < 3:
        empty = np.array([])
        return {
            "d2": 0.0,
            "r2": 0.0,
            "log_r": empty,
            "log_c": empty,
            "region": np.array([], dtype=bool),
        }
    rmin, rmax = np.quantile(dists, CORR_QUANTILES)
    radii = np.logspace(np.log10(rmin), np.log10(rmax), CORR_RADII)
    order = np.sort(dists)
    counts = np.searchsorted(order, radii, side="left") / order.size
    ok = counts > 0
    log_r, log_c = np.log(radii[ok]), np.log(counts[ok])
    lo, hi = np.quantile(log_c, SCALING_WINDOW)
    region = (log_c >= lo) & (log_c <= hi)
    if region.sum() < 3:
        region = np.ones_like(log_c, dtype=bool)
    slope, intercept = np.polyfit(log_r[region], log_c[region], 1)
    pred = slope * log_r[region] + intercept
    ss_res = np.sum((log_c[region] - pred) ** 2)
    ss_tot = np.sum((log_c[region] - log_c[region].mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {
        "d2": float(slope),
        "r2": float(r2),
        "log_r": log_r,
        "log_c": log_c,
        "region": region,
    }


def d2_vs_embedding(
    signal: np.ndarray, tau: int, dims: tuple[int, ...] = EMBED_DIMS
) -> tuple[np.ndarray, np.ndarray]:
    """
    Correlation dimension D2 at each embedding dimension m; the plateau is the
    attractor-dimension estimate.
    """

    ms, d2s = [], []

    for m in dims:
        try:
            embedded = delay_embed(signal, tau, m)
        except ValueError:
            break
        ms.append(m)
        d2s.append(correlation_dimension(embedded)["d2"])

    return np.array(ms), np.array(d2s)


def higuchi_fd(signal: np.ndarray, k_max: int = HIGUCHI_KMAX) -> float:
    """
    Higuchi fractal dimension of the raw waveform (no embedding).

    Slope of log(curve length) vs log(1/k) over scales k. Near 1 for a smooth
    curve, near 2 for a space-filling (noise-like) one.
    """

    x = np.asarray(signal, dtype=np.float64)
    n = len(x)
    ks = [k for k in range(1, k_max + 1) if n > k + 1]
    lengths = []

    for k in ks:
        per_start = []

        for start in range(k):
            idx = np.arange(start, n, k)
            if len(idx) < 2:
                continue
            # Higuchi normalisation: (N-1) / (points * k) with a further 1/k,
            # so a space-filling (noise) curve scores ~2 and a smooth one ~1.
            norm = (n - 1) / ((len(idx) - 1) * k * k)
            per_start.append(np.sum(np.abs(np.diff(x[idx]))) * norm)
        lengths.append(np.mean(per_start))
    log_k = np.log(1.0 / np.array(ks, dtype=np.float64))
    log_l = np.log(np.array(lengths))

    return float(np.polyfit(log_k, log_l, 1)[0])
