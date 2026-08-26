"""
Entropy and compressibility of a 1-D sequence.

Permutation entropy (ordinal-pattern predictability), sample entropy
(regularity), and Lempel-Ziv complexity (compressibility). All robust,
scale-free predictability measures well suited to noisy real time series.
"""

from __future__ import annotations

import math

import numpy as np

PERM_ORDER = 4
PERM_DELAY = 1
SAMPEN_M = 2
SAMPEN_R = 0.2  # tolerance as a multiple of the signal std
SAMPEN_MAX_POINTS = 2000  # sample entropy is O(N^2); subsample above this
LZ_SYMBOLS = 2  # binary symbolisation at the median
# Lempel-Ziv is superlinear and its normalised value is length-dependent, so
# it runs at a fixed strided length for speed and cross-signal comparability.
LZ_MAX_POINTS = 20000


def permutation_entropy(
    signal: np.ndarray, order: int = PERM_ORDER, delay: int = PERM_DELAY
) -> float:
    """
    Bandt-Pompe permutation entropy in [0, 1].

    Encodes each length-'order' window by the rank order of its samples and
    takes the entropy of the ordinal-pattern distribution, normalised by
    log(order!). Near 1 for unpredictable signals, near 0 for monotone ones.
    """

    x = np.asarray(signal, dtype=np.float64)
    count = len(x) - (order - 1) * delay
    if count <= 1:
        return 0.0
    columns = [x[i * delay : i * delay + count] for i in range(order)]
    patterns = np.argsort(np.stack(columns, axis=1), axis=1)
    _, counts = np.unique(patterns, axis=0, return_counts=True)
    p = counts / counts.sum()

    return float(-np.sum(p * np.log(p)) / np.log(math.factorial(order)))


def sample_entropy(
    signal: np.ndarray,
    m: int = SAMPEN_M,
    r: float = SAMPEN_R,
    max_points: int = SAMPEN_MAX_POINTS,
) -> float:
    """
    Sample entropy: the negative log likelihood that sequences close for m
    samples stay close for m+1.

    Higher means less regular. Tolerance is 'r' times the std; the signal is
    strided down to 'max_points' first, since the estimator is O(N^2).
    """

    x = np.asarray(signal, dtype=np.float64)
    if len(x) > max_points:
        x = x[np.linspace(0, len(x) - 1, max_points).astype(np.int64)]
    n = len(x)
    tol = r * np.std(x)
    if tol == 0 or n <= m + 1:
        return 0.0

    def matches(length: int) -> int:
        templates = np.stack([x[i : i + length] for i in range(n - length + 1)])
        total = 0

        for i in range(len(templates)):
            dist = np.max(np.abs(templates - templates[i]), axis=1)
            total += int(np.sum(dist <= tol)) - 1

        return total

    b, a = matches(m), matches(m + 1)
    if b == 0 or a == 0:
        return 0.0

    return float(-np.log(a / b))


def _lz76(symbols: str) -> int:
    """
    Lempel-Ziv (1976) production count of a symbol string.
    """

    n = len(symbols)
    i, c, length, k, k_max = 0, 1, 1, 1, 1

    while length + k <= n:
        if symbols[i + k - 1] == symbols[length + k - 1]:
            k += 1
        else:
            k_max = max(k_max, k)
            i += 1
            if i == length:
                c += 1
                length += k_max
                i, k, k_max = 0, 1, 1
            else:
                k = 1
    if k != 1:
        c += 1

    return c


def lempel_ziv(
    signal: np.ndarray,
    symbols: int = LZ_SYMBOLS,
    max_points: int = LZ_MAX_POINTS,
) -> float:
    """
    Normalised Lempel-Ziv complexity in roughly [0, 1].

    The signal is symbolised into 'symbols' equal-count levels; the LZ76
    production count is normalised by n / log_symbols(n), so a random sequence
    approaches 1 and a constant one approaches 0. Strided to 'max_points'
    first, since the value is length-dependent and the count superlinear.
    """

    x = np.asarray(signal, dtype=np.float64)
    if len(x) > max_points:
        x = x[np.linspace(0, len(x) - 1, max_points).astype(np.int64)]
    n = len(x)
    if n < 2:
        return 0.0
    edges = np.quantile(x, np.linspace(0.0, 1.0, symbols + 1)[1:-1])
    coded = "".join(str(v) for v in np.digitize(x, edges))
    norm = n / (math.log(n, symbols))

    return float(_lz76(coded) / norm)
