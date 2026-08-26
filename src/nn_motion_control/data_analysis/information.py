"""
Information flow between command and motion.

Mutual information (shared information) and transfer entropy (directed,
command -> position) via histogram estimates. Pure numpy.
"""

from __future__ import annotations

import numpy as np

MI_BINS = 16
TE_BINS = 8
TE_LAG = 1


def _discretise(v: np.ndarray, bins: int) -> np.ndarray:
    edges = np.quantile(v, np.linspace(0.0, 1.0, bins + 1)[1:-1])

    return np.clip(np.digitize(v, edges), 0, bins - 1)


def mutual_information(
    a: np.ndarray, b: np.ndarray, bins: int = MI_BINS
) -> float:
    """
    Mutual information between two signals (nats), from a 2-D histogram.
    """

    counts, _, _ = np.histogram2d(
        np.asarray(a, dtype=np.float64),
        np.asarray(b, dtype=np.float64),
        bins=bins,
    )
    joint = counts / counts.sum()
    p_a = joint.sum(axis=1)
    p_b = joint.sum(axis=0)
    outer = p_a[:, None] * p_b[None, :]
    mask = (joint > 0) & (outer > 0)

    return float(np.sum(joint[mask] * np.log(joint[mask] / outer[mask])))


def transfer_entropy(
    source: np.ndarray,
    target: np.ndarray,
    bins: int = TE_BINS,
    lag: int = TE_LAG,
) -> float:
    """
    Transfer entropy source -> target (nats).

    The information the present source adds about the next target beyond the
    present target alone: the conditional-mutual-information form
    I(future ; source | present). Each variable is discretised into 'bins'
    equal-count levels.
    """

    tgt = np.asarray(target, dtype=np.float64)
    src = np.asarray(source, dtype=np.float64)
    future = _discretise(tgt[lag:], bins)
    present = _discretise(tgt[:-lag], bins)
    driver = _discretise(src[:-lag], bins)

    joint = np.zeros((bins, bins, bins))
    np.add.at(joint, (future, present, driver), 1.0)
    joint /= joint.sum()
    p_present = joint.sum(axis=(0, 2))  # p(t)
    p_fp = joint.sum(axis=2)  # p(t+, t)
    p_ps = joint.sum(axis=0)  # p(t, s)
    numer = joint * p_present[None, :, None]
    denom = p_fp[:, :, None] * p_ps[None, :, :]
    mask = (joint > 0) & (denom > 0)

    return float(np.sum(joint[mask] * np.log(numer[mask] / denom[mask])))
