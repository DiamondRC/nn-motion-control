"""
Frequency-domain characterisation.

A Welch-style power spectral density and the quantities read off it: spectral
entropy and bandwidth (how the energy spreads), the 1/f slope and resonance
peaks (spectral shape), the input-output frequency response (Bode gain, phase,
coherence), and the high-frequency noise floor. All pure numpy.
"""

from __future__ import annotations

import numpy as np

# Welch segmenting: segment length (samples) and overlap fraction.
WELCH_SEGMENT = 4096
WELCH_OVERLAP = 0.5
# Cumulative-power fraction defining the effective bandwidth.
BANDWIDTH_FRACTION = 0.9
# 1/f slope is fit over this central fraction of the band (skip DC and the
# noise-floor tail).
SLOPE_BAND = (0.02, 0.4)
# Resonance peak picking: a local maximum counts when it exceeds this multiple
# of the local median PSD; at most this many are returned.
PEAK_PROMINENCE = 8.0
PEAK_HALF_WINDOW = 24
MAX_PEAKS = 6
# Top fraction of the band treated as the broadband noise floor.
NOISE_BAND_FRACTION = 0.8
_MIN_SEGMENT = 16


def psd(
    signal: np.ndarray,
    sample_rate: float,
    segment: int = WELCH_SEGMENT,
    overlap: float = WELCH_OVERLAP,
) -> tuple[np.ndarray, np.ndarray]:
    """
    One-sided Welch power spectral density: (freqs Hz, power units^2/Hz).

    Splits the signal into overlapping Hann-windowed segments and averages the
    periodograms; each segment is mean-removed so DC does not dominate.
    """

    x = np.asarray(signal, dtype=np.float64)
    seg = min(segment, len(x))
    if seg < _MIN_SEGMENT:
        raise ValueError("Signal too short for a PSD")
    step = max(1, int(seg * (1.0 - overlap)))
    window = np.hanning(seg)
    win_power = np.sum(window**2)
    starts = list(range(0, len(x) - seg + 1, step)) or [0]
    acc = np.zeros(seg // 2 + 1)

    for s in starts:
        block = x[s : s + seg]
        spec = np.fft.rfft((block - block.mean()) * window)
        acc += np.abs(spec) ** 2
    power = acc / (len(starts) * win_power * sample_rate)
    power[1:-1] *= 2.0  # one-sided: fold the negative frequencies in
    freqs = np.fft.rfftfreq(seg, d=1.0 / sample_rate)

    return freqs, power


def spectral_entropy(signal: np.ndarray, sample_rate: float, **kw) -> float:
    """
    Shannon entropy of the normalised PSD, in [0, 1].

    A flat (broadband) spectrum approaches 1; a pure tone approaches 0.
    """

    _, power = psd(signal, sample_rate, **kw)
    p = power[1:]  # drop DC
    total = p.sum()
    if total <= 0:
        return 0.0
    p = p / total
    nz = p[p > 0]

    return float(-np.sum(nz * np.log(nz)) / np.log(len(p)))


def bandwidth(
    signal: np.ndarray,
    sample_rate: float,
    fraction: float = BANDWIDTH_FRACTION,
    **kw,
) -> float:
    """
    Frequency (Hz) below which 'fraction' of the a-c power lies.
    """

    freqs, power = psd(signal, sample_rate, **kw)
    f, p = freqs[1:], power[1:]
    cumulative = np.cumsum(p)
    if cumulative[-1] <= 0:
        return 0.0
    cumulative /= cumulative[-1]

    return float(f[min(int(np.searchsorted(cumulative, fraction)), len(f) - 1)])


def one_over_f_slope(
    signal: np.ndarray,
    sample_rate: float,
    band: tuple[float, float] = SLOPE_BAND,
    **kw,
) -> float:
    """
    Slope of log10(power) vs log10(freq) over a central band (the 1/f^-slope
    spectral colour). Returns 0 when the band is too sparse to fit.
    """

    freqs, power = psd(signal, sample_rate, **kw)
    nyquist = freqs[-1]
    keep = (
        (freqs > band[0] * nyquist) & (freqs <= band[1] * nyquist) & (power > 0)
    )
    if keep.sum() < 3:
        return 0.0

    return float(np.polyfit(np.log10(freqs[keep]), np.log10(power[keep]), 1)[0])


def resonance_peaks(
    signal: np.ndarray,
    sample_rate: float,
    prominence: float = PEAK_PROMINENCE,
    max_peaks: int = MAX_PEAKS,
    **kw,
) -> list[tuple[float, float]]:
    """
    Dominant spectral peaks as (frequency Hz, prominence over local median).

    A local maximum is kept when it rises 'prominence' times above the median
    PSD in a window around it; the strongest 'max_peaks' are returned.
    """

    freqs, power = psd(signal, sample_rate, **kw)
    found: list[tuple[float, float]] = []

    for i in range(1, len(power) - 1):
        if power[i] <= power[i - 1] or power[i] < power[i + 1]:
            continue
        lo = max(0, i - PEAK_HALF_WINDOW)
        base = np.median(power[lo : i + PEAK_HALF_WINDOW + 1])
        if base > 0 and power[i] > prominence * base:
            found.append((float(freqs[i]), float(power[i] / base)))
    found.sort(key=lambda peak: -peak[1])

    return found[:max_peaks]


def frequency_response(
    x_in: np.ndarray,
    y_out: np.ndarray,
    sample_rate: float,
    segment: int = WELCH_SEGMENT,
    overlap: float = WELCH_OVERLAP,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Empirical transfer function input->output: (freqs, gain, phase_deg, coh).

    Welch-averaged cross-spectra give H = Sxy / Sxx (gain, phase) and the
    magnitude-squared coherence Sxy^2 / (Sxx Syy) in [0, 1].
    """

    x = np.asarray(x_in, dtype=np.float64)
    y = np.asarray(y_out, dtype=np.float64)
    seg = min(segment, len(x))
    if seg < _MIN_SEGMENT:
        raise ValueError("Signals too short for a frequency response")
    step = max(1, int(seg * (1.0 - overlap)))
    window = np.hanning(seg)
    starts = list(range(0, len(x) - seg + 1, step)) or [0]
    s_xx = np.zeros(seg // 2 + 1)
    s_yy = np.zeros(seg // 2 + 1)
    s_xy = np.zeros(seg // 2 + 1, dtype=complex)

    for s in starts:
        xb = (x[s : s + seg] - x[s : s + seg].mean()) * window
        yb = (y[s : s + seg] - y[s : s + seg].mean()) * window
        fx, fy = np.fft.rfft(xb), np.fft.rfft(yb)
        s_xx += np.abs(fx) ** 2
        s_yy += np.abs(fy) ** 2
        s_xy += fy * np.conj(fx)
    freqs = np.fft.rfftfreq(seg, d=1.0 / sample_rate)
    safe_xx = np.where(s_xx > 0, s_xx, 1.0)
    resp = s_xy / safe_xx
    denom = np.where(s_xx * s_yy > 0, s_xx * s_yy, 1.0)
    coherence_curve = np.abs(s_xy) ** 2 / denom

    return freqs, np.abs(resp), np.degrees(np.angle(resp)), coherence_curve


def mean_coherence(
    a: np.ndarray, b: np.ndarray, sample_rate: float, **kw
) -> float:
    """
    Band-averaged magnitude-squared coherence between two signals, in [0, 1].
    """

    _, _, _, coherence_curve = frequency_response(a, b, sample_rate, **kw)

    return float(coherence_curve[1:].mean())


def noise_floor(
    signal: np.ndarray,
    sample_rate: float,
    band_fraction: float = NOISE_BAND_FRACTION,
    **kw,
) -> dict[str, float]:
    """
    High-frequency noise floor and signal-to-noise ratio.

    'floor' is the median PSD above 'band_fraction' of the band; 'snr' is the
    total a-c power over the power that floor would contribute across the band.
    """

    freqs, power = psd(signal, sample_rate, **kw)
    band = power[freqs > band_fraction * freqs[-1]]
    floor = float(np.median(band)) if band.size else 0.0
    total = float(power[1:].sum())
    floor_power = floor * (len(power) - 1)
    snr = total / floor_power if floor_power > 0 else float("inf")

    return {"floor": floor, "snr": snr}
