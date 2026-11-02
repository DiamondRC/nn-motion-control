"""Plant FRF check: period and span detection, line response, scoring."""

import numpy as np
import pytest

from nn_motion_control.eval import frf as m

RATE = 1000.0
PERIOD = 200  # samples, 0.2 s
GAIN = 3.0
SETTLE_RTOL = 2e-3


def _multisine(n_periods, rng, offset=0.0, ramp=0):
    """A random-phase multisine, optional ramp onto and off an offset."""
    k = np.arange(1, 40)
    phases = rng.uniform(0, 2 * np.pi, len(k))
    t = np.arange(PERIOD * n_periods)
    u = np.sin(2 * np.pi * np.outer(t, k) / PERIOD + phases).sum(axis=1)
    if ramp:
        env = np.r_[
            np.linspace(0, 1, ramp), np.ones(len(u)), np.linspace(1, 0, ramp)
        ]
        u = np.r_[np.zeros(ramp), u, np.zeros(ramp)] + offset * env

    return u


def _rows(u):
    """
    Dataset rows of a stage that follows the command instantly with gain
    GAIN: row i holds position GAIN * u[i] beside the command one step
    earlier, u[i - 1], as the build stores it.
    """
    dac_row = np.r_[0.0, u[:-1]]
    pos = np.zeros((3, len(u)))
    pos[0] = GAIN * u
    dac = np.zeros((3, len(u)))
    dac[0] = dac_row

    return pos, dac


def test_period_and_span_survive_an_offset_ramp():
    rng = np.random.default_rng(0)
    u = _multisine(5, rng, offset=50.0, ramp=3 * PERIOD)
    assert m.find_period(u, RATE) == PERIOD
    start, whole = m.periodic_span(u, PERIOD)
    # The repeating stretch is the plateau between the ramps.
    assert 3 * PERIOD - 1 <= start <= 3 * PERIOD + 1
    assert whole == 5


def test_line_response_recovers_a_known_gain_and_references_the_command():
    rng = np.random.default_rng(1)
    u = _multisine(6, rng)
    pos, dac = _rows(u)
    freqs, h, snr = m.line_response(pos, dac, 0, PERIOD, RATE)
    assert len(freqs) == 39
    # The one-step row shift is undone: a static gain reads as a real H.
    assert np.allclose(h[0], GAIN, rtol=1e-6)
    assert np.all(snr[0] > m.MIN_SNR_DB)
    # The undriven axes read no response.
    assert np.allclose(h[1:], 0.0)


def test_too_few_periods_raise():
    rng = np.random.default_rng(2)
    u = _multisine(2, rng)
    pos, dac = _rows(u)
    with pytest.raises(ValueError, match="periods"):
        m.line_response(pos, dac, 0, PERIOD, RATE)


def test_band_scores_compare_prediction_to_measurement():
    freqs = np.array([10.0, 100.0, 1000.0])
    measured = np.ones((3, 3), dtype=complex)
    predicted = measured.copy()
    predicted[0, 1] = 2.0  # one line 6 dB high on the driven axis
    snr = np.full((3, 3), 40.0)
    sets = [
        m.LineSet(
            freqs, measured, predicted, snr, np.array([True, False, False])
        )
    ]
    diag = {s.band_hz: s for s in m.band_scores(sets, diagonal=True)}
    assert diag[(30, 300)].gain_ratio == pytest.approx(2.0)
    assert diag[(30, 300)].within_db == 0.0
    assert diag[(5, 30)].within_db == 1.0
    cross = m.band_scores(sets, diagonal=False)
    assert all(s.gain_ratio == pytest.approx(1.0) for s in cross)


def test_line_response_recovers_a_resonant_delayed_stage_through_drift():
    # A damped second-order stage with a two-sample delay, plus an offset
    # and a slow drift on the position, driven like the generator: a
    # smoothly faded-in settling period, then repeating periods.
    rng = np.random.default_rng(4)
    wave = _multisine(1, rng)
    fade = 0.5 - 0.5 * np.cos(np.pi * np.arange(PERIOD) / PERIOD)
    u = np.r_[fade * wave, np.tile(wave, 4)]
    a1, a2, b, delay = 1.6, -0.81, 0.3, 2
    y = np.zeros(len(u))

    for t in range(len(u)):
        drive = u[t - delay] if t >= delay else 0.0
        y[t] = b * drive + a1 * y[t - 1] + a2 * y[t - 2] if t >= 2 else 0.0
    pos = np.zeros((3, len(u)))
    pos[0] = y + 7.0 + 1e-3 * np.arange(len(u))
    dac = np.zeros((3, len(u)))
    dac[0] = np.r_[0.0, u[:-1]]  # rows hold the command one step behind
    freqs, h, _ = m.line_response(pos, dac, 0, PERIOD, RATE)
    z = np.exp(-2j * np.pi * freqs / RATE)
    want = b * z**delay / (1 - a1 * z - a2 * z**2)
    # The stage is still settling from the fade-in as the repeats begin,
    # which leaves a residual well under a percent.
    assert np.allclose(h[0], want, rtol=SETTLE_RTOL, atol=0)
