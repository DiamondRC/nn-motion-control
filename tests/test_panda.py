"""PandA capture reader tests: units, open-loop guards, exact and filtered
resampling.
"""

import json

import h5py
import numpy as np
import pytest

from nn_motion_control.data import panda as m
from nn_motion_control.data._hdf5 import as_dataset

CAPTURE_HZ = 40_000.0  # matches the conftest capture writer
DRIVE_HZ = 20_000.0
AXES = ("x", "y", "z")
RES_NM = 0.256
KAISER_SLACK_DB = 1.0
# Lag of a two-sample hold's centre behind its update, in drive samples.
ZOH_LAG_DRIVE_SAMPLES = 0.25
AA = m.AntiAlias(passband_hz=4000, stopband_hz=5000, attenuation_db=80)


def read(path, pgens):
    return m.read_capture(path, pgens, AXES, RES_NM, CAPTURE_HZ)


def test_raw_counts_become_nm(tmp_path, write_capture):
    counts = np.tile(np.arange(4000.0), (3, 1))
    path = write_capture(tmp_path / "c.h5", {}, counts=counts)
    cap = read(path, {})
    assert np.allclose(cap.position, counts[:, m.STALE_SAMPLES :] * RES_NM)


def test_prescaled_capture_is_not_scaled_twice(tmp_path, write_capture):
    counts = np.tile(np.arange(4000.0), (3, 1))
    path = write_capture(tmp_path / "c.h5", {}, counts=counts, scale=RES_NM)
    cap = read(path, {})
    assert np.allclose(cap.position, counts[:, m.STALE_SAMPLES :] * RES_NM)


def test_off_grid_counts_are_rejected(tmp_path, write_capture):
    path = write_capture(tmp_path / "c.h5", {}, counts=np.full((3, 4000), 0.3))
    with pytest.raises(ValueError, match="count grid"):
        read(path, {})


def test_pgen_drives_its_mapped_axis(tmp_path, write_capture):
    table = np.arange(2000.0)
    path = write_capture(tmp_path / "c.h5", {"PGEN1": table})
    cap = read(path, {"y": "PGEN1"})
    assert np.any(cap.command[1]) and not np.any(cap.command[[0, 2]])


def test_closed_loop_capture_is_rejected(tmp_path, write_capture):
    path = write_capture(tmp_path / "c.h5", {}, feedback=True)
    with pytest.raises(ValueError, match="not an open-loop capture"):
        read(path, {})


def test_unmapped_active_pgen_is_rejected(tmp_path, write_capture):
    path = write_capture(tmp_path / "c.h5", {"PGEN2": np.ones(2000)})
    with pytest.raises(ValueError, match="not mapped"):
        read(path, {"x": "PGEN1"})


def test_timestamp_gap_is_rejected(tmp_path, write_capture):
    path = write_capture(tmp_path / "c.h5", {}, gap_at=1000)
    with pytest.raises(ValueError, match="irregular"):
        read(path, {})


@pytest.mark.parametrize("phase", [0, 1])
def test_drive_rate_resampling_recovers_the_table(
    tmp_path, phase, write_capture
):
    table = np.random.default_rng(0).integers(-500, 500, 2000).astype(float)
    path = write_capture(tmp_path / "c.h5", {"PGEN1": table}, phase=phase)
    pos, cmd = m.resample(read(path, {"x": "PGEN1"}), DRIVE_HZ, DRIVE_HZ, None)
    assert pos.shape == cmd.shape
    # Every picked sample is a whole table entry, in order.
    first = int(np.flatnonzero(table == cmd[0, 5])[0]) - 5
    assert np.array_equal(cmd[0, 5:900], table[first + 5 : first + 900])


def test_mixed_update_phase_is_rejected(tmp_path, write_capture):
    path = write_capture(tmp_path / "c.h5", {"PGEN1": np.arange(2000.0)})
    with h5py.File(path, "a") as f:
        # A one-sample change off the drive phase.
        as_dataset(f, "PGEN1.OUT.Value")[100] = -1.0
    with pytest.raises(ValueError, match="phases"):
        m.resample(read(path, {"x": "PGEN1"}), DRIVE_HZ, DRIVE_HZ, None)


def test_filtered_resampling_keeps_the_band_and_drops_the_alias(
    tmp_path, write_capture
):
    n = 40_000
    t = np.arange(n // 2) / DRIVE_HZ
    low = 400 * np.sin(2 * np.pi * 100 * t)
    high = 400 * np.sin(2 * np.pi * 7000 * t)
    path = write_capture(tmp_path / "c.h5", {"PGEN1": low + high}, n=n)
    _, cmd = m.resample(read(path, {"x": "PGEN1"}), 10_000.0, DRIVE_HZ, AA)
    # The 100 Hz tone keeps unit gain, and the 7 kHz tone, above the 5 kHz
    # Nyquist, leaves no residual to alias into the band.
    amp, _, resid = _fit_tone(cmd[0], 100.0, 10_000.0)
    assert np.isclose(amp, 400, rtol=1e-3)
    assert resid < 0.5


def _fit_tone(x, freq, rate):
    """
    Amplitude, timing and residual of a sinusoid fit to 'x' at 'rate'.

    The timing is where the samples sit on the tone, in drive samples
    measured from the first output sample's grid point.
    """
    t = np.arange(len(x)) / rate
    basis = np.stack(
        [np.sin(2 * np.pi * freq * t), np.cos(2 * np.pi * freq * t)], axis=1
    )
    (a, b), *_ = np.linalg.lstsq(basis, x, rcond=None)
    lead = np.arctan2(b, a) / (2 * np.pi * freq) * DRIVE_HZ

    return np.hypot(a, b), lead, np.std(x - basis @ (a, b))


@pytest.mark.parametrize("phase", [0, 1])
def test_resampling_lands_on_command_updates(tmp_path, write_capture, phase):
    n = 40_000
    k = np.arange(n // 2)
    table = 400 * np.sin(2 * np.pi * 100 * k / DRIVE_HZ)
    path = write_capture(tmp_path / "c.h5", {"PGEN1": table}, n=n, phase=phase)
    cap = read(path, {"x": "PGEN1"})
    _, cmd20 = m.resample(cap, DRIVE_HZ, DRIVE_HZ, None)
    _, cmd10 = m.resample(cap, 10_000.0, DRIVE_HZ, AA)
    # At the drive rate each sample is a table entry, a whole number of
    # drive samples along the tone.
    _, lead20, _ = _fit_tone(cmd20[0], 100.0, DRIVE_HZ)
    assert abs(lead20 - round(lead20)) < 1e-3
    # Below it the zero-phase filter sees each value held for two capture
    # samples, so its centre lags the update by half a capture sample, a
    # quarter drive sample. A causal filter or an off-by-one pick shifts
    # this by a whole capture sample or more.
    _, lead10, _ = _fit_tone(cmd10[0], 100.0, 10_000.0)
    assert abs((lead10 % 1) - (1 - ZOH_LAG_DRIVE_SAMPLES)) < 1e-2


def test_frozen_encoder_is_rejected(tmp_path, write_capture):
    counts = np.tile(np.arange(4000) % 2, (3, 1)).astype(float)
    counts[2, 1000:] = 7.0  # z stops updating a quarter of the way in
    path = write_capture(tmp_path / "c.h5", {}, counts=counts)
    with pytest.raises(m.LostEncoderError, match="INENC3"):
        read(path, {})


def test_prescaled_value_off_the_grid_is_rejected(tmp_path, write_capture):
    # Large nm values must still sit on the grid, a relative tolerance
    # would let them through.
    counts = np.tile(2e6 + np.arange(4000) % 2, (3, 1))
    counts[0, 50] += 0.4
    path = write_capture(tmp_path / "c.h5", {}, counts=counts, scale=RES_NM)
    with pytest.raises(ValueError, match="count grid"):
        read(path, {})


def test_longest_hold():
    assert m.longest_hold(np.array([1.0, 1, 2, 2, 2, 3])) == 3
    assert m.longest_hold(np.array([1.0, 2, 3])) == 1
    assert m.longest_hold(np.array([5.0, 5, 5])) == 3


def test_command_and_position_are_filtered_alike(tmp_path, write_capture):
    table = np.random.default_rng(1).integers(-300, 300, 4000).astype(float)
    held = np.zeros(8000)
    held[1:] = np.repeat(table, 2)[:-1]
    counts = np.tile(held, (3, 1))
    path = write_capture(
        tmp_path / "c.h5", {"PGEN1": table}, counts=counts, n=8000
    )
    pos, cmd = m.resample(read(path, {"x": "PGEN1"}), 10_000.0, DRIVE_HZ, AA)
    # A position that equals the command stays equal after resampling, so
    # the filter adds no lag between input and response.
    assert np.allclose(pos[0] / RES_NM, cmd[0], atol=1e-6)


def test_resampling_below_the_drive_needs_a_filter(tmp_path, write_capture):
    path = write_capture(tmp_path / "c.h5", {"PGEN1": np.arange(2000.0)})
    with pytest.raises(ValueError, match="anti-alias"):
        m.resample(read(path, {"x": "PGEN1"}), 10_000.0, DRIVE_HZ, None)


def test_anti_alias_taps_have_unit_dc_gain():
    h = AA.taps(CAPTURE_HZ)
    assert len(h) % 2 == 1
    assert np.isclose(h.sum(), 1.0)
    freqs = np.fft.rfftfreq(1 << 16, 1 / CAPTURE_HZ)
    gain = np.abs(np.fft.rfft(h, 1 << 16))
    assert np.all(np.abs(gain[freqs < 4000] - 1) < 1e-3)
    # Kaiser's order estimate meets the attenuation to within a decibel.
    floor_db = AA.attenuation_db - KAISER_SLACK_DB
    assert np.all(gain[freqs > 5000] < 10 ** (-floor_db / 20))


def write_feedback(path, position, command, rate, key, scale=None):
    """
    A synthetic feedback capture: 'position' rows become INENC1.., the
    command goes to 'key', timestamps are regular at 'rate'.
    """
    n = position.shape[1]
    ts = np.r_[99.0, 0.0, (np.arange(n - 2) + 0.5) / rate]
    with h5py.File(path, "w") as f:
        f.create_dataset("PCAP.TS_START.Value", data=ts)

        for i, row in enumerate(position):
            f.create_dataset(f"INENC{i + 1}.VAL.Value", data=row)
        f.create_dataset(key, data=command)
        if scale is not None:
            f.attrs["janus_capture_scales"] = json.dumps(
                {f"INENC{i + 1}.VAL": scale for i in range(len(position))}
            )

    return path


def _held(values, period):
    """'values' held for 'period' samples each, as a capture records it."""
    return np.repeat(values, period)


BRETT = "BRETT_PID.REAL_OUTPUT.Value"
PANDA_PID = "PID.REAL_OUTPUT.Value"


def test_closed_loop_units_and_axes_are_read_off_the_data(tmp_path):
    rng = np.random.default_rng(0)
    n = 4000
    # Brett-style: one interferometer, already nm (on the 0.256 grid).
    nm = rng.integers(-1000, 1000, n) * RES_NM
    path = write_feedback(
        tmp_path / "b.h5",
        nm[None],
        _held(rng.normal(size=n // 2), 2),
        2e4,
        BRETT,
    )
    cap = m.read_closed_loop(path, {"x": BRETT}, AXES, RES_NM)
    assert cap.measured == (True, False, False)
    assert np.allclose(cap.position[0], nm[m.STALE_SAMPLES :])
    assert not np.any(cap.position[1:])
    assert np.isclose(cap.capture_rate_hz, 2e4)
    assert np.isclose(cap.drive_rate_hz, 1e4)  # each output held 2 samples
    # Raw counts on three axes are scaled to nm.
    counts = rng.integers(-1000, 1000, (3, n)).astype(float)
    path = write_feedback(
        tmp_path / "c.h5",
        counts,
        _held(rng.normal(size=n // 2), 2),
        4e4,
        PANDA_PID,
    )
    cap = m.read_closed_loop(path, {"x": PANDA_PID}, AXES, RES_NM)
    assert cap.measured == (True, True, True)
    assert np.allclose(cap.position, counts[:, m.STALE_SAMPLES :] * RES_NM)
    assert m.feedback_output(path) == PANDA_PID


def _driven_capture(tmp_path, rest_counts, gain, lead=0):
    """
    x held at a static spring equilibrium under a slowly varying command
    from the first sample (or after 'lead' zero-command drive samples).
    """
    n = 8000
    u = 200 * np.sin(np.arange(n // 2) / 300.0) + 300
    u[:lead] = 0.0
    held = _held(u, 2)
    counts = np.zeros((3, n))
    counts[0] = np.round(rest_counts + gain * held / RES_NM)
    counts[1:] = np.arange(n) % 2  # live, at rest
    path = write_feedback(tmp_path / "d.h5", counts, held, 4e4, PANDA_PID)

    return m.read_closed_loop(path, {"x": PANDA_PID}, AXES, RES_NM), held


def test_rest_frame_finds_rest_from_the_static_gain(tmp_path):
    gain = 1000.0  # nm per count
    cap, held = _driven_capture(tmp_path, rest_counts=5e5, gain=gain)
    frame = m.DatasetFrame(
        rate_hz=2e4,
        reference=m.REST_REFERENCE,
        rest_seconds=0.002,
        static_gain=np.array([gain, 900.0, 960.0]),
    )
    pos, cmd = frame.place(cap)
    # Rest removed: what is left is the spring's static response.
    assert np.allclose(pos[0], gain * cmd[0], atol=RES_NM)
    no_gain = m.DatasetFrame(
        rate_hz=2e4, reference=m.REST_REFERENCE, rest_seconds=0.002
    )
    with pytest.raises(ValueError, match="static gain"):
        no_gain.place(cap)


def test_rest_frame_prefers_a_zero_command_window(tmp_path):
    cap, _ = _driven_capture(tmp_path, rest_counts=5e5, gain=1000.0, lead=200)
    frame = m.DatasetFrame(
        rate_hz=2e4, reference=m.REST_REFERENCE, rest_seconds=0.002
    )
    pos, _ = frame.place(cap)
    assert abs(pos[0, :20].mean()) < RES_NM


def test_mean_frame_centres_and_picks_without_filtering(tmp_path):
    cap, _ = _driven_capture(tmp_path, rest_counts=5e5, gain=1000.0)
    # An older dataset: centred per recording, sampled by picking.
    frame = m.DatasetFrame(rate_hz=1e4, reference=m.MEAN_REFERENCE)
    pos, cmd = frame.place(cap)
    # 40 kHz to 10 kHz keeps every fourth sample.
    assert abs(pos.shape[1] - cap.position.shape[1] / 4) <= 1
    assert np.allclose(pos.mean(axis=1), 0.0, atol=1e-6)
    # Picked, not filtered: every command sample is a recorded value.
    assert np.isin(cmd[0], cap.command[0]).all()


def test_dataset_frame_reads_a_build_and_needs_a_rate_otherwise(tmp_path):
    v3 = tmp_path / "v3.h5"
    with h5py.File(v3, "w") as f:
        f.attrs.update(
            {
                "position_reference": m.REST_REFERENCE,
                "rate_hz": 1e4,
                "rest_seconds": 0.02,
                "anti_alias_hz": np.array([4000.0, 5000.0]),
                "anti_alias_db": 80.0,
                "static_gain_nm_per_count": np.array([1.0, 2.0, 3.0]),
            }
        )
    frame = m.DatasetFrame.of(v3, rate_hz=None)
    assert frame.anti_alias == AA and frame.rate_hz == 1e4
    v2 = tmp_path / "v2.h5"
    with h5py.File(v2, "w") as f:
        f.attrs["schema_version"] = 2
    with pytest.raises(ValueError, match="data_rate_hz"):
        m.DatasetFrame.of(v2, rate_hz=None)
    assert m.DatasetFrame.of(v2, rate_hz=1e4).reference == m.MEAN_REFERENCE


def test_update_period_survives_a_jittered_change():
    # A command held four samples per update, read off its changes even
    # where the value repeats across updates, and despite one stray gap.
    held = np.repeat(np.arange(1000) % 7, 4)[None].astype(float)
    assert m.update_period(held) == 4
    held[0, 1001] = 99.0
    assert m.update_period(held) == 4
