"""
Read open-loop PandA captures and resample them to a plant data rate.

A capture is one PandA PCAP HDF5 file: INENC1..3 carry the interferometer
position of x/y/z and PGEN1..3 the command table driven onto the amplifier
outputs (OUTENC). INENC is stored either as raw counts or, when the capture
tool applied an on-FPGA scale, already multiplied by that scale (recorded in
the 'janus_capture_scales' file attribute). Either way the reader returns
position in nm and command in DAC counts.

The command is a zero-order hold at the drive rate, captured at a higher
rate. Resampling to the drive rate picks one sample per command update, so
the command is exact and the position is the capture's own sample at that
instant. The interferometer refreshes at about 15 kHz, asynchronously to
the capture clock, so at 20 kHz some position samples repeat the previous
reading, which is the hardware's real behaviour and is kept. Resampling
below the drive rate cannot represent the command, so command and position
pass through the same zero-phase low-pass before decimation, which keeps
the input-output relation intact.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from nn_motion_control.data._hdf5 import as_dataset

# The first two samples of every capture are stale: sample 0 is the last
# sample of the previous arm and the first interval is partial.
STALE_SAMPLES = 2

# A sample interval this far from the median (as a fraction) is a gap.
_TS_TOLERANCE = 0.25

# Command-change gaps rarer than this share are jitter, not the drive.
RARE_GAP_FRACTION = 0.01

# Fewest samples a feedback capture needs to have a rate at all.
MIN_CAPTURE_SAMPLES = 3

# Controller outputs a feedback capture records, newest naming first.
FEEDBACK_OUTPUTS = ("PID.REAL_OUTPUT.Value", "BRETT_PID.REAL_OUTPUT.Value")

# Datasets that are non-zero only when a feedback loop drives the output.
_FEEDBACK_KEYS = (
    "PID.REAL_OUTPUT.Value",
    "LQR.U0.Value",
    "LQR.U1.Value",
    "LQR.U2.Value",
)

_TS_KEY = "PCAP.TS_START.Value"
_SCALES_ATTR = "janus_capture_scales"

# Position references a dataset can record ('position_reference').
REST_REFERENCE = "rest"
MEAN_REFERENCE = "mean"

# Raw INENC counts must sit on the integer grid to this tolerance.
_GRID_TOLERANCE = 1e-6

# A live interferometer reading never holds one value for longer than about
# half a millisecond, even at rest, a longer hold is a lost encoder.
FROZEN_SECONDS = 0.01


class LostEncoderError(ValueError):
    """
    An interferometer channel stopped updating during the capture.
    """


@dataclass(frozen=True)
class Capture:
    """
    One open-loop capture in physical units at its capture rate.

    'position' is [A, n] nm and 'command' [A, n] DAC counts, both float64.
    """

    path: Path
    position: np.ndarray
    command: np.ndarray
    capture_rate_hz: float


@dataclass(frozen=True)
class AntiAlias:
    """
    Kaiser-window low-pass used when resampling below the drive rate.
    """

    passband_hz: float
    stopband_hz: float
    attenuation_db: float

    def taps(self, rate_hz: float) -> np.ndarray:
        """
        Odd-length, unit-DC-gain linear-phase FIR at 'rate_hz'.

        Kaiser's order estimate is approximate: the stopband reaches the
        requested attenuation to within about a decibel.
        """

        if not 0 < self.passband_hz < self.stopband_hz < rate_hz / 2:
            raise ValueError(
                f"Anti-alias band {self.passband_hz}-{self.stopband_hz} Hz "
                f"does not fit below Nyquist of {rate_hz} Hz"
            )
        a = self.attenuation_db
        width = 2 * np.pi * (self.stopband_hz - self.passband_hz) / rate_hz
        # Kaiser's design formulas for order and window shape.
        n = int(np.ceil((a - 8.0) / (2.285 * width))) + 1
        n += 1 - n % 2
        beta = 0.1102 * (a - 8.7) if a > 50 else 0.5842 * (a - 21) ** 0.4
        cutoff = (self.passband_hz + self.stopband_hz) / rate_hz
        t = np.arange(n) - (n - 1) / 2
        h = cutoff * np.sinc(cutoff * t) * np.kaiser(n, beta)

        return h / h.sum()


def capture_scales(f: h5py.File) -> dict[str, float]:
    raw = f.attrs.get(_SCALES_ATTR)
    if raw is None:
        return {}

    return {k: float(v) for k, v in json.loads(raw).items()}


def check_timestamps(ts: np.ndarray, rate_hz: float, path: Path) -> None:
    """
    Raise when the capture has a gap or runs at an unexpected rate.
    """

    d = np.diff(ts)
    expected = 1.0 / rate_hz
    bad = np.abs(d - expected) > _TS_TOLERANCE * expected
    if bad.any():
        first = int(np.argmax(bad))
        raise ValueError(
            f"{path}: {int(bad.sum())} irregular sample interval(s), first "
            f"at sample {first + STALE_SAMPLES} ({d[first]:.3g} s, expected "
            f"{expected:.3g} s)"
        )


def read_capture(
    path: str | Path,
    pgens: Mapping[str, str],
    axes: Sequence[str],
    resolution_nm: float,
    capture_rate_hz: float,
    max_samples: int | None = None,
) -> Capture:
    """
    Read one open-loop capture as nm position and DAC-count command.

    'pgens' maps each driven axis to its PGEN block ('{"x": "PGEN1"}'),
    undriven axes get a zero command. INENC1..3 are taken as the axes in
    order. 'max_samples' caps the read after the stale samples are dropped.
    Raises on a feedback-driven capture, a non-zero PGEN that 'pgens' does
    not account for, a timestamp gap, or raw counts off the integer grid,
    and with LostEncoderError when an interferometer channel freezes.
    """

    path = Path(path)
    unknown = set(pgens) - set(axes)
    if unknown:
        raise ValueError(f"{path}: PGEN map names unknown axes {unknown}")
    stop = None if max_samples is None else STALE_SAMPLES + max_samples
    sel = slice(STALE_SAMPLES, stop)

    with h5py.File(path, "r") as f:
        for key in _FEEDBACK_KEYS:
            if key in f and np.any(as_dataset(f, key)[sel]):
                raise ValueError(
                    f"{path}: {key} is non-zero, not an open-loop capture"
                )
        check_timestamps(as_dataset(f, _TS_KEY)[sel], capture_rate_hz, path)

        position = [
            _position_nm(f, i, sel, resolution_nm, capture_rate_hz, path)
            for i in range(len(axes))
        ]

        driven = {pgen: axis for axis, pgen in pgens.items()}
        command = np.zeros((len(axes), len(position[0])))
        keys = [str(k) for k in f.keys()]

        for key in keys:
            if not (key.startswith("PGEN") and key.endswith(".OUT.Value")):
                continue
            block = key.split(".")[0]
            values = as_dataset(f, key)[sel]
            if block in driven:
                command[list(axes).index(driven[block])] = values
            elif np.any(values):
                raise ValueError(
                    f"{path}: {block} drives an output but is not mapped "
                    "to an axis"
                )
        missing = set(driven) - {k.split(".")[0] for k in keys}
        if missing:
            raise ValueError(f"{path}: mapped PGEN(s) {missing} not captured")

    return Capture(path, np.stack(position), command, capture_rate_hz)


def _position_nm(
    f: h5py.File,
    index: int,
    sel: slice,
    resolution_nm: float,
    rate_hz: float,
    path: Path,
) -> np.ndarray:
    """
    One interferometer channel (INENC 'index' + 1) in nm.

    The unit is read off the data: a 'janus_capture_scales' entry gives the
    scale applied on the FPGA, otherwise integer values are raw counts and
    values on the 'resolution_nm' grid are already nm. Anything else, or a
    channel that stops updating, raises.
    """

    key = f"INENC{index + 1}.VAL"
    values = as_dataset(f, f"{key}.Value")[sel]
    scale = capture_scales(f).get(key)
    candidates = [scale] if scale is not None else [1.0, resolution_nm]
    counts = None

    for s in candidates:
        c = values / s
        if not np.any(np.abs(c - np.round(c)) > _GRID_TOLERANCE):
            counts = c
            break
    if counts is None:
        raise ValueError(
            f"{path}: {key} is off the count grid for scale(s) {candidates}"
        )
    held = longest_hold(counts) / rate_hz
    if held > FROZEN_SECONDS:
        raise LostEncoderError(f"{path}: {key} held one value for {held:.3g} s")

    return counts * resolution_nm


@dataclass(frozen=True)
class ClosedLoopCapture(Capture):
    """
    A feedback capture: 'measured' flags the axes with an interferometer
    channel (the others read zero) and 'drive_rate_hz' is how often the
    controller output updates.
    """

    measured: tuple[bool, ...] = ()
    drive_rate_hz: float = 0.0


def read_closed_loop(
    path: str | Path,
    outputs: Mapping[str, str],
    axes: Sequence[str],
    resolution_nm: float,
) -> ClosedLoopCapture:
    """
    Read a feedback capture as nm position and the controller's command.

    'outputs' maps each driven axis to the dataset key of the command it
    received (a PID output), undriven axes get zero. The capture rate comes
    from the timestamps and the drive rate from how often the command
    changes. Only the interferometer channels present are read.
    """

    path = Path(path)
    sel = slice(STALE_SAMPLES, None)

    with h5py.File(path, "r") as f:
        ts = as_dataset(f, _TS_KEY)[sel]
        if len(ts) < MIN_CAPTURE_SAMPLES:
            raise ValueError(f"{path}: holds only {len(ts)} sample(s)")
        rate = 1.0 / float(np.median(np.diff(ts)))
        check_timestamps(ts, rate, path)
        measured = tuple(
            f"INENC{i + 1}.VAL.Value" in f for i in range(len(axes))
        )
        n = len(ts)
        position = np.zeros((len(axes), n))

        for i, present in enumerate(measured):
            if present:
                position[i] = _position_nm(f, i, sel, resolution_nm, rate, path)

        command = np.zeros((len(axes), n))

        for axis, key in outputs.items():
            if key not in f:
                raise ValueError(f"{path}: no command channel {key!r}")
            command[list(axes).index(axis)] = as_dataset(f, key)[sel]

    return ClosedLoopCapture(
        path=path,
        position=position,
        command=command,
        capture_rate_hz=rate,
        measured=measured,
        drive_rate_hz=rate / update_period(command),
    )


def feedback_output(path: str | Path) -> str:
    """
    The controller-output channel a feedback capture recorded.
    """

    with h5py.File(path, "r") as f:
        for key in FEEDBACK_OUTPUTS:
            if key in f:
                return key

    raise ValueError(f"{path}: none of {FEEDBACK_OUTPUTS} was captured")


def update_period(command: np.ndarray) -> int:
    """
    The command's update period in capture samples (1 when it never
    changes).

    A held command changes only on update instants, so every gap between
    changes is a whole number of periods and their greatest common divisor
    is the period, even when the value often holds across several updates.
    """

    changes = np.flatnonzero(np.any(np.diff(command, axis=1) != 0, axis=0))
    gaps = np.diff(changes)
    if not len(gaps):
        return 1
    # A rare jittered gap would collapse the divisor to one and hide the
    # drive rate, so gaps seen too seldom are left out (update_phase still
    # raises if the phases really are mixed).
    values, counts = np.unique(gaps, return_counts=True)
    common = values[counts >= RARE_GAP_FRACTION * len(gaps)]

    return int(np.gcd.reduce(common if len(common) else values))


def longest_hold(values: np.ndarray) -> int:
    """
    Length, in samples, of the longest run of one repeated value.
    """

    changes = np.flatnonzero(np.diff(values) != 0)
    if len(changes) == 0:
        return len(values)
    runs = np.diff(np.r_[-1, changes, len(values) - 1])

    return int(runs.max())


def update_phase(command: np.ndarray, factor: int, path: Path) -> int:
    """
    Sample offset (mod 'factor') at which the held command changes.

    Every change must share one phase, otherwise the capture is not a clean
    'factor'-times oversampling of the drive. A command that never changes
    has phase 0.
    """

    changes = np.nonzero(np.any(np.diff(command, axis=1) != 0, axis=0))[0] + 1
    if len(changes) == 0:
        return 0
    phases = np.unique(changes % factor)
    if len(phases) != 1:
        raise ValueError(
            f"{path}: command changes at {len(phases)} phases of the "
            f"{factor}x oversampled drive"
        )

    return int(phases[0])


def resample(
    capture: Capture,
    rate_hz: float,
    drive_rate_hz: float,
    anti_alias: AntiAlias | None,
    unfiltered: bool = False,
    filtered: bool | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Resample a capture to 'rate_hz', returning (position, command).

    At the drive rate both channels are picked at the command-update
    samples without filtering, matching what a controller at that rate
    reads and writes: the command exactly, the position as captured. Below
    the drive rate both channels pass through the same zero-phase FIR
    first, and the samples a half-filter from either end, which the filter
    cannot settle, are dropped. 'unfiltered' instead picks at the update
    phase below the drive rate too, matching a dataset that was itself
    sampled that way, and 'filtered' forces the filter on or off whatever
    the drive rate, so a capture is placed the way its dataset's rows were
    made. Above the drive rate (a controller updating slower
    than the plant steps) the held command is exact on the finer grid, so
    the samples are picked in step with the updates.
    """

    fs = capture.capture_rate_hz
    drive_factor = _ratio(fs, drive_rate_hz)
    factor = _ratio(fs, rate_hz)
    phase = update_phase(capture.command, drive_factor, capture.path)
    pos, cmd = capture.position, capture.command

    if filtered is None:
        filtered = rate_hz < drive_rate_hz and not unfiltered
    if filtered:
        if anti_alias is None:
            raise ValueError(
                f"Resampling to {rate_hz} Hz below the drive rate "
                f"{drive_rate_hz} Hz needs an anti-alias filter"
            )
        h = anti_alias.taps(fs)
        pos = _filter(pos, h)
        cmd = _filter(cmd, h)
        half = len(h) // 2
        pos, cmd = pos[:, half:-half], cmd[:, half:-half]
        phase = (phase - half) % drive_factor
    sel = slice(phase % factor, None, factor)

    return pos[:, sel], cmd[:, sel]


def _ratio(fs: float, rate_hz: float) -> int:
    factor = round(fs / rate_hz)
    if factor < 1 or not np.isclose(factor * rate_hz, fs):
        raise ValueError(
            f"Capture rate {fs} Hz is not an integer multiple of {rate_hz} Hz"
        )

    return factor


def _filter(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """
    Centred ('same') convolution of each row with 'h', by FFT.
    """

    n = x.shape[1] + len(h) - 1
    size = 1 << (n - 1).bit_length()
    full = np.fft.irfft(
        np.fft.rfft(x, size, axis=1) * np.fft.rfft(h, size), size, axis=1
    )
    half = len(h) // 2

    return full[:, half : half + x.shape[1]]


@dataclass(frozen=True)
class DatasetFrame:
    """
    How a dataset samples and references position, so an external capture
    can be placed in the same frame as the rows a plant was trained on.

    A schema-v3 build records its own rate, filter, rest window and the
    stage's static gain. An older dataset centred each recording on its
    mean and was sampled without an anti-alias filter, its rate comes from
    the SystemSpec.
    """

    rate_hz: float
    reference: str  # "rest" or "mean"
    rest_seconds: float = 0.0
    anti_alias: AntiAlias | None = None
    static_gain: np.ndarray | None = None
    # Rows pair each position with the command this many steps earlier.
    dac_shift: int = 1

    @classmethod
    def of(cls, h5_path: str | Path, rate_hz: float | None) -> DatasetFrame:
        with h5py.File(h5_path, "r") as f:
            attrs: dict[str, Any] = dict(f.attrs)
        if attrs.get("position_reference") != REST_REFERENCE:
            if not rate_hz:
                raise ValueError(
                    f"Dataset {h5_path} records no rate, pass the system's "
                    "data_rate_hz"
                )
            return cls(rate_hz=float(rate_hz), reference=MEAN_REFERENCE)
        band = np.asarray(attrs.get("anti_alias_hz", []), dtype=float)
        aa = (
            AntiAlias(band[0], band[1], float(attrs["anti_alias_db"]))
            if len(band) == 2
            else None
        )

        return cls(
            rate_hz=float(attrs["rate_hz"]),
            reference=REST_REFERENCE,
            rest_seconds=float(attrs["rest_seconds"]),
            anti_alias=aa,
            static_gain=np.asarray(attrs["static_gain_nm_per_count"]),
            dac_shift=int(attrs.get("dac_shift_steps", 1)),
        )

    def place(
        self, capture: ClosedLoopCapture
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        A capture resampled and referenced as the dataset's rows are.

        Rest-referenced datasets subtract each axis's rest: the mean of a
        leading zero-command window when the capture has one, else for a
        driven axis the mean of position minus static gain times command,
        the spring's own equilibrium (a feedback capture starts already
        driven), and for an undriven one its mean. Axes without an
        interferometer channel stay at the frame's zero. Returns (position,
        command), each [A, n] at the same instants, see 'rows' for the
        dataset's pairing.
        """

        pos, cmd = resample(
            capture,
            self.rate_hz,
            capture.drive_rate_hz,
            self.anti_alias,
            filtered=(
                self.anti_alias is not None
                and capture.capture_rate_hz > self.rate_hz
            ),
        )
        if self.reference == MEAN_REFERENCE:
            rest = pos.mean(axis=1)
        else:
            rest = self._rest(capture, pos, cmd)
        pos = pos - rest[:, None]
        pos[~np.asarray(capture.measured)] = 0.0

        return pos, cmd

    def _rest(
        self, capture: ClosedLoopCapture, pos: np.ndarray, cmd: np.ndarray
    ) -> np.ndarray:
        """
        Each axis's rest in the capture's own encoder frame.

        The zero-command test reads the raw command: a filtered one carries
        round-off where the drive was exactly zero. Its window covers the
        rows the dataset averages plus the filter's reach before them.
        """

        n = max(1, int(round(self.rest_seconds * self.rate_hz)))
        reach = (
            len(self.anti_alias.taps(capture.capture_rate_hz))
            if self.anti_alias is not None
            else 0
        )
        raw_n = reach + int(np.ceil(n * capture.capture_rate_hz / self.rate_hz))
        if not np.any(capture.command[:, :raw_n]):
            return pos[:, :n].mean(axis=1)
        rest = pos.mean(axis=1)
        driven = np.any(cmd != 0, axis=1)
        gain = self.static_gain
        if driven.any():
            if gain is None or np.isnan(gain[driven]).any():
                raise ValueError(
                    f"{capture.path}: starts driven and the dataset has no "
                    "static gain to find its rest"
                )
            # The spring's own equilibrium, the stage starting driven.
            rest[driven] = (
                pos[driven] - gain[driven, None] * cmd[driven]
            ).mean(axis=1)

        return rest

    def rows(
        self, pos: np.ndarray, cmd: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Placed (position, command) paired as the dataset's rows pair them:
        each position beside the command 'dac_shift' steps earlier.
        """

        k = self.dac_shift
        if k == 0:
            return pos, cmd

        return pos[:, k:], cmd[:, :-k]
