"""
Measured disturbance for closed-loop rollouts.

The plant model is noise-free, while the real stage holds under ambient
vibration and amplifier noise. This model takes a loop-open, zero-command
capture of the real stage, refers its position spectrum back to the
command through the measured FRF, and during a rollout adds a fresh
random-phase realisation of that command disturbance before the plant. No
physics is assumed: the spectrum is the measurement's own.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import h5py
import numpy as np
import torch

from nn_motion_control.control.workspace import MeasuredStage

# Welch segment length (samples at the servo rate) for the spectrum, and
# the samples skipped at the capture's start (the stale first frames).
SEGMENT = 1 << 15
SKIP_SECONDS = 1.0
# A generated sequence this long (samples) repeats seamlessly, being a
# periodic synthesis, so rollouts of any length can read it.
SEQUENCE = 1 << 18
# Below this frequency (Hz) the measured position is stage motion and is
# referred to the command; above it the stage barely responds, the record
# is sensor noise, and referring it back would invent huge commands.
DISTURBANCE_MAX_HZ = 200.0
# The sensor's white floor is read from this band (Hz) of the record.
SENSOR_BAND_HZ = 1000.0


@dataclass(frozen=True)
class DisturbanceSpectrum:
    """
    One-sided command-referred disturbance PSD per axis (counts^2 / Hz).
    """

    freqs: np.ndarray  # [F]
    psd: np.ndarray  # [A, F]
    rate_hz: float
    sensor_std: np.ndarray  # [A] white measurement noise, position unit

    @classmethod
    def from_capture(
        cls,
        path: str | Path,
        stage: MeasuredStage,
        rate_hz: float,
        n_axes: int,
    ) -> DisturbanceSpectrum:
        """
        From a loop-open, zero-command capture: each axis's position PSD
        divided by its measured FRF magnitude squared.
        """

        with h5py.File(path, "r") as f:
            ts = np.asarray(cast(h5py.Dataset, f["PCAP.TS_START.Value"])[()])
            pos = np.stack(
                [
                    np.asarray(
                        cast(h5py.Dataset, f[f"INENC{i + 1}.VAL.Value"])[()]
                    )
                    for i in range(n_axes)
                ]
            )
        capture_rate = 1.0 / float(np.median(np.diff(ts[1:])))
        factor = int(round(capture_rate / rate_hz))
        if factor < 1 or not math.isclose(
            capture_rate / factor, rate_hz, rel_tol=1e-3
        ):
            raise ValueError(
                f"Capture rate {capture_rate:g} Hz is not a multiple of "
                f"{rate_hz:g}"
            )
        pos = pos[:, int(SKIP_SECONDS * capture_rate) :]
        pos = _decimate(pos, factor)
        freqs, power = _welch(pos, rate_hz)
        mags = []

        for ax in range(n_axes):
            mags.append(
                np.interp(
                    freqs,
                    stage.freqs,
                    np.abs(stage.response[ax]),
                    left=stage.gain[ax],
                )
            )

        referred = power / np.stack(mags) ** 2
        referred[:, freqs > DISTURBANCE_MAX_HZ] = 0.0
        floor = power[:, freqs >= SENSOR_BAND_HZ].mean(axis=1)
        sensor_std = np.sqrt(floor * rate_hz / 2.0)

        return cls(freqs, referred, rate_hz, sensor_std)

    def sample(
        self, rows: int, generator: torch.Generator | None, device
    ) -> torch.Tensor:
        """
        A random-phase realisation [rows, SEQUENCE, A] in counts.
        """

        f = np.fft.rfftfreq(SEQUENCE, 1.0 / self.rate_hz)
        psd = np.stack(
            [np.interp(f, self.freqs, p) for p in self.psd]
        )  # [A, F]
        # Amplitude per bin so the sequence's variance matches the PSD.
        amp = np.sqrt(psd * self.rate_hz * SEQUENCE / 2.0)
        amp[:, 0] = 0.0
        amp_t = torch.as_tensor(amp, device=device, dtype=torch.float64)
        phase = (
            2.0
            * math.pi
            * torch.rand(
                (rows, *amp.shape),
                generator=generator,
                device=device,
                dtype=torch.float64,
            )
        )
        spec = amp_t * torch.exp(1j * phase)
        seq = torch.fft.irfft(spec, n=SEQUENCE, dim=-1)

        return seq.transpose(1, 2).float()


class CommandDisturbance:
    """
    Actuator model adding a measured disturbance to the command each step,
    composed on an optional inner model (applied first).

    'generator' makes the realisations reproducible (validation); None
    draws fresh ones each rollout (training).
    """

    def __init__(
        self,
        spectrum: DisturbanceSpectrum,
        scale: float = 1.0,
        inner=None,
        generator: torch.Generator | None = None,
    ):
        self.spectrum = spectrum
        self.scale = scale
        self.inner = inner
        self.generator = generator
        self._seq: torch.Tensor | None = None
        self._k = 0

    def reset(self, command: torch.Tensor) -> None:
        if self.inner is not None:
            self.inner.reset(command)
        self._seq = self.scale * self.spectrum.sample(
            command.shape[0], self.generator, command.device
        )
        self._k = 0

    def apply(self, command: torch.Tensor) -> torch.Tensor:
        if self.inner is not None:
            command = self.inner.apply(command)
        if self._seq is None:
            self.reset(command)
        seq = cast(torch.Tensor, self._seq)
        noise = seq[:, self._k % seq.shape[1]].to(command.dtype)
        self._k += 1

        return command + noise


def _decimate(x: np.ndarray, factor: int) -> np.ndarray:
    # Brick-wall low-pass at the new Nyquist in the frequency domain, then
    # keep every 'factor'-th sample.
    if factor == 1:
        return x
    spec = np.fft.rfft(x, axis=-1)
    keep = spec.shape[-1] // factor
    spec[..., keep:] = 0.0

    return np.fft.irfft(spec, n=x.shape[-1], axis=-1)[..., ::factor]


def _welch(x: np.ndarray, rate_hz: float) -> tuple[np.ndarray, np.ndarray]:
    # One-sided PSD per row, Hann windows at half overlap.
    n = min(SEGMENT, x.shape[-1])
    step = n // 2
    win = np.hanning(n)
    scale = 1.0 / (rate_hz * (win**2).sum())
    starts = range(0, x.shape[-1] - n + 1, step)
    acc = np.zeros((x.shape[0], n // 2 + 1))

    for s in starts:
        seg = x[:, s : s + n]
        seg = seg - seg.mean(axis=-1, keepdims=True)
        acc += np.abs(np.fft.rfft(seg * win, axis=-1)) ** 2
    psd = acc * scale / len(starts)
    psd[:, 1:-1] *= 2.0

    return np.fft.rfftfreq(n, 1.0 / rate_hz), psd


class NoisySensor:
    """
    Sensor model adding white measurement noise [A] to the observed
    position, composed on an optional inner model (applied after), with
    velocity from the noisy positions as the controller would see it.
    """

    def __init__(
        self,
        std: np.ndarray,
        inner=None,
        generator: torch.Generator | None = None,
    ):
        self.std = torch.as_tensor(std, dtype=torch.float32)
        self.inner = inner
        self.generator = generator
        self._prev: torch.Tensor | None = None

    def _noisy(self, position: torch.Tensor) -> torch.Tensor:
        noise = torch.randn(
            position.shape,
            generator=self.generator,
            device=position.device,
            dtype=position.dtype,
        )

        return position + noise * self.std.to(position.device, position.dtype)

    def reset(self, position: torch.Tensor) -> None:
        if self.inner is not None:
            self.inner.reset(position)
        self._prev = None

    def observe(
        self, position: torch.Tensor, velocity: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        seen = self._noisy(position)
        vel = seen - self._prev if self._prev is not None else velocity
        self._prev = seen
        if self.inner is not None:
            return self.inner.observe(seen, vel)

        return seen, vel
