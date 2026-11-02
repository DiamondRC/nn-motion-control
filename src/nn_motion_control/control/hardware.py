"""Sensor interface model for closed-loop controller training.

The plant simulates continuous, noiseless dynamics, but the deployed
controller reads an interferometer that reports position on a fixed
integer grid (its least count). Training the policy on the plant's
continuous output lets it chase sub-count position detail, which on
hardware is only sensor noise, so it learns a large high-frequency
command that does not transfer. Observing through the real quantiser
removes that detail and with it the incentive to chase it, while
leaving real moves (many counts) untouched. This is the actual sensor,
deterministic and calibrated from hardware, not an injected disturbance.
"""

from __future__ import annotations

import math

import torch


class ActuatorModel:
    """First-order low-pass on the DAC before the plant integrates it.

    The plant learned its DAC-to-position response only where the
    identification excited it (below about 25 Hz); above that it
    extrapolates a fantasy gain a real actuator and mechanism cannot
    deliver, and a controller trained through it stabilises by exciting
    that fantasy into a high-frequency limit cycle. Rolling the command
    off at the identification bandwidth restores the missing mechanical
    low-pass, so the plant is only driven where it is faithful and the
    fantasy modes cannot be excited. The rolled-off command is what
    drives the plant and what is scored, matching a real actuator that
    cannot inject motion faster than its bandwidth.

    'cutoff_hz' is the -3 dB corner and 'servo_rate_hz' the step rate;
    a non-positive cutoff disables the model (identity).
    """

    def __init__(self, cutoff_hz: float, servo_rate_hz: float):
        self.cutoff_hz = float(cutoff_hz)
        if self.cutoff_hz > 0.0:
            tau = 1.0 / (2.0 * math.pi * self.cutoff_hz)
            dt = 1.0 / float(servo_rate_hz)
            self.alpha = dt / (tau + dt)
        else:
            self.alpha = 1.0
        self._prev: torch.Tensor | None = None

    def reset(self, command: torch.Tensor) -> None:
        """Seed the filter with the current operating command."""

        self._prev = command

    def apply(self, command: torch.Tensor) -> torch.Tensor:
        if self.cutoff_hz <= 0.0:
            return command
        if self._prev is None:
            self._prev = command
        filtered = self._prev + self.alpha * (command - self._prev)
        self._prev = filtered

        return filtered


class QuantisedCommand:
    """
    Round the command to whole counts before the plant, as the DAC does,
    after an optional inner command model.

    A controller trained on continuous commands can rely on sub-count
    resolution the hardware does not have; at the count level, where a
    precision stage holds, that difference is the whole signal. The
    rounding passes its gradient straight through so a policy still trains.
    """

    def __init__(self, inner: ActuatorModel | None = None):
        self.inner = inner

    def reset(self, command: torch.Tensor) -> None:
        if self.inner is not None:
            self.inner.reset(command)

    def apply(self, command: torch.Tensor) -> torch.Tensor:
        if self.inner is not None:
            command = self.inner.apply(command)

        return command + (torch.round(command) - command).detach()


class SensorModel:
    """Quantise the observed position to the encoder grid and derive
    velocity from quantised position differences.

    'grid' is the encoder least count in the position unit (nm for the
    deltabot); a non-positive grid disables the model (identity). The
    quantiser is a straight-through estimator so a policy still trains
    through it. Velocity is the change in quantised position per step,
    the same units as the plant's velocity channel, so sub-count motion
    vanishes from both channels the controller sees.
    """

    def __init__(self, grid: float):
        self.grid = float(grid)
        self._prev: torch.Tensor | None = None

    def _quantise(self, position: torch.Tensor) -> torch.Tensor:
        q = torch.round(position / self.grid) * self.grid

        return position + (q - position).detach()

    def reset(self, position: torch.Tensor) -> None:
        """Seed the velocity reference with the initial quantised position."""

        self._prev = self._quantise(position) if self.grid > 0.0 else position

    def observe(
        self, position: torch.Tensor, velocity: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.grid <= 0.0:
            return position, velocity

        q = self._quantise(position)
        if self._prev is None:
            self._prev = q
        vel = q - self._prev
        self._prev = q

        return q, vel
