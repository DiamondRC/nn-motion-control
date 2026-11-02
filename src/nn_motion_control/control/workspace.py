"""
Workspace-wide reference trajectories for training a generalist controller.

Each draw is a unit-scale displacement shape from its seed, scaled per row
by an amplitude drawn log-uniformly up to the largest that row can realise:
the travel box from its own seed, and the stage's command budget, estimated
by inverting the dominant suspension mode. Shapes are set in seconds and
hertz, so one spec serves any servo rate and horizon, and positions are in
the dataset's rest frame (zero command holds the stage at zero).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

import h5py
import numpy as np
import torch

FAMILIES = (
    "hold",
    "move",
    "line",
    "arc",
    "spiral",
    "helix",
    "raster",
    "smooth",
)

# Travel kept clear of each measured end-stop (position unit, nm).
DEFAULT_MARGIN = 100_000.0
# A seed or reference may sit only where its static holding command uses at
# most this share of the safe range, leaving the rest for feedback.
HOLD_FRACTION = 0.9
# Peak displacement range (position unit, nm) the amplitude is drawn from,
# log-uniformly so nm-scale moves keep a fair share of the draws.
AMPLITUDE_RANGE = (10.0, 1.0e6)
# Shape rate range (Hz): rotation rate, line rate or component frequency.
FREQUENCY_RANGE_HZ = (0.5, 300.0)
# Highest shape rate drawn (Hz). Above the dominant mode's fit band the
# command estimate is corrected by the measured FRF (StageModel).
REFERENCE_MAX_HZ = 300.0
# Share of a seed's command headroom (from the command holding it to each
# side of the safe range) a reference may plan to use, leaving the rest for
# feedback.
COMMAND_FRACTION = 0.6
# A shape's content reaches about this multiple of its drawn rate
# (minimum-jerk ramps, rounded triangle harmonics), the band its command
# correction is read over.
CONTENT_SPAN = 2.0
# Dominant-mode fit: lines up to this frequency, over these grids.
FIT_MAX_HZ = 100.0
FIT_NATURAL_HZ = (3.0, 40.0)
FIT_DAMPING = (0.003, 0.7)
FIT_GRID = 200

# References are three-axis.
N_AXES = 3
# Every shape fades in with a minimum-jerk envelope over this many periods
# of its drawn rate, so a stage at rest is never asked for a velocity step.
FADE_CYCLES = 1.0
# Rest samples placed before step 0 when estimating the command, so the
# start-up acceleration counts against the budget.
REST_PAD = 4
# A move starts within this many seconds (under half the shortest training
# horizon), the same at every horizon.
MOVE_START_S = 0.02
# An arc sweeps between these angles, then holds.
ARC_SWEEP_RAD = (math.pi / 6.0, 2.0 * math.pi)
# A spiral's radius ends at this multiple of its start, reached after this
# many turns (log-uniform), the same at every horizon.
SPIRAL_RADIUS_RATIO = (0.1, 1.9)
SPIRAL_TURNS = (1.0, 20.0)
# Helix rise per turn and raster line pitch, as multiples of the radius or
# line length, and the turns or lines before the slow axis reverses.
PITCH_RANGE = (0.02, 1.0)
LEG_COUNT = (1, 6)
# Smooth paths sum this many components per axis, weighted down as 1 / f,
# spread over this ratio of frequencies below the row's drawn rate.
SMOOTH_COMPONENTS = 4
SMOOTH_SPREAD = 10.0
# Rounding of triangle-wave tips: 1 is a sharp triangle, lower is rounder.
TRIANGLE_ROUNDING = 0.98
# A raster's fast triangle runs this many times slower than the drawn rate,
# so its harmonics stay inside REFERENCE_MAX_HZ.
RASTER_HARMONICS = 5.0
# Floors for norms and ratios before dividing.
EPS = 1e-12


def _quintic(s: torch.Tensor) -> torch.Tensor:
    # Minimum-jerk 0 -> 1: zero velocity and acceleration at both ends.
    s = s.clamp(0.0, 1.0)

    return s**3 * (10.0 - 15.0 * s + 6.0 * s * s)


def _triangle(theta: torch.Tensor) -> torch.Tensor:
    """
    Rounded triangle wave in [-1, 1] with the phase of sin (0 at 0).
    """

    beta = TRIANGLE_ROUNDING

    return torch.asin(beta * torch.sin(theta)) / math.asin(beta)


def _central_rate(x: torch.Tensor) -> torch.Tensor:
    # d/dstep along dim 1, one-sided at the ends.
    if x.shape[1] < 2:
        return torch.zeros_like(x)

    return torch.gradient(x, dim=1)[0]


@dataclass(frozen=True)
class StageModel:
    """
    Dominant suspension mode per axis: a static gain (position per count)
    and a second-order resonance, fitted to the dataset's measured FRF.

    Its inverse estimates the command a reference needs, which is what
    bounds a reference to what the stage can physically realise. Where the
    real stage responds less than the mode (above its fit band) it needs
    more command than the inverse says: 'correction' [A, L] holds, per
    axis, the running maximum over frequency of the model-to-measured gain
    ratio (never below 1) at the measured lines 'correction_hz' [L], and
    'dynamic_scale' reads it for a band.
    """

    gain: torch.Tensor  # [A] position unit per count
    natural_hz: torch.Tensor  # [A]
    damping: torch.Tensor  # [A]
    correction_hz: torch.Tensor | None = None  # [L]
    correction: torch.Tensor | None = None  # [A, L]

    def dynamic_scale(self, top_hz: torch.Tensor) -> torch.Tensor:
        """
        Factor [B, A] the dynamic command must grow by for content up to
        'top_hz' [B]: the measured shortfall anywhere below that band.
        """

        n_axes = self.gain.numel()
        if self.correction is None or self.correction_hz is None:
            return torch.ones(top_hz.shape[0], n_axes, dtype=top_hz.dtype)
        lines = self.correction_hz.to(top_hz.device, top_hz.dtype)
        at = torch.searchsorted(lines, top_hz, right=True) - 1
        table = self.correction.to(top_hz.device, top_hz.dtype)
        scale = table[:, at.clamp_min(0)].T  # [B, A]

        # Below the first measured line the mode is trusted as it is.
        return torch.where(at[:, None] >= 0, scale, torch.ones_like(scale))

    def command(self, pos: torch.Tensor, rate_hz: float) -> torch.Tensor:
        """
        Command [B, H, A] that drives the mode along 'pos' [B, H, A].
        """

        dev, dt = pos.device, pos.dtype
        w = (2.0 * math.pi * self.natural_hz).to(dev, dt)
        zeta = self.damping.to(dev, dt)
        vel = _central_rate(pos) * rate_hz
        acc = _central_rate(vel) * rate_hz
        drive = pos + (2.0 * zeta / w) * vel + acc / (w * w)

        return drive / self.gain.to(dev, dt)

    @classmethod
    def from_dataset(cls, h5_path: str, axes: tuple[str, ...]) -> StageModel:
        """
        Fit each axis's diagonal FRF up to FIT_MAX_HZ, holding the gain at
        the dataset's travel-measured static gain.
        """

        with h5py.File(h5_path, "r") as f:
            gain = np.asarray(f.attrs["static_gain_nm_per_count"], float)
            frf = {
                a: (
                    np.asarray(cast(h5py.Dataset, f[f"frf/{a}/freqs"])[()]),
                    np.asarray(cast(h5py.Dataset, f[f"frf/{a}/H"])[()])[i],
                )
                for i, a in enumerate(axes)
            }
        f0_grid = np.geomspace(*FIT_NATURAL_HZ, FIT_GRID)[:, None, None]
        z_grid = np.geomspace(*FIT_DAMPING, FIT_GRID)[None, :, None]
        natural, damping, shortfall = [], [], []
        lines = frf[axes[0]][0]
        if any(not np.array_equal(frf[a][0], lines) for a in axes):
            raise ValueError("Every axis's FRF must share its frequency lines")

        for i, a in enumerate(axes):
            freqs, h = frf[a]
            keep = freqs <= FIT_MAX_HZ
            s = freqs[keep][None, None, :] / f0_grid
            model = gain[i] / (1.0 - s * s + 2j * z_grid * s)
            err = np.abs(np.log(model / h[keep][None, None, :])) ** 2
            j, k = np.unravel_index(np.argmin(err.mean(-1)), err.shape[:2])
            natural.append(float(f0_grid[j, 0, 0]))
            damping.append(float(z_grid[0, k, 0]))
            s = freqs / natural[-1]
            fitted = gain[i] / (1.0 - s * s + 2j * damping[-1] * s)
            ratio = np.maximum(np.abs(fitted) / np.abs(h), 1.0)
            shortfall.append(np.maximum.accumulate(ratio))

        return cls(
            torch.tensor(gain, dtype=torch.float32),
            torch.tensor(natural, dtype=torch.float32),
            torch.tensor(damping, dtype=torch.float32),
            torch.tensor(lines, dtype=torch.float32),
            torch.tensor(np.stack(shortfall), dtype=torch.float32),
        )


@dataclass(frozen=True)
class WorkspaceLimits:
    """
    What a reference may ask of the stage: the travel box (position unit,
    rest frame) and the command range (counts) per axis.
    """

    pos_lo: torch.Tensor  # [A]
    pos_hi: torch.Tensor
    cmd_lo: torch.Tensor  # [A], negative
    cmd_hi: torch.Tensor  # [A], positive
    stage: StageModel
    rate_hz: float

    def __post_init__(self):
        if bool((self.pos_lo >= self.pos_hi).any()):
            raise ValueError("Workspace box is empty on some axis")


def max_amplitude(
    base: torch.Tensor,
    coeff: torch.Tensor,
    lo: torch.Tensor,
    hi: torch.Tensor,
) -> torch.Tensor:
    """
    Largest 'a >= 0' per row with 'lo <= base + a * coeff <= hi' at every
    step and axis; 'base' is [B, A], 'coeff' [B, H, A].
    """

    up = (hi - base).clamp_min(0.0)[:, None, :]
    down = (base - lo).clamp_min(0.0)[:, None, :]
    inf = torch.full_like(coeff, math.inf)
    limit = torch.where(
        coeff > EPS,
        up / coeff.clamp_min(EPS),
        torch.where(coeff < -EPS, down / (-coeff).clamp_min(EPS), inf),
    )

    return limit.amin(dim=(1, 2))


class WorkspaceReference:
    """
    Callable '(origin, horizon, generator) -> (pos, vel)' drawing a mix of
    FAMILIES across the workspace.

    'spec' may set 'amplitude' and 'frequency_hz' ranges, per-family
    'weights' and an 'amplitude_curriculum' ({"start": nm, "epochs": n})
    that widens the amplitude ceiling geometrically from 'start' over
    'epochs'. 'set_epoch' moves it, None is the full range (validation).
    After each call 'last_families' holds every row's family index.
    """

    def __init__(self, spec: Mapping, limits: WorkspaceLimits):
        unknown = set(spec.get("weights", {})) - set(FAMILIES)
        if unknown:
            raise ValueError(f"Unknown workspace families: {sorted(unknown)}")
        self.limits = limits
        self.amplitude = tuple(
            float(v) for v in spec.get("amplitude", AMPLITUDE_RANGE)
        )
        f_lo, f_hi = (
            float(v) for v in spec.get("frequency_hz", FREQUENCY_RANGE_HZ)
        )
        nyquist = limits.rate_hz / 2.0
        self.frequency = (f_lo, min(f_hi, REFERENCE_MAX_HZ, nyquist))
        if not 0 < self.amplitude[0] < self.amplitude[1]:
            raise ValueError("Workspace amplitude must be 0 < lo < hi")
        if not 0 < self.frequency[0] < self.frequency[1]:
            raise ValueError("Workspace frequency_hz must be 0 < lo < hi")
        weights = spec.get("weights", {})
        self.weights = torch.tensor(
            [float(weights.get(name, 1.0)) for name in FAMILIES]
        )
        if not bool((self.weights >= 0).all()) or self.weights.sum() <= 0:
            raise ValueError("Workspace weights must be >= 0, not all zero")
        curriculum = spec.get("amplitude_curriculum")
        self._curriculum = None
        if curriculum:
            self._curriculum = (
                float(curriculum["start"]),
                int(curriculum["epochs"]),
            )
        self._ceiling = self.amplitude[1]
        self.last_families: torch.Tensor | None = None

    def set_epoch(self, epoch: int | None) -> None:
        if self._curriculum is None or epoch is None:
            self._ceiling = self.amplitude[1]

            return

        start, epochs = self._curriculum
        frac = min(1.0, epoch / max(epochs, 1))
        ceiling = start * (self.amplitude[1] / start) ** frac
        self._ceiling = min(max(ceiling, self.amplitude[0]), self.amplitude[1])

    def __call__(
        self,
        origin: torch.Tensor,
        horizon: int,
        generator: torch.Generator | None = None,
        hold: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        'hold' is the command [B, A] already holding each seed, which sets
        the command headroom left for the motion. Without it the hold is
        estimated from the static gain, which overstates it far from rest
        where the suspension softens.
        """

        b, a = origin.shape
        if a != N_AXES:
            raise ValueError("Workspace references need three axes")
        # Shapes and their command estimate are built in float64: a float32
        # phase of hundreds of radians carries enough rounding to swamp
        # the twice-differentiated command. Only the result is cast back.
        dev, out_dt, dt = origin.device, origin.dtype, torch.float64
        origin = origin.to(dt)
        if hold is not None:
            hold = hold.to(dt)
        lim = self.limits
        t = torch.arange(horizon, device=dev, dtype=dt) / lim.rate_hz

        def u(lo, hi, shape=(b,)):
            r = torch.rand(shape, generator=generator, device=dev, dtype=dt)

            return lo + (hi - lo) * r

        def log_u(lo, hi, shape=(b,)):
            return torch.exp(u(math.log(lo), math.log(hi), shape))

        fam = torch.multinomial(
            self.weights.to(dev), b, replacement=True, generator=generator
        )
        freq = log_u(*self.frequency)
        shape = torch.zeros(b, horizon, a, device=dev, dtype=dt)

        for i, name in enumerate(FAMILIES):
            rows = (fam == i).nonzero().flatten()
            # Every family draws its parameters for the whole batch so the
            # generator advances the same way whatever the family mix.
            d = _draw_shape(name, t, freq, u, log_u, horizon)
            if len(rows):
                shape[rows] = d[rows]

        fade = _quintic(t[None, :] * freq[:, None] / FADE_CYCLES)
        shape = shape * fade[:, :, None]
        # Unit peak displacement, so the amplitude is the peak excursion.
        peak = shape.abs().amax(dim=(1, 2)).clamp_min(EPS)
        shape = shape / peak[:, None, None]
        stage = lim.stage
        if hold is None:
            hold = origin / stage.gain.to(dev, dt)
        rest = torch.zeros(b, REST_PAD, a, device=dev, dtype=dt)
        path = torch.cat([rest, shape], 1)
        static = path / stage.gain.to(dev, dt)
        # The measured stage needs more than the mode's dynamic command
        # where it responds less; the static part fits to about 1 %.
        scale = stage.dynamic_scale(CONTENT_SPAN * freq)
        dynamic = stage.command(path, lim.rate_hz) - static
        unit_cmd = static + dynamic * scale[:, None, :]
        up = (lim.cmd_hi.to(dev, dt) - hold).clamp_min(0.0)
        down = (hold - lim.cmd_lo.to(dev, dt)).clamp_min(0.0)
        cmd_lo = hold - COMMAND_FRACTION * down
        cmd_hi = hold + COMMAND_FRACTION * up
        cap = torch.minimum(
            max_amplitude(
                origin, shape, lim.pos_lo.to(dev, dt), lim.pos_hi.to(dev, dt)
            ),
            max_amplitude(hold, unit_cmd, cmd_lo, cmd_hi),
        ).clamp(max=self._ceiling)
        floor = self.amplitude[0]
        lo = torch.full_like(cap, math.log(floor))
        hi = torch.log(cap.clamp_min(floor))
        r = torch.rand((b,), generator=generator, device=dev, dtype=dt)
        amp = torch.where(cap >= floor, torch.exp(lo + (hi - lo) * r), cap)
        pos = origin[:, None, :] + amp[:, None, None] * shape
        self.last_families = fam

        return pos.to(out_dt), _central_rate(pos).to(out_dt)


def _random_frame(u, b: int):
    # A random orthonormal triple per row, from Gram-Schmidt on Gaussians
    # built from uniforms (Box-Muller) so the shared generator drives it.
    def gauss():
        r = torch.sqrt(-2.0 * torch.log(u(EPS, 1.0, (b, 3))))

        return r * torch.cos(u(0.0, 2.0 * math.pi, (b, 3)))

    e1 = gauss()
    e1 = e1 / e1.norm(dim=1, keepdim=True).clamp_min(EPS)
    v = gauss()
    e2 = v - (v * e1).sum(1, keepdim=True) * e1
    e2 = e2 / e2.norm(dim=1, keepdim=True).clamp_min(EPS)

    return e1, e2, torch.linalg.cross(e1, e2)


def _draw_shape(name, t, freq, u, log_u, horizon) -> torch.Tensor:
    """
    One family's unit-free displacement [B, H, 3], zero at step 0.
    """

    b = freq.shape[0]
    e1, e2, e3 = _random_frame(u, b)
    tt = t[None, :]

    def along(vec, w):
        return w[:, :, None] * vec[:, None, :]

    if name == "hold":
        return torch.zeros(b, horizon, 3, device=t.device, dtype=t.dtype)
    if name == "move":
        start = u(0.0, MOVE_START_S)
        prog = _quintic((tt - start[:, None]) * freq[:, None])

        return along(e1, prog)
    if name == "line":
        return along(e1, torch.sin(2.0 * math.pi * freq[:, None] * tt))
    if name == "arc":
        sweep = u(*ARC_SWEEP_RAD)
        duration = sweep / (2.0 * math.pi * freq)
        phi = sweep[:, None] * _quintic(tt / duration[:, None])

        return along(e1, torch.cos(phi) - 1.0) + along(e2, torch.sin(phi))
    sign = torch.where(u(0.0, 1.0) < 0.5, -1.0, 1.0)
    phi = sign[:, None] * 2.0 * math.pi * freq[:, None] * tt
    if name == "spiral":
        ratio = u(*SPIRAL_RADIUS_RATIO)
        turns = log_u(*SPIRAL_TURNS)
        prog = (phi.abs() / (2.0 * math.pi * turns[:, None])).clamp(max=1.0)
        r = 1.0 + (ratio[:, None] - 1.0) * prog

        return along(e1, r * torch.cos(phi) - 1.0) + along(
            e2, r * torch.sin(phi)
        )
    pitch = log_u(*PITCH_RANGE)
    legs = torch.floor(u(float(LEG_COUNT[0]), float(LEG_COUNT[1]) + 1.0))
    # The slow axis climbs for 'legs' turns (a quarter of its triangle
    # period), then comes back.
    slow = (pitch * legs)[:, None] * _triangle(
        phi.abs() / (4.0 * legs[:, None])
    )
    if name == "helix":
        circle = along(e1, torch.cos(phi) - 1.0) + along(e2, torch.sin(phi))

        return circle + along(e3, slow)
    if name == "raster":
        fast = _triangle(phi / RASTER_HARMONICS)
        slow = (pitch * legs)[:, None] * _triangle(
            phi.abs() / (4.0 * RASTER_HARMONICS * legs[:, None])
        )

        return along(e1, fast) + along(e2, slow)
    if name == "smooth":
        m = SMOOTH_COMPONENTS
        spread = u(-math.log(SMOOTH_SPREAD), 0.0, (b, 3, m))
        f = freq[:, None, None] * torch.exp(spread)
        w = u(0.0, 1.0, (b, 3, m)) / f
        ph = u(0.0, 2.0 * math.pi, (b, 3, m))
        arg = (
            2.0 * math.pi * f[..., None] * tt[:, None, None, :] + ph[..., None]
        )
        wave = (w[..., None] * (torch.sin(arg) - torch.sin(ph)[..., None])).sum(
            2
        )

        return wave.transpose(1, 2)

    raise ValueError(f"Unknown workspace family: {name!r}")


# The measured-FRF command check pads a path with this much rest (seconds)
# either side and returns to rest along a minimum-jerk ramp of this length,
# so the transform sees a smooth periodic signal.
FRF_PAD_S = 0.2
FRF_RETURN_S = 0.4


@dataclass(frozen=True)
class MeasuredStage:
    """
    The dataset's measured diagonal FRF per axis, for checking a reference
    against what the real stage needs rather than against a fitted mode.
    """

    freqs: np.ndarray  # [L] Hz
    response: np.ndarray  # [A, L] complex, position unit per count
    gain: np.ndarray  # [A] static gain

    @classmethod
    def from_dataset(cls, h5_path: str, axes: tuple[str, ...]) -> MeasuredStage:
        with h5py.File(h5_path, "r") as f:
            gain = np.asarray(f.attrs["static_gain_nm_per_count"], float)
            freqs = np.asarray(
                cast(h5py.Dataset, f[f"frf/{axes[0]}/freqs"])[()]
            )
            response = np.stack(
                [
                    np.asarray(cast(h5py.Dataset, f[f"frf/{a}/H"])[()])[i]
                    for i, a in enumerate(axes)
                ]
            )

        return cls(freqs, response, gain)

    def command(
        self, pos: np.ndarray, rate_hz: float, band_hz: float
    ) -> np.ndarray:
        """
        Command [B, H, A] the measured stage needs to follow 'pos' [B, H, A]
        (displacement from rest), counting content up to 'band_hz' only.

        The path rests before step 0 and returns smoothly to rest after
        its end, so the transform's wrap adds no step of its own; judge
        only a path that is itself at rest or slow at its end, or read
        just its early part.
        """

        b, h, a = pos.shape
        pad = int(round(FRF_PAD_S * rate_hz))
        back = int(round(FRF_RETURN_S * rate_hz))
        s = np.linspace(0.0, 1.0, back)[None, :, None]
        ramp = pos[:, -1:] * (1.0 - s**3 * (10.0 - 15.0 * s + 6.0 * s * s))
        rest = np.zeros((b, pad, a))
        path = np.concatenate([rest, pos, ramp, rest], axis=1)
        spec = np.fft.rfft(path, axis=1)
        f = np.fft.rfftfreq(path.shape[1], 1.0 / rate_hz)
        cmd = np.zeros_like(spec)

        for ax in range(a):
            mag = np.interp(
                f, self.freqs, np.abs(self.response[ax]), left=self.gain[ax]
            )
            phase = np.interp(
                f, self.freqs, np.unwrap(np.angle(self.response[ax])), left=0.0
            )
            inverse = 1.0 / (mag * np.exp(1j * phase))
            cmd[:, :, ax] = np.where(f <= band_hz, spec[:, :, ax] * inverse, 0)

        return np.fft.irfft(cmd, n=path.shape[1], axis=1)[:, pad : pad + h]
