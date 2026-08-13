"""
Randomised PVT reference trajectories for training a generalist controller.

Real scans are a mix, so the controller must track any reference, not one
spiral. Each family maps a physical origin [B, A] and a horizon to a position
and velocity demand [B, H, A] (velocity in units/step, the per-step change
the policy feeds forward). Every family is anchored so the trajectory starts
on the origin, the controller tracks from its current state, and each sample
draws its own params, so one batch spans many trajectories.

sample_mixed_reference picks a family per sample by weight and returns the
blend. Pass a seeded generator for reproducible validation, None for fresh
training.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch


def _u(lo, hi, shape, generator, device, dtype):
    """Uniform [lo, hi) sample of shape on the given device."""

    r = torch.rand(shape, generator=generator, device=device, dtype=dtype)

    return lo + (hi - lo) * r


def _z_axis(n_axes: int) -> int | None:
    """The ramp axis for helical / z-scan motion (axis 2 when it exists)."""

    return 2 if n_axes >= 3 else None


def spiral_family(origin, k, radius, angular, z_rate, xy):
    """
    Anchored circle/helix: starts on origin, sweeps angular rad per step.
    """

    b, a = origin.shape
    ang = angular[:, None] * k[None, :]  # [B, H]
    x0, y0 = xy
    pos = origin[:, None, :].expand(b, len(k), a).clone()
    r = radius[:, None]
    pos[:, :, x0] = origin[:, x0 : x0 + 1] - r + r * torch.cos(ang)
    pos[:, :, y0] = origin[:, y0 : y0 + 1] + r * torch.sin(ang)
    vel = torch.zeros_like(pos)
    vel[:, :, x0] = -r * angular[:, None] * torch.sin(ang)
    vel[:, :, y0] = r * angular[:, None] * torch.cos(ang)
    zc = _z_axis(a)
    if zc is not None:
        pos[:, :, zc] = origin[:, zc : zc + 1] + z_rate[:, None] * k[None, :]
        vel[:, :, zc] = z_rate[:, None].expand(b, len(k))

    return pos, vel


def line_family(origin, k, direction, amplitude, angular):
    """
    Sinusoidal traverse along a random unit direction (back-and-forth
    line scan).
    """

    s = torch.sin(
        angular[:, None] * k[None, :]
    )  # sin(0) = 0 -> starts on origin
    c = torch.cos(angular[:, None] * k[None, :])
    span = (amplitude[:, None] * s)[:, :, None] * direction[:, None, :]
    pos = origin[:, None, :] + span
    rate = (amplitude[:, None] * angular[:, None] * c)[:, :, None] * direction[
        :, None, :
    ]

    return pos, rate


def step_family(origin, k, delta, start, duration):
    """
    Point-to-point move: smoothstep from origin to origin + delta, then
    hold.
    """

    s = ((k[None, :] - start[:, None]) / duration[:, None]).clamp(
        0.0, 1.0
    )  # [B, H]
    smooth = s * s * (3.0 - 2.0 * s)
    pos = origin[:, None, :] + delta[:, None, :] * smooth[:, :, None]
    inside = (k[None, :] >= start[:, None]) & (
        k[None, :] <= start[:, None] + duration[:, None]
    )
    dsmooth = (6.0 * s * (1.0 - s) / duration[:, None]) * inside
    vel = delta[:, None, :] * dsmooth[:, :, None]

    return pos, vel


def random_smooth_family(origin, k, amps, angulars, phases):
    """
    Smooth arbitrary path: a sum of low-frequency sinusoids per axis,
    anchored.
    """

    # amps/angulars/phases: [B, A, M]. Subtract the k=0 value so pos[0]
    # equals origin.
    ang = (
        angulars[..., None] * k[None, None, None, :] + phases[..., None]
    )  # [B, A, M, H]
    wave = torch.sin(ang) - torch.sin(phases)[..., None]
    pos = origin[:, None, :] + (amps[..., None] * wave).sum(dim=2).transpose(
        1, 2
    )
    rate = (amps[..., None] * angulars[..., None] * torch.cos(ang)).sum(dim=2)
    vel = rate.transpose(1, 2)

    return pos, vel


def sample_mixed_reference(
    origin: torch.Tensor,
    horizon: int,
    spec: Mapping | None = None,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Draw a per-sample mix of trajectory families, returning
    (position, velocity).

    spec overrides the per-family parameter ranges and mixing weights,
    sensible physical defaults (nm, rad/step, nm/step) apply otherwise.
    generator makes the draw reproducible (validation), None gives fresh
    randomness (training).
    """

    spec = dict(spec or {})
    b, a = origin.shape
    dev, dt = origin.device, origin.dtype
    k = torch.arange(horizon, device=dev, dtype=dt)
    gen = generator

    def u(lo, hi, shape):
        return _u(lo, hi, shape, gen, dev, dt)

    def rng(key, lo, hi):
        r = spec.get(key, (lo, hi))

        return float(r[0]), float(r[1])

    xy = tuple(int(v) for v in spec.get("xy", (0, 1)))

    # Spiral: random radius, signed angular rate, z ramp.
    r_lo, r_hi = rng("radius", 50.0, 1500.0)
    w_lo, w_hi = rng("angular", 0.002, 0.03)
    zr_lo, zr_hi = rng("z_rate", -30.0, 30.0)
    sign = torch.where(u(0, 1, (b,)) < 0.5, -1.0, 1.0)
    sp = spiral_family(
        origin,
        k,
        u(r_lo, r_hi, (b,)),
        sign * u(w_lo, w_hi, (b,)),
        u(zr_lo, zr_hi, (b,)),
        xy,
    )

    # Line scan: random unit direction, amplitude, rate.
    d = u(-1.0, 1.0, (b, a))
    d = d / d.norm(dim=1, keepdim=True).clamp_min(1e-6)
    ln = line_family(origin, k, d, u(r_lo, r_hi, (b,)), u(w_lo, w_hi, (b,)))

    # Step: random per-axis target offset, start time, duration.
    st_lo, st_hi = rng("step", -1000.0, 1000.0)
    start = u(0.0, max(1.0, horizon * 0.4), (b,))
    dur = u(max(1.0, horizon * 0.15), max(2.0, horizon * 0.6), (b,))
    stp = step_family(origin, k, u(st_lo, st_hi, (b, a)), start, dur)

    # Random smooth: M sinusoids per axis.
    m = int(spec.get("components", 3))
    a_lo, a_hi = rng("smooth_amp", 0.0, 300.0)
    rs = random_smooth_family(
        origin,
        k,
        u(a_lo, a_hi, (b, a, m)),
        u(w_lo, w_hi * 2.0, (b, a, m)),
        u(0.0, 2.0 * math.pi, (b, a, m)),
    )

    families = [sp, ln, stp, rs]
    pos_stack = torch.stack([p for p, _ in families], dim=0)  # [F, B, H, A]
    vel_stack = torch.stack([v for _, v in families], dim=0)
    weights = torch.tensor(
        [float(w) for w in spec.get("weights", (1.0, 1.0, 1.0, 1.0))],
        device=dev,
        dtype=torch.float32,
    )
    fam = torch.multinomial(weights, b, replacement=True, generator=gen)  # [B]
    idx = torch.arange(b, device=dev)

    return pos_stack[fam, idx], vel_stack[fam, idx]
