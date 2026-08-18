"""
Randomised PVT reference trajectories for training a generalist controller.

Real scans are a mix, so the controller must track any reference, not one
spiral. Each family maps a physical origin [B, A] and a horizon to a position
and velocity demand [B, H, A] (velocity in units/step, the per-step change
the policy feeds forward). Every family is anchored so the trajectory starts
on the origin, the controller tracks from its current state, and each sample
draws its own params, so one batch spans many trajectories.

sample_mixed_reference picks a family per sample by weight and returns the
blend. morph_family interpolates one shape into another across the horizon;
sequence_reference concatenates shape segments in time. Pass a seeded
generator for reproducible draws (validation, seeded visualisation), None
for fresh training randomness.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import torch

# Default per-family parameter ranges (physical units: nm, rad/step,
# nm/step); a spec may override any of them by key.
SPIRAL_RADIUS_RANGE = (50.0, 1500.0)
SPIRAL_ANGULAR_RANGE = (0.002, 0.03)
SPIRAL_Z_RATE_RANGE = (-30.0, 30.0)
# A helix forces a non-zero z ramp so the circle climbs; the flat-circle
# (z rate 0) case is already reachable through the spiral family.
HELIX_Z_RATE_RANGE = (10.0, 30.0)
STEP_OFFSET_RANGE = (-1000.0, 1000.0)
SMOOTH_AMP_RANGE = (0.0, 300.0)
SMOOTH_COMPONENTS = 3
# Smooth paths reach twice the spiral angular rate at the top end.
SMOOTH_ANGULAR_SCALE = 2.0
# Step-move timing as fractions of the horizon.
STEP_START_MAX_FRAC = 0.4
STEP_DURATION_MIN_FRAC = 0.15
STEP_DURATION_MAX_FRAC = 0.6
# Floor for a random direction's norm before normalising.
DIRECTION_EPS = 1e-6
# The four randomised families, in mix-weight order.
FAMILY_NAMES = ("spiral", "line", "step", "smooth")


def _u(lo, hi, shape, generator, device, dtype):
    """Uniform [lo, hi) sample of shape on the given device."""

    r = torch.rand(shape, generator=generator, device=device, dtype=dtype)

    return lo + (hi - lo) * r


def _z_axis(n_axes: int) -> int | None:
    """The ramp axis for helical / z-scan motion (axis 2 when it exists)."""

    return 2 if n_axes >= 3 else None


def _tilt_matrix(tilt_x, tilt_y):
    """
    Per-sample rotation Rx(tilt_x) @ Ry(tilt_y) as [B, 3, 3].

    Tilts a planar (x, y) trajectory out of its plane so it sweeps all
    three axes instead of a flat x-y circle.
    """

    cx, sx = torch.cos(tilt_x), torch.sin(tilt_x)
    cy, sy = torch.cos(tilt_y), torch.sin(tilt_y)
    z = torch.zeros_like(tilt_x)
    row0 = torch.stack([cy, z, sy], dim=-1)
    row1 = torch.stack([sx * sy, cx, -sx * cy], dim=-1)
    row2 = torch.stack([-cx * sy, sx, cx * cy], dim=-1)

    return torch.stack([row0, row1, row2], dim=1)


def spiral_family(
    origin,
    k,
    radius,
    angular,
    z_rate,
    xy,
    tilt_x=None,
    tilt_y=None,
    radius_end=None,
):
    """
    Anchored spiral: starts on origin, sweeps angular rad per step.

    The radius sweeps 'radius' -> 'radius_end' across the horizon, so the
    path winds into (or out of) a centre — a real vortex; radius_end None
    keeps the radius constant (a plain circle). z_rate adds a ramp on the
    third axis (a helix / conical vortex). tilt_x/tilt_y (per-sample
    radians) rotate the plane out of x-y so it sweeps all three axes.
    """

    b, a = origin.shape
    h = len(k)
    ang = angular[:, None] * k[None, :]  # [B, H]
    x0, y0 = xy
    cos, sin = torch.cos(ang), torch.sin(ang)
    w = angular[:, None]  # [B, 1]
    r0 = radius[:, None]  # [B, 1] initial radius (anchors the centre)
    # Radius over time and its per-step rate; constant for a plain circle.
    if radius_end is None:
        r = r0
        dr = torch.zeros_like(r0)
    else:
        denom = max(h - 1, 1)
        prog = k[None, :] / denom  # [1, H] in [0, 1]
        r = r0 + (radius_end[:, None] - r0) * prog  # [B, H]
        dr = (radius_end[:, None] - r0) / denom  # [B, 1]
    pos = origin[:, None, :].expand(b, h, a).clone()
    # The centre sits one initial radius from the origin, so step 0 (r=r0,
    # angle 0) lands on the origin regardless of the radius schedule.
    pos[:, :, x0] = origin[:, x0 : x0 + 1] - r0 + r * cos
    pos[:, :, y0] = origin[:, y0 : y0 + 1] + r * sin
    vel = torch.zeros_like(pos)
    # d/dk of (centre + r(k)[cos, sin]): the moving radius adds the dr term.
    vel[:, :, x0] = dr * cos - r * sin * w
    vel[:, :, y0] = dr * sin + r * cos * w
    zc = _z_axis(a)
    if zc is not None:
        pos[:, :, zc] = origin[:, zc : zc + 1] + z_rate[:, None] * k[None, :]
        vel[:, :, zc] = z_rate[:, None].expand(b, h)

    # Rotate the (x, y, z) offset about the anchor so the plane tilts;
    # anchoring holds since the offset is zero at step 0.
    if tilt_x is not None and zc is not None:
        ty = tilt_y if tilt_y is not None else torch.zeros_like(tilt_x)
        rot = _tilt_matrix(tilt_x, ty)
        cols = [x0, y0, zc]
        base = origin[:, None, cols]
        off = pos[:, :, cols] - base
        pos[:, :, cols] = base + torch.einsum("bij,bhj->bhi", rot, off)
        vel[:, :, cols] = torch.einsum("bij,bhj->bhi", rot, vel[:, :, cols])

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


def build_family(
    name: str,
    origin: torch.Tensor,
    k: torch.Tensor,
    spec: Mapping,
    generator: torch.Generator | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Draw one randomised family with params from spec, returning (pos, vel).

    'name' is one of FAMILY_NAMES plus 'helix' (a spiral with a forced
    non-zero z ramp). The per-family ranges and draw order live here so the
    mixer, morph and sequence all share one source of truth.
    """

    b, a = origin.shape
    dev, dt = origin.device, origin.dtype

    def u(lo, hi, shape):
        return _u(lo, hi, shape, generator, dev, dt)

    def rng(key, default):
        r = spec.get(key, default)

        return float(r[0]), float(r[1])

    xy = tuple(int(v) for v in spec.get("xy", (0, 1)))
    horizon = len(k)

    if name in ("spiral", "helix"):
        r_lo, r_hi = rng("radius", SPIRAL_RADIUS_RANGE)
        w_lo, w_hi = rng("angular", SPIRAL_ANGULAR_RANGE)
        z_default = (
            HELIX_Z_RATE_RANGE if name == "helix" else SPIRAL_Z_RATE_RANGE
        )
        zr_lo, zr_hi = rng("z_rate", z_default)
        sign = torch.where(u(0.0, 1.0, (b,)) < 0.5, -1.0, 1.0)

        return spiral_family(
            origin,
            k,
            u(r_lo, r_hi, (b,)),
            sign * u(w_lo, w_hi, (b,)),
            u(zr_lo, zr_hi, (b,)),
            xy,
        )
    if name == "line":
        r_lo, r_hi = rng("radius", SPIRAL_RADIUS_RANGE)
        w_lo, w_hi = rng("angular", SPIRAL_ANGULAR_RANGE)
        d = u(-1.0, 1.0, (b, a))
        d = d / d.norm(dim=1, keepdim=True).clamp_min(DIRECTION_EPS)

        return line_family(
            origin, k, d, u(r_lo, r_hi, (b,)), u(w_lo, w_hi, (b,))
        )
    if name == "step":
        st_lo, st_hi = rng("step", STEP_OFFSET_RANGE)
        start = u(0.0, max(1.0, horizon * STEP_START_MAX_FRAC), (b,))
        dur = u(
            max(1.0, horizon * STEP_DURATION_MIN_FRAC),
            max(2.0, horizon * STEP_DURATION_MAX_FRAC),
            (b,),
        )

        return step_family(origin, k, u(st_lo, st_hi, (b, a)), start, dur)
    if name == "smooth":
        m = int(spec.get("components", SMOOTH_COMPONENTS))
        a_lo, a_hi = rng("smooth_amp", SMOOTH_AMP_RANGE)
        w_lo, w_hi = rng("angular", SPIRAL_ANGULAR_RANGE)

        return random_smooth_family(
            origin,
            k,
            u(a_lo, a_hi, (b, a, m)),
            u(w_lo, w_hi * SMOOTH_ANGULAR_SCALE, (b, a, m)),
            u(0.0, 2.0 * math.pi, (b, a, m)),
        )

    raise ValueError(f"Unknown trajectory family: {name!r}")


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
    b, _ = origin.shape
    dev = origin.device
    k = torch.arange(horizon, device=dev, dtype=origin.dtype)

    families = [
        build_family(name, origin, k, spec, generator) for name in FAMILY_NAMES
    ]
    pos_stack = torch.stack([p for p, _ in families], dim=0)  # [F, B, H, A]
    vel_stack = torch.stack([v for _, v in families], dim=0)
    weights = torch.tensor(
        [float(w) for w in spec.get("weights", (1.0, 1.0, 1.0, 1.0))],
        device=dev,
        dtype=torch.float32,
    )
    fam = torch.multinomial(weights, b, replacement=True, generator=generator)
    idx = torch.arange(b, device=dev)

    return pos_stack[fam, idx], vel_stack[fam, idx]


def morph_family(
    origin: torch.Tensor,
    horizon: int,
    from_name: str,
    to_name: str,
    spec: Mapping,
    generator: torch.Generator | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Interpolate 'from_name' into 'to_name' across the horizon with a
    smoothstep blend, returning (pos, vel).

    Both endpoint families are anchored on the origin, so the morph is too.
    Velocity is analytic: the blended velocities plus the blend's own rate
    times the shape gap (the product-rule cross term).
    """

    dev = origin.device
    k = torch.arange(horizon, device=dev, dtype=origin.dtype)
    pos_a, vel_a = build_family(from_name, origin, k, spec, generator)
    pos_b, vel_b = build_family(to_name, origin, k, spec, generator)
    denom = max(horizon - 1, 1)
    s = (k / denom).clamp(0.0, 1.0)  # [H]
    w = (s * s * (3.0 - 2.0 * s))[None, :, None]
    dw = (6.0 * s * (1.0 - s) / denom)[None, :, None]  # d weight / d step
    pos = (1.0 - w) * pos_a + w * pos_b
    vel = (1.0 - w) * vel_a + w * vel_b + dw * (pos_b - pos_a)

    return pos, vel


def _even_split(total: int, n: int) -> list[int]:
    """Split total into n integer lengths differing by at most one."""

    base, rem = divmod(total, n)

    return [base + (1 if i < rem else 0) for i in range(n)]


def sequence_reference(
    origin: torch.Tensor,
    horizon: int,
    segments: Sequence[str],
    spec: Mapping,
    generator: torch.Generator | None,
    durations: Sequence[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Concatenate family segments in time, returning (pos, vel).

    Each segment is anchored on the previous segment's end position, so
    position is continuous across the seams; velocity may step at a seam
    because each family carries its own initial velocity. durations
    (summing to the horizon) set the per-segment lengths, else the horizon
    splits evenly.
    """

    if not segments:
        raise ValueError("Sequence needs at least one segment")
    if durations is None:
        lengths = _even_split(horizon, len(segments))
    else:
        lengths = [int(d) for d in durations]
        if sum(lengths) != horizon:
            raise ValueError("Sequence durations must sum to the horizon")
    dev = origin.device
    pos_parts, vel_parts = [], []
    anchor = origin

    for name, length in zip(segments, lengths, strict=True):
        if length <= 0:
            continue
        seg_k = torch.arange(length, device=dev, dtype=origin.dtype)
        seg_pos, seg_vel = build_family(name, anchor, seg_k, spec, generator)
        pos_parts.append(seg_pos)
        vel_parts.append(seg_vel)
        anchor = seg_pos[:, -1, :]  # next segment starts where this ended

    if not pos_parts:
        raise ValueError("Sequence produced no segments; check durations")

    return torch.cat(pos_parts, dim=1), torch.cat(vel_parts, dim=1)
