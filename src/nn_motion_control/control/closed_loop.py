"""
Closed-loop evaluation: run a control policy inside the plant and score the
tracking.

The plant advances one step per call given the DAC a policy produces from
the current state (see Plant.closed_loop_rollout). This module supplies the
pieces around that loop: reference-trajectory generators, baseline policies
to sanity-check the harness (before any learned controller), and tracking
metrics.

Everything runs at the plant's rate, one control step per plant transition.
The 200 kHz-vs-20 kHz servo sub-stepping is a deliberate later step, starting
1:1 keeps the number of unvalidated assumptions minimal while the loop is
brought up.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from nn_motion_control.eval.tables import PERCENTILES

# A policy maps the current physical (position, velocity, reference), each
# [B, A], to a physical DAC command [B, A]. Matches
# Plant.closed_loop_rollout's expectation.


def zero_policy(
    position: torch.Tensor,
    velocity: torch.Tensor,
    reference: torch.Tensor,
    reference_velocity: torch.Tensor,
) -> torch.Tensor:
    """Baseline: no command (open circuit), the plant drifts on its own."""

    del velocity, reference, reference_velocity

    return torch.zeros_like(position)


@dataclass
class PDPolicy:
    """
    A per-axis PD regulator with optional velocity feedforward:
    dac = kp * (reference - position) - kd * velocity
        + kv * reference_velocity.

    Not a serious controller, a sanity baseline. With a plant where the DAC
    drives motion, a positive kp pulls the position toward the reference, kd
    damps it, and kv feeds the demanded velocity forward (helps on moving
    trajectories). kp/kd/kv are scalars or per-axis [A] tensors.
    """

    kp: float | torch.Tensor
    kd: float | torch.Tensor = 0.0
    kv: float | torch.Tensor = 0.0

    def __call__(
        self,
        position: torch.Tensor,
        velocity: torch.Tensor,
        reference: torch.Tensor,
        reference_velocity: torch.Tensor,
    ) -> torch.Tensor:
        return (
            self.kp * (reference - position)
            - self.kd * velocity
            + self.kv * reference_velocity
        )


def constant_reference(
    target: torch.Tensor, horizon: int, batch: int | None = None
) -> torch.Tensor:
    """
    A held-target reference [B, H, A] from a per-axis target [A] or [B, A].

    A constant target is the regulation / step-response case: hold (or step
    to) a fixed position for horizon steps.
    """

    if target.dim() == 1:
        if batch is None:
            raise ValueError("Pass batch when target is [A]")
        target = target.unsqueeze(0).expand(batch, -1)

    return target.unsqueeze(1).expand(-1, horizon, -1).contiguous()


def tracking_metrics(
    positions: torch.Tensor, reference: torch.Tensor
) -> dict[str, torch.Tensor]:
    """
    Per-axis tracking summary from a closed-loop rollout.

    positions/reference are [B, H, A] (physical). Returns per-axis [A]
    tensors: root-mean-square tracking error over the whole horizon, and the
    mean absolute error at the final step (a coarse settling proxy).
    """

    err = positions - reference  # [B, H, A]
    rms = err.pow(2).mean(dim=(0, 1)).sqrt()  # [A]
    final = err[:, -1, :].abs().mean(dim=0)  # [A]

    return {"rms": rms, "final_abs": final}


def tracking_percentiles(
    positions: torch.Tensor,
    reference: torch.Tensor,
    pcts: tuple[float, ...] = PERCENTILES,
) -> dict[float, torch.Tensor]:
    """
    Per-axis percentiles of the per-rollout RMS tracking error.

    positions/reference are [B, H, A]. Each rollout's RMS error over the
    horizon is taken, then percentiles across the batch per axis. The
    default is a latency-style five-number summary (P50/P95/P99/P99.9/max):
    P50-P99 close means predictable, P99.9 exposes the tail and max the
    worst case. Returns {pct: [A]}.
    """

    per_rollout_rms = (
        (positions - reference).pow(2).mean(dim=1).sqrt()
    )  # [B, A]
    q = torch.tensor(
        [p / 100.0 for p in pcts],
        device=positions.device,
        dtype=positions.dtype,
    )
    out = torch.quantile(per_rollout_rms, q, dim=0)  # [len(pcts), A]

    return {p: out[i] for i, p in enumerate(pcts)}


def _batched(vec: torch.Tensor, batch: int | None) -> torch.Tensor:
    """Broadcast a per-axis [A] vector to [B, A] ([B, A] passes through)."""

    if vec.dim() == 1:
        if batch is None:
            raise ValueError("Pass batch when the origin/center is [A]")

        return vec.unsqueeze(0).expand(batch, -1)

    return vec


def step_reference(
    origin: torch.Tensor,
    amplitude: float | torch.Tensor,
    horizon: int,
    step_at: int = 0,
    batch: int | None = None,
) -> torch.Tensor:
    """
    Hold origin then step to origin + amplitude at step_at.

    origin is [A] (needs batch) or [B, A], amplitude is a scalar or
    per-axis [A]. Returns a [B, H, A] reference, the step-response case.
    """

    origin = _batched(origin, batch)
    amp = torch.as_tensor(amplitude, dtype=origin.dtype, device=origin.device)
    b, a = origin.shape
    traj = origin.unsqueeze(1).expand(b, horizon, a).clone()
    if step_at < horizon:
        traj[:, step_at:, :] = (origin + amp).unsqueeze(1)

    return traj


def settling_time(
    positions: torch.Tensor,
    reference: torch.Tensor,
    tol: float,
    dt: float = 1.0,
) -> torch.Tensor:
    """
    Per-axis time (steps times dt) after which every sample stays within tol.

    positions/reference are [B, H, A]. An axis that never settles (some
    sample outside tol at the final step) returns inf.
    """

    within = ((positions - reference).abs() <= tol).all(dim=0)  # [H, A]
    horizon, axes = within.shape
    # sustained[k, a] is True when the band holds from step k to the end.
    flipped = torch.flip(within.to(torch.int64), dims=[0])
    sustained = torch.flip(
        torch.cumprod(flipped, dim=0), dims=[0]
    ).bool()  # [H, A]
    idx = torch.arange(horizon, device=positions.device).unsqueeze(1)  # [H, 1]
    sentinel = torch.full((horizon, axes), horizon, device=positions.device)
    first = torch.where(sustained, idx, sentinel).min(dim=0).values  # [A]
    settle = first.to(torch.float32) * dt
    settle[first == horizon] = float("inf")

    return settle


def overshoot(
    positions: torch.Tensor,
    reference: torch.Tensor,
    origin: torch.Tensor,
    batch: int | None = None,
) -> torch.Tensor:
    """
    Per-axis peak excursion past the target, as a fraction of the
    commanded step.

    The step is reference[:, -1] - origin, overshoot is the largest move
    beyond the target in the step's direction over the horizon. Axes with
    no commanded step (zero amplitude) report zero. positions/reference are
    [B, H, A].
    """

    origin = _batched(origin, batch)
    target = reference[:, -1, :]  # [B, A]
    step = target - origin  # [B, A]
    beyond = (positions - target.unsqueeze(1)) * torch.sign(step).unsqueeze(1)
    peak = beyond.clamp_min(0.0).amax(dim=1)  # [B, A]
    frac = peak / step.abs().clamp_min(1e-12)
    frac = torch.where(step.abs() < 1e-9, torch.zeros_like(frac), frac)

    return frac.mean(dim=0)  # [A]


def disturbance_response(
    positions: torch.Tensor,
    reference: torch.Tensor,
    disturb_step: int,
    tol: float,
    dt: float = 1.0,
) -> dict[str, torch.Tensor]:
    """
    Per-axis peak error after a disturbance and the time to re-settle
    within tol.

    Pure scorer over a rollout in which the caller has injected the
    disturbance at disturb_step (e.g. an impulse on the DAC or a jump in
    the reference). Returns {"peak": [A], "recovery": [A]}, recovery is
    measured from disturb_step.
    """

    post_err = (positions - reference).abs()[:, disturb_step:, :]  # [B, H', A]
    peak = post_err.amax(dim=(0, 1))  # [A]
    recovery = settling_time(
        positions[:, disturb_step:, :], reference[:, disturb_step:, :], tol, dt
    )

    return {"peak": peak, "recovery": recovery}
