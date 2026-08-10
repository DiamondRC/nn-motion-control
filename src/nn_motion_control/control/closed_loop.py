"""
Closed-loop evaluation: run a control policy inside the plant and score the tracking.

The plant advances one step per call given the DAC a policy produces from the current
state (see ``Plant.closed_loop_rollout``). This module supplies the pieces around that
loop: reference-trajectory generators, baseline policies to sanity-check the harness
(before any learned controller), and tracking metrics.

Everything runs at the plant's rate — one control step per plant transition. The
200 kHz-vs-20 kHz servo sub-stepping is a deliberate later step; starting 1:1 keeps the
number of unvalidated assumptions minimal while the loop is brought up.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

# A policy maps the current physical (position, velocity, reference), each [B, A], to a
# physical DAC command [B, A]. Matches ``Plant.closed_loop_rollout``'s expectation.


def zero_policy(
    position: torch.Tensor, velocity: torch.Tensor, reference: torch.Tensor
) -> torch.Tensor:
    """Baseline: apply no command (open circuit) — the plant drifts on its own."""

    del velocity, reference
    return torch.zeros_like(position)


@dataclass
class PDPolicy:
    """
    A per-axis PD regulator: ``dac = kp * (reference - position) - kd * velocity``.

    Not a serious controller — a sanity baseline. With a plant where the DAC drives
    motion, a positive ``kp`` should pull the position toward the reference and ``kd``
    should damp it, so tracking error falls over the rollout. ``kp``/``kd`` are scalars
    or per-axis ``[A]`` tensors (physical DAC units per metre / per metre-per-step).
    """

    kp: float | torch.Tensor
    kd: float | torch.Tensor = 0.0

    def __call__(
        self, position: torch.Tensor, velocity: torch.Tensor, reference: torch.Tensor
    ) -> torch.Tensor:
        return self.kp * (reference - position) - self.kd * velocity


def constant_reference(
    target: torch.Tensor, horizon: int, batch: int | None = None
) -> torch.Tensor:
    """
    A held-target reference ``[B, H, A]`` from a per-axis target ``[A]`` or ``[B, A]``.

    A constant target is the regulation / step-response case: hold (or step to) a fixed
    position for ``horizon`` steps.
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

    ``positions``/``reference`` are ``[B, H, A]`` (physical). Returns per-axis ``[A]``
    tensors: root-mean-square tracking error over the whole horizon, and the mean
    absolute error at the final step (a coarse settling proxy).
    """

    err = positions - reference  # [B, H, A]
    rms = err.pow(2).mean(dim=(0, 1)).sqrt()  # [A]
    final = err[:, -1, :].abs().mean(dim=0)  # [A]
    return {"rms": rms, "final_abs": final}
