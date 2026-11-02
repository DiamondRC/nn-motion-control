"""
The PandA brett_pid block as a batched closed-loop policy.

A float translation of info/vhdl-panda-motion-controllers/src/pids/
brett_pid/brett_pid.vhd, run once per servo tick in nm: position error,
velocity feedback, an integral that only accumulates while the setpoint
moves, derivative, and velocity, acceleration and position feedforward.
Replayed on a hardware capture it reproduces the recorded output to under a
count while holding, so the same controller can run on the plant model as
the baseline every learned controller is measured against.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path

import torch

# The block's read-back gain names, in the order of the dataclass fields.
_GAIN_KEYS = {
    "kp": "KP_I",
    "kv": "KV_I",
    "ki": "KI_I",
    "kd": "KD_I",
    "kvff": "KVFF_I",
    "kaff": "KAFF_I",
    "kpff0": "KPFF0_I",
    "kpff1": "KPFF1_I",
    "k_tot": "K_TOT_I",
    "dt": "DT_I",
    "dt_inv": "DT_INV_I",
    "max_integral": "MAX_INTEGRAL_I",
    "max_output": "MAX_OUTPUT_I",
}


@dataclass(frozen=True)
class PidGains:
    """
    Gains as the block reads them back (real values, scales applied).
    """

    kp: float
    kv: float
    ki: float
    kd: float
    kvff: float
    kaff: float
    kpff0: float
    kpff1: float
    k_tot: float
    dt: float
    dt_inv: float
    max_integral: float
    max_output: float

    @classmethod
    def from_manifest(cls, path: str | Path) -> PidGains:
        """
        Gains from a janus-motus drive manifest ('pid' -> 'gains').
        """

        gains = json.loads(Path(path).read_text())["pid"]["gains"]
        missing = [k for k in _GAIN_KEYS.values() if k not in gains]
        if missing:
            raise ValueError(f"PID manifest {path} lacks gains {missing}")

        return cls(
            **{f.name: float(gains[_GAIN_KEYS[f.name]]) for f in fields(cls)}
        )


class PandaPID:
    """
    Policy '(position, velocity, reference, reference_velocity) -> command',
    each [B, A] physical, running the block's law on every axis.

    'position_ff_scale' [A] scales the position feedforward per axis, so
    gains tuned on one axis can carry over to axes of a different static
    gain. Commands are rounded to whole counts, as the block outputs them.
    """

    def __init__(self, gains: PidGains, position_ff_scale: torch.Tensor):
        self.g = gains
        self._ff_scale = position_ff_scale
        self._state: dict[str, torch.Tensor] | None = None
        self._primed: torch.Tensor | None = None

    def reset(self, batch: int, device: torch.device | str) -> None:
        self._state = None
        self._primed = None

    def prime(self, command: torch.Tensor) -> None:
        """
        Take over bumplessly: the first output equals 'command' [B, A], the
        integral absorbing what the other terms do not supply.
        """

        self._primed = command

    def __call__(
        self,
        position: torch.Tensor,
        velocity: torch.Tensor,
        reference: torch.Tensor,
        reference_velocity: torch.Tensor,
    ) -> torch.Tensor:
        g = self.g
        err = reference - position
        ff_scale = self._ff_scale.to(position.device, position.dtype)
        if self._state is None:
            zero = torch.zeros_like(position)
            self._state = {
                "pos": position,
                "set": reference,
                "err": err,
                "v_des": zero,
                "integral": zero,
                "sum": zero,
            }
        st = self._state
        vel = (position - st["pos"]) * g.dt_inv
        v_des = (reference - st["set"]) * g.dt_inv
        others = (
            g.kp * err
            - g.kv * vel
            + g.kd * g.dt_inv * (err - st["err"])
            + g.kvff * v_des
            + g.kaff * (v_des - st["v_des"])
            + ff_scale
            * (g.kpff1 * reference + g.kpff0 * reference.abs() * reference)
        )
        integral = st["integral"]
        if self._primed is not None:
            integral = (self._primed / g.k_tot - others).clamp(
                -g.max_integral, g.max_integral
            )
            self._primed = None
        else:
            # The block integrates only while the setpoint moves and its
            # last output was under the cap (a one-sided test, as built).
            live = (st["sum"] < g.max_output) & (v_des != 0)
            stepped = (integral + g.ki * g.dt * err).clamp(
                -g.max_integral, g.max_integral
            )
            integral = torch.where(live, stepped, integral)
        total = others + integral
        command = torch.round(g.k_tot * total).clamp(
            -g.max_output, g.max_output
        )
        self._state = {
            "pos": position,
            "set": reference,
            "err": err,
            "v_des": v_des,
            "integral": integral,
            "sum": total,
        }

        return command
