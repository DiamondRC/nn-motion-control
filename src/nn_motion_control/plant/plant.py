"""
Forward-dynamics plant: wrap a one-step model and roll it forward over a horizon.

The plant predicts the position change ``ΔP``; a rollout reconstructs the state and
feeds it back — ``next_position = current + ΔP`` and ``velocity = the change in fed
position`` (so velocity stays consistent whether a step is free-running or
truth-anchored). Commands are exogenous (the recorded DAC stream). The reconstruction
runs in float32 for numerical stability even when the model computes in bf16.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from nn_motion_control.core.config import RunConfiguration
from nn_motion_control.data.normalize import NormStats
from nn_motion_control.models.builder import JsonModel
from nn_motion_control.plant.rollout import (
    RecurrentStepper,
    RolloutStepper,
    StreamingConvStepper,
    WindowedStepper,
)

# A control policy maps the current physical (position, velocity, reference target),
# each [B, A], to a physical DAC command [B, A]. Kept a plain callable so the plant does
# not depend on the control package.
ControlPolicy = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class RolloutLayout:
    """
    Which input columns hold position / velocity / command, derived per axis.
    """

    pos_cols: list[int]  # position columns (one per axis; the predicted state)
    vel_cols: list[int]  # order-1 derived velocity columns (may be empty)
    dac_cols: list[int]  # command columns (exogenous)
    n_features: int

    @classmethod
    def from_config(cls, config: RunConfiguration) -> RolloutLayout:
        system = config.system
        in_names = config.input_channels
        axes = system.axes
        n_in = len(in_names)

        commands = [n for n in in_names if system.channel(n).kind == "command"]
        if len(commands) != 1:
            raise NotImplementedError(
                f"Rollout supports exactly one command channel, found {commands}"
            )
        if len(config.target_channels) != 1:
            raise NotImplementedError(
                "Rollout supports predicting one state channel (position); "
                f"found targets {config.target_channels}"
            )
        pos_name = config.target_channels[0]
        if system.channel(pos_name).kind != "measured":
            raise NotImplementedError(
                f"Predicted channel '{pos_name}' must be a measured state"
            )
        vel_name = next(
            (
                n
                for n in in_names
                if system.channel(n).kind == "derived"
                and system.channel(n).order == 1
                and system.channel(n).source == pos_name
            ),
            None,
        )

        def cols(name: str) -> list[int]:
            j = in_names.index(name)
            return [a * n_in + j for a in range(len(axes))]

        pos_cols = cols(pos_name)
        dac_cols = cols(commands[0])
        vel_cols = cols(vel_name) if vel_name is not None else []

        covered = set(pos_cols) | set(vel_cols) | set(dac_cols)
        if covered != set(range(n_in * len(axes))):
            raise NotImplementedError(
                "Rollout only supports position/order-1-velocity/command inputs; "
                f"config inputs {in_names} include an unsupported channel"
            )
        return cls(pos_cols, vel_cols, dac_cols, n_in * len(axes))


class Plant:
    """
    A one-step model plus the bookkeeping to roll it forward over a horizon.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        input_stats: NormStats,
        target_stats: NormStats,
        layout: RolloutLayout,
        device: str = "cpu",
        rollout_kind: str = "windowed",
    ):
        self.model = model
        self.layout = layout
        self.device = device
        self.rollout_kind = rollout_kind
        self._im = input_stats.mean.to(device=device, dtype=torch.float32)
        self._istd = input_stats.std.to(device=device, dtype=torch.float32)
        self._tm = target_stats.mean.to(device=device, dtype=torch.float32)
        self._tstd = target_stats.std.to(device=device, dtype=torch.float32)
        self._pos = torch.tensor(layout.pos_cols, device=device)
        self._vel = torch.tensor(layout.vel_cols, device=device)
        self._dac = torch.tensor(layout.dac_cols, device=device)

    @property
    def pos_std(self) -> torch.Tensor:
        """
        Per-axis z-score std of the position inputs, for denormalising position error.
        """

        return self._istd[self._pos]

    @classmethod
    def from_checkpoint(
        cls, config: RunConfiguration, ckpt_path: str, device: str = "cpu"
    ) -> Plant:
        """
        Build a frozen plant from an artifact config and a schema-v2 checkpoint.
        """

        bundle = torch.load(ckpt_path, map_location=device)
        model = JsonModel(config=config).to(device)
        model.load_state_dict(bundle["model_state_dict"])
        model.eval()

        def stats(key: str) -> NormStats:
            s = bundle[key]
            mean = s["mean"]
            return NormStats(
                mean=mean, std=s["std"], normalizable=torch.ones_like(mean)
            )

        rollout_kind = (config.rollout or {}).get("rollout_kind", "windowed")
        return cls(
            model,
            stats("input_stats"),
            stats("target_stats"),
            RolloutLayout.from_config(config),
            device,
            rollout_kind=rollout_kind,
        )

    def make_stepper(self) -> RolloutStepper:
        """
        Build the per-rollout stepper for this plant's ``rollout_kind``.

        The stepper advances ``self.model`` one step at a time; every other rollout
        concern (reconstruction, scheduled sampling, next-frame assembly) stays in
        ``roll_forward``. A fresh stepper per rollout keeps any cached state local.
        """

        if self.rollout_kind == "windowed":
            return WindowedStepper(self.model)
        if self.rollout_kind == "streaming":
            return StreamingConvStepper(self.model)
        if self.rollout_kind == "recurrent":
            return RecurrentStepper(self.model)
        raise ValueError(f"Unknown rollout_kind: {self.rollout_kind!r}")

    def roll_forward(
        self,
        warmup: torch.Tensor,
        dac_future: torch.Tensor,
        horizon: int,
        teacher_pos: torch.Tensor | None = None,
        ss_prob: float = 0.0,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """
        Roll the plant ``horizon`` steps, returning normalised predicted positions.

        ``warmup`` ``[B, F, W]`` seeds the window; ``dac_future`` ``[B, H, A]`` is the
        recorded command stream; ``teacher_pos`` ``[B, H, A]`` (with ``ss_prob > 0``)
        enables scheduled sampling — each step, with probability ``ss_prob``, the truth
        is fed forward instead of the prediction. Returns ``[B, H, A]``.

        This is the architecture-agnostic reconstruction loop: model advancement is
        delegated to a ``RolloutStepper`` (``reset``/``step``), so a streaming or
        recurrent model differs only in the stepper, not the physical bookkeeping.
        """

        pos, vel, dac = self._pos, self._vel, self._dac
        tm, tstd = self._tm, self._tstd
        # Gather the per-axis normalisation constants once; they are constant over the
        # rollout, so indexing them per step is wasted work.
        im_pos, istd_pos = self._im[pos], self._istd[pos]
        im_vel, istd_vel = self._im[vel], self._istd[vel]
        # Reconstruction runs in float32 for stability even under a bf16 autocast, so
        # cast the (possibly bf16) loader tensors up front to match the state window.
        warmup = warmup.float()
        dac_future = dac_future.float()
        if teacher_pos is not None:
            teacher_pos = teacher_pos.float()
        b = warmup.shape[0]
        cur_p = warmup[:, pos, -1] * istd_pos + im_pos  # physical current position

        stepper = self.make_stepper()
        raw_delta = stepper.reset(warmup)
        preds = []
        for k in range(horizon):
            d_p = raw_delta.float() * tstd + tm  # physical ΔP  [B, A]
            pred_p = cur_p + d_p  # model's next position
            preds.append((pred_p - im_pos) / istd_pos)  # store normalised (for loss)

            fed_p = pred_p
            if teacher_pos is not None and ss_prob > 0:
                truth_p = teacher_pos[:, k, :] * istd_pos + im_pos
                anchor = torch.rand(b, 1, device=warmup.device, generator=generator)
                fed_p = torch.where(anchor < ss_prob, truth_p, pred_p)

            fed_v = fed_p - cur_p  # velocity = change in fed position (consistent)
            new_col = warmup.new_zeros(b, self.layout.n_features)
            new_col[:, pos] = (fed_p - im_pos) / istd_pos
            if vel.numel():
                new_col[:, vel] = (fed_v - im_vel) / istd_vel
            new_col[:, dac] = dac_future[:, k, :]
            cur_p = fed_p

            if k < horizon - 1:
                raw_delta = stepper.step(new_col)

        return torch.stack(preds, dim=1)  # [B, H, A]

    def closed_loop_rollout(
        self,
        warmup: torch.Tensor,
        reference: torch.Tensor,
        policy: ControlPolicy,
        horizon: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Free-run the plant with the DAC produced by ``policy`` each step (closed loop).

        ``warmup`` ``[B, F, W]`` seeds the window; ``reference`` ``[B, H, A]`` is the
        target position trajectory (physical units). Each step the policy sees the
        current physical ``(position, velocity, reference)`` and returns a physical DAC
        command ``[B, A]``; the plant predicts the next position from a window whose
        last frame holds that command. Returns ``(positions, dacs)`` in physical units,
        both ``[B, H, A]``. Differentiable through the plant, so a policy trains by
        rolling it out and backpropagating a tracking loss (analytic policy gradient).

        This runs at the plant's own rate (one control step per plant transition); the
        200 kHz-vs-20 kHz servo sub-stepping ("ten chances") is a deliberate later step.
        """

        pos, vel, dac = self._pos, self._vel, self._dac
        tm, tstd = self._tm, self._tstd
        im_pos, istd_pos = self._im[pos], self._istd[pos]
        im_vel, istd_vel = self._im[vel], self._istd[vel]
        im_dac, istd_dac = self._im[dac], self._istd[dac]
        window = warmup.float()
        reference = reference.float()
        b = window.shape[0]
        cur_p = window[:, pos, -1] * istd_pos + im_pos  # physical position
        if vel.numel():
            cur_v = window[:, vel, -1] * istd_vel + im_vel
        else:
            cur_v = torch.zeros_like(cur_p)

        positions, dacs = [], []
        for k in range(horizon):
            dac_phys = policy(cur_p, cur_v, reference[:, k, :])  # [B, A]
            # Carry the command in the current (last) window frame, then predict.
            dac_norm = (dac_phys - im_dac) / istd_dac
            last = window[:, :, -1].index_copy(1, dac, dac_norm)
            win = torch.cat([window[:, :, :-1], last.unsqueeze(-1)], dim=2)
            d_p = self.model(win).float() * tstd + tm
            next_p = cur_p + d_p
            positions.append(next_p)
            dacs.append(dac_phys)

            # Advance one step: slide in the next frame (its DAC is set next iteration).
            next_v = next_p - cur_p
            new_col = window.new_zeros(b, self.layout.n_features)
            new_col[:, pos] = (next_p - im_pos) / istd_pos
            if vel.numel():
                new_col[:, vel] = (next_v - im_vel) / istd_vel
            window = torch.cat([win[:, :, 1:], new_col.unsqueeze(-1)], dim=2)
            cur_p, cur_v = next_p, next_v

        return torch.stack(positions, dim=1), torch.stack(dacs, dim=1)
