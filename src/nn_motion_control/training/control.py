"""
Policy-gradient controller training through a frozen differentiable plant.

The controller is trained by rolling the plant forward under the
controller's own DAC commands ('Plant.closed_loop_rollout') and
backpropagating a tracking loss to the controller weights (the
analytic policy gradient). The plant is frozen; only the controller
learns. The shared AMP / accumulation / early-stopping loop comes from
the base 'Trainer' -- this module supplies the objective and the
per-batch rollout.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F  # noqa: N812  (conventional alias)

from nn_motion_control.control.config import (
    ControllerConfig,
    save_controller_checkpoint,
)
from nn_motion_control.control.controller import ControllerNet, NNPolicy
from nn_motion_control.control.resource import score_controller
from nn_motion_control.plant.plant import Plant
from nn_motion_control.training.rollout import (
    curriculum_horizon,
    normalise_axis_weights,
)
from nn_motion_control.training.rollout import (
    horizon_weights as horizon_weight_vector,
)
from nn_motion_control.training.trainer import Trainer

# A reference generator maps a physical origin [B, A], a horizon and an
# optional RNG to the PVT demand (position [B, H, A], velocity
# [B, H, A]) the controller tracks. The RNG seeds a reproducible draw
# for validation; training passes None (fresh draws).
ReferenceGen = Callable[
    [torch.Tensor, int, "torch.Generator | None"],
    tuple[torch.Tensor, torch.Tensor],
]


def control_loss(
    positions: torch.Tensor,
    reference: torch.Tensor,
    dacs: torch.Tensor,
    *,
    tracking_weight: float,
    effort_weight: float,
    rate_weight: float,
    velocity_weight: float = 0.0,
    huber_delta: float = 0.0,
    reference_velocity: torch.Tensor | None = None,
    horizon_weights: torch.Tensor | None = None,
    axis_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Closed-loop objective: a config-weighted sum of the control concerns.

    'positions'/'reference'/'dacs' are [B, H, A]. The concerns are:
    position tracking (horizon- and axis-weighted, Huber if
    huber_delta > 0 so divergent rollouts do not dominate), velocity
    tracking (achieved per-step motion vs 'reference_velocity', the
    PVT demand), control effort (command magnitude) and command rate
    (smoothness). Each is scaled by its weight, so a run dials its own
    priorities. 'horizon_weights' ([H]) defaults to a uniform mean
    over horizon.
    """

    if huber_delta > 0:
        err_el = F.huber_loss(
            positions, reference, reduction="none", delta=huber_delta
        )  # [B, H, A]
    else:
        err_el = (positions - reference) ** 2
    if axis_weights is not None:
        err_el = err_el * axis_weights
    per_step = err_el.mean(dim=(0, 2))  # [H]
    if horizon_weights is not None:
        tracking = (horizon_weights * per_step).sum()
    else:
        tracking = per_step.mean()

    effort = dacs.pow(2).mean()
    if dacs.shape[1] > 1:
        rate = (dacs[:, 1:, :] - dacs[:, :-1, :]).pow(2).mean()
    else:
        rate = dacs.new_zeros(())

    velocity = positions.new_zeros(())
    if (
        velocity_weight > 0
        and reference_velocity is not None
        and positions.shape[1] > 1
    ):
        vel_err = (
            positions[:, 1:, :] - positions[:, :-1, :]
        ) - reference_velocity[:, 1:, :]
        vel_sq = vel_err**2
        if axis_weights is not None:
            vel_sq = vel_sq * axis_weights
        velocity = vel_sq.mean()

    return (
        tracking_weight * tracking
        + velocity_weight * velocity
        + effort_weight * effort
        + rate_weight * rate
    )


class ControlTrainer(Trainer):
    """
    Train a controller by policy gradient through a frozen plant.

    'model' (passed to the base) is the controller, so the optimiser,
    gradient clip and checkpoint all target it; the plant is frozen
    and held separately. Each batch's warmup window seeds a
    closed-loop rollout against a generated reference; the horizon
    grows on a curriculum. 'controller_config' may be None for tests
    that only touch '_forward_loss' (saving a checkpoint needs it).
    """

    def __init__(
        self,
        plant: Plant,
        controller: ControllerNet,
        policy: NNPolicy,
        controller_config: ControllerConfig | None,
        reference_gen: ReferenceGen,
        *,
        max_horizon: int,
        curriculum_start: int = 8,
        curriculum_ramp: int = 20,
        tracking_weight: float = 1.0,
        effort_weight: float = 1e-3,
        rate_weight: float = 1e-3,
        velocity_weight: float = 0.0,
        huber_delta: float = 0.0,
        axis_weights: list[float] | None = None,
        hw_mode: str = "uniform",
        **trainer_kwargs,
    ):
        super().__init__(model=controller, **trainer_kwargs)

        # Freeze the plant: it is the simulator, not a learner.
        plant.model.eval()

        for param in plant.model.parameters():
            param.requires_grad_(False)

        self.plant = plant
        self.policy = policy
        self.controller_config = controller_config
        self.reference_gen = reference_gen
        self.max_horizon = max_horizon
        self.curriculum_start = curriculum_start
        self.curriculum_ramp = curriculum_ramp
        self.tracking_weight = tracking_weight
        self.effort_weight = effort_weight
        self.rate_weight = rate_weight
        self.velocity_weight = velocity_weight
        self.huber_delta = huber_delta
        # Normalise to mean 1 (like RolloutTrainer) so a focus vector
        # such as [3, 1, 1] rebalances axes without changing the
        # overall tracking-loss magnitude the early-stopping threshold
        # sees.
        self.axis_weights = (
            normalise_axis_weights(axis_weights, self.device)
            if axis_weights is not None
            else None
        )
        self.hw_mode = hw_mode

        # Reproducible validation reference draws: reseeded once per
        # validation epoch (in _validate_epoch), not per batch, so the
        # batches within an epoch stay diverse while the metric
        # remains comparable across epochs.
        self._val_gen: torch.Generator | None = None

        self._cur_h = curriculum_start
        self._weights = horizon_weight_vector(
            curriculum_start, hw_mode, device=self.device
        )
        self._resource_report = (
            score_controller(
                controller.resource_layers(), controller_config.servo_rate_hz
            ).as_dict()
            if controller_config is not None
            else None
        )

    def _on_epoch_start(self, epoch: int) -> None:
        self._cur_h = curriculum_horizon(
            epoch, self.curriculum_start, self.max_horizon, self.curriculum_ramp
        )
        self._weights = horizon_weight_vector(
            self._cur_h, self.hw_mode, device=self.device
        )

    def _forward_loss(self, batch):
        warmup = batch[0].to(self.device, non_blocking=True)
        # The reference is a target, not a learnable quantity -> no
        # grad from the seed.
        with torch.no_grad():
            origin, _ = self.plant.seed_state(warmup)
        # Training draws fresh random trajectories; validation reuses
        # the per-epoch generator (seeded once in _validate_epoch) so
        # draws are reproducible across epochs yet diverse across
        # batches within an epoch (a no-op for fixed kinds).
        gen = None if self.model.training else self._val_gen
        reference, ref_v = self.reference_gen(origin, self._cur_h, gen)
        positions, dacs = self.plant.closed_loop_rollout(
            warmup,
            reference,
            self.policy,
            self._cur_h,
            reference_velocity=ref_v,
        )
        return control_loss(
            positions,
            reference,
            dacs,
            tracking_weight=self.tracking_weight,
            effort_weight=self.effort_weight,
            rate_weight=self.rate_weight,
            velocity_weight=self.velocity_weight,
            huber_delta=self.huber_delta,
            reference_velocity=ref_v,
            horizon_weights=self._weights,
            axis_weights=self.axis_weights,
        )

    def _validate_epoch(self):
        # Validate at the full horizon so early stopping tracks
        # end-to-end tracking.
        saved_h, saved_w = self._cur_h, self._weights
        self._cur_h = self.max_horizon
        self._weights = horizon_weight_vector(
            self.max_horizon, self.hw_mode, device=self.device
        )
        # Reseed the reference generator once per validation epoch:
        # the batches within this epoch advance the same generator
        # (diverse references) while every epoch sees the same
        # sequence (comparable early-stopping metric).
        self._val_gen = torch.Generator(device=self.device).manual_seed(
            self.seed
        )
        try:
            return super()._validate_epoch()
        finally:
            self._cur_h, self._weights = saved_h, saved_w

    def _save_checkpoint(self):
        if self.controller_config is None:
            raise ValueError(
                "Cannot save a controller checkpoint without a config"
            )
        save_controller_checkpoint(
            self.save_path,
            self.model,
            self.controller_config,
            self._resource_report,
        )
