"""
Rollout-training objective and schedules.

Pure functions the rollout trainer composes: the joint loss (accumulated-position spine
plus an optional per-step increment term), horizon weighting and the epoch schedules
for scheduled sampling and the horizon curriculum. Kept free of model/loop state so they
can be tested in isolation.
"""

from __future__ import annotations

import logging
import os
from typing import cast

import torch

from nn_motion_control.plant.plant import Plant
from nn_motion_control.training.trainer import Trainer

logger = logging.getLogger(os.path.basename(__file__))


def linear_schedule(epoch: int, start: float, end: float, over_epochs: int) -> float:
    """
    Linearly interpolate start -> end across over_epochs, then hold end.
    """

    if over_epochs <= 0:
        return end
    frac = min(1.0, max(0.0, epoch / over_epochs))
    return start + (end - start) * frac


def curriculum_horizon(epoch: int, start_h: int, max_h: int, ramp_epochs: int) -> int:
    """
    Grow the rollout horizon start_h -> max_h across ramp_epochs (then hold).
    """

    if ramp_epochs <= 0:
        return max_h
    frac = min(1.0, max(0.0, epoch / ramp_epochs))
    return min(max_h, int(round(start_h + (max_h - start_h) * frac)))


def horizon_weights(
    horizon: int, mode: str = "uniform", gamma: float = 0.99, device: str = "cpu"
) -> torch.Tensor:
    """
    Per-step loss weights over the horizon, normalised to sum to 1.

    uniform weights every step equally; discount = gamma**k (emphasise
    near-term); increasing = k (emphasise long-horizon robustness).
    """

    k = torch.arange(horizon, device=device, dtype=torch.float32)
    if mode == "uniform":
        w = torch.ones(horizon, device=device)
    elif mode == "discount":
        w = gamma**k
    elif mode == "increasing":
        w = k + 1.0
    else:
        raise ValueError(f"Unknown horizon weighting mode: {mode!r}")
    return w / w.sum()


def rollout_loss(
    preds: torch.Tensor,
    gt_pos: torch.Tensor,
    seed_last: torch.Tensor,
    weights: torch.Tensor,
    step_weight: float = 0.0,
    axis_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Joint rollout loss: horizon-weighted accumulated-position error + optional per-step.

    preds/gt_pos are [B, H, A] (normalised positions); seed_last is
    [B, A], the window's last true position. The accumulated (L_acc) term is the
    per-step position MSE; the per-step (L_step) term matches the step-to-step
    increment (the local velocity profile) and is added with step_weight.

    axis_weights [A] (normalised to mean 1 by the caller) re-balances the per-axis
    contributions so a hard axis does not dominate the objective; None weights equally.
    """

    sq = (preds - gt_pos) ** 2  # [B, H, A]
    if axis_weights is not None:
        sq = sq * axis_weights
    per_step = sq.mean(dim=(0, 2))  # [H]
    loss = (weights * per_step).sum()

    if step_weight > 0:
        prev_pred = torch.cat([seed_last.unsqueeze(1), preds[:, :-1, :]], dim=1)
        prev_gt = torch.cat([seed_last.unsqueeze(1), gt_pos[:, :-1, :]], dim=1)
        inc_sq = ((preds - prev_pred) - (gt_pos - prev_gt)) ** 2  # [B, H, A]
        if axis_weights is not None:
            inc_sq = inc_sq * axis_weights
        loss = loss + step_weight * inc_sq.mean()

    return loss


class RolloutTrainer(Trainer):
    """
    Trainer for the multi-step rollout objective.

    Reuses the base AMP / gradient-accumulation / early-stopping loop and only overrides
    the per-batch objective (roll the plant, joint loss) and a per-epoch hook that
    advances the scheduled-sampling probability and the horizon curriculum. The rollout
    loader yields (warmup, dac_future, gt_pos) already on the device; the curriculum
    slices the first horizon(t) steps from tensors built at the maximum horizon.
    """

    def __init__(
        self,
        plant: Plant,
        *,
        max_horizon: int,
        curriculum_start: int = 4,
        curriculum_ramp: int = 20,
        ss_start: float = 1.0,
        ss_end: float = 0.0,
        ss_ramp: int = 30,
        step_weight: float = 0.1,
        hw_mode: str = "uniform",
        auto_balance: bool = False,
        **trainer_kwargs,
    ):
        super().__init__(model=plant.model, **trainer_kwargs)
        self.plant = plant
        self.max_horizon = max_horizon
        self.curriculum_start = curriculum_start
        self.curriculum_ramp = curriculum_ramp
        self.ss_start, self.ss_end, self.ss_ramp = ss_start, ss_end, ss_ramp
        self.step_weight = step_weight
        self.hw_mode = hw_mode
        self._cur_h = max_horizon
        self._cur_ss = 0.0
        self._ss_gen = torch.Generator(device=self.device).manual_seed(self.seed)
        # Per-axis loss weights so a hard axis does not dominate; calibrated once from
        # the warm-started plant's per-axis free-run error (None -> equal weighting).
        self.axis_weights = self._calibrate_axis_weights() if auto_balance else None
        # Cache the per-horizon weight vectors; the curriculum revisits the same few
        # horizons every epoch, so there is no need to rebuild them each batch.
        self._weight_cache: dict[int, torch.Tensor] = {}
        # Validation is graded free-running at the full horizon regardless of the
        # curriculum, so the early-stopping metric tracks the deployment quantity (the
        # error-vs-horizon drift) and is comparable across epochs. Grading validation at
        # the current curriculum H and scheduled-sampling probability instead makes the
        # earliest, shortest-horizon, most teacher-forced epoch look best and saves it.
        self._eval_mode = False

    def _calibrate_axis_weights(self) -> torch.Tensor:
        """
        Per-axis weights ~ 1/mse so each axis contributes equally to the loss.

        Estimated once from a single validation batch, free-running the warm-started
        plant at the full horizon, then normalised to mean 1 (so the overall loss scale
        is unchanged). Prevents the hardest axis from monopolising the objective.
        """

        self.plant.model.eval()
        with torch.no_grad():
            batch = cast(
                tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                next(iter(self.val_loader)),
            )
            warmup, dac_future, gt_pos = batch
            h = self.max_horizon
            preds = self.plant.roll_forward(warmup, dac_future[:, :h], h, ss_prob=0.0)
            mse_axis = ((preds - gt_pos[:, :h]) ** 2).mean(dim=(0, 1))  # [A]
        self.plant.model.train()
        w = 1.0 / (mse_axis + 1e-12)
        w = w / w.mean()
        logger.info(
            "Auto-balanced axis weights (1/mse): %s",
            [round(float(x), 3) for x in w],
        )
        return w

    def _on_epoch_start(self, epoch: int) -> None:
        self._cur_h = curriculum_horizon(
            epoch, self.curriculum_start, self.max_horizon, self.curriculum_ramp
        )
        self._cur_ss = linear_schedule(epoch, self.ss_start, self.ss_end, self.ss_ramp)

    def _validate_epoch(self):
        # Grade validation free-running at the full horizon (see _eval_mode note above).
        self._eval_mode = True
        try:
            return super()._validate_epoch()
        finally:
            self._eval_mode = False

    def _weights_for(self, h: int) -> torch.Tensor:
        w = self._weight_cache.get(h)
        if w is None:
            w = horizon_weights(h, self.hw_mode, device=self.device)
            self._weight_cache[h] = w
        return w

    def _forward_loss(self, batch):
        warmup, dac_future, gt_pos = batch
        if self._eval_mode:
            h, ss_prob = self.max_horizon, 0.0
        else:
            h, ss_prob = self._cur_h, self._cur_ss
        dac_h, gt_h = dac_future[:, :h], gt_pos[:, :h]
        preds = self.plant.roll_forward(
            warmup,
            dac_h,
            h,
            teacher_pos=gt_h,
            ss_prob=ss_prob,
            generator=self._ss_gen,
        )
        # last true position of the seed window, for the per-step increment term
        seed_last = warmup[:, :, -1][:, self.plant.layout.pos_cols].float()
        weights = self._weights_for(h)
        return rollout_loss(
            preds, gt_h, seed_last, weights, self.step_weight, self.axis_weights
        )
