"""
Rollout-training objective and schedules.
"""

from __future__ import annotations

import logging
import os
from typing import cast

import torch

from nn_motion_control.models.builder import JsonModel
from nn_motion_control.plant.plant import Plant
from nn_motion_control.training.trainer import Trainer

logger = logging.getLogger(os.path.basename(__file__))

# Guards the 1/mse axis-weight calibration against a near-zero mse.
AXIS_WEIGHT_EPS = 1e-12


def linear_schedule(
    epoch: int, start: float, end: float, over_epochs: int
) -> float:
    """
    Linearly interpolate start -> end across over_epochs, then hold end.
    """

    if over_epochs <= 0:
        return end

    frac = min(1.0, max(0.0, epoch / over_epochs))
    return start + (end - start) * frac


def curriculum_horizon(
    epoch: int, start_h: int, max_h: int, ramp_epochs: int
) -> int:
    """
    Grow the rollout horizon start_h -> max_h across ramp_epochs (then hold).
    """

    if ramp_epochs <= 0:
        return max_h

    frac = min(1.0, max(0.0, epoch / ramp_epochs))
    return min(max_h, int(round(start_h + (max_h - start_h) * frac)))


def horizon_weights(
    horizon: int,
    mode: str = "uniform",
    gamma: float = 0.99,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Per-step loss weights over the horizon, normalised to sum to 1.

    'uniform' weights every step equally
    'discount' = gamma**k emphasises the near-term
    'increasing' = k emphasises long-horizon robustness.
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
    row_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Joint rollout loss: horizon-weighted accumulated error plus an
    optional per-step term. 'row_weights' [B] scales each row's error.
    """

    # Position equence loss
    sq = (preds - gt_pos) ** 2  # [B, H, A]

    # Weight by axis
    if axis_weights is not None:
        sq = sq * axis_weights
    rows = None if row_weights is None else row_weights[:, None, None]
    if rows is not None:
        sq = sq * rows

    per_step = sq.mean(dim=(0, 2))  # [H]
    loss = (weights * per_step).sum()

    # Match the velocity shape of the tarjectory
    if step_weight > 0:
        prev_pred = torch.cat([seed_last.unsqueeze(1), preds[:, :-1, :]], dim=1)
        prev_gt = torch.cat([seed_last.unsqueeze(1), gt_pos[:, :-1, :]], dim=1)
        inc_sq = ((preds - prev_pred) - (gt_pos - prev_gt)) ** 2  # [B, H, A]

        if axis_weights is not None:
            inc_sq = inc_sq * axis_weights
        if rows is not None:
            inc_sq = inc_sq * rows
        loss = loss + step_weight * inc_sq.mean()

    return loss


def normalise_axis_weights(
    weights: list[float], device: str = "cpu"
) -> torch.Tensor:
    """
    Per-axis loss weights normalised to mean 1, so the overall loss
    scale is unchanged.

    Use to focus an under-fit axis without changing the loss
    magnitude the early-stopping threshold sees.
    """

    w = torch.tensor(weights, dtype=torch.float32, device=device)
    return w / w.mean()


# Seed windows whose command moves less than this (counts rms) carry no
# motion to scale and are left out of the flatness penalty.
FLATNESS_MIN_RIPPLE = 0.1
# Guards the penalty's normalisation when no row is kept.
FLATNESS_EPS = 1e-12


class SmallSignalFlatness:
    """
    Penalty holding a model's temporal stack flat to small same-axis
    changes, so the linear path alone carries the small-signal response.

    Each seed window's own motion on one random axis (its command,
    position and velocity about their window means) is scaled to 'counts'
    rms of command and added to that axis's columns. The stack's change in
    that axis's output, over the linear path's change, squared, is roughly
    the squared gain error the stack adds to the small-signal response.
    Other axes' outputs are left free, so cross-axis coupling can still be
    learned.
    """

    def __init__(
        self,
        model: JsonModel,
        plant: Plant,
        weight: float,
        counts: float,
        seed: int,
    ):
        linear = getattr(model, "linear_skip", None)
        if not isinstance(linear, torch.nn.Linear):
            raise ValueError("Small_signal_flatness needs a linear_skip model")
        layout = plant.layout
        if not layout.vel_cols:
            raise ValueError("Small_signal_flatness needs velocity columns")
        self.model = model
        self.linear = linear
        self.weight = float(weight)
        self.counts = float(counts)
        dev = plant.device
        # Per axis, its position, velocity and command columns [A, 3].
        self.cols = torch.tensor(
            list(
                zip(
                    layout.pos_cols,
                    layout.vel_cols,
                    layout.dac_cols,
                    strict=True,
                )
            ),
            device=dev,
        )
        self.dac_std = plant.feature_stats(["command"])[1][:, 0].to(dev).float()
        self.generator = torch.Generator(device=dev).manual_seed(seed)

    def __call__(self, warmup: torch.Tensor) -> torch.Tensor:
        w = warmup.float()
        b, _, t = w.shape
        axis = torch.randint(
            len(self.cols), (b,), generator=self.generator, device=w.device
        )
        idx = self.cols[axis][:, :, None].expand(-1, -1, t)  # [B, 3, T]
        own = w.gather(1, idx)
        own = own - own.mean(dim=2, keepdim=True)
        ripple = own[:, 2].std(dim=1) * self.dac_std[axis]  # counts rms
        keep = (ripple > FLATNESS_MIN_RIPPLE).float()
        scale = self.counts / ripple.clamp_min(FLATNESS_MIN_RIPPLE)
        nudge = torch.zeros_like(w).scatter(1, idx, own * scale[:, None, None])
        stack = self.model.stack(w + nudge) - self.model.stack(w)
        lin = nudge.flatten(start_dim=1) @ self.linear.weight.T
        pick = axis[:, None]
        num = (keep * stack.gather(1, pick).squeeze(1) ** 2).sum()
        den = (keep * lin.gather(1, pick).squeeze(1).detach() ** 2).sum()

        return self.weight * num / den.clamp_min(FLATNESS_EPS)


class RolloutTrainer(Trainer):
    """
    Trainer for the multi-step rollout objective.

    Overrides the per-batch objective (roll the plant, joint loss)
    and a per-epoch hook that advances the scheduled-sampling
    probability and the horizon curriculum.
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
        axis_weights: list[float] | None = None,
        small_signal_flatness: dict | None = None,
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
        self._ss_gen = torch.Generator(device=self.device).manual_seed(
            self.seed
        )

        # Per-axis loss weights so a hard axis does not dominate.
        if axis_weights is not None:
            self.axis_weights = normalise_axis_weights(
                axis_weights, self.device
            )
        elif auto_balance:
            self.axis_weights = self._calibrate_axis_weights()
        else:
            self.axis_weights = None

        # Cache the per-horizon weight vectors:
        # the curriculum revisits the same few horizons every epoch,
        # => do not rebuild each batch.
        self._weight_cache: dict[int, torch.Tensor] = {}

        # Validation is graded free-running at the full horizon
        # regardless of the curriculum, so the early-stopping metric
        # tracks the deployment quantity (the error-vs-horizon drift)
        # and is comparable across epochs. Grading validation at the
        # current curriculum H and scheduled-sampling probability
        # instead makes the earliest, shortest-horizon, most
        # teacher-forced epoch look best and saves it.
        self._eval_mode = False

        # The model unwrapped from any compile, for the stack-only passes.
        raw = getattr(plant.model, "_orig_mod", plant.model)
        self.flatness = (
            None
            if small_signal_flatness is None
            else SmallSignalFlatness(
                cast(JsonModel, raw),
                plant,
                weight=small_signal_flatness["weight"],
                counts=small_signal_flatness.get("counts", 1.0),
                seed=self.seed,
            )
        )

    def _calibrate_axis_weights(self) -> torch.Tensor:
        """
        Per-axis weights ~ 1/mse so each axis contributes equally to
        the loss.

        Estimated once from a single validation batch, free-running
        the warm-started plant at the full horizon, then normalised to
        mean 1 (so the overall loss scale is unchanged). Prevents the
        hardest axis from monopolising the objective.
        """

        self.plant.model.eval()
        with torch.no_grad():
            batch = cast(
                tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                next(iter(self.val_loader)),
            )
            warmup, dac_future, gt_pos = batch
            h = self.max_horizon
            preds = self.plant.roll_forward(
                warmup, dac_future[:, :h], h, ss_prob=0.0
            )
            mse_axis = ((preds - gt_pos[:, :h]) ** 2).mean(dim=(0, 1))  # [A]

        self.plant.model.train()
        w = 1.0 / (mse_axis + AXIS_WEIGHT_EPS)
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
        self._cur_ss = linear_schedule(
            epoch, self.ss_start, self.ss_end, self.ss_ramp
        )

    def _validate_epoch(self):
        # Grade validation free-running at the full horizon
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
        warmup, dac_future, gt_pos = batch[:3]
        row_weights = batch[3] if len(batch) > 3 else None

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
        # Last true position of the seed window,
        # for the per-step increment term.
        seed_last = warmup[:, :, -1][:, self.plant.layout.pos_cols].float()
        weights = self._weights_for(h)
        loss = rollout_loss(
            preds,
            gt_h,
            seed_last,
            weights,
            self.step_weight,
            self.axis_weights,
            row_weights,
        )
        # Validation stays the plain rollout error, comparable across runs.
        if self.flatness is not None and not self._eval_mode:
            loss = loss + self.flatness(warmup)

        return loss
