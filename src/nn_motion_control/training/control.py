"""
Policy-gradient controller training through a frozen differentiable plant.

The controller is trained by rolling the plant forward under the
controller's own DAC commands. The plant is frozen, only the controller
learns.

Supplies the objective and the per-batch rollout.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import torch
import torch.nn.functional as F  # noqa: N812

from nn_motion_control.control.config import (
    Controller,
    ControllerConfig,
    ControllerNetwork,
    ControllerPolicy,
    save_controller_checkpoint,
)
from nn_motion_control.control.resource import score_controller
from nn_motion_control.control.workspace import FAMILIES, WorkspaceReference
from nn_motion_control.data.device_loader import RolloutWindowLoader
from nn_motion_control.plant.plant import Plant
from nn_motion_control.training.rollout import (
    curriculum_horizon,
    normalise_axis_weights,
)
from nn_motion_control.training.rollout import (
    horizon_weights as horizon_weight_vector,
)
from nn_motion_control.training.trainer import Trainer

logger = logging.getLogger(__name__)

# A reference generator maps a physical origin [B, A],
# a horizon and an optional RNG to the PVT demand the controller tracks.
#
# The RNG seeds a reproducible draw for validation,
# training passes None (fresh draws).
#
# (position [B, H, A], velocity [B, H, A])

# Trajectory-growth defaults: a perturbation well above the float32 floor yet
# small against the error scale, rerun on a quarter of each batch and judged
# over the second half of the rollout.
DEFAULT_GROWTH_PERTURB = 10.0
DEFAULT_GROWTH_FRACTION = 0.25
DEFAULT_GROWTH_TAIL = 0.5
# Growth is only judged once the rollout spans the loop's slow dynamics:
# over a short horizon even a stable loop has not yet removed a perturbation.
DEFAULT_GROWTH_MIN_HORIZON = 500
GAP_EPS = 1e-12
GROWTH_SEED_OFFSET = 1
# A late gap below this share of the perturbation is the numerical floor.
GROWTH_FLOOR_FRACTION = 0.05
# Cap on a projected log-growth, keeping exp() finite.
MAX_LOG_GROWTH = 20.0
# Validation's per-family report: the error is read over this final share
# of the horizon, and a row past VAL_DIVERGED (position unit) has diverged.
VAL_TAIL = 0.1
VAL_DIVERGED = 1000.0
VAL_PERCENTILE = 0.95
# Command energy above this frequency (Hz) marks a limit cycle or thrash.
VAL_THRASH_HZ = 200.0
# The checkpoint consistency check holds this many validation seeds for
# this many steps, with the in-memory controller and with the saved one.
CHECK_ROWS = 64
CHECK_STEPS = 2000
PERCENT = 100.0

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
    tracking_scale: float = 0.0,
) -> torch.Tensor:
    """
    Closed-loop objective: a config-weighted sum of the control concerns.

    The concerns are:
     - Position tracking: horizon-and axis-weighted, Huber if huber_delta > 0
        so divergent rollouts do not dominate. A tracking_scale > 0 (position
        unit) takes log(1 + (error / scale)^2) instead: every tenfold of error
        costs alike at any scale, so nm holds weigh as much as mm moves, and
        the gradient of a large error falls as 1 / error.
     - Velocity tracking: achieved per-step motion vs the reference velocity.
     - Control effort: the command magnitude.
     - Command rate: the smoothness.

    Each is scaled by its weight, so a run dials its own priorities.
    horizon_weights [H] default to a uniform mean over horizon.
    """

    if tracking_scale > 0:
        err_el = torch.log1p(((positions - reference) / tracking_scale) ** 2)
    elif huber_delta > 0:
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


def command_activity(
    dacs: torch.Tensor, rate_hz: float, above_hz: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Per row of 'dacs' [B, T, A]: the command's standard deviation and the
    share of its varying energy above 'above_hz', each the largest across
    axes.
    """

    varying = dacs - dacs.mean(dim=1, keepdim=True)
    power = torch.fft.rfft(varying, dim=1).abs().pow(2)  # [B, F, A]
    freqs = torch.fft.rfftfreq(dacs.shape[1], 1.0 / rate_hz).to(dacs.device)
    high = power[:, freqs > above_hz].sum(dim=1)
    share = high / power.sum(dim=1).clamp_min(GAP_EPS)

    return varying.std(dim=1).amax(dim=-1), share.amax(dim=-1)


def perturbation_growth(
    nominal: torch.Tensor,
    perturbed: torch.Tensor,
    perturb: float,
    tail_fraction: float,
) -> torch.Tensor:
    """
    Per-rollout growth [B] of a perturbation along the closed loop.

    'nominal' and 'perturbed' are [B, H, A] positions from the same
    reference, the perturbed run started 'perturb' (position unit) away. The
    growth is the RMS gap over the last 'tail_fraction' of the horizon over
    the starting gap: below one a disturbance dies away, above one the loop
    amplifies it along the very trajectory it is tracking.
    """

    tail = max(1, int(round(nominal.shape[1] * tail_fraction)))
    gap = (perturbed[:, -tail:] - nominal[:, -tail:]).pow(2).mean(dim=(1, 2))

    # The eps keeps the square root's gradient finite at an exact zero gap.
    return (gap + GAP_EPS).sqrt() / perturb


def projected_growth(
    nominal: torch.Tensor,
    perturbed: torch.Tensor,
    perturb: float,
    project_steps: int,
) -> torch.Tensor:
    """
    Per-rollout growth [B] projected 'project_steps' ahead from the rate
    at which the gap changes between the last two quarters of the horizon.

    A slowly unstable mode can still end a short rollout below the size it
    started at, so the end-over-start growth misses it, the rate does not:
    the gap ratio r over a quarter of the horizon becomes r^(project_steps
    / quarter). Rows whose late gap has decayed into the numerical floor
    (GROWTH_FLOOR_FRACTION of the perturbation) read as no growth.
    """

    q = max(1, nominal.shape[1] // 4)
    gap = (perturbed - nominal).pow(2).mean(dim=2)  # [B, H]
    late = (gap[:, -q:].mean(dim=1) + GAP_EPS).sqrt()
    before = (gap[:, -2 * q : -q].mean(dim=1) + GAP_EPS).sqrt()
    ratio = late / before
    # The exponent is clamped so a fast blow-up stays finite in float32.
    growth = torch.exp(
        (torch.log(ratio) * project_steps / q).clamp(max=MAX_LOG_GROWTH)
    )
    live = late > GROWTH_FLOOR_FRACTION * perturb

    return torch.where(live, growth, torch.zeros_like(growth))


def growth_penalty(growth: torch.Tensor) -> torch.Tensor:
    """
    log(1 + relu(growth - 1)^2): zero while a perturbation dies away, so a
    stable but slow loop is not pushed toward a faster, higher-gain one,
    quadratic once it grows, and only logarithmic once it explodes, so a
    divergent rollout cannot swamp the tracking terms.
    """

    return torch.log1p(F.relu(growth - 1.0).pow(2)).mean()


class ControlTrainer(Trainer):
    """
    Train a controller by policy gradient through a frozen plant.
    """

    def __init__(
        self,
        plant: Plant,
        controller: ControllerNetwork,
        policy: ControllerPolicy,
        controller_config: ControllerConfig | None,
        reference_gen: ReferenceGen,
        *,
        max_horizon: int,
        val_horizon: int | None = None,
        curriculum_start: int = 8,
        curriculum_ramp: int = 20,
        tracking_weight: float = 1.0,
        effort_weight: float = 1e-3,
        rate_weight: float = 1e-3,
        velocity_weight: float = 0.0,
        huber_delta: float = 0.0,
        tracking_scale: float = 0.0,
        axis_weights: list[float] | None = None,
        hw_mode: str = "uniform",
        sensor=None,
        actuator=None,
        growth_weight: float = 0.0,
        growth_perturb: float = DEFAULT_GROWTH_PERTURB,
        growth_fraction: float = DEFAULT_GROWTH_FRACTION,
        growth_tail: float = DEFAULT_GROWTH_TAIL,
        growth_min_horizon: int = DEFAULT_GROWTH_MIN_HORIZON,
        row_error_cap: float | None = None,
        seed_curriculum: tuple[float, int] | None = None,
        row_grad_bound: float | None = None,
        growth_project_steps: int = 0,
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
        # Validation may run past the training horizon: a slowly unstable
        # policy looks excellent over a short rollout, so early stopping
        # needs to see the horizon the controller will be deployed over.
        self.val_horizon = val_horizon or max_horizon
        self.curriculum_start = curriculum_start
        self.curriculum_ramp = curriculum_ramp
        self.tracking_weight = tracking_weight
        self.effort_weight = effort_weight
        self.rate_weight = rate_weight
        self.velocity_weight = velocity_weight
        self.huber_delta = huber_delta
        self.tracking_scale = tracking_scale
        self.sensor = sensor
        self.actuator = actuator
        # Closed-loop stability along the tracked trajectory: a rerun of
        # part of the batch from a slightly offset start, penalised when
        # the offset grows (see perturbation_growth).
        if growth_weight < 0 or growth_perturb <= 0:
            raise ValueError(
                "Growth_weight must be >= 0 and growth_perturb > 0, got "
                f"{growth_weight} and {growth_perturb}"
            )
        if not (0 < growth_fraction <= 1 and 0 < growth_tail <= 1):
            raise ValueError(
                "Growth_fraction and growth_tail must lie in (0, 1], got "
                f"{growth_fraction} and {growth_tail}"
            )
        self.growth_weight = growth_weight
        self.growth_perturb = growth_perturb
        self.growth_fraction = growth_fraction
        self.growth_tail = growth_tail
        self.growth_min_horizon = growth_min_horizon
        # Steps ahead the late gap rate is projected over (0 disables it).
        self.growth_project_steps = growth_project_steps
        if row_error_cap is not None and row_error_cap <= 0:
            raise ValueError(f"Row_error_cap must be > 0, got {row_error_cap}")
        # A row whose largest error over the tail of the rollout passes this
        # (position unit) has diverged: its backpropagated gradient grows
        # with the divergence and would dominate the clipped step, so
        # training leaves it out of the tracking terms (the growth penalty
        # still sees it). The tail, not the whole row, so a late divergence
        # is caught.
        self.row_error_cap = row_error_cap
        # (start radius in the position unit, epochs): training seeds widen
        # from within 'start' of rest to the whole travel, geometrically.
        if seed_curriculum is not None and seed_curriculum[0] <= 0:
            raise ValueError("Seed_curriculum start must be > 0")
        self.seed_curriculum = seed_curriculum
        if row_grad_bound is not None and row_grad_bound <= 0:
            raise ValueError("Row_grad_bound must be > 0")
        self.row_grad_bound = row_grad_bound
        self._seed_offsets: torch.Tensor | None = None
        # Its own stream, so switching the penalty on leaves the training
        # references the same draws as a run without it.
        self._growth_gen = torch.Generator(device=self.device).manual_seed(
            self.seed + GROWTH_SEED_OFFSET
        )
        self._terms = {
            "tracking": torch.zeros((), device=self.device),
            "growth": torch.zeros((), device=self.device),
            "dropped": torch.zeros((), device=self.device),
        }

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

        # Reproducible validation reference draws:
        # reseeded once per validation epoch (in _validate_epoch),
        # not per batch, so the batches within an epoch stay
        # diverse while the metric remains comparable across epochs.
        self._val_gen: torch.Generator | None = None
        self._epoch = 0
        # Validation rows' tail error and family, for the per-family report.
        self._val_rows: list[tuple[torch.Tensor, ...]] = []

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
        self._epoch = epoch
        self._apply_seed_curriculum(epoch)
        set_epoch = getattr(self.reference_gen, "set_epoch", None)
        if set_epoch is not None:
            set_epoch(epoch)
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

        # Training draws fresh random trajectories and validation reuses
        # the per-epoch generator => draws are reproducible across epochs
        # yet diverse across batches within an epoch (no-op for fixed kinds).
        gen = None if self.model.training else self._val_gen
        workspace = (
            self.reference_gen
            if isinstance(self.reference_gen, WorkspaceReference)
            else None
        )
        if workspace is not None:
            # The command already holding the seed sets the headroom the
            # reference may use.
            hold = self.plant.seed_command(warmup)
            reference, ref_v = workspace(origin, self._cur_h, gen, hold=hold)
        else:
            reference, ref_v = self.reference_gen(origin, self._cur_h, gen)
        b = warmup.shape[0]
        growing = (
            self.growth_weight > 0
            and self.model.training
            and self._cur_h >= self.growth_min_horizon
        )
        if growing:
            # The perturbed rows ride in the same rollout as the nominal
            # ones, so no second sequential pass is needed.
            n = max(1, int(round(b * self.growth_fraction)))
            warmup = torch.cat([warmup, self._offset_rows(warmup[:n])])
            reference = torch.cat([reference, reference[:n]])
            ref_v = torch.cat([ref_v, ref_v[:n]])
        positions, dacs = self.plant.closed_loop_rollout(
            warmup,
            reference,
            self.policy,
            self._cur_h,
            reference_velocity=ref_v,
            sensor=self.sensor,
            actuator=self.actuator,
            row_grad_bound=self.row_grad_bound if self.model.training else None,
        )

        rows = self._tracked_rows(positions[:b], reference[:b])
        if not bool(rows.any()):
            if self.model.training:
                self._terms["dropped"] += 1.0
                discard = getattr(self.model, "discard_act_peaks", None)
                if discard is not None:
                    discard()
            # Every row has diverged: a non-finite loss makes the trainer
            # skip the step rather than train on exploding gradients.
            return positions.sum() * float("nan")
        loss = control_loss(
            positions[:b][rows],
            reference[:b][rows],
            dacs[:b][rows],
            tracking_weight=self.tracking_weight,
            effort_weight=self.effort_weight,
            rate_weight=self.rate_weight,
            velocity_weight=self.velocity_weight,
            huber_delta=self.huber_delta,
            tracking_scale=self.tracking_scale,
            reference_velocity=ref_v[:b][rows],
            horizon_weights=self._weights,
            axis_weights=self.axis_weights,
        )
        if self.model.training:
            self._terms["tracking"] += loss.detach().float()
            self._terms["dropped"] += 1.0 - rows.float().mean()
            commit = getattr(self.model, "commit_act_scales", None)
            if commit is not None:
                commit()
        elif workspace is not None and workspace.last_families is not None:
            families = workspace.last_families
            tail = max(1, int(round(VAL_TAIL * positions.shape[1])))
            err = positions[:b, -tail:] - reference[:b, -tail:]
            rms = err.pow(2).mean(dim=1).sqrt().amax(dim=-1)
            std, thrash = command_activity(
                dacs[:b, -tail:], workspace.limits.rate_hz, VAL_THRASH_HZ
            )
            self._val_rows.append(
                (rms.detach(), families[:b], std.detach(), thrash.detach())
            )
        n = positions.shape[0] - b
        # Pairs follow their base row: a diverged pair would otherwise bring
        # back the gradients the row cap drops.
        pair = rows[:n]
        # A training term only: validation keeps scoring tracking alone, so
        # its metric stays comparable across runs.
        if growing and bool(pair.any()):
            growth = perturbation_growth(
                positions[:n][pair],
                positions[b:][pair],
                self.growth_perturb,
                self.growth_tail,
            )
            if self.growth_project_steps > 0:
                growth = torch.maximum(
                    growth,
                    projected_growth(
                        positions[:n][pair],
                        positions[b:][pair],
                        self.growth_perturb,
                        self.growth_project_steps,
                    ),
                )
            term = self.growth_weight * growth_penalty(growth)
            self._terms["growth"] += term.detach().float()
            loss = loss + term

        return loss

    def _tracked_rows(
        self, positions: torch.Tensor, reference: torch.Tensor
    ) -> torch.Tensor:
        """
        Mask [B] of the rows the tracking terms train on: every row in
        validation or without a cap, else those whose largest error over
        the last 'growth_tail' of the rollout stays within the cap.
        """

        keep = torch.ones(
            positions.shape[0], dtype=torch.bool, device=positions.device
        )
        if self.row_error_cap is None or not self.model.training:
            return keep
        tail = max(1, int(round(positions.shape[1] * self.growth_tail)))
        err = (positions[:, -tail:] - reference[:, -tail:]).detach().abs()

        return err.amax(dim=(1, 2)) <= self.row_error_cap

    def _offset_rows(self, warmup: torch.Tensor) -> torch.Tensor:
        """
        'warmup' rows with every axis's position offset by +-'growth_perturb'
        across the whole window, so the velocity history is unchanged.
        """

        pos_cols = torch.as_tensor(
            self.plant.layout.pos_cols, device=warmup.device
        )
        sign = torch.randint(
            0,
            2,
            (warmup.shape[0], len(pos_cols)),
            device=warmup.device,
            generator=self._growth_gen,
        )
        offset = (2 * sign - 1) * self.growth_perturb / self.plant.pos_std
        shifted = warmup.clone()
        shifted[:, pos_cols, :] += offset[:, :, None].to(shifted.dtype)

        return shifted

    def _train_epoch(self):
        for v in self._terms.values():
            v.zero_()
        mean = super()._train_epoch()
        n = max(len(self.train_loader), 1)
        logger.info(
            "Train terms: tracking %.4g, growth %.4g, rows dropped as "
            "diverged %.1f%% (mean per batch, horizon %d, growth judged "
            "from %d)",
            float(self._terms["tracking"].item()) / n,
            float(self._terms["growth"].item()) / n,
            PERCENT * float(self._terms["dropped"].item()) / n,
            self._cur_h,
            self.growth_min_horizon,
        )

        return mean

    def _validate_epoch(self):
        # Validate at the full validation horizon so early stopping
        # tracks end-to-end tracking.
        saved_h, saved_w = self._cur_h, self._weights
        self._cur_h = self.val_horizon
        self._weights = horizon_weight_vector(
            self.val_horizon, self.hw_mode, device=self.device
        )

        # Reseed the reference generator once per validation epoch:
        # the batches within this epoch advance the same generator
        # (diverse references) while every epoch sees the same
        # sequence (comparable early-stopping metric).
        self._val_gen = torch.Generator(device=self.device).manual_seed(
            self.seed
        )

        # Validation always scores the full reference distribution, so the
        # metric stays comparable while an amplitude curriculum widens.
        set_epoch = getattr(self.reference_gen, "set_epoch", None)
        if set_epoch is not None:
            set_epoch(None)
        self._val_rows = []

        try:
            loss = super()._validate_epoch()
            self._log_val_families()

            return loss
        finally:
            self._cur_h, self._weights = saved_h, saved_w
            if set_epoch is not None:
                set_epoch(self._epoch)

    def _log_val_families(self) -> None:
        if not self._val_rows:
            return

        rms, fam, std, thrash = (
            torch.cat([row[k] for row in self._val_rows]).float().cpu()
            for k in range(4)
        )
        logger.info(
            "Validation tail by family: error median / P95 (nm), diverged "
            "rows, command std median (counts), share of command energy "
            "above %g Hz (median):",
            VAL_THRASH_HZ,
        )

        for i, name in enumerate(FAMILIES):
            mask = fam == i
            sel = rms[mask]
            if sel.numel() == 0:
                continue
            finite = sel[torch.isfinite(sel)]
            diverged = int((~torch.isfinite(sel) | (sel > VAL_DIVERGED)).sum())
            med, p95 = (
                (
                    float(finite.median()),
                    float(finite.quantile(VAL_PERCENTILE)),
                )
                if finite.numel()
                else (float("nan"), float("nan"))
            )
            logger.info(
                "  %-7s %5d rows  %10.3g / %10.3g  %3d diverged  "
                "std %7.3g  hf %4.2f",
                name,
                sel.numel(),
                med,
                p95,
                diverged,
                float(std[mask].nanmedian()),
                float(thrash[mask].nanmedian()),
            )

    def _apply_seed_curriculum(self, epoch: int) -> None:
        loader = self.train_loader
        if self.seed_curriculum is None or not isinstance(
            loader, RolloutWindowLoader
        ):
            return

        if self._seed_offsets is None:
            frames = loader.last_frames()[:, :, None]
            pos, _ = self.plant.seed_state(frames)
            self._seed_offsets = pos.abs().amax(dim=1)
        start, epochs = self.seed_curriculum
        widest = float(self._seed_offsets.max())
        frac = min(1.0, epoch / max(epochs, 1))
        radius = start * (max(widest, start) / start) ** frac
        if radius >= widest:
            loader.restrict(None)
        else:
            loader.restrict(self._seed_offsets <= radius)
        logger.info(
            "Seed curriculum: %d of %d training seeds within %.0f of rest",
            len(loader.dataset),
            len(self._seed_offsets),
            radius,
        )

    def calibrate_activations(self) -> None:
        """
        Observe one training batch so the controller's fixed activation
        scales exist before the first validation of a warm start.
        """

        was_training = self.model.training
        self.model.train()

        with torch.no_grad():
            self._forward_loss(next(iter(self.train_loader)))
        self.model.train(was_training)

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
        self._check_saved()

    def _check_saved(self) -> None:
        """
        Hold validation seeds with the in-memory controller and with the
        one just saved, logging both, so a checkpoint that does not behave
        as validated shows at once.
        """

        if not isinstance(self.val_loader, RolloutWindowLoader):
            return

        warmup = self.val_loader.spread_warmups(CHECK_ROWS).to(self.device)
        origin, _ = self.plant.seed_state(warmup)
        hold = origin[:, None, :].expand(-1, CHECK_STEPS, -1).contiguous()
        saved = Controller.from_checkpoint(self.save_path, self.device)
        was_training = self.model.training
        self.model.eval()
        tails = []

        for policy in (self.policy, saved.policy):
            with torch.no_grad():
                pos, _ = self.plant.closed_loop_rollout(
                    warmup,
                    hold,
                    policy,
                    CHECK_STEPS,
                    sensor=self.sensor,
                    actuator=self.actuator,
                )
            tail = max(1, int(round(VAL_TAIL * CHECK_STEPS)))
            err = (pos[:, -tail:] - hold[:, -tail:]).pow(2).mean(1).sqrt()
            tails.append(float(err.amax(dim=-1).median()))
        self.model.train(was_training)
        logger.info(
            "Checkpoint check, hold tail error median: in memory %.3g nm, "
            "saved %.3g nm",
            *tails,
        )
