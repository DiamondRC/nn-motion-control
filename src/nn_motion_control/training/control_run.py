"""
Orchestrate a controller training run: frozen plant plus
policy-gradient trainer.

Loads the referenced plant checkpoint (frozen), sources warmup
windows from the plant's dataset, builds the quantised controller and
its policy, trains it closed-loop against a generated reference, and
logs the FPGA resource cost of the result.
"""

from __future__ import annotations

import logging
import math
import os
import time
from datetime import datetime

import torch

from nn_motion_control.control.closed_loop import step_reference
from nn_motion_control.control.config import (
    ControllerConfig,
    build_controller_net,
    build_policy,
)
from nn_motion_control.control.resource import score_controller
from nn_motion_control.control.trajectories import (
    build_family,
    morph_family,
    sample_mixed_reference,
    sequence_reference,
    spiral_family,
)
from nn_motion_control.core.config import RunConfiguration
from nn_motion_control.core.seeding import seed_everything
from nn_motion_control.plant.plant import (
    Plant,
    RolloutLayout,
    rollout_splits_from_config,
)
from nn_motion_control.training.control import ControlTrainer
from nn_motion_control.training.logging_setup import ModelLogger
from nn_motion_control.training.trainer import config_overrides

# Rollout horizon (steps) used when the config omits 'horizon'.
DEFAULT_HORIZON = 32


def make_reference_gen(spec: dict, device: str):
    """
    Build a PVT reference generator: (origin, horizon,
    generator=None) -> (pos, vel).

    Both are [B, H, A]; velocity is the demand's analytic derivative.
    'generator' (used only by the 'mixed' kind) seeds a reproducible
    draw for validation; the deterministic single-trajectory kinds
    ignore it.
    """

    kind = spec.get("kind", "step")
    if kind == "step":
        amplitude = spec.get("amplitude", 0.0)
        step_at = int(spec.get("step_at", 0))
        amp: float | torch.Tensor
        if isinstance(amplitude, (list, tuple)):
            amp = torch.tensor(amplitude, dtype=torch.float32, device=device)
        else:
            amp = float(amplitude)

        def step_gen(origin, horizon, generator=None):
            pos = step_reference(origin, amp, horizon, step_at)
            return pos, torch.zeros_like(pos)

        return step_gen
    if kind == "spiral":
        radius = float(spec["radius"])
        angular = float(spec.get("angular_step", 0.1))
        z_rate = float(spec.get("z_rate", 0.0))
        xy = tuple(int(v) for v in spec.get("xy", (0, 1)))
        tilt = spec.get("tilt")
        radius_end = spec.get("radius_end")

        def spiral_gen(origin, horizon, generator=None):
            b = origin.shape[0]
            dev, dt = origin.device, origin.dtype
            k = torch.arange(horizon, device=dev, dtype=dt)

            def col(value):
                return torch.full((b,), value, device=dev, dtype=dt)

            tilt_x = tilt_y = None
            if tilt is not None:
                if isinstance(tilt, (tuple, list)):
                    t_min, t_max = float(tilt[0]), float(tilt[1])
                else:
                    t_min, t_max = 0.0, float(tilt)
                theta = t_min + (t_max - t_min) * torch.rand(
                    1, generator=generator, device=dev, dtype=dt
                )
                azimuth = (2.0 * math.pi) * torch.rand(
                    1, generator=generator, device=dev, dtype=dt
                )
                tilt_x = (theta * torch.sin(azimuth)).expand(b)
                tilt_y = (theta * torch.cos(azimuth)).expand(b)

            r_end = None if radius_end is None else col(float(radius_end))

            return spiral_family(
                origin,
                k,
                col(radius),
                col(angular),
                col(z_rate),
                xy,
                tilt_x=tilt_x,
                tilt_y=tilt_y,
                radius_end=r_end,
            )

        return spiral_gen
    if kind == "mixed":

        def mixed_gen(origin, horizon, generator=None):
            return sample_mixed_reference(origin, horizon, spec, generator)

        return mixed_gen
    if kind in ("line", "helix", "smooth"):

        def single_gen(origin, horizon, generator=None):
            k = torch.arange(horizon, device=origin.device, dtype=origin.dtype)
            return build_family(kind, origin, k, spec, generator)

        return single_gen
    if kind == "morph":
        from_name = str(spec.get("from", "spiral"))
        to_name = str(spec.get("to", "line"))

        def morph_gen(origin, horizon, generator=None):
            return morph_family(
                origin, horizon, from_name, to_name, spec, generator
            )

        return morph_gen
    if kind == "sequence":
        segments = [
            str(s) for s in spec.get("segments", ("spiral", "line", "step"))
        ]
        durations = spec.get("durations")

        def sequence_gen(origin, horizon, generator=None):
            return sequence_reference(
                origin, horizon, segments, spec, generator, durations
            )

        return sequence_gen
    raise ValueError(f"Unknown reference kind: {kind!r}")


# Trajectory shapes the track command can select with --shape; 'config'
# means keep the controller config's own reference block unchanged.
TRACK_SHAPES = (
    "config",
    "spiral",
    "helix",
    "line",
    "step",
    "smooth",
    "mixed",
    "morph",
    "sequence",
)

# Spiral/step/helix shapes for visualisation.
_SPIRAL_VIZ_RADIUS = 1000.0
_SPIRAL_VIZ_RADIUS_END = 0.0
_SPIRAL_VIZ_ANGULAR = 0.05
_SPIRAL_VIZ_TILT = (math.pi / 6, math.pi / 3)
_STEP_VIZ_AMPLITUDE = 800.0
_STEP_VIZ_AT = 8
_HELIX_VIZ_RADIUS = 1000.0
_HELIX_VIZ_ANGULAR = 0.03
_HELIX_VIZ_Z_RATE = 20.0


def shape_spec(shape: str) -> dict:
    """
    Reference spec for a named track shape.

    step and helix are deterministic (seed-independent); the spiral tilts
    to a seed-dependent orientation, and the remaining shapes draw
    randomised params, so all of those vary with the seed.
    """

    presets: dict[str, dict] = {
        "spiral": {
            "kind": "spiral",
            "radius": _SPIRAL_VIZ_RADIUS,
            "radius_end": _SPIRAL_VIZ_RADIUS_END,
            "angular_step": _SPIRAL_VIZ_ANGULAR,
            "tilt": _SPIRAL_VIZ_TILT,
        },
        "step": {
            "kind": "step",
            "amplitude": _STEP_VIZ_AMPLITUDE,
            "step_at": _STEP_VIZ_AT,
        },
        "helix": {
            "kind": "spiral",
            "radius": _HELIX_VIZ_RADIUS,
            "angular_step": _HELIX_VIZ_ANGULAR,
            "z_rate": _HELIX_VIZ_Z_RATE,
        },
        "line": {"kind": "line"},
        "smooth": {"kind": "smooth"},
        "mixed": {"kind": "mixed"},
        "morph": {"kind": "morph", "from": "spiral", "to": "line"},
        "sequence": {
            "kind": "sequence",
            "segments": ["spiral", "line", "step"],
        },
    }
    if shape not in presets:
        raise ValueError(f"Unknown track shape: {shape!r}")

    return presets[shape]


class ControlRun:
    """
    Train a controller by policy gradient through a frozen plant, then
    report cost.
    """

    def __init__(self, controller_cfg_path: str):
        self.config = ControllerConfig(controller_cfg_path)
        self._setup_logging()
        self._set_seed()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        plant = self._load_plant()
        loaders = self._build_loaders()
        # Normalise the controller's raw physical inputs with the
        # plant's fitted stats.
        feat_mean, feat_std = plant.feature_stats(self.config.features)
        net = build_controller_net(self.config, feat_mean, feat_std).to(
            self.device
        )
        policy = build_policy(self.config, net)

        train = self.config.training
        horizon = int(train.get("horizon", DEFAULT_HORIZON))
        reference_gen = make_reference_gen(
            train.get("reference", {}), self.device
        )
        # Loss and curriculum knobs the config may override
        overrides = config_overrides(
            train,
            {
                "curriculum_start": int,
                "curriculum_ramp": int,
                "tracking_weight": float,
                "effort_weight": float,
                "rate_weight": float,
                "velocity_weight": float,
                "huber_delta": float,
                "axis_weights": list,
            },
        )

        trainer = ControlTrainer(
            plant,
            net,
            policy,
            self.config,
            reference_gen,
            max_horizon=horizon,
            **overrides,
            train_loader=loaders.trn_loader,
            val_loader=loaders.val_loader,
            device=self.device,
            scaler_class=self.plant_config.grad_scaler,
            optimizer_class=getattr(torch.optim, self.config.optimiser),
            criterion_class=torch.nn.MSELoss,
            node_info=loaders.node_info,
            max_epochs=self.config.max_epochs,
            learning_rate=self.config.learning_rate,
            min_delta=self.config.min_delta,
            patience=self.config.patience,
            model_name=self.config.model_name,
            save_path=self.config.save_path,
            logging=True,
            accumulation_steps=1,
            # bfloat16 autocast (float32 autocast is unsupported); the
            # plant's own reconstruction stays float32 internally, as
            # in plant rollout training.
            training_dtype=torch.bfloat16,
            window_size=self.plant_config.window_size,
            seed=self.config.seed,
        )
        trainer.train()

        report = score_controller(
            net.resource_layers(), self.config.servo_rate_hz
        )

        for reason in report.reasons:
            self.logger.info("Resource: %s", reason)
        self.logger.info(
            "Controller cost: params=%d, BRAM bits=%d, DSP-cycles=%d, "
            "max rate=%.0f Hz",
            report.params,
            report.bram_bits,
            report.dsp_cycles,
            report.max_servo_rate_hz,
        )
        self.logger.info("Run took %.1fs", time.perf_counter() - self._t0)

    def _setup_logging(self):
        self._t0 = time.perf_counter()
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        ModelLogger(self.config.logging_dir, timestamp)
        self.logger = logging.getLogger(os.path.basename(__file__))
        self.logger.info("Starting controller run: %s", self.config.model_name)

    def _set_seed(self):
        seed_everything(self.config.seed)

    def _load_plant(self) -> Plant:
        self.plant_config = RunConfiguration(self.config.plant_config_path)
        return Plant.from_checkpoint(
            self.plant_config, self.config.plant_checkpoint, self.device
        )

    def _build_loaders(self):
        layout = RolloutLayout.from_config(self.plant_config)
        train = self.config.training
        return rollout_splits_from_config(
            self.plant_config,
            layout,
            max_horizon=int(train.get("horizon", DEFAULT_HORIZON)),
            batch_size=self.config.batch_size,
            seed=self.config.seed,
            device=self.device,
            # Adjacent warmup windows overlap heavily; stride the
            # starts so an epoch is a representative subsample rather
            # than every window.
            train_start_stride=int(train.get("start_stride", 1)),
            val_start_stride=int(train.get("val_start_stride", 1)),
            # Seed the controller from quiescent relaxation holds (its
            # operating start), not the plant-ID excitation transients
            # that fill an arbitrary window.
            quiescent_seed=train.get("seed_quiescence"),
        )
