"""
Orchestrate a controller training run: frozen plant plus
policy-gradient trainer.

Loads the referenced plant checkpoint (frozen), sources warmup
windows from the plant's dataset, builds the quantised controller and
its policy, trains it closed-loop against a generated reference, and
logs the FPGA resource cost of the result.
"""

from __future__ import annotations

import glob
import logging
import math
import os
import time
from datetime import datetime

import torch

from nn_motion_control.control.closed_loop import step_reference
from nn_motion_control.control.config import (
    ControllerConfig,
    build_actuator,
    build_controller_net,
    build_policy,
    build_sensor,
)
from nn_motion_control.control.pid import PidGains
from nn_motion_control.control.resource import score_controller
from nn_motion_control.control.trajectories import (
    build_family,
    morph_family,
    sample_mixed_reference,
    sequence_reference,
    spiral_family,
)
from nn_motion_control.control.workspace import (
    DEFAULT_MARGIN,
    HOLD_FRACTION,
    StageModel,
    WorkspaceLimits,
    WorkspaceReference,
)
from nn_motion_control.core.checkpoints import load_checkpoint_bundle
from nn_motion_control.core.config import RunConfiguration
from nn_motion_control.core.precision import strict_float32
from nn_motion_control.core.seeding import seed_everything
from nn_motion_control.plant.plant import (
    Plant,
    RolloutLayout,
    rollout_splits_from_config,
)
from nn_motion_control.training.control import ControlTrainer
from nn_motion_control.training.logging_setup import ModelLogger
from nn_motion_control.training.preflight import (
    FAIL,
    Check,
    PreflightError,
    run_preflight,
)
from nn_motion_control.training.trainer import config_overrides

logger = logging.getLogger(__name__)

CONTROL_DTYPES = {"bfloat16": torch.bfloat16, "float32": torch.float32}

PREFLIGHT_MODES = ("enforce", "warn", "off")

# Rollout horizon (steps) used when the config omits 'horizon'.
DEFAULT_HORIZON = 32


def make_reference_gen(
    spec: dict, device: str, limits: WorkspaceLimits | None = None
):
    """
    Build a PVT reference generator: (origin, horizon,
    generator=None) -> (pos, vel).

    Both are [B, H, A]; velocity is the demand's analytic derivative.
    'generator' (used only by the 'mixed' kind) seeds a reproducible
    draw for validation; the deterministic single-trajectory kinds
    ignore it.
    """

    kind = spec.get("kind", "step")
    if kind == "workspace":
        if limits is None:
            raise ValueError("A workspace reference needs the workspace limits")

        return WorkspaceReference(spec, limits)
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


def warm_start(
    net: torch.nn.Module, config: ControllerConfig, path: str
) -> None:
    """
    Load a saved controller's weights into 'net' after checking it was the
    same controller for the same plant: architecture, layers, error scale,
    integrator taps and safe range. Quantisation widths and the error clip
    may change, since the weights are a valid start either way, and a
    change is logged. The config may add input features: their first-layer
    weights start at zero, so the net behaves as the saved one until
    training moves them. The feature normalisation travels with the
    weights, so the warm-started net keeps the scales it was trained with.
    """

    bundle = load_checkpoint_bundle(path, "cpu", weights_only=False)
    expected = {
        "artifact_type": "controller",
        "architecture": config.architecture,
        "hidden": list(config.hidden),
        "error_scale": config.error_scale,
        "integral_lambdas": list(config.integral_lambdas),
        "safe_range": config.safe_range.cpu().tolist(),
        "plant_checkpoint": config.plant_checkpoint,
    }
    found = {k: bundle.get(k) for k in expected}
    if isinstance(found["safe_range"], torch.Tensor):
        found["safe_range"] = found["safe_range"].tolist()
    old_features = list(bundle.get("features", []))
    added = [f for f in config.features if f not in old_features]
    if (
        found != expected
        or [f for f in config.features if f in old_features] != old_features
    ):
        raise ValueError(
            f"Cannot warm-start from {path}: it is {found} with features "
            f"{old_features}, the config asks for {expected} with features "
            f"{list(config.features)}"
        )
    retrained = {
        "quant": (
            bundle.get("quant"),
            [
                {"weight_bits": q.weight_bits, "act_bits": q.act_bits}
                for q in config.quant
            ],
        ),
        "error_clip": (bundle.get("error_clip"), config.error_clip),
    }

    for key, (old, new) in retrained.items():
        if old != new:
            logger.info("Warm start changes %s from %s to %s", key, old, new)
    state = bundle["model_state_dict"]
    if added:
        state = _widen_inputs(
            state,
            net.state_dict(),
            _input_slots(old_features, config),
            _input_slots(list(config.features), config),
        )
        logger.info("Warm start adds input features %s at zero weight", added)
    net.load_state_dict(state)
    logger.info("Warm-started the controller from %s", path)


def _input_slots(features: list[str], config: ControllerConfig) -> list[str]:
    # Axis-major input layout: per axis its features, then integrator taps.
    taps = [f"tap{i}" for i in range(len(config.integral_lambdas))]

    return [
        f"{axis}:{name}" for axis in config.axes for name in features + taps
    ]


def _widen_inputs(
    old: dict, fresh: dict, old_slots: list[str], new_slots: list[str]
) -> dict:
    """
    The saved state laid onto a wider input layout: each saved input column
    moves to its slot, new slots take zero weight and the fresh net's
    normalisation.
    """

    out = dict(old)
    dev = fresh["linears.0.weight"].device
    where = {slot: i for i, slot in enumerate(old_slots)}
    index = torch.tensor([where.get(s, -1) for s in new_slots], device=dev)
    known = index >= 0
    weight = torch.zeros_like(fresh["linears.0.weight"])
    weight[:, known] = old["linears.0.weight"].to(dev)[:, index[known]]
    out["linears.0.weight"] = weight

    for key in ("feat_mean", "feat_std"):
        merged = fresh[key].clone()
        merged[known] = old[key].to(dev)[index[known]]
        out[key] = merged

    return out


def reference_limits(
    config: ControllerConfig, plant_config: RunConfiguration, spec: dict
) -> WorkspaceLimits | None:
    """
    The travel box and command range a workspace reference may use, None
    for the other kinds.

    The box is the system's measured end-stops brought in by the spec's
    'margin' (position unit), cut to where the static holding command stays
    within HOLD_FRACTION of the DAC safe range (it binds before the stops on
    the PandA stage). The command range is the safe range, and the stage
    model is fitted to the plant dataset's FRF.
    """

    if spec.get("kind") != "workspace":
        return None
    limits = config.system.channel("position").limits
    if limits is None:
        raise ValueError("A workspace reference needs position limits")
    stops = torch.tensor(
        [limits[axis] for axis in config.axes], dtype=torch.float32
    )
    margin = float(spec.get("margin", DEFAULT_MARGIN))
    safe = config.safe_range
    stage = StageModel.from_dataset(
        plant_config.datafile_dir, tuple(config.axes)
    )
    holdable = HOLD_FRACTION * stage.gain[:, None] * safe

    return WorkspaceLimits(
        torch.maximum(stops[:, 0] + margin, holdable[:, 0]),
        torch.minimum(stops[:, 1] - margin, holdable[:, 1]),
        safe[:, 0],
        safe[:, 1],
        stage,
        config.servo_rate_hz,
    )


def control_dtype(train: dict) -> torch.dtype:
    name = str(train.get("training_dtype", "bfloat16"))
    if name not in CONTROL_DTYPES:
        raise ValueError(
            f"Controller training_dtype must be one of {list(CONTROL_DTYPES)}"
            f", got {name!r}"
        )

    return CONTROL_DTYPES[name]


class ControlRun:
    """
    Train a controller by policy gradient through a frozen plant, then
    report cost.
    """

    def __init__(self, controller_cfg_path: str, run_training: bool = True):
        """
        Build the run, pass preflight, then train unless 'run_training' is
        False (a preflight-only run). The config's training 'preflight' is
        "enforce" (default: a failed check stops the run), "warn" or "off".
        """

        self.config = ControllerConfig(controller_cfg_path)
        self._setup_logging()
        self._set_seed()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        plant = self._load_plant()
        loaders = self._build_loaders()
        # Normalise the controller's raw physical inputs with the
        # plant's fitted stats.
        feat_mean, feat_std = plant.feature_stats(
            self.config.features,
            n_integral=len(self.config.integral_lambdas),
            n_state=self.config.n_state,
            error_scale=self.config.error_scale,
        )
        net = build_controller_net(self.config, feat_mean, feat_std).to(
            self.device
        )
        if self.config.init_checkpoint:
            warm_start(net, self.config, self.config.init_checkpoint)
        policy = build_policy(self.config, net)
        sensor = build_sensor(self.config)
        actuator = build_actuator(self.config)

        train = self.config.training
        dtype = control_dtype(train)
        if dtype == torch.float32:
            strict_float32()
        horizon = int(train.get("horizon", DEFAULT_HORIZON))
        reference_gen = make_reference_gen(
            train.get("reference", {}), self.device, self.limits
        )
        # Loss and curriculum knobs the config may override
        overrides = config_overrides(
            train,
            {
                "curriculum_start": int,
                "curriculum_ramp": int,
                "val_horizon": int,
                "tracking_weight": float,
                "effort_weight": float,
                "rate_weight": float,
                "velocity_weight": float,
                "huber_delta": float,
                "tracking_scale": float,
                "axis_weights": list,
                "growth_weight": float,
                "growth_perturb": float,
                "growth_fraction": float,
                "growth_tail": float,
                "growth_min_horizon": int,
                "row_error_cap": float,
                "row_grad_bound": float,
            },
        )

        # Growth projection horizon is given in seconds, the trainer counts
        # steps.
        project_s = float(train.get("growth_project_s", 0.0))
        seeds = train.get("seed_curriculum")
        trainer = ControlTrainer(
            plant,
            net,
            policy,
            self.config,
            reference_gen,
            max_horizon=horizon,
            growth_project_steps=int(
                round(project_s * self.config.servo_rate_hz)
            ),
            sensor=sensor,
            actuator=actuator,
            seed_curriculum=(
                None
                if seeds is None
                else (float(seeds["start"]), int(seeds["epochs"]))
            ),
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
            # bfloat16 by default. Its 8-bit mantissa resolves the plant's
            # normalised position input only to tens of nm, so nm-level
            # control can ask for float32.
            training_dtype=dtype,
            # A warm start is scored before training so it sets the bar.
            validate_first=self.config.init_checkpoint is not None,
            window_size=self.plant_config.window_size,
            seed=self.config.seed,
        )
        if self.config.init_checkpoint:
            trainer.calibrate_activations()
        self.trainer = trainer
        self.preflight = self._preflight(trainer)
        if not run_training:
            return
        trainer.train()

        report = score_controller(
            net.resource_layers(), self.config.servo_rate_hz
        )

        for reason in report.reasons:
            self.logger.info("Resource: %s", reason)
        self.logger.info(
            (
                "Controller cost: params=%d, BRAM bits=%d, "
                "DSP-cycles=%d, max rate=%.0f Hz"
            ),
            report.params,
            report.bram_bits,
            report.dsp_cycles,
            report.max_servo_rate_hz,
        )
        self.logger.info("Run took %.1fs", time.perf_counter() - self._t0)

    def _preflight(self, trainer: ControlTrainer) -> list[Check]:
        mode = str(self.config.training.get("preflight", "enforce"))
        if mode not in PREFLIGHT_MODES:
            raise ValueError(f"Preflight must be one of {PREFLIGHT_MODES}")
        if mode == "off":
            return []
        manifest = self.config.training.get("pid_baseline")
        pid = (
            PidGains.from_manifest(self.config.resolve_path(manifest))
            if manifest
            else None
        )
        checks = run_preflight(
            trainer,
            self.config,
            self.plant_config.datafile_dir,
            pid,
            self.config.logging_dir,
        )
        failed = [c.name for c in checks if c.status == FAIL]
        if failed and mode == "enforce":
            raise PreflightError(
                f"Preflight failed: {failed}; see preflight.md in "
                f"{self.config.logging_dir}"
            )

        return checks

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
        self.limits = reference_limits(
            self.config, self.plant_config, train.get("reference", {})
        )
        quiescence = train.get("seed_quiescence")
        if quiescence is not None and self.limits is not None:
            # Seeds outside the box would leave their references no room.
            box = torch.stack([self.limits.pos_lo, self.limits.pos_hi], 1)
            quiescence = {**quiescence, "bounds": box.tolist()}
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
            quiescent_seed=quiescence,
            # Additionally seed from recorded closed-loop captures, the
            # mid-scan operating states the open-loop ID set never visits.
            capture_seed=self._capture_seed(train.get("seed_captures")),
        )

    def _capture_seed(self, spec: dict | None) -> dict | None:
        """
        Resolve a capture-seed spec's glob into concrete capture paths.
        """

        if not spec:
            return None

        pattern = self.config.resolve_path(str(spec["glob"]))
        paths = sorted(glob.glob(pattern, recursive=True))
        if not paths:
            raise ValueError(f"No captures matched: {pattern}")
        limit = int(spec.get("max_files", len(paths)))

        return {**spec, "paths": paths[:limit]}
