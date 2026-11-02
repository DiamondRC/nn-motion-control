"""
Controller artifact: config-per-artifact reader plus checkpoint save/load.

A controller config references a SystemSpec and a trained plant, and
declares the network shape, per-layer quantisation and closed-loop
training settings. Unlike a plant, its inputs include a generated
reference (not a dataset channel) and its output is the DAC command, so
I/O sizing derives from the system axes and a feature spec rather than
the plant's channel-expansion path.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import cast

import torch

from nn_motion_control.control.controller import (
    ControllerNet,
    FeatureSpec,
    NNPolicy,
)
from nn_motion_control.control.hardware import (
    ActuatorModel,
    QuantisedCommand,
    SensorModel,
)
from nn_motion_control.control.recurrent import (
    RecurrentControllerNet,
    RecurrentPolicy,
    RecurrentSpec,
)
from nn_motion_control.control.resource import QuantSpec
from nn_motion_control.core.champions import (
    CHAMPION_PREFIX,
    is_champion_ref,
    registry_path,
    resolve_champion,
    resolved_paths,
)
from nn_motion_control.core.checkpoints import (
    load_checkpoint_bundle,
    write_json_sidecar,
)
from nn_motion_control.core.system import SystemSpec

ARCHITECTURES = ("mlp", "recurrent")

# Either controller network and its matching policy adapter.
ControllerNetwork = ControllerNet | RecurrentControllerNet
ControllerPolicy = NNPolicy | RecurrentPolicy


def _command_channel(system: SystemSpec) -> str:
    """
    Name of the system's single command (DAC) channel.
    """

    commands = [n for n, ch in system.channels.items() if ch.kind == "command"]
    if len(commands) != 1:
        raise ValueError(
            f"Controller needs exactly one command channel, found {commands}"
        )

    return commands[0]


def _safe_range_tensor(system: SystemSpec, dac_name: str) -> torch.Tensor:
    """
    Per-axis DAC safe range as a '[A, 2]' (lo, hi) tensor in
    system-axis order.
    """

    channel = system.channel(dac_name)
    if channel.safe_range is None:
        raise ValueError(f"Command channel '{dac_name}' has no safe_range")
    rows = [channel.safe_range[axis] for axis in system.axes]

    return torch.tensor(rows, dtype=torch.float32)


class ControllerConfig:
    """
    Parsed controller artifact config.

    Sizing derives from the system: 'in_features = len(axes) *
    len(features)' and 'out_features = len(axes)'. The plant is
    referenced by path only (loaded later by the trainer), so parsing
    needs neither the plant checkpoint nor the dataset.
    """

    def __init__(self, json_path: str):
        self.json_path = json_path
        self._dir = Path(json_path).parent
        cfg = json.loads(Path(json_path).read_text())

        system_path = self._resolve(cfg["system"])
        self.system = SystemSpec.from_toml(system_path)
        self.axes = list(self.system.axes)

        run = cfg.get("run", {})
        self.model_name = cfg["model_name"]
        self.seed = run.get("seed", 42)
        # Optional controller checkpoint to warm-start the weights from.
        init = run.get("init_checkpoint")
        self.init_checkpoint: str | None = self._resolve(init) if init else None
        out_dir = self._resolve(run.get("out_dir", "runs"))
        # Each artifact owns a directory so its checkpoint, run history and
        # eval plots do not collide with other models under a shared out_dir.
        model_dir = f"{out_dir}/{self.model_name}"
        self.save_path = f"{model_dir}/{self.model_name}.pth"
        self.logging_dir = f"{model_dir}/history"
        self.eval_dir = f"{model_dir}/eval"

        # The plant may be referenced by champion label
        # ("champion:plant", or {"champion": "plant"}) or by explicit
        # config/checkpoint paths. A champion reference resolves via the
        # registry beside the SystemSpec, so re-promoting a plant flows
        # in without editing this config.
        plant = cfg["plant"]
        ref = None
        if is_champion_ref(plant):
            ref = plant
        elif isinstance(plant, dict) and "champion" in plant:
            ref = CHAMPION_PREFIX + plant["champion"]
        if ref is not None:
            reg = registry_path(system_path)
            self.plant_config_path, self.plant_checkpoint = resolved_paths(
                resolve_champion(ref, reg), reg
            )
        else:
            self.plant_config_path = self._resolve(plant["config"])
            self.plant_checkpoint = self._resolve(plant["checkpoint"])

        ctrl = cfg["controller"]
        self.features = tuple(ctrl["features"])
        FeatureSpec(self.features)  # validate names early
        self.hidden = list(ctrl["hidden"])
        self.quant = [
            QuantSpec(weight_bits=q["weight_bits"], act_bits=q["act_bits"])
            for q in ctrl["quant"]
        ]
        if len(self.quant) != len(self.hidden) + 1:
            raise ValueError(
                "Quant must have one entry per linear layer "
                f"(len(hidden) + 1 = {len(self.hidden) + 1}), "
                f"got {len(self.quant)}"
            )
        self.servo_rate_hz = float(ctrl["servo_rate_hz"])
        # The closed-loop rollout takes one controller step per plant step,
        # so every per-step quantity (integrator decays, the actuator
        # low-pass, reference speeds) assumes the two rates are equal.
        data_rate = self.system.data_rate_hz
        if data_rate and not math.isclose(self.servo_rate_hz, data_rate):
            raise ValueError(
                f"Controller servo_rate_hz {self.servo_rate_hz:g} must equal "
                f"the system's data_rate_hz {data_rate:g}"
            )

        # Optional leaky-integrator error memory: an EMA of the per-axis
        # tracking error at each time constant (seconds), converted to a
        # decay factor lambda = exp(-1 / (tau * servo_rate)). Gives the
        # otherwise memoryless MLP the low-frequency state a memoryless law
        # cannot form, so it can reject a slowly accumulating bias.
        taus = ctrl.get("integral_taus", [])
        self.integral_lambdas = [
            math.exp(-1.0 / (float(t) * self.servo_rate_hz)) for t in taus
        ]

        # 'mlp' is the memoryless net (with optional fixed taps above);
        # 'recurrent' carries a learned per-axis error-state bank instead.
        self.architecture = ctrl.get("architecture", "mlp")
        if self.architecture not in ARCHITECTURES:
            raise ValueError(f"Architecture must be one of {ARCHITECTURES}")
        self.recurrent: RecurrentSpec | None = None
        if self.architecture == "recurrent":
            if self.integral_lambdas:
                raise ValueError(
                    "A recurrent controller carries its own error states, "
                    "drop integral_taus"
                )
            self.recurrent = RecurrentSpec.from_dict(ctrl.get("recurrent", {}))
        # Tracking-error normalisation scale (position unit). None keeps
        # the workspace position std; the recurrent net requires one.
        scale = ctrl.get("error_scale")
        self.error_scale = None if scale is None else float(scale)
        if self.recurrent is not None and self.error_scale is None:
            raise ValueError("A recurrent controller needs an error_scale")
        # Soft clip on the tracking error (position unit) before it reaches
        # the net and the integrator taps. None leaves it linear.
        clip = ctrl.get("error_clip")
        self.error_clip = None if clip is None else float(clip)
        if self.error_clip is not None and self.recurrent is not None:
            raise ValueError("Error_clip is only supported on the MLP")
        if "command" in self.features and self.recurrent is not None:
            raise ValueError("The command feature is only supported on the MLP")
        # Steps the gradient may run back through the fed-back command, 0
        # cutting it at every step (a training setting, not deployed).
        self.command_backprop_steps = int(ctrl.get("command_backprop_steps", 0))

        per_axis = len(self.features) + len(self.integral_lambdas)
        self.in_features = len(self.axes) * (per_axis + self.n_state)
        self.out_features = len(self.axes)
        self._dac_name = _command_channel(self.system)
        self.safe_range = _safe_range_tensor(self.system, self._dac_name)

        # Hardware observation model: the sensor grid the deployed
        # controller reads, in the position unit. Empty means the policy
        # trains on the plant's continuous state (no sensor model).
        self.sensor = ctrl.get("sensor", {})

        # Hardware command model: rolls the command off before the plant
        # integrates it, keeping the plant within its identified band.
        # Empty means the raw command drives the plant.
        self.actuator = ctrl.get("actuator", {})

        # Training block (consumed by the policy-gradient trainer).
        self.training = ctrl.get("training", {})
        self.batch_size = ctrl["batch_size"]
        self.max_epochs = ctrl.get("max_epochs", 200)
        self.patience = ctrl.get("patience", 30)
        self.min_delta = ctrl.get("min_delta", 0.0)
        self.learning_rate = ctrl["learning_rate"]
        self.optimiser = ctrl.get("optimiser", "Adam")

    @property
    def n_state(self) -> int:
        """Recurrent error states per axis (0 for the MLP)."""

        return 0 if self.recurrent is None else self.recurrent.per_axis_states

    def _resolve(self, rel: str) -> str:
        """
        Resolve a path against the config's directory (absolute paths
        pass through).
        """

        return rel if os.path.isabs(rel) else str(self._dir / rel)

    def resolve_path(self, rel: str) -> str:
        """Public path resolution for callers outside this module."""

        return self._resolve(rel)


def _make_net(
    n_axes: int,
    features: tuple[str, ...],
    hidden: list[int],
    quant: list[QuantSpec],
    servo_rate_hz: float,
    integral_lambdas: list[float],
    recurrent: RecurrentSpec | None,
    error_scale: float | None,
    feat_mean: torch.Tensor | None = None,
    feat_std: torch.Tensor | None = None,
) -> ControllerNetwork:
    # Shared by a fresh build and a checkpoint rebuild so both agree.
    if recurrent is not None:
        return RecurrentControllerNet(
            n_axes,
            len(features),
            recurrent,
            servo_rate_hz,
            hidden,
            quant,
            float(cast(float, error_scale)),
            feat_mean=feat_mean,
            feat_std=feat_std,
        )
    in_features = n_axes * (len(features) + len(integral_lambdas))

    return ControllerNet(
        in_features,
        n_axes,
        hidden,
        quant,
        feat_mean=feat_mean,
        feat_std=feat_std,
    )


def _make_policy(
    net: ControllerNetwork,
    features: tuple[str, ...],
    safe_range: torch.Tensor,
    n_axes: int,
    integral_lambdas: list[float],
    error_clip: float | None = None,
    command_backprop_steps: int = 0,
) -> ControllerPolicy:
    if isinstance(net, RecurrentControllerNet):
        return RecurrentPolicy(net, FeatureSpec(features), safe_range, n_axes)

    return NNPolicy(
        net,
        FeatureSpec(features),
        safe_range,
        n_axes,
        integral_lambdas=integral_lambdas,
        error_clip=error_clip,
        command_backprop_steps=command_backprop_steps,
    )


def build_controller_net(
    config: ControllerConfig,
    feat_mean: torch.Tensor | None = None,
    feat_std: torch.Tensor | None = None,
) -> ControllerNetwork:
    """
    Instantiate a fresh controller network from a parsed config.

    'feat_mean'/'feat_std' (from 'Plant.feature_stats') normalise the
    raw physical inputs, omitting them leaves normalisation as identity.
    """

    return _make_net(
        len(config.axes),
        config.features,
        config.hidden,
        config.quant,
        config.servo_rate_hz,
        config.integral_lambdas,
        config.recurrent,
        config.error_scale,
        feat_mean=feat_mean,
        feat_std=feat_std,
    )


def build_policy(
    config: ControllerConfig, net: ControllerNetwork
) -> ControllerPolicy:
    """
    Wrap a controller network in the DAC-clamping policy adapter for this
    system.
    """

    return _make_policy(
        net,
        config.features,
        config.safe_range,
        len(config.axes),
        config.integral_lambdas,
        config.error_clip,
        config.command_backprop_steps,
    )


def build_sensor(config: ControllerConfig) -> SensorModel | None:
    """
    Build the hardware sensor model from a parsed config, or None.

    A positive 'grid' enables encoder quantisation; anything else (an
    empty or absent 'sensor' block) leaves the policy on the plant's
    continuous state.
    """

    grid = float(config.sensor.get("grid", 0.0))

    return SensorModel(grid) if grid > 0.0 else None


def build_actuator(
    config: ControllerConfig,
) -> ActuatorModel | QuantisedCommand | None:
    """
    Build the hardware command model from a parsed config, or None.

    A positive 'cutoff_hz' enables the command roll-off and 'quantise'
    rounds the command to whole counts as the DAC does; an empty or absent
    'actuator' block drives the plant with the raw command.
    """

    cutoff = float(config.actuator.get("cutoff_hz", 0.0))
    rolloff = (
        ActuatorModel(cutoff, config.servo_rate_hz) if cutoff > 0.0 else None
    )
    if config.actuator.get("quantise", False):
        return QuantisedCommand(rolloff)

    return rolloff


# Schema 2 records the command stage ('output_mapping'). Schema 1 bundles
# are read with the same per-sign mapping: those with a symmetric safe range
# (every one before the PandA specs) behave identically under it, and the
# asymmetric ones were trained after it was introduced. Schema 3 adds the
# error soft clip and the fixed activation scales, absent (no clip, per-row
# scales) in older bundles.
CONTROLLER_SCHEMA = 3
READABLE_SCHEMAS = (1, 2, 3)
OUTPUT_MAPPING = "tanh, per-sign scale to safe_range"


def save_controller_checkpoint(
    path: str,
    net: ControllerNetwork,
    config: ControllerConfig,
    resource_report: dict | None = None,
) -> None:
    """
    Save the controller weights plus everything needed to rebuild and deploy it.
    """

    recurrent = config.recurrent
    bundle = {
        "architecture": config.architecture,
        "recurrent": None if recurrent is None else recurrent.to_dict(),
        "error_scale": config.error_scale,
        "error_clip": config.error_clip,
        "schema_version": CONTROLLER_SCHEMA,
        "artifact_type": "controller",
        "output_mapping": OUTPUT_MAPPING,
        "model_state_dict": net.state_dict(),
        "model_name": config.model_name,
        "seed": config.seed,
        "axes": list(config.axes),
        "features": list(config.features),
        "hidden": list(config.hidden),
        "quant": [
            {"weight_bits": q.weight_bits, "act_bits": q.act_bits}
            for q in config.quant
        ],
        "input_act_bits": net.input_act_bits,
        "integral_lambdas": list(config.integral_lambdas),
        "sensor": dict(config.sensor),
        "actuator": dict(config.actuator),
        "safe_range": config.safe_range.cpu(),
        "plant_config": config.plant_config_path,
        "plant_checkpoint": config.plant_checkpoint,
        "servo_rate_hz": config.servo_rate_hz,
        "resource_report": resource_report,
        "bram_export": net.export_bram().to_dict(),
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(bundle, path)

    # The recurrent net's input normalisation lives on its readout MLP.
    norm_net = net.mlp if isinstance(net, RecurrentControllerNet) else net
    sidecar = {
        "model_name": config.model_name,
        "architecture": config.architecture,
        "axes": bundle["axes"],
        "features": bundle["features"],
        "hidden": bundle["hidden"],
        "quant": bundle["quant"],
        "servo_rate_hz": config.servo_rate_hz,
        "error_scale": config.error_scale,
        # Applied to reference - position before the error input and the
        # integrator taps: error_clip * tanh(error / error_clip).
        "error_clip": config.error_clip,
        "input_norm": {
            "mean": norm_net.get_buffer("feat_mean").tolist(),
            "std": norm_net.get_buffer("feat_std").tolist(),
        },
        "resource_report": resource_report,
        "bram_export": bundle["bram_export"],
        # The command stage after the net: tanh, then each sign scaled to
        # its own limit (positive to hi, negative to -lo) per axis.
        "output_mapping": OUTPUT_MAPPING,
        "safe_range": config.safe_range.cpu().tolist(),
        "actuator": dict(config.actuator),
    }
    if "command" in config.features:
        # The command input is the controller's own previous output: the
        # bounded value before any DAC rounding, held in a register that
        # starts at the command already holding the stage at handover.
        sidecar["command_feedback"] = (
            "previous bounded output before DAC rounding; initialise to the "
            "command holding the stage at handover"
        )
    if isinstance(net, RecurrentControllerNet):
        sidecar["recurrent"] = bundle["recurrent"]
        sidecar["poles"] = net.modes.describe()
    write_json_sidecar(path, sidecar, suffix=".export.json")


class Controller:
    """
    A trained controller loaded for evaluation or deployment (net + policy).
    """

    def __init__(self, net: ControllerNetwork, policy: ControllerPolicy):
        self.net = net
        self.policy = policy

    @classmethod
    def from_checkpoint(cls, path: str, device: str = "cpu") -> Controller:
        """
        Rebuild a controller and its policy from a saved bundle.

        Bundles written before the architecture key default to the MLP.
        """

        bundle = load_checkpoint_bundle(path, device, weights_only=False)
        schema = bundle.get("schema_version")
        if schema not in READABLE_SCHEMAS:
            raise ValueError(
                f"Controller checkpoint {path} has schema {schema!r}, "
                f"expected one of {READABLE_SCHEMAS}"
            )
        axes = bundle["axes"]
        features = tuple(bundle["features"])
        lambdas = bundle.get("integral_lambdas", [])
        spec = bundle.get("recurrent")
        net = _make_net(
            len(axes),
            features,
            bundle["hidden"],
            [QuantSpec(**q) for q in bundle["quant"]],
            float(bundle["servo_rate_hz"]),
            lambdas,
            None if spec is None else RecurrentSpec.from_dict(spec),
            bundle.get("error_scale"),
        ).to(device)
        net.load_state_dict(bundle["model_state_dict"])
        net.eval()
        safe_range = bundle["safe_range"].to(device)
        policy = _make_policy(
            net,
            features,
            safe_range,
            len(axes),
            lambdas,
            bundle.get("error_clip"),
        )

        return cls(net, policy)
