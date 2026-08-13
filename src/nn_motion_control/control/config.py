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
import os
from pathlib import Path

import torch

from nn_motion_control.control.controller import (
    ControllerNet,
    FeatureSpec,
    NNPolicy,
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

        self.in_features = len(self.axes) * len(self.features)
        self.out_features = len(self.axes)
        self._dac_name = _command_channel(self.system)
        self.safe_range = _safe_range_tensor(self.system, self._dac_name)

        # Training block (consumed by the policy-gradient trainer).
        self.training = ctrl.get("training", {})
        self.batch_size = ctrl["batch_size"]
        self.max_epochs = ctrl.get("max_epochs", 200)
        self.patience = ctrl.get("patience", 30)
        self.min_delta = ctrl.get("min_delta", 0.0)
        self.learning_rate = ctrl["learning_rate"]
        self.optimiser = ctrl.get("optimiser", "Adam")

    def _resolve(self, rel: str) -> str:
        """
        Resolve a path against the config's directory (absolute paths
        pass through).
        """

        return rel if os.path.isabs(rel) else str(self._dir / rel)


def build_controller_net(
    config: ControllerConfig,
    feat_mean: torch.Tensor | None = None,
    feat_std: torch.Tensor | None = None,
) -> ControllerNet:
    """
    Instantiate a fresh 'ControllerNet' from a parsed config.

    'feat_mean'/'feat_std' (from 'Plant.feature_stats') normalise the
    raw physical inputs, omitting them leaves normalisation as identity.
    """

    return ControllerNet(
        config.in_features,
        config.out_features,
        config.hidden,
        config.quant,
        feat_mean=feat_mean,
        feat_std=feat_std,
    )


def build_policy(config: ControllerConfig, net: ControllerNet) -> NNPolicy:
    """
    Wrap a 'ControllerNet' in the DAC-clamping policy adapter for this system.
    """

    return NNPolicy(
        net, FeatureSpec(config.features), config.safe_range, len(config.axes)
    )


def save_controller_checkpoint(
    path: str,
    net: ControllerNet,
    config: ControllerConfig,
    resource_report: dict | None = None,
) -> None:
    """
    Save the controller weights plus everything needed to rebuild and deploy it.
    """

    bundle = {
        "schema_version": 1,
        "artifact_type": "controller",
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
        "safe_range": config.safe_range.cpu(),
        "plant_config": config.plant_config_path,
        "plant_checkpoint": config.plant_checkpoint,
        "servo_rate_hz": config.servo_rate_hz,
        "resource_report": resource_report,
        "bram_export": net.export_bram().to_dict(),
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(bundle, path)

    write_json_sidecar(
        path,
        {
            "model_name": config.model_name,
            "axes": bundle["axes"],
            "features": bundle["features"],
            "hidden": bundle["hidden"],
            "quant": bundle["quant"],
            "servo_rate_hz": config.servo_rate_hz,
            "input_norm": {
                "mean": net.get_buffer("feat_mean").tolist(),
                "std": net.get_buffer("feat_std").tolist(),
            },
            "resource_report": resource_report,
            "bram_export": bundle["bram_export"],
        },
        suffix=".export.json",
    )


class Controller:
    """
    A trained controller loaded for evaluation or deployment (net + policy).
    """

    def __init__(self, net: ControllerNet, policy: NNPolicy):
        self.net = net
        self.policy = policy

    @classmethod
    def from_checkpoint(cls, path: str, device: str = "cpu") -> Controller:
        """
        Rebuild a controller and its policy from a saved bundle.
        """

        bundle = load_checkpoint_bundle(
            path, device, weights_only=False, expected_schema=1
        )
        axes = bundle["axes"]
        features = tuple(bundle["features"])
        quant = [QuantSpec(**q) for q in bundle["quant"]]
        net = ControllerNet(
            len(axes) * len(features), len(axes), bundle["hidden"], quant
        ).to(device)
        net.load_state_dict(bundle["model_state_dict"])
        net.eval()
        safe_range = bundle["safe_range"].to(device)
        policy = NNPolicy(net, FeatureSpec(features), safe_range, len(axes))

        return cls(net, policy)
