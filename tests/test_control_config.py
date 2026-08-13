"""
control.config: controller artifact parsing, sizing and checkpoint round-trip.
"""

import json
from pathlib import Path

import pytest
import torch

from nn_motion_control.control.config import (
    Controller,
    ControllerConfig,
    build_controller_net,
    save_controller_checkpoint,
)
from nn_motion_control.control.resource import score_controller

REPO = Path(__file__).resolve().parents[1]
SYSTEM = REPO / "examples/deltabot/system.toml"
PLANT_CFG = REPO / "examples/deltabot/configs/plant_tcn.json"


def _write_config(
    tmp_path, quant, hidden=(8,), features=("position", "velocity", "reference")
):
    cfg = {
        "model_name": "test_controller",
        "system": str(SYSTEM),
        "run": {"out_dir": str(tmp_path), "seed": 7},
        "plant": {
            "config": str(PLANT_CFG),
            "checkpoint": str(tmp_path / "plant.pth"),
        },
        "controller": {
            "features": list(features),
            "hidden": list(hidden),
            "quant": quant,
            "servo_rate_hz": 10000,
            "batch_size": 32,
            "learning_rate": 1e-3,
        },
    }
    path = tmp_path / "controller.json"
    path.write_text(json.dumps(cfg))
    return str(path)


def test_sizing_from_axes_and_features(tmp_path):
    path = _write_config(
        tmp_path,
        quant=[
            {"weight_bits": 16, "act_bits": 48},
            {"weight_bits": 16, "act_bits": 16},
        ],
    )
    config = ControllerConfig(path)
    assert config.axes == ["x", "y", "z"]
    assert config.in_features == 3 * 3  # axes * features
    assert config.out_features == 3
    assert config.safe_range.shape == (3, 2)
    # deltabot DAC safe range is the recorded HV piezo rail on every axis.
    assert torch.equal(config.safe_range[0], torch.tensor([-737.25, 737.25]))


def test_quant_length_must_match_layers(tmp_path):
    path = _write_config(
        tmp_path, quant=[{"weight_bits": 16, "act_bits": 48}]
    )  # need 2
    with pytest.raises(ValueError, match="one entry per linear layer"):
        ControllerConfig(path)


def test_checkpoint_round_trip(tmp_path):
    path = _write_config(
        tmp_path,
        quant=[
            {"weight_bits": 8, "act_bits": 48},
            {"weight_bits": 8, "act_bits": 16},
        ],
    )
    config = ControllerConfig(path)
    net = build_controller_net(config)
    report = score_controller(
        net.resource_layers(), config.servo_rate_hz
    ).as_dict()

    ckpt = str(tmp_path / "ctrl.pth")
    save_controller_checkpoint(ckpt, net, config, resource_report=report)
    assert Path(ckpt).with_suffix(".export.json").exists()

    loaded = Controller.from_checkpoint(ckpt)

    for k, v in net.state_dict().items():
        assert torch.allclose(v, loaded.net.state_dict()[k])
    assert torch.equal(loaded.policy._lo, config.safe_range[:, 0])
    # The rebuilt policy honours the same I/O contract.
    dac = loaded.policy(
        torch.zeros(2, 3),
        torch.zeros(2, 3),
        torch.zeros(2, 3),
        torch.zeros(2, 3),
    )
    assert dac.shape == (2, 3)
