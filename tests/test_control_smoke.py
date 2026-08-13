"""
End-to-end controller smoke: train a tiny controller and write a
deployable bundle.
"""

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from nn_motion_control.control.closed_loop import step_reference
from nn_motion_control.control.config import (
    Controller,
    ControllerConfig,
    build_controller_net,
    build_policy,
)
from nn_motion_control.data.dataset import DatasetMetadata
from nn_motion_control.data.normalize import NormStats
from nn_motion_control.plant.plant import Plant, RolloutLayout
from nn_motion_control.training.control import ControlTrainer

TINY_SYSTEM = """
name = "tiny"
axes = ["x"]

[channels.position]
kind = "measured"

[channels.velocity]
kind = "derived"
from = "position"
order = 1

[channels.dac]
kind = "command"
[channels.dac.range]
x = [-2.0, 2.0]
[channels.dac.safe_range]
x = [-1.0, 1.0]
"""


class _DacDriven(nn.Module):
    def __init__(self, gain: float = 1.0):
        super().__init__()
        self.gain = gain

    def forward(self, window):
        return self.gain * window[:, 2:3, -1]


class _PassScaler:
    """A no-op GradScaler for float32 CPU training."""

    def __init__(self, device, enabled):
        pass

    def scale(self, loss):
        return loss

    def unscale_(self, optimizer):
        pass

    def step(self, optimizer):
        optimizer.step()

    def update(self):
        pass


def _dac_plant() -> Plant:
    layout = RolloutLayout(
        pos_cols=[0], vel_cols=[1], dac_cols=[2], n_features=3
    )
    ones = torch.ones(3)
    in_stats = NormStats(
        mean=torch.zeros(3), std=ones, normalizable=ones.bool()
    )
    t_stats = NormStats(
        mean=torch.zeros(1),
        std=torch.ones(1),
        normalizable=torch.ones(1).bool(),
    )
    return Plant(_DacDriven(1.0), in_stats, t_stats, layout, device="cpu")


def _fake_node_info() -> DatasetMetadata:
    ones = torch.ones(1)
    return DatasetMetadata(
        input_labels=np.array(["x_pos"]),
        target_labels=np.array(["x_dac"]),
        input_denorm_params={"mean": {"x_pos": 0.0}, "std": {"x_pos": 1.0}},
        target_denorm_params={"mean": {"x_dac": 0.0}, "std": {"x_dac": 1.0}},
        loss_weights=ones,
        input_stats=NormStats(torch.zeros(1), ones, ones.bool()),
        target_stats=NormStats(torch.zeros(1), ones, ones.bool()),
    )


def _write_controller_config(tmp_path) -> str:
    system_path = tmp_path / "tiny.toml"
    system_path.write_text(TINY_SYSTEM)
    cfg = {
        "model_name": "tiny_ctrl",
        "system": str(system_path),
        "run": {"out_dir": str(tmp_path), "seed": 0},
        "plant": {"config": "plant.json", "checkpoint": "plant.pth"},
        "controller": {
            "features": ["error"],
            "hidden": [4],
            "quant": [
                {"weight_bits": 32, "act_bits": 48},
                {"weight_bits": 32, "act_bits": 16},
            ],
            "servo_rate_hz": 10000,
            "training": {
                "horizon": 5,
                "curriculum_start": 5,
                "curriculum_ramp": 0,
                "reference": {"kind": "step", "amplitude": 0.5, "step_at": 0},
            },
            "batch_size": 8,
            "max_epochs": 2,
            "patience": 2,
            "learning_rate": 0.05,
        },
    }
    path = tmp_path / "controller.json"
    path.write_text(json.dumps(cfg))
    return str(path)


def test_controller_smoke_trains_and_exports(tmp_path):
    config = ControllerConfig(_write_controller_config(tmp_path))
    net = build_controller_net(config)
    policy = build_policy(config, net)
    plant = _dac_plant()

    b, w, h = 8, 4, 5
    batch = (torch.zeros(b, 3, w), torch.zeros(b, h, 1), torch.zeros(b, h, 1))
    loader = [batch, batch]

    trainer = ControlTrainer(
        plant,
        net,
        policy,
        config,
        reference_gen=lambda origin, horizon, generator=None: (
            step_reference(origin, 0.5, horizon, 0),
            torch.zeros_like(step_reference(origin, 0.5, horizon, 0)),
        ),
        max_horizon=h,
        curriculum_start=h,
        curriculum_ramp=0,
        train_loader=loader,
        val_loader=loader,
        device="cpu",
        scaler_class=_PassScaler,
        optimizer_class=torch.optim.Adam,
        criterion_class=nn.MSELoss,
        node_info=_fake_node_info(),
        max_epochs=2,
        learning_rate=0.05,
        min_delta=0.0,
        patience=2,
        model_name=config.model_name,
        save_path=config.save_path,
        logging=False,
        accumulation_steps=1,
        training_dtype=torch.bfloat16,
        window_size=w,
        seed=0,
    )
    trainer.train()

    # A deployable bundle and its tensor-free sidecar were written.
    assert Path(config.save_path).exists()
    assert Path(config.save_path).with_suffix(".export.json").exists()
    bundle = torch.load(config.save_path, weights_only=False)
    assert bundle["artifact_type"] == "controller"
    assert bundle["resource_report"]["fits"] is True
    assert bundle["bram_export"]["layers"]  # BRAM-ready weights present

    # The saved controller reloads and honours the policy contract.
    loaded = Controller.from_checkpoint(config.save_path)
    dac = loaded.policy(
        torch.zeros(2, 1),
        torch.zeros(2, 1),
        torch.ones(2, 1),
        torch.zeros(2, 1),
    )
    assert dac.shape == (2, 1)
    assert torch.all(
        dac.abs() <= 1.0
    )  # clamped to the tiny system's safe range
