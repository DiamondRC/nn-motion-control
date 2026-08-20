"""Recorded-command replay: batch construction, shapes, anchoring."""

import numpy as np
import pytest
import torch

from nn_motion_control.core.config import RunConfiguration
from nn_motion_control.data.normalize import NormStats
from nn_motion_control.eval.replay import build_replay_batch, replay_errors
from nn_motion_control.models.builder import JsonModel
from nn_motion_control.plant.plant import Plant, RolloutLayout

WINDOW = 8
TCN_LAYERS = [
    {"TemporalConv": [9, 8, 3, 1, 1, 0.0]},
    {"AdaptiveAvgPool1d": [1]},
    {"Flatten": [1]},
    {"Linear": [8, 3]},
]


@pytest.fixture
def plant(config_factory):
    path = config_factory(
        window_size=WINDOW,
        hidden_layers=TCN_LAYERS,
        inputs=["position", "velocity", "dac"],
        targets={"position": {"form": "delta", "weight": 1}},
    )
    model = JsonModel(config=RunConfiguration(path)).eval()
    layout = RolloutLayout.from_config(RunConfiguration(path))
    ident = NormStats(
        mean=torch.zeros(9),
        std=torch.ones(9),
        normalizable=torch.ones(9).bool(),
    )
    tgt = NormStats(
        mean=torch.zeros(3),
        std=torch.ones(3),
        normalizable=torch.ones(3).bool(),
    )

    return Plant(model, ident, tgt, layout, device="cpu")


def _capture(n=200):
    t = np.arange(n)
    return 100.0 * np.sin(t / 20.0), 5.0 * np.cos(t / 15.0)  # pos, dac


def test_build_replay_batch_shapes_and_anchoring(plant):
    pos, dac = _capture()
    warmup, dac_future, truth = build_replay_batch(
        plant, pos, dac, axis=1, window=WINDOW, horizon=6, stride=10
    )
    starts = np.arange(0, len(pos) - WINDOW - 6, 10)
    b = len(starts)
    assert warmup.shape == (b, 9, WINDOW)  # [B, F, W]
    assert dac_future.shape == (b, 6, 3)  # [B, H, A]
    assert truth.shape == (b, 6)
    # Truth is the active axis's real future position (identity stats).
    want = pos[starts[0] + WINDOW]
    assert truth[0, 0].item() == pytest.approx(want, abs=1e-3)
    # Inactive axes are held quiescent (zero) in the warmup command rows.
    assert torch.count_nonzero(warmup[:, (2, 8), :]) == 0  # x, z dac cols


def test_replay_errors_shape_finite_and_nonneg(plant):
    pos, dac = _capture()
    err = replay_errors(
        plant, pos, dac, axis=0, window=WINDOW, horizon=6, stride=10
    )
    assert err.shape[1] == 6
    assert torch.isfinite(err).all()
    assert (err >= 0).all()


def test_capture_too_short_raises(plant):
    pos, dac = _capture(n=WINDOW + 2)
    with pytest.raises(ValueError, match="too short"):
        build_replay_batch(
            plant, pos, dac, axis=0, window=WINDOW, horizon=6, stride=10
        )
