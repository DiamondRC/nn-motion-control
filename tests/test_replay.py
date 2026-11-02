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
    """Every axis measured: position and command, each [A, n]."""
    t = np.arange(n)
    pos = np.stack([100.0 * np.sin(t / (20.0 + a)) for a in range(3)])
    dac = np.stack([5.0 * np.cos(t / (15.0 + a)) for a in range(3)])

    return pos, dac


def test_build_replay_batch_shapes_and_anchoring(plant):
    pos, dac = _capture()
    warmup, dac_future, truth = build_replay_batch(
        plant, pos, dac, axis=1, window=WINDOW, horizon=6, stride=10
    )
    starts = np.arange(0, pos.shape[1] - WINDOW - 6, 10)
    b = len(starts)
    assert warmup.shape == (b, 9, WINDOW)  # [B, F, W]
    assert dac_future.shape == (b, 6, 3)  # [B, H, A]
    assert truth.shape == (b, 6)
    # Truth is the scored axis's real future position (identity stats).
    want = pos[1, starts[0] + WINDOW]
    assert truth[0, 0].item() == pytest.approx(want, abs=1e-3)
    # Every axis carries its own recorded command in the warmup (columns
    # 2, 5, 8 are the x, y, z DAC under identity stats).
    for a, col in enumerate((2, 5, 8)):
        got = warmup[0, col].numpy()
        assert np.allclose(
            got, dac[a, starts[0] : starts[0] + WINDOW], atol=1e-5
        )


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
