"""Streaming-TCN stepper.

RF <= window, and streaming == windowed rollout in eval.
"""

import pytest
import torch

from nn_motion_control.core.config import RunConfiguration
from nn_motion_control.data.normalize import NormStats
from nn_motion_control.models.builder import JsonModel
from nn_motion_control.models.layers.heads import AvgPoolLastK, LastFrame
from nn_motion_control.plant.plant import Plant, RolloutLayout
from nn_motion_control.plant.rollout import (
    RolloutStepper,
    StreamingConvStepper,
)

# The retired plant_stream.json architecture, built hermetically so
# the streaming-TCN code stays tested without depending on an
# example config (9 inputs = 3 axes).
STREAM_LAYERS = [
    {"TemporalConv": [9, 64, 3, 1, 1, 0.1]},
    {"TemporalConv": [64, 64, 3, 1, 2, 0.1]},
    {"TemporalConv": [64, 64, 3, 1, 4, 0.1]},
    {"TemporalConv": [64, 64, 3, 1, 8, 0.1]},
    "LastFrame",
    {"Flatten": [1]},
    {"Linear": [64, 128]},
    "ReLU",
    {"Linear": [128, 3]},
]


@pytest.fixture
def stream_cfg(config_factory):
    """Path to a hermetic streaming-TCN plant config matching
    plant_stream.json.
    """

    return config_factory(
        window_size=64,
        hidden_layers=STREAM_LAYERS,
        inputs=["position", "velocity", "dac"],
        targets={"position": {"form": "delta", "weight": 1}},
    )


@pytest.fixture
def stream_model(stream_cfg):
    """A freshly-built streaming-TCN JsonModel from the hermetic config."""

    return JsonModel(config=RunConfiguration(stream_cfg))


def test_last_frame_selects_final_step():
    x = torch.arange(2 * 3 * 5, dtype=torch.float32).reshape(2, 3, 5)
    out = LastFrame()(x)
    assert out.shape == (2, 3, 1)
    assert torch.equal(out[:, :, 0], x[:, :, -1])


def test_stream_model_builds_and_has_receptive_field_within_window(
    stream_model,
):
    m = stream_model
    # 4 blocks, kernel 3, dilations 1/2/4/8, 2 convs/block -> RF = 1
    # + 4*(1+2+4+8).
    rf = 1 + 4 * (1 + 2 + 4 + 8)
    assert rf == 61 <= 64
    # Forward runs and yields [B, A].
    out = m.eval()(torch.randn(4, 9, 64))
    assert out.shape == (4, 3)


def test_streaming_stepper_satisfies_protocol(stream_model):
    assert isinstance(StreamingConvStepper(stream_model), RolloutStepper)


def test_reset_matches_windowed_forward_last_frame(stream_model):
    m = stream_model.eval()
    torch.manual_seed(0)
    window = torch.randn(4, 9, 64)
    stepper = StreamingConvStepper(m)
    with torch.no_grad():
        ref = m(window)  # windowed forward, last-frame head
        got = stepper.reset(window)
    assert torch.allclose(ref, got, atol=1e-4, rtol=1e-3)


def _identity_stream_plant(model, kind, cfg_path):
    # Real 3-axis deltabot layout (9 features = 3 axes x
    # [pos,vel,dac]), identity norm.
    layout = RolloutLayout.from_config(RunConfiguration(cfg_path))
    ones, zeros = torch.ones(9), torch.zeros(9)
    in_stats = NormStats(mean=zeros, std=ones, normalizable=ones.bool())
    t_stats = NormStats(
        mean=torch.zeros(3),
        std=torch.ones(3),
        normalizable=torch.ones(3).bool(),
    )
    return Plant(
        model, in_stats, t_stats, layout, device="cpu", rollout_kind=kind
    )


def test_streaming_rollout_matches_windowed_rollout_in_eval(
    stream_model, stream_cfg
):
    # The whole point: under RF <= window + last-frame head,
    # streaming the model one frame at a time reproduces recomputing
    # the full window each step (float rounding).
    m = stream_model.eval()
    torch.manual_seed(1)
    warmup = torch.randn(3, 9, 64)
    dac = torch.randn(3, 12, 3)

    win = _identity_stream_plant(m, "windowed", stream_cfg)
    stream = _identity_stream_plant(m, "streaming", stream_cfg)
    with torch.no_grad():
        ref = win.roll_forward(warmup, dac, horizon=12)
        got = stream.roll_forward(warmup, dac, horizon=12)
    assert ref.shape == got.shape == (3, 12, 3)
    assert torch.allclose(ref, got, atol=1e-4, rtol=1e-3)


def test_avg_pool_last_k_averages_final_frames():
    x = torch.arange(2 * 3 * 8, dtype=torch.float32).reshape(2, 3, 8)
    out = AvgPoolLastK(4)(x)
    assert out.shape == (2, 3, 1)
    assert torch.allclose(out[:, :, 0], x[:, :, -4:].mean(dim=2))


class _AvgPoolStreamModel(torch.nn.Module):
    """A 3-block streaming-TCN (RF 29) with an AvgPoolLastK(16) readout."""

    def __init__(self):
        super().__init__()
        from nn_motion_control.models.layers.tcn import TemporalBlock

        self.network = torch.nn.Sequential(
            TemporalBlock(9, 16, 3, 1, 1, 0.1),
            TemporalBlock(16, 16, 3, 1, 2, 0.1),
            TemporalBlock(16, 16, 3, 1, 4, 0.1),
            AvgPoolLastK(16),
            torch.nn.Flatten(1),
            torch.nn.Linear(16, 3),
        )

    def forward(self, x):
        return self.network(x)


def test_streaming_matches_windowed_with_avg_pool_readout():
    # Pooled readout must stream identically to the windowed
    # forward: RF 29 <= 64-16, so all 16 pooled frames are in-window
    # and streaming reproduces it (float rounding).
    m = _AvgPoolStreamModel().eval()
    stepper = StreamingConvStepper(m)
    assert stepper.pool == 16
    torch.manual_seed(3)
    window = torch.randn(4, 9, 64)
    frames = [torch.randn(4, 9) for _ in range(10)]
    with torch.no_grad():
        # Windowed reference: slide a 64-window and re-run the full
        # forward each step.
        got = [stepper.reset(window)]
        win = window
        want = [m(win)]

        for fr in frames:
            got.append(stepper.step(fr))
            win = torch.cat([win[:, :, 1:], fr.unsqueeze(-1)], dim=2)
            want.append(m(win))

    for g, w_ in zip(got, want, strict=True):
        assert torch.allclose(g, w_, atol=1e-4, rtol=1e-3)


def test_streaming_rollout_is_differentiable(stream_model, stream_cfg):
    # BPTT-correct: activations stay in-graph, so a loss on the
    # rollout backprops.
    m = stream_model.train()
    torch.manual_seed(2)
    warmup = torch.randn(2, 9, 64)
    dac = torch.zeros(2, 6, 3)
    plant = _identity_stream_plant(m, "streaming", stream_cfg)
    preds = plant.roll_forward(warmup, dac, horizon=6)
    preds.pow(2).mean().backward()
    grads = [p.grad for p in m.parameters() if p.grad is not None]
    assert grads and any(g.abs().sum() > 0 for g in grads)
