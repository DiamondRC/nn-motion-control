"""Model build-path tests.

Ensures string activations are not silently dropped, LayerNorm is
registered and the windowed-Linear construction is correct. The
shipped TCN config is ~1.6 B params, so the temporal path is
exercised with a small inline config instead of instantiating it.
"""

from pathlib import Path

import pytest
import torch

from nn_motion_control.core.config import RunConfiguration
from nn_motion_control.models.builder import (
    JsonModel,
    ModelComponents,
)

REPO = Path(__file__).resolve().parents[1]
TEMPORAL = REPO / "examples/deltabot/configs/plant_tcn.json"

# A windowed-MLP architecture (retired plant_mlp.json) built
# hermetically to exercise the string-activation, LayerNorm and
# windowed-Linear build paths (9 inputs = 3 axes).
MLP_LAYERS = [
    {"Linear": [9, 256]},
    "ReLU",
    {"LayerNorm": [256]},
    {"Dropout": [0.1]},
    {"Linear": [256, 128]},
    "ReLU",
    {"LayerNorm": [128]},
    {"Dropout": [0.1]},
    {"Linear": [128, 3]},
]


@pytest.fixture
def mlp_cfg(config_factory):
    """Path to a hermetic windowed-MLP plant config matching plant_mlp.json."""

    return config_factory(
        window_size=32,
        hidden_layers=MLP_LAYERS,
        inputs=["position", "velocity", "dac"],
        targets={"position": {"form": "delta", "weight": 1}},
    )


def test_plant_mlp_builds_and_forwards(mlp_cfg):
    rc = RunConfiguration(mlp_cfg)
    model = JsonModel(config=rc)
    x = torch.randn(2, rc.input_size, rc.window_size)  # [B, features, window]
    y = model(x)
    assert y.shape == (2, rc.target_size)


def test_plant_mlp_keeps_activations_and_layernorm(mlp_cfg):
    rc = RunConfiguration(mlp_cfg)
    model = JsonModel(config=rc)
    names = [m.__class__.__name__ for m in model.network]
    # Bare-string "ReLU" entries must survive; LayerNorm must resolve.
    assert names.count("ReLU") > 0
    assert names.count("LayerNorm") > 0


def test_windowed_linear_uses_window_size(mlp_cfg):
    rc = RunConfiguration(mlp_cfg)
    model = JsonModel(config=rc)
    first_linear = next(
        m for m in model.network if m.__class__.__name__ == "Linear"
    )
    # First Linear must accept in_features * window_size (flattened window).
    assert first_linear.in_features == rc.input_size * rc.window_size


def test_recurrent_registry_entries_removed():
    # nn.LSTM/nn.GRU return (output, state) tuples that crash
    # nn.Sequential, they were dead landmines and must not resolve.
    components = ModelComponents()

    for name in ("LSTM", "GRU"):
        with pytest.raises(ValueError):
            components.get(name)


def test_temporal_config_layers_all_registered():
    # Every layer name in the shipped TCN config resolves without
    # instantiating it.
    rc = RunConfiguration(str(TEMPORAL))
    components = ModelComponents()

    for layer in rc.hidden_layers:
        name = layer if isinstance(layer, str) else next(iter(layer))
        assert components.get(name) is not None


def test_small_temporal_model_builds(config_factory):
    cfg = config_factory(
        window_size=8,
        hidden_layers=[
            {"TemporalConv": [15, 8, 3, 1, 1, 0.1]},
            {"AdaptiveAvgPool1d": [1]},
            {"Flatten": [1]},
            {"Linear": [8, 12]},
        ],
    )
    rc = RunConfiguration(cfg)
    model = JsonModel(config=rc)
    assert model.is_temporal_model
    x = torch.randn(2, rc.input_size, rc.window_size)  # [B, C, T]
    assert model(x).shape == (2, 12)


SMALL_TEMPORAL = [
    {"TemporalConv": [15, 8, 3, 1, 1, 0.0]},
    {"AdaptiveAvgPool1d": [1]},
    {"Flatten": [1]},
    {"Linear": [8, 12]},
]


def test_linear_skip_starts_as_the_plain_model(config_factory):
    torch.manual_seed(0)
    plain = JsonModel(
        config=RunConfiguration(
            config_factory(window_size=8, hidden_layers=SMALL_TEMPORAL)
        )
    ).eval()
    rc = RunConfiguration(
        config_factory(
            architecture={
                "window_size": 8,
                "hidden_layers": SMALL_TEMPORAL,
                "linear_skip": True,
            }
        )
    )
    skip = JsonModel(config=rc).eval()
    # A checkpoint without the branch loads, the branch reading as zero.
    skip.load_state_dict(plain.state_dict())
    x = torch.randn(2, rc.input_size, rc.window_size)
    with torch.no_grad():
        assert torch.equal(skip(x), plain(x))


def test_linear_skip_adds_an_exactly_linear_branch(config_factory):
    rc = RunConfiguration(
        config_factory(
            architecture={
                "window_size": 8,
                "hidden_layers": SMALL_TEMPORAL,
                "linear_skip": True,
            }
        )
    )
    model = JsonModel(config=rc).eval()
    assert model.linear_skip is not None
    with torch.no_grad():
        model.linear_skip.weight.normal_()
        model.linear_skip.bias.normal_()
    x = torch.randn(2, rc.input_size, rc.window_size)
    expected = model.network(x) + model.linear_skip(x.flatten(start_dim=1))
    with torch.no_grad():
        torch.testing.assert_close(model(x), expected)


def test_frozen_linear_skip_takes_no_gradient(config_factory):
    rc = RunConfiguration(
        config_factory(
            architecture={
                "window_size": 8,
                "hidden_layers": SMALL_TEMPORAL,
                "linear_skip": True,
                "freeze_linear_skip": True,
            }
        )
    )
    model = JsonModel(config=rc)
    assert model.linear_skip is not None
    model(torch.randn(2, rc.input_size, rc.window_size)).sum().backward()
    assert model.linear_skip.weight.grad is None
    assert model.network[-1].weight.grad is not None


def test_linear_skip_refuses_tf32_training(config_factory):
    from nn_motion_control.training.run import CompleteRun

    run = CompleteRun.__new__(CompleteRun)
    run.m = RunConfiguration(
        config_factory(
            architecture={
                "window_size": 8,
                "hidden_layers": SMALL_TEMPORAL,
                "linear_skip": True,
            },
            training={"allow_tf32": True},
        )
    )
    with pytest.raises(ValueError, match="allow_tf32 off"):
        run._cuda_profiling()


def test_every_cli_command_starts_at_full_precision():
    from typer.testing import CliRunner

    from nn_motion_control.__main__ import cli

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    CliRunner().invoke(cli, ["champions"])
    assert not torch.backends.cuda.matmul.allow_tf32
    assert not torch.backends.cudnn.allow_tf32
    assert torch.get_float32_matmul_precision() == "highest"
