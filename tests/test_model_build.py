"""Model build-path tests (Workstream D).

Ensures string activations are not silently dropped, LayerNorm is registered, and the
windowed-Linear construction is correct. The shipped TCN config is ~1.6 B params, so we
exercise the temporal path with a small inline config instead of instantiating it.
"""

from pathlib import Path

import torch

from nn_motion_control.core.config import RunConfiguration
from nn_motion_control.models.builder import (
    JsonModel,
    ModelComponents,
)

REPO = Path(__file__).resolve().parents[1]
PLANT_MLP = REPO / "examples/deltabot/configs/plant_mlp.json"
TEMPORAL = REPO / "examples/deltabot/configs/plant_tcn.json"


def test_plant_mlp_builds_and_forwards():
    rc = RunConfiguration(str(PLANT_MLP))
    model = JsonModel(config=rc)
    x = torch.randn(2, rc.input_size, rc.window_size)  # [B, features, window]
    y = model(x)
    assert y.shape == (2, rc.target_size)


def test_plant_mlp_keeps_activations_and_layernorm():
    rc = RunConfiguration(str(PLANT_MLP))
    model = JsonModel(config=rc)
    names = [m.__class__.__name__ for m in model.network]
    # Bare-string "ReLU" entries must survive; LayerNorm must resolve.
    assert names.count("ReLU") > 0
    assert names.count("LayerNorm") > 0


def test_windowed_linear_uses_window_size():
    rc = RunConfiguration(str(PLANT_MLP))
    model = JsonModel(config=rc)
    first_linear = next(m for m in model.network if m.__class__.__name__ == "Linear")
    # First Linear must accept in_features * window_size (flattened window).
    assert first_linear.in_features == rc.input_size * rc.window_size


def test_temporal_config_layers_all_registered():
    # Every layer name in the shipped TCN config resolves without instantiating it.
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
