import logging
import os
from dataclasses import dataclass, field

import torch.nn as nn

from nn_motion_control.core.config import RunConfiguration
from nn_motion_control.models.layers.tcn import TemporalBlock

logger = logging.getLogger(os.path.basename(__file__))


@dataclass(frozen=True)
class ModelComponents:
    registry: dict = field(
        default_factory=lambda: {
            # Expects input shape [batch, features * time]
            "Linear": {"cls": nn.Linear, "temporal": False},
            "ReLU": {"cls": nn.ReLU, "temporal": False},
            "Dropout": {"cls": nn.Dropout, "temporal": False},
            "Flatten": {"cls": nn.Flatten, "temporal": False},
            "LayerNorm": {"cls": nn.LayerNorm, "temporal": False},
            # Expects input shape [batch, features, time]
            "AdaptiveAvgPool1d": {"cls": nn.AdaptiveAvgPool1d, "temporal": True},
            "Conv1d": {"cls": nn.Conv1d, "temporal": True},
            "LSTM": {"cls": nn.LSTM, "temporal": True},
            "GRU": {"cls": nn.GRU, "temporal": True},
            # Custom
            "TemporalConv": {"cls": TemporalBlock, "temporal": True},
        }
    )

    def get(self, name: str) -> nn.Module:
        entry = self.registry.get(name)
        if entry is None:
            raise ValueError(f"Unsupported layer {name}.")
        return entry["cls"]

    def is_temporal(self, name: str) -> bool:
        entry = self.registry.get(name)
        if entry is None:
            raise ValueError(f"Unsupported layer {name}.")
        return entry["temporal"]


class JsonModel(nn.Module):
    """
    A feed-forward network assembled from a JSON ``hidden_layers`` spec.

    Temporal vs MLP mode is auto-detected from the first layer. For a windowed MLP the
    first Linear's in-features are scaled by the window size (the window is flattened
    in ``forward``); temporal models consume ``[B, C, T]`` directly.
    """

    def __init__(self, config: RunConfiguration):
        self.c = config

        self.logging: bool = self.c.do_verb_log
        self.hidden_layers: list[dict[type[nn.Module], list[int] | None]] = []

        # Model config
        super().__init__()
        self._build_model()
        self.apply(self._init_weights)

    # Constructs the model based on the provided config
    def _build_model(self):
        output_size = self.c.target_size
        window_size = self.c.window_size
        if self.logging:
            logger.debug("Building Model...")
            logger.debug(f"Model Inputs: {self.c.input_params}")
            logger.debug(f"Model Outputs: {self.c.target_params}")
            logger.debug(f"Model Window Size: {window_size}")
            logger.debug(
                f"Model Input Size: {self.c.input_size} * {window_size} "
                f"= {self.c.input_size * window_size}"
            )
            logger.debug(f"Model Output Size: {output_size}")

        layers = []
        components = ModelComponents()

        # Normalise every layer spec to a (name, args) pair. A bare string such as
        # "ReLU" is shorthand for a layer with no constructor args; a single-key dict
        # {"Linear": [15, 2048]} carries positional args. Anything else is an error
        # (previously non-dict entries were silently dropped, so string activations
        # vanished from the model).
        def _as_name_args(layer):
            if isinstance(layer, str):
                return layer, []
            if isinstance(layer, dict):
                name, args = next(iter(layer.items()))
                return name, list(args) if args else []
            raise TypeError(
                f"Unsupported layer spec {layer!r}; expected a str or single-key dict."
            )

        # Auto-detect whether we operate on time-major data (based on the first layer)
        first_name, _ = _as_name_args(self.c.hidden_layers[0])
        self.is_temporal_model = components.is_temporal(first_name)

        for idx, layer in enumerate(self.c.hidden_layers):
            name, args = _as_name_args(layer)
            layer_class = components.get(name)  # validates the layer is supported

            if (
                name == "Linear"
                and not self.is_temporal_model
                and idx == 0
                and window_size != 1
            ):
                # MLP path flattens [B, features, window] -> [B, features * window]
                # in forward(), so the first Linear must accept in_features * window.
                layers.append(layer_class(args[0] * window_size, *args[1:]))
            else:
                layers.append(layer_class(*args))

        self.network = nn.Sequential(*layers)

        if self.logging:
            logger.debug("Model configuration:")
            for layer in self.network:
                logger.debug(f"{layer}")

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # nn.init.xavier_normal_(m.weight, gain=0.5)  # ReLU scaling
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")  # He initialization
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        if self.is_temporal_model:
            if x.dim() != 3:
                raise ValueError(f"Expected [B, C, T], got {x.shape}")
            return self.network(x)  # [batch, features, time]
        else:
            # MLP: flatten to [batch, features*time]
            return self.network(x.flatten(start_dim=1))
