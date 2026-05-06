import logging
import os
from dataclasses import dataclass, field

import torch.nn as nn

from deltabot_nn_controller.model_utils.instantiate_model import RunConfiguration
from deltabot_nn_controller.model_zoo.layers.tcn import TemporalBlock

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
    TODO
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

        # Auto-detect if we need to flatten data
        first_layer = next(iter(self.c.hidden_layers))
        first_name = next(iter(first_layer.keys()))
        self.is_temporal_model = components.is_temporal(first_name)

        for idx, layer in enumerate(self.c.hidden_layers):
            if isinstance(layer, dict):
                name, args = next(iter(layer.items()))

                # Validate layer type exists
                layer_class = components.get(name)

                if name == "Linear":
                    if self.is_temporal_model:
                        # Never multiply by window_size for temporal models
                        layers.append(layer_class(*args) if args else layer_class())
                    else:
                        # Non-temporal flatten time dimension including the window size
                        if idx == 0 and window_size != 1:
                            layers.append(
                                layer_class((args[0] * window_size, *args[1:]))
                                if args
                                else layer_class()
                            )
                        else:
                            layers.append(layer_class(*args) if args else layer_class())

                elif name == "ReLU":
                    layers.append(layer_class())

                elif name in ["Dropout", "AdaptiveAvgPool1d", "Flatten"]:
                    layers.append(layer_class(*args))

                elif name == "Conv1d":
                    layers.append(
                        layer_class((args[0], window_size, *args[1:]))
                        if args
                        else layer_class()
                    )

                elif name == "TemporalConv":
                    layers.append(layer_class(*args))

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
