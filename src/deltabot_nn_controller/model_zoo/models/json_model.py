import logging
import os

import torch.nn as nn

from deltabot_nn_controller.model_utils.instantiate_model import (
    RunConfiguration,
    resolve_class,
)

logger = logging.getLogger(os.path.basename(__file__))


class JsonModel(nn.Module):
    """
    TODO
    """

    def __init__(self, config: RunConfiguration):
        self.c = config

        self.logging: bool = self.c.do_verb_log
        self.hidden_layers: list[dict[type[nn.Module], list[int] | None]] = []

        # Unpack the model configuration
        for layer in self.c.hidden_layers:
            if isinstance(layer, dict):
                for name, size in layer.items():
                    cls = resolve_class(name, module_paths=("torch.nn",))
                    self.hidden_layers.append({cls: size})
            elif isinstance(layer, str):
                cls = resolve_class(layer, module_paths=("torch.nn",))
                self.hidden_layers.append({cls: None})
            else:
                raise TypeError(f"Unknown arg type in model config: {layer}")

        # Model config
        super().__init__()
        self._build_model()
        self.apply(self._init_weights)

    # Constructs the model based on the provided config
    def _build_model(self):
        output_size = self.c.target_size
        window_size = self.c.window_size
        # Flatten input if using windows
        input_size = self.c.input_size * window_size

        if self.logging:
            logger.debug("Building Model...")
            logger.debug(f"Model Inputs: {self.c.input_params}")
            logger.debug(f"Model Outputs: {self.c.target_params}")
            logger.debug(f"Model Input size {input_size}")
            logger.debug(f"Model Output size {output_size}")

        layers = []

        layer_type = next(iter(self.hidden_layers[0].keys()))
        layers.append(
            layer_type(input_size, *next(iter(self.hidden_layers[0].values())))
        )

        for entry in self.hidden_layers[1:-1]:
            for layer, args in entry.items():
                layers.append(layer(*args) if args else layer())

        layer_type = next(iter(self.hidden_layers[-1].keys()))
        layers.append(
            layer_type(*next(iter(self.hidden_layers[-1].values())), output_size)
        )

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
        """
        Handle flattening for sliding window input if necessary,
        then pass through the network.
        """

        if x.dim() == 3:  # [B, window_size, features]
            x = x.flatten(start_dim=1)  # [B, window_size*features]
        # if x.dim() == 2, already correct shape ([B, features]), return
        return self.network(x)
