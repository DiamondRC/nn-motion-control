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
        self.activations: list[type[nn.Module]] = []
        self.hidden_layer: list[type[nn.Module]] = []
        self.hidden_sizes: list[int] = []

        for layer_dict in self.c.hidden_layers:
            for name, size in layer_dict.items():
                cls = resolve_class(name, module_paths=("torch.nn",))
                self.hidden_layer.append(cls)
                self.hidden_sizes.append(size)
                break

        for name in self.c.activations:
            cls = resolve_class(name, module_paths=("torch.nn",))
            self.activations.append(cls)

        # Model config
        super().__init__()
        self._build_model()
        self.apply(self._init_weights)

    # Constructs the model based on the provided config
    def _build_model(self):
        output_size = self.c.target_size
        dropout = self.c.dropout
        layer_norm = self.c.layer_norm
        window_size = self.c.window_size
        # Flatten input if using windows
        input_size = self.c.input_size * window_size

        if self.logging:
            logger.debug("Building Model...")
            logger.debug(f"Model Inputs: {self.c.input_params}")
            logger.debug(f"Model Outputs: {self.c.target_params}")
            logger.debug(f"Model Input size {input_size}")
            logger.debug(f"Model Output size {output_size}")
            logger.debug(f"LayerNorm enabled: {layer_norm}")
            logger.debug(f"Using Dropout: {dropout}")

        layers = []

        layers.append(self.hidden_layer[0](input_size, self.hidden_sizes[0]))
        for idx, layer in enumerate(self.hidden_layer[1:]):
            layers.append(layer(self.hidden_sizes[idx], self.hidden_sizes[idx + 1]))
        layers.append(self.hidden_layer[-1](self.hidden_sizes[-1], output_size))

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
