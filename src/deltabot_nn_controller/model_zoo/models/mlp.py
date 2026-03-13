import json
import logging

import torch.nn as nn

logger = logging.getLogger(__name__)


class MLP(nn.Module):
    """
    Multilayer Perceptron (MLP) implementation with configurable architecture.

    The architecture is defined by a JSON config file specifying:
    - input_size: Number of input features
    - output_size: Number of output features
    - hidden_layers: List of hidden layer sizes
    - activations: List of activation functions for each hidden layer
    - dropout: Dropout rate between layers
    - layer_norm: Boolean if we apply LayerNorm after each hidden layer
    - window_size: Number of timesteps in input window
    """

    # TODO - rewrite logic to handle variable activations per layer,
    # currently assumes all ReLU
    ACTIVATIONS = {"ReLU": nn.ReLU, "Tanh": nn.Tanh, "Sigmoid": nn.Sigmoid}

    def __init__(self, logging, config):
        # User params
        self.logging = logging

        # Model config
        super().__init__()
        self.network_type = config["network_type"]
        if self.network_type == "mlp":
            self._build_mlp(config)
        else:
            raise ValueError(f"Unsupported network type: {self.network_type}")
        self.apply(self._init_weights)

    # Constructs the MLP based on the provided config
    def _build_mlp(self, config):
        output_size = config["output_size"]
        hidden_layers = config["hidden_layers"]
        activations = config.get("activations", ["ReLU"] * len(hidden_layers))
        dropout = config.get("dropout", 0.0)
        layer_norm = config.get("layer_norm", False)

        window_size = config.get("window_size", 1)
        # Flatten input if using windows
        input_size = config["input_size"] * window_size

        if self.logging:
            logger.debug(f"Building MLP with input size {input_size}")
            logger.debug(f"LayerNorm enabled: {config.get('layer_norm', False)}")
            logger.debug(f"Using Dropout: {config.get('dropout')}")

        layers = []
        prev_size = input_size

        for i, size in enumerate(hidden_layers):
            layers.append(nn.Linear(prev_size, size))
            if i < len(activations):
                act_fn = self.ACTIVATIONS.get(activations[i], nn.ReLU)
                layers.append(act_fn())
            if layer_norm:
                layers.append(nn.LayerNorm(size))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_size = size

        layers.append(nn.Linear(prev_size, output_size))
        self.network = nn.Sequential(*layers)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # nn.init.xavier_normal_(m.weight, gain=0.5)  # ReLU scaling
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")  # He initialization
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def return_layer_shapes(self):
        """
        Utility function to return the shape of each layer in the model.
        """
        shapes = []
        for layer in self.network:
            shapes.append((layer.in_features, layer.out_features))
        return shapes

    def forward(self, x):
        """
        Handle flattening for sliding window input if necessary,
        then pass through the network.
        """

        if x.dim() == 3:  # [B, window_size, features]
            x = x.flatten(start_dim=1)  # [B, window_size*features]
        # if x.dim() == 2, already correct shape ([B, features]), return
        return self.network(x)


# Load
with open("src/deltabot_nn_controller/model_zoo/basic_mlp.json") as f:
    config = json.load(f)
    if len(config["activations"]) != len(config["hidden_layers"]):
        raise ValueError("Length of activations must match length of hidden layers")
