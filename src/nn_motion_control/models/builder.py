import logging
import os
from dataclasses import dataclass, field

import torch.nn as nn

from nn_motion_control.core.config import RunConfiguration
from nn_motion_control.models.layers.heads import AvgPoolLastK, LastFrame
from nn_motion_control.models.layers.tcn import TemporalBlock
from nn_motion_control.models.ssm import DiagSSM

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
            "AdaptiveAvgPool1d": {
                "cls": nn.AdaptiveAvgPool1d,
                "temporal": True,
            },
            "LastFrame": {"cls": LastFrame, "temporal": True},
            "AvgPoolLastK": {"cls": AvgPoolLastK, "temporal": True},
            "Conv1d": {"cls": nn.Conv1d, "temporal": True},
            # Custom
            "TemporalConv": {"cls": TemporalBlock, "temporal": True},
            # Recurrent (streams via a carried state, not a windowed conv)
            "DiagSSM": {"cls": DiagSSM, "temporal": True, "recurrent": True},
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

    def is_recurrent(self, name: str) -> bool:
        entry = self.registry.get(name)
        if entry is None:
            raise ValueError(f"Unsupported layer {name}.")

        return entry.get("recurrent", False)


class JsonModel(nn.Module):
    """
    A feed-forward network assembled from a JSON hidden_layers spec.

    Temporal vs MLP mode is auto-detected from the first layer. For a
    windowed MLP the first Linear's in-features are scaled by the
    window size (the window is flattened in forward). Temporal models
    consume [B, C, T] directly.
    """

    def __init__(self, config: RunConfiguration):
        self.c = config

        self.logging: bool = self.c.do_verb_log
        self.hidden_layers: list[dict[type[nn.Module], list[int] | None]] = []
        # When set, temporal forward runs channels-last (see
        # enable_channels_last).
        self.use_channels_last = False

        super().__init__()
        self._build_model()
        self.apply(self._init_weights)

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

        # Normalise each layer spec to a (name, args) pair. A bare
        # string such as "ReLU" is shorthand for a layer with no
        # constructor args, a single-key dict such as
        # {"Linear": [15, 2048]} carries positional args. Anything else
        # is an error (previously non-dict entries were silently
        # dropped, so string activations vanished from the model).
        def _as_name_args(layer):
            if isinstance(layer, str):
                return layer, []
            if isinstance(layer, dict):
                name, args = next(iter(layer.items()))
                return name, list(args) if args else []
            raise TypeError(
                f"Unsupported layer spec {layer!r}; expected a str or "
                f"single-key dict."
            )

        # Temporal vs MLP mode is inferred from the first layer.
        first_name, _ = _as_name_args(self.c.hidden_layers[0])
        self.is_temporal_model = components.is_temporal(first_name)
        # A recurrent (SSM) stack is time-major but streams via a
        # carried state, so it gets its own forward branch (transpose to
        # [B, T, C], run the scan, then head).
        self.is_ssm = components.is_recurrent(first_name)

        for idx, layer in enumerate(self.c.hidden_layers):
            name, args = _as_name_args(layer)
            layer_class = components.get(
                name
            )  # validates the layer is supported

            if (
                name == "Linear"
                and not self.is_temporal_model
                and idx == 0
                and window_size != 1
            ):
                # MLP path flattens [B, features, window] to
                # [B, features * window] in forward(), so the first
                # Linear must accept in_features * window.
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
            nn.init.kaiming_normal_(
                m.weight, nonlinearity="relu"
            )  # He initialization
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def enable_channels_last(self) -> None:
        """
        Run the temporal forward channels-last (identical maths, no NHWC
        transposes).

        Only valid for a channels-last-able TemporalConv stack, raises
        otherwise.
        """

        from nn_motion_control.models.channels_last import is_channels_last_able

        if not (self.is_temporal_model and is_channels_last_able(self.network)):
            raise ValueError(
                "Model is not a channels-last-able TemporalConv stack"
            )
        self.use_channels_last = True

    def ssm_section(self) -> tuple[list[nn.Module], list[nn.Module]]:
        """
        Split the network into the leading DiagSSM layers and the
        trailing head.

        The SSM layers run time-major ([B, T, C]), the head
        (pool/flatten/Linear) runs channel-major ([B, C, T]) exactly
        like the TCN head. Used by the SSM forward branch and by the
        recurrent rollout stepper so both see the same split.
        """

        layers = list(self.network)
        n = 0

        while n < len(layers) and isinstance(layers[n], DiagSSM):
            n += 1

        return layers[:n], layers[n:]

    def _ssm_forward(self, x):
        ssm_layers, head_layers = self.ssm_section()
        u = x.transpose(1, 2)  # [B, C, T] -> [B, T, C]

        for layer in ssm_layers:
            u = layer(u)
        out = u.transpose(1, 2)  # back to [B, C, T] for the head

        for layer in head_layers:
            out = layer(out)

        return out

    def forward(self, x):
        if self.is_ssm:
            if x.dim() != 3:
                raise ValueError(f"Expected [B, C, T], got {x.shape}")
            return self._ssm_forward(x)
        if self.is_temporal_model:
            if x.dim() != 3:
                raise ValueError(f"Expected [B, C, T], got {x.shape}")
            if self.use_channels_last:
                from nn_motion_control.models.channels_last import (
                    channels_last_tcn_forward,
                )

                return channels_last_tcn_forward(self.network, x)
            return self.network(x)  # [batch, features, time]
        else:
            # MLP: flatten to [batch, features*time]
            return self.network(x.flatten(start_dim=1))
