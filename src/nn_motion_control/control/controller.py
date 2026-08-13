"""
Quantisation-configurable neural controller (the FPGA deployment target).

A coupled MLP maps the per-axis control features (position, velocity,
reference, error) to a per-axis DAC command. Per-layer weight and
activation bit-widths are explicit, so the network trains in float with
straight-through fake-quant to the exact widths the PandA MAC engine
will use, and its weights export in a BRAM-ready integer layout. The
policy adapter assembles the feature vector and clamps the command to
the DAC safe band.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import torch
import torch.nn.functional as F  # noqa: N812  (conventional alias)
from torch import nn

from nn_motion_control.control.resource import (
    ALLOWED_FEATURES,
    LayerCost,
    QuantSpec,
    dense_layers,
)

_QUANT_EPS = 1e-12
# Floor for feat_std so a near-constant input channel doesn't divide by
# ~0 in normalisation.
_FEAT_STD_FLOOR = 1e-8


def _qmax(bits: int) -> int:
    """
    Largest positive code of a symmetric signed 'bits'-bit integer.
    """

    return (1 << (bits - 1)) - 1


def _quant_scale(x: torch.Tensor, bits: int) -> torch.Tensor:
    """
    Symmetric per-tensor scale mapping the largest magnitude to the top code.
    """

    return x.detach().abs().amax().clamp_min(_QUANT_EPS) / _qmax(bits)


def fake_quantize(x: torch.Tensor, bits: int) -> torch.Tensor:
    """
    Symmetric per-tensor fake-quant with a straight-through estimator.

    'bits' outside 1..31 is a pass-through, which is how the wide
    (48-bit) interferometer input is represented at full precision in
    simulation. The rounding is non-differentiable, so the gradient is
    passed straight through to 'x'.
    """

    if bits <= 0 or bits >= 32:
        return x
    qmax = _qmax(bits)
    scale = _quant_scale(x, bits)
    q = torch.clamp(torch.round(x / scale), -qmax, qmax) * scale

    return x + (q - x).detach()


def quantize_to_int(x: torch.Tensor, bits: int) -> tuple[torch.Tensor, float]:
    """
    Quantise a tensor to symmetric signed integers plus the shared float scale.
    """

    qmax = _qmax(bits)
    scale = _quant_scale(x, bits)
    codes = torch.clamp(torch.round(x / scale), -qmax, qmax).to(torch.int64)

    return codes, float(scale)


@dataclass(frozen=True)
class FeatureSpec:
    """
    Which per-axis control features the controller consumes, in input
    order.

    'position'/'velocity' come from the plant state, 'reference' is the
    setpoint and 'error' is 'reference - position'. The assembled vector
    is axis-major (axis outer, feature inner), matching
    'SystemSpec.labels' ordering.
    """

    names: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.names:
            raise ValueError("Controller needs at least one feature")
        unknown = [n for n in self.names if n not in ALLOWED_FEATURES]
        if unknown:
            raise ValueError(f"Unknown controller features: {unknown}")

    @property
    def count(self) -> int:
        """
        Features per axis.
        """

        return len(self.names)

    def assemble(
        self,
        position: torch.Tensor,
        velocity: torch.Tensor,
        reference: torch.Tensor,
        reference_velocity: torch.Tensor,
    ) -> torch.Tensor:
        """
        Build the axis-major feature vector '[B, A * count]' from
        '[B, A]' inputs.
        """

        available = {
            "position": position,
            "velocity": velocity,
            "reference": reference,
            "reference_velocity": reference_velocity,
            "error": reference - position,
            "velocity_error": reference_velocity - velocity,
        }
        cols = [available[name] for name in self.names]  # each [B, A]
        stacked = torch.stack(cols, dim=-1)  # [B, A, F]

        return stacked.reshape(stacked.shape[0], -1)  # [B, A * F], axis-major


@dataclass(frozen=True)
class BramLayer:
    """
    One layer's weights as BRAM-ready integers plus the float
    reconstruction scales.
    """

    weight_codes: list[list[int]]  # [out][in], row-major by output neuron
    weight_scale: float
    bias_codes: list[int]  # [out]
    bias_scale: float
    weight_bits: int


@dataclass(frozen=True)
class BramExport:
    """
    The controller's weights laid out for the MAC engine's gain BRAM.
    """

    layers: list[BramLayer]
    layout: str = "row-major-by-output"

    def to_dict(self) -> dict:
        """
        JSON-serialisable view (no tensors) for the checkpoint sidecar.
        """

        return {
            "layout": self.layout,
            "layers": [
                {
                    "weight_codes": layer.weight_codes,
                    "weight_scale": layer.weight_scale,
                    "bias_codes": layer.bias_codes,
                    "bias_scale": layer.bias_scale,
                    "weight_bits": layer.weight_bits,
                }
                for layer in self.layers
            ],
        }


class ControllerNet(nn.Module):
    """
    A coupled dense MLP with per-layer fake-quantised weights and
    activations.

    'hidden' lists the hidden widths, 'quant' gives one 'QuantSpec' per
    linear layer (so 'len(quant) == len(hidden) + 1'). The first
    layer's 'act_bits' is the wide interferometer input width. ReLU
    follows every layer but the last.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden: Sequence[int],
        quant: Sequence[QuantSpec],
        feat_mean: torch.Tensor | None = None,
        feat_std: torch.Tensor | None = None,
    ):
        super().__init__()
        dims = [in_features, *hidden, out_features]
        if len(quant) != len(dims) - 1:
            raise ValueError("Quant must have one entry per linear layer")
        self.dims = dims
        self.quant = tuple(quant)
        self.linears = nn.ModuleList(
            nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)
        )
        # Input normalisation (fit from the plant's stats) travels as
        # buffers so it is saved/loaded with the weights, identity by
        # default. Physical inputs are in raw units (counts), so this
        # keeps the first layer out of saturation.
        if feat_mean is None:
            feat_mean = torch.zeros(in_features)
        if feat_std is None:
            feat_std = torch.ones(in_features)
        self.register_buffer("feat_mean", feat_mean.reshape(-1).float())
        self.register_buffer(
            "feat_std", feat_std.reshape(-1).float().clamp_min(_FEAT_STD_FLOOR)
        )

    @property
    def input_act_bits(self) -> int:
        """
        Activation width of the (wide) input layer.
        """

        return self.quant[0].act_bits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = cast(torch.Tensor, self.feat_mean)
        std = cast(torch.Tensor, self.feat_std)
        x = (x - mean) / std  # normalise raw physical features to ~unit scale
        last = len(self.linears) - 1

        for i, layer in enumerate(self.linears):
            linear = cast(nn.Linear, layer)
            x = fake_quantize(x, self.quant[i].act_bits)
            w = fake_quantize(linear.weight, self.quant[i].weight_bits)
            x = F.linear(x, w, linear.bias)
            if i != last:
                x = F.relu(x)

        return x

    def resource_layers(self) -> list[LayerCost]:
        """
        Per-layer FPGA cost (bridges to 'control.resource.score_controller').
        """

        return dense_layers(
            self.dims,
            [q.weight_bits for q in self.quant],
            [q.act_bits for q in self.quant],
        )

    def export_bram(self) -> BramExport:
        """
        Quantise every layer's weights and bias to BRAM-ready integers.
        """

        layers: list[BramLayer] = []

        for layer, spec in zip(self.linears, self.quant, strict=True):
            linear = cast(nn.Linear, layer)
            w_codes, w_scale = quantize_to_int(linear.weight, spec.weight_bits)
            b_codes, b_scale = quantize_to_int(linear.bias, spec.weight_bits)
            layers.append(
                BramLayer(
                    weight_codes=w_codes.tolist(),
                    weight_scale=w_scale,
                    bias_codes=b_codes.tolist(),
                    bias_scale=b_scale,
                    weight_bits=spec.weight_bits,
                )
            )

        return BramExport(layers=layers)


class NNPolicy:
    """
    Adapt a 'ControllerNet' to the plant's
    '(position, velocity, reference) -> dac' policy contract, bounding
    the command to the per-axis DAC safe range.

    The net's raw output is mapped to 'safe_range' with a 'tanh' (a
    smooth bounded head): 'dac = centre + half_range * tanh(raw)'.
    Unlike a hard clamp this keeps a live gradient everywhere, so the
    policy can always be improved: a hard clamp saturates and zeroes the
    policy gradient once the command hits a rail.
    """

    def __init__(
        self,
        net: ControllerNet,
        features: FeatureSpec,
        safe_range: torch.Tensor,
        n_axes: int,
    ):
        expected = n_axes * features.count
        if net.dims[0] != expected:
            raise ValueError(
                f"Controller input {net.dims[0]} != axes*features {expected}"
            )
        if safe_range.shape != (n_axes, 2):
            raise ValueError(
                f"Safe_range must be [A, 2], got {tuple(safe_range.shape)}"
            )
        self.net = net
        self.features = features
        self.n_axes = n_axes
        self._lo = safe_range[:, 0]
        self._hi = safe_range[:, 1]
        self._centre = (self._lo + self._hi) / 2
        self._half = (self._hi - self._lo) / 2

    def __call__(
        self,
        position: torch.Tensor,
        velocity: torch.Tensor,
        reference: torch.Tensor,
        reference_velocity: torch.Tensor,
    ) -> torch.Tensor:
        x = self.features.assemble(
            position, velocity, reference, reference_velocity
        )
        raw = self.net(x)
        centre = self._centre.to(raw.device, raw.dtype)
        half = self._half.to(raw.device, raw.dtype)

        return centre + half * torch.tanh(raw)
