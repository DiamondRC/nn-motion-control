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
# Fixed activation scales: each training batch records its peak, over the
# rollout, of the ACT_QUANTILE row peak (so a single wild row cannot set
# it). Committed once per batch, the range only ever grows: a fixed-point
# range must cover the whole operating envelope, not the typical batch, or
# the rarer large inputs (far holds, long moves) saturate. ACT_HEADROOM
# sits above it before the code saturates, as it does on the FPGA.
ACT_QUANTILE = 0.99
ACT_HEADROOM = 4.0
# Floor for feat_std so a near-constant input channel doesn't divide by
# ~0 in normalisation.
_FEAT_STD_FLOOR = 1e-8


def _qmax(bits: int) -> int:
    """
    Largest positive code of a symmetric signed 'bits'-bit integer.
    """

    return (1 << (bits - 1)) - 1


def _quant_scale(
    x: torch.Tensor, bits: int, per_row: bool = False
) -> torch.Tensor:
    """
    Symmetric scale mapping the largest magnitude to the top code, over the
    whole tensor or, with 'per_row', over each row's last dimension.
    """

    mag = x.detach().abs()
    peak = mag.amax(dim=-1, keepdim=True) if per_row else mag.amax()

    return peak.clamp_min(_QUANT_EPS) / _qmax(bits)


def fake_quantize(
    x: torch.Tensor, bits: int, per_row: bool = False
) -> torch.Tensor:
    """
    Symmetric fake-quant with a straight-through estimator.

    'bits' outside 1..31 is a pass-through, which is how the wide
    (48-bit) interferometer input is represented at full precision in
    simulation. The rounding is non-differentiable, so the gradient is
    passed straight through to 'x'. 'per_row' gives each batch row its own
    scale, which is the deployed single-sample behaviour and keeps one
    diverged row from coarsening or poisoning the others.
    """

    if bits <= 0 or bits >= 32:
        return x
    qmax = _qmax(bits)
    scale = _quant_scale(x, bits, per_row)
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


def soft_clip(x: torch.Tensor, limit: float | None) -> torch.Tensor:
    """
    'limit * tanh(x / limit)': linear well inside the limit, saturating
    smoothly at it. None passes 'x' through.
    """

    if limit is None:
        return x

    return limit * torch.tanh(x / limit)


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
        error_clip: float | None = None,
        command: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Build the axis-major feature vector '[B, A * count]' from
        '[B, A]' inputs, the error soft-clipped by 'error_clip' if set.
        'command' is the previous output, required by that feature.
        """

        if "command" in self.names and command is None:
            raise ValueError("The command feature needs the previous command")

        available = {
            "position": position,
            "velocity": velocity,
            "reference": reference,
            "reference_velocity": reference_velocity,
            "error": soft_clip(reference - position, error_clip),
            "velocity_error": reference_velocity - velocity,
        }
        if command is not None:
            available["command"] = command
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
    # Fixed-point step of the layer's input activation, None when it is
    # unquantised (the wide interferometer input) or was never observed.
    act_scale: float | None = None
    act_bits: int | None = None


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
                    "act_scale": layer.act_scale,
                    "act_bits": layer.act_bits,
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
        # Per-layer activation range for the fixed scales, learnt while
        # training. Zero means not yet observed: that layer falls back to a
        # per-row scale, which is how bundles saved before it behave.
        self.register_buffer("act_amax", torch.zeros(len(dims) - 1))
        # This batch's observed peaks, folded into 'act_amax' by
        # 'commit_act_scales'.
        self.register_buffer(
            "act_peak", torch.zeros(len(dims) - 1), persistent=False
        )
        # Optional count of activations clipped by the fixed range, read by
        # the preflight check; off by default so training pays nothing.
        self.track_saturation = False
        self.saturated = torch.zeros(())
        self.observed = torch.zeros(())

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        key = prefix + "act_amax"
        if key not in state_dict:
            state_dict[key] = torch.zeros_like(self.get_buffer("act_amax"))
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    @torch.no_grad()
    def commit_act_scales(self) -> None:
        """
        Fold the batch's observed peaks into the fixed activation ranges.
        """

        amax = self.get_buffer("act_amax")
        peaks = self.get_buffer("act_peak")
        amax.copy_(torch.maximum(amax, peaks))
        peaks.zero_()

    def discard_act_peaks(self) -> None:
        """
        Drop the batch's observed peaks, for a batch that is not trained on.
        """

        self.get_buffer("act_peak").zero_()

    def act_scale(self, i: int) -> float | None:
        """
        Layer 'i' input activation's fixed-point step, None if unquantised
        or not yet observed.
        """

        bits = self.quant[i].act_bits
        amax = float(self.get_buffer("act_amax")[i])
        if bits <= 0 or bits >= 32 or amax <= 0.0:
            return None

        return ACT_HEADROOM * amax / _qmax(bits)

    def _quantize_act(self, x: torch.Tensor, i: int) -> torch.Tensor:
        bits = self.quant[i].act_bits
        if bits <= 0 or bits >= 32:
            return x
        amax = self.get_buffer("act_amax")[i]
        if self.training:
            peaks = self.get_buffer("act_peak")
            with torch.no_grad():
                rows = torch.nan_to_num(
                    x.detach().abs().amax(dim=-1), nan=0.0, posinf=0.0
                )
                peak = torch.quantile(rows.float(), ACT_QUANTILE)
                peaks[i] = torch.maximum(peaks[i], peak)
            # A new peak is covered within the batch that meets it.
            amax = torch.maximum(amax, peaks[i])
        qmax = _qmax(bits)
        if self.track_saturation and float(amax) > 0:
            with torch.no_grad():
                clipped = x.detach().abs() > ACT_HEADROOM * amax
                self.saturated = self.saturated + clipped.sum().cpu()
                self.observed = self.observed + clipped.numel()
        fixed = (ACT_HEADROOM * amax / qmax).clamp_min(_QUANT_EPS)
        scale = torch.where(amax > 0, fixed, _quant_scale(x, bits, True))
        q = torch.clamp(torch.round(x / scale), -qmax, qmax) * scale

        return x + (q - x).detach()

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
            x = self._quantize_act(x, i)
            w = fake_quantize(linear.weight, self.quant[i].weight_bits)
            x = F.linear(x, w, linear.bias)
            if i != last:
                x = F.relu(x)

        return x

    def resource_layers(self) -> list[LayerCost]:
        """
        Per-layer FPGA cost.
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

        for i, (layer, spec) in enumerate(
            zip(self.linears, self.quant, strict=True)
        ):
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
                    act_scale=self.act_scale(i),
                    act_bits=spec.act_bits,
                )
            )

        return BramExport(layers=layers)


def check_safe_range(safe_range: torch.Tensor, n_axes: int) -> None:
    if safe_range.shape != (n_axes, 2):
        raise ValueError(
            f"Safe_range must be [A, 2], got {tuple(safe_range.shape)}"
        )
    lo, hi = safe_range[:, 0], safe_range[:, 1]
    if bool(((lo > 0) | (hi < 0)).any()):
        raise ValueError(
            "Safe_range must contain zero on every axis, a zero command "
            f"is the stage at rest, got {safe_range.tolist()}"
        )


def bounded_command(
    squashed: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor
) -> torch.Tensor:
    """
    Map a tanh output in (-1, 1) onto the per-axis safe range [lo, hi].

    Each sign scales to its own limit, so a zero output is a zero command
    however asymmetric the range is, and each side saturates at its own
    limit.
    """

    lo = lo.to(squashed.device, squashed.dtype)
    hi = hi.to(squashed.device, squashed.dtype)

    return torch.where(squashed >= 0, hi, -lo) * squashed


class NNPolicy:
    """
    Adapt a 'ControllerNet' to the plant's
    '(position, velocity, reference) -> dac' policy contract, bounding
    the command to the per-axis DAC safe range.

    The net's raw output is mapped to 'safe_range' with a 'tanh', each
    sign scaled to its own limit (see 'bounded_command'). Unlike a hard
    clamp this keeps a live gradient everywhere, so the
    policy can always be improved: a hard clamp saturates and zeroes the
    policy gradient once the command hits a rail.
    """

    def __init__(
        self,
        net: ControllerNet,
        features: FeatureSpec,
        safe_range: torch.Tensor,
        n_axes: int,
        integral_lambdas: Sequence[float] | None = None,
        error_clip: float | None = None,
        command_backprop_steps: int = 0,
    ):
        taps = len(integral_lambdas) if integral_lambdas else 0
        if command_backprop_steps < 0:
            raise ValueError("Command_backprop_steps must be >= 0")
        # How many steps the gradient may run back through the fed-back
        # command before it is cut, 0 treating it as a measured input. The
        # plant and the policy already form a long recurrence, and a second
        # loop through the controller's own output lets gradients compound.
        self.command_backprop_steps = command_backprop_steps
        self._step = 0
        if error_clip is not None and error_clip <= 0:
            raise ValueError(f"Error_clip must be > 0, got {error_clip}")
        self.error_clip = error_clip
        expected = n_axes * (features.count + taps)
        if net.dims[0] != expected:
            raise ValueError(
                f"Controller input {net.dims[0]} != axes*(features+integral) "
                f"{expected}"
            )
        check_safe_range(safe_range, n_axes)
        self.net = net
        self.features = features
        self.n_axes = n_axes
        self._lo = safe_range[:, 0]
        self._hi = safe_range[:, 1]

        # Leaky-integrator error memory:
        # per-axis EMA of the tracking error at each decay lambda,
        # giving the memoryless MLP low-frequency state.
        self._lambda = (
            torch.tensor([float(x) for x in integral_lambdas])
            if integral_lambdas
            else None
        )
        self._s: torch.Tensor | None = None
        # The last command issued, fed back when 'command' is a feature.
        self._u: torch.Tensor | None = None

    def reset(self, batch: int, device: torch.device | str) -> None:
        """Clear the integrator state at the start of a rollout."""

        if self._lambda is not None:
            self._s = torch.zeros(
                batch, self.n_axes, self._lambda.numel(), device=device
            )
        self._u = torch.zeros(batch, self.n_axes, device=device)
        self._step = 0

    def prime(self, command: torch.Tensor) -> None:
        """
        Start from the command [B, A] already holding the stage, so taking
        over is bumpless.
        """

        self._u = command

    def _augment(self, x: torch.Tensor, error: torch.Tensor) -> torch.Tensor:
        """Advance the integrator and append its taps to the features."""

        lam = cast(torch.Tensor, self._lambda).to(error.device, error.dtype)

        if self._s is None:
            self.reset(error.shape[0], error.device)

        s = cast(torch.Tensor, self._s).to(error.dtype)  # [B, A, n_tau]
        s = lam * s + (1.0 - lam) * error[..., None]  # unit-DC-gain EMA
        self._s = s
        base = x.reshape(x.shape[0], self.n_axes, -1)  # [B, A, F]

        return torch.cat([base, s], dim=-1).reshape(x.shape[0], -1)

    def __call__(
        self,
        position: torch.Tensor,
        velocity: torch.Tensor,
        reference: torch.Tensor,
        reference_velocity: torch.Tensor,
    ) -> torch.Tensor:
        if self._u is None:
            self._u = torch.zeros_like(position)
        if self._u.shape != position.shape:
            raise ValueError(
                f"Previous command {tuple(self._u.shape)} does not match the "
                f"batch {tuple(position.shape)}, reset the policy first"
            )
        x = self.features.assemble(
            position,
            velocity,
            reference,
            reference_velocity,
            self.error_clip,
            self._u.to(position.dtype),
        )
        if self._lambda is not None:
            # The taps integrate the clipped error, so a long move cannot
            # wind them up.
            x = self._augment(
                x, soft_clip(reference - position, self.error_clip)
            )

        raw = self.net(x)
        command = bounded_command(torch.tanh(raw), self._lo, self._hi)
        self._step += 1
        n = self.command_backprop_steps
        cut = n == 0 or self._step % n == 0
        self._u = command.detach() if cut else command

        return command
