"""
Recurrent error-state controller: per-axis linear modes driven by the
tracking error, read out by the quantised MLP.

Each axis carries an exact integrator (pole fixed at 1, the internal
model of a constant disturbance, so a steady bias is rejected to zero
error), learned leaky real modes, and learned resonant pairs whose
frequency is capped below the band the plant was identified on (the
internal model of a sinusoid such as an ambient line). The recurrence is
linear and driven only by error, so it stays bounded whenever the loop is
stable, and its poles can be read back from a trained controller. The
error is scaled by a configured error scale rather than the workspace
span, so nanometre errors reach the network well above the fixed-point
floor.

Coefficients are held in decrement form, h <- h - a * (h - e), storing
the small quantity 1 - lambda rather than lambda, so slow poles keep
their precision at the coefficient width.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import cast

import torch
from torch import nn

from nn_motion_control.control.controller import (
    BramExport,
    ControllerNet,
    FeatureSpec,
    bounded_command,
    check_safe_range,
    fake_quantize,
    quantize_to_int,
)
from nn_motion_control.control.resource import (
    LayerCost,
    QuantSpec,
    axis_readout_cost,
    recurrence_cost,
)

READOUTS = ("mlp", "linear+mlp")

DEFAULT_N_LEAKY = 2
DEFAULT_N_RESONANT = 2
# Leaky time constants (s) are spread log-uniformly across this span at
# init, from servo-scale to well past the training horizon.
DEFAULT_LEAKY_TAU_S = (1e-3, 1e-1)
# Resonant frequencies are capped here (Hz): the plant is only identified
# below about 25 Hz, so memory above it would model plant fantasy.
DEFAULT_RESONANT_MAX_HZ = 25.0
DEFAULT_RESONANT_TAU_S = 0.05
# Time (s) for a constant unit error to charge the integrator to unity.
DEFAULT_INTEGRATOR_TAU_S = 0.01
DEFAULT_ANTI_WINDUP = 1.0
DEFAULT_COEFF_BITS = 16

# Resonant magnitudes stay below this cap, bounding a pair's time constant
# (1 s at 10 kHz) so rounding cannot lift a pole onto the unit circle.
RESONANT_MAG_CAP = 0.9999
# Initial resonant frequencies span this fraction band of the cap.
RESONANT_FREQ_INIT_FRAC = (0.1, 0.9)
# States are wide accumulators, matching the DSP48E1 accumulator.
STATE_BITS = 48
# With a parallel linear path the MLP starts as a small residual, so the
# learned linear law leads early training.
RESIDUAL_INIT_SCALE = 0.1


@dataclass(frozen=True)
class RecurrentSpec:
    """
    Shape and initialisation of the per-axis error-state bank.

    Per axis the state is [integrator, leaky modes, resonant (re, im)
    pairs], so 'per_axis_states' = 1 + n_leaky + 2 * n_resonant.
    """

    n_leaky: int = DEFAULT_N_LEAKY
    n_resonant: int = DEFAULT_N_RESONANT
    leaky_tau_s: tuple[float, float] = DEFAULT_LEAKY_TAU_S
    resonant_max_hz: float = DEFAULT_RESONANT_MAX_HZ
    resonant_tau_s: float = DEFAULT_RESONANT_TAU_S
    integrator_tau_s: float = DEFAULT_INTEGRATOR_TAU_S
    anti_windup: float = DEFAULT_ANTI_WINDUP
    readout: str = "mlp"
    coeff_bits: int = DEFAULT_COEFF_BITS

    def __post_init__(self) -> None:
        if self.n_leaky < 0 or self.n_resonant < 0:
            raise ValueError("Mode counts must be non-negative")
        lo, hi = self.leaky_tau_s
        if not 0.0 < lo <= hi:
            raise ValueError("Leaky tau span must satisfy 0 < lo <= hi")
        if self.resonant_max_hz <= 0.0 or self.resonant_tau_s <= 0.0:
            raise ValueError("Resonant cap and time constant must be positive")
        if self.integrator_tau_s <= 0.0:
            raise ValueError("Integrator time constant must be positive")
        if self.anti_windup < 0.0:
            raise ValueError("Anti-windup gain must be non-negative")
        if self.readout not in READOUTS:
            raise ValueError(f"Readout must be one of {READOUTS}")
        if self.coeff_bits < 2:
            raise ValueError("Coefficient bits must be >= 2")

    @property
    def per_axis_states(self) -> int:
        return 1 + self.n_leaky + 2 * self.n_resonant

    @classmethod
    def from_dict(cls, spec: dict) -> RecurrentSpec:
        spec = dict(spec)
        if "leaky_tau_s" in spec:
            lo, hi = spec["leaky_tau_s"]
            spec["leaky_tau_s"] = (float(lo), float(hi))

        return cls(**spec)

    def to_dict(self) -> dict:
        out = asdict(self)
        out["leaky_tau_s"] = list(self.leaky_tau_s)

        return out


def _quantise(x: torch.Tensor, bits: int) -> torch.Tensor:
    # Fake-quant that tolerates an empty mode group (no scale to take).
    return fake_quantize(x, bits) if x.numel() else x


def _codes(x: torch.Tensor, bits: int) -> dict:
    if not x.numel():
        return {"codes": [], "scale": 0.0, "bits": bits}
    codes, scale = quantize_to_int(x.detach(), bits)

    return {"codes": codes.tolist(), "scale": scale, "bits": bits}


class ErrorModes(nn.Module):
    """
    The per-axis linear error-state bank and its quantised coefficients.

    Leaky modes are unit-DC-gain low-passes of the error, resonant pairs
    have unit gain at their resonance, and the integrator reaches unity
    after 'integrator_tau_s' of unit error.
    """

    def __init__(self, n_axes: int, spec: RecurrentSpec, servo_rate_hz: float):
        super().__init__()
        self.n_axes = n_axes
        self.spec = spec
        self.dt = 1.0 / float(servo_rate_hz)
        self.integrator_gain = self.dt / spec.integrator_tau_s

        # A leaky pole lambda = exp(-exp(log_rate)); at init exp(log_rate)
        # is dt / tau so the pole has time constant tau.
        lo, hi = spec.leaky_tau_s
        taus = torch.logspace(
            math.log10(lo), math.log10(hi), spec.n_leaky, dtype=torch.float32
        )
        log_rate = torch.log(self.dt / taus).expand(n_axes, -1).clone()
        self.log_rate = nn.Parameter(log_rate)

        # A resonant magnitude is RESONANT_MAG_CAP * exp(-exp(log_decay)),
        # its frequency resonant_max_hz * sigmoid(freq_logit).
        mag0 = math.exp(-self.dt / spec.resonant_tau_s)
        if mag0 >= RESONANT_MAG_CAP:
            raise ValueError(
                "Resonant time constant exceeds the magnitude cap at this rate"
            )
        log_decay = math.log(-math.log(mag0 / RESONANT_MAG_CAP))
        self.log_decay = nn.Parameter(
            torch.full((n_axes, spec.n_resonant), log_decay)
        )
        frac = torch.linspace(*RESONANT_FREQ_INIT_FRAC, spec.n_resonant)
        freq_logit = torch.log(frac / (1.0 - frac)).expand(n_axes, -1)
        self.freq_logit = nn.Parameter(freq_logit.clone())

    def coefficients(self) -> dict[str, torch.Tensor]:
        """
        Quantised coefficients, each [A, n]: the leaky decrement 'alpha'
        (1 - lambda), and per resonant pair its decrement 'dec'
        (1 - |lambda| cos theta), rotation 'rot' (|lambda| sin theta) and
        input 'gain' (1 - |lambda|).
        """

        bits = self.spec.coeff_bits
        alpha = -torch.expm1(-torch.exp(self.log_rate))
        mag = RESONANT_MAG_CAP * torch.exp(-torch.exp(self.log_decay))
        freq = self.spec.resonant_max_hz * torch.sigmoid(self.freq_logit)
        theta = 2.0 * math.pi * freq * self.dt
        # 1 - |lambda| cos theta without cancellation for poles near 1.
        dec = (1.0 - mag) + 2.0 * mag * torch.sin(theta / 2.0) ** 2

        return {
            "alpha": _quantise(alpha, bits),
            "dec": _quantise(dec, bits),
            "rot": _quantise(mag * torch.sin(theta), bits),
            "gain": _quantise(1.0 - mag, bits),
        }

    def step(
        self,
        error: torch.Tensor,
        state: torch.Tensor,
        deficit: torch.Tensor,
        coeffs: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Advance the bank one step: error and deficit [B, A] (scaled error
        and the tanh deficit of the last command), state [B, A, S].
        """

        n_leaky, n_res = self.spec.n_leaky, self.spec.n_resonant
        b = state.shape[0]
        integ = state[..., 0] + self.integrator_gain * (
            error - self.spec.anti_windup * deficit
        )
        leaky = state[..., 1 : 1 + n_leaky]
        leaky = leaky - coeffs["alpha"] * (leaky - error[..., None])
        pairs = state[..., 1 + n_leaky :].reshape(b, self.n_axes, n_res, 2)
        re, im = pairs[..., 0], pairs[..., 1]
        new_re = (
            re
            - coeffs["dec"] * re
            - coeffs["rot"] * im
            + coeffs["gain"] * error[..., None]
        )
        new_im = im - coeffs["dec"] * im + coeffs["rot"] * re
        res = torch.stack([new_re, new_im], dim=-1).reshape(
            b, self.n_axes, 2 * n_res
        )

        return torch.cat([integ[..., None], leaky, res], dim=-1)

    def describe(self) -> dict:
        """
        The quantised poles as time constants (ms) and frequencies (Hz),
        per axis: what the deployed hardware would implement.
        """

        with torch.no_grad():
            c = self.coefficients()
            leaky_tau = -self.dt / torch.log1p(-c["alpha"])
            re = 1.0 - c["dec"]
            mag = torch.sqrt(re**2 + c["rot"] ** 2)
            freq = torch.atan2(c["rot"], re) / (2.0 * math.pi * self.dt)
            res_tau = -self.dt / torch.log(mag)
        ms = 1e3

        return {
            "integrator_tau_ms": self.spec.integrator_tau_s * ms,
            "leaky_tau_ms": (leaky_tau * ms).tolist(),
            "resonant_hz": freq.tolist(),
            "resonant_tau_ms": (res_tau * ms).tolist(),
        }

    def export(self) -> dict:
        """
        Coefficients as integer codes plus scales, for the sidecar.
        """

        bits = self.spec.coeff_bits
        with torch.no_grad():
            c = self.coefficients()

        return {
            "layout": "per-axis [integrator, leaky, resonant (re, im)]",
            "integrator_gain": self.integrator_gain,
            "anti_windup": self.spec.anti_windup,
            **{name: _codes(value, bits) for name, value in c.items()},
        }


@dataclass(frozen=True)
class RecurrentExport:
    """
    The recurrent controller's deployable coefficients.
    """

    readout: BramExport
    modes: dict
    linear: dict | None

    def to_dict(self) -> dict:
        return {
            "architecture": "recurrent",
            "readout": self.readout.to_dict(),
            "modes": self.modes,
            "linear": self.linear,
        }


class RecurrentControllerNet(nn.Module):
    """
    Error-state bank plus quantised readout.

    The readout MLP sees the instantaneous features and every axis's
    states, axis-major. With 'linear+mlp' a per-axis linear law over
    [scaled error, states] runs in parallel and the MLP is a residual.
    """

    def __init__(
        self,
        n_axes: int,
        n_features: int,
        spec: RecurrentSpec,
        servo_rate_hz: float,
        hidden: Sequence[int],
        quant: Sequence[QuantSpec],
        error_scale: float,
        feat_mean: torch.Tensor | None = None,
        feat_std: torch.Tensor | None = None,
    ):
        super().__init__()
        if error_scale <= 0.0:
            raise ValueError("Error scale must be positive")
        self.n_axes = n_axes
        self.n_features = n_features
        self.spec = spec
        self.modes = ErrorModes(n_axes, spec, servo_rate_hz)
        width = n_features + spec.per_axis_states
        self.mlp = ControllerNet(
            n_axes * width,
            n_axes,
            hidden,
            quant,
            feat_mean=feat_mean,
            feat_std=feat_std,
        )
        self.register_buffer("error_scale", torch.tensor(float(error_scale)))

        self.gain: nn.Parameter | None = None
        if spec.readout == "linear+mlp":
            self.gain = nn.Parameter(
                torch.zeros(n_axes, 1 + spec.per_axis_states)
            )
            last = cast(nn.Linear, self.mlp.linears[-1])
            with torch.no_grad():
                last.weight.mul_(RESIDUAL_INIT_SCALE)
                last.bias.mul_(RESIDUAL_INIT_SCALE)

    @property
    def input_act_bits(self) -> int:
        return self.mlp.input_act_bits

    def forward(
        self, x: torch.Tensor, error_state: torch.Tensor
    ) -> torch.Tensor:
        """
        x [B, A * (F + S)] features and states, error_state [B, A, 1 + S]
        (scaled error then states) to the raw pre-squash command [B, A].
        """

        raw = self.mlp(x)
        if self.gain is not None:
            gain = fake_quantize(self.gain, self.spec.coeff_bits)
            raw = raw + (gain * error_state).sum(dim=-1)

        return raw

    def resource_layers(self) -> list[LayerCost]:
        layers = self.mlp.resource_layers()
        layers.append(
            recurrence_cost(
                self.n_axes,
                self.spec.n_leaky,
                self.spec.n_resonant,
                self.spec.coeff_bits,
                STATE_BITS,
            )
        )
        if self.gain is not None:
            layers.append(
                axis_readout_cost(
                    self.n_axes,
                    1 + self.spec.per_axis_states,
                    self.spec.coeff_bits,
                    STATE_BITS,
                )
            )

        return layers

    def export_bram(self) -> RecurrentExport:
        linear = None
        if self.gain is not None:
            linear = _codes(self.gain, self.spec.coeff_bits)

        return RecurrentExport(
            readout=self.mlp.export_bram(),
            modes=self.modes.export(),
            linear=linear,
        )


class RecurrentPolicy:
    """
    Adapt a 'RecurrentControllerNet' to the plant's policy contract,
    carrying the error-state bank across steps and bounding the command
    to the DAC safe range with the same tanh guard as the MLP policy.

    The tanh deficit (raw - tanh(raw)) of each command feeds back into the
    integrator as back-calculation anti-windup: it is near zero while the
    command is in the linear region and grows only under saturation.
    """

    def __init__(
        self,
        net: RecurrentControllerNet,
        features: FeatureSpec,
        safe_range: torch.Tensor,
        n_axes: int,
    ):
        width = features.count + net.spec.per_axis_states
        if net.mlp.dims[0] != n_axes * width:
            raise ValueError(
                f"Controller input {net.mlp.dims[0]} != axes*(features+states)"
                f" {n_axes * width}"
            )
        check_safe_range(safe_range, n_axes)
        self.net = net
        self.features = features
        self.n_axes = n_axes
        self._lo, self._hi = safe_range[:, 0], safe_range[:, 1]
        self._state: torch.Tensor | None = None
        self._deficit: torch.Tensor | None = None
        self._coeffs: dict[str, torch.Tensor] | None = None

    def reset(self, batch: int, device: torch.device | str) -> None:
        """
        Clear the error states and snapshot the coefficients (the weights
        are fixed within a rollout, so the bank reuses one quantisation).
        """

        states = self.net.spec.per_axis_states
        self._state = torch.zeros(batch, self.n_axes, states, device=device)
        self._deficit = torch.zeros(batch, self.n_axes, device=device)
        self._coeffs = self.net.modes.coefficients()

    def __call__(
        self,
        position: torch.Tensor,
        velocity: torch.Tensor,
        reference: torch.Tensor,
        reference_velocity: torch.Tensor,
    ) -> torch.Tensor:
        if self._state is None:
            self.reset(position.shape[0], position.device)
        state = cast(torch.Tensor, self._state)
        deficit = cast(torch.Tensor, self._deficit)
        coeffs = cast(dict[str, torch.Tensor], self._coeffs)

        x = self.features.assemble(
            position, velocity, reference, reference_velocity
        )
        error = (reference - position) / self.net.get_buffer("error_scale")
        state = self.net.modes.step(error, state, deficit, coeffs)
        self._state = state

        b = x.shape[0]
        base = x.reshape(b, self.n_axes, -1)
        x_in = torch.cat([base, state], dim=-1).reshape(b, -1)
        error_state = torch.cat([error[..., None], state], dim=-1)
        raw = self.net(x_in, error_state)
        squashed = torch.tanh(raw)
        self._deficit = raw - squashed

        return bounded_command(squashed, self._lo, self._hi)
