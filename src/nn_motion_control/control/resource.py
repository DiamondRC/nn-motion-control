"""
FPGA resource model for the neural controller (torch-free).

Scores a controller architecture against the PandA MAC-engine datapath:
the DSP cycles a forward pass costs, the BRAM bits its weights occupy,
and the maximum servo rate the hardware can sustain. Pure Python so it
runs without torch or a checkpoint and is exercised directly by
doctests. The controller and CLI feed it layer dimensions.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

# Feature roles a controller may consume: position/velocity come from the
# plant state, reference/reference_velocity are the PVT demand, error is
# reference - position and velocity_error is reference_velocity -
# velocity (all synthesised at run time).
ALLOWED_FEATURES: tuple[str, ...] = (
    "position",
    "velocity",
    "reference",
    "reference_velocity",
    "error",
    "velocity_error",
)


def num_chunks(width: int, port: int) -> int:
    """
    Number of DSP port-width chunks a signed operand of 'width' bits needs.

    A DSP48E1 multiplies operands up to 'port' bits. Wider operands split
    into chunks that overlap by one sign bit, so a 48-bit operand needs
    two 25-bit chunks.

    >>> num_chunks(18, 18)
    1
    >>> num_chunks(48, 25)
    2
    >>> num_chunks(50, 25)
    3
    """

    if width <= 0:
        raise ValueError("Operand width must be positive")
    if port <= 1:
        raise ValueError("Port width must be at least 2 bits")
    if width <= port:
        return 1

    return 1 + (width - 2) // (port - 1)


@dataclass(frozen=True)
class HardwareModel:
    """
    PandA / XC7Z030 datapath budget for the controller.

    Defaults match the target board: 125 MHz fabric, ~360 usable DSP48E1
    slices, the DSP's native 25x18 operand ports and 48-bit accumulator.
    BRAM capacity and read bandwidth are optional gates (left None when
    the controller owns the whole board).
    """

    clock_hz: float = 125_000_000.0
    dsp_budget: int = 360
    act_port_bits: int = 25  # DSP48E1 25-bit (A/coeff) port
    weight_port_bits: int = 18  # weights map to the 18-bit (B/data) port
    accum_bits: int = 48
    bram_total_bits: int | None = None
    bram_read_bits_per_cycle: int | None = None


# Shared default so functions can take an hw argument without a call in
# their defaults.
_DEFAULT_HW = HardwareModel()


def dsp_per_mac(
    act_bits: int, weight_bits: int, hw: HardwareModel = _DEFAULT_HW
) -> int:
    """
    DSP48E1 cells one multiply-accumulate needs at the given operand widths.

    A wide (48-bit interferometer) activation costs two A-port chunks,
    quantised hidden activations and weights fit one chunk each, so a
    hidden MAC is a single DSP.
    """

    return num_chunks(act_bits, hw.act_port_bits) * num_chunks(
        weight_bits, hw.weight_port_bits
    )


@dataclass(frozen=True)
class QuantSpec:
    """
    Per-layer fixed-point widths: the weight width and the incoming
    activation width.
    """

    weight_bits: int
    act_bits: int

    def __post_init__(self):
        # 1-bit symmetric quantisation is degenerate: qmax =
        # 2**(bits-1) - 1 = 0, which divides by zero in the fake-quant
        # scale and silently produces NaN weights.
        for name, bits in (
            ("weight_bits", self.weight_bits),
            ("act_bits", self.act_bits),
        ):
            if bits < 2:
                raise ValueError(f"{name} must be >= 2, got {bits}")


@dataclass(frozen=True)
class LayerCost:
    """
    Cost of one dense (matmul + bias) controller layer.
    """

    in_features: int
    out_features: int
    weight_bits: int
    act_bits_in: int
    params: int
    macs: int
    dsp_per_mac: int
    dsp_cycles: int
    bram_bits: int

    @classmethod
    def dense(
        cls,
        in_features: int,
        out_features: int,
        weight_bits: int,
        act_bits_in: int,
        hw: HardwareModel = _DEFAULT_HW,
    ) -> LayerCost:
        """
        Cost a dense layer from its dimensions and operand widths.
        """

        if in_features <= 0 or out_features <= 0:
            raise ValueError("Layer dimensions must be positive")
        macs = in_features * out_features
        params = macs + out_features  # weights plus one bias per output
        per_mac = dsp_per_mac(act_bits_in, weight_bits, hw)

        return cls(
            in_features=in_features,
            out_features=out_features,
            weight_bits=weight_bits,
            act_bits_in=act_bits_in,
            params=params,
            macs=macs,
            dsp_per_mac=per_mac,
            dsp_cycles=macs * per_mac,
            bram_bits=params * weight_bits,
        )

    def as_dict(self) -> dict:
        """
        Plain-dict view for JSON sidecars and checkpoint bundles.
        """

        return {
            "in_features": self.in_features,
            "out_features": self.out_features,
            "weight_bits": self.weight_bits,
            "act_bits_in": self.act_bits_in,
            "params": self.params,
            "macs": self.macs,
            "dsp_per_mac": self.dsp_per_mac,
            "dsp_cycles": self.dsp_cycles,
            "bram_bits": self.bram_bits,
        }


def dense_layers(
    dims: Sequence[int],
    weight_bits: Sequence[int],
    act_bits: Sequence[int],
    hw: HardwareModel = _DEFAULT_HW,
) -> list[LayerCost]:
    """
    Build the LayerCosts for an MLP whose widths are 'dims' = [in, h0,
    ..., out].

    'weight_bits' and 'act_bits' give each linear layer's weight width
    and incoming activation width, both have length 'len(dims) - 1' (the
    input layer's 'act_bits' is the wide interferometer width).
    """

    n_layers = len(dims) - 1
    if n_layers < 1:
        raise ValueError(
            "An MLP needs at least an input and an output dimension"
        )
    if len(weight_bits) != n_layers or len(act_bits) != n_layers:
        raise ValueError(
            "Weight_bits and act_bits must have one entry per layer"
        )

    return [
        LayerCost.dense(dims[i], dims[i + 1], weight_bits[i], act_bits[i], hw)
        for i in range(n_layers)
    ]


@dataclass(frozen=True)
class ResourceReport:
    """
    Whether a controller fits the hardware at a servo rate, and by how
    much.

    'cycles_needed' assumes the work parallelises across 'dsp_budget'
    DSPs, 'max_servo_rate_hz' is the rate at which that fully-parallel
    pass just fills the step. 'bram_bits_per_cycle' is a
    weight-read-bandwidth proxy for the same pass.
    """

    layers: tuple[LayerCost, ...]
    params: int
    bram_bits: int
    macs: int
    dsp_cycles: int
    dsp_budget: int
    servo_rate_hz: float
    cycles_available: float
    cycles_needed: int
    max_servo_rate_hz: float
    bram_bits_per_cycle: float
    fits: bool
    reasons: tuple[str, ...]

    def as_dict(self) -> dict:
        """
        Plain-dict view for JSON sidecars and checkpoint bundles.
        """

        return {
            "layers": [layer.as_dict() for layer in self.layers],
            "params": self.params,
            "bram_bits": self.bram_bits,
            "macs": self.macs,
            "dsp_cycles": self.dsp_cycles,
            "dsp_budget": self.dsp_budget,
            "servo_rate_hz": self.servo_rate_hz,
            "cycles_available": self.cycles_available,
            "cycles_needed": self.cycles_needed,
            "max_servo_rate_hz": self.max_servo_rate_hz,
            "bram_bits_per_cycle": self.bram_bits_per_cycle,
            "fits": self.fits,
            "reasons": list(self.reasons),
        }


def score_controller(
    layers: Sequence[LayerCost],
    servo_rate_hz: float,
    hw: HardwareModel = _DEFAULT_HW,
) -> ResourceReport:
    """
    Score a controller's layers against the hardware budget at a servo rate.
    """

    layers = tuple(layers)
    if not layers:
        raise ValueError("Controller must have at least one layer")
    if servo_rate_hz <= 0:
        raise ValueError("Servo rate must be positive")

    params = sum(layer.params for layer in layers)
    bram_bits = sum(layer.bram_bits for layer in layers)
    macs = sum(layer.macs for layer in layers)
    dsp_cycles = sum(layer.dsp_cycles for layer in layers)

    cycles_available = hw.clock_hz / servo_rate_hz
    cycles_needed = math.ceil(dsp_cycles / hw.dsp_budget)
    max_rate = hw.clock_hz * hw.dsp_budget / dsp_cycles
    bram_bits_per_cycle = bram_bits / cycles_available

    reasons: list[str] = []
    fits = True
    if cycles_needed > cycles_available:
        fits = False
        reasons.append(
            f"Needs {cycles_needed} cycles/step "
            f"but only {cycles_available:.0f} available at "
            f"{servo_rate_hz:.0f} Hz (max rate {max_rate:.0f} Hz)"
        )
    if hw.bram_total_bits is not None and bram_bits > hw.bram_total_bits:
        fits = False
        reasons.append(
            f"Weights need {bram_bits} BRAM bits, "
            f"budget is {hw.bram_total_bits}"
        )
    if (
        hw.bram_read_bits_per_cycle is not None
        and bram_bits_per_cycle > hw.bram_read_bits_per_cycle
    ):
        fits = False
        reasons.append(
            f"Needs {bram_bits_per_cycle:.0f} weight bits/cycle, "
            f"bandwidth is {hw.bram_read_bits_per_cycle}"
        )
    if fits:
        reasons.append(
            f"Fits: {cycles_needed} of {cycles_available:.0f} "
            f"cycles/step used at "
            f"{servo_rate_hz:.0f} Hz (max rate {max_rate:.0f} Hz)"
        )

    return ResourceReport(
        layers=layers,
        params=params,
        bram_bits=bram_bits,
        macs=macs,
        dsp_cycles=dsp_cycles,
        dsp_budget=hw.dsp_budget,
        servo_rate_hz=servo_rate_hz,
        cycles_available=cycles_available,
        cycles_needed=cycles_needed,
        max_servo_rate_hz=max_rate,
        bram_bits_per_cycle=bram_bits_per_cycle,
        fits=fits,
        reasons=tuple(reasons),
    )


def max_servo_rate(
    layers: Sequence[LayerCost], hw: HardwareModel = _DEFAULT_HW
) -> float:
    """
    Highest servo rate (Hz) the fully-parallel forward pass can sustain.
    """

    dsp_cycles = sum(layer.dsp_cycles for layer in layers)
    if dsp_cycles <= 0:
        raise ValueError("Controller must have at least one MAC")

    return hw.clock_hz * hw.dsp_budget / dsp_cycles


def fits_rate(
    layers: Sequence[LayerCost],
    servo_rate_hz: float,
    hw: HardwareModel = _DEFAULT_HW,
) -> bool:
    """
    Whether the controller fits the hardware budget at the given servo rate.
    """

    return score_controller(layers, servo_rate_hz, hw).fits
