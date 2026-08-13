"""
control.resource: DSP/BRAM/cycle cost model for the FPGA controller
(torch-free).
"""

import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from nn_motion_control.control.resource import (
    HardwareModel,
    LayerCost,
    QuantSpec,
    dense_layers,
    dsp_per_mac,
    fits_rate,
    max_servo_rate,
    num_chunks,
    score_controller,
)


def test_quantspec_rejects_sub_2_bits():
    # 1-bit symmetric quant gives qmax = 0, which divides by zero in
    # the fake-quant scale and silently yields NaN weights, reject it
    # at construction.
    with pytest.raises(ValueError):
        QuantSpec(weight_bits=1, act_bits=8)
    with pytest.raises(ValueError):
        QuantSpec(weight_bits=8, act_bits=1)
    QuantSpec(weight_bits=2, act_bits=2)  # boundary is valid


def test_num_chunks_known_values():
    # A DSP port fits an operand up to its width in one chunk, wider
    # operands split with a one-bit sign overlap.
    assert num_chunks(18, 18) == 1
    assert num_chunks(25, 25) == 1
    assert (
        num_chunks(48, 25) == 2
    )  # wide interferometer input on the 25-bit A port
    assert num_chunks(50, 25) == 3
    assert num_chunks(26, 25) == 2


def test_num_chunks_rejects_bad_args():
    with pytest.raises(ValueError, match="width must be positive"):
        num_chunks(0, 25)
    with pytest.raises(ValueError, match="Port width"):
        num_chunks(16, 1)


def test_dsp_per_mac_input_vs_hidden():
    # Input layer: 48-bit activation (2 A-chunks) x <=18-bit weight
    # (1 B-chunk) = 2 DSP.
    assert dsp_per_mac(48, 16) == 2
    # Hidden/output: quantised activation and weight each fit one chunk = 1 DSP.
    assert dsp_per_mac(16, 16) == 1
    assert dsp_per_mac(25, 18) == 1
    # A 32-bit weight (the LQR width) would cost a second B-chunk.
    assert dsp_per_mac(16, 32) == 2


def test_layer_cost_hand_computed():
    layer = LayerCost.dense(9, 64, weight_bits=16, act_bits_in=48)
    assert layer.macs == 9 * 64
    assert layer.params == 9 * 64 + 64  # plus one bias per output
    assert layer.dsp_per_mac == 2  # wide input
    assert layer.dsp_cycles == 9 * 64 * 2
    assert layer.bram_bits == (9 * 64 + 64) * 16


def test_score_controller_totals_and_fit():
    layers = dense_layers([9, 64, 3], weight_bits=[16, 16], act_bits=[48, 16])
    report = score_controller(layers, servo_rate_hz=10_000)
    assert report.macs == 576 + 192
    assert report.dsp_cycles == 576 * 2 + 192 * 1
    assert report.params == 640 + 195
    assert report.bram_bits == 640 * 16 + 195 * 16
    assert report.cycles_available == 125_000_000 / 10_000
    assert report.cycles_needed == math.ceil(
        report.dsp_cycles / report.dsp_budget
    )
    assert report.fits
    assert report.reasons and report.reasons[0].startswith("Fits")


def test_same_net_fits_slow_rate_but_flagged_fast():
    # Single-DSP budget forces the DSP bound so the fit flip is deterministic.
    hw = HardwareModel(dsp_budget=1)
    layers = dense_layers([9, 64, 3], weight_bits=[16, 16], act_bits=[48, 16])
    assert fits_rate(layers, 10_000, hw)  # 1344 cycles <= 12500 available
    fast = score_controller(layers, 200_000, hw)  # only 625 available
    assert not fast.fits
    assert any("cycles/step" in r for r in fast.reasons)


def test_bram_capacity_gate():
    layers = dense_layers([9, 64, 3], weight_bits=[16, 16], act_bits=[48, 16])
    tight = HardwareModel(bram_total_bits=100)
    report = score_controller(layers, 10_000, tight)
    assert not report.fits
    assert any("BRAM bits" in r for r in report.reasons)


@given(
    extra=st.integers(min_value=1, max_value=256),
    base=st.integers(min_value=8, max_value=128),
)
@settings(deadline=None, max_examples=40)
def test_max_servo_rate_decreases_with_width(base, extra):
    # A wider hidden layer means more MACs, hence a lower sustainable
    # servo rate.
    narrow = dense_layers([9, base, 3], [16, 16], [48, 16])
    wide = dense_layers([9, base + extra, 3], [16, 16], [48, 16])
    assert max_servo_rate(wide) < max_servo_rate(narrow)


def test_score_controller_rejects_empty_and_bad_rate():
    layers = dense_layers([9, 3], [16], [48])
    with pytest.raises(ValueError, match="Servo rate"):
        score_controller(layers, 0)
    with pytest.raises(ValueError, match="at least one layer"):
        score_controller([], 10_000)
