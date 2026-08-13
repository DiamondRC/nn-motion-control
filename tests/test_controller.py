"""
control.controller: fake-quant, coupled MLP, BRAM export and the
DAC-clamping policy.
"""

import json

import pytest
import torch

from nn_motion_control.control.controller import (
    ControllerNet,
    FeatureSpec,
    NNPolicy,
    fake_quantize,
    quantize_to_int,
)
from nn_motion_control.control.resource import QuantSpec, score_controller


def _net(
    in_features=6, out_features=2, hidden=(8,), weight_bits=16, act_bits=16
):
    quant = [QuantSpec(weight_bits, 48)] + [
        QuantSpec(weight_bits, act_bits) for _ in hidden
    ]
    torch.manual_seed(0)
    return ControllerNet(in_features, out_features, list(hidden), quant)


def test_feature_assembly_is_axis_major():
    spec = FeatureSpec(("position", "velocity", "error"))
    pos = torch.tensor([[1.0, 2.0]])  # [B=1, A=2]
    vel = torch.tensor([[3.0, 4.0]])
    ref = torch.tensor([[10.0, 20.0]])
    x = spec.assemble(pos, vel, ref, torch.zeros_like(ref))
    # Axis-major: [x_pos, x_vel, x_err, y_pos, y_vel, y_err]; error = ref - pos.
    assert x.shape == (1, 6)
    assert torch.equal(x[0], torch.tensor([1.0, 3.0, 9.0, 2.0, 4.0, 18.0]))


def test_feedforward_features_assembled():
    spec = FeatureSpec(("reference_velocity", "velocity_error"))
    pos = torch.zeros(1, 1)
    vel = torch.tensor([[2.0]])
    ref = torch.zeros(1, 1)
    ref_v = torch.tensor([[5.0]])
    x = spec.assemble(pos, vel, ref, ref_v)  # [B=1, A*F=2]
    # reference_velocity = 5, velocity_error = ref_v - vel = 3.
    assert torch.equal(x[0], torch.tensor([5.0, 3.0]))


def test_feature_spec_rejects_unknown():
    with pytest.raises(ValueError, match="Unknown controller features"):
        FeatureSpec(("position", "acceleration"))


def test_controller_forward_shape():
    net = _net(in_features=6, out_features=2, hidden=(8, 8))
    out = net(torch.randn(5, 6))
    assert out.shape == (5, 2)


def test_fake_quantize_passthrough_and_grid():
    x = torch.randn(1000)
    # 32-bit (and wider) is a full-precision pass-through.
    assert torch.equal(fake_quantize(x, 32), x)
    # 4-bit lands every value on the symmetric integer grid.
    q = fake_quantize(x, 4)
    codes, scale = quantize_to_int(x, 4)
    assert torch.allclose(q, codes.to(x.dtype) * scale, atol=1e-6)
    assert (
        int(codes.abs().max()) <= (1 << 3) - 1
    )  # symmetric 4-bit -> |code| <= 7


def test_fake_quantize_straight_through_gradient():
    w = torch.randn(20, requires_grad=True)
    fake_quantize(w, 8).sum().backward()
    # STE routes a unit gradient straight through the rounding.
    assert w.grad is not None
    assert torch.allclose(w.grad, torch.ones_like(w))


def test_policy_bounds_to_safe_range():
    net = _net(in_features=6, out_features=2, hidden=(8,))
    with torch.no_grad():
        for param in net.parameters():  # push the tanh head into saturation
            param.mul_(1000.0)
    safe = torch.tensor([[-1.0, 1.0], [-2.0, 3.0]])
    policy = NNPolicy(
        net, FeatureSpec(("position", "velocity", "error")), safe, 2
    )
    dac = policy(
        torch.randn(7, 2),
        torch.randn(7, 2),
        torch.randn(7, 2),
        torch.randn(7, 2),
    )
    assert dac.shape == (7, 2)
    # tanh bounds the command inside the per-axis safe range.
    assert torch.all(dac[:, 0] >= -1.0) and torch.all(dac[:, 0] <= 1.0)
    assert torch.all(dac[:, 1] >= -2.0) and torch.all(dac[:, 1] <= 3.0)


def test_input_normalization_is_applied():
    net = _net(in_features=6, out_features=2, hidden=(8,))
    x = torch.randn(3, 6)
    base = net(x).clone()
    with torch.no_grad():
        net.get_buffer("feat_mean").add_(5.0)  # shift the input normalisation
    assert not torch.allclose(base, net(x))  # forward actually uses the buffer


def test_policy_is_differentiable_through_net():
    net = _net(in_features=6, out_features=2, hidden=(8,))
    safe = torch.tensor([[-1e6, 1e6], [-1e6, 1e6]])  # wide so nothing saturates
    policy = NNPolicy(
        net, FeatureSpec(("position", "velocity", "error")), safe, 2
    )
    dac = policy(
        torch.randn(4, 2),
        torch.randn(4, 2),
        torch.randn(4, 2),
        torch.randn(4, 2),
    )
    dac.pow(2).sum().backward()
    grads = [p.grad for p in net.parameters() if p.grad is not None]
    assert grads and any(g.abs().sum() > 0 for g in grads)


def test_policy_rejects_mismatched_shapes():
    net = _net(in_features=6, out_features=2, hidden=(8,))
    with pytest.raises(ValueError, match="axes\\*features"):
        NNPolicy(
            net, FeatureSpec(("position",)), torch.zeros(2, 2), 2
        )  # 2*1 != 6
    with pytest.raises(ValueError, match="Safe_range must be"):
        NNPolicy(
            net,
            FeatureSpec(("position", "velocity", "error")),
            torch.zeros(3),
            2,
        )


def test_resource_layers_feed_score_controller():
    net = _net(in_features=6, out_features=2, hidden=(8, 8))
    layers = net.resource_layers()
    assert [(la.in_features, la.out_features) for la in layers] == [
        (6, 8),
        (8, 8),
        (8, 2),
    ]
    report = score_controller(layers, servo_rate_hz=10_000)
    assert report.macs == 6 * 8 + 8 * 8 + 8 * 2
    assert layers[0].dsp_per_mac == 2  # wide 48-bit input layer


def test_bram_export_round_trips_json():
    net = _net(in_features=6, out_features=2, hidden=(8,), weight_bits=8)
    export = net.export_bram()
    blob = json.dumps(
        export.to_dict()
    )  # must be JSON-serialisable (no tensors)
    back = json.loads(blob)
    assert len(back["layers"]) == 2
    first = back["layers"][0]
    assert len(first["weight_codes"]) == 8  # out neurons of layer 0
    assert len(first["weight_codes"][0]) == 6  # in features
    assert (
        max(abs(c) for row in first["weight_codes"] for c in row)
        <= (1 << 7) - 1
    )
