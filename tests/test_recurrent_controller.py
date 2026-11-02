"""
control.recurrent: the error-state bank, its readouts, the policy
contract, config dispatch and checkpoint round-trip.
"""

import json
import math
from pathlib import Path

import pytest
import torch
from torch import nn

from nn_motion_control.control.config import (
    Controller,
    ControllerConfig,
    build_controller_net,
    build_policy,
    save_controller_checkpoint,
)
from nn_motion_control.control.controller import FeatureSpec
from nn_motion_control.control.recurrent import (
    ErrorModes,
    RecurrentControllerNet,
    RecurrentPolicy,
    RecurrentSpec,
)
from nn_motion_control.control.resource import QuantSpec, score_controller
from nn_motion_control.data.normalize import NormStats
from nn_motion_control.plant.plant import Plant, RolloutLayout

REPO = Path(__file__).resolve().parents[1]
SYSTEM = REPO / "examples/deltabot/system.toml"
PLANT_CFG = REPO / "examples/deltabot/configs/plant_tcn.json"
SERVO_HZ = 10000.0
# Coefficient width at or above 32 bits is a fake-quant pass-through, for
# tests that check the exact recurrence.
EXACT_BITS = 32
QUANT = [QuantSpec(32, 48), QuantSpec(32, 16)]


def _modes(**spec) -> ErrorModes:
    return ErrorModes(1, RecurrentSpec(**spec), SERVO_HZ)


def _run(modes, errors, deficit=0.0):
    state = torch.zeros(1, 1, modes.spec.per_axis_states)
    coeffs = modes.coefficients()
    d = torch.full((1, 1), deficit)

    for e in errors:
        state = modes.step(torch.full((1, 1), float(e)), state, d, coeffs)

    return state[0, 0].detach()


def test_integrator_is_exact():
    # A constant unit error charges the integrator linearly with no leak.
    modes = _modes(n_leaky=0, n_resonant=0, coeff_bits=EXACT_BITS)
    steps = 500
    state = _run(modes, [1.0] * steps)
    assert math.isclose(
        float(state[0]), steps * modes.integrator_gain, rel_tol=1e-5
    )


def test_leaky_modes_are_unit_dc_gain():
    modes = _modes(n_leaky=2, n_resonant=0, leaky_tau_s=(1e-3, 5e-3))
    state = _run(modes, [2.0] * 5000)
    assert torch.allclose(state[1:3], torch.full((2,), 2.0), atol=1e-3)


def test_resonant_frequency_is_capped():
    modes = _modes(n_leaky=0, n_resonant=2, resonant_max_hz=25.0)
    with torch.no_grad():
        modes.freq_logit.fill_(50.0)  # sigmoid -> 1, the cap
    freqs = modes.describe()["resonant_hz"][0]
    assert all(f <= 25.0 + 1e-3 for f in freqs)


def test_resonant_pair_has_unit_gain_at_resonance():
    modes = _modes(
        n_leaky=0,
        n_resonant=1,
        resonant_max_hz=25.0,
        resonant_tau_s=0.02,
        coeff_bits=EXACT_BITS,
    )
    f0 = modes.describe()["resonant_hz"][0][0]
    n = 4000  # many time constants, past the transient
    drive = [math.sin(2 * math.pi * f0 * k / SERVO_HZ) for k in range(n)]
    coeffs = modes.coefficients()
    state = torch.zeros(1, 1, 3)
    peak = 0.0

    for k, e in enumerate(drive):
        state = modes.step(
            torch.full((1, 1), e), state, torch.zeros(1, 1), coeffs
        )
        if k > n // 2:
            peak = max(peak, float(state[0, 0, 1:].detach().norm()))
    assert 0.8 < peak < 1.2


def test_anti_windup_bleeds_the_integrator():
    modes = _modes(n_leaky=0, n_resonant=0, anti_windup=1.0)
    free = _run(modes, [1.0] * 100)
    bled = _run(modes, [1.0] * 100, deficit=0.5)
    assert float(bled[0]) < float(free[0])


def test_coefficients_sit_on_the_quantisation_grid():
    modes = _modes(n_leaky=3, n_resonant=2, coeff_bits=8)
    alpha = modes.coefficients()["alpha"].detach()
    scale = float(alpha.abs().max()) / (2**7 - 1)
    codes = alpha / scale
    assert torch.allclose(codes, codes.round(), atol=1e-4)


def _net(readout="mlp", n_leaky=1, n_resonant=1, features=("error",)):
    spec = RecurrentSpec(
        n_leaky=n_leaky, n_resonant=n_resonant, readout=readout
    )
    torch.manual_seed(0)

    return RecurrentControllerNet(
        1, len(features), spec, SERVO_HZ, [8], QUANT, error_scale=1.0
    )


def test_policy_contract_bounds_and_resets():
    net = _net()
    policy = RecurrentPolicy(
        net, FeatureSpec(("error",)), torch.tensor([[-1.0, 1.0]]), 1
    )
    z = torch.zeros(4, 1)
    policy.reset(4, "cpu")
    dac = policy(z, z, torch.full((4, 1), 50.0), z)
    assert dac.shape == (4, 1) and torch.all(dac.abs() <= 1.0)
    assert policy._state is not None and policy._state.abs().sum() > 0
    policy.reset(4, "cpu")
    assert float(policy._state.abs().sum()) == 0.0


@pytest.mark.parametrize("readout", ["mlp", "linear+mlp"])
def test_rollout_is_differentiable_through_the_bank(readout):
    net = _net(readout=readout)
    policy = RecurrentPolicy(
        net, FeatureSpec(("error",)), torch.tensor([[-1.0, 1.0]]), 1
    )
    policy.reset(3, "cpu")
    ref = torch.randn(3, 1)
    z = torch.zeros(3, 1)
    loss = torch.stack([policy(z, z, ref, z) for _ in range(6)]).pow(2).mean()
    loss.backward()
    params = [net.modes.log_rate, net.modes.log_decay, net.modes.freq_logit]
    if readout == "linear+mlp":
        assert net.gain is not None
        params.append(net.gain)
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in params)


class _BiasedDacPlant(nn.Module):
    """Delta-P = gain * (last-frame DAC + bias): a constant input bias."""

    def __init__(self, gain: float, bias: float):
        super().__init__()
        self.gain, self.bias = gain, bias

    def forward(self, window):  # DAC is feature column 2
        return self.gain * (window[:, 2:3, -1] + self.bias)


def _biased_plant(bias: float) -> Plant:
    layout = RolloutLayout(
        pos_cols=[0], vel_cols=[1], dac_cols=[2], n_features=3
    )
    ones = torch.ones(3)
    in_stats = NormStats(
        mean=torch.zeros(3), std=ones, normalizable=ones.bool()
    )
    t_stats = NormStats(
        mean=torch.zeros(1),
        std=torch.ones(1),
        normalizable=torch.ones(1).bool(),
    )
    return Plant(
        _BiasedDacPlant(1.0, bias), in_stats, t_stats, layout, device="cpu"
    )


def _pi_policy(kp: float, ki: float) -> RecurrentPolicy:
    # A hand-set PI law on the linear readout, residual MLP silenced.
    net = _net(readout="linear+mlp", n_leaky=0, n_resonant=0)
    with torch.no_grad():
        last = net.mlp.linears[-1]
        assert isinstance(last, nn.Linear)
        last.weight.zero_()
        last.bias.zero_()
        assert net.gain is not None
        net.gain.zero_()
        net.gain[0, 0] = kp  # error column
        net.gain[0, 1] = ki  # integrator column
    safe = torch.tensor([[-10.0, 10.0]])

    return RecurrentPolicy(net, FeatureSpec(("error",)), safe, 1)


def test_integrator_rejects_a_constant_bias_in_closed_loop():
    # On a biased plant a proportional law settles at the textbook offset,
    # while the exact integrator drives the steady-state error to zero.
    bias, kp, horizon = 0.2, 0.05, 3000
    plant = _biased_plant(bias)
    warmup = torch.zeros(2, 3, 4)
    ref = torch.zeros(2, horizon, 1)

    with torch.no_grad():
        p_only, _ = plant.closed_loop_rollout(
            warmup, ref, _pi_policy(kp, 0.0), horizon
        )
        pi, _ = plant.closed_loop_rollout(
            warmup, ref, _pi_policy(kp, 1.0), horizon
        )
    p_err = float((p_only[:, -1, 0] - ref[:, -1, 0]).abs().mean())
    pi_err = float((pi[:, -1, 0] - ref[:, -1, 0]).abs().mean())
    # safe half-range 10 scales the readout into DAC units near tanh's
    # linear region, so the P-only offset is bias / (10 * kp).
    assert math.isclose(p_err, bias / (10.0 * kp), rel_tol=0.05)
    assert pi_err < 1e-3


def test_feature_stats_error_scale_and_state_columns():
    plant = _biased_plant(0.0)
    mean, std = plant.feature_stats(
        ("error", "position"), n_integral=1, n_state=2, error_scale=5.0
    )
    assert mean.shape == std.shape == (1, 5)
    assert torch.allclose(std[0], torch.tensor([5.0, 1.0, 5.0, 1.0, 1.0]))


def _write_config(tmp_path, controller_overrides):
    ctrl = {
        "architecture": "recurrent",
        "error_scale": 100.0,
        "recurrent": {"n_leaky": 2, "n_resonant": 1, "readout": "linear+mlp"},
        "features": ["position", "velocity", "error"],
        "hidden": [8],
        "quant": [
            {"weight_bits": 16, "act_bits": 48},
            {"weight_bits": 16, "act_bits": 16},
        ],
        "servo_rate_hz": 10000,
        "batch_size": 32,
        "learning_rate": 1e-3,
    }
    ctrl.update(controller_overrides)
    cfg = {
        "model_name": "test_recurrent",
        "system": str(SYSTEM),
        "run": {"out_dir": str(tmp_path), "seed": 7},
        "plant": {
            "config": str(PLANT_CFG),
            "checkpoint": str(tmp_path / "plant.pth"),
        },
        "controller": ctrl,
    }
    path = tmp_path / "controller.json"
    path.write_text(json.dumps(cfg))

    return str(path)


def test_config_sizes_the_recurrent_input(tmp_path):
    config = ControllerConfig(_write_config(tmp_path, {}))
    assert config.architecture == "recurrent"
    assert config.recurrent is not None
    states = 1 + 2 + 2 * 1  # integrator, 2 leaky, 1 resonant pair
    assert config.n_state == states
    assert config.in_features == 3 * (3 + states)


def test_config_rejects_taps_and_missing_scale(tmp_path):
    with pytest.raises(ValueError, match="drop integral_taus"):
        ControllerConfig(_write_config(tmp_path, {"integral_taus": [0.01]}))
    with pytest.raises(ValueError, match="needs an error_scale"):
        ControllerConfig(_write_config(tmp_path, {"error_scale": None}))


def test_checkpoint_round_trip_and_cost(tmp_path):
    config = ControllerConfig(_write_config(tmp_path, {}))
    net = build_controller_net(config)
    policy = build_policy(config, net)
    assert isinstance(net, RecurrentControllerNet)
    assert isinstance(policy, RecurrentPolicy)
    report = score_controller(net.resource_layers(), config.servo_rate_hz)
    assert report.fits
    # MLP layers, the recurrence and the parallel linear readout.
    assert len(report.layers) == len(config.hidden) + 1 + 2

    ckpt = str(tmp_path / "ctrl.pth")
    save_controller_checkpoint(ckpt, net, config, report.as_dict())
    sidecar = json.loads(Path(ckpt).with_suffix(".export.json").read_text())
    assert sidecar["architecture"] == "recurrent"
    assert len(sidecar["poles"]["leaky_tau_ms"]) == 3  # one row per axis

    loaded = Controller.from_checkpoint(ckpt)
    assert isinstance(loaded.policy, RecurrentPolicy)

    for k, v in net.state_dict().items():
        assert torch.allclose(v, loaded.net.state_dict()[k])
    z = torch.zeros(2, 3)
    assert loaded.policy(z, z, z, z).shape == (2, 3)
