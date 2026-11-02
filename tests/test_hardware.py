"""control.hardware: the encoder sensor and actuator models in a rollout."""

import torch
import torch.nn as nn

from nn_motion_control.control.closed_loop import PDPolicy, constant_reference
from nn_motion_control.control.hardware import ActuatorModel, SensorModel
from nn_motion_control.data.normalize import NormStats
from nn_motion_control.plant.plant import Plant, RolloutLayout

GRID = 0.256
SERVO_HZ = 10000.0


class _DacDriven(nn.Module):
    """A plant where the command moves the state: delta-P = gain * DAC."""

    def __init__(self, gain: float):
        super().__init__()
        self.gain = gain

    def forward(self, window):  # window [B, F, W]; DAC is feature column 2
        return self.gain * window[:, 2:3, -1]


def _dac_plant(gain=1.0):
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
    return Plant(_DacDriven(gain), in_stats, t_stats, layout, device="cpu")


def test_quantises_to_grid():
    s = SensorModel(GRID)
    s.reset(torch.zeros(1, 1))
    pos = torch.tensor([[0.1, 0.2, 0.13, -0.13]])
    q, _ = s.observe(pos, torch.zeros_like(pos))
    # Each value rounds to the nearest multiple of the grid.
    expected = torch.round(pos / GRID) * GRID
    assert torch.allclose(q, expected)


def test_sub_grid_motion_vanishes():
    # Two positions closer than one least count read as the same code, so
    # the derived velocity is zero: the controller cannot see sub-count drift.
    s = SensorModel(GRID)
    s.reset(torch.zeros(2, 1))
    _, v0 = s.observe(torch.full((2, 1), 0.05), torch.zeros(2, 1))
    _, v1 = s.observe(torch.full((2, 1), 0.10), torch.zeros(2, 1))
    assert float(v0.abs().sum()) == 0.0
    assert float(v1.abs().sum()) == 0.0


def test_velocity_from_quantised_differences():
    s = SensorModel(GRID)
    s.reset(torch.zeros(1, 1))
    s.observe(torch.zeros(1, 1), torch.zeros(1, 1))  # prev = 0
    _, v = s.observe(torch.full((1, 1), 3.0 * GRID), torch.zeros(1, 1))
    assert torch.allclose(v, torch.full((1, 1), 3.0 * GRID))


def test_straight_through_gradient():
    s = SensorModel(GRID)
    s.reset(torch.zeros(1, 1))
    pos = torch.tensor([[0.4]], requires_grad=True)
    q, _ = s.observe(pos, torch.zeros(1, 1))
    q.sum().backward()
    # Rounding is non-differentiable; the estimator passes gradient through.
    assert pos.grad is not None and float(pos.grad) == 1.0


def test_disabled_grid_is_identity():
    s = SensorModel(0.0)
    s.reset(torch.zeros(1, 1))
    pos = torch.tensor([[0.13, 0.9]])
    vel = torch.tensor([[1.5, -2.0]])
    q, v = s.observe(pos, vel)
    assert torch.equal(q, pos) and torch.equal(v, vel)


def test_rollout_without_sensor_is_unchanged():
    # sensor=None must leave the closed-loop rollout byte-identical.
    plant = _dac_plant(gain=1.0)
    warmup = torch.zeros(3, 3, 6)
    ref = constant_reference(torch.tensor([1.0]), horizon=12, batch=3)
    pos_a, dac_a = plant.closed_loop_rollout(warmup, ref, PDPolicy(kp=0.3), 12)
    pos_b, dac_b = plant.closed_loop_rollout(
        warmup, ref, PDPolicy(kp=0.3), 12, sensor=None
    )
    assert torch.equal(pos_a, pos_b) and torch.equal(dac_a, dac_b)


def test_actuator_lags_a_step():
    # A step command is approached gradually: the first output moves only
    # a fraction alpha of the way, and it converges to the command.
    act = ActuatorModel(cutoff_hz=25.0, servo_rate_hz=SERVO_HZ)
    act.reset(torch.zeros(1, 1))
    cmd = torch.ones(1, 1)
    first = act.apply(cmd)
    assert 0.0 < float(first) < 1.0
    out = first

    for _ in range(2000):
        out = act.apply(cmd)
    assert torch.allclose(out, cmd, atol=1e-3)


def test_actuator_attenuates_high_frequency():
    # An every-step sign flip is the Nyquist rate; the low-pass shrinks
    # its amplitude far below the command's.
    act = ActuatorModel(cutoff_hz=25.0, servo_rate_hz=SERVO_HZ)
    act.reset(torch.zeros(1, 1))
    outs = []
    for k in range(200):
        outs.append(float(act.apply(torch.tensor([[1.0 if k % 2 else -1.0]]))))
    amp = max(abs(v) for v in outs[100:])  # after the transient settles
    assert amp < 0.2


def test_actuator_disabled_is_identity():
    act = ActuatorModel(cutoff_hz=0.0, servo_rate_hz=SERVO_HZ)
    act.reset(torch.zeros(1, 1))
    cmd = torch.tensor([[3.0, -5.0]])
    assert torch.equal(act.apply(cmd), cmd)


def test_rollout_with_actuator_rolls_off_command():
    # The applied command changes more slowly than an unfiltered one: its
    # per-step differences are strictly smaller than the raw policy's.
    plant = _dac_plant(gain=1.0)
    warmup = torch.zeros(2, 3, 6)
    ref = constant_reference(torch.tensor([1.0]), horizon=20, batch=2)
    _, dac_raw = plant.closed_loop_rollout(warmup, ref, PDPolicy(kp=0.5), 20)
    _, dac_lp = plant.closed_loop_rollout(
        warmup,
        ref,
        PDPolicy(kp=0.5),
        20,
        actuator=ActuatorModel(cutoff_hz=25.0, servo_rate_hz=SERVO_HZ),
    )
    assert torch.isfinite(dac_lp).all()
    raw_slew = dac_raw[:, 1:, :].sub(dac_raw[:, :-1, :]).abs().max()
    lp_slew = dac_lp[:, 1:, :].sub(dac_lp[:, :-1, :]).abs().max()
    assert float(lp_slew) < float(raw_slew)


def test_rollout_with_sensor_runs_and_tracks():
    plant = _dac_plant(gain=1.0)
    warmup = torch.zeros(2, 3, 6)
    ref = constant_reference(torch.tensor([1.0]), horizon=20, batch=2)
    pos, dac = plant.closed_loop_rollout(
        warmup, ref, PDPolicy(kp=0.3), 20, sensor=SensorModel(GRID)
    )
    assert torch.isfinite(pos).all() and torch.isfinite(dac).all()
    # A coarse sensor still lets a proportional policy close most of the step.
    assert (pos[:, -1, 0] - ref[:, -1, 0]).abs().max() < GRID
