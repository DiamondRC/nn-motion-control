"""Closed-loop harness.

A PD baseline tracks a reference, the loop is differentiable.
"""

import torch
import torch.nn as nn

from nn_motion_control.control.closed_loop import (
    PDPolicy,
    constant_reference,
    tracking_metrics,
    zero_policy,
)
from nn_motion_control.data.normalize import NormStats
from nn_motion_control.plant.plant import Plant, RolloutLayout


class _DacDriven(nn.Module):
    """A plant where the command moves the state.

    Delta-P = gain * (last-frame DAC).
    """

    def __init__(self, gain: float):
        super().__init__()
        self.gain = gain

    def forward(self, window):  # window [B, F, W]; DAC is feature column 2
        return self.gain * window[:, 2:3, -1]


def _dac_plant(gain=1.0):
    # 1 axis, F=3 [pos, vel, dac]; identity norm so physical == normalised.
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


def test_constant_reference_shape():
    ref = constant_reference(torch.tensor([1.0, 2.0]), horizon=5, batch=3)
    assert ref.shape == (3, 5, 2)
    assert torch.equal(ref[0, :, 0], torch.full((5,), 1.0))


def test_closed_loop_shapes():
    plant = _dac_plant()
    warmup = torch.zeros(4, 3, 6)
    ref = constant_reference(torch.tensor([1.0]), horizon=8, batch=4)
    pos, dac = plant.closed_loop_rollout(warmup, ref, zero_policy, horizon=8)
    assert pos.shape == (4, 8, 1)
    assert dac.shape == (4, 8, 1)


def test_pd_tracks_better_than_zero():
    # DAC drives motion: a proportional policy should pull position
    # toward the target, so its tracking error is far below the
    # do-nothing baseline's.
    plant = _dac_plant(gain=1.0)
    warmup = torch.zeros(2, 3, 6)  # start at position 0
    ref = constant_reference(torch.tensor([1.0]), horizon=20, batch=2)

    pos_zero, _ = plant.closed_loop_rollout(warmup, ref, zero_policy, 20)
    pos_pd, dac_pd = plant.closed_loop_rollout(
        warmup, ref, PDPolicy(kp=0.3), 20
    )

    m_zero = tracking_metrics(pos_zero, ref)
    m_pd = tracking_metrics(pos_pd, ref)

    # Zero policy never moves, error stays ~1 (the full step), PD
    # converges toward it.
    assert m_zero["final_abs"].item() > 0.9
    assert m_pd["final_abs"].item() < 0.05
    assert m_pd["rms"].item() < m_zero["rms"].item()
    # The command shrinks as the error shrinks (proportional).
    assert dac_pd[0, -1, 0].abs() < dac_pd[0, 0, 0].abs()


def test_closed_loop_is_differentiable_through_policy():
    # A learnable policy: dac = gain * error. Rolling the plant and
    # backpropagating a tracking loss must produce a gradient on the
    # policy parameter (policy gradient).
    plant = _dac_plant(gain=1.0)
    gain = torch.nn.Parameter(torch.tensor(0.2))

    def policy(position, velocity, reference, reference_velocity):
        del velocity, reference_velocity
        return gain * (reference - position)

    warmup = torch.zeros(3, 3, 6)
    ref = constant_reference(torch.tensor([1.0]), horizon=10, batch=3)
    pos, _ = plant.closed_loop_rollout(warmup, ref, policy, 10)
    loss = tracking_metrics(pos, ref)["rms"].sum()
    loss.backward()
    assert gain.grad is not None and gain.grad.abs() > 0


def test_reanchor_bounds_long_horizon_drift():
    # A zero policy never moves, so on a ramping reference the
    # un-anchored error grows to the full ramp. Re-anchoring position
    # onto the reference every K steps (what feedback does) caps it
    # at the per-window drift and resets to ~0 at each anchor, so it
    # stays bounded over an arbitrarily long run.
    plant = _dac_plant(gain=1.0)
    horizon, k = 40, 8
    warmup = torch.zeros(2, 3, 6)
    ramp = 0.1 * torch.arange(horizon, dtype=torch.float32)
    ref = ramp.view(1, horizon, 1).expand(2, horizon, 1).contiguous()

    pos_drift, _ = plant.closed_loop_rollout(warmup, ref, zero_policy, horizon)
    pos_anch, _ = plant.closed_loop_rollout(
        warmup, ref, zero_policy, horizon, reanchor_every=k
    )
    err_drift = (pos_drift - ref).abs()
    err_anch = (pos_anch - ref).abs()

    assert err_drift[:, -1].max() > 3.0  # drifts to ~the full ramp (0.1 * 39)
    assert (
        err_anch[:, -1].max() < 1.0
    )  # bounded by the per-window drift (~0.1 * k)
    # Position matches the reference exactly right after an anchor correction.
    assert torch.allclose(pos_anch[:, k, 0], ref[:, k, 0], atol=1e-5)
