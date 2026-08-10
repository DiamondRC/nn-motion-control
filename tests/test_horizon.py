"""
Error-vs-horizon eval: error collection, curve reduction, resolution lookup.
"""

import numpy as np
import torch

from nn_motion_control.core.config import RunConfiguration
from nn_motion_control.data.normalize import NormStats
from nn_motion_control.eval.horizon import (
    collect_horizon_errors,
    collect_horizon_trajectories,
    horizon_channel_table,
    horizon_curves,
    position_resolution,
)
from nn_motion_control.plant.plant import Plant, RolloutLayout


class _ConstDelta(torch.nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.value = value

    def forward(self, window):
        return torch.full((window.shape[0], 1), self.value)


def _identity_plant(const=0.5):
    layout = RolloutLayout(pos_cols=[0], vel_cols=[1], dac_cols=[2], n_features=3)
    ones = torch.ones(3)
    in_stats = NormStats(mean=torch.zeros(3), std=ones, normalizable=ones.bool())
    t_stats = NormStats(
        mean=torch.zeros(1), std=torch.ones(1), normalizable=torch.ones(1).bool()
    )
    return Plant(_ConstDelta(const), in_stats, t_stats, layout, device="cpu")


def test_collect_errors_grow_with_horizon():
    plant = _identity_plant(0.5)
    b, h = 4, 6
    warmup = torch.zeros(b, 3, 4)  # seed position 0
    dac = torch.zeros(b, h, 1)
    gt = torch.zeros(b, h, 1)  # truth stays at 0, model marches away by 0.5/step
    errs = collect_horizon_errors(plant, [(warmup, dac, gt)], h)
    assert errs.shape == (b, h, 1)
    # constant delta 0.5 vs zero truth -> error at step k is (k+1)*0.5 (identity norm).
    expected = torch.tensor([(k + 1) * 0.5 for k in range(h)])
    assert torch.allclose(errs[0, :, 0], expected, atol=1e-6)
    assert errs[:, -1, :].mean() > errs[:, 0, :].mean()  # drift grows


def test_max_batches_caps_sampling():
    plant = _identity_plant()
    b, h = 2, 3
    batch = (torch.zeros(b, 3, 4), torch.zeros(b, h, 1), torch.zeros(b, h, 1))
    errs = collect_horizon_errors(plant, [batch, batch, batch], h, max_batches=2)
    assert errs.shape[0] == 2 * b  # only the first two batches consumed


def test_horizon_curves_reduce_over_samples():
    n, h, a = 10, 5, 2
    e = torch.zeros(n, h, a)
    for k in range(h):
        e[:, k, :] = k  # every sample has error k at step k
    c = horizon_curves(e)
    assert c["mean"].shape == (h, a)
    assert np.allclose(c["mean"][:, 0], np.arange(h))
    assert np.allclose(c["p99"][:, 0], np.arange(h))  # identical samples


def test_trajectories_and_channel_table():
    # Constant-delta plant marching away from a zero truth: at step k the predicted
    # position is (k+1)*0.5 while truth stays 0, so derived velocity is a steady 0.5.
    plant = _identity_plant(0.5)
    b, h = 5, 6
    warmup = torch.zeros(b, 3, 4)  # seed position 0 (identity norm)
    dac = torch.zeros(b, h, 1)
    gt = torch.zeros(b, h, 1)
    preds, truth, anchor = collect_horizon_trajectories(plant, [(warmup, dac, gt)], h)
    assert preds.shape == (b, h, 1)
    assert truth.shape == (b, h, 1)
    assert anchor.shape == (b, 1)
    assert torch.allclose(
        preds[0, :, 0], torch.tensor([(k + 1) * 0.5 for k in range(h)])
    )

    to_nm = np.array([1.0])  # identity units so nm == raw counts
    table = horizon_channel_table(preds, truth, anchor, ["x"], [1, h], to_nm)
    # Two rows per axis: position then velocity.
    assert [m.name for m in table[1]] == ["x pos", "x vel"]
    # Step 1: position and velocity coincide (free-run has not diverged) at 0.5.
    assert table[1][0].p99_abs == table[1][1].p99_abs
    assert np.isclose(table[1][1].p99_abs, 0.5)
    # Later step: position drift grows with the horizon; velocity error stays ~0.5.
    assert table[h][0].p99_abs > table[1][0].p99_abs
    assert np.isclose(table[h][1].p99_abs, 0.5)


def test_position_resolution_from_config():
    rc = RunConfiguration("examples/deltabot/configs/plant_tcn.json")
    res = position_resolution(rc)
    assert res.shape == (3,)
    assert np.allclose(res, 0.256e-9)  # interferometer resolution, m per count
