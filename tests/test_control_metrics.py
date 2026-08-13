"""
control.closed_loop: settling / overshoot / disturbance metrics.
"""

import torch

from nn_motion_control.control.closed_loop import (
    disturbance_response,
    overshoot,
    settling_time,
    tracking_percentiles,
)


def test_settling_time_finds_first_sustained_step():
    ref = torch.zeros(2, 6, 1)
    pos = torch.zeros(2, 6, 1)
    pos[:, :3, 0] = 1.0  # outside the tol band for the first three steps
    settle = settling_time(pos, ref, tol=0.1, dt=1.0)
    assert settle.shape == (1,)
    assert settle.item() == 3.0


def test_settling_time_infinite_when_never_settled():
    ref = torch.zeros(1, 5, 1)
    pos = torch.ones(1, 5, 1)  # always outside the band
    assert torch.isinf(settling_time(pos, ref, tol=0.1)).all()


def test_overshoot_fraction_of_step():
    origin = torch.zeros(1, 1)
    ref = torch.ones(1, 4, 1)  # target 1.0, so the step is 1.0
    pos = torch.tensor([[[0.5], [1.2], [1.0], [1.0]]])  # peaks at 1.2
    assert torch.allclose(
        overshoot(pos, ref, origin), torch.tensor([0.2]), atol=1e-6
    )


def test_overshoot_zero_without_a_step():
    origin = torch.ones(1, 1)
    ref = torch.ones(1, 4, 1)  # target equals origin: no commanded step
    pos = torch.full((1, 4, 1), 5.0)
    assert torch.allclose(overshoot(pos, ref, origin), torch.zeros(1))


def test_tracking_percentiles_split_typical_from_tail():
    ref = torch.zeros(4, 3, 1)
    pos = torch.zeros(4, 3, 1)

    for i in range(4):
        pos[i] = float(
            i + 1
        )  # per-rollout RMS = 1, 2, 3, 4 (constant over horizon)
    out = tracking_percentiles(pos, ref, pcts=(50.0, 100.0))
    assert torch.allclose(out[100.0], torch.tensor([4.0]))  # worst rollout
    assert out[50.0].item() < out[100.0].item()  # P50 typical below the tail


def test_disturbance_peak_and_recovery():
    ref = torch.zeros(1, 8, 1)
    pos = torch.zeros(1, 8, 1)
    pos[0, 3, 0] = 2.0  # impulse at the disturbance step
    pos[0, 4, 0] = 0.5  # decaying back
    out = disturbance_response(pos, ref, disturb_step=3, tol=0.1)
    assert torch.allclose(out["peak"], torch.tensor([2.0]))
    # Post-disturbance the band holds from relative step 2 (absolute
    # step 5) onward.
    assert out["recovery"].item() == 2.0
