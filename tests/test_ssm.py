"""Diagonal SSM.

Dual-form equivalence (parallel scan == recurrent step), streaming.
"""

import pytest
import torch

from nn_motion_control.core.config import RunConfiguration
from nn_motion_control.data.normalize import NormStats
from nn_motion_control.models.builder import JsonModel
from nn_motion_control.models.ssm import DiagSSM
from nn_motion_control.plant.plant import Plant, RolloutLayout
from nn_motion_control.plant.rollout import RecurrentStepper, RolloutStepper

# The retired plant_ssm.json architecture, built hermetically so the
# SSM code stays tested without depending on an example config (9
# inputs = 3 axes x [pos, vel, dac]).
SSM_LAYERS = [
    {"DiagSSM": [9, 64, 32]},
    {"DiagSSM": [64, 64, 32]},
    {"DiagSSM": [64, 64, 32]},
    "LastFrame",
    {"Flatten": [1]},
    {"Linear": [64, 128]},
    "ReLU",
    {"Linear": [128, 3]},
]


@pytest.fixture
def ssm_cfg(config_factory):
    """Path to a hermetic SSM plant config matching the retired
    plant_ssm.json.
    """

    return config_factory(
        window_size=64,
        hidden_layers=SSM_LAYERS,
        inputs=["position", "velocity", "dac"],
        targets={"position": {"form": "delta", "weight": 1}},
    )


def test_diagssm_scan_equals_sequential_steps():
    torch.manual_seed(1)
    layer = DiagSSM(d_in=4, d_out=6, d_state=8).eval()
    u = torch.randn(3, 20, 4)
    y_par, h_last = layer.scan(u)
    # Recurrent form must reproduce the parallel scan output frame for frame.
    h = layer.init_state(3, u.device)
    ys = []

    for t in range(u.shape[1]):
        y_t, h = layer.step(u[:, t], h)
        ys.append(y_t)
    y_seq = torch.stack(ys, dim=1)
    assert torch.allclose(y_par, y_seq, atol=1e-5)
    assert torch.allclose(h_last, h, atol=1e-5)


def test_ssm_model_builds_and_forwards(ssm_cfg):
    m = JsonModel(config=RunConfiguration(ssm_cfg))
    assert m.is_ssm
    ssm_layers, head = m.ssm_section()
    assert len(ssm_layers) == 3 and all(
        isinstance(x, DiagSSM) for x in ssm_layers
    )
    assert len(head) == 5  # LastFrame, Flatten, Linear, ReLU, Linear
    out = m.eval()(torch.randn(4, 9, 64))
    assert out.shape == (4, 3)


def test_recurrent_stepper_satisfies_protocol(ssm_cfg):
    m = JsonModel(config=RunConfiguration(ssm_cfg))
    assert isinstance(RecurrentStepper(m), RolloutStepper)


def test_recurrent_reset_matches_windowed_forward(ssm_cfg):
    m = JsonModel(config=RunConfiguration(ssm_cfg)).eval()
    torch.manual_seed(2)
    window = torch.randn(4, 9, 64)
    stepper = RecurrentStepper(m)
    with torch.no_grad():
        ref = m(window)
        got = stepper.reset(window)
    assert torch.allclose(ref, got, atol=1e-4, rtol=1e-3)


def test_recurrent_streaming_equals_full_sequence_scan(ssm_cfg):
    # Streaming the SSM frame by frame must equal running the full
    # growing sequence at once: the SSM carries all history, so no
    # window-truncation (unlike a TCN).
    m = JsonModel(config=RunConfiguration(ssm_cfg)).eval()
    torch.manual_seed(3)
    window = torch.randn(2, 9, 64)
    frames = [torch.randn(2, 9) for _ in range(8)]
    stepper = RecurrentStepper(m)
    with torch.no_grad():
        got = [stepper.reset(window)]
        seq = window
        want = [m(seq)]

        for fr in frames:
            got.append(stepper.step(fr))
            seq = torch.cat(
                [seq, fr.unsqueeze(-1)], dim=2
            )  # grow; keep all old frames
            want.append(m(seq))

    for g, w_ in zip(got, want, strict=True):
        assert torch.allclose(g, w_, atol=1e-4, rtol=1e-3)


def _identity_ssm_plant(model, cfg_path):
    layout = RolloutLayout.from_config(RunConfiguration(cfg_path))
    ones, zeros = torch.ones(9), torch.zeros(9)
    ins = NormStats(mean=zeros, std=ones, normalizable=ones.bool())
    tgt = NormStats(
        mean=torch.zeros(3),
        std=torch.ones(3),
        normalizable=torch.ones(3).bool(),
    )
    return Plant(
        model, ins, tgt, layout, device="cpu", rollout_kind="recurrent"
    )


def test_ssm_rollout_runs_and_is_differentiable(ssm_cfg):
    m = JsonModel(config=RunConfiguration(ssm_cfg)).train()
    torch.manual_seed(4)
    warmup = torch.randn(2, 9, 64)
    dac = torch.zeros(2, 6, 3)
    plant = _identity_ssm_plant(m, ssm_cfg)
    preds = plant.roll_forward(warmup, dac, horizon=6)
    assert preds.shape == (2, 6, 3)
    preds.pow(2).mean().backward()
    grads = [p.grad for p in m.parameters() if p.grad is not None]
    assert grads and any(g.abs().sum() > 0 for g in grads)
