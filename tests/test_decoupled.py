"""Decoupled plant: input routing, and streamed == windowed equivalence."""

import pytest
import torch

from nn_motion_control.core.config import RunConfiguration
from nn_motion_control.models.builder import build_model
from nn_motion_control.models.decoupled import DecoupledSSMPlant
from nn_motion_control.plant.rollout import DecoupledStepper

WINDOW = 8
DECOUPLED_ARCH = {
    "window_size": WINDOW,
    "decoupled": {
        "dynamic": [{"DiagSSM": [6, 8, 4]}, {"DiagSSM": [8, 8, 4]}],
        "static": [{"Linear": [3, 4]}, "ReLU"],
        "head": [{"Linear": [12, 8]}, "ReLU", {"Linear": [8, 3]}],
    },
}


@pytest.fixture
def decoupled_cfg(config_factory):
    return config_factory(
        architecture=DECOUPLED_ARCH,
        inputs=["position", "velocity", "dac"],
        targets={"position": {"form": "delta", "weight": 1}},
    )


def test_builds_and_routes_position_out_of_the_dynamic_path(decoupled_cfg):
    cfg = RunConfiguration(decoupled_cfg)
    assert cfg.is_decoupled and cfg.target_form == "delta"
    model = build_model(cfg)
    assert isinstance(model, DecoupledSSMPlant)
    # Position (cols 0,3,6) feeds the static path; velocity+command the SSM.
    assert model.static_cols == [0, 3, 6]
    assert model.dyn_cols == [1, 2, 4, 5, 7, 8]
    out = model.eval()(torch.randn(4, 9, WINDOW))
    assert out.shape == (4, 3)


def test_stepper_reset_matches_windowed_forward(decoupled_cfg):
    model = build_model(RunConfiguration(decoupled_cfg)).eval()
    window = torch.randn(4, 9, WINDOW)
    stepper = DecoupledStepper(model)
    with torch.no_grad():
        ref = model(window)
        got = stepper.reset(window)
    assert torch.allclose(ref, got, atol=1e-4, rtol=1e-3)


def test_streaming_equals_growing_sequence(decoupled_cfg):
    # Carrying the SSM state frame by frame must equal re-running the whole
    # growing sequence: the dynamic path keeps all history, the static path
    # reads the current position each step.
    model = build_model(RunConfiguration(decoupled_cfg)).eval()
    window = torch.randn(2, 9, WINDOW)
    frames = [torch.randn(2, 9) for _ in range(6)]
    stepper = DecoupledStepper(model)
    with torch.no_grad():
        got = [stepper.reset(window)]
        seq = window
        want = [model(seq)]

        for fr in frames:
            got.append(stepper.step(fr))
            seq = torch.cat([seq, fr.unsqueeze(-1)], dim=2)
            want.append(model(seq))

    for g, w in zip(got, want, strict=True):
        assert torch.allclose(g, w, atol=1e-4, rtol=1e-3)
