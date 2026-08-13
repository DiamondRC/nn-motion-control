"""Channels-last TCN forward.

Capability detection and equivalence to the conv1d path.
"""

import torch
import torch.nn as nn

from nn_motion_control.core.config import RunConfiguration
from nn_motion_control.models.builder import JsonModel
from nn_motion_control.models.channels_last import (
    channels_last_tcn_forward,
    is_channels_last_able,
)

TCN_CFG = "examples/deltabot/configs/plant_tcn_rollout.json"


def _tcn():
    return JsonModel(config=RunConfiguration(TCN_CFG))


def test_tcn_stack_is_channels_last_able():
    assert is_channels_last_able(_tcn().network)


def test_non_tcn_stack_is_not_able():
    # No AdaptiveAvgPool1d -> not handled; caller must fall back.
    assert not is_channels_last_able(
        nn.Sequential(nn.Flatten(), nn.Linear(10, 3))
    )


def test_channels_last_matches_conv1d_in_eval():
    m = (
        _tcn().eval()
    )  # eval -> dropout off, so both paths match to float rounding
    torch.manual_seed(0)
    window = torch.randn(4, 9, 64)
    with torch.no_grad():
        ref = m.network(window)
        got = channels_last_tcn_forward(m.network, window)
    assert torch.allclose(ref, got, atol=1e-4, rtol=1e-3)


def test_enable_channels_last_switches_forward_equivalently():
    m = _tcn().eval()
    torch.manual_seed(1)
    window = torch.randn(4, 9, 64)
    with torch.no_grad():
        ref = m(window)  # conv1d path
        m.enable_channels_last()
        got = m(window)  # channels-last path
    assert m.use_channels_last
    assert torch.allclose(ref, got, atol=1e-4, rtol=1e-3)
