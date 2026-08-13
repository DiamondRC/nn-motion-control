"""
Channels-last forward for the temporal-conv stack.

On tensor-core GPUs cuDNN runs conv kernels in NHWC (channels-last), so
a plain [B, C, L] conv1d spends a large share of its time transposing
tensors NCHW to NHWC around every convolution. Running the identical
convolution as a channels-last conv2d (a singleton height dim) lets
cuDNN use its native NHWC kernels with no transpose — same maths,
markedly less memory traffic. This module reproduces JsonModel's
temporal forward that way. It is numerically equivalent (to float
rounding) to running the stack as conv1d.
"""

from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812  (conventional alias)

from nn_motion_control.models.layers.tcn import TemporalBlock


def _pool_index(network: nn.Sequential) -> int:
    for i, layer in enumerate(network):
        if isinstance(layer, nn.AdaptiveAvgPool1d):
            return i

    return -1


def is_channels_last_able(network: nn.Sequential) -> bool:
    """
    Whether network is a TemporalConv stack this module can run
    channels-last.

    Requires: one or more TemporalBlock layers, then
    AdaptiveAvgPool1d(1), then Flatten, then a non-temporal head
    (Linear/ReLU/Dropout/LayerNorm). Anything else (LSTM/GRU,
    non-pooled head, ...) is not handled and the caller should fall
    back.
    """

    layers = list(network)
    pi = _pool_index(network)
    if pi <= 0:
        return False
    if not all(isinstance(layer, TemporalBlock) for layer in layers[:pi]):
        return False
    if layers[pi].output_size != 1:
        return False
    if pi + 1 >= len(layers) or not isinstance(layers[pi + 1], nn.Flatten):
        return False
    head_types = (nn.Linear, nn.ReLU, nn.Dropout, nn.LayerNorm)

    return all(isinstance(layer, head_types) for layer in layers[pi + 2 :])


def _causal_conv2d(conv: nn.Conv1d, x: torch.Tensor) -> torch.Tensor:
    # Run a causal Conv1d as a conv2d over [B, Cin, 1, L], reproducing
    # its left-pad + Chomp1d causal trim. conv.weight materialises the
    # weight-norm parametrisation.
    p = cast(tuple, conv.padding)[0]
    d = cast(tuple, conv.dilation)[0]
    w = conv.weight.unsqueeze(2)
    out = F.conv2d(x, w, conv.bias, padding=(0, p), dilation=(1, d))

    return out[:, :, :, : out.shape[-1] - p] if p > 0 else out


def _block(blk: TemporalBlock, x: torch.Tensor) -> torch.Tensor:
    # Mirror TemporalBlock.forward: net1 (conv/chomp/relu/drop), net2,
    # residual, relu.
    conv1, drop1 = cast(nn.Conv1d, blk.net1[0]), cast(nn.Dropout, blk.net1[-1])
    conv2, drop2 = cast(nn.Conv1d, blk.net2[0]), cast(nn.Dropout, blk.net2[-1])
    out = F.dropout(F.relu(_causal_conv2d(conv1, x)), drop1.p, drop1.training)
    out = F.dropout(F.relu(_causal_conv2d(conv2, out)), drop2.p, drop2.training)
    down = cast("nn.Conv1d | None", blk.downsample)
    res = (
        x if down is None else F.conv2d(x, down.weight.unsqueeze(2), down.bias)
    )

    return F.relu(out + res)


def channels_last_tcn_forward(
    network: nn.Sequential, window: torch.Tensor
) -> torch.Tensor:
    """
    Run a channels-last-able TemporalConv network: window [B, C, L] to
    [B, out].
    """

    pi = _pool_index(network)
    x = window.unsqueeze(2).to(
        memory_format=torch.channels_last
    )  # [B, C, 1, L]

    for i in range(pi):
        x = _block(cast(TemporalBlock, network[i]), x)
    x = x.mean(dim=(2, 3))  # AdaptiveAvgPool1d(1) + Flatten

    for layer in list(network)[pi + 2 :]:
        x = layer(x)

    return x
