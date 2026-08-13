"""
Chunked dense forward for the temporal-conv stack (cached-conv
training).

Runs the causal TCN once over a contiguous chunk and emits a prediction
at every position that has a full pooling window, instead of
re-convolving each length-W window independently. The shared
convolution is thus amortised across overlapping windows: one pass over
L positions replaces L - W + 1 length-W ones.

This is not numerically identical to windowed training. The windowed
model pools conv features that were zero-padded at each window's start.
The chunked model gives those positions their real history — the
streaming-consistent computation the FPGA actually runs at deploy time.
It is therefore an opt-in modelling variant, gated on accuracy, not a
drop-in speedup.
"""

from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812  (conventional alias)

from nn_motion_control.models.channels_last import _block, _pool_index
from nn_motion_control.models.layers.tcn import TemporalBlock


def chunked_dense_forward(
    network: nn.Sequential, chunk: torch.Tensor, window: int
) -> torch.Tensor:
    """
    Run network over chunk [B, C, L] and predict at every full-window
    position.

    Returns [B, L - window + 1, out]: the prediction whose length-window
    pooling window ends at each position p in [window - 1, L - 1]. The
    leading positions carry a startup transient (their pooling window
    still sees zero-padded conv features) and should be discarded as
    warmup by the caller.
    """

    pi = _pool_index(network)
    x = chunk.unsqueeze(2).to(memory_format=torch.channels_last)  # [B, C, 1, L]

    for i in range(pi):
        x = _block(cast(TemporalBlock, network[i]), x)
    feats = x.squeeze(2)  # [B, C', L]
    # Sliding mean of width window reproduces AdaptiveAvgPool1d(1) per
    # end-position.
    pooled = F.avg_pool1d(feats, kernel_size=window, stride=1)  # [B, C', P]
    h = pooled.transpose(1, 2)  # [B, P, C']
    b, p, c = h.shape
    h = h.reshape(b * p, c)

    for layer in list(network)[pi + 2 :]:  # skip AdaptiveAvgPool1d + Flatten
        h = layer(h)

    return h.reshape(b, p, -1)
