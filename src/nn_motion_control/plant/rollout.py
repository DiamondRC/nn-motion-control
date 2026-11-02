"""
Rollout steppers: how a model advances one step during a free-run.

A windowed conv/MLP recomputes the whole window,
a streaming conv updates cached activations, a recurrent cell steps
a hidden state.
"""

from __future__ import annotations

from typing import Any, Protocol, cast, runtime_checkable

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812  (conventional alias)

from nn_motion_control.models.layers.heads import AvgPoolLastK, LastFrame
from nn_motion_control.models.layers.tcn import TemporalBlock
from nn_motion_control.models.ssm import DiagSSM

__all__ = [
    "bound_row_gradient",
    "RolloutStepper",
    "WindowedStepper",
    "StreamingConvStepper",
    "RecurrentStepper",
    "DecoupledStepper",
]


# Floor on a row's gradient norm before dividing by it.
_NORM_FLOOR = 1e-30


class _BoundRowGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, limit: float) -> torch.Tensor:
        ctx.limit = limit

        return x.view_as(x)

    @staticmethod
    def backward(ctx, *grads: torch.Tensor) -> tuple[torch.Tensor, None]:
        g = grads[0]
        norm = g.flatten(1).norm(dim=1).clamp_min(_NORM_FLOOR)
        scale = (ctx.limit / norm).clamp(max=1.0)
        shape = (-1,) + (1,) * (g.dim() - 1)

        return g * scale.view(shape), None


def bound_row_gradient(x: torch.Tensor, limit: float | None) -> torch.Tensor:
    """
    Identity on the forward pass, and on the backward pass each batch row's
    gradient is scaled down to norm 'limit' if it exceeds it.

    Applied to a rollout's state every step, a row whose closed loop is
    unstable, and whose gradient therefore grows exponentially with the
    horizon, keeps its direction but cannot outweigh the rest of the batch.
    None passes 'x' through untouched.
    """

    if limit is None or not x.requires_grad:
        return x

    return cast(torch.Tensor, _BoundRowGradient.apply(x, limit))


@runtime_checkable
class RolloutStepper(Protocol):
    """
    Advances a model one rollout step, returning the raw (normalised) delta
    [B, A].
    """

    def reset(self, warmup: torch.Tensor) -> torch.Tensor:
        """
        Prime state from the seed window [B, F, W]; return the step-0
        delta.
        """
        ...

    def step(self, frame: torch.Tensor) -> torch.Tensor:
        """Advance by one input frame [B, F]; return the next delta [B, A]."""
        ...


class WindowedStepper:
    """
    Baseline stepper for windowed models (TCN/MLP): recompute the window
    each step.

    'reset' stores the seed window and runs the model on it; 'step' slides
    the window (drop the oldest frame, append the new one) and re-runs the
    full model. This is the behaviour the plant had before the stepper
    split: every step is an independent forward over the current window.
    """

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self._window: torch.Tensor | None = None

    def reset(self, warmup: torch.Tensor) -> torch.Tensor:
        self._window = warmup
        return self.model(warmup)

    def step(self, frame: torch.Tensor) -> torch.Tensor:
        assert self._window is not None, "Step() called before reset()"
        self._window = torch.cat(
            [self._window[:, :, 1:], frame.unsqueeze(-1)], dim=2
        )
        return self.model(self._window)


def _conv_of(block: TemporalBlock, which: int) -> nn.Conv1d:
    net = block.net1 if which == 1 else block.net2
    return cast(nn.Conv1d, net[0])


def _dropout_of(block: TemporalBlock, which: int) -> nn.Dropout:
    net = block.net1 if which == 1 else block.net2
    return cast(nn.Dropout, net[-1])


def _causal_conv_seq(conv: nn.Conv1d, x: torch.Tensor) -> torch.Tensor:
    """
    Causal (left-pad + chomp) dilated conv over a sequence; [B,C,L] in/out.
    """
    p = cast("tuple[int, ...]", conv.padding)[0]
    d = cast("tuple[int, ...]", conv.dilation)[0]
    out = F.conv1d(x, conv.weight, conv.bias, padding=p, dilation=d)
    return out[:, :, : out.shape[-1] - p] if p > 0 else out


def _streaming_conv_frame(conv: nn.Conv1d, buf: torch.Tensor) -> torch.Tensor:
    """
    One causal output frame from a span buffer [B, C, (k-1)*d+1] to
    [B, C'].
    """
    d = cast("tuple[int, ...]", conv.dilation)[0]
    out = F.conv1d(
        buf, conv.weight, conv.bias, dilation=d
    )  # length collapses to 1
    return out[:, :, 0]


def _span(conv: nn.Conv1d) -> int:
    """Receptive span of one dilated conv frame: (kernel - 1) * dilation + 1."""
    k = cast("tuple[int, ...]", conv.kernel_size)[0]
    d = cast("tuple[int, ...]", conv.dilation)[0]
    return (k - 1) * d + 1


def _readout_pool(head_layers: list[nn.Module]) -> int:
    """
    Number of final-block frames the readout pools over: 1 for 'LastFrame',
    k for 'AvgPoolLastK(k)'. Both take the last 'pool' frames, so the
    streaming ring buffer holds exactly that many.
    """

    for layer in head_layers:
        if isinstance(layer, AvgPoolLastK):
            return cast(int, layer.k)
        if isinstance(layer, LastFrame):
            return 1

    raise ValueError("Streaming head needs a LastFrame or AvgPoolLastK readout")


class StreamingConvStepper:
    """
    O(1)/step stepper for a streaming-TCN (receptive field <= window,
    causal readout).

    Each 'TemporalBlock' keeps two causal ring buffers (one per internal
    conv) holding the last (k-1)*d + 1 input frames, and a final ring
    buffer holds the last 'pool' block outputs the readout averages over
    (1 for 'LastFrame', k for 'AvgPoolLastK(k)'). 'reset' runs the seed
    window once to prime every buffer and returns the readout; 'step'
    advances every block by a single new frame, WaveNet-style streaming
    inference, so a rollout costs one full windowed pass plus O(1) work
    per horizon step instead of a full window per step.

    Under receptive-field <= window - pool the pooled frames are all
    in-window, so this reproduces the windowed forward exactly (float
    rounding) in eval; activations stay in-graph so the rollout is
    BPTT-correct in training.
    """

    def __init__(self, model: nn.Module):
        network = cast(nn.Sequential, model.network)
        layers = list(network)
        n_blocks = sum(isinstance(layer, TemporalBlock) for layer in layers)
        if n_blocks == 0 or not all(
            isinstance(layer, TemporalBlock) for layer in layers[:n_blocks]
        ):
            raise ValueError(
                "StreamingConvStepper needs a leading TemporalBlock stack"
            )
        self.blocks = [
            cast(TemporalBlock, layer) for layer in layers[:n_blocks]
        ]
        # Everything after the conv stack (readout pool, Flatten, Linear
        # head). Fed a [B, C, pool] tensor so the pool + Flatten shape
        # handling is reused verbatim.
        self.head = nn.Sequential(*layers[n_blocks:])
        self.pool = _readout_pool(layers[n_blocks:])
        self._buf1: list[torch.Tensor] = []
        self._buf2: list[torch.Tensor] = []
        self._out: torch.Tensor | None = None  # last 'pool' final-block frames

    def _apply_head(self, frames: torch.Tensor) -> torch.Tensor:
        # frames is [B, C, pool]; the readout pool collapses it to [B, C, 1].
        return self.head(frames)

    def _block_frame(self, i: int, x_t: torch.Tensor) -> torch.Tensor:
        block = self.blocks[i]
        conv1, conv2 = _conv_of(block, 1), _conv_of(block, 2)
        drop1, drop2 = _dropout_of(block, 1), _dropout_of(block, 2)
        self._buf1[i] = torch.cat(
            [self._buf1[i][:, :, 1:], x_t.unsqueeze(-1)], dim=2
        )
        c1 = _streaming_conv_frame(conv1, self._buf1[i])
        out1 = F.dropout(F.relu(c1), drop1.p, drop1.training)
        self._buf2[i] = torch.cat(
            [self._buf2[i][:, :, 1:], out1.unsqueeze(-1)], dim=2
        )
        c2 = _streaming_conv_frame(conv2, self._buf2[i])
        out2 = F.dropout(F.relu(c2), drop2.p, drop2.training)
        down = cast("nn.Conv1d | None", block.downsample)
        if down is None:
            res = x_t
        else:
            res = F.conv1d(x_t.unsqueeze(-1), down.weight, down.bias)[:, :, 0]
        return F.relu(out2 + res)

    def reset(self, warmup: torch.Tensor) -> torch.Tensor:
        self._buf1, self._buf2 = [], []
        x_seq = warmup

        for block in self.blocks:
            conv1, conv2 = _conv_of(block, 1), _conv_of(block, 2)
            drop1, drop2 = _dropout_of(block, 1), _dropout_of(block, 2)
            # Windowed causal pass over the whole seed window, matching
            # the model.
            out1_seq = F.dropout(
                F.relu(_causal_conv_seq(conv1, x_seq)), drop1.p, drop1.training
            )
            out2_seq = F.dropout(
                F.relu(_causal_conv_seq(conv2, out1_seq)),
                drop2.p,
                drop2.training,
            )
            down = cast("nn.Conv1d | None", block.downsample)
            res_seq = (
                x_seq
                if down is None
                else F.conv1d(x_seq, down.weight, down.bias)
            )
            block_out = F.relu(out2_seq + res_seq)
            # Keep only the receptive span of each internal conv as the
            # streaming state.
            self._buf1.append(x_seq[:, :, -_span(conv1) :])
            self._buf2.append(out1_seq[:, :, -_span(conv2) :])
            x_seq = block_out
        # Prime the readout ring buffer with the last 'pool' final-block
        # frames.
        self._out = x_seq[:, :, -self.pool :]

        return self._apply_head(self._out)

    def step(self, frame: torch.Tensor) -> torch.Tensor:
        assert self._out is not None, "Step() called before reset()"
        x_t = frame

        for i in range(len(self.blocks)):
            x_t = self._block_frame(i, x_t)
        self._out = torch.cat([self._out[:, :, 1:], x_t.unsqueeze(-1)], dim=2)

        return self._apply_head(self._out)


class DecoupledStepper:
    """
    O(1)/step stepper: stream the dynamic SSM, add the static position
    term each step.
    """

    def __init__(self, model: nn.Module):
        m = cast(Any, model)
        self.dyn = [cast(DiagSSM, layer) for layer in m.dynamic]
        self.static = m.static
        self.head = m.head
        self.dyn_cols = m.dyn_cols
        self.static_cols = m.static_cols
        self._states: list[torch.Tensor] = []

    def _readout(
        self, dyn_feat: torch.Tensor, pos: torch.Tensor
    ) -> torch.Tensor:
        if self.static is not None:
            return self.head(torch.cat([dyn_feat, self.static(pos)], dim=-1))

        return self.head(dyn_feat)

    def reset(self, warmup: torch.Tensor) -> torch.Tensor:
        u = warmup[:, self.dyn_cols, :].transpose(1, 2)  # [B, W, d_dyn]
        self._states = []

        for layer in self.dyn:
            u, h_last = layer.scan(u)
            self._states.append(h_last)

        return self._readout(u[:, -1, :], warmup[:, self.static_cols, -1])

    def step(self, frame: torch.Tensor) -> torch.Tensor:
        u_t = frame[:, self.dyn_cols]

        for i, layer in enumerate(self.dyn):
            u_t, self._states[i] = layer.step(u_t, self._states[i])

        return self._readout(u_t, frame[:, self.static_cols])


class RecurrentStepper:
    """
    O(1)/step stepper for a recurrent (SSM) model - carries the state, no
    window.

    'reset' runs each 'DiagSSM' layer's parallel scan over the seed window
    to prime its state (and returns the readout); 'step' advances every
    layer by one recurrent update. The state summarises all history, so
    there is no receptive-field ceiling: the streaming rollout and the
    windowed forward compute the same readout (to float rounding), and
    activations stay in-graph so the rollout is BPTT-correct.
    """

    def __init__(self, model: nn.Module):
        ssm_layers, head_layers = cast(
            "tuple[list[nn.Module], list[nn.Module]]",
            cast(Any, model).ssm_section(),
        )
        if not ssm_layers:
            raise ValueError("RecurrentStepper expects a leading DiagSSM stack")

        self.ssm = [cast(DiagSSM, layer) for layer in ssm_layers]
        self.head = nn.Sequential(*head_layers)
        self.pool = _readout_pool(head_layers)
        self._states: list[torch.Tensor] = []
        self._out: torch.Tensor | None = None

    def reset(self, warmup: torch.Tensor) -> torch.Tensor:
        # warmup [B,F,W] to time-major [B,W,F]; scan each layer, keep its
        # final state.
        u = warmup.transpose(1, 2)
        self._states = []

        for layer in self.ssm:
            u, h_last = layer.scan(u)
            self._states.append(h_last)
        final_seq = u.transpose(1, 2)  # [B, d, W]
        self._out = final_seq[:, :, -self.pool :]

        return self.head(self._out)

    def step(self, frame: torch.Tensor) -> torch.Tensor:
        assert self._out is not None, "Step() called before reset()"
        u_t = frame

        for i, layer in enumerate(self.ssm):
            u_t, self._states[i] = layer.step(u_t, self._states[i])
        self._out = torch.cat([self._out[:, :, 1:], u_t.unsqueeze(-1)], dim=2)

        return self.head(self._out)
