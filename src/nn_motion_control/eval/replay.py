"""
Replay a recorded command stream through a frozen plant.

Drives the plant open-loop with a real command trajectory from an
external capture (not the training set) and measures how its predicted
position tracks the recorded position: the one-step error is the plant's
local fidelity out of distribution, the K-step free-run error is how fast
it diverges. The capture is read in the units its file records and placed
in the plant dataset's frame (see 'panda.DatasetFrame'): resampled and
referenced as the training rows were, every measured axis carrying its
recorded position.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from nn_motion_control.data.panda import (
    DatasetFrame,
    feedback_output,
    read_closed_loop,
)
from nn_motion_control.plant.plant import Plant


def load_capture(
    path: str,
    frame: DatasetFrame,
    axes: Sequence[str],
    resolution_nm: float,
    axis: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load (position, command), each [A, n] float64, from a feedback capture
    whose controller drove 'axis', in the plant dataset's frame and paired
    as its rows are (each position beside the command a step earlier).
    """

    outputs = {axes[axis]: feedback_output(path)}
    capture = read_closed_loop(path, outputs, axes, resolution_nm)

    return frame.rows(*frame.place(capture))


def build_replay_batch(
    plant: Plant,
    pos: np.ndarray,
    dac: np.ndarray,
    axis: int,
    window: int,
    horizon: int,
    stride: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build batched rollout seeds from a capture ('pos', 'dac' [A, n]).

    Slides a start every 'stride' samples, each seed carrying every axis's
    recorded position, velocity and command. Velocity is the per-step
    position delta (the plant's own rollout convention). Returns (warmup
    [B, F, W], dac_future [B, H, A], truth [B, H]) with 'axis''s true
    future position, on the plant's device.
    """

    dev = plant.device
    vel = np.diff(pos, axis=1, prepend=pos[:, :1])
    starts = np.arange(0, pos.shape[1] - window - horizon + 1, stride)
    if starts.size == 0:
        raise ValueError("Capture too short for the requested window/horizon")

    def gather(arr: np.ndarray, offset: int, length: int) -> np.ndarray:
        idx = starts[:, None] + offset + np.arange(length)[None, :]

        return arr[:, idx].transpose(1, 2, 0)  # [B, length, A]

    pos_win = gather(pos, 0, window)
    vel_win = gather(vel, 0, window)
    dac_win = gather(dac, 0, window)
    dac_fut = gather(dac, window, horizon)
    truth = gather(pos, window, horizon)[..., axis]  # [B, H] physical

    def tens(arr: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(arr, dtype=torch.float32, device=dev)

    frames = plant.encode_inputs(tens(pos_win), tens(vel_win), tens(dac_win))
    warmup = frames.transpose(1, 2).contiguous()  # [B, F, W]
    dac_future = plant.normalise_command(tens(dac_fut))  # [B, H, A]

    return warmup, dac_future, tens(truth)


@torch.no_grad()
def replay_errors(
    plant: Plant,
    pos: np.ndarray,
    dac: np.ndarray,
    axis: int,
    window: int,
    horizon: int,
    stride: int,
) -> torch.Tensor:
    """
    Free-run the plant on the captured command and return |position
    error| on 'axis' per step, [B, H], in the units of the passed 'pos'.

    Step 0 is the one-step (local) error; later steps are the compounding
    free-run drift.
    """

    warmup, dac_future, truth = build_replay_batch(
        plant, pos, dac, axis, window, horizon, stride
    )
    preds = plant.roll_forward(warmup, dac_future, horizon)[..., axis]
    pred_phys = preds * plant.pos_std[axis] + plant.pos_mean[axis]

    return (pred_phys - truth).abs()  # [B, H]
