"""
Replay a recorded command stream through a frozen plant.

Drives the plant open-loop with a real command trajectory from an
external capture (not the training set) and measures how its predicted
position tracks the recorded position: the one-step error is the plant's
local fidelity out of distribution, the K-step free-run error is how fast
it diverges. The capture is single-axis (one interferometer); the other
axes are held quiescent, matching a single-axis move.

Position and command in the capture are assumed to already be in the
plant's channel units (encoder / command counts), so they map straight
onto the model inputs; only the sample rate may differ (decimate to the
plant rate).
"""

from __future__ import annotations

import numpy as np
import torch

from nn_motion_control.plant.plant import Plant

# PandA capture channel names: interferometer position and applied
# command (both in the plant's counts).
CAPTURE_POSITION = "INENC1.VAL.Value"
CAPTURE_COMMAND = "BRETT_PID.REAL_OUTPUT.Value"


def load_capture(
    path: str,
    decimate: int = 1,
    pos_key: str = CAPTURE_POSITION,
    dac_key: str = CAPTURE_COMMAND,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load (position, command) from an HDF5 capture, decimated to the plant
    rate.

    Returns two float64 [N] arrays. 'decimate' keeps every d-th sample (a
    20 kHz capture against a 10 kHz plant uses d=2).
    """

    import h5py

    from nn_motion_control.data._hdf5 import as_dataset

    with h5py.File(path, "r") as f:
        pos = as_dataset(f, pos_key)[::decimate].astype(np.float64)
        dac = as_dataset(f, dac_key)[::decimate].astype(np.float64)

    return pos, dac


def build_replay_batch(
    plant: Plant,
    pos: np.ndarray,
    dac: np.ndarray,
    axis: int,
    window: int,
    horizon: int,
    stride: int,
    quiescent: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build batched rollout seeds from a single-axis capture.

    Slides a start every 'stride' samples; each seed's active 'axis'
    carries the real position/velocity/command and the other axes are
    held at 'quiescent' (undriven). Velocity is the per-step position
    delta (the plant's own rollout convention). Returns (warmup [B, F, W],
    dac_future [B, H, A], truth [B, H]) with the active axis's true
    future position, on the plant's device.
    """

    dev = plant.device
    axes = len(plant.layout.pos_cols)
    vel = np.diff(pos, prepend=pos[0])
    starts = np.arange(0, len(pos) - window - horizon, stride)
    if starts.size == 0:
        raise ValueError("Capture too short for the requested window/horizon")

    def gather(arr: np.ndarray, offset: int, length: int) -> np.ndarray:
        idx = starts[:, None] + offset + np.arange(length)[None, :]

        return arr[idx]  # [B, length]

    batch = starts.size

    def axis_field(win: np.ndarray, length: int, fill: float) -> np.ndarray:
        field = np.full((batch, length, axes), fill, dtype=np.float64)
        field[..., axis] = win

        return field

    pos_win = axis_field(gather(pos, 0, window), window, quiescent)
    vel_win = axis_field(gather(vel, 0, window), window, 0.0)
    dac_win = axis_field(gather(dac, 0, window), window, 0.0)
    dac_fut = axis_field(gather(dac, window, horizon), horizon, 0.0)
    truth = gather(pos, window, horizon)  # [B, H] physical, active axis

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
    quiescent: float = 0.0,
) -> torch.Tensor:
    """
    Free-run the plant on the captured command and return |position
    error| per step, [B, H], in the position channel's physical units.

    Step 0 is the one-step (local) error; later steps are the compounding
    free-run drift. Convert to nm at the call site with the channel
    resolution, as the horizon eval does.
    """

    warmup, dac_future, truth = build_replay_batch(
        plant, pos, dac, axis, window, horizon, stride, quiescent
    )
    preds = plant.roll_forward(warmup, dac_future, horizon)[..., axis]
    pred_phys = preds * plant.pos_std[axis] + plant.pos_mean[axis]

    return (pred_phys - truth).abs()  # [B, H]
