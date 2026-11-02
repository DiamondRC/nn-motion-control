"""
Seed controller rollouts from recorded closed-loop captures.

The identification dataset holds open-loop excitation, so the states a
controller actually operates in mid-scan (moving at scan speed, carrying a
small following error, the flexure loaded away from rest) are structurally
absent from it; quiescent seeding covers only the at-rest start. A recorded
closed-loop capture supplies those states as ground truth.

Only the recorded state is used, never the recording controller's commands
as a target: a capture window seeds the plant's history and initial
(position, velocity), and the policy under training issues every command
from step 0 onward. Which controller drove the stage there does not matter,
the manifold of reachable states is a property of the stage.

Each capture is read in the units its file records and placed in the
dataset's own frame (see 'panda.DatasetFrame'): resampled as the dataset was
and referenced to the stage's rest, or centred, as the dataset's recordings
were. Every axis with an interferometer channel carries its measured
position, an axis without one sits at the frame's zero with no command.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence

import numpy as np
import torch

from nn_motion_control.data.panda import DatasetFrame

logger = logging.getLogger(os.path.basename(__file__))

# Rows to leave between consecutive capture seeds. Adjacent windows overlap
# in all but one row, so seeds taken every row are near-duplicates.
DEFAULT_SEED_STRIDE = 64


def _capture_rows(
    pos: np.ndarray,
    dac: np.ndarray,
    n_features: int,
    pos_cols: Sequence[int],
    vel_cols: Sequence[int],
    dac_cols: Sequence[int],
) -> np.ndarray:
    """
    Lay a capture ([A, n] position and command) out as physical [n, F] rows
    in the dataset's axis-major channel layout, velocity being the per-step
    position difference (the rollout's own convention).
    """

    rows = np.zeros((pos.shape[1], n_features), dtype=np.float64)

    for a, col in enumerate(pos_cols):
        rows[:, col] = pos[a]
        rows[:, dac_cols[a]] = dac[a]
        if len(vel_cols):
            rows[:, vel_cols[a]] = np.diff(pos[a], prepend=pos[a, 0])

    return rows


def append_capture_seeds(
    inputs: torch.Tensor,
    in_mean: torch.Tensor,
    in_std: torch.Tensor,
    capture_paths: Sequence[str],
    frame: DatasetFrame,
    axes: Sequence[str],
    resolution_nm: float,
    axis: int,
    window_size: int,
    max_horizon: int,
    pos_cols: Sequence[int],
    vel_cols: Sequence[int],
    dac_cols: Sequence[int],
    stride: int = DEFAULT_SEED_STRIDE,
) -> tuple[torch.Tensor, np.ndarray]:
    """
    Append normalised capture rows to the shared input tensor and return
    (extended inputs, new window starts).

    'axis' is the axis the recorded controller drove. The starts index the
    extended tensor, so they drop straight into the same rollout loader the
    dataset seeds use. Each capture contributes only starts whose window and
    horizon stay inside that capture, so no rollout crosses a recording
    boundary. The rows are normalised with the dataset's own fitted stats.
    """

    from nn_motion_control.eval.replay import load_capture

    dev = inputs.device
    n_features = inputs.shape[1]
    span = window_size + max_horizon
    chunks: list[torch.Tensor] = []
    starts: list[np.ndarray] = []
    offset = inputs.shape[0]

    for path in capture_paths:
        pos, dac = load_capture(path, frame, axes, resolution_nm, axis)
        n = pos.shape[1]
        if n < span:
            logger.warning("Capture too short to seed, skipping: %s", path)
            continue
        rows = _capture_rows(pos, dac, n_features, pos_cols, vel_cols, dac_cols)
        tens = torch.as_tensor(rows, dtype=inputs.dtype, device=dev)
        chunks.append((tens - in_mean.to(dev)) / in_std.to(dev))
        starts.append(offset + np.arange(0, n - span + 1, stride))
        offset += n
    if not chunks:
        raise ValueError("No capture supplied a usable seed window")
    extended = torch.cat([inputs, *chunks], dim=0)
    seed_starts = np.concatenate(starts)
    logger.info(
        "Capture seeds: %d starts from %d capture(s)",
        len(seed_starts),
        len(chunks),
    )

    return extended, seed_starts
