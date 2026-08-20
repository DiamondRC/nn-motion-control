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

Captures are single-axis (one interferometer, the other axes undriven), so a
seed places the recorded axis on its measured trajectory and holds the others
at their quiescent value.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence

import numpy as np
import torch

logger = logging.getLogger(os.path.basename(__file__))

# Rows to leave between consecutive capture seeds. Adjacent windows overlap
# in all but one row, so seeds taken every row are near-duplicates.
DEFAULT_SEED_STRIDE = 64


def _capture_rows(
    pos: np.ndarray,
    dac: np.ndarray,
    axis: int,
    n_features: int,
    pos_cols: Sequence[int],
    vel_cols: Sequence[int],
    dac_cols: Sequence[int],
    quiescent_pos: np.ndarray,
) -> np.ndarray:
    """
    Lay a single-axis capture out as physical [T, F] rows in the dataset's
    axis-major channel layout.

    The recorded axis carries measured position, its per-step difference as
    velocity (the rollout's own convention) and the applied command; the
    other axes hold 'quiescent_pos' with zero velocity and command.
    """

    rows = np.zeros((len(pos), n_features), dtype=np.float64)
    for a, col in enumerate(pos_cols):
        rows[:, col] = quiescent_pos[a]
    rows[:, pos_cols[axis]] = pos
    if len(vel_cols):
        rows[:, vel_cols[axis]] = np.diff(pos, prepend=pos[0])
    rows[:, dac_cols[axis]] = dac

    return rows


def append_capture_seeds(
    inputs: torch.Tensor,
    in_mean: torch.Tensor,
    in_std: torch.Tensor,
    capture_paths: Sequence[str],
    axis: int,
    window_size: int,
    max_horizon: int,
    pos_cols: Sequence[int],
    vel_cols: Sequence[int],
    dac_cols: Sequence[int],
    decimate: int = 1,
    stride: int = DEFAULT_SEED_STRIDE,
) -> tuple[torch.Tensor, np.ndarray]:
    """
    Append normalised capture rows to the shared input tensor and return
    (extended inputs, new window starts).

    The starts index the extended tensor, so they drop straight into the
    same rollout loader the dataset seeds use. Each capture contributes
    only starts whose window and horizon stay inside that capture, so no
    rollout crosses a recording boundary. Position units are the dataset's
    (the captures are recorded in the same encoder counts), so the rows are
    normalised with the dataset's own fitted stats.
    """

    from nn_motion_control.eval.replay import load_capture

    dev = inputs.device
    n_features = inputs.shape[1]
    span = window_size + max_horizon
    # Hold the undriven axes at the dataset's own quiescent value (the
    # normalisation mean is the recordings' average position).
    quiescent = in_mean[list(pos_cols)].cpu().numpy()
    chunks: list[torch.Tensor] = []
    starts: list[np.ndarray] = []
    offset = inputs.shape[0]

    for path in capture_paths:
        pos, dac = load_capture(path, decimate=decimate)
        if len(pos) < span:
            logger.warning("Capture too short to seed, skipping: %s", path)
            continue
        rows = _capture_rows(
            pos,
            dac,
            axis,
            n_features,
            pos_cols,
            vel_cols,
            dac_cols,
            quiescent,
        )
        tens = torch.as_tensor(rows, dtype=inputs.dtype, device=dev)
        chunks.append((tens - in_mean.to(dev)) / in_std.to(dev))
        starts.append(offset + np.arange(0, len(pos) - span, stride))
        offset += len(pos)
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
