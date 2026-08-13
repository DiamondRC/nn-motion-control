"""
Global RNG seeding for reproducible runs.
"""

from __future__ import annotations

import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """
    Seed Python, NumPy and torch (including CUDA) so weight init, dropout
    and shuffling are reproducible across a run.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
