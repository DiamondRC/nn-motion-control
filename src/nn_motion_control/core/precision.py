"""
Numeric precision switches shared by plant and controller training.
"""

from __future__ import annotations

import torch


def strict_float32() -> None:
    """
    Run float32 matmuls and convolutions at full precision.

    The container enables TF32, whose 10-bit mantissa resolves a normalised
    position only to hundreds of nm, too coarse for nm-level work.
    """

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def allow_tf32() -> None:
    """
    Let float32 matmuls and convolutions run in TF32, for runs whose config
    opts in to trade resolution for speed.
    """

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
