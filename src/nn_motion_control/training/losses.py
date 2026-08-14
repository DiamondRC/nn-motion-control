"""
Per-target weighted mean-squared-error loss.
"""

import torch
import torch.nn as nn
from torch import Tensor


class WeightedMSELoss(nn.Module):
    """
    MSE scaled per target channel.
    The weight vector is applied before the mean.
    """

    def __init__(self, weights: Tensor | None = None):
        super().__init__()
        if weights is not None:
            self.register_buffer(
                "weights", torch.as_tensor(weights, dtype=torch.float32)
            )
        else:
            self.weights = None

    def forward(self, preds: Tensor, targets: Tensor) -> Tensor:
        loss = (preds - targets) ** 2

        if self.weights is not None:
            loss = loss * self.weights

        return loss.mean()
