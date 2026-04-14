import torch
import torch.nn as nn
from torch import Tensor


class WeightedMSELoss(nn.Module):
    def __init__(self, weights: Tensor | None = None):
        super().__init__()
        # self.register_buffer("weights", torch.as_tensor(weights, dtype=torch.float32))

        if weights is not None:
            self.register_buffer(
                "weights", torch.as_tensor(weights, dtype=torch.float32)
            )
        else:
            self.weights = None

    def forward(self, preds, targets):
        loss = (preds - targets) ** 2

        if self.weights is not None:
            loss = loss * self.weights

        return loss.mean()
