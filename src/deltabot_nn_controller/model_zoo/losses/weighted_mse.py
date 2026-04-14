import torch
import torch.nn as nn


class WeightedMSELoss(nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.register_buffer("weights", torch.as_tensor(weights, dtype=torch.float32))

    def forward(self, preds, targets):
        loss = (preds - targets) ** 2
        loss = loss * self.weights
        return loss.mean()
