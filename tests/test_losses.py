"""WeightedMSELoss tests: the per-target weighting must actually run."""

import torch

from nn_motion_control.training.losses import WeightedMSELoss


def test_two_arg_forward_does_not_raise():
    loss = WeightedMSELoss(weights=torch.tensor([1.0, 2.0]))
    value = loss(torch.zeros(4, 2), torch.ones(4, 2))
    assert torch.isfinite(value)


def test_weights_change_the_loss():
    preds, targets = (
        torch.zeros(3, 2),
        torch.ones(3, 2),
    )  # squared error 1 per element
    unweighted = WeightedMSELoss()(preds, targets)
    weighted = WeightedMSELoss(weights=torch.tensor([1.0, 3.0]))(preds, targets)
    assert torch.isclose(unweighted, torch.tensor(1.0))
    # mean over weights [1, 3] applied to unit errors -> (1 + 3) / 2 = 2
    assert torch.isclose(weighted, torch.tensor(2.0))
