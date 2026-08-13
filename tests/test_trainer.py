"""Trainer early-stopping.

The improvement threshold is relative (scale-invariant).
"""

import torch
from torch import nn

from nn_motion_control.training.trainer import Trainer, config_overrides


def test_config_overrides_casts_present_and_omits_absent():
    casts = {
        "curriculum_start": int,
        "step_weight": float,
        "axis_weights": list,
    }
    source = {
        "curriculum_start": 8.0,
        "axis_weights": [1.0, 2.0],
        "unrelated": 5,
    }
    out = config_overrides(source, casts)
    # Present keys are cast to their declared type, a float horizon
    # becomes an int.
    assert out == {"curriculum_start": 8, "axis_weights": [1.0, 2.0]}
    assert isinstance(out["curriculum_start"], int)
    # An absent key is omitted so the trainer signature's own default applies.
    assert "step_weight" not in out
    # Keys outside the cast map are ignored.
    assert "unrelated" not in out


def _trainer(min_delta: float) -> Trainer:
    # Bypass the heavy __init__; _is_improvement only reads self.min_delta.
    t = Trainer.__new__(Trainer)
    t.min_delta = min_delta
    return t


class _FunctionalScaler:
    """A no-scaling GradScaler stand-in that actually applies the
    optimiser step.
    """

    def scale(self, loss):
        return loss

    def unscale_(self, optimizer):
        pass

    def step(self, optimizer):
        optimizer.step()

    def update(self):
        pass


def test_validate_epoch_averages_over_counted_batches():
    # A skipped NaN batch must not deflate the mean: divide by
    # counted, not len().
    t = Trainer.__new__(Trainer)
    t.device = "cpu"
    t.training_dtype = torch.bfloat16
    t.model = nn.Identity()
    outs = iter([torch.tensor(float("nan")), torch.tensor(4.0)])
    t._forward_loss = lambda batch: next(outs)
    t.val_loader = [0, 1]
    assert t._validate_epoch() == 4.0  # 4.0 / 1 counted, not 4.0 / 2


def test_train_epoch_flushes_incomplete_accumulation_cycle():
    # One batch with accumulation_steps=4 never completes a full
    # cycle, without the end-of-epoch flush its gradient would be
    # silently discarded and no step taken.
    torch.manual_seed(0)
    t = Trainer.__new__(Trainer)
    t.device = "cpu"
    t.training_dtype = torch.bfloat16
    t.accumulation_steps = 4
    t.grad_clip_norm = 1.0
    t.scaler = _FunctionalScaler()
    model = nn.Linear(1, 1)
    t.model = model
    t.optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    t.train_loader = [0]  # single batch, incomplete cycle
    t._forward_loss = lambda batch: (model(torch.ones(1, 1)).sum() - 5.0) ** 2

    before = model.weight.detach().clone()
    t._train_epoch()
    assert not torch.equal(
        model.weight.detach(), before
    )  # the flush applied a step


def test_first_epoch_beats_inf():
    assert _trainer(0.01)._is_improvement(1.0, float("inf"))


def test_relative_threshold_is_scale_invariant():
    t = _trainer(0.01)  # require a 1% relative improvement
    # 2% better counts, 0.5% better does not, identically at any
    # loss magnitude.
    assert t._is_improvement(0.98, 1.0)
    assert not t._is_improvement(0.995, 1.0)
    assert t._is_improvement(0.98e-5, 1.0e-5)
    assert not t._is_improvement(0.995e-5, 1.0e-5)


def test_zero_min_delta_saves_any_strict_improvement():
    t = _trainer(0.0)
    assert t._is_improvement(0.999999e-5, 1.0e-5)
    assert not t._is_improvement(1.0e-5, 1.0e-5)  # equal is not an improvement
