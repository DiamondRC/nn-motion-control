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


def test_a_divergent_validation_batch_fails_the_epoch():
    # Averaging only the finite batches would rank a partly unstable model
    # as the best, so one non-finite batch scores the epoch inf.
    t = Trainer.__new__(Trainer)
    t.device = "cpu"
    t.training_dtype = torch.bfloat16
    t.model = nn.Identity()
    outs = iter([torch.tensor(float("nan")), torch.tensor(4.0)])
    t._forward_loss = lambda batch: next(outs)
    t.val_loader = [0, 1]
    assert t._validate_epoch() == float("inf")


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


def _adam_trainer(batches, accumulation=1, seed=0):
    torch.manual_seed(seed)
    t = Trainer.__new__(Trainer)
    t.device = "cpu"
    t.training_dtype = torch.bfloat16
    t.accumulation_steps = accumulation
    t.grad_clip_norm = 1.0
    t.scaler = _FunctionalScaler()
    model = nn.Linear(1, 1)
    t.model = model
    t.optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    t.train_loader = batches

    def forward_loss(batch):
        x = torch.full((1, 1), 1.0 + batch)  # NaN input on a NaN batch
        return (model(x).sum() - 5.0) ** 2

    t._forward_loss = forward_loss

    return t, model


def test_non_finite_batch_keeps_the_rest_of_its_accumulation_cycle():
    # A NaN batch after a finite one in the same cycle must leave the
    # finite batch's gradient in place: the step matches a cycle of the
    # finite batch alone.
    clean, clean_model = _adam_trainer([0.0], accumulation=2)
    clean._train_epoch()
    t, model = _adam_trainer([0.0, float("nan")], accumulation=2)
    t._train_epoch()
    assert torch.allclose(model.weight, clean_model.weight)
    assert not torch.equal(model.weight, torch.zeros_like(model.weight))


def test_non_finite_step_moves_neither_weights_nor_optimiser_state():
    t, model = _adam_trainer([0.0, 0.0])
    t._train_epoch()
    weight = model.weight.detach().clone()
    state = {k: v.clone() for k, v in t.optimizer.state[model.weight].items()}
    t.train_loader = [float("nan")]
    loss = t._train_epoch()
    assert torch.equal(model.weight, weight)
    for k, v in t.optimizer.state[model.weight].items():
        assert torch.equal(v, state[k]), k
    # Nothing finite was averaged, so the epoch cannot read as an
    # improvement.
    assert loss == float("inf")


def test_non_finite_loss_with_finite_gradients_does_not_step():
    # An overflow behind finite local derivatives backpropagates finite
    # zero gradients, the norm alone reads clean, yet the step must still
    # be skipped or Adam would move on its stale momentum.
    t, model = _adam_trainer([0.0, 0.0])
    t._train_epoch()
    weight = model.weight.detach().clone()
    state = {k: v.clone() for k, v in t.optimizer.state[model.weight].items()}

    def overflow(batch):
        return model(torch.ones(1, 1)).sum() * 0.0 + float("inf")

    t._forward_loss = overflow
    t.train_loader = [0.0]
    t._train_epoch()
    assert torch.equal(model.weight, weight)

    for k, v in t.optimizer.state[model.weight].items():
        assert torch.equal(v, state[k]), k


def test_all_non_finite_validation_is_never_the_best():
    t = Trainer.__new__(Trainer)
    t.device = "cpu"
    t.training_dtype = torch.bfloat16
    t.model = nn.Identity()
    t._forward_loss = lambda batch: torch.tensor(float("nan"))
    t.val_loader = [0, 1]
    assert t._validate_epoch() == float("inf")


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
