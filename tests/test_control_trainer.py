"""
training.control: the closed-loop objective and policy-gradient trainer.
"""

import numpy as np
import pytest
import torch
from torch import nn

from nn_motion_control.control.controller import (
    ControllerNet,
    FeatureSpec,
    NNPolicy,
)
from nn_motion_control.control.resource import QuantSpec
from nn_motion_control.data.dataset import DatasetMetadata
from nn_motion_control.data.normalize import NormStats
from nn_motion_control.plant.plant import Plant, RolloutLayout
from nn_motion_control.training.control import ControlTrainer, control_loss


class _DacDriven(nn.Module):
    """Toy plant whose position change is proportional to the applied DAC."""

    def __init__(self, gain: float = 1.0):
        super().__init__()
        self.gain = gain

    def forward(self, window):
        return (
            self.gain * window[:, 2:3, -1]
        )  # delta-position = gain * last DAC


def _dac_plant(gain: float = 1.0) -> Plant:
    layout = RolloutLayout(
        pos_cols=[0], vel_cols=[1], dac_cols=[2], n_features=3
    )
    ones = torch.ones(3)
    in_stats = NormStats(
        mean=torch.zeros(3), std=ones, normalizable=ones.bool()
    )
    t_stats = NormStats(
        mean=torch.zeros(1),
        std=torch.ones(1),
        normalizable=torch.ones(1).bool(),
    )
    return Plant(_DacDriven(gain), in_stats, t_stats, layout, device="cpu")


def _tiny_controller():
    torch.manual_seed(0)
    net = ControllerNet(1, 1, [4], [QuantSpec(32, 48), QuantSpec(32, 16)])
    policy = NNPolicy(
        net, FeatureSpec(("error",)), torch.tensor([[-10.0, 10.0]]), 1
    )
    return net, policy


def _step_to(origin, horizon, amplitude=1.0):
    target = origin + amplitude
    return target.unsqueeze(1).expand(-1, horizon, -1)


def test_control_loss_terms():
    ref = torch.zeros(2, 4, 1)
    # Effort penalises command magnitude; zero DAC -> zero effort.
    zero = control_loss(
        ref,
        ref,
        torch.zeros(2, 4, 1),
        tracking_weight=0.0,
        effort_weight=1.0,
        rate_weight=1.0,
    )
    assert torch.allclose(zero, torch.zeros(()))
    # A constant command has zero rate but nonzero effort.
    const = control_loss(
        ref,
        ref,
        torch.ones(2, 4, 1),
        tracking_weight=0.0,
        effort_weight=0.0,
        rate_weight=1.0,
    )
    assert torch.allclose(const, torch.zeros(()))
    effort = control_loss(
        ref,
        ref,
        torch.ones(2, 4, 1),
        tracking_weight=0.0,
        effort_weight=1.0,
        rate_weight=0.0,
    )
    assert torch.allclose(effort, torch.ones(()))


def test_huber_tames_the_tail():
    ref = torch.zeros(1, 1, 1)
    dac = torch.zeros(1, 1, 1)
    big = torch.full((1, 1, 1), 1000.0)  # a divergent-tail error
    mse = control_loss(
        big, ref, dac, tracking_weight=1.0, effort_weight=0.0, rate_weight=0.0
    )
    hub = control_loss(
        big,
        ref,
        dac,
        tracking_weight=1.0,
        effort_weight=0.0,
        rate_weight=0.0,
        huber_delta=100.0,
    )
    # Beyond delta Huber grows linearly: 100*(1000 - 50) = 95000, vs MSE 1e6.
    assert hub < mse
    assert torch.allclose(hub, torch.tensor(95000.0), rtol=1e-3)


def test_axis_weights_scale_per_axis_tracking():
    # Same per-axis error on 2 axes, weighting axis 0 up makes it
    # dominate the loss.
    pos = torch.tensor([[[1.0, 1.0]]])  # [B=1, H=1, A=2], error 1 on both axes
    ref = torch.zeros(1, 1, 2)
    dac = torch.zeros(1, 1, 2)
    uniform = control_loss(
        pos, ref, dac, tracking_weight=1.0, effort_weight=0.0, rate_weight=0.0
    )
    weighted = control_loss(
        pos,
        ref,
        dac,
        tracking_weight=1.0,
        effort_weight=0.0,
        rate_weight=0.0,
        axis_weights=torch.tensor([3.0, 1.0]),
    )
    # per-axis err^2 mean: uniform (1+1)/2=1; weighted (3+1)/2=2.
    assert torch.allclose(uniform, torch.tensor(1.0))
    assert torch.allclose(weighted, torch.tensor(2.0))


def test_feature_stats_covers_all_allowed_features():
    # Guard against drift between ALLOWED_FEATURES and
    # Plant.feature_stats' table.
    from nn_motion_control.control.resource import ALLOWED_FEATURES

    mean, std = _dac_plant().feature_stats(ALLOWED_FEATURES)
    assert mean.shape == (1, len(ALLOWED_FEATURES))
    assert std.shape == (1, len(ALLOWED_FEATURES))


def test_velocity_tracking_term():
    # positions step by 2/step, demanded velocity 2 -> zero velocity
    # error.
    pos = torch.tensor([[[0.0], [2.0], [4.0], [6.0]]])  # [B=1, H=4, A=1]
    ref = pos.clone()
    dac = torch.zeros(1, 4, 1)
    demand = torch.full((1, 4, 1), 2.0)  # matches the achieved per-step motion
    matched = control_loss(
        pos,
        ref,
        dac,
        tracking_weight=0.0,
        effort_weight=0.0,
        rate_weight=0.0,
        velocity_weight=1.0,
        reference_velocity=demand,
    )
    assert torch.allclose(matched, torch.zeros(()))  # achieved == demanded
    mismatched = control_loss(
        pos,
        ref,
        dac,
        tracking_weight=0.0,
        effort_weight=0.0,
        rate_weight=0.0,
        velocity_weight=1.0,
        reference_velocity=torch.zeros(1, 4, 1),
    )
    assert mismatched > matched  # demanding zero velocity while moving costs


def test_tracking_term_rewards_closeness():
    ref = torch.ones(2, 4, 1)
    dac = torch.zeros(2, 4, 1)
    near = control_loss(
        0.9 * ref,
        ref,
        dac,
        tracking_weight=1.0,
        effort_weight=0.0,
        rate_weight=0.0,
    )
    far = control_loss(
        0.1 * ref,
        ref,
        dac,
        tracking_weight=1.0,
        effort_weight=0.0,
        rate_weight=0.0,
    )
    assert near < far


def test_policy_gradient_reduces_tracking():
    plant = _dac_plant(1.0)
    net, policy = _tiny_controller()
    opt = torch.optim.Adam(net.parameters(), lr=0.1)
    b, w, h = 8, 4, 6
    warmup = torch.zeros(b, 3, w)

    losses = []

    for _ in range(120):
        origin, _ = plant.seed_state(warmup)
        ref = _step_to(origin, h)
        positions, dacs = plant.closed_loop_rollout(warmup, ref, policy, h)
        loss = control_loss(
            positions,
            ref,
            dacs,
            tracking_weight=1.0,
            effort_weight=0.0,
            rate_weight=0.0,
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0] * 0.5  # tracking loss falls substantially
    origin, _ = plant.seed_state(warmup)
    ref = _step_to(origin, h)
    positions, dacs = plant.closed_loop_rollout(warmup, ref, policy, h)
    assert (
        positions[:, -1, :] - ref[:, -1, :]
    ).abs().mean() < 0.3  # settles near target
    assert dacs.abs().max() <= 10.0 + 1e-4  # stays inside the safe range


class _StubScaler:
    """Minimal GradScaler stand-in (this test never runs the training step)."""

    def __init__(self, device, enabled):
        pass


def _fake_node_info() -> DatasetMetadata:
    ones = torch.ones(1)
    return DatasetMetadata(
        input_labels=np.array(["x_pos"]),
        target_labels=np.array(["x_dac"]),
        input_denorm_params={"mean": {"x_pos": 0.0}, "std": {"x_pos": 1.0}},
        target_denorm_params={"mean": {"x_dac": 0.0}, "std": {"x_dac": 1.0}},
        loss_weights=ones,
        input_stats=NormStats(torch.zeros(1), ones, ones.bool()),
        target_stats=NormStats(torch.zeros(1), ones, ones.bool()),
    )


def test_control_trainer_forward_loss_is_differentiable(tmp_path):
    plant = _dac_plant(1.0)
    net, policy = _tiny_controller()
    b, w, h = 4, 4, 5
    batch = (torch.zeros(b, 3, w), torch.zeros(b, h, 1), torch.zeros(b, h, 1))
    trainer = ControlTrainer(
        plant,
        net,
        policy,
        controller_config=None,
        reference_gen=lambda origin, horizon, generator=None: (
            _step_to(origin, horizon),
            torch.zeros_like(_step_to(origin, horizon)),
        ),
        max_horizon=h,
        curriculum_start=h,
        curriculum_ramp=0,
        train_loader=[batch],
        val_loader=[batch],
        device="cpu",
        scaler_class=_StubScaler,
        optimizer_class=torch.optim.Adam,
        criterion_class=nn.MSELoss,
        node_info=_fake_node_info(),
        max_epochs=1,
        learning_rate=1e-3,
        min_delta=0.0,
        patience=1,
        model_name="ctrl_smoke",
        save_path=str(tmp_path / "ctrl.pth"),
        logging=False,
        accumulation_steps=1,
        training_dtype=torch.float32,
        window_size=w,
        seed=0,
    )
    trainer._on_epoch_start(0)
    loss = trainer._forward_loss(batch)
    assert loss.ndim == 0 and loss.requires_grad
    loss.backward()
    assert net.linears[0].weight.grad is not None
    # The plant was frozen when the trainer took ownership of it.
    assert all(not p.requires_grad for p in plant.model.parameters())


def _make_trainer(
    tmp_path, *, reference_gen=None, axis_weights=None, max_horizon=6
):
    plant = _dac_plant(1.0)
    net, policy = _tiny_controller()

    def _default_gen(origin, horizon, generator=None):
        pos = _step_to(origin, horizon)
        return pos, torch.zeros_like(pos)

    return ControlTrainer(
        plant,
        net,
        policy,
        controller_config=None,
        reference_gen=reference_gen or _default_gen,
        max_horizon=max_horizon,
        curriculum_start=max_horizon,
        curriculum_ramp=0,
        axis_weights=axis_weights,
        train_loader=[],
        val_loader=[],
        device="cpu",
        scaler_class=_StubScaler,
        optimizer_class=torch.optim.Adam,
        criterion_class=nn.MSELoss,
        node_info=_fake_node_info(),
        max_epochs=1,
        learning_rate=1e-3,
        min_delta=0.0,
        patience=1,
        model_name="ctrl",
        save_path=str(tmp_path / "ctrl.pth"),
        logging=False,
        accumulation_steps=1,
        training_dtype=torch.float32,
        window_size=4,
        seed=0,
    )


def test_control_trainer_normalises_axis_weights(tmp_path):
    # Like RolloutTrainer, a focus vector is normalised to mean 1 so
    # it rebalances axes without changing the overall loss magnitude
    # the early-stopping threshold sees.
    trainer = _make_trainer(tmp_path, axis_weights=[3.0, 1.0, 1.0])
    assert torch.allclose(trainer.axis_weights.mean(), torch.tensor(1.0))
    assert torch.allclose(trainer.axis_weights, torch.tensor([1.8, 0.6, 0.6]))


def test_validation_references_differ_across_batches(tmp_path):
    # The per-epoch validation generator must advance across batches
    # (diverse refs), not be reseeded per batch (which would collapse
    # every batch to the same reference).
    def stochastic_ref(origin, horizon, generator=None):
        target = origin + torch.rand(
            origin.shape, generator=generator, device=origin.device
        )
        pos = target.unsqueeze(1).expand(-1, horizon, -1)
        return pos, torch.zeros_like(pos)

    trainer = _make_trainer(tmp_path, reference_gen=stochastic_ref)
    trainer.model.eval()  # validation path -> uses self._val_gen
    trainer._on_epoch_start(0)
    batch = (torch.zeros(4, 3, 4), torch.zeros(4, 6, 1), torch.zeros(4, 6, 1))

    trainer._val_gen = torch.Generator(device="cpu").manual_seed(0)
    l1 = trainer._forward_loss(batch).item()
    l2 = trainer._forward_loss(batch).item()
    assert l1 != l2  # generator advanced -> different references across batches

    # Reseeding at the start of a validation epoch reproduces the
    # sequence (comparable).
    trainer._val_gen = torch.Generator(device="cpu").manual_seed(0)
    assert trainer._forward_loss(batch).item() == pytest.approx(l1)
