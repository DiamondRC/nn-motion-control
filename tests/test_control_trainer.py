"""
training.control: the closed-loop objective and policy-gradient trainer.
"""

import math
from types import SimpleNamespace

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
from nn_motion_control.training.control import (
    ControlTrainer,
    control_loss,
    growth_penalty,
    perturbation_growth,
)


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
    # Behind the plant's one-step command delay, error alone overshoots an
    # integrator, so the controller also sees velocity to damp it.
    torch.manual_seed(0)
    net = ControllerNet(2, 1, [8], [QuantSpec(32, 48), QuantSpec(32, 16)])
    policy = NNPolicy(
        net,
        FeatureSpec(("error", "velocity")),
        torch.tensor([[-10.0, 10.0]]),
        1,
    )
    opt = torch.optim.Adam(net.parameters(), lr=0.05)
    b, w, h = 8, 4, 12
    warmup = torch.zeros(b, 3, w)

    losses = []

    for _ in range(200):
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
    tmp_path,
    *,
    reference_gen=None,
    axis_weights=None,
    max_horizon=6,
    val_horizon=None,
    val_loader=None,
    **extra,
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
        val_horizon=val_horizon,
        curriculum_start=max_horizon,
        curriculum_ramp=0,
        axis_weights=axis_weights,
        train_loader=[],
        val_loader=val_loader or [],
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
        **extra,
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


# The helper trains in float32, which CPU autocast declines (and warns).
@pytest.mark.filterwarnings("ignore:In CPU autocast")
def test_validation_runs_at_the_validation_horizon(tmp_path):
    # Early stopping must see the long horizon, while training resumes at
    # its own curriculum horizon afterwards.
    seen = []

    def recording_ref(origin, horizon, generator=None):
        seen.append(horizon)
        pos = _step_to(origin, horizon)
        return pos, torch.zeros_like(pos)

    batch = (torch.zeros(4, 3, 4), torch.zeros(4, 6, 1), torch.zeros(4, 6, 1))
    trainer = _make_trainer(
        tmp_path,
        reference_gen=recording_ref,
        max_horizon=6,
        val_horizon=20,
        val_loader=[batch],
    )
    trainer._on_epoch_start(0)
    trainer._validate_epoch()
    assert seen == [20]
    assert trainer._cur_h == 6


def test_perturbation_growth_reads_decay_and_amplification():
    h = 40
    t = torch.arange(h, dtype=torch.float32)[None, :, None]
    nominal = torch.zeros(2, h, 1)
    decaying = 10.0 * torch.exp(-t / 5.0).expand(2, h, 1)
    growing = 10.0 * torch.exp(t / 10.0).expand(2, h, 1)
    assert (perturbation_growth(nominal, decaying, 10.0, 0.5) < 0.1).all()
    assert (perturbation_growth(nominal, growing, 10.0, 0.5) > 1.0).all()
    # The penalty stays modest however hard a rollout diverges.
    huge = perturbation_growth(nominal, 1e6 * growing, 10.0, 0.5)
    assert growth_penalty(huge) < 100.0


def test_growth_penalty_spares_decaying_perturbations():
    # A stable loop that has not yet removed the perturbation (growth below
    # one) costs nothing, only amplification is penalised.
    assert growth_penalty(torch.tensor([0.2, 0.9, 1.0])) == 0.0
    assert growth_penalty(torch.tensor([2.0])) > 0.0


def test_growth_term_is_gated_folded_and_training_only(tmp_path, monkeypatch):
    import nn_motion_control.training.control as ctl

    b, h = 4, 6
    batch = (torch.zeros(b, 3, 4), torch.zeros(b, h, 1), torch.zeros(b, h, 1))
    plain = _make_trainer(tmp_path)
    gated = _make_trainer(tmp_path, growth_weight=2.0, growth_min_horizon=h + 1)
    grown = _make_trainer(tmp_path, growth_weight=2.0, growth_min_horizon=h)

    for t in (gated, grown):
        t.model.load_state_dict(plain.model.state_dict())

    for t in (plain, gated, grown):
        t._on_epoch_start(0)
        t.model.train()
    base = plain._forward_loss(batch)
    # Below the minimum horizon the penalty is off entirely.
    assert torch.isclose(gated._forward_loss(batch), base)
    # Above it the term is added, its gradient reaching the controller.
    monkeypatch.setattr(ctl, "growth_penalty", lambda g: g.mean() + 1.0)
    loss = grown._forward_loss(batch)
    assert loss > base + 2.0 * 1.0 - 1e-4
    loss.backward()
    assert grown.model.linears[0].weight.grad is not None
    # Validation scores tracking alone, so runs stay comparable. The nets
    # are synced first: each learnt its own activation ranges in training.
    grown.model.load_state_dict(plain.model.state_dict())
    grown.model.eval()
    plain.model.eval()
    with torch.no_grad():
        assert torch.isclose(
            grown._forward_loss(batch), plain._forward_loss(batch)
        )


def test_growth_settings_are_range_checked(tmp_path):
    with pytest.raises(ValueError, match="growth_perturb"):
        _make_trainer(tmp_path, growth_weight=1.0, growth_perturb=0.0)
    with pytest.raises(ValueError, match="growth_tail"):
        _make_trainer(tmp_path, growth_weight=1.0, growth_tail=0.0)


def test_row_cap_drops_diverged_rows_in_training_only(tmp_path):
    t = _make_trainer(tmp_path, row_error_cap=5.0)
    pos = torch.zeros(3, 4, 1)
    ref = torch.zeros(3, 4, 1)
    pos[1] = 100.0  # one row far off its reference
    t.model.train()
    assert t._tracked_rows(pos, ref).tolist() == [True, False, True]
    # Validation scores every row.
    t.model.eval()
    assert t._tracked_rows(pos, ref).all()
    # A row that diverges only at the end is caught by its tail.
    late = torch.zeros(1, 8, 1)
    late[0, -1] = 100.0
    t.model.train()
    assert not t._tracked_rows(late, torch.zeros_like(late)).any()
    with pytest.raises(ValueError, match="Row_error_cap"):
        _make_trainer(tmp_path, row_error_cap=0.0)


def test_projected_growth_catches_a_slow_mode_the_end_ratio_misses():
    from nn_motion_control.training.control import projected_growth

    h, perturb = 400, 10.0
    t = torch.arange(h, dtype=torch.float32)[None, :, None]
    nominal = torch.zeros(1, h, 1)
    # A fast decay to 3 nm under a slow mode growing with a 3000-step
    # e-fold: it ends below the 10 nm it started at, yet it is unstable.
    slow = 7.0 * torch.exp(-t / 10.0) + 0.5 * torch.exp(t / 3000.0)
    assert perturbation_growth(nominal, slow, perturb, 0.5).item() < 1.0
    projected = projected_growth(nominal, slow, perturb, project_steps=10_000)
    assert projected.item() > 1.0
    # A decaying gap, or one sunk into the numerical floor, reads as none.
    fading = 10.0 * torch.exp(-t / 100.0)
    assert projected_growth(nominal, fading, perturb, 10_000).item() < 1.0
    floor = torch.full((1, h, 1), 1e-3)
    assert projected_growth(nominal, floor, perturb, 10_000).item() == 0.0


def test_an_all_diverged_batch_is_not_trained_on(tmp_path):
    # With every row past the cap the loss is non-finite, so the trainer
    # skips the step instead of following exploding gradients.
    b, h = 4, 6
    batch = (torch.zeros(b, 3, 4), torch.zeros(b, h, 1), torch.zeros(b, h, 1))
    t = _make_trainer(tmp_path, row_error_cap=1e-9)
    t._on_epoch_start(0)
    t.model.train()
    assert not torch.isfinite(t._forward_loss(batch))


def test_growth_pairs_follow_their_base_rows(tmp_path, monkeypatch):
    import nn_motion_control.training.control as ctl

    b, h = 4, 6
    batch = (torch.zeros(b, 3, 4), torch.zeros(b, h, 1), torch.zeros(b, h, 1))
    t = _make_trainer(
        tmp_path, growth_weight=1.0, growth_min_horizon=h, growth_fraction=0.5
    )
    seen = []

    def growth(nominal, perturbed, *_):
        seen.append(nominal.shape[0])
        return torch.ones(nominal.shape[0])

    monkeypatch.setattr(ctl, "perturbation_growth", growth)
    t._on_epoch_start(0)
    t.model.train()
    # The first base row diverged, so only the second pair is judged.
    keep = torch.tensor([False, True, True, True])
    monkeypatch.setattr(t, "_tracked_rows", lambda *_: keep)
    t._forward_loss(batch)
    assert seen == [1]
    # With every paired base row dropped the term is skipped, not NaN.
    keep = torch.tensor([False, False, True, True])
    monkeypatch.setattr(t, "_tracked_rows", lambda *_: keep)
    assert torch.isfinite(t._forward_loss(batch))
    assert seen == [1]


def test_warm_start_loads_a_matching_controller_and_refuses_others(tmp_path):
    from types import SimpleNamespace

    from nn_motion_control.control.resource import QuantSpec
    from nn_motion_control.training.control_run import warm_start

    net, _ = _tiny_controller()
    config = SimpleNamespace(
        architecture="mlp",
        features=("position",),
        hidden=[4],
        error_scale=None,
        error_clip=None,
        integral_lambdas=[],
        quant=[QuantSpec(16, 16), QuantSpec(16, 16)],
        safe_range=torch.tensor([[-1.0, 1.0]]),
        plant_checkpoint="plant.pth",
    )
    saved = {k: v.clone() for k, v in net.state_dict().items()}
    bundle = {
        "schema_version": 2,
        "artifact_type": "controller",
        "architecture": "mlp",
        "features": ["position"],
        "hidden": [4],
        "error_scale": None,
        "integral_lambdas": [],
        "quant": [{"weight_bits": 16, "act_bits": 16}] * 2,
        "safe_range": torch.tensor([[-1.0, 1.0]]),
        "plant_checkpoint": "plant.pth",
        "model_state_dict": saved,
    }
    path = tmp_path / "ctrl.pth"
    torch.save(bundle, path)
    fresh, _ = _tiny_controller()
    with torch.no_grad():
        for p in fresh.parameters():
            p.add_(1.0)  # genuinely different weights before the warm start
    warm_start(fresh, config, str(path))
    assert all(
        torch.equal(a, b)
        for a, b in zip(
            fresh.state_dict().values(), saved.values(), strict=True
        )
    )
    # Retraining at other widths is a valid start, and a bundle saved before
    # the fixed activation scales loads with none observed.
    old = {k: v for k, v in saved.items() if k != "act_amax"}
    wider = [{"weight_bits": 16, "act_bits": 24}] * 2
    torch.save({**bundle, "quant": wider, "model_state_dict": old}, path)
    warm_start(fresh, config, str(path))
    assert not fresh.get_buffer("act_amax").any()
    # A checkpoint trained on another plant is refused.
    torch.save({**bundle, "plant_checkpoint": "other.pth"}, path)
    with pytest.raises(ValueError, match="warm-start"):
        warm_start(fresh, config, str(path))


def test_command_feature_feeds_back_the_last_command_primed_from_the_seed():
    plant = _dac_plant(1.0)
    torch.manual_seed(0)
    net = ControllerNet(2, 1, [4], [QuantSpec(32, 48), QuantSpec(32, 16)])
    policy = NNPolicy(
        net,
        FeatureSpec(("error", "command")),
        torch.tensor([[-10.0, 10.0]]),
        1,
    )
    seen = []
    forward = net.forward

    def spy(x):
        seen.append(x[:, 1].clone())  # the command column

        return forward(x)

    net.forward = spy
    warmup = torch.zeros(2, 3, 4)
    warmup[:, 2, -1] = torch.tensor([3.0, -2.0])  # the commands holding them
    origin, _ = plant.seed_state(warmup)
    ref = _step_to(origin, 3)
    with torch.no_grad():
        _, dacs = plant.closed_loop_rollout(warmup, ref, policy, 3)
    assert torch.equal(seen[0], torch.tensor([3.0, -2.0]))
    assert torch.allclose(seen[1], dacs[:, 0, 0])
    assert torch.allclose(seen[2], dacs[:, 1, 0])
    with pytest.raises(ValueError, match="previous command"):
        FeatureSpec(("command",)).assemble(*[torch.zeros(1, 1)] * 4)


def test_warm_start_adds_an_input_at_zero_weight(tmp_path):
    from types import SimpleNamespace

    from nn_motion_control.control.resource import QuantSpec
    from nn_motion_control.training.control_run import warm_start

    quant = [QuantSpec(16, 48), QuantSpec(16, 16)]
    torch.manual_seed(0)
    old = ControllerNet(2, 1, [4], quant)  # features (error,) + one tap
    bundle = {
        "artifact_type": "controller",
        "architecture": "mlp",
        "features": ["error"],
        "hidden": [4],
        "error_scale": None,
        "integral_lambdas": [0.9],
        "quant": [{"weight_bits": 16, "act_bits": 48}] * 2,
        "safe_range": torch.tensor([[-1.0, 1.0]]),
        "plant_checkpoint": "plant.pth",
        "model_state_dict": old.state_dict(),
    }
    path = tmp_path / "ctrl.pth"
    torch.save(bundle, path)
    config = SimpleNamespace(
        architecture="mlp",
        axes=["x"],
        features=("error", "command"),
        hidden=[4],
        error_scale=None,
        error_clip=None,
        integral_lambdas=[0.9],
        quant=quant,
        safe_range=torch.tensor([[-1.0, 1.0]]),
        plant_checkpoint="plant.pth",
    )
    new = ControllerNet(3, 1, [4], quant)  # error, command, tap
    if torch.cuda.is_available():
        # The checkpoint loads on the CPU, the net may live on the GPU.
        new.cuda()
        warm_start(new, config, str(path))
        new.cpu()
    warm_start(new, config, str(path))
    old.eval()
    new.eval()
    x = torch.randn(5, 2)
    wide = torch.stack([x[:, 0], 7.0 * torch.randn(5), x[:, 1]], dim=1)
    assert torch.allclose(new(wide), old(x))
    # Reordering the saved features is refused.
    config.features = ("command", "error")
    torch.save({**bundle, "features": ["error", "command"]}, path)
    with pytest.raises(ValueError, match="warm-start"):
        warm_start(new, config, str(path))


def test_seed_curriculum_widens_the_training_seeds(tmp_path):
    from nn_motion_control.data.device_loader import RolloutWindowLoader

    w = 4
    inputs = torch.zeros(60, 3)
    inputs[:, 0] = torch.arange(60.0) * 10.0  # seed offset grows with row
    starts = torch.arange(0, 50, 5)
    loader = RolloutWindowLoader(
        inputs, starts, w, 2, batch_size=4, pos_cols=[0], dac_cols=[2]
    )
    t = _make_trainer(tmp_path, seed_curriculum=(100.0, 2))
    t.train_loader = loader
    t._on_epoch_start(0)
    assert len(loader.dataset) == 2  # last frames 30, 80 in, 130 out
    t._on_epoch_start(1)
    assert 2 < len(loader.dataset) < len(starts)
    t._on_epoch_start(2)
    assert len(loader.dataset) == len(starts)
    with pytest.raises(ValueError, match="Seed_curriculum"):
        _make_trainer(tmp_path, seed_curriculum=(0.0, 2))


def test_command_feedback_gradient_is_cut_by_default():
    torch.manual_seed(0)
    net = ControllerNet(2, 1, [4], [QuantSpec(32, 48), QuantSpec(32, 32)])

    def policy(steps):
        return NNPolicy(
            net,
            FeatureSpec(("error", "command")),
            torch.tensor([[-10.0, 10.0]]),
            1,
            command_backprop_steps=steps,
        )

    def fed_back_grad(p):
        p.reset(1, "cpu")
        pos = torch.zeros(1, 1)
        first = p(pos, pos, torch.ones(1, 1), pos)
        second = p(pos, pos, torch.ones(1, 1), pos)

        return torch.autograd.grad(second.sum(), first, allow_unused=True)[0]

    assert fed_back_grad(policy(0)) is None
    assert fed_back_grad(policy(4)) is not None


def test_log_tracking_loss_weighs_every_scale_alike():
    def cost(err):
        ref = torch.zeros(1, 4, 1)

        return control_loss(
            ref + err,
            ref,
            torch.zeros(1, 4, 1),
            tracking_weight=1.0,
            effort_weight=0.0,
            rate_weight=0.0,
            tracking_scale=1.0,
        )

    # A tenfold error costs about the same small or large.
    small = cost(100.0) - cost(10.0)
    large = cost(1e5) - cost(1e4)
    assert torch.isclose(small, large, rtol=0.01)
    # The gradient of a large error falls off instead of growing.
    err = torch.tensor([10.0, 1e4], requires_grad=True)
    pos = err[None, :, None]
    control_loss(
        pos,
        torch.zeros_like(pos),
        torch.zeros_like(pos),
        tracking_weight=1.0,
        effort_weight=0.0,
        rate_weight=0.0,
        tracking_scale=1.0,
    ).backward()
    assert err.grad is not None and err.grad[1] < err.grad[0]


def test_command_activity_flags_a_limit_cycle():
    from nn_motion_control.training.control import command_activity

    t = torch.arange(1000) / 10_000.0
    slow = torch.sin(2 * math.pi * 50.0 * t)
    ring = torch.sin(2 * math.pi * 800.0 * t)
    dacs = torch.stack([slow, ring], dim=0)[:, :, None] * 300.0
    std, share = command_activity(dacs, 10_000.0, 200.0)
    assert torch.allclose(
        std, torch.full((2,), 300.0 / math.sqrt(2)), rtol=0.05
    )
    assert share[0] < 0.05 and share[1] > 0.95


def _toy_val_loader():
    from nn_motion_control.data.device_loader import RolloutWindowLoader

    inputs = torch.zeros(400, 3)
    starts = torch.arange(0, 300, 5)
    return RolloutWindowLoader(
        inputs, starts, 4, 2, batch_size=8, pos_cols=[0], dac_cols=[2]
    )


class _Proportional:
    # Gain 0.5 is well inside stability for an integrator behind the
    # plant's one-step command delay (gain 1 would be marginal).
    def __call__(self, position, velocity, reference, reference_velocity):
        return (0.5 * (reference - position)).clamp(-10.0, 10.0)


def test_preflight_degenerate_check_passes_a_working_controller(tmp_path):
    from nn_motion_control.training import preflight as pf

    t = _make_trainer(tmp_path, val_loader=_toy_val_loader())
    t.policy = _Proportional()
    config = SimpleNamespace(
        axes=["x"], safe_range=torch.tensor([[-10.0, 10.0]]), seed=0
    )
    measured = SimpleNamespace(gain=np.ones(1))
    check = pf.check_degenerate(t, config, None, measured)
    assert check.status == pf.PASS


def test_preflight_degenerate_check_fails_a_loss_that_rewards_freezing(
    tmp_path,
):
    from nn_motion_control.training import preflight as pf

    # Tracking barely counts and any command change is expensive: a frozen
    # controller then scores as well as a working one.
    t = _make_trainer(
        tmp_path,
        val_loader=_toy_val_loader(),
        tracking_weight=1e-9,
        rate_weight=1.0,
    )
    t.policy = _Proportional()
    config = SimpleNamespace(
        axes=["x"], safe_range=torch.tensor([[-10.0, 10.0]]), seed=0
    )
    measured = SimpleNamespace(gain=np.ones(1))
    check = pf.check_degenerate(t, config, None, measured)
    assert check.status == pf.FAIL


def test_a_command_acts_a_step_after_the_position_it_was_decided_from():
    # Rows pair each position with the command issued a step earlier, so
    # the first position after the seed follows the seed's own command.
    plant = _dac_plant(1.0)

    class Constant:
        def __call__(self, position, velocity, reference, reference_velocity):
            return torch.full_like(position, 5.0)

    warmup = torch.zeros(1, 3, 4)
    ref = torch.zeros(1, 3, 1)
    with torch.no_grad():
        pos, dacs = plant.closed_loop_rollout(warmup, ref, Constant(), 3)
    assert pos[0, :, 0].tolist() == [0.0, 5.0, 10.0]
    assert dacs[0, :, 0].tolist() == [5.0, 5.0, 5.0]
