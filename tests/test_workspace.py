"""
Workspace references: stage fit, travel and command limits, seeds.
"""

import math

import h5py
import numpy as np
import pytest
import torch

from nn_motion_control.control.workspace import (
    COMMAND_FRACTION,
    FAMILIES,
    StageModel,
    WorkspaceLimits,
    WorkspaceReference,
    max_amplitude,
)

RATE = 10_000.0
GAIN = (1000.0, 900.0, 950.0)
NATURAL = (12.0, 13.0, 12.5)
DAMPING = (0.1, 0.09, 0.095)
BOX = 1.0e6
SAFE = 1500.0


def _write_frf(path):
    freqs = np.arange(5.0, 400.0, 1.0)
    with h5py.File(path, "w") as f:
        f.attrs["static_gain_nm_per_count"] = np.array(GAIN)

        for i, axis in enumerate("xyz"):
            s = freqs / NATURAL[i]
            h = np.zeros((3, len(freqs)), complex)
            h[i] = GAIN[i] / (1.0 - s * s + 2j * DAMPING[i] * s)
            f[f"frf/{axis}/freqs"] = freqs
            f[f"frf/{axis}/H"] = h


def _stage():
    return StageModel(
        torch.tensor(GAIN), torch.tensor(NATURAL), torch.tensor(DAMPING)
    )


def _limits(stage=None):
    box = torch.full((3,), BOX)
    safe = torch.full((3,), SAFE)

    return WorkspaceLimits(-box, box, -safe, safe, stage or _stage(), RATE)


def test_stage_fit_recovers_a_known_resonance(tmp_path):
    path = tmp_path / "frf.h5"
    _write_frf(path)
    stage = StageModel.from_dataset(str(path), ("x", "y", "z"))
    assert torch.allclose(stage.natural_hz, torch.tensor(NATURAL), rtol=0.02)
    assert torch.allclose(stage.damping, torch.tensor(DAMPING), rtol=0.05)


def test_stage_command_holds_a_static_offset_and_drives_a_sine():
    stage = _stage()
    t = torch.arange(20_000, dtype=torch.float64) / RATE
    hold = torch.full((1, len(t), 3), 5000.0, dtype=torch.float64)
    gain = torch.tensor(GAIN, dtype=torch.float64)
    assert torch.allclose(stage.command(hold, RATE), hold / gain, rtol=1e-5)
    # At resonance the command is 2 * zeta of the static one, 90 deg ahead.
    f0 = NATURAL[0]
    pos = torch.zeros(1, len(t), 3, dtype=torch.float64)
    pos[0, :, 0] = torch.sin(2 * math.pi * f0 * t)
    cmd = stage.command(pos, RATE)[0, 1000:-1000, 0]
    expected = 2 * DAMPING[0] / GAIN[0]
    assert math.isclose(float(cmd.abs().max()), expected, rel_tol=0.01)


def test_max_amplitude_is_the_tightest_side():
    base = torch.tensor([[0.0, 5.0]])
    coeff = torch.tensor([[[1.0, -1.0], [2.0, 0.0]]])
    lo, hi = torch.tensor([-10.0, -10.0]), torch.tensor([10.0, 10.0])
    # Axis 0 reaches 2 * a <= 10, axis 1 goes down 5 + 10 = 15.
    assert max_amplitude(base, coeff, lo, hi).item() == 5.0


def _draw(gen_seed=0, b=512, h=2000, spec=None, origin=None, hold=None):
    ref = WorkspaceReference(spec or {}, _limits())
    if origin is None:
        g = torch.Generator().manual_seed(1)
        origin = (torch.rand(b, 3, generator=g) * 2 - 1) * 0.5 * BOX
    gen = torch.Generator().manual_seed(gen_seed)
    pos, vel = ref(origin, h, gen, hold=hold)

    return ref, origin, pos, vel


def test_references_start_on_the_seed_inside_box_and_budget():
    ref, origin, pos, vel = _draw()
    assert torch.allclose(pos[:, 0], origin)
    assert bool((pos.abs() <= BOX + 1e-3).all())
    # The motion uses at most its share of each seed's command headroom.
    cmd = _stage().command(pos.double(), RATE)
    hold = (origin / torch.tensor(GAIN)).double()[:, None, :]
    up = hold + COMMAND_FRACTION * (SAFE - hold)
    down = hold - COMMAND_FRACTION * (SAFE + hold)
    assert bool((cmd <= up + 1e-3).all() and (cmd >= down - 1e-3).all())
    assert torch.allclose(vel, torch.gradient(pos, dim=1)[0], atol=0.25)
    # Every family is drawn.
    assert ref.last_families is not None
    assert set(ref.last_families.tolist()) == set(range(len(FAMILIES)))


def test_references_are_reproducible_from_the_generator():
    _, _, a, _ = _draw(gen_seed=3)
    _, _, b, _ = _draw(gen_seed=3)
    _, _, c, _ = _draw(gen_seed=4)
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


def test_amplitude_curriculum_widens_the_ceiling():
    spec = {"amplitude_curriculum": {"start": 100.0, "epochs": 4}}
    ref = WorkspaceReference(spec, _limits())
    origin = torch.zeros(256, 3)
    peaks = []

    for epoch in (0, 2, None):
        ref.set_epoch(epoch)
        pos, _ = ref(origin, 500, torch.Generator().manual_seed(0))
        peaks.append(float(pos.abs().max()))
    assert peaks[0] <= 100.0 + 1e-3
    assert peaks[0] < peaks[1] < peaks[2]


def test_a_seed_held_near_its_limit_leaves_little_command_headroom():
    origin = torch.zeros(256, 3)
    near_rail = torch.full((256, 3), COMMAND_FRACTION * SAFE * 0.999)
    _, _, free, _ = _draw(origin=origin, hold=torch.zeros(256, 3))
    _, _, held, _ = _draw(origin=origin, hold=near_rail)
    assert held.abs().max() < free.abs().max()


def test_bad_specs_are_refused():
    with pytest.raises(ValueError, match="Unknown workspace families"):
        WorkspaceReference({"weights": {"zigzag": 1.0}}, _limits())
    with pytest.raises(ValueError, match="amplitude"):
        WorkspaceReference({"amplitude": [10.0, 1.0]}, _limits())


def test_references_leave_rest_smoothly_within_budget():
    # From a seed at rest, the start-up acceleration counts too: the
    # command with rest samples before step 0 stays inside the budget.
    origin = torch.zeros(512, 3)
    _, _, pos, vel = _draw(origin=origin, hold=torch.zeros(512, 3))
    # The first step's one-sided difference across the minimum-jerk ramp.
    assert float(vel[:, 0].abs().max()) < 0.1
    rest = torch.zeros(512, 8, 3, dtype=torch.float64)
    path = torch.cat([rest, pos.double()], 1)
    cmd = _stage().command(path, RATE)
    assert float(cmd.abs().max()) <= COMMAND_FRACTION * SAFE * 1.01


def test_box_is_cut_to_where_the_stage_can_be_held(tmp_path):
    from types import SimpleNamespace

    from nn_motion_control.control.workspace import HOLD_FRACTION
    from nn_motion_control.training.control_run import reference_limits

    path = tmp_path / "frf.h5"
    _write_frf(path)
    stops = {"x": [-2.0e6, 2.0e6], "y": [-2.0e6, 0.5e6], "z": [-2.0e6, 2.0e6]}
    position = SimpleNamespace(limits=stops)
    config = SimpleNamespace(
        system=SimpleNamespace(channel=lambda name: position),
        axes=["x", "y", "z"],
        safe_range=torch.tensor([[-1000.0, 1000.0]] * 3),
        servo_rate_hz=RATE,
    )
    plant_config = SimpleNamespace(datafile_dir=str(path))
    spec = {"kind": "workspace", "margin": 1.0e5}
    lim = reference_limits(config, plant_config, spec)
    assert lim is not None
    held = HOLD_FRACTION * torch.tensor(GAIN) * 1000.0
    # The command binds everywhere except y's near stop, where it does not.
    assert torch.allclose(
        lim.pos_hi, torch.minimum(held, torch.tensor([1.9e6, 0.4e6, 1.9e6]))
    )
    assert torch.allclose(lim.pos_lo, -held)
    assert reference_limits(config, plant_config, {"kind": "mixed"}) is None


def test_command_is_corrected_where_the_stage_responds_less(tmp_path):
    # A stage that responds half as much as its mode above 100 Hz needs
    # twice the dynamic command there, and the mode's own below.
    path = tmp_path / "frf.h5"
    freqs = np.arange(5.0, 400.0, 1.0)
    with h5py.File(path, "w") as f:
        f.attrs["static_gain_nm_per_count"] = np.array(GAIN)

        for i, axis in enumerate("xyz"):
            s = freqs / NATURAL[i]
            h = np.zeros((3, len(freqs)), complex)
            h[i] = GAIN[i] / (1.0 - s * s + 2j * DAMPING[i] * s)
            h[i, freqs > 100.0] /= 2.0
            f[f"frf/{axis}/freqs"] = freqs
            f[f"frf/{axis}/H"] = h
    stage = StageModel.from_dataset(str(path), ("x", "y", "z"))
    scale = stage.dynamic_scale(torch.tensor([2.0, 50.0, 150.0, 300.0]))
    assert torch.allclose(scale[:2], torch.ones(2, 3), atol=0.05)
    assert torch.allclose(scale[2:], torch.full((2, 3), 2.0), rtol=0.05)
