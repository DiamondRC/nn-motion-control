"""
The PandA PID policy and the controller suite's scoring.
"""

import json
import math

import numpy as np
import torch

from nn_motion_control.control.pid import PandaPID, PidGains
from nn_motion_control.eval.controller_suite import score_rows, summarise

GAINS = {
    "KP_I": 0.004,
    "KV_I": 0.9,
    "KI_I": 0.0002,
    "KD_I": 0.0,
    "KVFF_I": 3.7,
    "KAFF_I": 100.0,
    "KPFF0_I": 0.0,
    "KPFF1_I": 0.00196,
    "K_TOT_I": 1.0,
    "DT_I": 1.0,
    "DT_INV_I": 1.0,
    "MAX_OUTPUT_I": 3932.0,
    "MAX_INTEGRAL_I": 28000.0,
}


def _pid(tmp_path, **override):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"pid": {"gains": {**GAINS, **override}}}))

    return PandaPID(PidGains.from_manifest(path), torch.ones(1))


def _step(pid, setpoint, position):
    def t(v):
        return torch.tensor([[float(v)]])

    return float(pid(t(position), t(0.0), t(setpoint), t(0.0)))


def test_pid_terms_follow_the_block(tmp_path):
    pid = _pid(tmp_path)
    pid.reset(1, "cpu")
    # First tick: no motion yet, so P plus the position feedforward.
    assert _step(pid, 1000.0, 900.0) == round(0.004 * 100 + 0.00196 * 1000)
    # The stage moves 10 nm with the setpoint still: velocity feedback.
    out = _step(pid, 1000.0, 910.0)
    assert out == round(0.004 * 90 - 0.9 * 10 + 0.00196 * 1000)


def test_pid_integrates_only_while_the_setpoint_moves(tmp_path):
    pid = _pid(tmp_path, KP_I=0.0, KPFF1_I=0.0, KV_I=0.0, KVFF_I=0.0)
    pid.reset(1, "cpu")
    _step(pid, 0.0, 0.0)
    held = out = float("nan")

    for _ in range(10):
        held = _step(pid, 0.0, -1e4)  # a held setpoint: frozen integral
    assert held == 0.0
    total = 0.0

    for k in range(1, 11):
        # A moving setpoint integrates the error (the acceleration
        # feedforward only sees the first step).
        out = _step(pid, float(k), -1e4)
        total += 0.0002 * (k + 1e4)
    assert math.isclose(out, round(total), abs_tol=1.0)


def test_pid_takes_over_bumplessly(tmp_path):
    pid = _pid(tmp_path)
    pid.reset(1, "cpu")
    pid.prime(torch.tensor([[-459.0]]))
    assert _step(pid, -5e5, -5e5 + 20.0) == -459.0
    # The next held tick keeps it: the integral absorbed the difference.
    assert abs(_step(pid, -5e5, -5e5 + 20.0) + 459.0) <= 1.0


def test_scoring_flags_divergence_and_thrash():
    rate, h = 10_000.0, 1000
    t = torch.arange(h) / rate
    ref = torch.zeros(3, h, 1)
    pos = torch.zeros(3, h, 1)
    pos[1, -50:] = 5000.0  # runs away at the end
    cmd = torch.zeros(3, h, 1)
    cmd[2, :, 0] = 300.0 * torch.sin(2 * math.pi * 800.0 * t)  # thrashes
    safe = torch.tensor([[-1000.0, 1000.0]])
    rows = score_rows(pos, cmd, ref, safe, rate)
    assert rows["lost"].tolist() == [False, True, False]
    assert rows["thrash"][2] > 0.95
    assert np.isnan(rows["thrash"][0])  # a still command has no spectrum
    assert np.isclose(rows["command_std"][2], 300.0 / math.sqrt(2), rtol=0.05)
    s = summarise(rows)
    assert s["lost"] == 1 and s["rows"] == 3


def test_a_large_reference_tracked_loosely_is_not_lost():
    rate, h = 10_000.0, 1000
    ref = torch.linspace(0.0, 1e6, h)[None, :, None]  # a 1 mm move
    pos = ref - 5e4  # 50 um behind, steadily
    cmd = torch.zeros_like(ref)
    rows = score_rows(pos, cmd, ref, torch.tensor([[-1e3, 1e3]]), rate)
    assert not rows["lost"][0]
    assert np.isclose(rows["offset"][0], 5e4) and rows["fluctuation"][0] < 1.0
