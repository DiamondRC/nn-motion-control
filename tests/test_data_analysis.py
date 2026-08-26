"""Data analysis: loader units, fit metrics, summaries, end-to-end report."""

import os
from pathlib import Path

import numpy as np

from nn_motion_control.core.system import SystemSpec
from nn_motion_control.data_analysis.analyse import run_analysis
from nn_motion_control.data_analysis.envelope import envelope_bounds
from nn_motion_control.data_analysis.loader import load_channels
from nn_motion_control.data_analysis.nonlinearity import (
    linear_fit,
    static_nonlinearity,
)
from nn_motion_control.data_analysis.stats import channel_summary

# Position column indices in the synthetic input layout (x_pos, y_pos, z_pos).
POS_COLS = (1, 5, 9)
SYSTEM_PATH = (
    Path(__file__).resolve().parents[1] / "examples/deltabot/system.toml"
)


def _system() -> SystemSpec:
    return SystemSpec.from_toml(SYSTEM_PATH)


def test_linear_fit_recovers_line():
    x = np.linspace(-5.0, 5.0, 200)
    y = 2.0 * x + 3.0
    fit = linear_fit(x, y)
    assert abs(fit["slope"] - 2.0) < 1e-6
    assert abs(fit["intercept"] - 3.0) < 1e-6
    assert fit["r2"] > 0.999999
    assert fit["resid_rms"] < 1e-6
    assert fit["max_dev_pct"] < 1e-6


def test_linear_fit_noise_has_low_r2_but_finite():
    rng = np.random.default_rng(0)
    x = rng.normal(size=500)
    y = rng.normal(size=500)  # unrelated
    fit = linear_fit(x, y)
    assert fit["r2"] < 0.2
    assert np.isfinite(fit["resid_rms"])
    assert np.isfinite(fit["max_dev_pct"])


def test_static_nonlinearity_flat_on_linear_mean():
    # A linear mean response with heavy noise: raw R2 is low, but the binned
    # conditional mean is straight, so the static bend is ~0.
    rng = np.random.default_rng(1)
    x = rng.uniform(-100.0, 100.0, 500000)
    y = 3.0 * x + rng.normal(scale=50.0, size=x.size)
    s = static_nonlinearity(x, y)
    assert s["r2"] > 0.999
    assert s["bend_pct"] < 2.0


def test_static_nonlinearity_detects_curvature():
    x = np.linspace(-5.0, 5.0, 20000)
    y = x**2  # curved mean response
    s = static_nonlinearity(x, y)
    assert s["r2"] < 0.5
    assert s["bend_pct"] > 10.0


def test_channel_summary_matches_numpy():
    v = np.array([-3.0, -1.0, 0.0, 2.0, 4.0])
    lo, hi, mean, std, p95, _p99 = channel_summary(v)
    assert lo == -3.0
    assert hi == 4.0
    assert abs(mean - v.mean()) < 1e-12
    assert abs(std - v.std()) < 1e-12
    assert p95 == np.percentile(np.abs(v), 95.0)


def test_load_channels_units_and_shapes(synth_h5):
    data = load_channels(_system(), synth_h5["path"])
    assert data.axes == ["x", "y", "z"]
    for name in ("position", "velocity", "acceleration", "jerk", "dac"):
        assert name in data.values
        assert data.values[name].shape == (synth_h5["inputs"].shape[0], 3)
    # Position stores nm already, so the factor is identity: values equal the
    # raw columns.
    expected = synth_h5["inputs"][:, list(POS_COLS)].astype(np.float64)
    assert np.allclose(data.values["position"], expected)
    assert data.units["position"] == "nm"
    assert data.units["velocity"] == "nm/step"
    assert data.units["dac"] == "V"


def test_envelope_bounds_are_per_axis_min_max(synth_h5):
    data = load_channels(_system(), synth_h5["path"])
    bounds = envelope_bounds(data)
    raw = synth_h5["inputs"][:, list(POS_COLS)]
    assert np.allclose(bounds["min"], raw.min(axis=0))
    assert np.allclose(bounds["max"], raw.max(axis=0))
    assert np.allclose(bounds["span"], bounds["max"] - bounds["min"])


def test_run_analysis_writes_report_and_plots(synth_h5, tmp_path):
    out_dir = str(tmp_path / "analysis")
    path = run_analysis(_system(), synth_h5["path"], out_dir=out_dir)
    assert os.path.dirname(path) == out_dir
    assert os.path.basename(path) == "report.md"
    assert os.path.getsize(path) > 0
    for name in (
        "envelope.svg",
        "nonlinearity.svg",
        "hist_position.svg",
        "hist_dac.svg",
    ):
        assert os.path.exists(os.path.join(out_dir, name))
    # Per-axis full-size figures are written by default.
    assert os.path.exists(
        os.path.join(out_dir, "per_axis", "hist_position_x.svg")
    )
