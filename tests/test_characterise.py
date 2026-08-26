"""
Characterisation battery: each measure is checked against synthetic signals of
known character (noise, sine, ramp, chaotic map, coupled pair, ...), asserting
it discriminates in the expected direction.
"""

import os
from pathlib import Path

import numpy as np

from nn_motion_control.core.system import SystemSpec
from nn_motion_control.data_analysis import (
    dimension,
    dynamics,
    entropy,
    information,
    physics,
    regime,
    spectral,
)
from nn_motion_control.data_analysis.characterise import run_characterisation

SYSTEM_PATH = (
    Path(__file__).resolve().parents[1] / "examples/deltabot/system.toml"
)

SR = 1000.0
N = 20000
RNG = np.random.default_rng(0)
T = np.arange(N) / SR


def _white():
    return RNG.standard_normal(N)


def _sine(f=50.0):
    return np.sin(2 * np.pi * f * T)


def _ramp():
    return np.linspace(0.0, 1.0, N)


def _ar1(phi=0.95):
    e = RNG.standard_normal(N)
    x = np.zeros(N)

    for i in range(1, N):
        x[i] = phi * x[i - 1] + e[i]

    return x


def _logistic(r=4.0):
    x = np.zeros(N)
    x[0] = 0.37

    for i in range(1, N):
        x[i] = r * x[i - 1] * (1.0 - x[i - 1])

    return x


# --- spectral ---


def test_spectral_entropy_noise_gt_sine():
    assert spectral.spectral_entropy(_white(), SR) > spectral.spectral_entropy(
        _sine(), SR
    )


def test_one_over_f_slope_red_more_negative_than_white():
    assert spectral.one_over_f_slope(_ar1(), SR) < spectral.one_over_f_slope(
        _white(), SR
    )


def test_resonance_peaks_finds_tone():
    peaks = spectral.resonance_peaks(_sine(50.0), SR)
    assert peaks
    assert min(abs(f - 50.0) for f, _ in peaks) < 5.0


def test_frequency_response_recovers_gain_and_coherence():
    x = _white()
    y = 2.0 * x + 0.01 * RNG.standard_normal(N)
    freqs, gain, _, coh = spectral.frequency_response(x, y, SR)
    mid = (freqs > 50) & (freqs < 400)
    assert abs(np.median(gain[mid]) - 2.0) < 0.1
    assert np.median(coh[mid]) > 0.9


def test_mean_coherence_high_for_coupled_low_for_independent():
    x = _white()
    y = x + 0.05 * RNG.standard_normal(N)
    assert spectral.mean_coherence(x, y, SR) > spectral.mean_coherence(
        x, _white(), SR
    )


def test_noise_floor_snr_sine_gt_noise():
    assert (
        spectral.noise_floor(_sine(), SR)["snr"]
        > spectral.noise_floor(_white(), SR)["snr"]
    )


# --- entropy ---


def test_permutation_entropy_orders_signals():
    assert entropy.permutation_entropy(_white()) > entropy.permutation_entropy(
        _sine()
    )
    assert entropy.permutation_entropy(_ramp()) < 0.1


def test_sample_entropy_noise_gt_sine():
    assert entropy.sample_entropy(_white()) > entropy.sample_entropy(_sine())


def test_lempel_ziv_noise_gt_periodic():
    assert entropy.lempel_ziv(_white()) > entropy.lempel_ziv(_sine())
    assert entropy.lempel_ziv(_white()) > 0.8


# --- dimension ---


def test_pca_dimension_line_vs_isotropic():
    line = np.stack([T, 2 * T, 3 * T], axis=1) + 1e-6 * RNG.standard_normal(
        (N, 3)
    )
    iso = RNG.standard_normal((N, 3))
    assert dimension.pca_dimension(line)["participation_ratio"] < 1.5
    assert dimension.pca_dimension(iso)["participation_ratio"] > 2.5


def test_correlation_dimension_plane_vs_line():
    plane = RNG.uniform(0, 1, (3000, 2))
    line = np.stack([np.linspace(0, 1, 3000)] * 2, axis=1)
    assert 1.5 < dimension.correlation_dimension(plane)["d2"] < 2.5
    assert dimension.correlation_dimension(line)["d2"] < 1.3


def test_higuchi_noise_gt_smooth():
    hn = dimension.higuchi_fd(_white())
    hr = dimension.higuchi_fd(_ramp())
    assert hn > 1.5
    assert hr < 1.4
    assert hn > hr


# --- dynamics ---


def test_lyapunov_positive_for_chaos_small_for_periodic():
    chaos = dynamics.largest_lyapunov(_logistic(), tau=1, m=4, sample_rate=1.0)
    periodic = dynamics.largest_lyapunov(
        _sine(), tau=dimension.estimate_tau(_sine()), m=4, sample_rate=1.0
    )
    assert chaos > 0.2
    assert periodic < 0.15


def test_dfa_hurst_white_near_half():
    assert 0.35 < dynamics.dfa_hurst(_white()) < 0.65


def test_rqa_determinism_sine_gt_noise():
    det_sine = dynamics.rqa(dimension.delay_embed(_sine(), tau=5, m=3))[
        "determinism"
    ]
    det_noise = dynamics.rqa(dimension.delay_embed(_white(), tau=1, m=3))[
        "determinism"
    ]
    assert det_sine > det_noise


def test_increment_kurtosis_gaussian_vs_heavy_tailed():
    gaussian = np.cumsum(RNG.standard_normal(N))
    heavy = np.cumsum(RNG.standard_t(3, N))
    assert abs(dynamics.increment_stats(gaussian)["kurtosis"]) < 0.5
    assert dynamics.increment_stats(heavy)["kurtosis"] > 2.0


# --- information ---


def test_mutual_information_self_vs_independent():
    x = _white()
    assert information.mutual_information(
        x, x
    ) > information.mutual_information(x, _white())


def test_transfer_entropy_directional():
    s = _white()
    t = np.zeros(N)

    for i in range(1, N):
        t[i] = 0.5 * t[i - 1] + 0.5 * s[i - 1]
    assert information.transfer_entropy(s, t) > information.transfer_entropy(
        t, s
    )


# --- physics ---


def test_kinematic_consistency_perfect_derivative():
    pos = _sine()
    vel = np.concatenate([[0.0], np.diff(pos)]) * SR
    acc = np.concatenate([[0.0], np.diff(vel)]) * SR
    out = physics.kinematic_consistency(pos, vel, acc, dt=1.0 / SR)
    assert out["vel_r2"] > 0.99


def test_eom_residual_recovers_coefficients():
    a = RNG.standard_normal(N)
    v = RNG.standard_normal(N)
    x = RNG.standard_normal(N)
    dac = 2.0 * a + 0.5 * v + 3.0 * x
    out = physics.eom_residual(x, v, a, dac)
    assert abs(out["m"] - 2.0) < 1e-6
    assert abs(out["c"] - 0.5) < 1e-6
    assert abs(out["k"] - 3.0) < 1e-6
    assert out["r2"] > 0.999


def test_hysteresis_loop_vs_line():
    theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
    loop = physics.hysteresis(np.cos(theta), np.sin(theta))["normalized"]
    line = physics.hysteresis(_ramp(), _ramp())["normalized"]
    assert loop > 0.5
    assert line < 0.05


def test_causal_lag_recovers_delay():
    dac = _white()
    lag = 7
    pos = np.concatenate([np.zeros(lag), dac[:-lag]])
    out = physics.causal_lag(dac, pos, SR, max_lag=50)
    assert out["lag_steps"] == lag


# --- regime ---


def test_stationarity_flags_drift():
    offsets = np.array([0, N])
    stationary = regime.stationarity(_white(), offsets)["mean_drift"]
    drifting = regime.stationarity(_white() + 5 * T, offsets)["mean_drift"]
    assert drifting > stationary


def test_spatial_homogeneity_detects_position_dependence():
    pos = np.linspace(-1.0, 1.0, N)
    homogeneous = RNG.standard_normal(N)
    growing = RNG.standard_normal(N) * (1.0 + 5.0 * np.abs(pos))
    assert (
        regime.spatial_homogeneity(pos, growing)["cv"]
        > regime.spatial_homogeneity(pos, homogeneous)["cv"]
    )


# --- end to end ---


def test_run_characterisation_writes_report(synth_h5, tmp_path):
    system = SystemSpec.from_toml(SYSTEM_PATH)
    out = str(tmp_path / "char")
    path = run_characterisation(system, synth_h5["path"], out_dir=out)
    assert os.path.basename(path) == "report.md"
    assert os.path.getsize(path) > 0
    for name in ("psd.svg", "bode.svg", "d2_saturation.svg"):
        assert os.path.exists(os.path.join(out, name))


def test_plant_sources_shapes_and_regimes(config_factory):
    import torch

    from nn_motion_control.core.config import RunConfiguration
    from nn_motion_control.data.normalize import NormStats
    from nn_motion_control.data_analysis.compare import plant_sources
    from nn_motion_control.models.builder import JsonModel
    from nn_motion_control.plant.plant import Plant, RolloutLayout

    layers = [
        {"TemporalConv": [9, 8, 3, 1, 1, 0.0]},
        {"AdaptiveAvgPool1d": [1]},
        {"Flatten": [1]},
        {"Linear": [8, 3]},
    ]
    path = config_factory(
        window_size=8,
        hidden_layers=layers,
        inputs=["position", "velocity", "dac"],
        targets={"position": {"form": "delta", "weight": 1}},
    )
    cfg = RunConfiguration(path)
    plant = Plant(
        JsonModel(config=cfg).eval(),
        NormStats(torch.zeros(9), torch.ones(9), torch.ones(9).bool()),
        NormStats(torch.zeros(3), torch.ones(3), torch.ones(3).bool()),
        RolloutLayout.from_config(cfg),
        device="cpu",
    )
    h = 200
    warmup = torch.randn(1, 9, 8)
    dac_future = torch.randn(1, h, 3)
    gt = torch.randn(1, h, 3)
    sources = plant_sources(plant, warmup, dac_future, gt, h)
    assert set(sources) == {
        "data",
        "plant (teacher-forced)",
        "plant (free-run)",
    }

    for arr in sources.values():
        assert arr.shape == (h, 3)
    assert np.isfinite(sources["plant (teacher-forced)"]).all()
    # Data source is the ground truth, unchanged.
    assert np.allclose(sources["data"], gt[0].numpy())
