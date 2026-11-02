"""
Measured disturbance, sensor noise and small-signal oversampling.
"""

import numpy as np
import torch

from nn_motion_control import data as dl
from nn_motion_control.control.disturbance import (
    SEQUENCE,
    CommandDisturbance,
    DisturbanceSpectrum,
    NoisySensor,
)


def test_a_realisation_has_the_spectrum_variance():
    rate = 10_000.0
    freqs = np.linspace(0.0, rate / 2, 1001)
    psd = np.zeros((1, freqs.size))
    psd[0, (freqs > 10) & (freqs < 100)] = 2.0  # counts^2 / Hz
    spec = DisturbanceSpectrum(freqs, psd, rate, np.zeros(1))
    seq = spec.sample(4, torch.Generator().manual_seed(0), "cpu")
    assert seq.shape == (4, SEQUENCE, 1)
    # The spectrum is interpolated onto the sequence's bins, edges included.
    f = np.fft.rfftfreq(SEQUENCE, 1.0 / rate)
    expected = np.trapezoid(np.interp(f, freqs, psd[0]), f)
    assert np.isclose(float(seq.var()), expected, rtol=0.02)


def test_disturbance_and_sensor_noise_are_added():
    rate = 10_000.0
    freqs = np.linspace(0.0, rate / 2, 11)
    spec = DisturbanceSpectrum(freqs, np.ones((2, 11)), rate, np.zeros(2))
    act = CommandDisturbance(spec, generator=torch.Generator().manual_seed(1))
    act.reset(torch.zeros(3, 2))
    out = torch.stack([act.apply(torch.zeros(3, 2)) for _ in range(500)])
    assert float(out.std()) > 0.0
    sensor = NoisySensor(
        np.array([2.0, 0.5]), generator=torch.Generator().manual_seed(2)
    )
    sensor.reset(torch.zeros(4000, 2))
    seen, _ = sensor.observe(torch.zeros(4000, 2), torch.zeros(4000, 2))
    assert np.allclose(seen.std(dim=0).numpy(), [2.0, 0.5], rtol=0.1)


def test_small_signal_windows_are_oversampled():
    n, w = 3000, 4
    inputs = torch.zeros(n, 2)
    t = torch.arange(n, dtype=torch.float32)
    inputs[1000:2000, 1] = torch.sin(t[1000:2000])  # ~0.7 count std
    inputs[2000:, 1] = 50.0 * torch.sin(t[2000:])  # large excitation
    starts = np.arange(0, n - w, dtype=np.int64)
    out = dl.oversample_small_signal(
        inputs,
        starts,
        w,
        [1],
        torch.zeros(2),
        torch.ones(2),
        {"band": [0.5, 5.0], "factor": 3, "span": 200},
    )
    counts = np.bincount(out, minlength=n)
    assert counts[1500] == 3  # quiet dither: tripled
    assert counts[500] == 1 and counts[2500] == 1  # idle and large: once


def test_commands_are_rounded_to_whole_counts_with_a_live_gradient():
    from nn_motion_control.control.hardware import QuantisedCommand

    q = QuantisedCommand()
    u = torch.tensor([[0.4, -1.6, 2.5]], requires_grad=True)
    out = q.apply(u)
    assert out.tolist() == [[0.0, -2.0, 2.0]]
    out.sum().backward()
    assert u.grad is not None and torch.equal(u.grad, torch.ones_like(u))


def test_linearity_augment_scales_rows_about_their_equilibrium():
    from nn_motion_control.data.augment import LinearityAugment
    from nn_motion_control.data.device_loader import RolloutWindowLoader

    # One axis: columns pos, vel, dac, with non-trivial normalisation.
    mean = torch.tensor([10.0, 0.0, 2.0])
    std = torch.tensor([100.0, 5.0, 4.0])
    n, w, h = 200, 4, 3
    t = torch.arange(n, dtype=torch.float32)
    phys = torch.stack(
        [
            500.0 + 50 * torch.sin(t / 7),
            torch.cos(t / 7),
            3.0 + torch.sin(t / 5),
        ],
        1,
    )
    inputs = (phys - mean) / std
    loader = RolloutWindowLoader(
        inputs,
        torch.arange(0, 150, 10),
        w,
        h,
        batch_size=15,
        pos_cols=[0],
        dac_cols=[2],
    )
    gain = torch.tensor([7.0])
    aug = LinearityAugment(
        loader,
        [0],
        [1],
        [2],
        mean,
        std,
        gain,
        {"fraction": 1.0, "scale": [0.5, 0.5]},
    )
    (warm, dac_f, pos_f), (w0, d0, p0) = next(iter(aug)), next(iter(loader))
    u = d0 * std[2] + mean[2]
    u_eq = (w0[:, 2, :] * std[2] + mean[2]).mean(dim=1)
    expect_u = u_eq[:, None, None] + 0.5 * (u - u_eq[:, None, None])
    assert torch.allclose(dac_f * std[2] + mean[2], expect_u, atol=1e-4)
    x = p0 * std[0] + mean[0]
    x_eq = 7.0 * u_eq
    expect_x = x_eq[:, None, None] + 0.5 * (x - x_eq[:, None, None])
    assert torch.allclose(pos_f * std[0] + mean[0], expect_x, atol=1e-3)
    vel = warm[:, 1, :] * std[1] + mean[1]
    assert torch.allclose(
        vel, 0.5 * (w0[:, 1, :] * std[1] + mean[1]), atol=1e-5
    )
    none = LinearityAugment(
        loader,
        [0],
        [1],
        [2],
        mean,
        std,
        gain,
        {"fraction": 1e-9, "scale": [0.5, 0.6]},
    )
    assert torch.equal(next(iter(none))[0], w0)


def test_weighted_linearity_augment_restores_each_rows_weight():
    from nn_motion_control.data.augment import LinearityAugment
    from nn_motion_control.data.device_loader import RolloutWindowLoader

    inputs = torch.randn(200, 3)
    loader = RolloutWindowLoader(
        inputs,
        torch.arange(0, 150, 10),
        4,
        3,
        batch_size=15,
        pos_cols=[0],
        dac_cols=[2],
    )
    args = (loader, [0], [1], [2], torch.zeros(3), torch.ones(3), torch.ones(1))
    weighted = LinearityAugment(
        *args, {"fraction": 1.0, "scale": [0.5, 0.5], "weighted": True}
    )
    batch = next(iter(weighted))
    assert len(batch) == 4 and torch.allclose(batch[3], torch.full((15,), 4.0))
    plain = LinearityAugment(*args, {"fraction": 1.0, "scale": [0.5, 0.5]})
    assert len(next(iter(plain))) == 3
