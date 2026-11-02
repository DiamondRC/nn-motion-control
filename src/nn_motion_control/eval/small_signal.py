"""
The plant's response to small commands, against the measured stage.

Drives the plant open-loop with band-limited white command noise at a few
amplitudes from near-rest seeds and compares its gain, band by band, with
the dataset's measured FRF. A precision stage holds by dithering a count or
so of command, so a plant that is accurate only for large excitation is
wrong exactly where nm control lives; this is the gate for that.
"""

from __future__ import annotations

import logging
import os

import numpy as np
import torch

from nn_motion_control.control.disturbance import _welch
from nn_motion_control.control.workspace import MeasuredStage
from nn_motion_control.core.config import RunConfiguration
from nn_motion_control.core.precision import strict_float32
from nn_motion_control.data.device_loader import RolloutWindowLoader
from nn_motion_control.plant.plant import (
    Plant,
    RolloutLayout,
    rollout_splits_from_config,
)

logger = logging.getLogger(__name__)

AMPLITUDES = (0.3, 1.0, 3.0, 10.0)  # command rms, counts
BANDS_HZ = ((1.0, 5.0), (5.0, 15.0), (15.0, 50.0), (50.0, 200.0))
DRIVE_MAX_HZ = 200.0
SECONDS = 8.0
SETTLE_SECONDS = 1.0
ROWS = 8
SEED = 7
# Seeds within this distance of rest (position unit).
NEAR_REST = 2000.0
# The gate: gain within TOLERANCE of the measured stage at these amplitudes.
GATED = (1.0, 3.0)
TOLERANCE = 0.1


class _Play:
    def __init__(self, drive: torch.Tensor):
        self.drive = drive
        self.k = 0
        self.held: torch.Tensor | None = None

    def reset(self, batch: int, device) -> None:
        self.k = 0

    def prime(self, command: torch.Tensor) -> None:
        self.held = command

    def __call__(self, position, velocity, reference, reference_velocity):
        base = self.held if self.held is not None else position * 0.0
        out = base + self.drive[:, self.k]
        self.k += 1

        return out


def small_signal_gains(
    plant: Plant, plant_cfg: RunConfiguration, device: str
) -> dict[float, np.ndarray]:
    """
    Plant over measured gain [A, bands] for each drive amplitude.
    """

    rate = float(plant_cfg.system.data_rate_hz or 0.0)
    if rate <= 0:
        raise ValueError("The small-signal test needs the system's data rate")
    axes = tuple(plant_cfg.system.axes)
    stage = MeasuredStage.from_dataset(plant_cfg.datafile_dir, axes)
    val = rollout_splits_from_config(
        plant_cfg,
        RolloutLayout.from_config(plant_cfg),
        max_horizon=1,
        batch_size=ROWS,
        device=device,
        quiescent_seed={"max_dac_ripple": 1.0, "max_speed": 0.1},
    ).val_loader
    if not isinstance(val, RolloutWindowLoader):
        raise ValueError("The small-signal test needs rollout seeds")
    pool = val.spread_warmups(64 * ROWS)
    origin, _ = plant.seed_state(pool)
    near = (origin.abs().amax(dim=1) < NEAR_REST).nonzero().flatten()[:ROWS]
    if near.numel() == 0:
        raise ValueError("No validation seeds near rest")
    warmup = pool[near]
    steps = int(SECONDS * rate)
    settle = int(SETTLE_SECONDS * rate)
    hold = origin[near][:, None, :].expand(-1, steps, -1).contiguous()
    out = {}

    for amp in AMPLITUDES:
        gen = torch.Generator(device=device).manual_seed(SEED)
        noise = torch.randn(
            near.numel(), steps, len(axes), generator=gen, device=device
        )
        spec = torch.fft.rfft(noise, dim=1)
        freqs = torch.fft.rfftfreq(steps, 1.0 / rate, device=device)
        spec[:, freqs > DRIVE_MAX_HZ] = 0
        drive = torch.fft.irfft(spec, n=steps, dim=1)
        drive = amp * drive / drive.std(dim=1, keepdim=True)
        with torch.no_grad():
            pos, _ = plant.closed_loop_rollout(
                warmup, hold, _Play(drive), steps
            )
        y = pos[:, settle:].double().cpu().numpy()
        x = drive[:, settle:].double().cpu().numpy()
        gains = np.zeros((len(axes), len(BANDS_HZ)))

        for ax in range(len(axes)):
            f, py = _welch(y[:, :, ax], rate)
            _, px = _welch(x[:, :, ax], rate)
            measured = np.interp(
                f, stage.freqs, np.abs(stage.response[ax]), left=stage.gain[ax]
            )

            for j, (lo, hi) in enumerate(BANDS_HZ):
                m = (f >= lo) & (f < hi)
                plant_gain = np.sqrt(py[:, m].mean(0) / px[:, m].mean(0))
                gains[ax, j] = float(np.mean(plant_gain / measured[m]))
        out[amp] = gains

    return out


def run_small_signal(plant_config: str, out_dir: str | None = None) -> bool:
    """
    Measure a plant's small-signal gains, write small_signal.md, and return
    whether it passes the gate.
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    strict_float32()
    cfg = RunConfiguration(plant_config)
    plant = Plant.from_checkpoint(cfg, cfg.m_save_dir, device)
    gains = small_signal_gains(plant, cfg, device)
    axes = list(cfg.system.axes)
    bands = [f"{lo:g}-{hi:g} Hz" for lo, hi in BANDS_HZ]
    lines = [
        f"# Small-signal gain: {cfg.model_name}",
        "",
        "Plant gain over the measured FRF, white command noise below "
        f"{DRIVE_MAX_HZ:g} Hz from rest. Gate: within {100 * TOLERANCE:g} % "
        f"at {', '.join(f'{a:g}' for a in GATED)} counts.",
        "",
        "| drive (counts rms) | axis | " + " | ".join(bands) + " |",
        "|---|---|" + "---|" * len(bands),
    ]
    passed = True

    for amp, g in gains.items():
        for ax, name in enumerate(axes):
            lines.append(
                f"| {amp:g} | {name} | "
                + " | ".join(f"{v:.2f}" for v in g[ax])
                + " |"
            )
        if amp in GATED and np.abs(g - 1.0).max() > TOLERANCE:
            passed = False
    lines += ["", f"Gate: {'PASS' if passed else 'FAIL'}"]
    out = out_dir or cfg.eval_dir
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "small_signal.md")
    with open(path, "w") as f:
        f.write("\n".join(lines))

    for line in lines:
        logger.info("%s", line)

    return passed
