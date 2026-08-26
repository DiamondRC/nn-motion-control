"""
Physical-law and input-output consistency.

Kinematic consistency (are the derived rates the derivatives of position?),
equation-of-motion residual (fit m*a+c*v+k*x = DAC), energy/passivity, the
DAC->position hysteresis loop, and the command-to-response causal lag. Pure
numpy; each takes plain arrays so it reruns on model output.
"""

from __future__ import annotations

import numpy as np

CAUSAL_MAX_LAG = 200
_EPS = 1e-12


def _r2(residual: np.ndarray, target: np.ndarray) -> float:
    ss_tot = float(np.sum((target - target.mean()) ** 2))
    if ss_tot <= _EPS:
        return 0.0

    return float(1.0 - np.sum(residual**2) / ss_tot)


def kinematic_consistency(
    pos: np.ndarray, vel: np.ndarray, acc: np.ndarray, dt: float
) -> dict:
    """
    How well the stored rates match finite differences of position.

    Compares velocity to d(pos)/dt and acceleration to d(vel)/dt: residual RMS
    (in the rate's units) and R2. Near-perfect on the ingested data (a sanity
    check); a real test of a model that predicts the rates jointly.
    """

    d_pos = np.diff(pos) / dt
    v_res = vel[1:] - d_pos
    d_vel = np.diff(vel) / dt
    a_res = acc[1:] - d_vel

    return {
        "vel_residual_rms": float(np.sqrt(np.mean(v_res**2))),
        "vel_r2": _r2(v_res, vel[1:]),
        "acc_residual_rms": float(np.sqrt(np.mean(a_res**2))),
        "acc_r2": _r2(a_res, acc[1:]),
    }


def eom_residual(
    pos: np.ndarray, vel: np.ndarray, acc: np.ndarray, dac: np.ndarray
) -> dict:
    """
    Linear equation-of-motion fit m*a + c*v + k*x = DAC.

    Least-squares (m, c, k), the fit R2 and residual RMS. R2 near 1 means the
    motion obeys a linear 2nd-order law; the residual is the unmodelled
    physics. Coefficients absorb the unit scaling between DAC and nm.
    """

    design = np.stack([acc, vel, pos], axis=1)
    coeffs, *_ = np.linalg.lstsq(design, dac, rcond=None)
    residual = dac - design @ coeffs

    return {
        "m": float(coeffs[0]),
        "c": float(coeffs[1]),
        "k": float(coeffs[2]),
        "r2": _r2(residual, dac),
        "residual_rms": float(np.sqrt(np.mean(residual**2))),
    }


def energy_passivity(vel: np.ndarray, dac: np.ndarray, dt: float) -> dict:
    """
    Actuator power balance.

    Instantaneous power is DAC times velocity; net_work is its time integral,
    inflow_fraction the share of time power is positive. A driven dissipative
    stage takes in net energy (net_work > 0); a model that injects energy
    (free-run instability) skews the balance.
    """

    power = np.asarray(dac, dtype=np.float64) * np.asarray(
        vel, dtype=np.float64
    )

    return {
        "net_work": float(np.sum(power) * dt),
        "mean_power": float(np.mean(power)),
        "inflow_fraction": float(np.mean(power > 0)),
    }


def hysteresis(dac: np.ndarray, pos: np.ndarray) -> dict:
    """
    Net signed area enclosed by the DAC-position trajectory (shoelace), raw and
    normalised by the bounding box. Non-zero means the command-to-position
    relationship carries a loop (dynamic lag or true hysteresis).
    """

    x = np.asarray(dac, dtype=np.float64)
    y = np.asarray(pos, dtype=np.float64)
    area = 0.5 * float(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]))
    box = (x.max() - x.min()) * (y.max() - y.min())

    return {
        "loop_area": abs(area),
        "normalized": abs(area) / box if box > _EPS else 0.0,
    }


def causal_lag(
    dac: np.ndarray,
    pos: np.ndarray,
    sample_rate: float,
    max_lag: int = CAUSAL_MAX_LAG,
) -> dict:
    """
    Command-to-response delay from the cross-correlation peak (position lagging
    DAC), as steps and milliseconds, with the peak correlation.
    """

    a = np.asarray(dac, dtype=np.float64)
    b = np.asarray(pos, dtype=np.float64)
    a = a - a.mean()
    b = b - b.mean()
    n = len(a)
    fa = np.fft.rfft(a, 2 * n)
    fb = np.fft.rfft(b, 2 * n)
    cc = np.fft.irfft(fb * np.conj(fa))[: max_lag + 1]
    norm = np.sqrt(np.sum(a * a) * np.sum(b * b))
    cc = cc / norm if norm > _EPS else cc
    lag = int(np.argmax(cc))

    return {
        "lag_steps": lag,
        "lag_ms": lag * 1000.0 / sample_rate,
        "peak_xcorr": float(cc[lag]),
    }
