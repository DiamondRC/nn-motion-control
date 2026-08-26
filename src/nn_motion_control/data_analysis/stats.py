"""
Per-channel descriptive statistics.

Min, max, mean, std and tail magnitudes for each channel, per axis and pooled
across axes, so the recorded regime (typical excursion and worst case) is
legible before any model metric is read against it.
"""

from __future__ import annotations

import logging

import numpy as np

from nn_motion_control.data_analysis.loader import ChannelData
from nn_motion_control.eval.tables import log_table, render_table

NM_PER_UM = 1000.0
TAIL_PERCENTILES = (95.0, 99.0)
_STAT_LABELS = ["min", "max", "mean", "std", "P95abs", "P99abs"]


def channel_summary(values: np.ndarray) -> list[float]:
    """
    Six-number summary of a 1-D array: min, max, mean, std and the 95th/99th
    percentiles of magnitude.
    """

    tails = np.percentile(np.abs(values), list(TAIL_PERCENTILES))

    return [
        float(values.min()),
        float(values.max()),
        float(values.mean()),
        float(values.std()),
        float(tails[0]),
        float(tails[1]),
    ]


def _channel_table(block: np.ndarray, axes: list[str]) -> list[str]:
    rows = [channel_summary(block[:, a]) for a in range(len(axes))]
    rows.append(channel_summary(block.reshape(-1)))

    return render_table(
        [*axes, "pooled"],
        _STAT_LABELS,
        rows,
        corner_label="axis",
        value_fmt="{:.1f}",
    )


def _velocity_headline(data: ChannelData) -> str | None:
    """
    Median and peak absolute speed in um/s, when velocity and the data rate
    are known.
    """

    if "velocity" not in data.values or not data.data_rate_hz:
        return None
    speed = np.abs(data.values["velocity"])  # [N, A] in units/step
    to_um_s = data.data_rate_hz / NM_PER_UM  # nm/step -> um/s

    return (
        f"Speed |v|: median {np.median(speed) * to_um_s:.1f} um/s, "
        f"peak {speed.max() * to_um_s:.1f} um/s"
    )


def run_stats(data: ChannelData, logger: logging.Logger) -> dict:
    """
    Log a per-channel descriptive-stats table (per axis plus pooled) and
    return a report section.
    """

    tables: list[tuple[str, list[str]]] = []
    logger.info("")
    logger.info("Per-channel descriptive statistics:")

    for name, block in data.values.items():
        unit = data.units[name]
        table = _channel_table(block, data.axes)
        tables.append((f"{name} ({unit})", table))
        logger.info("")
        logger.info("  [%s] (%s)", name, unit)
        log_table(logger, table)

    base_note = (
        "min/max/mean/std are signed; P95abs/P99abs are magnitude "
        "(|value|) percentiles."
    )
    if data.data_rate_hz:
        base_note += (
            f" Rates are per data step (data_rate_hz {data.data_rate_hz:g} Hz)."
        )
    notes = [base_note]
    speed = _velocity_headline(data)
    if speed:
        notes.append(speed)

    return {
        "title": "Descriptive statistics",
        "tables": tables,
        "plots": [],
        "notes": notes,
    }
