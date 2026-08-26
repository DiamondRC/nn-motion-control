"""
Position envelope: the reachable range the stage actually explored.

Per-axis min, max and span of the recorded position, in nm. These bounds are
the operating region for later setpoint generation, and the yardstick model
drift figures are read against.
"""

from __future__ import annotations

import logging
import os

import numpy as np

from nn_motion_control.data_analysis.loader import ChannelData
from nn_motion_control.eval.tables import log_table, render_table

NM_PER_UM = 1000.0


def envelope_bounds(data: ChannelData) -> dict[str, np.ndarray]:
    """
    Per-axis min, max and span of the recorded position, each [A] in nm.
    """

    pos = data.values["position"]  # [N, A]

    return {
        "min": pos.min(axis=0),
        "max": pos.max(axis=0),
        "span": pos.max(axis=0) - pos.min(axis=0),
    }


def _table(
    data: ChannelData,
    bounds: dict[str, np.ndarray],
    travel_nm: np.ndarray | None,
) -> list[str]:
    col_labels = ["min", "max", "span", "span (um)"]
    if travel_nm is not None:
        col_labels.append("% travel")
    rows = []

    for a in range(len(data.axes)):
        row = [
            float(bounds["min"][a]),
            float(bounds["max"][a]),
            float(bounds["span"][a]),
            float(bounds["span"][a]) / NM_PER_UM,
        ]
        if travel_nm is not None:
            row.append(100.0 * float(bounds["span"][a]) / float(travel_nm[a]))
        rows.append(row)

    return render_table(
        data.axes,
        col_labels,
        rows,
        corner_label="axis",
        unit="nm",
        value_fmt="{:.1f}",
    )


def _plot(data: ChannelData, bounds: dict[str, np.ndarray], path: str) -> None:
    from nn_motion_control.control.track import _agg_pyplot
    from nn_motion_control.data_analysis.plotting import FULL_W, style_axis

    plt = _agg_pyplot()
    lo = bounds["min"] / NM_PER_UM
    hi = bounds["max"] / NM_PER_UM
    y = np.arange(len(data.axes))
    fig, ax = plt.subplots(figsize=(FULL_W, 1.6 + 0.9 * len(data.axes)))
    ax.hlines(y, lo, hi, color="C0", lw=8, alpha=0.6)
    ax.plot(lo, y, "|", color="C0", ms=18)
    ax.plot(hi, y, "|", color="C0", ms=18)
    ax.axvline(0.0, color="0.4", lw=1.0, ls="--")

    for a in range(len(data.axes)):
        ax.annotate(
            f"{lo[a]:.1f}",
            (lo[a], y[a]),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8,
        )
        ax.annotate(
            f"{hi[a]:.1f}",
            (hi[a], y[a]),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8,
        )
    ax.set_yticks(y)
    ax.set_yticklabels(data.axes)
    ax.set_xlabel("position (um)")
    ax.set_title("Recorded position envelope (displacement from rest)")
    style_axis(ax)
    fig.tight_layout()
    fig.savefig(path, format="svg")
    plt.close(fig)


def run_envelope(
    data: ChannelData,
    out_dir: str,
    logger: logging.Logger,
    travel_nm: np.ndarray | None = None,
) -> dict:
    """
    Log the envelope table, save the envelope plot, return a report section.
    """

    bounds = envelope_bounds(data)
    table = _table(data, bounds, travel_nm)
    logger.info("")
    logger.info("Position envelope (reachable range):")
    log_table(logger, table)

    path = os.path.join(out_dir, "envelope.svg")
    _plot(data, bounds, path)

    return {
        "title": "Position envelope",
        "table": table,
        "plots": [path],
        "notes": [
            "Per-axis min/max/span of recorded position (nm). Positions are "
            "centred per file, so these are displacement-from-rest spans."
        ],
    }
