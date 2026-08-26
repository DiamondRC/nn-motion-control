"""
Per-axis marginal distributions of each channel.

A compact figure per channel (a panel per axis) for fast comparison, plus
full-size per-axis figures for detailed reading. Distribution shape exposes
what min/max/std hide: multimodality (dwell at scan turnarounds), command
saturation piling at the rail, skew and tail weight.
"""

from __future__ import annotations

import logging
import os

import numpy as np

from nn_motion_control.data_analysis.loader import ChannelData
from nn_motion_control.data_analysis.plotting import (
    COMPACT_PANEL_W,
    COMPACT_ROW_H,
    FULL_H,
    FULL_W,
    plot_scale_unit,
    style_axis,
)

HIST_BINS = 160


def _hist_ax(ax, col: np.ndarray, name: str, unit: str, axis: str) -> None:
    ax.hist(col, bins=HIST_BINS, color="C0", alpha=0.8)
    ax.axvline(
        float(col.mean()),
        color="0.2",
        lw=1.2,
        ls="--",
        label=f"mean {col.mean():.1f}",
    )
    ax.set_title(f"{axis}  [{col.min():.1f}, {col.max():.1f}] {unit}")
    ax.set_xlabel(f"{name} ({unit})")
    ax.set_ylabel("count")
    style_axis(ax)
    ax.legend(fontsize=8)


def _compact(
    name: str, block: np.ndarray, unit: str, axes: list[str], path: str
) -> None:
    from nn_motion_control.control.track import _agg_pyplot

    plt = _agg_pyplot()
    n_ax = len(axes)
    fig, axs = plt.subplots(
        1, n_ax, figsize=(COMPACT_PANEL_W * n_ax, COMPACT_ROW_H), squeeze=False
    )

    for a, axis in enumerate(axes):
        _hist_ax(axs[0, a], block[:, a], name, unit, axis)
    fig.suptitle(f"{name} distribution per axis")
    fig.tight_layout()
    fig.savefig(path, format="svg")
    plt.close(fig)


def _per_axis(
    name: str, block: np.ndarray, unit: str, axes: list[str], out_dir: str
) -> list[str]:
    from nn_motion_control.control.track import _agg_pyplot

    plt = _agg_pyplot()
    paths: list[str] = []

    for a, axis in enumerate(axes):
        fig, ax = plt.subplots(figsize=(FULL_W, FULL_H))
        _hist_ax(ax, block[:, a], name, unit, axis)
        fig.suptitle(f"{name} distribution: {axis}")
        fig.tight_layout()
        path = os.path.join(out_dir, f"hist_{name}_{axis}.svg")
        fig.savefig(path, format="svg")
        plt.close(fig)
        paths.append(path)

    return paths


def run_histograms(
    data: ChannelData,
    out_dir: str,
    logger: logging.Logger,
    per_axis: bool = True,
) -> dict:
    """
    Save one compact distribution figure per channel (plus full-size per-axis
    figures when 'per_axis') and return a report section.
    """

    paths: list[str] = []
    per_axis_dir = os.path.join(out_dir, "per_axis")
    if per_axis:
        os.makedirs(per_axis_dir, exist_ok=True)

    for name, block in data.values.items():
        scale, unit = plot_scale_unit(name, data.units[name])
        scaled = block * scale
        path = os.path.join(out_dir, f"hist_{name}.svg")
        _compact(name, scaled, unit, data.axes, path)
        paths.append(path)
        if per_axis:
            _per_axis(name, scaled, unit, data.axes, per_axis_dir)
    logger.info("")
    logger.info("Saved %d per-channel distribution plots.", len(paths))

    notes = [
        f"Per-axis histograms ({HIST_BINS} bins) of each channel; dashed "
        "line marks the mean, the title carries the exact min/max."
    ]
    if per_axis:
        notes.append("Full-size per-axis figures are in per_axis/.")

    return {"title": "Channel distributions", "plots": paths, "notes": notes}
