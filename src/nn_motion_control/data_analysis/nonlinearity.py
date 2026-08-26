"""
Demand-to-position non-linearity.

Per axis, fit a straight line to command (DAC) versus recorded position and
measure how far the data departs from it. The command drives the stage, so a
curved or scattered relationship flags actuator or plant non-linearity.

The relationship is dynamic, not static (DAC drives acceleration, which
integrates to position with phase lag), so part of the spread is dynamics
rather than non-linearity; the scatter shows the combined picture.
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
from nn_motion_control.eval.tables import log_table, render_table

# Points drawn on each scatter panel; the full set is used for the fit.
SCATTER_MAX_POINTS = 20000
_R2_EPS = 1e-12


def linear_fit(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """
    Least-squares fit y = slope * x + intercept and its deviation metrics.

    Returns slope, intercept, r2 (fraction of variance the line explains),
    resid_rms (root-mean-square residual, y units) and max_dev_pct (worst
    residual as a percentage of the y range).
    """

    design = np.stack([x, np.ones_like(x)], axis=1)  # [N, 2]
    (slope, intercept), *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - (slope * x + intercept)
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > _R2_EPS else 0.0
    y_range = float(y.max() - y.min())
    max_dev_pct = (
        100.0 * float(np.abs(resid).max()) / y_range
        if y_range > _R2_EPS
        else 0.0
    )

    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r2": r2,
        "resid_rms": float(np.sqrt(ss_res / len(y))),
        "max_dev_pct": max_dev_pct,
    }


# Bins for the static (conditional-mean) response curve.
BIN_COUNT = 50


def binned_mean(
    x: np.ndarray, y: np.ndarray, n_bins: int = BIN_COUNT
) -> tuple[np.ndarray, np.ndarray]:
    """
    Conditional mean E[y | x] over n_bins equal-width bins of x.

    Returns (centers, means) for populated bins only. Averaging within a bin
    cancels the dynamic scatter and leaves the systematic response shape.
    """

    edges = np.linspace(x.min(), x.max(), n_bins + 1)
    idx = np.clip(np.digitize(x, edges) - 1, 0, n_bins - 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    sums = np.bincount(idx, weights=y, minlength=n_bins)
    counts = np.bincount(idx, minlength=n_bins)
    ok = counts > 0
    means = np.zeros(n_bins)
    means[ok] = sums[ok] / counts[ok]

    return centers[ok], means[ok]


def static_nonlinearity(
    x: np.ndarray, y: np.ndarray, n_bins: int = BIN_COUNT
) -> dict:
    """
    Non-linearity of the static response curve E[y | x].

    Fits a line to the binned conditional means and measures the curve's
    departure from it: bin-mean R2, and the worst bend in y units and as a
    percentage of the bin-mean range. Averaging within bins removes the
    dynamic scatter, so this isolates static (shape) non-linearity.
    """

    centers, means = binned_mean(x, y, n_bins)
    fit = linear_fit(centers, means)
    line = fit["slope"] * centers + fit["intercept"]
    rng = float(means.max() - means.min())
    bend = float(np.abs(means - line).max())
    bend_pct = 100.0 * bend / rng if rng > _R2_EPS else 0.0

    return {
        "r2": fit["r2"],
        "bend_nm": bend,
        "bend_pct": bend_pct,
        "centers": centers,
        "means": means,
    }


def _subsample(n: int) -> np.ndarray:
    if n <= SCATTER_MAX_POINTS:
        return np.arange(n)

    return np.linspace(0, n - 1, SCATTER_MAX_POINTS).astype(np.int64)


def _scatter_ax(
    ax, x, y, fit, static, pos_scale, pos_unit, dac_unit, axis
) -> None:
    line = (fit["slope"] * x + fit["intercept"]) * pos_scale
    ax.scatter(x, y * pos_scale, s=2, alpha=0.2, color="C0", rasterized=True)
    order = np.argsort(x)
    ax.plot(x[order], line[order], "-", color="k", lw=1.3, label="linear fit")
    ax.plot(
        static["centers"],
        static["means"] * pos_scale,
        "-o",
        color="C3",
        ms=3,
        lw=1.6,
        label="mean E[pos|DAC]",
    )
    ax.set_title(
        f"{axis}: linear R2 {fit['r2']:.2f}  |  "
        f"static R2 {static['r2']:.3f}, bend {static['bend_pct']:.1f}%"
    )
    ax.set_xlabel(f"DAC ({dac_unit})")
    ax.set_ylabel(f"position ({pos_unit})")
    style_axis(ax)
    ax.legend(fontsize=8)


def _residual_ax(ax, x, y, fit, pos_scale, pos_unit, dac_unit) -> None:
    line = fit["slope"] * x + fit["intercept"]
    ax.scatter(
        x, (y - line) * pos_scale, s=2, alpha=0.2, color="C1", rasterized=True
    )
    ax.axhline(0.0, color="k", lw=0.9)
    ax.set_xlabel(f"DAC ({dac_unit})")
    ax.set_ylabel(f"residual ({pos_unit})")
    style_axis(ax)


def _compact(data: ChannelData, fits, statics, sel, scale_unit, path) -> None:
    from nn_motion_control.control.track import _agg_pyplot

    plt = _agg_pyplot()
    pos_scale, pos_unit, dac_unit = scale_unit
    dac, pos = data.values["dac"], data.values["position"]
    n_ax = len(data.axes)
    fig, axs = plt.subplots(
        2,
        n_ax,
        figsize=(COMPACT_PANEL_W * n_ax, 2 * COMPACT_ROW_H),
        squeeze=False,
    )

    for a, axis in enumerate(data.axes):
        x, y = dac[sel, a], pos[sel, a]
        _scatter_ax(
            axs[0, a],
            x,
            y,
            fits[a],
            statics[a],
            pos_scale,
            pos_unit,
            dac_unit,
            axis,
        )
        _residual_ax(axs[1, a], x, y, fits[a], pos_scale, pos_unit, dac_unit)
    fig.suptitle("Demand (DAC) vs position, linear fit and residual")
    fig.tight_layout()
    fig.savefig(path, format="svg")
    plt.close(fig)


def _per_axis(
    data: ChannelData, fits, statics, sel, scale_unit, out_dir: str
) -> list[str]:
    from nn_motion_control.control.track import _agg_pyplot

    plt = _agg_pyplot()
    pos_scale, pos_unit, dac_unit = scale_unit
    dac, pos = data.values["dac"], data.values["position"]
    paths: list[str] = []

    for a, axis in enumerate(data.axes):
        x, y = dac[sel, a], pos[sel, a]
        fig, axs = plt.subplots(2, 1, figsize=(FULL_W, 2 * FULL_H))
        _scatter_ax(
            axs[0],
            x,
            y,
            fits[a],
            statics[a],
            pos_scale,
            pos_unit,
            dac_unit,
            axis,
        )
        _residual_ax(axs[1], x, y, fits[a], pos_scale, pos_unit, dac_unit)
        fig.suptitle(f"Demand (DAC) vs position: {axis}")
        fig.tight_layout()
        path = os.path.join(out_dir, f"nonlinearity_{axis}.svg")
        fig.savefig(path, format="svg")
        plt.close(fig)
        paths.append(path)

    return paths


def run_nonlinearity(
    data: ChannelData,
    out_dir: str,
    logger: logging.Logger,
    per_axis: bool = True,
) -> dict:
    """
    Fit DAC-vs-position per axis, log the deviation table, save the plots.
    """

    if "dac" not in data.values or "position" not in data.values:
        return {"title": "Demand-to-position non-linearity", "plots": []}

    dac = data.values["dac"]
    pos = data.values["position"]
    n = len(data.axes)
    fits = [linear_fit(dac[:, a], pos[:, a]) for a in range(n)]
    statics = [static_nonlinearity(dac[:, a], pos[:, a]) for a in range(n)]

    raw_table = render_table(
        data.axes,
        ["slope", "R2", "resid RMS (nm)", "max dev %"],
        [[f["slope"], f["r2"], f["resid_rms"], f["max_dev_pct"]] for f in fits],
        corner_label="axis",
        value_fmt="{:.3f}",
    )
    static_table = render_table(
        data.axes,
        ["R2", "bend (nm)", "bend %"],
        [[s["r2"], s["bend_nm"], s["bend_pct"]] for s in statics],
        corner_label="axis",
        value_fmt="{:.3f}",
    )
    logger.info("")
    logger.info("Demand (DAC) vs position linearity -- raw (instantaneous):")
    log_table(logger, raw_table)
    logger.info("")
    logger.info("Demand (DAC) vs position linearity -- static (binned mean):")
    log_table(logger, static_table)

    pos_scale, pos_unit = plot_scale_unit("position", data.units["position"])
    _, dac_unit = plot_scale_unit("dac", data.units["dac"])
    scale_unit = (pos_scale, pos_unit, dac_unit)
    sel = _subsample(dac.shape[0])

    path = os.path.join(out_dir, "nonlinearity.svg")
    _compact(data, fits, statics, sel, scale_unit, path)
    if per_axis:
        per_axis_dir = os.path.join(out_dir, "per_axis")
        os.makedirs(per_axis_dir, exist_ok=True)
        _per_axis(data, fits, statics, sel, scale_unit, per_axis_dir)

    notes = [
        "Two complementary readings of position against DAC (a dynamic link: "
        "DAC drives acceleration, which integrates to position).",
        "Raw (instantaneous): a straight-line fit to every sample. Its low R2 "
        "and large residual are dominated by the dynamics (position is not an "
        "instantaneous function of DAC), so this is a predictability measure, "
        "not actuator non-linearity.",
        "Static (binned mean): a line fit to the conditional mean E[pos|DAC] "
        "over "
        f"{BIN_COUNT} DAC bins, which averages the dynamic scatter away. Its "
        "R2 and bend (the curve's worst departure from the line) are the "
        "genuine static non-linearity: bend near 0 means a linear mean "
        "response.",
    ]
    if per_axis:
        notes.append("Full-size per-axis figures are in per_axis/.")

    return {
        "title": "Demand-to-position non-linearity",
        "tables": [
            ("raw (instantaneous, all samples)", raw_table),
            ("static (binned mean E[pos|DAC])", static_table),
        ],
        "plots": [path],
        "notes": notes,
    }
