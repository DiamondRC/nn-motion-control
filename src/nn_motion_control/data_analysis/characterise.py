"""
Run the full characterisation battery on a dataset.

Size, complexity (dimension/spectral/entropy/dynamics) and physics/correctness
axes, per axis, on a representative recording. Every measure is a plain
function on a signal or state, so the same battery reruns on plant rollouts and
closed-loop trajectories in the comparison phase.
"""

from __future__ import annotations

import logging
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
from nn_motion_control.data_analysis.loader import load_channels
from nn_motion_control.data_analysis.plotting import (
    COMPACT_PANEL_W,
    COMPACT_ROW_H,
    plot_scale_unit,
    style_axis,
)
from nn_motion_control.data_analysis.report import write_report
from nn_motion_control.eval.tables import log_table, render_table

logger = logging.getLogger(os.path.basename(__file__))

DEFAULT_ROOT = "data_analysis"
# The battery runs on the longest recording, capped so the O(N^2) measures and
# the FFTs stay quick; the internal subsampling handles the rest.
MAX_SAMPLES = 300000


def _representative_window(segment_offsets: np.ndarray) -> slice:
    lengths = np.diff(segment_offsets)
    k = int(np.argmax(lengths))
    start = int(segment_offsets[k])
    end = min(int(segment_offsets[k + 1]), start + MAX_SAMPLES)

    return slice(start, end)


def _axis_measures(
    pos: np.ndarray,
    dac: np.ndarray,
    vel: np.ndarray,
    acc: np.ndarray,
    sample_rate: float,
) -> dict:
    """
    Every scalar measure for one axis's representative window.
    """

    # The stored rates are per data step (finite differences at ingest), so
    # kinematic and energy checks use a one-step dt, not seconds.
    dt = 1.0
    tau = dimension.estimate_tau(pos)
    d2_ms, d2_curve = dimension.d2_vs_embedding(pos, tau)
    plateau = (
        float(np.median(d2_curve[len(d2_curve) // 2 :]))
        if d2_curve.size
        else 0.0
    )
    embedded = dimension.delay_embed(pos, tau, dimension.EMBED_DIMS[-1])
    eom = physics.eom_residual(pos, vel, acc, dac)
    kin = physics.kinematic_consistency(pos, vel, acc, dt)
    energy = physics.energy_passivity(vel, dac, dt)
    hyst = physics.hysteresis(dac, pos)
    lag = physics.causal_lag(dac, pos, sample_rate)
    incr = dynamics.increment_stats(pos)

    return {
        "tau": tau,
        "d2": plateau,
        "d2_ms": d2_ms,
        "d2_curve": d2_curve,
        "higuchi": dimension.higuchi_fd(pos),
        "pca_embed": dimension.pca_dimension(embedded)["participation_ratio"],
        "spectral_entropy": spectral.spectral_entropy(pos, sample_rate),
        "bandwidth": spectral.bandwidth(pos, sample_rate),
        "one_over_f": spectral.one_over_f_slope(pos, sample_rate),
        "perm_entropy": entropy.permutation_entropy(pos),
        "sample_entropy": entropy.sample_entropy(pos),
        "lempel_ziv": entropy.lempel_ziv(pos),
        # Per data step (multiplying by the rate over-inflates on smooth,
        # oversampled position); near 0 means non-chaotic.
        "lyapunov": dynamics.largest_lyapunov(pos, tau, 4, 1.0),
        "dfa_hurst": dynamics.dfa_hurst(pos),
        "rqa_det": dynamics.rqa(embedded)["determinism"],
        "kurtosis": incr["kurtosis"],
        "eom_r2": eom["r2"],
        "eom_m": eom["m"],
        "eom_c": eom["c"],
        "eom_k": eom["k"],
        "kin_vel_r2": kin["vel_r2"],
        "energy_inflow": energy["inflow_fraction"],
        "hysteresis": hyst["normalized"],
        "causal_lag_ms": lag["lag_ms"],
        "io_coherence": spectral.mean_coherence(dac, pos, sample_rate),
        "te_dac_pos": information.transfer_entropy(dac, pos),
        "mi_dac_pos": information.mutual_information(dac, pos),
    }


def _table(axes, per_axis, keys, labels, fmt="{:.3f}"):
    rows = [[per_axis[a][k] for k in keys] for a in axes]

    return render_table(axes, labels, rows, corner_label="axis", value_fmt=fmt)


def _plot_psd(axes, pos, sample_rate, path):
    from nn_motion_control.control.track import _agg_pyplot

    plt = _agg_pyplot()
    fig, ax = plt.subplots(figsize=(COMPACT_PANEL_W * 2, COMPACT_ROW_H))

    for a, axis in enumerate(axes):
        freqs, power = spectral.psd(pos[:, a], sample_rate)
        ax.loglog(freqs[1:], power[1:], color=f"C{a}", lw=0.9, label=axis)

        for f, _ in spectral.resonance_peaks(pos[:, a], sample_rate):
            ax.axvline(f, color=f"C{a}", ls=":", lw=0.6, alpha=0.5)
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("PSD (nm^2/Hz)")
    ax.set_title("Position power spectral density (dotted = resonance peaks)")
    style_axis(ax)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, format="svg")
    plt.close(fig)


def _plot_bode(axes, dac, pos, sample_rate, path):
    from nn_motion_control.control.track import _agg_pyplot

    plt = _agg_pyplot()
    fig, axs = plt.subplots(
        3, 1, figsize=(COMPACT_PANEL_W * 2, 3 * COMPACT_ROW_H), sharex=True
    )

    for a, axis in enumerate(axes):
        freqs, gain, phase, coh = spectral.frequency_response(
            dac[:, a], pos[:, a], sample_rate
        )
        axs[0].loglog(freqs[1:], gain[1:], color=f"C{a}", lw=0.9, label=axis)
        axs[1].semilogx(freqs[1:], phase[1:], color=f"C{a}", lw=0.9)
        axs[2].semilogx(freqs[1:], coh[1:], color=f"C{a}", lw=0.9)
    axs[0].set_ylabel("gain")
    axs[1].set_ylabel("phase (deg)")
    axs[2].set_ylabel("coherence")
    axs[2].set_xlabel("frequency (Hz)")
    axs[0].set_title("DAC -> position frequency response (Bode)")

    for ax in axs:
        style_axis(ax)
    axs[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, format="svg")
    plt.close(fig)


def _plot_d2_saturation(axes, per_axis, path):
    from nn_motion_control.control.track import _agg_pyplot

    plt = _agg_pyplot()
    fig, ax = plt.subplots(figsize=(COMPACT_PANEL_W * 2, COMPACT_ROW_H))

    for a, axis in enumerate(axes):
        ax.plot(
            per_axis[axis]["d2_ms"],
            per_axis[axis]["d2_curve"],
            "-o",
            color=f"C{a}",
            ms=4,
            label=axis,
        )
    ax.set_xlabel("embedding dimension m")
    ax.set_ylabel("correlation dimension D2")
    ax.set_title("D2 vs embedding dimension (plateau = attractor dimension)")
    style_axis(ax)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, format="svg")
    plt.close(fig)


def run_characterisation(
    system: SystemSpec,
    data_path: str,
    out_dir: str | None = None,
    per_axis: bool = True,
) -> str:
    """
    Run the battery on a dataset and return the report path.
    """

    del per_axis  # reserved for future per-axis figure output
    out_dir = out_dir or os.path.join(
        DEFAULT_ROOT, Path(data_path).stem, "characterisation"
    )
    os.makedirs(out_dir, exist_ok=True)

    data = load_channels(system, data_path)
    sample_rate = data.data_rate_hz or 1.0
    axes = data.axes
    window = _representative_window(data.segment_offsets)
    pos = data.values["position"][window]
    dac = data.values["dac"][window]
    vel = data.values["velocity"][window]
    acc = data.values["acceleration"][window]
    logger.info(
        "Characterising %s: %d-sample window, %.0f Hz, axes %s",
        os.path.basename(data_path),
        pos.shape[0],
        sample_rate,
        axes,
    )

    per_axis_m = {
        axis: _axis_measures(
            pos[:, a], dac[:, a], vel[:, a], acc[:, a], sample_rate
        )
        for a, axis in enumerate(axes)
    }

    to_um = plot_scale_unit("position", data.units["position"])[0]
    size_rows = [
        [
            (pos[:, a].max() - pos[:, a].min()) * to_um,
            pos[:, a].std(),
            per_axis_m[axis]["bandwidth"],
        ]
        for a, axis in enumerate(axes)
    ]
    size_table = render_table(
        axes,
        ["range(um)", "std(nm)", "bandwidth(Hz)"],
        size_rows,
        corner_label="axis",
        value_fmt="{:.2f}",
    )
    spatial = dimension.pca_dimension(pos)["participation_ratio"]

    dim_table = _table(
        axes,
        per_axis_m,
        ["d2", "higuchi", "pca_embed", "spectral_entropy", "one_over_f"],
        ["D2", "higuchi", "PCA-embed", "spec-ent", "1/f slope"],
    )
    dyn_table = _table(
        axes,
        per_axis_m,
        [
            "perm_entropy",
            "sample_entropy",
            "lempel_ziv",
            "lyapunov",
            "dfa_hurst",
            "rqa_det",
            "kurtosis",
        ],
        ["perm-e", "samp-e", "LZ", "lyap", "DFA-H", "RQA-det", "kurt"],
    )
    phys_table = _table(
        axes,
        per_axis_m,
        ["eom_r2", "eom_m", "eom_c", "eom_k", "kin_vel_r2", "energy_inflow"],
        ["EOM R2", "m", "c", "k", "kin velR2", "E inflow"],
    )
    io_table = _table(
        axes,
        per_axis_m,
        [
            "hysteresis",
            "causal_lag_ms",
            "io_coherence",
            "te_dac_pos",
            "mi_dac_pos",
        ],
        ["hyst", "lag(ms)", "IO coh", "TE dac>x", "MI dac,x"],
    )

    couple_rows = []
    for i in range(len(axes)):
        for j in range(i + 1, len(axes)):
            couple_rows.append(
                [
                    information.mutual_information(pos[:, i], pos[:, j]),
                    spectral.mean_coherence(pos[:, i], pos[:, j], sample_rate),
                ]
            )
    couple_labels = [
        f"{axes[i]}-{axes[j]}"
        for i in range(len(axes))
        for j in range(i + 1, len(axes))
    ]
    couple_table = render_table(
        couple_labels,
        ["MI", "coherence"],
        couple_rows,
        corner_label="pair",
        value_fmt="{:.3f}",
    )

    homogeneity = {
        axis: regime.spatial_homogeneity(
            data.values["position"][:, a], data.values["acceleration"][:, a]
        )["cv"]
        for a, axis in enumerate(axes)
    }
    stat = {
        axis: regime.stationarity(
            data.values["position"][:, a], data.segment_offsets
        )["mean_drift"]
        for a, axis in enumerate(axes)
    }
    regime_table = render_table(
        axes,
        ["homogeneity cv", "stationarity drift"],
        [[homogeneity[a], stat[a]] for a in axes],
        corner_label="axis",
        value_fmt="{:.4f}",
    )

    for title, table in [
        ("size", size_table),
        ("complexity (dimension/spectral)", dim_table),
        ("complexity (entropy/dynamics)", dyn_table),
        ("physics: equation of motion + energy", phys_table),
        ("input-output", io_table),
        ("cross-axis coupling", couple_table),
        ("regime: homogeneity + stationarity", regime_table),
    ]:
        logger.info("")
        logger.info("%s:", title)
        log_table(logger, table)
    logger.info("")
    logger.info("3-axis spatial PCA participation ratio: %.3f", spatial)

    psd_path = os.path.join(out_dir, "psd.svg")
    bode_path = os.path.join(out_dir, "bode.svg")
    d2_path = os.path.join(out_dir, "d2_saturation.svg")
    _plot_psd(axes, pos, sample_rate, psd_path)
    _plot_bode(axes, dac, pos, sample_rate, bode_path)
    _plot_d2_saturation(axes, per_axis_m, d2_path)

    sections = [
        {
            "title": "Size",
            "table": size_table,
            "plots": [],
            "notes": ["Per-axis range, std and effective bandwidth."],
        },
        {
            "title": "Complexity",
            "tables": [
                ("dimension / spectral", dim_table),
                ("entropy / dynamics", dyn_table),
            ],
            "plots": [d2_path, psd_path],
            "notes": [
                f"3-axis spatial PCA participation ratio: {spatial:.3f}.",
                "D2 is the correlation-dimension plateau; PCA-embed the linear "
                "DOF of the delay embedding; higuchi the waveform fractal; "
                "spec-ent the spectral entropy; lyap the largest Lyapunov "
                "exponent (per step, ~0 = non-chaotic); DFA-H the Hurst "
                "exponent; RQA-det "
                "determinism; kurt the increment excess kurtosis.",
            ],
        },
        {
            "title": "Physics and input-output",
            "tables": [
                ("equation of motion + energy", phys_table),
                ("input-output", io_table),
            ],
            "plots": [bode_path],
            "notes": [
                "EOM R2 is the fit of m*a+c*v+k*x = DAC (near 1 = linear "
                "2nd-order); kin velR2 the kinematic consistency; E inflow the "
                "fraction of time the actuator adds energy; hyst the "
                "normalised DAC-position loop area; lag the "
                "command-to-response delay (for an integrating DAC->position "
                "this saturates at the search window; read the Bode phase "
                "instead); IO coh / TE / MI the DAC->position coupling.",
            ],
        },
        {
            "title": "Coupling and regime",
            "tables": [
                ("cross-axis coupling", couple_table),
                ("homogeneity + stationarity", regime_table),
            ],
            "plots": [],
            "notes": [
                "Homogeneity cv is the position-dependence of acceleration "
                "spread (0 = uniform across the workspace); stationarity drift "
                "the within-recording statistic drift."
            ],
        },
    ]
    header = [
        f"Dataset: `{data_path}`",
        "",
        f"- Representative window: {pos.shape[0]} samples at "
        f"{sample_rate:g} Hz, axes {', '.join(axes)}",
    ]
    path = write_report("Signal characterisation", header, sections, out_dir)
    logger.info("")
    logger.info("Report saved to %s", path)

    return path
