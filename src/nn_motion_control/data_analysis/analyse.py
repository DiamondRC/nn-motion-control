"""
Run the full data analysis: envelope, statistics, distributions, linearity.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np

from nn_motion_control.core.system import SystemSpec
from nn_motion_control.data_analysis.envelope import run_envelope
from nn_motion_control.data_analysis.histograms import run_histograms
from nn_motion_control.data_analysis.loader import load_channels
from nn_motion_control.data_analysis.nonlinearity import run_nonlinearity
from nn_motion_control.data_analysis.report import write_report
from nn_motion_control.data_analysis.stats import run_stats

logger = logging.getLogger(os.path.basename(__file__))


# Default output root; a per-dataset subfolder keeps runs from different
# captures side by side.
DEFAULT_ROOT = "data_analysis"


def _position_travel_nm(system: SystemSpec) -> np.ndarray | None:
    """
    Per-axis nameplate travel span (nm) from the position limits, if declared.
    """

    limits = system.channel("position").limits
    if limits is None:
        return None

    return np.array([limits[a][1] - limits[a][0] for a in system.axes])


def run_analysis(
    system: SystemSpec,
    data_path: str,
    out_dir: str | None = None,
    per_axis: bool = True,
) -> str:
    """
    Analyse a dataset against its SystemSpec and return the report path.

    Output defaults to data_analysis/<dataset stem>/ in the working directory;
    'per_axis' also writes full-size per-axis figures under per_axis/.
    """

    stem = Path(data_path).stem
    out_dir = out_dir or os.path.join(DEFAULT_ROOT, stem)
    os.makedirs(out_dir, exist_ok=True)

    data = load_channels(system, data_path)
    n_samples = next(iter(data.values.values())).shape[0]
    n_segments = len(data.segment_offsets) - 1
    logger.info(
        "Analysing %s: %d samples, %d recordings, axes %s",
        os.path.basename(data_path),
        n_samples,
        n_segments,
        data.axes,
    )

    sections = [
        run_envelope(data, out_dir, logger, _position_travel_nm(system)),
        run_stats(data, logger),
        run_histograms(data, out_dir, logger, per_axis),
        run_nonlinearity(data, out_dir, logger, per_axis),
    ]

    header = [
        f"Dataset: `{data_path}`",
        "",
        f"- {n_samples} samples across {n_segments} recordings, "
        f"axes {', '.join(data.axes)}",
        f"- Channels: {', '.join(data.values)}",
    ]
    path = write_report("Data analysis", header, sections, out_dir)
    logger.info("")
    logger.info("Report saved to %s", path)

    return path
