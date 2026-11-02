"""
Load raw dataset channels in physical units for analysis.

Reads the ingested HDF5 directly (not through the training loaders), so the
analysis sees the recorded values in their own units: position scaled to nm,
the derived rates in per-step units, and the command in its native unit.
Only the analysed columns are read, kept in their stored precision.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nn_motion_control.core.config import unit_to_nm
from nn_motion_control.core.system import SystemSpec
from nn_motion_control.data._hdf5 import as_dataset, read_columns
from nn_motion_control.data.splits import validate_labels

# Channels analysed, in report order. A system missing any of these simply
# contributes fewer channels.
ANALYSIS_CHANNELS = (
    "position",
    "velocity",
    "acceleration",
    "jerk",
    "dac",
)


@dataclass(frozen=True)
class ChannelData:
    """
    Per-channel recorded values in physical units.

    'values' maps a channel name to a [N, A] array (rows are samples, columns
    are axes) and 'units' its unit label. 'segment_offsets' bounds each source
    recording so later per-file work does not cross a seam.
    """

    axes: list[str]
    values: dict[str, np.ndarray]
    units: dict[str, str]
    segment_offsets: np.ndarray
    data_rate_hz: float | None


def _channel_unit(
    declared_unit: str | None, position_unit: str, channel: str
) -> str:
    """
    Unit label for a channel.

    A channel that declares a unit (position, command) uses it. The derived
    rates carry none in the SystemSpec, so they inherit position's unit per
    step, per step squared and per step cubed.
    """

    if declared_unit:
        return declared_unit

    rate_units = {
        "velocity": f"{position_unit}/step",
        "acceleration": f"{position_unit}/step^2",
        "jerk": f"{position_unit}/step^3",
    }

    return rate_units.get(channel, position_unit)


def load_channels(system: SystemSpec, data_path: str) -> ChannelData:
    """
    Load the analysis channels from a dataset in physical units.

    Position is scaled to nm via the position channel's unit; the command and
    derived rates are read as stored. Channels absent from the system are
    skipped.
    """

    import h5py

    axes = list(system.axes)
    pos_unit = system.channel("position").unit or "nm"
    pos_scale = unit_to_nm(pos_unit)

    present = [c for c in ANALYSIS_CHANNELS if c in system.channels]

    with h5py.File(data_path, "r") as f:
        labels = [
            s.decode() if isinstance(s, bytes) else s
            for s in as_dataset(f, "input_labels")[:]
        ]
        cols = {
            ch: validate_labels(system.labels([ch]), labels, "input")
            for ch in present
        }
        flat = [c for ch in present for c in cols[ch]]
        block = read_columns(as_dataset(f, "inputs"), flat)  # [N, sum(A)]
        segment_offsets = as_dataset(f, "segment_offsets")[:].astype(np.int64)

    values: dict[str, np.ndarray] = {}
    units: dict[str, str] = {}
    start = 0

    for channel in present:
        width = len(cols[channel])
        values[channel] = block[:, start : start + width]  # [N, A]
        start += width
        if channel == "position" and pos_scale != 1.0:
            values[channel] = values[channel] * pos_scale
        declared = system.channel(channel).unit
        units[channel] = _channel_unit(declared, pos_unit, channel)

    return ChannelData(
        axes=axes,
        values=values,
        units=units,
        segment_offsets=segment_offsets,
        data_rate_hz=system.data_rate_hz,
    )
