"""Unit tests for core.system.SystemSpec.

Broadcast/per-axis, label expansion.
"""

from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from nn_motion_control.core.system import ChannelSpec, SystemSpec

REPO = Path(__file__).resolve().parents[1]
DELTABOT = REPO / "examples/deltabot/system.toml"


def test_per_axis_label_requires_axis_placeholder():
    # A per-axis template without {axis} would collapse all axes to
    # one column, reject.
    with pytest.raises(ValueError):
        ChannelSpec(
            name="pos", kind="measured", per_axis=True, label_template="{name}"
        )
    # A non-per-axis channel may omit it.
    ChannelSpec(
        name="temp", kind="measured", per_axis=False, label_template="temp"
    )


def test_loads_deltabot_spec():
    spec = SystemSpec.from_toml(DELTABOT)
    assert spec.name == "deltabot"
    assert spec.axes == ["x", "y", "z"]
    assert spec.board == "xc7z030"
    assert spec.clock_hz == 125_000_000
    assert spec.servo_rate_hz == 200_000
    assert spec.data_rate_hz == 10_000


def test_broadcast_field_fans_out_to_all_axes():
    spec = SystemSpec.from_toml(DELTABOT)
    pos = spec.channel("position")
    # A single 'limits' value must appear identically for every axis.
    assert pos.limits == {
        "x": [-0.004, 0.004],
        "y": [-0.004, 0.004],
        "z": [-0.004, 0.004],
    }


def test_per_axis_table_is_keyed_by_axis():
    spec = SystemSpec.from_toml(DELTABOT)
    dac = spec.channel("dac")
    assert dac.range is not None and dac.safe_range is not None
    assert set(dac.range) == {"x", "y", "z"}
    assert dac.safe_range["z"] == [-737.25, 737.25]


def test_label_expansion_is_axis_major():
    spec = SystemSpec.from_toml(DELTABOT)
    # Each axis's selected channels are contiguous, using the legacy labels.
    assert spec.labels(["position", "dac"]) == [
        "x_pos",
        "x_DAC_real",
        "y_pos",
        "y_DAC_real",
        "z_pos",
        "z_DAC_real",
    ]


def test_full_input_expansion_matches_legacy_15():
    spec = SystemSpec.from_toml(DELTABOT)
    labels = spec.labels(
        ["position", "velocity", "acceleration", "jerk", "dac"]
    )
    assert len(labels) == 15
    assert labels[:5] == ["x_pos", "x_vel", "x_acc", "x_jer", "x_DAC_real"]


def test_derived_channel_source_validated():
    with pytest.raises(ValueError, match="unknown source"):
        SystemSpec.from_dict(
            {
                "name": "s",
                "axes": ["x"],
                "channels": {
                    "velocity": {"kind": "derived", "from": "nope", "order": 1}
                },
            }
        )


def test_per_axis_key_mismatch_rejected():
    with pytest.raises(ValueError, match="must match system axes"):
        SystemSpec.from_dict(
            {
                "name": "s",
                "axes": ["x", "y"],
                "channels": {
                    "dac": {
                        "kind": "command",
                        "range": {"x": [-1, 1]},
                    },  # missing y
                },
            }
        )


def test_clocks_per_step():
    # deltabot: 125 MHz fabric clock / 200 kHz servo rate = 625 clocks per step.
    spec = SystemSpec.from_toml(DELTABOT)
    assert spec.clocks_per_step() == 625.0
    # None when either rate is missing.
    spec2 = SystemSpec.from_dict(
        {
            "name": "s",
            "axes": ["x"],
            "target": {"clock_hz": 125_000_000},
            "channels": {},
        }
    )
    assert spec2.clocks_per_step() is None


def test_control_substeps():
    # deltabot: 200 kHz control rate / 10 kHz data rate = 20 control steps per
    # plant transition ("twenty chances").
    spec = SystemSpec.from_toml(DELTABOT)
    assert spec.control_substeps() == 20.0
    # None when either rate is missing.
    spec2 = SystemSpec.from_dict(
        {"name": "s", "axes": ["x"], "servo_rate_hz": 200_000, "channels": {}}
    )
    assert spec2.control_substeps() is None


def test_unknown_channel_raises():
    spec = SystemSpec.from_toml(DELTABOT)
    with pytest.raises(KeyError, match="Unknown channel"):
        spec.channel("nope")


def test_bad_kind_rejected():
    with pytest.raises(ValueError, match="kind must be one of"):
        SystemSpec.from_dict(
            {"name": "s", "axes": ["x"], "channels": {"p": {"kind": "bogus"}}}
        )


def test_empty_axes_rejected():
    with pytest.raises(ValueError, match="at least one axis"):
        SystemSpec.from_dict({"name": "s", "axes": [], "channels": {}})


def test_duplicate_axes_rejected():
    with pytest.raises(ValueError, match="unique"):
        SystemSpec.from_dict({"name": "s", "axes": ["x", "x"], "channels": {}})


def test_non_per_axis_channel_label_ignores_axis():
    spec = SystemSpec.from_dict(
        {
            "name": "s",
            "axes": ["x"],
            "channels": {"temp": {"kind": "measured", "per_axis": False}},
        }
    )
    assert spec.channel("temp").label() == "temp"


@given(
    st.lists(
        st.sampled_from(
            ["position", "velocity", "acceleration", "jerk", "dac"]
        ),
        min_size=1,
        max_size=5,
        unique=True,
    )
)
def test_label_count_is_axes_times_channels(channels):
    spec = SystemSpec.from_toml(DELTABOT)
    assert len(spec.labels(channels)) == len(spec.axes) * len(channels)
