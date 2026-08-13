"""Shared per-axis metric table renderer."""

import pytest

from nn_motion_control.eval.tables import (
    PERCENTILE_LABELS,
    PERCENTILES,
    render_table,
)


def test_percentiles_and_labels_in_lockstep():
    assert len(PERCENTILES) == len(PERCENTILE_LABELS)


def test_render_table_generalises_to_any_axis_count():
    # Five axes, not the hardcoded three: one row each, plus header and rule.
    axes = ["a", "b", "c", "d", "e"]
    rows = [[float(i)] * len(PERCENTILE_LABELS) for i in range(len(axes))]
    lines = render_table(
        axes, list(PERCENTILE_LABELS), rows, corner_label="axis"
    )
    assert len(lines) == 2 + len(axes)
    assert lines[0].strip().startswith("axis")
    assert "P99.9" in lines[0]


def test_render_table_rule_marks_every_separator():
    lines = render_table(["x"], ["P50", "P95"], [[1.0, 2.0]])
    header, rule = lines[0], lines[1]
    assert len(rule) == len(header)
    # A '+' sits under each '|' and nowhere else.
    assert all(
        (rule[i] == "+") == (header[i] == "|") for i in range(len(header))
    )


def test_render_table_carries_unit_in_header_cell():
    # The unit belongs in the header so the table stands alone (no
    # external caption).
    lines = render_table(["x"], ["P50", "P95"], [[1.0, 2.0]], unit="nm")
    assert "axis (nm)" in lines[0]


def test_render_table_accepts_preformatted_string_cells():
    lines = render_table(["x pos"], ["FIT%", "verdict"], [["99.90", "PASS"]])
    assert "PASS" in lines[-1] and "99.90" in lines[-1]


def test_render_table_rejects_ragged_rows():
    with pytest.raises(ValueError):
        render_table(["x"], ["P50", "P95"], [[1.0]])
