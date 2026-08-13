"""
Shared console table rendering for per-axis metric reports.

One renderer for every per-axis percentile table the plant and controller
log, so the format (unit-labelled header, aligned columns, separator
rule) is defined once and generalises to any number of axes. The
percentile ladder is the single source of truth for the latency-style
five-number summary used throughout: P50 (typical), P95 (the acceptance
gate), P99, P99.9 (the tail) and max (the worst case).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

# Values and their column labels, kept in lockstep. Callers compute
# percentiles at PERCENTILES and render under PERCENTILE_LABELS.
PERCENTILES: tuple[float, ...] = (50.0, 95.0, 99.0, 99.9, 100.0)
PERCENTILE_LABELS: tuple[str, ...] = ("P50", "P95", "P99", "P99.9", "max")


def render_table(
    row_labels: Sequence[str],
    col_labels: Sequence[str],
    rows: Sequence[Sequence[float | str]],
    *,
    corner_label: str = "axis",
    unit: str | None = None,
    value_fmt: str = "{:.2f}",
    min_col_width: int = 8,
) -> list[str]:
    """
    Render a fixed-width table (header, rule, one row per label) as lines.

    rows[i][j] is the cell for row_labels[i] under col_labels[j]: a float
    formatted with value_fmt or an already-formatted string. When unit is
    given it is shown in the corner header cell (e.g. "axis (nm)"), so
    the numeric body's unit travels with the table. Columns are
    separated by " | " and the rule places "+" exactly under each
    separator, for any number of rows and columns.
    """

    if any(len(r) != len(col_labels) for r in rows):
        raise ValueError("Every row must have one cell per column label")

    corner = f"{corner_label} ({unit})" if unit else corner_label
    label_w = (
        max(len(corner), *(len(s) for s in row_labels)) if row_labels else 0
    )
    label_w = max(label_w, len(corner))
    cell_w = max(min_col_width, *(len(c) for c in col_labels))

    def cell(value: float | str) -> str:
        text = value if isinstance(value, str) else value_fmt.format(value)

        return f"{text:>{cell_w}}"

    header = f"{corner:>{label_w}} | " + " | ".join(cell(c) for c in col_labels)
    rule = "".join("+" if ch == "|" else "-" for ch in header)
    lines = [header, rule]

    for label, row in zip(row_labels, rows, strict=True):
        lines.append(
            f"{label:>{label_w}} | " + " | ".join(cell(v) for v in row)
        )

    return lines


def log_table(
    logger: logging.Logger, lines: Sequence[str], *, indent: str = "  "
) -> None:
    """
    Emit rendered table lines through logger.info with a leading indent.
    """

    for line in lines:
        logger.info("%s%s", indent, line)


def log_axis_percentiles(
    logger: logging.Logger,
    title: str,
    unit: str,
    axes: Sequence[str],
    values: Sequence[Sequence[float]],
    *,
    labels: Sequence[str] = PERCENTILE_LABELS,
    value_fmt: str = "{:.2f}",
) -> None:
    """
    Log a per-axis percentile table with the unit carried in the header.

    values[a] is the row of percentile figures for axes[a] (already in
    the given unit), one per entry of labels. The unit sits in the
    corner header cell so the table is self-contained; title is a plain
    caption. Any number of axes.
    """

    logger.info("")
    logger.info("%s:", title)
    log_table(
        logger,
        render_table(
            list(axes), list(labels), values, unit=unit, value_fmt=value_fmt
        ),
    )
