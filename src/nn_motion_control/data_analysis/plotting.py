"""
Shared plot styling for the analysis figures.

Static SVGs cannot be zoomed like a live plot, so the figures need to be
legible as saved: larger panels, major and minor gridlines, and readable
axis units (position in um rather than nm with a 1e6 offset).
"""

from __future__ import annotations

from typing import Any

NM_PER_UM = 1000.0

# Panel widths/heights in inches. Compact figures stack axes for fast
# comparison; full figures give one axis its own large panel to read.
COMPACT_PANEL_W = 5.2
COMPACT_ROW_H = 4.2
FULL_W = 9.0
FULL_H = 6.0


def plot_scale_unit(channel: str, base_unit: str) -> tuple[float, str]:
    """
    Plot scale factor and unit label for a channel.

    Position is shown in um to avoid an unreadable 1e6 axis offset; other
    channels keep their stored unit.
    """

    if channel == "position":
        return 1.0 / NM_PER_UM, "um"

    return 1.0, base_unit


def style_axis(ax: Any) -> None:
    """
    Readable gridlines: major plus fainter minor, with minor ticks on.
    """

    ax.minorticks_on()
    ax.grid(which="major", alpha=0.4, lw=0.7)
    ax.grid(which="minor", alpha=0.15, lw=0.4)
