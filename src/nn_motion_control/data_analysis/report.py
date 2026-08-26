"""
Assemble the analysis sections into a single markdown report.

Each section is a dict with a title, optional rendered table(s), plot paths
and notes; tables go in fenced blocks and plots as image links relative to
the report so it renders in place.
"""

from __future__ import annotations

import os


def _section_md(section: dict) -> list[str]:
    lines = [f"## {section['title']}", ""]

    for note in section.get("notes", []):
        lines += [note, ""]

    single = section.get("table")
    if single:
        lines += ["```", *single, "```", ""]

    for label, table in section.get("tables", []):
        lines += [f"**{label}**", "", "```", *table, "```", ""]

    for path in section.get("plots", []):
        name = os.path.basename(path)
        lines += [f"![{name}]({name})", ""]

    return lines


def write_report(
    title: str, header_lines: list[str], sections: list[dict], out_dir: str
) -> str:
    """
    Write report.md from the sections and return its path.
    """

    lines = [f"# {title}", ""]
    lines += header_lines + [""]

    for section in sections:
        lines += _section_md(section)

    path = os.path.join(out_dir, "report.md")
    with open(path, "w") as f:
        f.write("\n".join(lines).rstrip() + "\n")

    return path
