"""
Champion registry: the bookkeeping record of the best model per role.

A champion is a label, not a filesystem link. The registry (a small JSON
beside the SystemSpec) records, per role, which artifact is current and why,
so a config can reference 'champion:<role>' instead of a hardcoded path,
letting promoting a new model flow downstream without editing configs.
Checkpoints stay gitignored; the registry is the durable tracked record.
Paths in it are relative to the registry file's directory.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

CHAMPION_PREFIX = "champion:"
REGISTRY_NAME = "champions.json"
REGISTRY_GLOB = "champions*.json"
DEFAULT_SPEC_STEM = "system"


@dataclass(frozen=True)
class Champion:
    """
    One role's champion: which artifact is current, plus why it was promoted.
    """

    role: str
    model: str
    config: str
    checkpoint: str
    promoted: str = ""
    metric: str = ""
    note: str = ""


def is_champion_ref(value: object) -> bool:
    """
    Whether a config value is a 'champion:<role>' reference.
    """

    return isinstance(value, str) and value.startswith(CHAMPION_PREFIX)


def registry_path(system_path: str) -> Path:
    """
    Path to the champion registry beside the given SystemSpec file.

    Each spec owns its registry, so specs sharing a folder (one stage
    identified at different rates) never resolve each other's champions:
    'system.toml' keeps 'champions.json', any other spec gets
    'champions_<stem>.json'.
    """

    spec = Path(system_path).resolve()
    name = (
        REGISTRY_NAME
        if spec.stem == DEFAULT_SPEC_STEM
        else f"{Path(REGISTRY_NAME).stem}_{spec.stem}.json"
    )

    return spec.parent / name


def load_champions(path: str | Path) -> dict[str, Champion]:
    """
    Load the registry into {role: Champion}; an absent file yields {}.
    """

    p = Path(path)
    if not p.exists():
        return {}
    raw = json.loads(p.read_text())

    return {role: Champion(role=role, **spec) for role, spec in raw.items()}


def resolve_champion(ref: str, path: str | Path) -> Champion:
    """
    Resolve a 'champion:<role>' reference to its registry entry.
    """

    role = ref[len(CHAMPION_PREFIX) :]
    champions = load_champions(path)
    if role not in champions:
        raise ValueError(f"No champion registered for role '{role}' in {path}")

    return champions[role]


def resolved_paths(champion: Champion, path: str | Path) -> tuple[str, str]:
    """
    Absolute (config, checkpoint) for a champion, resolved against the
    registry dir.
    """

    base = Path(path).resolve().parent
    return (
        str((base / champion.config).resolve()),
        str((base / champion.checkpoint).resolve()),
    )


def promote(path: str | Path, champion: Champion) -> None:
    """
    Add or replace a role's entry in the registry, preserving the other roles.
    """

    champions = load_champions(path)
    champions[champion.role] = champion
    ordered = {
        role: {k: v for k, v in asdict(c).items() if k != "role"}
        for role, c in champions.items()
    }
    Path(path).write_text(json.dumps(ordered, indent=2) + "\n")
