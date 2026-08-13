"""
Interactive terminal pickers: run a command with no config and choose one
in-terminal.

A picker is only offered on an interactive TTY; with a piped or redirected
stdin (CI, scripts) a missing argument is an error instead, so
non-interactive use stays deterministic and every command remains
scriptable when its argument is passed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def is_interactive() -> bool:
    """
    Whether stdin and stdout are both attached to a terminal.
    """

    return sys.stdin.isatty() and sys.stdout.isatty()


def find_configs() -> list[Path]:
    """
    Artifact configs discoverable from the working directory, de-duplicated
    and sorted.
    """

    found: list[Path] = []

    for pattern in ("configs/*.json", "examples/*/configs/*.json"):
        found.extend(Path().glob(pattern))

    return sorted(set(found))


def _champion_roles() -> dict[str, str]:
    """
    Map an absolute config path to its champion role, for any registry
    under examples/.
    """

    from nn_motion_control.core.champions import load_champions

    tags: dict[str, str] = {}

    for registry in sorted(Path().glob("examples/*/champions.json")):
        for role, champion in load_champions(registry).items():
            tags[str((registry.parent / champion.config).resolve())] = role

    return tags


def pick_config(purpose: str = "config") -> str:
    """
    Prompt for one artifact config from those discoverable in the working
    directory.

    Configs registered as champions are listed first, tagged with their
    role, so the best models are the obvious pick.
    """

    import questionary

    configs = find_configs()
    if not configs:
        raise SystemExit(
            "No configs found under configs/ or examples/*/configs/; "
            "pass a config path explicitly."
        )

    tags = _champion_roles()
    champions, others = [], []

    for cfg in configs:
        role = tags.get(str(cfg.resolve()))
        if role:
            champions.append(
                questionary.Choice(
                    title=f"* {cfg}  (champion:{role})", value=str(cfg)
                )
            )
        else:
            others.append(questionary.Choice(title=str(cfg), value=str(cfg)))

    answer = questionary.select(
        f"Select a {purpose}:", choices=champions + others
    ).ask()
    if answer is None:
        raise SystemExit("No selection made.")

    return answer


PREFS_FILE = Path(".nnmc_prefs.json")


def load_prefs(command: str) -> dict | None:
    """
    The last-used interactive settings for a command, or None if
    absent/unreadable.
    """

    if not PREFS_FILE.exists():
        return None
    try:
        return json.loads(PREFS_FILE.read_text()).get(command)
    except (json.JSONDecodeError, OSError, AttributeError):
        return None


def save_prefs(command: str, data: dict) -> None:
    """
    Persist a command's interactive settings, merging with any other
    commands' prefs.
    """

    try:
        existing = (
            json.loads(PREFS_FILE.read_text()) if PREFS_FILE.exists() else {}
        )
        if not isinstance(existing, dict):
            existing = {}
    except (json.JSONDecodeError, OSError):
        existing = {}
    existing[command] = data
    try:
        PREFS_FILE.write_text(json.dumps(existing, indent=2) + "\n")
    except OSError:
        pass


def _ask(question: str, kind: type, default) -> object:
    """
    Prompt for one option value of the given kind (int, bool or str); abort
    on cancel.
    """

    import questionary

    if kind is bool:
        answer = questionary.confirm(question, default=bool(default)).ask()
    else:
        answer = questionary.text(question, default=str(default)).ask()
    if answer is None:
        raise SystemExit("Cancelled.")

    return answer if kind is bool else kind(answer)


def _summarise(values: dict, options: list[tuple]) -> str:
    """
    A one-line summary of the config and option values for the reuse prompt.
    """

    parts = [f"config={Path(values['config']).name}"]
    parts += [f"{key}={values[key]}" for key, _, _, _ in options]

    return ", ".join(parts)


def interactive_setup(
    command: str, pick_purpose: str, options: list[tuple]
) -> dict:
    """
    Interactive config + options flow with a "use previous settings?"
    shortcut.

    'options' is a list of (key, prompt, kind, default) where kind is int,
    bool or str. If the last-used settings are still valid the flow offers
    to reuse them (default yes); otherwise it picks a config and prompts
    each option, pre-filled with the last value or default. The chosen
    settings are saved for next time. TTY only.
    """

    import questionary

    prev = load_prefs(command)
    if not isinstance(prev, dict):
        prev = {}
    prev_valid = (
        bool(prev)
        and Path(prev.get("config", "")).exists()
        and all(key in prev for key, _, _, _ in options)
    )
    if prev_valid:
        reuse = questionary.confirm(
            f"Use previous settings? [{_summarise(prev, options)}]",
            default=True,
        ).ask()
        if reuse is None:
            raise SystemExit("Cancelled.")
        if reuse:
            return dict(prev)

    result: dict = {"config": pick_config(pick_purpose)}

    for key, prompt, kind, default in options:
        result[key] = _ask(prompt, kind, prev.get(key, default))
    save_prefs(command, result)

    return result


def offer_promotion(config_path: str, role: str, promoted: str) -> None:
    """
    After a run, offer to record the artifact as 'champion:<role>'
    (interactive only).

    Reads only the config's 'system'/'model_name'/'out_dir' to locate the
    registry and the just-written checkpoint, so it needs no artifact
    classes. A no-op off a TTY.
    """

    if not is_interactive():
        return

    import os

    import questionary

    from nn_motion_control.core.champions import (
        Champion,
        promote,
        registry_path,
    )

    cfg = json.loads(Path(config_path).read_text())
    base_dir = Path(config_path).parent
    system_path = (base_dir / cfg["system"]).resolve()
    registry = registry_path(system_path)
    model_name = cfg["model_name"]
    out_dir = (base_dir / cfg.get("run", {}).get("out_dir", "runs")).resolve()
    checkpoint = out_dir / model_name / f"{model_name}.pth"

    if not questionary.confirm(
        f"Record {model_name} as champion:{role}?", default=False
    ).ask():
        return
    metric = questionary.text("One-line metric/why (optional):").ask() or ""

    registry_dir = registry.parent
    promote(
        registry,
        Champion(
            role=role,
            model=model_name,
            config=os.path.relpath(config_path, registry_dir),
            checkpoint=os.path.relpath(checkpoint, registry_dir),
            promoted=promoted,
            metric=metric,
        ),
    )
