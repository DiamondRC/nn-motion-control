"""Champion registry.

Load, resolve, promote, and config champion-ref resolution.
"""

import json
from pathlib import Path

import pytest

from nn_motion_control.core.champions import (
    Champion,
    is_champion_ref,
    load_champions,
    promote,
    registry_path,
    resolve_champion,
    resolved_paths,
)

REPO = Path(__file__).resolve().parents[1]


def _write(tmp_path):
    reg = tmp_path / "champions.json"
    reg.write_text(
        json.dumps(
            {
                "plant": {
                    "model": "m",
                    "config": "configs/p.json",
                    "checkpoint": "runs/p.pth",
                    "promoted": "2026-01-01",
                    "metric": "x",
                    "note": "y",
                }
            }
        )
    )
    return reg


def test_is_champion_ref():
    assert is_champion_ref("champion:plant")
    assert not is_champion_ref("configs/p.json")
    assert not is_champion_ref({"champion": "plant"})


def test_load_and_resolve(tmp_path):
    reg = _write(tmp_path)
    assert set(load_champions(reg)) == {"plant"}
    champ = resolve_champion("champion:plant", reg)
    assert champ.model == "m"
    cfg, ckpt = resolved_paths(champ, reg)
    assert cfg.endswith("configs/p.json") and ckpt.endswith("runs/p.pth")


def test_resolve_unknown_role_errors(tmp_path):
    with pytest.raises(ValueError):
        resolve_champion("champion:controller", _write(tmp_path))


def test_missing_registry_is_empty(tmp_path):
    assert load_champions(tmp_path / "nope.json") == {}


def test_promote_adds_and_replaces(tmp_path):
    reg = _write(tmp_path)
    promote(
        reg,
        Champion(
            "controller",
            "c",
            "configs/c.json",
            "runs/c.pth",
            promoted="2026-02",
        ),
    )
    champs = load_champions(reg)
    assert set(champs) == {"plant", "controller"}  # existing role preserved
    promote(reg, Champion("plant", "m2", "configs/p2.json", "runs/p2.pth"))
    assert load_champions(reg)["plant"].model == "m2"  # existing role replaced


def test_registry_path_sits_beside_system(tmp_path):
    (tmp_path / "system.toml").write_text("")
    assert (
        registry_path(tmp_path / "system.toml") == tmp_path / "champions.json"
    )


def test_controller_config_resolves_champion_plant():
    # The shipped generalist config references champion:plant, it must
    # resolve via the registry to the plant's config + checkpoint.
    from nn_motion_control.control.config import ControllerConfig

    cfg = ControllerConfig(
        str(REPO / "examples/deltabot/configs/controller_10k_general.json")
    )
    assert cfg.plant_config_path.endswith("configs/plant_tcn_rollout.json")
    assert cfg.plant_checkpoint.endswith(
        "runs/deltabot_plant_tcn_rollout/deltabot_plant_tcn_rollout.pth"
    )
