"""
Checkpoint bundle I/O shared by every artifact.

A checkpoint is a bundle (weights + fitted norm stats + provenance), not a
bare state_dict. Loading, schema validation, the 'weights_only' policy and
the JSON sidecar are centralised here so plant, controller and eval agree
rather than each reimplementing their own unwrap. Bundles are trusted local
files, so 'weights_only' defaults False.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


def load_checkpoint_bundle(
    path: str,
    device: str = "cpu",
    *,
    weights_only: bool = False,
    expected_schema: int | None = None,
) -> Any:
    """
    Load a checkpoint bundle, optionally asserting its 'schema_version'.

    'expected_schema' (when given) must match the bundle's 'schema_version'
    or a clear error is raised instead of a buried KeyError deep in
    reconstruction.
    """

    bundle = torch.load(path, map_location=device, weights_only=weights_only)
    if expected_schema is not None:
        got = bundle.get("schema_version") if isinstance(bundle, dict) else None
        if got != expected_schema:
            raise ValueError(
                f"Unsupported checkpoint schema at {path}: expected "
                f"{expected_schema}, got {got!r}"
            )

    return bundle


def state_dict_from(bundle: Any) -> dict:
    """
    Extract 'model_state_dict' from a bundle, tolerating a bare state_dict.
    """

    if isinstance(bundle, dict) and "model_state_dict" in bundle:
        return bundle["model_state_dict"]

    return bundle


def write_json_sidecar(
    checkpoint_path: str, fields: dict, *, suffix: str = ".norm.json"
) -> Path:
    """
    Write a tensor-free JSON sidecar beside a checkpoint for human inspection.
    """

    sidecar = Path(checkpoint_path).with_suffix(suffix)
    sidecar.write_text(json.dumps(fields, indent=2))

    return sidecar
