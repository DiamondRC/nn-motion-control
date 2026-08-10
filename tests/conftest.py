"""Shared test fixtures.

All fixtures build small synthetic data so the suite is fast and needs neither the
real ~10M-row dataset nor a GPU.
"""

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from nn_motion_control.data.ingest import (
    INPUT_LABELS,
    TARGET_LABELS,
)

# The 15/12 features a model actually consumes (everything except the timestep index).
# TARGETS_12 is the absolute next-state subset; the targets dataset also carries the
# parallel `_delta` columns, which these fixtures do not exercise.
INPUTS_15 = [lbl for lbl in INPUT_LABELS if lbl != "timestep"]
TARGETS_12 = [
    lbl for lbl in TARGET_LABELS if lbl.endswith("_nxt") and lbl != "timestep_nxt"
]

# The deltabot SystemSpec: its channel labels match the synthetic dataset's columns,
# so the same spec drives both the real instance and the hermetic tests.
DELTABOT_SYSTEM = Path(__file__).resolve().parents[1] / "examples/deltabot/system.toml"


@pytest.fixture
def synth_h5(tmp_path):
    """A small schema-v2 dataset with the real label names and two segments."""
    rng = np.random.default_rng(0)
    seg = [60, 40]
    n = sum(seg)
    n_in, n_tgt = len(INPUT_LABELS), len(TARGET_LABELS)
    inputs = rng.normal(size=(n, n_in)).astype("float32")
    inputs[:, 0] = np.arange(n)  # timestep counter
    targets = rng.normal(size=(n, n_tgt)).astype("float32")
    targets[:, 0] = inputs[:, 0] + 1  # timestep_nxt
    offsets = np.concatenate([[0], np.cumsum(seg)]).astype("int64")

    path = tmp_path / "synth.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("inputs", data=inputs)
        f.create_dataset("targets", data=targets)
        f.create_dataset("segment_offsets", data=offsets)
        f.create_dataset("input_labels", data=list(INPUT_LABELS))
        f.create_dataset("target_labels", data=list(TARGET_LABELS))
        f.attrs["schema_version"] = 2

    return {
        "path": str(path),
        "inputs": inputs,
        "targets": targets,
        "offsets": offsets,
    }


@pytest.fixture
def config_factory(tmp_path, synth_h5):
    """Return a factory that writes an artifact-config JSON (new schema).

    ``window_size``, ``hidden_layers`` and a ``training`` dict may be overridden;
    remaining kwargs override top-level keys.
    """

    def _make(**overrides):
        arch = {
            "window_size": overrides.pop("window_size", 4),
            "hidden_layers": overrides.pop(
                "hidden_layers", [{"Linear": [15, 16]}, "ReLU", {"Linear": [16, 12]}]
            ),
        }
        training = {
            "batch_size": 64,
            "max_epochs": 2,
            "patience": 5,
            "min_delta": 1e-5,
            "learning_rate": 1e-3,
            "optimiser": "AdamW",
            "grad_scaler": "GradScaler",
            "loss_function": "WeightedMSELoss",
            "training_dtype": "float32",
            "accumulation_steps": 1,
            "train_ratio": 0.8,
            "validation_ratio": 0.1,
            "test_display_num": 2,
            "dataloader": {
                "auto_tune": False,
                "cpu_util": 50,
                "num_workers": 0,
                "prefetch_factor": 2,
            },
        }
        training.update(overrides.pop("training", {}))
        cfg = {
            "system": str(DELTABOT_SYSTEM),
            "role": "plant",
            "model_name": "TestModel",
            "architecture": arch,
            "inputs": ["position", "velocity", "acceleration", "jerk", "dac"],
            "targets": {
                ch: {"predict": "next", "weight": 1}
                for ch in ("position", "velocity", "acceleration", "jerk")
            },
            "data": synth_h5["path"],
            "training": training,
            "run": {
                "seed": 42,
                "out_dir": str(tmp_path / "runs"),
                "verbose_logging": False,
            },
        }
        cfg.update(overrides)
        (tmp_path / "runs").mkdir(exist_ok=True)
        path = tmp_path / "config.json"
        path.write_text(json.dumps(cfg, indent=2))
        return str(path)

    return _make
