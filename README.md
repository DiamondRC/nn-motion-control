[![CI](https://github.com/DiamondRC/nn-motion-control/actions/workflows/ci.yml/badge.svg)](https://github.com/DiamondRC/nn-motion-control/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)

# nn_motion_control

A framework for building **neural-network motion controllers that deploy to FPGA**.

High-precision motion stages are dominated by non-linear, history-dependent effects
(hysteresis, resonance) that classical controllers struggle with. This project learns
the controller instead: first a neural **plant model** that captures the stage's
non-linear dynamics, then a compact **controller** trained against that model and
deployed to an FPGA for real-time, nm-scale control.

The framework is **instance-agnostic** — a motion system is described declaratively by a
`SystemSpec`, and `examples/deltabot/` is the first reference instance (a 3-axis stage).

> **Status:** early development; the plant → controller → FPGA-export pipeline is being
> built up in milestones.

## How it works
- **`SystemSpec`** (`examples/*/system.toml`) — axes, channels, sample/clock rates,
  ranges, per-axis safety limits. The data schema and model I/O all derive from it.
- **Artifact configs** (`examples/*/configs/*.json`) — one JSON per trainable model; it
  references a `SystemSpec` and a JSON-defined architecture.
- **Pipeline** — HDF5 ingest → windowed, leakage-aware dataset → configurable model →
  trainer (mixed precision, early stopping) → evaluation, with provenance-rich checkpoints.

## Getting started
Runs in the dev container (PyTorch is provided by the NVIDIA base image).

```bash
# Train + evaluate a model from its config
python -m nn_motion_control model examples/deltabot/configs/plant_tcn.json

# Run the tests
pytest
```

## Development
- `uv` manages the environment (`uv sync`). `ruff` (lint + format), `pyright` (types) and
  `prettier` (JSON) run via `pre-commit` and CI.
- Build a training dataset from raw logs: `python -m nn_motion_control.data.ingest --help`

## License
Apache-2.0
