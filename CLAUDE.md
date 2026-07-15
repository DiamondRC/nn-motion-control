# nn_motion_control

A framework for building neural-network motion controllers that deploy to FPGA.
`examples/deltabot/` is the first reference instance (a 3-axis, nm-precision stage); the
framework itself is instance-agnostic. See `MILESTONES.md` for the roadmap
(plant model → controller → FPGA export) and current status.

## Layout
- `src/nn_motion_control/`
  - `core/` — `SystemSpec` (the system description), run-config schema, registry, checkpoints
  - `data/` — HDF5 ingest, windowed dataset, leakage-aware splits, z-score normalisation, loaders
  - `models/` — `JsonModel` (builds an `nn.Sequential` from a JSON `hidden_layers` spec) + layers
  - `training/` — trainer (AMP + early stopping), losses, run orchestration, logging
  - `eval/` — evaluation metrics (reliability checks land here from M1)
  - `plant/` · `control/` · `deploy/` — stubs for M1 / M2+ / M5
- `examples/deltabot/` — `system.toml`, `configs/*.json`; `data/` + `runs/` (gitignored)

## Key concepts
- **SystemSpec** (`examples/*/system.toml`): the "what am I controlling" description —
  axes, channels (measured/derived/command), rates, ranges, per-axis safety limits.
  Data schema, model I/O sizing and controller clamping all derive from it.
- **Config-per-artifact** (`configs/*.json`): one JSON per trainable thing (a plant, a
  controller). It references a SystemSpec and lists `inputs`/`targets` as *channel names*,
  which expand per-axis into concrete dataset labels — configs never hold column lists.
- **Checkpoints** are bundles (schema v2): weights + fitted norm stats + provenance, so
  inference recovers physical units (normalisation is fit at train time, not stored in
  the dataset).

## Run / test
- Train + test a model: `python -m nn_motion_control model examples/deltabot/configs/plant_tcn.json`
- Tests: `pytest` — hermetic (tiny synthetic data; no GPU, no real dataset needed)
- Lint / type: `ruff check` · `ruff format` · `pyright src/nn_motion_control`

## Conventions & gotchas
- **PyTorch is not a declared dependency** — the dev container provides it from the NVIDIA
  base image; CI installs the CPU wheel per-job. Do **not** add `torch` to `pyproject`.
- **JSON configs are hand-authored and Prettier-formatted** (`printWidth 100` keeps model
  specs compact). VS Code's built-in JSON formatter is disabled for JSON — don't re-enable it.
- Tooling: `uv` (env/lock), `ruff` (lint+format), `pyright`, `prettier` (JSON) — all wired
  into `pre-commit` and `.github/workflows/ci.yml`.
- Python house style: docstring summary on its own line with a blank line after;
  capitalised error/validation messages.

## Working agreement
- **Start of a session:** read `HANDOFF.md` (current state) and `MILESTONES.md` (status).
- **End of each successful task:** rewrite `HANDOFF.md` and tick `MILESTONES.md`. Keep both
  small — detailed history lives in git, not in these files.
