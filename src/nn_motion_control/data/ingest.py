"""
Build a training dataset (HDF5, schema v2).

Each raw text file recording is parsed into aligned (state, next-state) rows:
 - Positions are centred per file
 - DAC demands are shifted by one step so DAC[i] is the demand that
   produced pos[i]
 - Velocity/acceleration/jerk are derived by successive finite
   differences.

Targets are stored in two forms so an artifact config can select either:
the absolute next state ('*_nxt') and the change from the current step
('*_delta' = next - current).

The rows from every file are concatenated into one HDF5 file that stores raw
(un-normalised) features plus per-file boundary and provenance metadata.

Normalisation deliberately lives in the training pipeline.

Run from console:
    python -m nn_motion_control.data.ingest \
        --data-dir ./data/ --output ./data/plant_dataset.h5
"""

from __future__ import annotations

import argparse
import logging
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import h5py
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 2

_DEFAULT_STORAGE_DTYPE = np.dtype(np.float32)

# Columns as they appear in the raw whitespace-separated logs.
RAW_COLUMNS = [
    "timestep",
    "x_input",
    "x_DAC_real",
    "y_input",
    "y_DAC_real",
    "z_input",
    "z_DAC_real",
    "x_pos",
    "y_pos",
    "z_pos",
]

# Columns kept from the raw file (the *_input demand columns are discarded).
KEEP_COLUMNS = [
    "timestep",
    "x_DAC_real",
    "y_DAC_real",
    "z_DAC_real",
    "x_pos",
    "y_pos",
    "z_pos",
]

POS_COLS = ("x_pos", "y_pos", "z_pos")
DAC_COLS = ("x_DAC_real", "y_DAC_real", "z_DAC_real")

INPUT_LABELS = [
    "timestep",
    "x_pos",
    "x_vel",
    "x_acc",
    "x_jer",
    "y_pos",
    "y_vel",
    "y_acc",
    "y_jer",
    "z_pos",
    "z_vel",
    "z_acc",
    "z_jer",
    "x_DAC_real",
    "y_DAC_real",
    "z_DAC_real",
]

# Per-axis state channels a plant can predict.
STATE_LABELS = [
    "x_pos",
    "x_vel",
    "x_acc",
    "x_jer",
    "y_pos",
    "y_vel",
    "y_acc",
    "y_jer",
    "z_pos",
    "z_vel",
    "z_acc",
    "z_jer",
]

NXT_LABELS = ["timestep_nxt", *(f"{c}_nxt" for c in STATE_LABELS)]
DELTA_LABELS = [f"{c}_delta" for c in STATE_LABELS]
TARGET_LABELS = [*NXT_LABELS, *DELTA_LABELS]

# Rows lost per file during alignment:
# 1 (DAC shift) + 3 (V/A/J warm-up) + 1 (next-state).
ROWS_LOST_PER_FILE = 5

# Guards diff(timestep) against zero/negative dt before it is divided into.
_DT_EPS = 1e-9

# Rows per HDF5 chunk for the inputs/targets datasets.
_HDF5_CHUNK_ROWS = 8192


@dataclass(frozen=True)
class FileResult:
    """
    Aligned features for a single recording.
    """

    name: str
    inputs: np.ndarray  # [n, len(INPUT_LABELS)]
    targets: np.ndarray  # [n, len(TARGET_LABELS)]
    pos_offset: np.ndarray  # [3] float64 (per-file centring means)
    n_rows: int


def discover_files(data_dir: Path, pattern: str = "*.txt") -> list[Path]:
    """
    Return matching files in deterministic (sorted) order.
    """

    files = sorted(data_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching {pattern!r} in {data_dir}")

    return files


def read_raw_frame(path: Path) -> pd.DataFrame:
    """
    Parse and validate a single raw log into the kept columns.
    """

    # Peek the first non-empty line and confirm it is 10 numeric fields.
    first = ""
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                first = line
                break
    fields = first.split()
    if len(fields) != len(RAW_COLUMNS):
        raise ValueError(
            f"{path}: expected {len(RAW_COLUMNS)} "
            f"whitespace-separated columns, "
            f"got {len(fields)} on the first data line"
        )
    try:
        [float(x) for x in fields]
    except ValueError as exc:
        raise ValueError(
            f"{path}: first line is not numeric (stray header?): "
            f"{first.strip()!r}"
        ) from exc

    df = cast(
        pd.DataFrame,
        pd.read_csv(path, sep=r"\s+", header=None, names=RAW_COLUMNS)[
            KEEP_COLUMNS
        ],
    )
    if df.isnull().to_numpy().any():
        raise ValueError(
            f"{path}: contains missing/non-numeric values after parsing"
        )

    ts = df["timestep"].to_numpy()
    steps = np.diff(ts)
    if steps.size and not np.all(steps > 0):
        n_bad = int((steps <= 0).sum())
        logger.warning(
            "%s: timestep not strictly increasing (%d non-positive steps); "
            "dt will be clipped to avoid divide-by-zero",
            path.name,
            n_bad,
        )

    return df


def derive_pvaj(
    pos_xyz: np.ndarray, dt_safe: np.ndarray
) -> dict[str, np.ndarray]:
    """
    Velocity/acceleration/jerk by successive finite differences.
    """

    vel_u = np.diff(pos_xyz, axis=1) / dt_safe  # [3, n-1]
    acc_u = np.diff(vel_u, axis=1) / dt_safe[:-1]  # [3, n-2]
    jer_u = np.diff(acc_u, axis=1) / dt_safe[:-2]  # [3, n-3]

    vel = np.pad(vel_u, ((0, 0), (1, 0)), constant_values=np.nan)
    acc = np.pad(acc_u, ((0, 0), (2, 0)), constant_values=np.nan)
    jer = np.pad(jer_u, ((0, 0), (3, 0)), constant_values=np.nan)

    return {"vel": vel, "acc": acc, "jer": jer}


def align_rows(
    timestep: np.ndarray, pos_xyz: np.ndarray, dac_xyz: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Aligned (state, next-state) rows from one recording, in float64.

    'pos_xyz' and 'dac_xyz' are [3, n] sampled at 'timestep' [n]. The DAC
    is shifted by one step so DAC[i] is the demand that produced pos[i],
    velocity/acceleration/jerk are successive finite differences, and the
    targets are the next state ('*_nxt') and its change ('*_delta').
    Returns (inputs, targets) laid out as INPUT_LABELS / TARGET_LABELS with
    'n - ROWS_LOST_PER_FILE' rows.
    """

    ts = np.asarray(timestep, dtype=np.int64)[1:]
    pos = np.asarray(pos_xyz, dtype=np.float64)[:, 1:]
    dac = np.asarray(dac_xyz, dtype=np.float64)[:, :-1]
    dt_safe = np.clip(np.diff(ts), _DT_EPS, None)
    pvaj = derive_pvaj(pos, dt_safe)
    state = np.stack(
        [pos, pvaj["vel"], pvaj["acc"], pvaj["jer"]], axis=1
    ).reshape(len(STATE_LABELS), -1)
    rel = (ts - ts[0]).astype(np.float64)

    # Rows [3, m - 1) are current states with a full V/A/J history and a
    # next state, row k + 1 is the next state of row k.
    m = pos.shape[1]
    cur, nxt = slice(3, m - 1), slice(4, m)
    inputs = np.concatenate([rel[None, cur], state[:, cur], dac[:, cur]]).T
    targets = np.concatenate(
        [rel[None, nxt], state[:, nxt], state[:, nxt] - state[:, cur]]
    ).T

    return inputs, targets


def parse_raw_file(path: Path, storage_dtype: np.dtype) -> FileResult:
    """
    Full per-file pipeline: raw log -> aligned (state, next-state) rows.
    """

    df = read_raw_frame(path)
    n_raw = len(df)

    # Centre positions at the origin, keep the offset so absolute positions
    # can be reconstructed later.
    pos_offset = df[list(POS_COLS)].mean().to_numpy(dtype=np.float64)
    pos = df[list(POS_COLS)].to_numpy(dtype=np.float64).T - pos_offset[:, None]
    dac = df[list(DAC_COLS)].to_numpy(dtype=np.float64).T
    aligned_in, aligned_tg = align_rows(df["timestep"].to_numpy(), pos, dac)

    n_out = len(aligned_in)
    expected = n_raw - ROWS_LOST_PER_FILE
    if n_out != expected:
        raise ValueError(
            f"{path}: alignment produced {n_out} rows, expected {expected} "
            f"(n_raw={n_raw} - {ROWS_LOST_PER_FILE})"
        )

    inputs = aligned_in.astype(storage_dtype)
    targets = aligned_tg.astype(storage_dtype)
    if not (np.isfinite(inputs).all() and np.isfinite(targets).all()):
        raise ValueError(f"{path}: non-finite values remain after alignment")

    return FileResult(
        name=path.name,
        inputs=inputs,
        targets=targets,
        pos_offset=pos_offset,
        n_rows=n_out,
    )


def _git_commit() -> str:
    """
    Best-effort current git commit for provenance,
    'unknown' if unavailable.
    """

    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except (subprocess.SubprocessError, OSError):
        return "unknown"


def build_dataset(
    data_dir: Path,
    output_file: Path,
    pattern: str = "*.txt",
    storage_dtype: np.dtype = _DEFAULT_STORAGE_DTYPE,
    overwrite: bool = True,
    compression: str | None = "lzf",
) -> None:
    """
    Stream every matching raw file into one schema-v2 HDF5 dataset.
    """

    data_dir = Path(data_dir)
    output_file = Path(output_file)
    storage_dtype = np.dtype(storage_dtype)

    files = discover_files(data_dir, pattern)
    logger.info(
        "Found %d file(s) matching %r in %s", len(files), pattern, data_dir
    )

    if output_file.exists():
        if not overwrite:
            raise FileExistsError(
                f"{output_file} already exists. Pass --overwrite to replace it"
            )
        output_file.unlink()

    n_in, n_tgt = len(INPUT_LABELS), len(TARGET_LABELS)
    offsets = [0]
    file_names: list[str] = []
    row_counts: list[int] = []
    pos_offsets: list[np.ndarray] = []
    total = 0

    with h5py.File(output_file, "w") as f:
        dset_in = f.create_dataset(
            "inputs",
            shape=(0, n_in),
            maxshape=(None, n_in),
            dtype=storage_dtype,
            chunks=(_HDF5_CHUNK_ROWS, n_in),
            compression=compression,
        )
        dset_tg = f.create_dataset(
            "targets",
            shape=(0, n_tgt),
            maxshape=(None, n_tgt),
            dtype=storage_dtype,
            chunks=(_HDF5_CHUNK_ROWS, n_tgt),
            compression=compression,
        )

        for path in files:
            res = parse_raw_file(path, storage_dtype)
            new_total = total + res.n_rows
            dset_in.resize(new_total, axis=0)
            dset_in[total:new_total] = res.inputs
            dset_tg.resize(new_total, axis=0)
            dset_tg[total:new_total] = res.targets
            total = new_total
            offsets.append(total)
            file_names.append(res.name)
            row_counts.append(res.n_rows)
            pos_offsets.append(res.pos_offset)
            logger.info("  %-32s -> %8d rows", res.name, res.n_rows)

        if total == 0:
            raise ValueError("No rows were produced from any input file")

        f.create_dataset(
            "segment_offsets", data=np.asarray(offsets, dtype=np.int64)
        )
        f.create_dataset("file_names", data=file_names)
        f.create_dataset(
            "file_row_counts", data=np.asarray(row_counts, dtype=np.int64)
        )
        f.create_dataset(
            "file_position_offsets",
            data=np.asarray(pos_offsets, dtype=np.float64),
        )
        f.create_dataset("input_labels", data=INPUT_LABELS)
        f.create_dataset("target_labels", data=TARGET_LABELS)

        f.attrs["schema_version"] = SCHEMA_VERSION
        f.attrs["created_utc"] = datetime.now(UTC).isoformat()
        f.attrs["source_dir"] = str(data_dir)
        f.attrs["git_commit"] = _git_commit()
        f.attrs["n_files"] = len(files)
        f.attrs["n_rows"] = total
        f.attrs["storage_dtype"] = storage_dtype.name

    logger.info(
        "Wrote %s: %d rows from %d file(s) (schema v%d)",
        output_file,
        total,
        len(files),
        SCHEMA_VERSION,
    )


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    parser = argparse.ArgumentParser(
        description=(
            "Build the PVAJ->next-state training dataset (HDF5 schema v2)."
        )
    )
    parser.add_argument("--data-dir", type=Path, default=Path("./data/"))
    parser.add_argument(
        "--output", type=Path, default=Path("./data/plant_dataset.h5")
    )
    parser.add_argument("--pattern", default="*.txt")
    parser.add_argument(
        "--storage-dtype", default="float32", choices=["float32", "float64"]
    )
    parser.add_argument(
        "--compression",
        default="lzf",
        choices=["lzf", "gzip", "none"],
        help="HDF5 chunk compression (none disables it)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    parser.add_argument(
        "--overwrite", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=args.log_level, format="%(levelname)s %(name)s: %(message)s"
    )
    build_dataset(
        data_dir=args.data_dir,
        output_file=args.output,
        pattern=args.pattern,
        storage_dtype=np.dtype(args.storage_dtype),
        overwrite=args.overwrite,
        compression=None if args.compression == "none" else args.compression,
    )

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
