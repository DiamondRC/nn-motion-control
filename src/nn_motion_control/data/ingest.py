"""
Build a training dataset (HDF5, schema v2).

Each raw text file recording is parsed into aligned (state, next-state) rows:
 - Positions are centred per file
 - DAC demands are shifted by one step so DAC[i] is the demand that produced pos[i]
 - Velocity/acceleration/jerk are derived by successive finite differences.

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

TARGET_LABELS = [
    "timestep_nxt",
    "x_pos_nxt",
    "x_vel_nxt",
    "x_acc_nxt",
    "x_jer_nxt",
    "y_pos_nxt",
    "y_vel_nxt",
    "y_acc_nxt",
    "y_jer_nxt",
    "z_pos_nxt",
    "z_vel_nxt",
    "z_acc_nxt",
    "z_jer_nxt",
]

# Rows lost per file during alignment:
# 1 (DAC shift) + 3 (V/A/J warm-up) + 1 (next-state).
ROWS_LOST_PER_FILE = 5


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
            f"{path}: expected {len(RAW_COLUMNS)} whitespace-separated columns, "
            f"got {len(fields)} on the first data line"
        )
    try:
        [float(x) for x in fields]
    except ValueError as exc:
        raise ValueError(
            f"{path}: first line is not numeric (stray header?): {first.strip()!r}"
        ) from exc

    df = cast(
        pd.DataFrame,
        pd.read_csv(path, sep=r"\s+", header=None, names=RAW_COLUMNS)[KEEP_COLUMNS],
    )
    if df.isnull().to_numpy().any():
        raise ValueError(f"{path}: contains missing/non-numeric values after parsing")

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


def derive_pvaj(pos_xyz: np.ndarray, dt_safe: np.ndarray) -> dict[str, np.ndarray]:
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


def parse_raw_file(path: Path, storage_dtype: np.dtype) -> FileResult:
    """
    Full per-file pipeline: raw log -> aligned (state, next-state) rows.
    """

    df = read_raw_frame(path)
    n_raw = len(df)

    # Centre positions at the origin; keep the offset so absolute positions can be
    # reconstructed later.
    pos_offset = df[list(POS_COLS)].mean().to_numpy(dtype=np.float64)
    for i, col in enumerate(POS_COLS):
        df[col] = df[col] - pos_offset[i]

    # Shift DAC demands down by one so DAC[i] is the demand that produced pos[i].
    # The first row loses its DAC and is dropped explicitly.
    for col in DAC_COLS:
        df[col] = df[col].shift(1)
    df = df.iloc[1:].reset_index(drop=True)

    # Time deltas (guarded against non-positive dt) and a relative timestep index.
    ts = df["timestep"].to_numpy(dtype=np.int64)
    dt_safe = np.clip(np.diff(ts), 1e-9, None)
    df["timestep"] = ts - ts[0]

    # Derive V/A/J - drop the first 3 warm-up rows explicitly.
    pvaj = derive_pvaj(
        np.stack([df["x_pos"], df["y_pos"], df["z_pos"]]).astype(np.float64), dt_safe
    )
    for i, axis in enumerate("xyz"):
        df[f"{axis}_vel"] = pvaj["vel"][i]
        df[f"{axis}_acc"] = pvaj["acc"][i]
        df[f"{axis}_jer"] = pvaj["jer"][i]
    df = df.iloc[3:].reset_index(drop=True)

    # Next-state targets via a -1 shift
    # The final row loses its target and is dropped explicitly.
    for out_label in TARGET_LABELS:
        base_label = out_label.replace("_nxt", "")
        df[out_label] = df[base_label].shift(-1)
    df = df.iloc[:-1].reset_index(drop=True)

    n_out = len(df)
    expected = n_raw - ROWS_LOST_PER_FILE
    if n_out != expected:
        raise ValueError(
            f"{path}: alignment produced {n_out} rows, expected {expected} "
            f"(n_raw={n_raw} - {ROWS_LOST_PER_FILE})"
        )

    inputs = df[INPUT_LABELS].to_numpy(dtype=storage_dtype)
    targets = df[TARGET_LABELS].to_numpy(dtype=storage_dtype)
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
    logger.info("Found %d file(s) matching %r in %s", len(files), pattern, data_dir)

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
            chunks=(8192, n_in),
            compression=compression,
        )
        dset_tg = f.create_dataset(
            "targets",
            shape=(0, n_tgt),
            maxshape=(None, n_tgt),
            dtype=storage_dtype,
            chunks=(8192, n_tgt),
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

        f.create_dataset("segment_offsets", data=np.asarray(offsets, dtype=np.int64))
        f.create_dataset("file_names", data=file_names)
        f.create_dataset("file_row_counts", data=np.asarray(row_counts, dtype=np.int64))
        f.create_dataset(
            "file_position_offsets", data=np.asarray(pos_offsets, dtype=np.float64)
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
        description="Build the PVAJ->next-state training dataset (HDF5 schema v2)."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("./data/"))
    parser.add_argument("--output", type=Path, default=Path("./data/plant_dataset.h5"))
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
