"""
Build a training dataset (HDF5, schema v3) from open-loop PandA captures.

A JSON build spec names the capture sources (globs under a runs root), the
split each one feeds, an optional per-record length cap and the target
rate. Each capture is read in nm and DAC counts (data/panda.py), resampled
to the rate, referenced to its own rest position and aligned into the same
(state, next-state) rows as schema v2 (ingest.align_rows), so every v2
reader still works.

Schema v3 adds, per segment (one segment per capture): its split, kind,
group, source run, driven axes, level, band and peak command. Segments are
stored train first, then val, then test. A capture flagged as touching an
end stop, or whose excursion comes within a margin of a measured stop, is
kept apart in the 'contact' group, never in a split, and the measured
frequency responses are stored under 'frf'.

Run from console:
    python -m nn_motion_control build-dataset \
        examples/deltabot/datasets/panda_20k.json
"""

from __future__ import annotations

import fnmatch
import hashlib
import inspect
import json
import logging
import sys
from collections import defaultdict
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from nn_motion_control.data import ingest, panda
from nn_motion_control.data.ingest import (
    INPUT_LABELS,
    ROWS_LOST_PER_FILE,
    TARGET_LABELS,
    _git_commit,
    align_rows,
)
from nn_motion_control.data.panda import AntiAlias, read_capture, resample

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 3

AXES = ("x", "y", "z")
SPLITS = ("train", "val", "test")
CONTACT_GROUP = "contact"
FRF_GROUP = "frf"

NM_PER_UM = 1000.0

_STORAGE_DTYPE = np.dtype(np.float32)
_HDF5_CHUNK_ROWS = 8192

# Default length of the leading window whose mean position is the record's
# rest reference.
_DEFAULT_REST_SECONDS = 0.02

# A travel hold counts toward the static gain once its command has held this
# long, the second half of the hold (after the resonance has rung down) is
# averaged.
HOLD_MIN_SECONDS = 0.2


@dataclass(frozen=True)
class Record:
    """
    One capture and what it was: the metadata a segment is tagged with.
    """

    path: Path
    name: str  # path relative to the runs root
    source: str  # the run directory, relative to the runs root
    kind: str
    group: str
    pgens: dict[str, str]
    level: float
    band_hz: tuple[float, float]
    contact: bool
    split: str = ""
    max_seconds: float | None = None

    @property
    def axes(self) -> str:
        return "".join(a for a in AXES if a in self.pgens)


@dataclass(frozen=True)
class Source:
    glob: str
    split: str | list[str]
    max_seconds: float | None = None
    exclude: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class BuildSpec:
    """
    A parsed build spec. Relative paths resolve against the spec's folder.
    """

    path: Path
    runs_root: Path
    output: Path
    rate_hz: float
    drive_rate_hz: float
    capture_rate_hz: float
    resolution_nm: float
    sources: list[Source]
    exclude: list[str] = field(default_factory=list)
    anti_alias: AntiAlias | None = None
    rest_seconds: float = _DEFAULT_REST_SECONDS
    stops_um: dict[str, list[float]] = field(default_factory=dict)
    stop_margin_um: float = 0.0
    frf: dict[str, str] = field(default_factory=dict)
    seed: int = 0
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_json(cls, path: str | Path) -> BuildSpec:
        path = Path(path).resolve()
        raw = json.loads(path.read_text())
        base = path.parent

        def split_of(value: Any) -> str | list[str]:
            names = [value] if isinstance(value, str) else list(value)
            bad = [s for s in names if s not in SPLITS]
            if bad or not names:
                raise ValueError(
                    f"{path}: split must be one of {SPLITS} or a list of "
                    f"them, got {value!r}"
                )
            return value

        sources = [
            Source(
                glob=s["glob"],
                split=split_of(s["split"]),
                max_seconds=s.get("max_seconds"),
                exclude=list(s.get("exclude", [])),
            )
            for s in raw["sources"]
        ]
        aa = raw.get("anti_alias")

        return cls(
            path=path,
            runs_root=(base / raw["runs_root"]).resolve(),
            output=(base / raw["output"]).resolve(),
            rate_hz=float(raw["rate_hz"]),
            drive_rate_hz=float(raw["drive_rate_hz"]),
            capture_rate_hz=float(raw["capture_rate_hz"]),
            resolution_nm=float(raw["resolution_nm"]),
            sources=sources,
            exclude=list(raw.get("exclude", [])),
            anti_alias=AntiAlias(**aa) if aa else None,
            rest_seconds=float(raw.get("rest_seconds", _DEFAULT_REST_SECONDS)),
            stops_um={k: list(v) for k, v in raw.get("stops_um", {}).items()},
            stop_margin_um=float(raw.get("stop_margin_um", 0.0)),
            frf=dict(raw.get("frf", {})),
            seed=int(raw.get("seed", 0)),
            raw=raw,
        )


def _load_json(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text()) if path.exists() else None


def _band_of(meta: Mapping[str, Any]) -> tuple[float, float]:
    lo, hi = meta.get("f_start"), meta.get("f_end")
    if lo is None or hi is None:
        return (np.nan, np.nan)

    return (float(lo), float(hi))


def describe(path: Path, runs_root: Path) -> Record:
    """
    Tag a capture from its sidecar json, or its folder's manifest.

    A campaign record or travel level has a same-stem sidecar, a profile
    band capture is described by the 'manifest.json' beside it (the axis
    and design) and the run's 'profile.json' (the band edges).
    """

    name = path.relative_to(runs_root).as_posix()
    source = "/".join(name.split("/")[:2])
    side = _load_json(path.with_suffix(".json"))

    if side is not None and "commands" in side:
        return Record(
            path=path,
            name=name,
            source=source,
            kind="travel",
            group=path.parent.name,
            pgens=dict(side["pgens"]),
            level=np.nan,
            band_hz=(np.nan, np.nan),
            contact=bool(side.get("contact")),
        )
    if side is not None:
        pgens = side.get("pgens") or {side["axis"]: "PGEN1"}
        return Record(
            path=path,
            name=name,
            source=source,
            kind=str(side["kind"]),
            group=str(side.get("band") or side["segment"]),
            pgens=dict(pgens),
            level=float(side.get("level", np.nan)),
            band_hz=_band_of(side),
            contact=False,
        )

    manifest = _load_json(path.parent / "manifest.json")
    if manifest is None:
        raise ValueError(f"{path}: no sidecar json or manifest describes it")
    axis, _, stem = path.stem.partition("_")
    entry = manifest["axes"][axis][stem]
    band = path.parent.name
    profile = _load_json(path.parents[1] / "profile.json")
    design = (profile or {}).get("bands", {}).get(band, {}).get("multisine")

    return Record(
        path=path,
        name=name,
        source=source,
        kind=str(entry["kind"]),
        group=band,
        pgens={entry["axis"]: "PGEN1"},
        level=np.nan,
        band_hz=_band_of(design or {}),
        contact=False,
    )


def discover(spec: BuildSpec) -> list[Record]:
    """
    Every capture the spec selects, tagged and assigned a split.

    A list-valued split is dealt within each (kind, group) of that source:
    the group is shuffled with the spec's seed and dealt round-robin, the
    first split rotating from group to group, so each split sees every
    kind of record and a lone record does not always land in the same one.
    Paths are resolved, so a capture reached again through a run alias
    such as 'latest' is taken once, and one matched by two sources is an
    error.
    """

    records: list[Record] = []
    seen: dict[Path, str] = {}
    rng = np.random.default_rng(spec.seed)

    for src in spec.sources:
        excludes = [*spec.exclude, *src.exclude]
        paths = [
            p
            for p in sorted(spec.runs_root.glob(src.glob))
            if not any(
                fnmatch.fnmatch(p.relative_to(spec.runs_root).as_posix(), x)
                for x in excludes
            )
        ]
        if not paths:
            raise FileNotFoundError(
                f"Source {src.glob!r} matches no captures in {spec.runs_root}"
            )
        groups: dict[tuple[str, str], list[Record]] = defaultdict(list)

        for p in dict.fromkeys(q.resolve() for q in paths):
            if p in seen:
                raise ValueError(
                    f"{p} is matched by both {seen[p]!r} and {src.glob!r}"
                )
            seen[p] = src.glob
            rec = replace(
                describe(p, spec.runs_root), max_seconds=src.max_seconds
            )
            groups[(rec.kind, rec.group)].append(rec)

        for k, key in enumerate(sorted(groups)):
            if isinstance(src.split, str):
                records += [replace(r, split=src.split) for r in groups[key]]
                continue
            names = src.split
            order = rng.permutation(len(groups[key]))

            for j, i in enumerate(order):
                split = names[(k + j) % len(names)]
                records.append(replace(groups[key][i], split=split))

    return records


@dataclass(frozen=True)
class Segment:
    record: Record
    inputs: np.ndarray
    targets: np.ndarray
    rest_nm: np.ndarray
    peak_command: np.ndarray
    excursion_nm: np.ndarray  # [A, 2] min and max rest-referenced position


def process(rec: Record, spec: BuildSpec) -> Segment:
    """
    One capture to aligned rows at the spec's rate, rest-referenced.
    """

    max_samples = (
        None
        if rec.max_seconds is None
        else int(round(rec.max_seconds * spec.capture_rate_hz))
    )
    cap = read_capture(
        rec.path,
        rec.pgens,
        AXES,
        spec.resolution_nm,
        spec.capture_rate_hz,
        max_samples,
    )
    pos, cmd = resample(cap, spec.rate_hz, spec.drive_rate_hz, spec.anti_alias)
    n_rest = max(1, int(round(spec.rest_seconds * spec.rate_hz)))
    if pos.shape[1] < n_rest + ROWS_LOST_PER_FILE:
        raise ValueError(f"{rec.path}: too short for the rest window")
    rest = pos[:, :n_rest].mean(axis=1)
    pos = pos - rest[:, None]
    inputs, targets = align_rows(np.arange(pos.shape[1]), pos, cmd)

    return Segment(
        record=rec,
        inputs=inputs.astype(_STORAGE_DTYPE),
        targets=targets.astype(_STORAGE_DTYPE),
        rest_nm=rest,
        peak_command=np.abs(cmd).max(axis=1),
        excursion_nm=np.stack([pos.min(axis=1), pos.max(axis=1)], axis=1),
    )


def near_stop(seg: Segment, spec: BuildSpec) -> bool:
    """
    Whether the capture came within the margin of a measured end stop.

    The janus contact flag only fires once a level falls short of the
    spring line, so a level resting on the stop can pass unflagged, the
    recorded excursion catches it.
    """

    margin = spec.stop_margin_um

    for i, axis in enumerate(AXES):
        if axis not in spec.stops_um:
            continue
        lo, hi = spec.stops_um[axis]
        lo_um, hi_um = seg.excursion_nm[i] / NM_PER_UM
        if lo_um <= lo + margin or hi_um >= hi - margin:
            return True

    return False


def _ordered(records: list[Record]) -> Iterator[Record]:
    """
    Split-major order: every train capture, then val, then test.
    """

    for split in SPLITS:
        yield from (r for r in records if r.split == split)


class _RowWriter:
    """
    Appends segments to an inputs/targets pair and collects their metadata.
    """

    def __init__(self, group: h5py.Group, compression: str | None):
        n_in, n_tg = len(INPUT_LABELS), len(TARGET_LABELS)
        self.group = group
        self.inputs = group.create_dataset(
            "inputs",
            shape=(0, n_in),
            maxshape=(None, n_in),
            dtype=_STORAGE_DTYPE,
            chunks=(_HDF5_CHUNK_ROWS, n_in),
            compression=compression,
        )
        self.targets = group.create_dataset(
            "targets",
            shape=(0, n_tg),
            maxshape=(None, n_tg),
            dtype=_STORAGE_DTYPE,
            chunks=(_HDF5_CHUNK_ROWS, n_tg),
            compression=compression,
        )
        self.offsets = [0]
        self.segments: list[Segment] = []

    def append(self, seg: Segment) -> None:
        lo = self.offsets[-1]
        hi = lo + len(seg.inputs)
        self.inputs.resize(hi, axis=0)
        self.inputs[lo:hi] = seg.inputs
        self.targets.resize(hi, axis=0)
        self.targets[lo:hi] = seg.targets
        self.offsets.append(hi)
        # Keep the metadata only, the rows are on disk.
        self.segments.append(
            replace(seg, inputs=seg.inputs[:0], targets=seg.targets[:0])
        )

    def finish(self) -> None:
        g, segs = self.group, self.segments
        recs = [s.record for s in segs]
        g.create_dataset("segment_offsets", data=np.asarray(self.offsets))
        g.create_dataset("file_names", data=[r.name for r in recs])
        g.create_dataset(
            "file_row_counts", data=np.diff(np.asarray(self.offsets))
        )
        g.create_dataset(
            "file_position_offsets",
            data=np.asarray([s.rest_nm for s in segs]).reshape(-1, len(AXES)),
        )
        g.create_dataset("segment_split", data=[r.split for r in recs])
        g.create_dataset("segment_kind", data=[r.kind for r in recs])
        g.create_dataset("segment_group", data=[r.group for r in recs])
        g.create_dataset("segment_source", data=[r.source for r in recs])
        g.create_dataset("segment_axes", data=[r.axes for r in recs])
        g.create_dataset(
            "segment_level", data=np.asarray([r.level for r in recs])
        )
        g.create_dataset(
            "segment_band_hz",
            data=np.asarray([r.band_hz for r in recs]).reshape(-1, 2),
        )
        g.create_dataset(
            "segment_peak_command",
            data=np.asarray([s.peak_command for s in segs]).reshape(
                -1, len(AXES)
            ),
        )
        g.create_dataset("input_labels", data=INPUT_LABELS)
        g.create_dataset("target_labels", data=TARGET_LABELS)


def write_frf(f: h5py.File, spec: BuildSpec) -> None:
    """
    Store each driven axis's measured FRF (all three responses).

    The FRFs come from the full-rate captures, so in a build low-passed
    below the drive rate the lines above the anti-alias band describe the
    stage, not the stored rows.
    """

    for axis, rel in spec.frf.items():
        src = spec.runs_root / rel
        d = np.load(src)
        g = f.create_group(f"{FRF_GROUP}/{axis}")
        g.create_dataset("freqs", data=d["freqs"])
        g.create_dataset("H", data=np.stack([d[f"H_{a}"] for a in AXES]))
        g.create_dataset(
            "snr_db", data=np.stack([d[f"snr_db_{a}"] for a in AXES])
        )
        g.attrs["source"] = rel
        g.attrs["unit"] = "nm/count"
        g.attrs["reference"] = "command at capture time, no DAC shift"
        if spec.anti_alias and spec.rate_hz < spec.drive_rate_hz:
            g.attrs["rows_low_passed_above_hz"] = spec.anti_alias.passband_hz


def code_digest() -> str:
    """
    SHA-256 over the source files that turn captures into rows.

    The git commit alone does not pin the code when the tree is dirty.
    """

    h = hashlib.sha256()

    for module in (panda, ingest, sys.modules[__name__]):
        h.update(Path(inspect.getfile(module)).read_bytes())

    return h.hexdigest()


def held_points(seg: Segment, rate_hz: float) -> list[np.ndarray]:
    """
    Per axis, (command, settled position) of every non-zero command hold.

    Holds come from the stored rows, so the position is rest-referenced
    and the command is the DAC that produced it.
    """

    labels = INPUT_LABELS
    min_rows = int(HOLD_MIN_SECONDS * rate_hz)
    out: list[np.ndarray] = []

    for axis in AXES:
        u = seg.inputs[:, labels.index(f"{axis}_DAC_real")].astype(float)
        p = seg.inputs[:, labels.index(f"{axis}_pos")].astype(float)
        edges = np.r_[0, np.flatnonzero(np.diff(u) != 0) + 1, len(u)]
        pts = [
            (u[s], p[(s + e) // 2 : e].mean())
            for s, e in zip(edges[:-1], edges[1:], strict=True)
            if e - s >= min_rows and u[s] != 0
        ]
        out.append(np.asarray(pts).reshape(-1, 2))

    return out


def static_gain(points: list[list[np.ndarray]]) -> np.ndarray:
    """
    Per-axis spring gain, nm per count, from travel holds (origin fit).

    NaN for an axis with no holds. It is what references a capture with no
    zero-command window to the rest frame.
    """

    gains = []

    for a in range(len(AXES)):
        held = [p[a] for p in points if len(p[a])]
        if not held:
            gains.append(np.nan)
            continue
        c, x = np.concatenate(held).T
        gains.append(float((c * x).sum() / (c * c).sum()))

    return np.asarray(gains)


def _check_splits(main: _RowWriter) -> None:
    present = {s.record.split for s in main.segments}
    missing = [s for s in SPLITS if s not in present]
    if missing:
        raise ValueError(
            f"No recordings left in split(s) {missing} after holding apart "
            "end-stop and lost-encoder captures"
        )


def build_capture_dataset(
    spec_path: str | Path,
    overwrite: bool = True,
    compression: str | None = "lzf",
) -> dict[str, Any]:
    """
    Build the schema-v3 dataset a spec describes, returning a summary.

    The file is written beside the output and renamed into place only once
    complete, so a failed build leaves any previous dataset intact.
    Captures whose interferometer froze are dropped and listed in the
    'excluded_lost_encoder' attribute.
    """

    spec = BuildSpec.from_json(spec_path)
    records = discover(spec)
    out = spec.output
    if out.exists() and not overwrite:
        raise FileExistsError(
            f"{out} already exists, build with overwrite to replace it"
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".partial")
    lost: list[str] = []
    holds: list[list[np.ndarray]] = []
    logger.info("%d capture(s) at %g Hz", len(records), spec.rate_hz)

    try:
        with h5py.File(tmp, "w") as f:
            main = _RowWriter(f, compression)
            side = _RowWriter(f.create_group(CONTACT_GROUP), compression)

            for rec in _ordered(records):
                try:
                    seg = process(rec, spec)
                except panda.LostEncoderError as exc:
                    lost.append(rec.name)
                    logger.warning("  lost  %s", exc)
                    continue
                if rec.contact or near_stop(seg, spec):
                    side.append(
                        replace(seg, record=replace(rec, split="contact"))
                    )
                    logger.info("  stop  %-62s held apart", rec.name)
                    continue
                if rec.kind == "travel":
                    holds.append(held_points(seg, spec.rate_hz))
                main.append(seg)
                logger.info(
                    "  %-5s %-62s %9d rows",
                    rec.split,
                    rec.name,
                    len(seg.inputs),
                )
            _check_splits(main)
            main.finish()
            side.finish()
            write_frf(f, spec)
            logger.info(
                "%d capture(s) kept, %d at an end stop held apart, %d with "
                "a lost encoder dropped",
                len(main.segments),
                len(side.segments),
                len(lost),
            )
            f.attrs.update(_attrs(spec, main, lost))
            f.attrs["static_gain_nm_per_count"] = static_gain(holds)
        tmp.replace(out)
    finally:
        tmp.unlink(missing_ok=True)

    return summarise(main.segments, main.offsets, spec.rate_hz)


def _attrs(
    spec: BuildSpec, main: _RowWriter, lost: list[str]
) -> dict[str, Any]:
    aa = spec.anti_alias
    filtered = spec.rate_hz < spec.drive_rate_hz
    resampling = (
        "zero-phase fir on command and position, then decimation"
        if filtered
        else "pick at the command update: command exact, position as captured"
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(UTC).isoformat(),
        "source_dir": str(spec.runs_root),
        "build_spec": json.dumps(spec.raw),
        "git_commit": _git_commit(),
        "code_sha256": code_digest(),
        "n_files": len(main.segments),
        "n_rows": main.offsets[-1],
        "storage_dtype": _STORAGE_DTYPE.name,
        "rate_hz": spec.rate_hz,
        "drive_rate_hz": spec.drive_rate_hz,
        "capture_rate_hz": spec.capture_rate_hz,
        "resampling": resampling,
        "anti_alias_hz": np.asarray(
            [aa.passband_hz, aa.stopband_hz] if filtered and aa else [],
            dtype=np.float64,
        ),
        "anti_alias_db": aa.attenuation_db if filtered and aa else np.nan,
        "position_unit": "nm",
        "command_unit": "counts",
        "derivative_unit": "per step",
        "position_frame": (
            f"rest: each capture minus the mean of its first "
            f"{spec.rest_seconds:g} s, file_position_offsets holds that rest "
            "in the capture's own encoder frame, which is not shared across "
            "sessions"
        ),
        "position_reference": panda.REST_REFERENCE,
        "rest_seconds": spec.rest_seconds,
        "dac_shift_steps": 1,
        "excluded_lost_encoder": json.dumps(lost),
    }


def summarise(
    segments: list[Segment], offsets: list[int], rate_hz: float
) -> dict[str, Any]:
    """
    Seconds per (split, kind), peak |command| per axis per split, rows.
    """

    rows = np.diff(np.asarray(offsets))
    seconds: dict[str, dict[str, float]] = defaultdict(dict)
    peak: dict[str, np.ndarray] = {}

    for seg, n in zip(segments, rows, strict=True):
        r = seg.record
        by_kind = seconds[r.split]
        by_kind[r.kind] = by_kind.get(r.kind, 0.0) + n / rate_hz
        peak[r.split] = np.maximum(peak.get(r.split, 0.0), seg.peak_command)

    return {
        "rows": int(rows.sum()),
        "seconds": dict(seconds),
        "peak_command": {k: v.tolist() for k, v in peak.items()},
    }
