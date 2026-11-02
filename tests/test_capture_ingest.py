"""Capture-dataset build tests: discovery and tagging, split-major layout,
rest referencing, the contact group and the stored FRF.
"""

import json

import h5py
import numpy as np
import pytest

from nn_motion_control.data import capture_ingest as m
from nn_motion_control.data._hdf5 import as_dataset
from nn_motion_control.data.ingest import ROWS_LOST_PER_FILE

N = 4000  # capture samples per record
RES_NM = 0.256
REST_COUNTS = 1000.0  # where every synthetic stage sits at rest
RAMP_COUNTS = 300  # how far the campaign records drive x from rest


def _strings(f, key):
    return [x.decode() for x in as_dataset(f, key)[:]]


def _record(write_capture, path, axis="x", n=N, pgens=None, **side):
    """A campaign-style capture with its sidecar json."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ramp = np.round(np.linspace(0, RAMP_COUNTS, n // 2 - 200))
    # A one-count dither makes the command change every drive step, as a
    # real command does, so its update rate can be read off it.
    ramp += np.arange(len(ramp)) % 2
    table = np.r_[np.zeros(200), ramp]
    # Rest plus a one-count dither, the interferometer is live.
    counts = REST_COUNTS + np.tile(np.arange(n) % 2, (3, 1)).astype(float)
    counts[0, 1:] += np.repeat(table, 2)[: n - 1]
    blocks = dict.fromkeys((pgens or {axis: "PGEN1"}).values(), table)
    write_capture(path, blocks, counts=counts, n=n)
    meta = {"kind": "multisine", "segment": "low", "axis": axis, "level": 1.0}
    if pgens:
        meta["pgens"] = pgens
    meta.update(side)
    path.with_suffix(".json").write_text(json.dumps(meta))


@pytest.fixture
def runs(tmp_path, write_capture):
    root = tmp_path / "runs"
    a, b = root / "campaign/run_a", root / "campaign/run_b"
    _record(write_capture, a / "low/x_l1.h5", f_start=5.0, f_end=2000.0)
    _record(write_capture, a / "quiet/x_q.h5", kind="quiet", segment="quiet")
    _record(
        write_capture,
        a / "mimo/xyz_l1.h5",
        kind="mimo",
        segment="mimo",
        pgens={"x": "PGEN1", "y": "PGEN2", "z": "PGEN3"},
    )
    _record(write_capture, b / "low/x_l1.h5")
    _record(write_capture, b / "low/y_l1.h5", axis="y")
    _record(write_capture, b / "closed_loop/x_t.h5")

    # A capture whose z interferometer froze partway through.
    lost = a / "low/z_l1.h5"
    counts = np.tile(np.arange(N) % 2, (3, 1)).astype(float)
    counts[2, N // 4 :] = 3.0
    write_capture(lost, {"PGEN1": np.zeros(N // 2)}, counts=counts)
    lost.with_suffix(".json").write_text(
        json.dumps({"kind": "multisine", "segment": "low", "axis": "z"})
    )

    # A profile band capture, described by its folder's manifest.
    band = root / "profile/run_p/band_low"
    band.mkdir(parents=True)
    write_capture(band / "z_multisine.h5", {"PGEN1": np.ones(N // 2)})
    (band / "manifest.json").write_text(
        json.dumps(
            {"axes": {"z": {"multisine": {"kind": "multisine", "axis": "z"}}}}
        )
    )
    (band.parent / "profile.json").write_text(
        json.dumps(
            {"bands": {"band_low": {"multisine": {"f_start": 5, "f_end": 9}}}}
        )
    )

    # Two travel levels, the second touched the stop.
    for level, contact in (("l01", []), ("l02", ["x"])):
        p = root / f"travel/run_t/x+/x+_{level}.h5"
        p.parent.mkdir(parents=True, exist_ok=True)
        write_capture(p, {"PGEN1": np.full(N // 2, 500.0)})
        p.with_suffix(".json").write_text(
            json.dumps(
                {
                    "pgens": {"x": "PGEN1"},
                    "commands": {"x": 500.0},
                    "contact": contact,
                }
            )
        )
    # A run alias must not ingest the same captures twice.
    (root / "travel/latest").symlink_to(root / "travel/run_t")

    freqs = np.array([5.0, 10.0])
    frf = {"freqs": freqs}

    for ax in "xyz":
        frf[f"H_{ax}"] = np.array([1.0 + 0j, 2.0 + 0j])
        frf[f"snr_db_{ax}"] = np.array([30.0, 20.0])
    np.savez(root / "profile/run_p/x_stitched.npz", **frf)

    return root


def _spec(tmp_path, runs, rate_hz=20_000, **extra):
    spec = {
        "runs_root": str(runs),
        "output": str(tmp_path / "out.h5"),
        "rate_hz": rate_hz,
        "drive_rate_hz": 20_000,
        "capture_rate_hz": 40_000,
        "resolution_nm": RES_NM,
        "rest_seconds": 0.002,
        "sources": [
            {"glob": "campaign/run_a/*/*.h5", "split": "train"},
            {"glob": "campaign/run_b/*/*.h5", "split": ["val", "test"]},
            {"glob": "profile/*/band_*/*.h5", "split": "train"},
            {"glob": "travel/*/*/*.h5", "split": "train"},
        ],
        "exclude": ["*/closed_loop/*"],
        "frf": {"x": "profile/run_p/x_stitched.npz"},
    }
    spec.update(extra)
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec))

    return path


def test_discovery_tags_and_deals_splits(tmp_path, runs):
    recs = {
        r.name: r
        for r in m.discover(m.BuildSpec.from_json(_spec(tmp_path, runs)))
    }
    assert "campaign/run_b/closed_loop/x_t.h5" not in recs
    assert not any(n.startswith("travel/latest") for n in recs)
    assert recs["campaign/run_a/low/x_l1.h5"].band_hz == (5.0, 2000.0)
    assert recs["campaign/run_a/quiet/x_q.h5"].kind == "quiet"
    # Run b's two low records are dealt one to val, one to test.
    held = {recs[f"campaign/run_b/low/{a}_l1.h5"].split for a in "xy"}
    assert held == {"val", "test"}
    z = recs["profile/run_p/band_low/z_multisine.h5"]
    assert (z.axes, z.group, z.band_hz) == ("z", "band_low", (5.0, 9.0))
    mimo = recs["campaign/run_a/mimo/xyz_l1.h5"]
    assert (mimo.kind, mimo.axes) == ("mimo", "xyz")
    assert mimo.pgens == {"x": "PGEN1", "y": "PGEN2", "z": "PGEN3"}
    assert recs["travel/run_t/x+/x+_l02.h5"].contact
    assert not recs["travel/run_t/x+/x+_l01.h5"].contact


def test_build_layout_and_rest_reference(tmp_path, runs):
    summary = m.build_capture_dataset(_spec(tmp_path, runs))

    with h5py.File(tmp_path / "out.h5", "r") as f:
        assert f.attrs["schema_version"] == 3
        assert f.attrs["rate_hz"] == 20_000
        assert f.attrs["resampling"].startswith("pick at the command update")
        assert json.loads(f.attrs["excluded_lost_encoder"]) == [
            "campaign/run_a/low/z_l1.h5"
        ]
        assert len(f.attrs["code_sha256"]) == 64
        split = _strings(f, "segment_split")
        # Split-major: every train recording before val, val before test.
        order = [m.SPLITS.index(s) for s in split]
        assert order == sorted(order)
        # 3 + 2 campaign (the lost-encoder one dropped), 1 profile, 1 travel.
        assert len(split) == 7
        offsets = as_dataset(f, "segment_offsets")[:]
        rows = N // 2 - ROWS_LOST_PER_FILE - 1  # stale samples, then 2x
        assert np.all(np.diff(offsets) == rows)
        assert summary["rows"] == offsets[-1]

        # Positions are rest-referenced, the rest is kept as the offset.
        rest = as_dataset(f, "file_position_offsets")[:]
        names = _strings(f, "file_names")
        campaign = [n.startswith("campaign") for n in names]
        # The rest window sees the one-count dither, half a count on average.
        assert np.allclose(rest[campaign], REST_COUNTS * RES_NM, atol=RES_NM)
        labels = _strings(f, "input_labels")
        x0 = as_dataset(f, "inputs")[0, labels.index("x_pos")]
        assert abs(x0) < RES_NM

        peak = as_dataset(f, "segment_peak_command")[:]
        kinds = _strings(f, "segment_kind")
        assert peak[kinds.index("travel"), 0] == 500.0

        contact = _strings(f, "contact/file_names")
        assert contact == ["travel/run_t/x+/x+_l02.h5"]
        assert as_dataset(f, "contact/inputs").shape[0] == rows

        assert as_dataset(f, "frf/x/H").shape == (3, 2)
        assert f["frf/x"].attrs["unit"] == "nm/count"


def test_ten_kilohertz_build_filters(tmp_path, runs):
    spec = _spec(
        tmp_path,
        runs,
        rate_hz=10_000,
        anti_alias={
            "passband_hz": 4000,
            "stopband_hz": 5000,
            "attenuation_db": 80,
        },
    )
    m.build_capture_dataset(spec)
    with h5py.File(tmp_path / "out.h5", "r") as f:
        assert f.attrs["resampling"].startswith("zero-phase fir")
        assert list(f.attrs["anti_alias_hz"]) == [4000, 5000]
        assert f["frf/x"].attrs["rows_low_passed_above_hz"] == 4000
        offsets = as_dataset(f, "segment_offsets")[:]
        rows = np.diff(offsets)
        assert np.all(rows == rows[0]) and rows[0] > 0
        # Content survives the filter: across the middle of the record the
        # campaign ramp climbs at its commanded rate, two drive samples per
        # 10 kHz row.
        names = _strings(f, "file_names")
        labels = _strings(f, "input_labels")
        k = names.index("campaign/run_a/low/x_l1.h5")
        x = as_dataset(f, "inputs")[offsets[k] : offsets[k + 1]]
        x = x[:, labels.index("x_pos")].astype(float)
        mid = slice(len(x) // 4, 3 * len(x) // 4)
        slope = np.polyfit(np.arange(len(x))[mid], x[mid], 1)[0]
        ramp_samples = N // 2 - 200
        want = 2 * RAMP_COUNTS / ramp_samples * RES_NM
        assert np.isclose(slope, want, rtol=0.02)


def test_overlapping_sources_are_rejected(tmp_path, runs):
    spec = _spec(tmp_path, runs)
    raw = json.loads(spec.read_text())
    raw["sources"].append({"glob": "campaign/*/low/*.h5", "split": "train"})
    spec.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="matched by both"):
        m.discover(m.BuildSpec.from_json(spec))


def test_unknown_split_is_rejected(tmp_path, runs):
    spec = _spec(tmp_path, runs)
    raw = json.loads(spec.read_text())
    raw["sources"][0]["split"] = "holdout"
    spec.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="split must be"):
        m.BuildSpec.from_json(spec)


def test_build_refuses_to_empty_a_split(tmp_path, runs):
    m.build_capture_dataset(_spec(tmp_path, runs))
    before = (tmp_path / "out.h5").read_bytes()
    # The campaign records move x 77 nm from rest, a stop just inside that
    # holds them all apart, leaving val and test empty, so the build fails
    # and the previous dataset survives.
    spec = _spec(tmp_path, runs, stops_um={"x": [-1000.0, 0.05]})
    with pytest.raises(ValueError, match="No recordings left"):
        m.build_capture_dataset(spec)
    assert (tmp_path / "out.h5").read_bytes() == before
    assert not (tmp_path / "out.h5.partial").exists()


def _segment(lo_nm, hi_nm):
    excursion = np.zeros((3, 2))
    excursion[0] = (lo_nm, hi_nm)
    return m.Segment(
        record=None,  # type: ignore[arg-type]
        inputs=np.empty((0, 0)),
        targets=np.empty((0, 0)),
        rest_nm=np.zeros(3),
        peak_command=np.zeros(3),
        excursion_nm=excursion,
    )


@pytest.mark.parametrize(
    ("lo_um", "hi_um", "near"),
    [
        (-100.0, 100.0, False),  # well inside both stops
        (-955.0, 0.0, True),  # inside the low margin
        (-945.0, 0.0, False),  # just outside it
        (0.0, 955.0, True),  # inside the high margin
        (0.0, 945.0, False),
        (-1010.0, 0.0, True),  # past the stop
    ],
)
def test_near_stop_both_sides_with_margin(tmp_path, runs, lo_um, hi_um, near):
    spec = m.BuildSpec.from_json(
        _spec(
            tmp_path,
            runs,
            stops_um={"x": [-1000.0, 1000.0]},
            stop_margin_um=50.0,
        )
    )
    seg = _segment(lo_um * m.NM_PER_UM, hi_um * m.NM_PER_UM)
    assert m.near_stop(seg, spec) is near


def test_dealing_rotates_lone_records(tmp_path, runs, write_capture):
    b = runs / "campaign/run_b"
    _record(
        write_capture,
        b / "quiet_start/x_q.h5",
        kind="quiet",
        segment="quiet_start",
    )
    _record(
        write_capture, b / "quiet_end/x_q.h5", kind="quiet", segment="quiet_end"
    )
    recs = m.discover(m.BuildSpec.from_json(_spec(tmp_path, runs)))
    quiet = {
        r.group: r.split
        for r in recs
        if r.kind == "quiet" and r.source == "campaign/run_b"
    }
    # Two single-record groups land in different splits.
    assert sorted(quiet.values()) == ["test", "val"]


@pytest.mark.parametrize("rate_hz", [20_000, 10_000])
def test_placed_capture_reproduces_the_stored_rows(tmp_path, runs, rate_hz):
    # A capture placed through the dataset's own frame must give exactly the
    # rows the build stored for it: same resampling, same rest, same pairing
    # of each position with the command a step earlier.
    from nn_motion_control.data import panda
    from nn_motion_control.data.ingest import INPUT_LABELS

    aa = {"passband_hz": 4000, "stopband_hz": 5000, "attenuation_db": 80}
    extra = {"anti_alias": aa} if rate_hz < 20_000 else {}
    m.build_capture_dataset(_spec(tmp_path, runs, rate_hz=rate_hz, **extra))
    out = tmp_path / "out.h5"
    name = "campaign/run_a/low/x_l1.h5"
    with h5py.File(out, "r") as f:
        k = _strings(f, "file_names").index(name)
        lo, hi = as_dataset(f, "segment_offsets")[k : k + 2]
        stored = as_dataset(f, "inputs")[lo:hi].astype(float)

    frame = panda.DatasetFrame.of(out, rate_hz=None)
    cap = panda.read_closed_loop(
        runs / name, {"x": "PGEN1.OUT.Value"}, m.AXES, RES_NM
    )
    pos, cmd = frame.rows(*frame.place(cap))
    # Stored row k came from the paired sample k + 3: the first-row DAC
    # shift is already in 'rows', the V/A/J warm-up drops three more.
    lead = ROWS_LOST_PER_FILE - 2
    n = len(stored)
    for a, axis in enumerate(m.AXES):
        col_p = INPUT_LABELS.index(f"{axis}_pos")
        col_u = INPUT_LABELS.index(f"{axis}_DAC_real")
        assert np.allclose(pos[a, lead : lead + n], stored[:, col_p], atol=1e-3)
        assert np.allclose(cmd[a, lead : lead + n], stored[:, col_u], atol=1e-3)
