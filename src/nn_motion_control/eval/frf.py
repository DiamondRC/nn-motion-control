"""
Frequency-response faithfulness of a plant against the measured stage.

A time-domain error is dominated by the slow, large motion, so a plant can
score well while inventing dynamics where the stage barely moves. This check
works in the frequency domain instead: it free-runs the plant on the
recorded command of every held-out periodic multisine record, estimates the
transfer function on the excited lines by period averaging, the same way
for the prediction and for the recording, and reports their ratio per band.
Both come from the same rows, so sampling, filtering and the one-step DAC
shift cancel.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import h5py
import numpy as np
import torch

from nn_motion_control.core.config import RunConfiguration
from nn_motion_control.data._hdf5 import as_dataset
from nn_motion_control.plant.plant import Plant

logger = logging.getLogger(__name__)

# Bands the report aggregates over, in Hz.
BANDS_HZ = ((5, 30), (30, 300), (300, 1200), (1200, 5000), (5000, 9600))

# A DFT bin is an excited line when its command amplitude is at least this
# fraction of the strongest line.
LINE_FRACTION = 0.05

# A line counts only where the recording's response stands this far above
# its own period-to-period scatter.
MIN_SNR_DB = 10.0

# Within this many dB of the measured gain a line is a match.
MATCH_DB = 3.0

# Search range for a record's period, seconds.
MIN_PERIOD_S = 0.1
MAX_PERIOD_S = 20.0

# A record must hold this many whole repeating periods.
MIN_PERIODS = 2

# Keeps an exactly silent axis at 0 dB rather than a log of zero.
POWER_FLOOR = 1e-30

# Relative tolerance on a repeated command sample (float32 storage).
PERIODIC_TOL = 1e-5

SPLITS = ("val", "test")


@dataclass(frozen=True)
class Record:
    """
    One single-axis periodic record, physical [A, n] rows as stored.
    """

    name: str
    driven: int
    pos: np.ndarray
    vel: np.ndarray
    dac: np.ndarray


@dataclass
class LineSet:
    """
    Excited lines of one record: frequency and [A, L] responses.
    """

    freqs: np.ndarray
    measured: np.ndarray
    predicted: np.ndarray
    snr_db: np.ndarray
    driven: np.ndarray  # [A] bool, the axis the record drove
    name: str = ""


@dataclass
class BandScore:
    band_hz: tuple[float, float]
    lines: int
    gain_ratio: float  # median |predicted / measured|
    phase_deg: float  # circular mean phase of predicted / measured
    within_db: float  # fraction of lines within MATCH_DB


@dataclass
class FrfReport:
    model: str
    records: list[str]
    diagonal: list[BandScore] = field(default_factory=list)
    coupling: list[BandScore] = field(default_factory=list)
    by_axis: dict[str, list[BandScore]] = field(default_factory=dict)
    by_record: dict[str, list[BandScore]] = field(default_factory=dict)
    skipped: list[str] = field(default_factory=list)


def find_period(command: np.ndarray, rate_hz: float) -> int:
    """
    Period of a periodic command, in samples, from the autocorrelation of
    its first difference, which an offset or a ramp onto it does not swamp.
    """

    x = np.diff(command)
    n = len(x)
    spec = np.fft.rfft(x, 2 * n)
    acf = np.fft.irfft(spec * np.conj(spec))[:n]
    lo = int(MIN_PERIOD_S * rate_hz)
    hi = min(int(MAX_PERIOD_S * rate_hz), n // (MIN_PERIODS + 1))
    if hi <= lo:
        raise ValueError("Record too short to hold its period")

    return lo + int(np.argmax(acf[lo:hi]))


def periodic_span(command: np.ndarray, period: int) -> tuple[int, int]:
    """
    (start, whole periods) of the stretch where the command repeats.

    A periodic input stays periodic through any linear filter, so the test
    u[t + period] == u[t] is exact up to rounding. The ramps onto and off
    an offset, and the zero-command lead-in, fall outside it.
    """

    u = command
    tol = PERIODIC_TOL * (np.abs(u).max() + 1.0)
    same = np.abs(u[period:] - u[:-period]) <= tol
    edges = np.flatnonzero(np.diff(np.r_[0, same.astype(np.int8), 0]))
    starts, stops = edges[::2], edges[1::2]
    if not len(starts):
        return 0, 0
    k = int(np.argmax(stops - starts))

    return int(starts[k]), int((stops[k] - starts[k] + period) // period)


def line_response(
    pos: np.ndarray,
    dac: np.ndarray,
    driven: int,
    period: int,
    rate_hz: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Transfer function from the driven command to every axis on its lines.

    Only the stretch where the command repeats is used, which starts
    after the generator's own settling period. Rows hold the command one
    step behind the position it produced, which is undone so the result
    references the command at its own instant. Returns (freqs [L],
    H [A, L], snr_db [A, L]).
    """

    start, n_per = periodic_span(dac[driven], period)
    if start == 0:
        # The response difference needs the sample before the block.
        start, n_per = period, n_per - 1
    if n_per < MIN_PERIODS:
        raise ValueError(f"Record holds {n_per} periods, need {MIN_PERIODS}")
    block = slice(start, start + period * n_per)
    # First differences scale response and command lines alike, so their
    # ratio is unchanged for any linear stage while an offset vanishes and
    # a linear drift becomes DC, which is not a line. The response, a
    # continuous signal, differences against its true previous sample. The
    # command is exactly periodic over the block, so each period wraps onto
    # itself and its lines stay exact.
    y = np.diff(pos[:, start - 1 : block.stop], axis=1)
    y = y.reshape(pos.shape[0], n_per, period)
    u_per = dac[driven, block].reshape(n_per, period)
    u = u_per - np.roll(u_per, 1, axis=1)
    spec_y = np.fft.rfft(y, axis=-1)
    spec_u = np.fft.rfft(u, axis=-1).mean(axis=0)
    freqs = np.fft.rfftfreq(period, 1.0 / rate_hz)
    mean_y = spec_y.mean(axis=1)
    noise = spec_y.var(axis=1, ddof=1) / n_per
    # Lines are picked on the command itself: differencing would suppress
    # the lowest ones below the threshold.
    level = np.abs(np.fft.rfft(u_per, axis=-1)).mean(axis=0)
    ac = freqs > 0
    lines = ac & (level >= LINE_FRACTION * level[ac].max())
    shift = np.exp(-2j * np.pi * freqs[lines] / rate_hz)
    h = mean_y[:, lines] / spec_u[lines] * shift
    power = np.abs(mean_y[:, lines]) ** 2
    snr = 10 * np.log10((power + POWER_FLOOR) / (noise[:, lines] + POWER_FLOOR))

    return freqs[lines], h, snr


def load_records(
    data_path: str, axes: list[str], splits: tuple[str, ...] = SPLITS
) -> list[Record]:
    """
    Every single-axis multisine record of the given splits, as stored.
    """

    with h5py.File(data_path, "r") as f:
        names = [x.decode() for x in as_dataset(f, "file_names")[:]]
        kinds = [x.decode() for x in as_dataset(f, "segment_kind")[:]]
        split = [x.decode() for x in as_dataset(f, "segment_split")[:]]
        driven = [x.decode() for x in as_dataset(f, "segment_axes")[:]]
        offsets = as_dataset(f, "segment_offsets")[:]
        stored = [x.decode() for x in as_dataset(f, "input_labels")[:]]
        cols = {
            ch: [stored.index(f"{a}_{ch}") for a in axes]
            for ch in ("pos", "vel", "DAC_real")
        }
        records = []

        for k, name in enumerate(names):
            if kinds[k] != "multisine" or split[k] not in splits:
                continue
            if len(driven[k]) != 1:
                continue
            rows = as_dataset(f, "inputs")[
                int(offsets[k]) : int(offsets[k + 1])
            ]
            block = {ch: rows[:, c].T for ch, c in cols.items()}
            records.append(
                Record(
                    name=name,
                    driven=axes.index(driven[k]),
                    pos=block["pos"].astype(np.float64),
                    vel=block["vel"].astype(np.float64),
                    dac=block["DAC_real"].astype(np.float64),
                )
            )

    return records


@torch.no_grad()
def free_run(plant: Plant, records: list[Record], window: int) -> np.ndarray:
    """
    Predicted physical position [B, A, n - window] for equal-length records,
    warmed up on each record's first 'window' rows and then driven by its
    recorded command alone.
    """

    def tens(x: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(x, dtype=torch.float32, device=plant.device)

    pos = tens(np.stack([r.pos for r in records])).transpose(1, 2)
    vel = tens(np.stack([r.vel for r in records])).transpose(1, 2)
    dac = tens(np.stack([r.dac for r in records])).transpose(1, 2)
    frames = plant.encode_inputs(
        pos[:, :window], vel[:, :window], dac[:, :window]
    )
    warmup = frames.transpose(1, 2).contiguous()  # [B, F, W]
    future = plant.normalise_command(dac[:, window:])
    preds = plant.roll_forward(warmup, future, future.shape[1])
    phys = preds * plant.pos_std + plant.pos_mean  # [B, H, A]

    return phys.transpose(1, 2).cpu().numpy().astype(np.float64)


def band_scores(
    sets: list[LineSet], diagonal: bool, axis: int | None = None
) -> list[BandScore]:
    """
    Aggregate per band the driven-axis responses ('diagonal') or the
    cross-axis ones, over every record or only those driving 'axis'.
    """

    scores = []

    for lo, hi in BANDS_HZ:
        ratios = []

        for s in sets:
            if axis is not None and not s.driven[axis]:
                continue
            band = (s.freqs >= lo) & (s.freqs < hi)

            for a in range(s.measured.shape[0]):
                if bool(s.driven[a]) != diagonal:
                    continue
                keep = band & (s.snr_db[a] >= MIN_SNR_DB)
                ratios.append(s.predicted[a, keep] / s.measured[a, keep])
        r = np.concatenate(ratios) if ratios else np.empty(0, complex)
        if not len(r):
            continue
        db = 20 * np.log10(np.abs(r))
        # A circular mean, so phases either side of +-180 do not cancel.
        phase = np.angle(np.mean(r / np.abs(r)))
        scores.append(
            BandScore(
                band_hz=(lo, hi),
                lines=len(r),
                gain_ratio=float(np.median(np.abs(r))),
                phase_deg=float(np.degrees(phase)),
                within_db=float(np.mean(np.abs(db) <= MATCH_DB)),
            )
        )

    return scores


def run_plant_frf(
    config_path: str,
    checkpoint: str | None = None,
    out_dir: str | None = None,
    device: str | None = None,
) -> FrfReport:
    """
    Free-run a trained plant on the held-out multisines and score its
    frequency response against the recorded one, writing a report.
    """

    config = RunConfiguration(config_path)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    plant = Plant.from_checkpoint(
        config, checkpoint or config.m_save_dir, device
    )
    rate = config.system.data_rate_hz
    if not rate:
        raise ValueError("The system needs a data_rate_hz for an FRF check")
    axes = list(config.system.axes)
    window = config.window_size
    records = load_records(config.datafile_dir, axes)
    if not records:
        raise ValueError(
            f"No held-out multisine records in {config.datafile_dir}"
        )
    by_len: dict[int, list[Record]] = {}

    for r in records:
        by_len.setdefault(r.pos.shape[1], []).append(r)
    sets: list[LineSet] = []
    skipped: list[str] = []

    for n, group in sorted(by_len.items()):
        logger.info("Free-running %d record(s) of %d steps", len(group), n)
        preds = free_run(plant, group, window)

        for r, pred in zip(group, preds, strict=True):
            truth = r.pos[:, window:]
            dac = r.dac[:, window:]
            try:
                period = find_period(dac[r.driven], rate)
                f, h_true, snr = line_response(
                    truth, dac, r.driven, period, rate
                )
                _, h_pred, _ = line_response(pred, dac, r.driven, period, rate)
            except ValueError as exc:
                logger.warning("Skipping %s: %s", r.name, exc)
                skipped.append(r.name)
                continue
            driven = np.arange(len(axes)) == r.driven
            sets.append(LineSet(f, h_true, h_pred, snr, driven, r.name))

    report = FrfReport(
        model=config.model_name,
        records=[s.name for s in sets],
        diagonal=band_scores(sets, diagonal=True),
        coupling=band_scores(sets, diagonal=False),
        by_axis={
            ax: band_scores(sets, diagonal=True, axis=i)
            for i, ax in enumerate(axes)
        },
        by_record={s.name: band_scores([s], diagonal=True) for s in sets},
        skipped=skipped,
    )
    out = Path(out_dir or Path(config.eval_dir) / "frf")
    out.mkdir(parents=True, exist_ok=True)
    (out / "frf.json").write_text(json.dumps(asdict(report), indent=2))
    (out / "report.md").write_text(format_report(report))
    logger.info("FRF report written to %s", out)

    return report


def format_report(report: FrfReport) -> str:
    def table(scores: list[BandScore]) -> list[str]:
        rows = [
            "| band (Hz) | lines | gain ratio | phase (deg) | within "
            f"{MATCH_DB:g} dB |",
            "|---|---|---|---|---|",
        ]
        rows += [
            f"| {lo:g}-{hi:g} | {s.lines} | {s.gain_ratio:.3f} | "
            f"{s.phase_deg:+.1f} | {s.within_db:.0%} |"
            for s in scores
            for lo, hi in [s.band_hz]
        ]

        return rows

    lines = [
        f"# Plant FRF check: {report.model}",
        "",
        f"Free-run on {len(report.records)} held-out multisine record(s); "
        "ratio of predicted to recorded response on lines whose recorded "
        f"SNR is at least {MIN_SNR_DB:g} dB.",
        "",
        "## Driven axis",
        "",
        *table(report.diagonal),
        "",
        "## Cross-axis coupling",
        "",
        *table(report.coupling),
        "",
        *[
            line
            for ax, scores in report.by_axis.items()
            for line in (f"## Driven axis {ax} only", "", *table(scores), "")
        ],
        *(
            [f"Skipped (too short to analyse): {', '.join(report.skipped)}", ""]
            if report.skipped
            else []
        ),
    ]

    return "\n".join(lines)
