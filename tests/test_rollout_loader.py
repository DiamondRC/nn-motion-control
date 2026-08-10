"""
Rollout data: horizon-aware split reservation/gap and RolloutWindowLoader gathering.
"""

import numpy as np
import torch

from nn_motion_control.data.device_loader import RolloutWindowLoader
from nn_motion_control.data.splits import (
    build_valid_window_starts,
    split_window_starts_contiguous,
)

W, H, F_IN = 4, 3, 9
POS_COLS, DAC_COLS = [0, 3, 6], [2, 5, 8]


def test_valid_starts_reserve_window_plus_horizon():
    # One segment [0, 20). A sample reads rows [s, s+W+H); last valid start = 20-W-H.
    offsets = np.array([0, 20], dtype=np.int64)
    starts = build_valid_window_starts(offsets, W, horizon=H)
    assert starts[0] == 0
    assert starts[-1] == 20 - W - H  # 13
    # horizon=0 recovers the plain window reservation.
    assert build_valid_window_starts(offsets, W)[-1] == 20 - W


def test_no_rollout_sample_crosses_a_boundary():
    offsets = np.array([0, 20, 45], dtype=np.int64)
    starts = build_valid_window_starts(offsets, W, horizon=H)
    # The furthest row a sample reads is s + W + H - 1; it must stay in its segment.
    for s in starts:
        end = next(int(e) for e in offsets[1:] if e > s)
        assert s + W + H - 1 < end


def test_contiguous_gap_includes_horizon():
    starts = np.arange(100, dtype=np.int64)
    tr, va, te = split_window_starts_contiguous(starts, 0.8, 0.1, W, horizon=H)
    # gap = W + H - 1 dropped from the tail of train and val.
    gap = W + H - 1
    assert tr[-1] == 80 - 1 - gap
    assert va[-1] == 90 - 1 - gap
    assert te[-1] == 99  # test keeps its tail


def _encoded_inputs(n=60):
    # inputs[r, c] = r*10 + c, so gathered rows/cols are decodable.
    rows = np.arange(n)[:, None] * 10
    cols = np.arange(F_IN)[None, :]
    return torch.tensor(rows + cols, dtype=torch.float32)


def _loader(starts, **kw):
    return RolloutWindowLoader(
        _encoded_inputs(),
        torch.as_tensor(starts, dtype=torch.int64),
        W,
        H,
        batch_size=kw.pop("batch_size", 8),
        pos_cols=POS_COLS,
        dac_cols=DAC_COLS,
        device="cpu",
        **kw,
    )


def test_shapes():
    warmup, dac, gt = next(iter(_loader([0, 1, 2, 3, 4])))
    assert warmup.shape == (5, F_IN, W)
    assert dac.shape == (5, H, len(DAC_COLS))
    assert gt.shape == (5, H, len(POS_COLS))


def test_gathers_correct_rows_and_columns():
    warmup, dac, gt = next(iter(_loader([0, 5])))
    # start 0: warmup rows [0,4) x all cols -> inputs[t, c] = t*10 + c.
    assert warmup[0, 0, :].tolist() == [0, 10, 20, 30]  # col 0 over the window
    assert warmup[0, :, 0].tolist() == list(range(F_IN))  # all cols at t=0
    # start 5: future rows [5+W, 5+W+H) = [9,10,11]; dac cols [2,5,8], pos cols [0,3,6].
    assert dac[1, 0, :].tolist() == [9 * 10 + 2, 9 * 10 + 5, 9 * 10 + 8]
    assert gt[1, 0, :].tolist() == [9 * 10 + 0, 9 * 10 + 3, 9 * 10 + 6]
    assert gt[1, 2, :].tolist() == [11 * 10 + 0, 11 * 10 + 3, 11 * 10 + 6]


def test_len_and_full_coverage():
    starts = list(range(20))
    loader = _loader(starts, batch_size=8)
    assert len(loader) == 3  # ceil(20/8)
    seen = sum(w.shape[0] for w, _, _ in loader)
    assert seen == 20


def test_drop_last_gives_constant_batch_shape():
    # drop_last drops the ragged tail so every yielded batch is full (compile-safe).
    loader = _loader(range(20), batch_size=8, drop_last=True)
    assert len(loader) == 2  # floor(20/8)
    sizes = [w.shape[0] for w, _, _ in loader]
    assert sizes == [8, 8]


def test_shuffle_is_seed_deterministic():
    a = torch.cat([w for w, _, _ in _loader(range(16), shuffle=True, seed=1)])
    b = torch.cat([w for w, _, _ in _loader(range(16), shuffle=True, seed=1)])
    c = torch.cat([w for w, _, _ in _loader(range(16), shuffle=True, seed=2)])
    assert torch.equal(a, b)
    assert not torch.equal(a, c)
