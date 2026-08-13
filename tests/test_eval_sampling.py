"""Representative strided sampling from timeline-ordered rollout loaders."""

import pytest
import torch

from nn_motion_control.eval.sampling import (
    representative_windows,
    sampled_batches,
)


class _Loader:
    """A minimal loader: a fixed list of batches with a length."""

    def __init__(self, batches):
        self._batches = batches

    def __len__(self):
        return len(self._batches)

    def __iter__(self):
        return iter(self._batches)


def test_sampled_batches_strides_across_loader():
    loader = _Loader([(torch.tensor([i]),) for i in range(10)])
    got = [b[0].item() for b in sampled_batches(loader, 3)]
    assert got == [0, 3, 6]  # strided across the timeline, not the first three


def test_sampled_batches_none_yields_all():
    loader = _Loader([(torch.tensor([i]),) for i in range(4)])
    assert [b[0].item() for b in sampled_batches(loader, None)] == [0, 1, 2, 3]


def test_representative_windows_spans_batches():
    loader = _Loader([(torch.full((2, 1, 4), float(i)),) for i in range(8)])
    windows = representative_windows(loader, max_batches=4)
    assert windows.shape == (8, 1, 4)  # 4 strided batches x 2 rows
    # Drawn from batches 0, 2, 4, 6 -> representative of the whole
    # range, not the quiet leading batch that next(iter(loader))
    # would give.
    assert sorted(set(windows[:, 0, 0].tolist())) == [0.0, 2.0, 4.0, 6.0]


def test_representative_windows_rejects_empty():
    with pytest.raises(ValueError):
        representative_windows(_Loader([]), max_batches=4)
