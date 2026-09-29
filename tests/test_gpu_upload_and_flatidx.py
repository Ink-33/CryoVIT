"""Tests for per-batch GPU upload and flat_idxs lookup."""

import numpy as np
import torch

from cryovit.run.dino_features import _dino_features, _sam_features
from cryovit.types import BatchedTomogramData, BatchedTomogramMetadata


class _FakeDino:
    def forward_features(self, vec):
        b = vec.shape[0]
        # 2x2 patches of dim 8 → (B, 4, 8)
        return {"x_norm_patchtokens": torch.ones(b, 4, 8)}


class _FakeSam:
    def __init__(self):
        self._cursor = 0

    def forward_features(self, vec):
        b = vec.shape[0]
        start = self._cursor
        self._cursor += b
        idx = torch.arange(start, start + b, dtype=torch.float32)
        return {
            "backbone_fpn": [idx.view(b, 1, 1, 1).expand(b, 2, 2, 2).clone()],
            "vision_pos_enc": [idx.view(b, 1, 1, 1).expand(b, 2, 2, 2).clone()],
            "vision_features": idx.view(b, 1),
        }


def test_dino_uploads_per_batch_not_full_volume(monkeypatch):
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, *a, **k: self)
    uploaded = []

    def fake_cuda(self, *a, **k):
        uploaded.append(tuple(self.shape))
        return self

    monkeypatch.setattr(torch.Tensor, "cuda", fake_cuda)

    data = torch.zeros(5, 3, 28, 28)  # 5 slices
    _dino_features(data, _FakeDino(), batch_size=2)

    assert uploaded, "expected at least one .cuda() call"
    assert all(shape[0] <= 2 for shape in uploaded), uploaded
    assert max(shape[0] for shape in uploaded) <= 2


def test_sam_uploads_per_batch_not_full_volume(monkeypatch):
    uploaded = []

    def fake_cuda(self, *a, **k):
        uploaded.append(tuple(self.shape))
        return self

    monkeypatch.setattr(torch.Tensor, "cuda", fake_cuda)

    data = torch.zeros(7, 3, 8, 8)
    out = _sam_features(data, _FakeSam(), batch_size=3)
    assert out["backbone_fpn"][0].shape[0] == 7
    assert uploaded
    assert all(shape[0] <= 3 for shape in uploaded), uploaded


def _batched(d_sizes, c=1, h=4, w=4):
    b = len(d_sizes)
    max_d = max(d_sizes)
    tomo = torch.zeros(b, max_d, c, h, w)
    labels = torch.zeros(b, max_d, h, w)
    sizes = torch.tensor(d_sizes, dtype=torch.int)
    meta = BatchedTomogramMetadata(
        samples=[f"s{i}" for i in range(b)],
        tomo_names=[f"t{i}" for i in range(b)],
        unique_id=torch.arange(b).view(b, 1).repeat(1, 2).long(),
        split_id=None,
    )
    return BatchedTomogramData(
        tomo_batch=tomo,
        tomo_sizes=sizes,
        labels=labels,
        metadata=meta,
        min_slices=min(d_sizes),
        aux_data={},
    )


def test_flat_idxs_by_slice_matches_index_to_flat_batch():
    """One-pass map must agree with the per-slice helper on every slice."""
    batch = _batched([3, 5, 2])
    table = batch.flat_idxs_by_slice()
    assert len(table) == batch.num_slices
    for i in range(batch.num_slices):
        assert torch.equal(table[i], batch.index_to_flat_batch(i))
        assert int(table[i].max()) < sum([3, 5, 2])


def test_flat_idxs_by_slice_single_prefix_sum_not_n_argwhere(monkeypatch):
    """Building the full table must not call index_to_flat_batch / argwhere per slice."""
    batch = _batched([4, 6, 3])
    calls = {"index": 0, "argwhere": 0}

    real_index = batch.index_to_flat_batch

    def counting_index(idx):
        calls["index"] += 1
        return real_index(idx)

    monkeypatch.setattr(
        type(batch), "index_to_flat_batch", lambda self, idx: counting_index(idx)
    )
    real_argwhere = torch.argwhere

    def counting_argwhere(*a, **k):
        calls["argwhere"] += 1
        return real_argwhere(*a, **k)

    monkeypatch.setattr(torch, "argwhere", counting_argwhere)

    table = batch.flat_idxs_by_slice()
    assert len(table) == 6
    assert calls["index"] == 0
    assert calls["argwhere"] == 0


def test_tracking_loop_uses_lookup_not_index_helper(monkeypatch):
    """Simulated forward_tracking inner loop must only look up the table."""
    batch = _batched([3, 5, 2])
    calls = {"index": 0}
    real_index = type(batch).index_to_flat_batch

    def counting_index(self, idx):
        calls["index"] += 1
        return real_index(self, idx)

    monkeypatch.setattr(type(batch), "index_to_flat_batch", counting_index)

    flat_idxs_list = batch.flat_idxs_by_slice()
    processing_order = [0, 2, 1, 3, 4]
    seen = []
    for slice_id in processing_order:
        flat_idxs = flat_idxs_list[slice_id]
        seen.append(flat_idxs.tolist())
    assert calls["index"] == 0
    # slice 0 is present in all 3 tomos → 3 flat indices (same packing as helper)
    assert seen[0] == [0, 3, 8]


def test_flat_idxs_skips_padded_slices():
    batch = _batched([2, 5])
    table = batch.flat_idxs_by_slice()
    # slice 3 exists only in tomogram 1
    assert table[3].numel() == 1
    # slice index 0 exists in both
    assert table[0].numel() == 2
