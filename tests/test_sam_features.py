"""Regression tests for _sam_features cross-batch accumulation."""

import numpy as np
import torch

from cryovit.run.dino_features import _sam_features


class _FakeSamModel:
    """Deterministic multi-level features; value encodes the global slice index."""

    def __init__(self, slices_per_call: int | None = None):
        self._cursor = 0
        self.slices_per_call = slices_per_call

    def forward_features(self, vec: torch.Tensor):
        b = vec.shape[0]
        start = self._cursor
        self._cursor += b
        idx = torch.arange(start, start + b, dtype=torch.float32)
        lvl0 = idx.view(b, 1, 1, 1).expand(b, 2, 4, 4).clone()
        lvl1 = (idx * 10).view(b, 1, 1, 1).expand(b, 3, 2, 2).clone()
        pos0 = idx.view(b, 1, 1, 1).expand(b, 2, 4, 4).clone()
        pos1 = (idx * 10).view(b, 1, 1, 1).expand(b, 3, 2, 2).clone()
        return {
            "backbone_fpn": [lvl0, lvl1],
            "vision_pos_enc": [pos0, pos1],
            "vision_features": idx.view(b, 1).expand(b, 8).clone(),
        }


def _input(n_slices: int) -> torch.Tensor:
    return torch.zeros(n_slices, 3, 8, 8)


def test_sam_features_keeps_all_slices_across_batches(monkeypatch):
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, *a, **k: self)

    n_slices = 5
    batch_size = 2  # forces 3 batches: 2 + 2 + 1

    model = _FakeSamModel()
    out = _sam_features(_input(n_slices), model, batch_size)

    assert set(out) == {"backbone_fpn", "vision_pos_enc"}
    assert len(out["backbone_fpn"]) == 2
    assert len(out["vision_pos_enc"]) == 2

    # level 0: every slice present once, in order
    fpn0 = out["backbone_fpn"][0]
    assert fpn0.shape == (n_slices, 2, 4, 4)
    assert np.allclose(fpn0[:, 0, 0, 0], np.arange(n_slices))

    # level 1: same for the other pyramid level
    fpn1 = out["backbone_fpn"][1]
    assert fpn1.shape == (n_slices, 3, 2, 2)
    assert np.allclose(fpn1[:, 0, 0, 0], np.arange(n_slices) * 10)

    # pos encs must follow the same accumulation
    assert out["vision_pos_enc"][0].shape == (n_slices, 2, 4, 4)
    assert np.allclose(out["vision_pos_enc"][0][:, 0, 0, 0], np.arange(n_slices))


def test_sam_features_matches_per_slice_reference(monkeypatch):
    """Batched extraction must equal concatenating batch_size=1 runs."""
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, *a, **k: self)

    n_slices = 7
    data = _input(n_slices)

    ref_model = _FakeSamModel()
    reference = _sam_features(data, ref_model, batch_size=1)

    for bs in (2, 3, 5, n_slices, n_slices + 3):
        model = _FakeSamModel()
        out = _sam_features(data, model, batch_size=bs)
        assert set(out) == set(reference)
        for key in reference:
            assert len(out[key]) == len(reference[key])
            for lvl, (got, exp) in enumerate(zip(out[key], reference[key], strict=True)):
                assert got.shape == exp.shape, (bs, key, lvl)
                assert np.array_equal(got, exp), (bs, key, lvl)


def test_sam_features_single_batch_unchanged(monkeypatch):
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, *a, **k: self)
    data = _input(4)
    out = _sam_features(data, _FakeSamModel(), batch_size=4)
    assert out["backbone_fpn"][0].shape[0] == 4
    assert np.allclose(out["backbone_fpn"][0][:, 0, 0, 0], np.arange(4))


def test_sam_features_concatenates_once_per_level(monkeypatch):
    """Growth via per-batch np.concatenate is forbidden (O(n²) copies)."""
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, *a, **k: self)

    calls = {"n": 0}
    real_concat = np.concatenate

    def counting_concat(*args, **kwargs):
        calls["n"] += 1
        return real_concat(*args, **kwargs)

    monkeypatch.setattr(np, "concatenate", counting_concat)

    n_slices = 6
    _sam_features(_input(n_slices), _FakeSamModel(), batch_size=2)
    # Fake model exposes 2 keys × 2 levels (vision_features is skipped).
    # Exactly one concatenate per (key, level) at the end — 4 total — and
    # zero growth concatenates inside the 3-batch loop.
    assert calls["n"] == 4, calls
