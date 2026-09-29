"""Tests for dataset-identity reuse of the raw volume."""

import numpy as np
import pytest

from cryovit.datasets.file_dataset import FileDataset
from cryovit.types import FileData


h5py = pytest.importorskip("h5py")


def _write_pair(path, raw_key="data", feat_key="dino_features"):
    """Write a file with a raw volume and (optionally) feature volume.

    Raw and features share the same shape on purpose — shape must not drive reuse.
    """
    raw = np.arange(2 * 4 * 4, dtype=np.float32).reshape(1, 2, 4, 4) / 255.0
    feat = raw + 1.0  # same shape, different values/key
    with h5py.File(path, "w") as fh:
        fh.create_dataset(raw_key, data=raw)
        if feat_key is not None:
            fh.create_dataset(feat_key, data=feat.astype(np.float16))


def test_resolved_data_key_reuses_single_read(tmp_path, monkeypatch):
    path = tmp_path / "t.hdf"
    _write_pair(path)

    calls = {"n": 0}
    import cryovit.datasets.file_dataset as fd

    real_load = fd.load_data

    def counting_load(p, key=None):
        calls["n"] += 1
        return real_load(p, key=key)

    monkeypatch.setattr(fd, "load_data", counting_load)

    ds = FileDataset(
        [FileData(tomo_path=path)], input_key="data", label_key="mito", train=False
    )
    sample = ds[0]
    assert calls["n"] == 1
    assert sample.aux_data is not None
    aux = np.asarray(sample.aux_data["data"])
    inp = np.asarray(sample.data)
    if inp.ndim == 4:  # (C, D, H, W) → drop channel like squeeze(0)
        inp = inp[0]
    # aux raw must be the same values as input (not a second dataset)
    assert np.allclose(aux, inp)


def test_features_and_raw_are_separate_reads(tmp_path, monkeypatch):
    """dino_features vs data: different keys → two reads even if shapes match."""
    path = tmp_path / "t.hdf"
    _write_pair(path)

    calls = {"keys": []}
    import cryovit.datasets.file_dataset as fd

    real_load = fd.load_data

    def tracking_load(p, key=None):
        calls["keys"].append(key)
        return real_load(p, key=key)

    monkeypatch.setattr(fd, "load_data", tracking_load)

    ds = FileDataset(
        [FileData(tomo_path=path)],
        input_key="dino_features",
        label_key="mito",
        train=False,
    )
    sample = ds[0]
    # first call with input_key, second with key="data"
    assert calls["keys"] == ["dino_features", "data"]
    assert sample.aux_data is not None
    aux = np.asarray(sample.aux_data["data"])
    inp = np.asarray(sample.data)
    if inp.ndim == 4:
        inp = inp[0]
    # aux is the raw volume, not the feature volume (values differ)
    assert not np.allclose(aux, inp)


def test_none_input_key_resolved_to_data_reuses(tmp_path, monkeypatch):
    """input_key=None must use the resolved key from _key_cache for reuse."""
    path = tmp_path / "t.hdf"
    _write_pair(path, feat_key=None)

    calls = {"n": 0}
    import cryovit.datasets.file_dataset as fd

    real_load = fd.load_data

    def counting_load(p, key=None):
        calls["n"] += 1
        return real_load(p, key=key)

    monkeypatch.setattr(fd, "load_data", counting_load)

    ds = FileDataset(
        [FileData(tomo_path=path)], input_key=None, label_key="mito", train=False
    )
    sample = ds[0]
    # heuristic resolves to "data" → must NOT re-read
    assert calls["n"] == 1
    assert ds._key_cache[path] == "data"
    assert sample.aux_data is not None


def test_same_shape_different_key_not_reused(tmp_path):
    """Same shape alone must never cause reuse."""
    path = tmp_path / "t.hdf"
    _write_pair(path)
    ds = FileDataset(
        [FileData(tomo_path=path)],
        input_key="dino_features",
        label_key="mito",
        train=False,
    )
    data = ds._load_tomogram(ds.files[0])
    assert data["resolved_key"] == "dino_features"
    assert data["input"].shape[-3:] == (2, 4, 4)
