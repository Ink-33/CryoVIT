"""Tests for layered load statistics and label mapping."""

import numpy as np
import pytest

from cryovit.utils import (
    _match_label_keys_to_data,
    load_data,
    load_labels,
    read_hdf,
)


def test_load_data_hdf5_explicit_key_no_stats(tmp_path):
    path = tmp_path / "t.hdf5"
    import h5py

    with h5py.File(path, "w") as fh:
        fh.create_dataset("dino_features", data=np.zeros((2, 3, 4), dtype=np.float16))

    data, key = load_data(path, key="dino_features")
    assert key == "dino_features"
    assert data.shape == (1, 2, 3, 4)


def test_load_data_missing_key_raises(tmp_path):
    path = tmp_path / "t.hdf5"
    import h5py

    with h5py.File(path, "w") as fh:
        fh.create_dataset("data", data=np.zeros((2, 2, 2), dtype=np.float32))

    with pytest.raises(KeyError, match="missing"):
        load_data(path, key="missing")


def test_hdf5_single_label_direct_read_preserves_values(tmp_path):
    """Caller-guaranteed legal values: {0,1} must stay {0,1} after int8."""
    path = tmp_path / "lab.hdf5"
    import h5py

    arr = np.zeros((3, 3, 3), dtype=np.uint8)
    arr[1, 1, 1] = 1
    with h5py.File(path, "w") as fh:
        fh.create_dataset("mito", data=arr)

    labels = load_labels(path, label_keys=["mito"], key="mito")
    assert labels["mito"].dtype == np.int8
    assert set(np.unique(labels["mito"]).tolist()) <= {0, 1}
    assert labels["mito"][1, 1, 1] == 1


def test_mrc_tiff_map_0_255_mask_via_match(tmp_path):
    """MRC/TIFF single-class {0,255} must map to {0,1}, not int8 wrap to -1."""
    data = np.zeros((4, 4, 4), dtype=np.uint8)
    data[2, 2, 2] = 255

    out = _match_label_keys_to_data(data, ["mito"])
    assert set(np.unique(out["mito"]).tolist()) <= {0, 1}
    assert out["mito"][2, 2, 2] == 1
    assert out["mito"][0, 0, 0] == 0


def test_match_label_keys_single_unique_scan(monkeypatch):
    data = np.array([[1, 1], [2, 2]], dtype=np.int16)
    calls = {"n": 0}
    real_unique = np.unique

    def counting_unique(a, *args, **kwargs):
        calls["n"] += 1
        return real_unique(a, *args, **kwargs)

    monkeypatch.setattr(np, "unique", counting_unique)
    out = _match_label_keys_to_data(data, ["a", "b"])
    assert calls["n"] == 1
    assert set(out) == {"a", "b"}
    assert out["a"][0, 0] == 1
    assert out["b"][1, 0] == 1


def test_match_label_keys_with_ignore_value():
    data = np.array([[-1, 1], [2, 2]], dtype=np.int16)
    out = _match_label_keys_to_data(data, ["a", "b"])
    assert out["a"][0, 1] == 1  # value 1 → key a
    assert out["b"][1, 0] == 1  # value 2 → key b
    assert out["a"][0, 0] == -1  # ignore preserved


def test_match_label_keys_multi_with_background():
    data = np.array([[0, 1], [2, 0]], dtype=np.int16)
    out = _match_label_keys_to_data(data, ["a", "b"])
    # nunique(3) == len(keys)+1 and 0 present → 1,2 map to a,b
    assert out["a"][0, 1] == 1
    assert out["b"][1, 0] == 1
    assert out["a"][1, 1] == 0  # background


def test_read_hdf_compute_stats_false_leaves_nunique_zero(tmp_path):
    path = tmp_path / "t.hdf5"
    import h5py

    with h5py.File(path, "w") as fh:
        fh.create_dataset("x", data=np.arange(10, dtype=np.int16))

    _, _, meta = read_hdf(path, key="x", compute_stats=False)
    assert meta.nunique == 0
    _, _, meta2 = read_hdf(path, key="x", compute_stats=True)
    assert meta2.nunique == 10
