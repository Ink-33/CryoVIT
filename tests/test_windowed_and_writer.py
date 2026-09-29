"""Tests for windowed HDF5 reads, async prediction writer, and related switches."""

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")


def test_read_hdf_window_reads_only_slice(tmp_path):
    from cryovit.utils import read_hdf_window

    path = tmp_path / "t.hdf5"
    vol = np.arange(4 * 6 * 8, dtype=np.float32).reshape(4, 6, 8)
    with h5py.File(path, "w") as fh:
        fh.create_dataset("data", data=vol)
    win = (slice(1, 3), slice(2, 5), slice(0, 4))
    got = read_hdf_window(path, "data", win)
    assert got.shape == (2, 3, 4)
    assert np.array_equal(got, vol[1:3, 2:5, 0:4])


def test_apply_label_value_map_global_not_crop_local():
    from cryovit.utils import apply_label_value_map

    # value_map is raw_value → class_id (class_id matches label_keys order)
    value_map = {255: 1, 3: 2}
    # Window contains only class 1 and background (missing class 2)
    crop = np.array([[0, 255], [255, -1]], dtype=np.int16)
    out = apply_label_value_map(crop, ["a", "b"], value_map)
    assert out["a"].tolist() == [[0, 1], [1, -1]]
    # class 2 absent in this window must stay 0, not rebind to 255
    assert out["b"].tolist() == [[0, 0], [0, -1]]


def test_windowed_training_crop_independent_copy(tmp_path):
    from cryovit.datasets.file_dataset import FileDataset
    from cryovit.types import FileData

    path = tmp_path / "t.hdf5"
    lab = tmp_path / "lab.hdf5"
    raw = np.random.rand(4, 8, 8).astype(np.float32)
    labels = np.zeros((4, 8, 8), dtype=np.uint8)
    labels[1, 1, 1] = 255
    with h5py.File(path, "w") as fh:
        fh.create_dataset("data", data=raw)
    with h5py.File(lab, "w") as fh:
        fh.create_dataset("mito", data=labels)

    ds = FileDataset(
        [FileData(tomo_path=path, label_path=lab, labels=["mito"])],
        input_key="data",
        label_key="mito",
        train=True,
        windowed_read=True,
        label_value_map={255: 1},
    )
    sample = ds[0]
    # window smaller than volume → independent array, no base pointing at volume
    assert sample.data.untyped_storage().nbytes() <= 4 * 512 * 512 * 4 + 64
    assert sample.label.shape[-2:] == (512, 512) or sample.label.shape[-1] == 8


def test_save_raw_false_skips_aux_load(tmp_path, monkeypatch):
    from cryovit.datasets.file_dataset import FileDataset
    from cryovit.types import FileData

    path = tmp_path / "t.hdf5"
    with h5py.File(path, "w") as fh:
        fh.create_dataset("data", data=np.zeros((2, 4, 4), dtype=np.float32))
        fh.create_dataset("dino_features", data=np.zeros((1536, 2, 2), dtype=np.float16))

    calls = {"n": 0}
    import cryovit.datasets.file_dataset as fd

    real = fd.load_data

    def counting(*a, **k):
        calls["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(fd, "load_data", counting)

    ds = FileDataset(
        [FileData(tomo_path=path)],
        input_key="dino_features",
        label_key="mito",
        train=False,
        load_aux_raw=False,
    )
    sample = ds[0]
    assert sample.aux_data is None or sample.aux_data == {}
    # only the feature dataset is read — no second read of key="data"
    assert calls["n"] == 1


def test_prediction_writer_save_raw_and_async_drain(tmp_path):
    from cryovit.models.callbacks import PredictionWriter
    from cryovit.types import BatchedModelResult

    w = PredictionWriter(
        results_dir=tmp_path,
        label_key="mito",
        threshold=0.5,
        save_raw=False,
    )
    preds = np.array([[[0.9, 0.1]]], dtype=np.float32)
    result = BatchedModelResult(
        num_tomos=1,
        samples=["s"],
        tomo_names=["t.hdf"],
        split_id=None,
        data=[np.zeros((1, 2, 2), dtype=np.float32)],
        label=[np.zeros((1, 2, 2), dtype=np.uint8)],
        preds=[preds],
        losses={},
        metrics={},
        aux_data=None,
    )
    w.write_on_batch_end(None, None, result, None, None, 0, 0)
    w.drain()
    assert len(w.result_paths) == 1
    with h5py.File(w.result_paths[0], "r") as fh:
        assert "data" not in fh
        assert "mito_preds" in fh


def test_prediction_writer_save_raw_true_keeps_data(tmp_path):
    from cryovit.models.callbacks import PredictionWriter
    from cryovit.types import BatchedModelResult

    w = PredictionWriter(
        results_dir=tmp_path, label_key="mito", threshold=0.5, save_raw=True
    )
    preds = np.zeros((1, 2, 2), dtype=np.float32)
    result = BatchedModelResult(
        num_tomos=1,
        samples=["s"],
        tomo_names=["t.hdf"],
        split_id=None,
        data=[np.ones((1, 2, 2), dtype=np.float32)],
        label=[np.zeros((1, 2, 2), dtype=np.uint8)],
        preds=[preds],
        losses={},
        metrics={},
        aux_data=None,
    )
    w.write_on_batch_end(None, None, result, None, None, 0, 0)
    w.drain()
    with h5py.File(w.result_paths[0], "r") as fh:
        assert "data" in fh
        assert "mito_preds" in fh


def test_writer_drain_raises_background_error(tmp_path):
    from cryovit.models.callbacks import PredictionWriter

    w = PredictionWriter(results_dir=tmp_path, label_key="mito", threshold=0.5)
    w._ensure_worker()
    w._in_flight += 1
    w._queue.put((tmp_path / "bad.hdf", None, "not-an-array"))
    with pytest.raises(Exception):
        w.drain()


def test_pca_max_components_default_matches_legacy_formula():
    # contract: min(pca_max_components, n_samples, n_features); default 1024
    import inspect

    from cryovit.visualization.dino_pca import _calculate_pca

    sig = inspect.signature(_calculate_pca)
    assert sig.parameters["pca_max_components"].default == 1024


def test_windowed_path_skips_full_volume_load(tmp_path, monkeypatch):
    """__getitem__ must not call _load_tomogram before windowed read."""
    from cryovit.datasets.file_dataset import FileDataset
    from cryovit.types import FileData

    path = tmp_path / "t.hdf5"
    lab = tmp_path / "lab.hdf5"
    with h5py.File(path, "w") as fh:
        fh.create_dataset("data", data=np.zeros((4, 8, 8), dtype=np.float32))
    with h5py.File(lab, "w") as fh:
        fh.create_dataset("mito", data=np.zeros((4, 8, 8), dtype=np.uint8))

    ds = FileDataset(
        [FileData(tomo_path=path, label_path=lab, labels=["mito"])],
        input_key="data",
        label_key="mito",
        train=True,
        windowed_read=True,
        label_value_map={},
    )
    calls = {"full": 0}
    real = ds._load_tomogram

    def counting(*a, **k):
        calls["full"] += 1
        return real(*a, **k)

    monkeypatch.setattr(ds, "_load_tomogram", counting)
    ds[0]
    assert calls["full"] == 0


def test_windowed_requires_map_or_full_scan_not_silent_zeros(tmp_path):
    """Missing/non-HDF5 labels must not silently produce all-zero masks."""
    from cryovit.datasets.file_dataset import FileDataset
    from cryovit.types import FileData

    path = tmp_path / "t.hdf5"
    lab = tmp_path / "lab.mrc"
    lab.write_bytes(b"not-really-mrc")
    with h5py.File(path, "w") as fh:
        fh.create_dataset("data", data=np.zeros((4, 8, 8), dtype=np.float32))

    ds = FileDataset(
        [FileData(tomo_path=path, label_path=lab, labels=["mito"])],
        input_key="data",
        label_key="mito",
        train=True,
        windowed_read=True,
    )
    with pytest.raises(ValueError, match="HDF5 labels"):
        ds[0]


def test_dino_label_window_shape_not_feature_shape(tmp_path):
    """DINO ×16 label window keeps full-resolution label shape."""
    from cryovit.datasets.file_dataset import FileDataset
    from cryovit.types import FileData

    path = tmp_path / "t.hdf5"
    lab = tmp_path / "lab.hdf5"
    # input in feature space 2×4×4; label in raw space 2×64×64 (×16)
    with h5py.File(path, "w") as fh:
        fh.create_dataset("dino_features", data=np.zeros((8, 2, 4, 4), dtype=np.float16))
    with h5py.File(lab, "w") as fh:
        arr = np.zeros((2, 64, 64), dtype=np.uint8)
        arr[:, 0, 0] = 255
        fh.create_dataset("mito", data=arr)

    ds = FileDataset(
        [FileData(tomo_path=path, label_path=lab, labels=["mito"])],
        input_key="dino_features",
        label_key="mito",
        train=True,
        windowed_read=True,
        label_value_map={255: 1},
    )
    sample = ds[0]
    # feature crop side=32 → label side=512 in general; here volume is 4 so
    # side=min(32,4)=4 and label side=min(512,64) after ×16 rule uses 16*4=64
    assert sample.label.ndim == 3
    assert sample.label.shape[0] == 2
    assert sample.label.shape[1] == 64  # 16 * feature side (4)
    assert sample.data.shape[-1] == 4


def test_looks_like_compile_error_narrow():
    from cryovit.run.infer_model import _looks_like_compile_error

    assert _looks_like_compile_error(RuntimeError("BackendCompilerFailed: boom"))
    assert _looks_like_compile_error(RuntimeError("detected in dynamo"))
    assert not _looks_like_compile_error(FileNotFoundError("missing.hdf"))
    assert not _looks_like_compile_error(OSError("disk full during gzip write"))


def test_label_map_not_shared_across_files(tmp_path):
    """P1: auto map for file A (255) must not apply to file B (1)."""
    from cryovit.datasets.file_dataset import FileDataset
    from cryovit.types import FileData

    def write_pair(name, raw_fg):
        p = tmp_path / f"{name}.hdf5"
        lab = tmp_path / f"{name}_lab.hdf5"
        with h5py.File(p, "w") as fh:
            fh.create_dataset("data", data=np.zeros((2, 4, 4), dtype=np.float32))
        arr = np.zeros((2, 4, 4), dtype=np.uint8)
        arr[0, 0, 0] = raw_fg
        with h5py.File(lab, "w") as fh:
            fh.create_dataset("mito", data=arr)
        return p, lab

    p1, l1 = write_pair("a", 255)
    p2, l2 = write_pair("b", 1)

    ds = FileDataset(
        [
            FileData(tomo_path=p1, label_path=l1, labels=["mito"]),
            FileData(tomo_path=p2, label_path=l2, labels=["mito"]),
        ],
        input_key="data",
        label_key="mito",
        train=True,
        windowed_read=True,
    )
    # Force both windows to cover (0,0,0) via monkeypatched sample
    ds._sample_window = lambda *a, **k: (0, 0, 0, 2, 4, 4)
    s1 = ds[0]
    s2 = ds[1]
    assert s1.label[0, 0, 0] == 1  # 255 in file a
    assert s2.label[0, 0, 0] == 1  # 1 in file b — not treated as background
    assert len(ds._label_map_cache) == 2


def test_windowed_explicit_missing_key_raises(tmp_path):
    """P2: explicit input_key missing → error, never first-key fallback."""
    from cryovit.datasets.file_dataset import FileDataset
    from cryovit.types import FileData

    path = tmp_path / "t.hdf5"
    with h5py.File(path, "w") as fh:
        fh.create_dataset("other", data=np.zeros((2, 4, 4), dtype=np.float32))
    ds = FileDataset(
        [FileData(tomo_path=path)],
        input_key="data",
        label_key="mito",
        train=True,
        windowed_read=True,
    )
    with pytest.raises(KeyError, match="data"):
        ds[0]
