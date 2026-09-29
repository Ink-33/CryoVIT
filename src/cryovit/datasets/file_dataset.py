"""Dataset class for loading tomograms for CryoViT scripts."""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from torch.utils.data import Dataset
from torchvision.transforms import Normalize

from cryovit.config import DINO_PATCH_SIZE
from cryovit.datasets.vit_dataset import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
)
from cryovit.types import FileData, TomogramData
from cryovit.utils import load_data, load_labels


class FileDataset(Dataset):
    """A dataset class for handling and preprocessing tomographic data for CryoVIT models."""

    def __init__(
        self,
        files: list[FileData],
        input_key: str | None,
        label_key: str | None,
        train: bool = False,
        for_dino: bool = False,
        use_sam: bool = False,
        load_aux_raw: bool = True,
        windowed_read: bool = False,
        label_value_map: dict[int, int] | None = None,
        feat_noise_std: float = 0.0,
        feat_gain_jitter: float = 0.0,
    ) -> None:
        """Creates a new FileDataset object.

        Args:
            files: A list of FileData objects containing file paths and metadata.
            input_key: The key in a HDF5 file to access input features.
            label_key: The key in a HDF5 file to access labels.
            train: Enables random crop transformations.
            for_dino: Enables DINO transformations.
            use_sam: Enables SAM transformations.
            load_aux_raw: When False, eval/predict does not extra-load a raw
                volume just for saving/visualization. Model-required inputs
                are still read.
            windowed_read: When True and the source is HDF5, training crops are
                window-read instead of loading the full volume first.
            label_value_map: Global raw_value→class_id map for windowed labels.
                Must come from config or a full-volume scan — never per-crop.
        """

        self.files = files
        self.input_key = input_key
        self.label_key = label_key
        self.train = train
        self.for_dino = for_dino
        self.use_sam = use_sam
        self.load_aux_raw = load_aux_raw
        self.windowed_read = windowed_read
        self.label_value_map = label_value_map
        self.feat_noise_std = float(feat_noise_std)
        self.feat_gain_jitter = float(feat_gain_jitter)

        self.transform = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
        self._key_cache = {}
        # Auto-inferred maps are cached per (label_path, label_key, labels-tuple).
        # Explicit self.label_value_map is never overwritten by inference.
        self._label_map_cache: dict[tuple, dict[int, int]] = {}
        self._printed_resize_warning = False

    def __len__(self) -> int:
        """Returns the total number of tomograms in the dataset."""

        return len(self.files)

    def __getitem__(self, idx: int) -> TomogramData:  # type: ignore
        """Retrieves a single item from the dataset.

        Args:
            idx (int): The index of the item.

        Returns:
            TomogramData: A dataclass containing the loaded data, labels, and metadata.

        Raises:
            IndexError: If index is out of the range of the dataset.
        """

        if idx >= len(self):
            raise IndexError

        file_data = self.files[idx]
        aux_data = {}

        # Windowed training path must not load the full volume first.
        if (
            self.train
            and self.windowed_read
            and self._can_window_read(file_data)
            and not self.for_dino
        ):
            data = self._windowed_load_tomogram(file_data)
            return TomogramData(
                sample=file_data.sample,
                tomo_name=file_data.tomo_path.name,
                split_id=None,
                data=self._augment_features(data["input"]) if self.train else data["input"],
                label=data["label"],
                aux_data=aux_data,
            )  # type: ignore

        data = self._load_tomogram(file_data)

        if self.for_dino:
            feature_data = (
                self._sam_transform(data["input"])
                if self.use_sam
                else self._dino_transform(data["input"])
            )
            return TomogramData(
                sample=file_data.sample,
                tomo_name=file_data.tomo_path.name,
                split_id=None,
                data=feature_data,
                label=torch.zeros(
                    data["input"].shape, dtype=torch.bool
                ),  # dummy label,
                aux_data={"data": data["input"].squeeze(0)},
            )  # type: ignore
        if self.train:
            self._random_crop(data)
        elif not self.train and self.load_aux_raw:
            # Aux raw volume reuses the already-loaded array only when the
            # resolved dataset key IS "data" (same file + same key + same
            # load_data conversion). Shape must not decide reuse.
            resolved_key = data.get("resolved_key")
            if resolved_key == "data":
                aux_data = {"data": data["input"].squeeze(0)}
            else:
                aux_data = {
                    "data": load_data(file_data.tomo_path, key="data")[
                        0
                    ].squeeze(0)
                }

        return TomogramData(
            sample=file_data.sample,
            tomo_name=file_data.tomo_path.name,
            split_id=None,
            data=self._augment_features(data["input"]) if self.train else data["input"],
            label=data["label"],
            aux_data=aux_data,
        )  # type: ignore


    def _augment_features(self, x):
        """Train-time feature-space jitter: gain + Gaussian noise.

        Proxy for contrast/dose variation when DINO features are precomputed.
        """
        if self.feat_gain_jitter <= 0 and self.feat_noise_std <= 0:
            return x
        if not torch.is_tensor(x):
            return x
        y = x.clone()
        if self.feat_gain_jitter > 0:
            g = 1.0 + (torch.rand((), device=y.device, dtype=y.dtype) * 2 - 1) * self.feat_gain_jitter
            y = y * g
        if self.feat_noise_std > 0:
            y = y + torch.randn_like(y) * self.feat_noise_std
        return y

    def _load_tomogram(self, file_data: FileData) -> dict[str, Any]:
        """Loads a single tomogram based on the file information.

        Args:
            file_data (FileData): An object containing the file paths and metadata.

        Returns:
            data (dict[str, Any]): A dictionary with input data, label, and any auxiliary data.
        """

        tomo_path = file_data.tomo_path
        label_path = file_data.label_path

        # Cache the resolved dataset key for each tomogram to avoid redundant reads
        if tomo_path in self._key_cache:
            resolved_key = self._key_cache[tomo_path]
            data, _ = load_data(tomo_path, key=resolved_key)
        else:
            data, resolved_key = load_data(tomo_path, key=self.input_key)
            self._key_cache[tomo_path] = resolved_key
        labels = (
            load_labels(
                label_path, label_keys=file_data.labels, key=self.label_key
            )
            if label_path is not None and file_data.labels is not None
            else None
        )
        assert data is not None, f"Failed to load data from {tomo_path}"

        data_dict = {
            "input": data,
            "resolved_key": resolved_key,
            "label": (
                labels[self.label_key]
                if labels is not None and self.label_key is not None
                else np.zeros(
                    (1, *data.shape[1:]), dtype=np.int8
                )  # replace channel
            ),
        }
        return data_dict

    def _random_crop(self, data: dict[str, Any]) -> None:
        """Applies a random crop to the input data in the record dictionary.

        Args:
            data (dict[str, Any]): The record dictionary containing 'input' and 'label' data.
        """

        max_depth = 128
        side = 32 if self.input_key == "dino_features" else 512
        d, h, w = data["input"].shape[-3:]
        x, y, z = min(d, max_depth), side, side

        if (d, h, w) == (x, y, z):
            return  # nothing to be done

        delta_d = d - x + 1
        delta_h = h - y + 1
        delta_w = w - z + 1

        di = np.random.choice(delta_d) if delta_d > 0 else 0
        hi = np.random.choice(delta_h) if delta_h > 0 else 0
        wi = np.random.choice(delta_w) if delta_w > 0 else 0

        data["input"] = data["input"][
            ..., di : di + x, hi : hi + y, wi : wi + z
        ]

        if self.input_key == "dino_features":
            hi, wi, y, z = 16 * np.array([hi, wi, y, z])

        data["label"] = data["label"][di : di + x, hi : hi + y, wi : wi + z]

    def _can_window_read(self, file_data: FileData) -> bool:
        return (
            file_data.tomo_path.suffix in {".h5", ".hdf", ".hdf5"}
            and self.input_key is not None
        )

    def _sample_window(self, d: int, h: int, w: int):
        max_depth = 128
        side = 32 if self.input_key == "dino_features" else 512
        x, y, z = min(d, max_depth), min(side, h), min(side, w)
        delta_d = d - x + 1
        delta_h = h - y + 1
        delta_w = w - z + 1
        di = int(np.random.choice(delta_d)) if delta_d > 0 else 0
        hi = int(np.random.choice(delta_h)) if delta_h > 0 else 0
        wi = int(np.random.choice(delta_w)) if delta_w > 0 else 0
        return di, hi, wi, x, y, z

    def _ensure_label_value_map(
        self, label_path, label_keys: list[str] | None = None
    ) -> dict[int, int]:
        """Return raw_value→class_id for this label file.

        Explicit ``label_value_map`` (config) is always used as-is. Auto-inferred
        maps are cached **per label file + dataset key + label key order** so
        volumes with different raw encodings (e.g. 255 vs 1) do not share one
        map.
        """
        if self.label_value_map is not None:
            return self.label_value_map
        from cryovit.utils import read_hdf

        cache_key = (
            Path(label_path),
            self.label_key,
            tuple(label_keys) if label_keys is not None else (self.label_key,),
        )
        if cache_key in self._label_map_cache:
            return self._label_map_cache[cache_key]

        if Path(label_path).suffix not in {".h5", ".hdf", ".hdf5"}:
            raise ValueError(
                "windowed_read requires HDF5 labels or an explicit label_value_map; "
                f"got {Path(label_path).suffix}"
            )
        _, lab, _ = read_hdf(label_path, key=self.label_key, compute_stats=False)
        raw_values = [int(v) for v in np.unique(lab) if v not in (0, -1)]
        # Ascending raw values → class_id 1..N matching label_keys order.
        inferred = {
            raw_val: class_id for class_id, raw_val in enumerate(raw_values, start=1)
        }
        self._label_map_cache[cache_key] = inferred
        return inferred

    def _windowed_load_tomogram(self, file_data: FileData) -> dict[str, Any]:
        """Sample a crop window and read only that window from HDF5.

        Label values are mapped with the global ``label_value_map`` (never a
        crop-local unique). Returned arrays are independent copies.
        """
        import h5py

        from cryovit.utils import apply_label_value_map, read_hdf_window

        tomo_path = file_data.tomo_path
        if tomo_path in self._key_cache:
            resolved_key = self._key_cache[tomo_path]
        else:
            with h5py.File(tomo_path, "r") as fh:
                if self.input_key is not None:
                    # Explicit key: missing key must error.
                    if self.input_key not in fh:
                        raise KeyError(
                            f"Key '{self.input_key}' not found in {tomo_path}. "
                            f"Top-level keys: {list(fh.keys())}"
                        )
                    resolved_key = self.input_key
                else:
                    resolved_key = next(iter(fh.keys()))
            self._key_cache[tomo_path] = resolved_key

        with h5py.File(tomo_path, "r") as fh:
            shape = fh[resolved_key].shape
        d, h, w = shape[-3:]
        di, hi, wi, x, y, z = self._sample_window(d, h, w)
        win = (slice(di, di + x), slice(hi, hi + y), slice(wi, wi + z))

        inp = read_hdf_window(tomo_path, resolved_key, win)
        # Independent copy so the HDF5 window can release its base.
        inp = np.ascontiguousarray(inp).copy()
        if inp.dtype == np.uint8:
            inp = inp.astype(np.float32) / 255.0
        elif inp.dtype in (np.int8, np.uint16, np.int16):
            inp = inp.astype(np.float32)
        if inp.ndim == 3:
            inp = inp[np.newaxis, ...]

        # Label window: same spatial rule as _random_crop (DINO uses ×16 H/W).
        l_di, l_hi, l_wi = di, hi, wi
        l_x, l_y, l_z = x, y, z
        if self.input_key == "dino_features":
            l_hi, l_wi, l_y, l_z = 16 * np.array([hi, wi, y, z], dtype=int)
        lwin = (
            slice(l_di, l_di + l_x),
            slice(int(l_hi), int(l_hi) + int(l_y)),
            slice(int(l_wi), int(l_wi) + int(l_z)),
        )

        label = np.zeros((int(l_x), int(l_y), int(l_z)), dtype=np.int8)
        if file_data.label_path is not None and file_data.labels is not None:
            if self.label_key is None:
                raise ValueError("windowed_read requires label_key")
            value_map = self._ensure_label_value_map(
                file_data.label_path, file_data.labels
            )
            if file_data.label_path.suffix in {".h5", ".hdf", ".hdf5"}:
                raw = np.ascontiguousarray(
                    read_hdf_window(file_data.label_path, self.label_key, lwin)
                ).copy()
                mapped = apply_label_value_map(
                    raw, file_data.labels, value_map
                )
                label = mapped[self.label_key].copy()
            else:
                raise ValueError(
                    "windowed_read currently supports HDF5 labels only; "
                    f"got {file_data.label_path.suffix}"
                )

        return {
            "input": inp,
            "resolved_key": resolved_key,
            "label": label,
        }

    def _dino_transform(self, data: NDArray[np.float32]) -> torch.Tensor:
        """Applies normalization and resizing transformations to the tomogram.

        Args:
            data (NDArray[np.float32]): The loaded tomogram data as a numpy array.

        Returns:
            torch.Tensor: The transformed data as a PyTorch tensor.
        """

        scale = (DINO_PATCH_SIZE / 16, DINO_PATCH_SIZE / 16)
        h, w = data.shape[-2:]
        # Resize height and width to be multiples of 16
        H = int(np.ceil(h / 16) * 16)
        W = int(np.ceil(w / 16) * 16)
        if h != H or w != W:
            if not self._printed_resize_warning:
                logging.warning(
                    "Resizing tomogram from %s to %s", (h, w), (H, W)
                )
                self._printed_resize_warning = True
            data = np.pad(
                data, ((0, 0), (0, 0), (0, H - h), (0, W - w)), mode="edge"
            )
            h, w = H, W
        assert (
            h % 16 == 0 and w % 16 == 0
        ), f"Invalid height: {h} or width: {w}"

        np_data = data.transpose((1, 0, 2, 3))  # D, C, H, W
        torch_data = torch.from_numpy(np_data).float()  # D, C, H, W
        # expand (not repeat): 3-channel view without a 3× copy — Normalize
        # materializes on write.
        torch_data = torch_data.expand(-1, 3, -1, -1)
        torch_data: torch.Tensor = self.transform(torch_data)  # D, C, H, W
        torch_data: torch.Tensor = F.interpolate(
            torch_data, scale_factor=scale, mode="bicubic"
        )
        return torch_data

    def _sam_transform(self, data: NDArray[np.float32]) -> torch.Tensor:
        """Applies normalization and resizing transformations to the tomogram for SAM model.

        Args:
            data (NDArray[np.float32]): Loaded tomogram as (D,H,W) or (C,D,H,W).

        Returns:
            torch.Tensor: The transformed data as a PyTorch tensor (B,D,C,H,W).
        """

        data = np.asarray(data)
        if data.ndim == 4:
            # load_data adds a channel axis — SAM slice stack wants D,H,W
            data = data[0] if data.shape[0] == 1 else data[:, 0]
        _, h, w = data.shape
        # (D,H,W) -> (B,D,C,H,W); match VITDataset (axis 0 and 2)
        np_data = np.expand_dims(data, axis=(0, 2))
        torch_data = torch.from_numpy(np.ascontiguousarray(np_data)).float()
        torch_data = torch_data.expand(-1, -1, 3, -1, -1)
        # expand shares storage — pin_memory requires an owned tensor
        return torch_data.contiguous()
