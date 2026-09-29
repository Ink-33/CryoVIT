import logging
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import torch
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.callbacks.prediction_writer import BasePredictionWriter
from pytorch_lightning.utilities.types import STEP_OUTPUT

from cryovit.perf_probe import probe_stage
from cryovit.types import BatchedModelResult


class TestPredictionWriter(Callback):
    """Callback to write predictions to disk during model evaluation."""

    def __init__(self, results_dir: Path, label_key: str, **kwargs) -> None:
        """Creates a callback to save predictions on the test data.

        Args:
            results_dir (Path): directory in which the predictions should be saved.
        """

        self.results_dir = (
            results_dir if isinstance(results_dir, Path) else Path(results_dir)
        )
        self.label_key = label_key

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Handles the end of a test batch to save outputs."""

        assert isinstance(outputs, BatchedModelResult)
        for n in range(outputs.num_tomos):
            output_file = (
                self.results_dir / outputs.samples[n] / outputs.tomo_names[n]
            )
            output_file.parent.mkdir(parents=True, exist_ok=True)

            # Binary classify predictions into segmentations and convert to correct formats
            data = outputs.data[n]
            labels = outputs.label[n]
            preds = outputs.preds[n]

            with h5py.File(output_file, "w") as fh:
                fh.create_dataset(
                    "data", data=data, shape=data.shape, dtype=data.dtype
                )
                fh.create_dataset(self.label_key, data=labels, shape=labels.shape, dtype=labels.dtype, compression="gzip")  # type: ignore
                fh.create_dataset(f"{self.label_key}_preds", data=preds, shape=preds.shape, dtype=preds.dtype, compression="gzip")  # type: ignore


class PredictionWriter(BasePredictionWriter):
    """Callback to write predictions to disk during model prediction.

    Writes overlap with compute on a single background thread (h5py is
    process-global and not multi-writer safe). ``drain()`` must be called before
    reading results; ``run_inference`` does this before returning.
    """

    def __init__(
        self,
        results_dir: Path,
        label_key: str,
        threshold: float,
        save_raw: bool = True,
        max_queue: int = 2,
        **kwargs,
    ) -> None:
        """Creates a callback to save predictions on the test data.

        Args:
            results_dir (Path): directory in which the predictions should be saved.
            save_raw (bool): also write the raw volume as ``data`` (format-compatible
                default). When False only ``{label_key}_preds`` is written.
            max_queue (int): bounded number of in-flight write jobs.
        """

        super().__init__(write_interval="batch", **kwargs)
        self.results_dir = (
            results_dir if isinstance(results_dir, Path) else Path(results_dir)
        )
        self.label_key = label_key
        self.threshold = threshold
        self.save_raw = save_raw
        self.result_paths = []
        self._max_queue = max_queue
        self._in_flight = 0
        self._errors: list[BaseException] = []
        self._thread = None
        self._queue = None

    def _ensure_worker(self) -> None:
        import queue
        import threading

        if self._thread is not None:
            return
        self._queue = queue.Queue(maxsize=self._max_queue)

        def _worker():
            while True:
                job = self._queue.get()
                if job is None:
                    self._queue.task_done()
                    return
                result_path, data, segs = job
                try:
                    self._write_file(result_path, data, segs)
                    self.result_paths.append(result_path)
                except BaseException as exc:  # noqa: BLE001
                    self._errors.append(exc)
                finally:
                    self._in_flight -= 1
                    self._queue.task_done()

        self._thread = threading.Thread(target=_worker, daemon=True, name="pred-writer")
        self._thread.start()

    @staticmethod
    def _to_numpy(tensor) -> np.ndarray:
        """Non-blocking GPU→CPU into a pinned buffer, then wait on an event."""
        with probe_stage("infer.d2h"):
            if not hasattr(tensor, "detach"):
                return np.asarray(tensor)
            t = tensor.detach()
            if not t.is_cuda:
                return t.cpu().numpy()
            try:
                pinned = torch.empty(t.shape, dtype=t.dtype, pin_memory=True)
                pinned.copy_(t, non_blocking=True)
                event = torch.cuda.Event()
                event.record()
                event.synchronize()
                return pinned.numpy()
            except Exception:
                return t.cpu().numpy()

    def _write_file(self, result_path: Path, data, segs) -> None:
        segs = self._to_numpy(segs)
        if data is not None:
            data = self._to_numpy(data)
        segs = (segs >= self.threshold).astype(np.uint8)
        if data is not None:
            data = data.astype(np.float32)
        # h5py writes are serialized on this single worker thread on purpose.
        with probe_stage("infer.write"):
            with h5py.File(result_path, "w") as fh:
                if self.save_raw and data is not None:
                    fh.create_dataset("data", data=data, compression="gzip")
                fh.create_dataset(
                    f"{self.label_key}_preds", data=segs, compression="gzip"
                )

    def drain(self) -> None:
        """Wait for all in-flight writes; re-raise the first background error."""
        if self._queue is not None and self._thread is not None:
            self._queue.join()
            self._queue.put(None)
            self._thread.join(timeout=5.0)
            self._thread = None
        if self._errors:
            raise self._errors[0]

    def write_on_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        prediction: BatchedModelResult,
        batch_indices,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """Queue the end of a prediction batch for asynchronous save."""

        if self._errors:
            raise self._errors[0]
        self._ensure_worker()
        preds = prediction.preds
        data_src = prediction.data
        for n in range(prediction.num_tomos):
            result_path = self.results_dir / prediction.tomo_names[n]
            result_path = result_path.with_suffix(".hdf")
            result_path.parent.mkdir(parents=True, exist_ok=True)
            raw = data_src[n] if (self.save_raw and data_src is not None) else None
            pred_n = preds[n]
            self._in_flight += 1
            self._queue.put((result_path, raw, pred_n))


class CsvWriter(Callback):
    """Callback to save model performance metrics to a .csv."""

    def __init__(self, results_dir: Path, **kwargs) -> None:
        """Creates a callback to save performance metrics on the test data.

        Args:
            results_dir (Path): directory in which the .csv files should be saved.
        """

        self.results_dir = (
            results_dir if isinstance(results_dir, Path) else Path(results_dir)
        )
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def _create_metric_df(
        self,
        sample: str,
        tomo_name: str,
        split_id: int | None = None,
        **metrics,
    ) -> pd.DataFrame:
        result_dict: dict[str, Any] = {
            "sample": sample,
            "tomo_name": tomo_name,
        }
        for name, value in metrics.items():
            result_dict[name] = [value]
        if split_id is not None:
            result_dict["split_id"] = [split_id]

        return pd.DataFrame(result_dict)

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Handles the end of a test batch to save metrics."""

        assert isinstance(outputs, BatchedModelResult)
        assert (
            outputs.num_tomos == 1
        ), "TestPredictionWriter only supports single-tomogram batches."
        sample, tomo_name, split_id = (
            outputs.samples[0],
            outputs.tomo_names[0],
            outputs.split_id[0] if outputs.split_id is not None else None,
        )

        csv_result_path = (
            self.results_dir
            / f"{sample}{'' if split_id is None else f'_{split_id}'}.csv"
        )

        # Create results .csv if it doesn't exist
        metric_names = list(outputs.metrics)
        column_names = ["sample", "tomo_name"] + metric_names
        if split_id is not None:
            column_names += ["split_id"]
        if not csv_result_path.exists():
            results_df = pd.DataFrame(columns=column_names)
        else:
            results_df = pd.read_csv(csv_result_path)

        # Warn if row already exists and remove (i.e., replace)
        matching_rows = (results_df["tomo_name"] == tomo_name) & (
            results_df["sample"] == sample
        )
        if split_id is not None and "split_id" in results_df.columns:
            matching_rows = matching_rows & (
                results_df["split_id"] == split_id
            )
        if matching_rows.any():
            logging.warning(
                "Data with sample %s, name %s, and split %s already has an entry. Replacing %d rows...",
                sample,
                tomo_name,
                split_id,
                matching_rows.sum(),
            )
            results_df = results_df[~matching_rows]
        # Add metrics to df
        metrics_df = self._create_metric_df(
            sample, tomo_name, split_id, **outputs.metrics
        )
        if results_df.empty:
            results_df = metrics_df
        else:
            results_df = pd.concat([results_df, metrics_df], ignore_index=True)
        results_df.to_csv(csv_result_path, mode="w", index=False)
