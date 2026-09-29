"""Script for evaluating CryoVIT models based on configuration files."""

import logging
from pathlib import Path

import torch
from hydra import compose, initialize
from hydra.utils import instantiate

from cryovit.models.callbacks import PredictionWriter
from cryovit.perf_probe import probe_dump
from cryovit.utils import load_model

torch.set_float32_matmul_precision("high")

## For Scripts


_COMPILE_ERROR_MARKERS = (
    "dynamo",
    "inductor",
    "backendcompiler",
    "loweringexception",
    "compileerror",
    "torch.compiler",
    "recompilelimit",
)


def _looks_like_compile_error(exc: BaseException) -> bool:
    msg = f"{type(exc).__module__}.{type(exc).__name__}: {exc}".lower().replace(
        "_", ""
    )
    return any(marker.replace("_", "") in msg for marker in _COMPILE_ERROR_MARKERS)


def try_compile_forward(model: torch.nn.Module) -> torch.nn.Module:
    """Optionally wrap model.forward in torch.compile.

    Keeps the eager forward so first-execution compile failures can fall back
    without reconstructing the model.
    """
    if getattr(model, "_cryovit_compile_enabled", True) is False:
        return model
    eager_forward = model.forward
    try:
        model.forward = torch.compile(model.forward)  # type: ignore[method-assign]
        model._cryovit_eager_forward = eager_forward  # type: ignore[attr-defined]
        model._cryovit_compile_wrapped = True  # type: ignore[attr-defined]
    except Exception as exc:  # noqa: BLE001
        logging.warning("torch.compile unavailable at wrap time: %s", exc)
    return model


def restore_eager_forward(model: torch.nn.Module) -> torch.nn.Module:
    eager = getattr(model, "_cryovit_eager_forward", None)
    if eager is not None:
        model.forward = eager  # type: ignore[method-assign]
        model._cryovit_compile_wrapped = False  # type: ignore[attr-defined]
    return model


def run_inference(
    data_files: list[Path],
    model_path: Path,
    result_dir: Path,
    threshold: float = 0.5,
    save_raw: bool = True,
    use_compile: bool = False,
) -> list[Path]:
    """Run inference on the specified data files and saves the results.

    Args:
        data_files (list[Path]): List of paths to the input data files.
        model_path (Path): Path to the trained model file.
        result_dir (Path): Directory where the inference results will be saved.
        threshold (float, optional): Threshold for binary classification. Defaults to 0.5.
        save_raw (bool, optional): Write the raw volume into each result file.
            Defaults to True (output-format compatible). When False, only
            ``*_preds`` is written and the pipeline does not extra-load a raw
            volume just for saving.
        use_compile (bool, optional): Try torch.compile on forward. Defaults to
            False. First-execution failures fall back to eager.

    Returns:
        list[Path]: List of paths to the saved result files.
    """

    # Get model information
    model, model_type, model_name, label_key = load_model(model_path)
    assert model is not None, "Loaded model is None."
    ## Setup hydra config
    with initialize(
        version_base="1.2",
        config_path="../configs",
        job_name="infer_model",
    ):
        cfg = compose(
            config_name="infer_model",
            overrides=[
                f"name={model_name}",
                f"label_key={label_key}",
                f"model={model_type.value}",
                "datamodule=file",
            ],
        )
    cfg.paths.model_dir = Path(__file__).parent.parent / "foundation_models"
    cfg.paths.results_dir = result_dir

    # Check input key
    if cfg.model.input_key != "dino_features":
        cfg.model.input_key = None  # find available data instead

    ## Setup dataset
    dataset_fn = instantiate(cfg.datamodule.dataset)
    dataloader_fn = instantiate(cfg.datamodule.dataloader)
    datamodule = instantiate(cfg.datamodule, _convert_="all")(
        data_paths=data_files,
        val_paths=None,
        dataloader_fn=dataloader_fn,
        dataset_fn=dataset_fn,
        load_aux_raw=save_raw,
    )
    logging.info("Setup dataset.")

    ## Setup training
    pred_writer = PredictionWriter(
        results_dir=result_dir,
        label_key=label_key,
        threshold=threshold,
        save_raw=save_raw,
    )
    callbacks = [instantiate(cb_cfg) for cb_cfg in cfg.callbacks.values()]
    callbacks.append(pred_writer)
    loggers = []
    trainer = instantiate(cfg.trainer, callbacks=callbacks, logger=loggers)

    if use_compile and model_type.value != "sam2":
        model = try_compile_forward(model)

    logging.info("Starting prediction.")
    predict_err: BaseException | None = None
    try:
        trainer.predict(model, datamodule=datamodule)
    except BaseException as exc:  # noqa: BLE001
        predict_err = exc
    finally:
        try:
            pred_writer.drain()
        except BaseException as drain_exc:  # noqa: BLE001
            if predict_err is None:
                raise
            logging.error("writer drain failed after predict error: %s", drain_exc)

    if predict_err is not None:
        # Fallback only for compile-attributable failures AND only when nothing
        # has been written yet — never replay data/write errors or partial runs.
        if (
            getattr(model, "_cryovit_compile_wrapped", False)
            and _looks_like_compile_error(predict_err)
            and not pred_writer.result_paths
        ):
            logging.warning(
                "Compiled forward failed at first execution (%s); retrying eager.",
                predict_err,
            )
            model = restore_eager_forward(model)
            retry_err: BaseException | None = None
            try:
                trainer.predict(model, datamodule=datamodule)
            except BaseException as exc:  # noqa: BLE001
                retry_err = exc
            finally:
                try:
                    pred_writer.drain()
                except BaseException as drain_exc:  # noqa: BLE001
                    if retry_err is None:
                        raise
                    logging.error(
                        "writer drain failed after eager retry error: %s", drain_exc
                    )
            if retry_err is not None:
                raise retry_err
        else:
            raise predict_err

    probe_dump(result_dir / "perf_probe.jsonl", case="infer", model=str(model_path))
    result_paths = pred_writer.result_paths
    return result_paths
