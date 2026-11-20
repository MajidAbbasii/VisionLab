"""Reproducible, resumable YOLO hyperparameter search with Optuna."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import optuna
import pandas as pd
from ultralytics import YOLO

from ..common import configure_logging, load_yaml, project_path, save_json, set_seed
from .dataset_validation import validate_train_val_separation

LOGGER = configure_logging(__name__)
METRIC_TOKEN = "map50-95"


def _sha256(path: Path) -> str | None:
    """Return a file's SHA-256 digest, or None when the path is not a file."""
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _legacy_config_fingerprint(cfg: dict[str, Any]) -> str:
    """Fingerprint used by studies created before sampler/pruner were tracked."""
    relevant = {key: value for key, value in cfg.items() if key != "tuning"}
    relevant["search_space"] = cfg["tuning"]["search_space"]
    relevant["direction"] = cfg["tuning"].get("direction", "maximize")
    payload = json.dumps(relevant, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _config_fingerprint(cfg: dict[str, Any]) -> str:
    """Hash training and search settings to prevent incompatible study resumes."""
    relevant = {key: value for key, value in cfg.items() if key != "tuning"}
    relevant["search_space"] = cfg["tuning"]["search_space"]
    relevant["direction"] = cfg["tuning"].get("direction", "maximize")
    relevant["sampler"] = cfg["tuning"].get("sampler", {})
    relevant["pruner"] = cfg["tuning"].get("pruner", {})
    payload = json.dumps(relevant, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _environment_metadata(cfg: dict[str, Any]) -> dict[str, Any]:
    """Capture runtime, package, accelerator, dataset, and training provenance."""
    packages = {}
    for package in ("ultralytics", "torch", "optuna", "numpy", "pandas"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None

    gpu = None
    cuda_available = False
    cuda_version = None
    try:
        import torch
        cuda_available = bool(torch.cuda.is_available())
        cuda_version = torch.version.cuda
        if cuda_available:
            gpu = torch.cuda.get_device_name(0)
    except ImportError:
        pass

    data_path = project_path(cfg["data"])
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "cuda_available": cuda_available,
        "cuda_runtime_version": cuda_version,
        "gpu": gpu,
        "dataset_yaml": str(data_path),
        "dataset_yaml_sha256": _sha256(data_path),
        "model": cfg["model"],
        "optuna": {
            "sampler": cfg.get("tuning", {}).get("sampler", {}),
            "pruner": cfg.get("tuning", {}).get("pruner", {}),
        },
        "training_config": {
            key: value for key, value in cfg.items() if key != "tuning"
        },
    }


def _read_best_score(run_dir: Path) -> float:
    """Read the highest mAP50-95 value from an Ultralytics run's results.csv."""
    results_file = run_dir / "results.csv"
    if not results_file.is_file():
        raise FileNotFoundError(f"Ultralytics results.csv not found: {results_file}")

    frame = pd.read_csv(results_file)
    column = next(
        (name for name in frame.columns if METRIC_TOKEN in name.lower().replace("_", "-")),
        None,
    )
    if column is None:
        raise KeyError(f"mAP50-95 column not found in {results_file}; columns={list(frame.columns)}")
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if values.empty:
        raise ValueError(f"No valid mAP50-95 values found in {results_file}")
    return float(values.max())


def _validation_map50_95(metrics: dict[str, Any]) -> float | None:
    """Extract the validation mAP50-95 value exposed by an Ultralytics trainer."""
    for name, value in metrics.items():
        normalized = name.lower().replace("_", "-")
        if "map50-95" not in normalized:
            continue
        try:
            score = float(value)
        except (TypeError, ValueError):
            continue
        if pd.notna(score):
            return score
    return None


def _make_pruning_callback(trial: optuna.Trial, total_epochs: int):
    """Report each completed validation epoch and stop unpromising trials."""
    def on_fit_epoch_end(trainer: Any) -> None:
        """Report completed validation scores and prune weak trials when eligible."""
        epoch = int(getattr(trainer, "epoch", -1)) + 1
        # Do not prune on the final epoch: the trial has already paid the full cost.
        if epoch < 1 or epoch >= total_epochs:
            return

        score = _validation_map50_95(getattr(trainer, "metrics", {}) or {})
        if score is None:
            return

        trial.report(score, step=epoch)
        if trial.should_prune():
            trial.set_user_attr("pruned_epoch", epoch)
            trial.set_user_attr("pruned_map50_95", score)
            save_dir = getattr(trainer, "save_dir", None)
            if save_dir is not None:
                trial.set_user_attr("run_dir", str(save_dir))
            raise optuna.TrialPruned(
                f"Pruned after epoch {epoch}: validation mAP50-95={score:.5f}"
            )

    return on_fit_epoch_end


def _trial_records(study: optuna.Study) -> list[dict[str, Any]]:
    """Flatten Optuna trial state, timing, parameters, and metadata for CSV export."""
    records = []
    for trial in study.trials:
        duration = trial.duration.total_seconds() if trial.duration else None
        record = {
            "trial_number": trial.number,
            "state": trial.state.name,
            "objective_map50_95": trial.value,
            "datetime_start": trial.datetime_start.isoformat() if trial.datetime_start else None,
            "datetime_complete": trial.datetime_complete.isoformat() if trial.datetime_complete else None,
            "duration_seconds": duration,
            "params_json": json.dumps(trial.params, ensure_ascii=False, sort_keys=True),
            "intermediate_values_json": json.dumps(trial.intermediate_values, sort_keys=True),
            "user_attrs_json": json.dumps(trial.user_attrs, ensure_ascii=False, sort_keys=True, default=str),
        }
        record.update({f"param_{key}": value for key, value in trial.params.items()})
        records.append(record)
    return records


def _export_artifacts(study: optuna.Study, cfg: dict[str, Any], db_path: Path) -> dict[str, Any]:
    """Persist trial tables, best checkpoint, provenance, summary JSON, and plots."""
    study_name = study.study_name
    safe_name = "".join(char if char.isalnum() or char in "-_" else "_" for char in study_name)

    metrics_path = project_path(f"results/metrics/optuna_{safe_name}_trials.csv")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    trials_frame = pd.DataFrame(_trial_records(study))
    trials_frame.to_csv(metrics_path, index=False)
    # Keep the consolidated report's historical input path available as well.
    trials_frame.to_csv(project_path("results/metrics/hyperparameter_trials.csv"), index=False)

    completed = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    result: dict[str, Any] = {
        "study_name": study_name,
        "storage": str(db_path),
        "total_trials": len(study.trials),
        "completed_trials": len(completed),
        "failed_trials": sum(t.state == optuna.trial.TrialState.FAIL for t in study.trials),
        "pruned_trials": sum(t.state == optuna.trial.TrialState.PRUNED for t in study.trials),
        "trials_csv": str(metrics_path),
        "sampler": study.user_attrs.get("sampler_config"),
        "pruner": study.user_attrs.get("pruner_config"),
    }

    if completed:
        best_trial = study.best_trial
        result.update({
            "best_trial": best_trial.number,
            "best_map50_95": best_trial.value,
            "best_params": best_trial.params,
            "best_run_dir": best_trial.user_attrs.get("run_dir"),
        })
        best_run = best_trial.user_attrs.get("run_dir")
        if best_run:
            checkpoint = Path(best_run) / "weights" / "best.pt"
            if checkpoint.is_file():
                destination = project_path(f"models/trained/best_tuned_{safe_name}.pt")
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(checkpoint, destination)
                result["best_checkpoint"] = str(destination)

    save_json(result, f"results/reports/hyperparameter_tuning_{safe_name}.json")
    save_json(_environment_metadata(cfg), f"results/reports/optuna_{safe_name}_environment.json")

    if completed:
        import matplotlib.pyplot as plt
        from optuna.visualization.matplotlib import plot_optimization_history

        plot_dir = project_path(f"results/plots/optuna/{safe_name}")
        plot_dir.mkdir(parents=True, exist_ok=True)
        plotters = [("optimization_history.png", plot_optimization_history)]
        if len(completed) >= 2:
            from optuna.visualization.matplotlib import plot_param_importances
            plotters.append(("parameter_importance.png", plot_param_importances))
        for filename, plotter in plotters:
            try:
                ax = plotter(study)
                ax.figure.tight_layout()
                ax.figure.savefig(plot_dir / filename, dpi=160)
                plt.close(ax.figure)
            except Exception as exc:
                LOGGER.warning("Could not create %s: %s", filename, exc)

    LOGGER.info("Optuna artifacts exported: %s", result)
    return result


def run_tuning(
    config_path: str = "configs/experiments.yaml",
    trials: int | None = None,
    overrides: dict[str, Any] | None = None,
    study_name_override: str | None = None,
) -> dict[str, Any]:
    """Run or resume a reproducible Optuna study for YOLO training.

    The study is bound to a fingerprint of its dataset and search/training
    configuration. Existing compatible studies can resume; incompatible
    configurations must use a different study name. Each trial trains a fresh
    model, records its run metadata, and returns its best validation mAP50-95.
    """
    cfg = load_yaml(config_path)
    cfg.update(overrides or {})
    tuning = cfg["tuning"]
    ranges = tuning["search_space"]
    # Fail fast before creating/resuming a study or starting any GPU work.
    split_validation = validate_train_val_separation(project_path(cfg["data"]))
    LOGGER.info("Train/validation split preflight passed: %s", split_validation)
    set_seed(int(cfg.get("seed", 42)))

    study_name = str(study_name_override or tuning.get("study_name", "visionlab_yolo_tuning"))
    db_path = project_path(tuning.get("storage", "experiments/tuning/optuna.db"))
    db_path.parent.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{db_path.as_posix()}"
    fingerprint = _config_fingerprint(cfg)

    existing_names = optuna.study.get_all_study_names(storage=storage)
    existing_study = (
        optuna.load_study(study_name=study_name, storage=storage)
        if study_name in existing_names else None
    )
    legacy_mode = False
    if existing_study is not None:
        stored_fingerprint = existing_study.user_attrs.get("config_fingerprint")
        has_optimizer_metadata = (
            "sampler_config" in existing_study.user_attrs
            or "pruner_config" in existing_study.user_attrs
        )
        legacy_mode = (
            not has_optimizer_metadata
            and stored_fingerprint == _legacy_config_fingerprint(cfg)
        )

    sampler_cfg = tuning.get("sampler", {})
    sampler_type = str(sampler_cfg.get("type", "tpe")).lower()
    if sampler_type != "tpe":
        raise ValueError(f"Unsupported Optuna sampler: {sampler_type}")
    pruner_cfg = tuning.get("pruner", {})
    pruner_type = str(pruner_cfg.get("type", "median")).lower()
    if pruner_type != "median":
        raise ValueError(f"Unsupported Optuna pruner: {pruner_type}")

    if legacy_mode:
        # Preserve Optuna's original defaults for the existing legacy study:
        # unseeded TPESampler and NopPruner. Do not silently change its search policy.
        sampler = optuna.samplers.TPESampler()
        pruner = optuna.pruners.NopPruner()
        LOGGER.info("Resuming legacy study with its original Optuna sampler/pruner defaults")
    else:
        sampler = optuna.samplers.TPESampler(
            seed=int(sampler_cfg.get("seed", cfg.get("seed", 42))),
            n_startup_trials=int(sampler_cfg.get("n_startup_trials", 10)),
            n_ei_candidates=int(sampler_cfg.get("n_ei_candidates", 24)),
        )
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=int(pruner_cfg.get("n_startup_trials", 5)),
            n_warmup_steps=int(pruner_cfg.get("n_warmup_steps", 15)),
            interval_steps=int(pruner_cfg.get("interval_steps", 1)),
        )

    study = optuna.create_study(
        direction=tuning.get("direction", "maximize"),
        study_name=study_name,
        storage=storage,
        load_if_exists=bool(tuning.get("resume", True)),
        sampler=sampler,
        pruner=pruner,
    )
    existing_fingerprint = study.user_attrs.get("config_fingerprint")
    current_fingerprint = _config_fingerprint(cfg)
    legacy_fingerprint = _legacy_config_fingerprint(cfg)
    if existing_fingerprint and existing_fingerprint not in {
        current_fingerprint, legacy_fingerprint if legacy_mode else current_fingerprint
    }:
        raise ValueError(
            "This study already exists with a different training/search configuration. "
            "Use a new tuning.study_name to keep experiments separate."
        )
    if not existing_fingerprint:
        study.set_user_attr("config_fingerprint", current_fingerprint)
    study.set_user_attr("dataset_yaml_sha256", _sha256(project_path(cfg["data"])))
    study.set_user_attr("dataset_split_validation", split_validation)
    if not legacy_mode:
        study.set_user_attr("sampler_config", {
            "type": sampler_type,
            "seed": int(sampler_cfg.get("seed", cfg.get("seed", 42))),
            "n_startup_trials": int(sampler_cfg.get("n_startup_trials", 10)),
            "n_ei_candidates": int(sampler_cfg.get("n_ei_candidates", 24)),
        })
        study.set_user_attr("pruner_config", {
            "type": pruner_type,
            "n_startup_trials": int(pruner_cfg.get("n_startup_trials", 5)),
            "n_warmup_steps": int(pruner_cfg.get("n_warmup_steps", 15)),
            "interval_steps": int(pruner_cfg.get("interval_steps", 1)),
        })

    def objective(trial: optuna.Trial) -> float:
        """Train one sampled configuration and return its best validation score."""
        params = {
            "lrf": trial.suggest_float("lrf", *ranges["lrf"], log=True),
            "weight_decay": trial.suggest_float("weight_decay", *ranges["weight_decay"], log=True),
            "warmup_epochs": trial.suggest_float("warmup_epochs", *ranges["warmup_epochs"]),
            "hsv_h": trial.suggest_float("hsv_h", *ranges["hsv_h"]),
            "hsv_s": trial.suggest_float("hsv_s", *ranges["hsv_s"]),
            "hsv_v": trial.suggest_float("hsv_v", *ranges["hsv_v"]),
            "mosaic": trial.suggest_float("mosaic", *ranges["mosaic"]),
            "mixup": trial.suggest_float("mixup", *ranges["mixup"]),
        }
        run_dir: Path | None = None
        try:
            model = YOLO(str(cfg["model"]))
            train_args = {
                "data": cfg["data"],
                "epochs": int(cfg["epochs"]),
                "imgsz": int(cfg["imgsz"]),
                "batch": int(cfg["batch"]),
                "workers": int(cfg["workers"]),
                "project": str(project_path("experiments/tuning/runs")),
                "name": f"{study_name}_trial_{trial.number:04d}",
                "exist_ok": False,
                "plots": False,
                "seed": int(cfg.get("seed", 42)),
                "verbose": False,
                **cfg.get("train_args", {}),
                **params,
            }
            trial.set_user_attr("training_args", train_args)
            model.add_callback(
                "on_fit_epoch_end",
                _make_pruning_callback(trial, int(cfg["epochs"])),
            )
            model.train(**train_args)
            run_dir = Path(model.trainer.save_dir)
            trial.set_user_attr("run_dir", str(run_dir))
            score = _read_best_score(run_dir)
            trial.set_user_attr("best_epoch_metric", score)
            checkpoint = run_dir / "weights" / "best.pt"
            trial.set_user_attr("checkpoint_exists", checkpoint.is_file())
            return score
        except optuna.TrialPruned:
            raise
        except Exception as exc:
            trial.set_user_attr("error", f"{type(exc).__name__}: {exc}")
            trainer = getattr(locals().get("model"), "trainer", None)
            failed_run_dir = getattr(trainer, "save_dir", None)
            if failed_run_dir is not None:
                trial.set_user_attr("run_dir", str(failed_run_dir))
            if run_dir is not None:
                trial.set_user_attr("run_dir", str(run_dir))
            LOGGER.exception("Optuna trial %s failed", trial.number)
            raise

    requested_trials = int(trials if trials is not None else tuning.get("trials", 1))
    if requested_trials < 1:
        raise ValueError("trials must be at least 1")
    study.optimize(
        objective,
        n_trials=requested_trials,
        timeout=tuning.get("timeout_seconds"),
        gc_after_trial=True,
        catch=(Exception,),
    )
    return _export_artifacts(study, cfg, db_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiments.yaml")
    parser.add_argument("--trials", type=int)
    parser.add_argument("--epochs", type=int, help="Temporary epoch override for a smoke test")
    parser.add_argument("--batch", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--study-name", help="Use a separate study name for smoke tests or new experiments")
    args = parser.parse_args()
    overrides = {key: value for key, value in {
        "epochs": args.epochs, "batch": args.batch, "workers": args.workers
    }.items() if value is not None}
    run_tuning(
        config_path=args.config,
        trials=args.trials,
        overrides=overrides,
        study_name_override=args.study_name,
    )
