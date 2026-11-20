"""Controlled augmentation ablations; one experiment variant per training run."""
from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from ultralytics import YOLO
from ..common import configure_logging, load_yaml, project_path, save_json, set_seed
from .dataset_validation import validate_train_val_separation

LOGGER = configure_logging(__name__)
VARIANTS = {
    # Each variant changes only the named augmentation(s), leaving all other
    # training settings shared so the comparison remains interpretable.
    "baseline": {},
    "no_mosaic": {"mosaic": 0.0},
    "no_mixup": {"mixup": 0.0},
    "no_fliplr": {"fliplr": 0.0},
    "reduced_augmentation": {"mosaic": 0.0, "mixup": 0.0, "copy_paste": 0.0,
                             "hsv_h": 0.0, "hsv_s": 0.0, "hsv_v": 0.0},
}

def run_ablation(config_path: str = "configs/experiments.yaml") -> pd.DataFrame:
    """Train controlled augmentation variants and compare their best validation scores.

    The configured variants are trained independently from the same base model
    initialization. Scores are taken from each run's best recorded validation
    epoch rather than comparing only the final epoch.
    """
    cfg = load_yaml(config_path)
    split_validation = validate_train_val_separation(project_path(cfg["data"]))
    LOGGER.info("Train/validation split preflight passed: %s", split_validation)
    set_seed(int(cfg.get("seed", 42)))
    study_name = str(cfg.get("tuning", {}).get("study_name", "ablation"))
    safe_name = "".join(char if char.isalnum() or char in "-_" else "_" for char in study_name)
    rows = []
    for variant in cfg["ablation"]["variants"]:
        if variant not in VARIANTS: raise ValueError(f"Unknown ablation variant: {variant}")
        LOGGER.info("Training ablation variant: %s", variant)
        model = YOLO(cfg["model"])
        # Merge shared training settings with variant overrides before calling
        # Ultralytics; passing duplicate keyword arguments raises TypeError.
        # Variant values intentionally override shared augmentation settings.
        train_args = {**cfg.get("train_args", {}), **VARIANTS[variant]}
        metrics = model.train(data=cfg["data"], epochs=int(cfg["epochs"]), imgsz=int(cfg["imgsz"]),
            batch=int(cfg["batch"]), workers=int(cfg["workers"]), seed=int(cfg.get("seed", 42)),
            project=str(project_path(f"experiments/ablation/{safe_name}")), name=variant, exist_ok=False,
            plots=True, **train_args)
        # Use the best epoch recorded by Ultralytics, not only the final epoch metrics.
        results_file = Path(model.trainer.save_dir) / "results.csv"
        if not results_file.is_file():
            raise FileNotFoundError(f"Training metrics not found: {results_file}")
        history = pd.read_csv(results_file)
        metric_column = next((column for column in history.columns
                              if "map50-95" in column.lower().replace("_", "-")), None)
        if metric_column is None:
            raise KeyError(f"mAP50-95 column not found in {results_file}")
        # Ignore malformed/missing CSV values and retain the strongest epoch score.
        score = float(pd.to_numeric(history[metric_column], errors="coerce").max())
        best = Path(model.trainer.save_dir) / "weights" / "best.pt"
        rows.append({"variant": variant, "map50_95": score, "best_checkpoint": str(best),
                     "changes": str(VARIANTS[variant])})
    frame = pd.DataFrame(rows)
    metrics_path = project_path(f"results/metrics/ablation_{safe_name}.csv")
    frame.to_csv(metrics_path, index=False)
    # Report absolute and relative changes against the baseline variant.
    baseline = float(frame.loc[frame.variant == "baseline", "map50_95"].iloc[0])
    frame["delta_vs_baseline"] = frame.map50_95 - baseline
    frame["delta_percent"] = frame.delta_vs_baseline / baseline * 100 if baseline else 0.0
    frame.to_csv(metrics_path, index=False)
    ax = frame.plot.bar(x="variant", y="map50_95", legend=False, rot=20, figsize=(9, 5))
    ax.set_ylabel("Best validation mAP50-95"); ax.set_title("Augmentation ablation study")
    ax.figure.tight_layout(); ax.figure.savefig(project_path(f"results/plots/ablation_{safe_name}.png"), dpi=160)
    plt.close(ax.figure)
    save_json({"dataset_split_validation": split_validation,
               "results": frame.to_dict(orient="records")},
              f"results/reports/ablation_{safe_name}.json")
    return frame

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiments.yaml")
    args = parser.parse_args()
    run_ablation(args.config)

