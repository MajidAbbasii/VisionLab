"""One-factor-at-a-time inference sensitivity analysis on a fixed validation set."""
from __future__ import annotations
import argparse
import re
import matplotlib.pyplot as plt
import pandas as pd
from ultralytics import YOLO
from ..common import configure_logging, load_yaml, project_path, save_json
from .dataset_validation import validate_train_val_separation

LOGGER = configure_logging(__name__)

def run_sensitivity(model_path: str | None = None,
                    config_path: str = "configs/experiments.yaml") -> pd.DataFrame:
    """Measure one-factor-at-a-time inference sensitivity on the configured val split.

    Confidence, IoU, and input size are varied independently while the other
    inference settings remain at their baseline values. The configured
    train/validation split is checked before any validation run is started.
    """
    cfg = load_yaml(config_path)
    split_validation = validate_train_val_separation(project_path(cfg["data"]))
    LOGGER.info("Train/validation split preflight passed: %s", split_validation)
    model_path = model_path or cfg.get("sensitivity", {}).get("model_path", "models/trained/best.pt")
    model = YOLO(str(project_path(model_path)))
    study_name = str(cfg.get("tuning", {}).get("study_name", "sensitivity"))
    safe_name = re.sub(r"[^A-Za-z0-9_-]+", "_", study_name)
    # Build a one-dimensional sweep for each parameter; do not combine values
    # into a Cartesian grid, which would change multiple factors at once.
    settings = [("conf", v) for v in cfg["sensitivity"]["conf_values"]]
    settings += [("iou", v) for v in cfg["sensitivity"]["iou_values"]]
    settings += [("imgsz", v) for v in cfg["sensitivity"]["imgsz_values"]]
    rows = []
    for parameter, value in settings:
        args = {"data": cfg["data"], "split": "val", "imgsz": int(cfg["imgsz"]),
                "conf": 0.25, "iou": 0.45, "batch": 1,
                "workers": int(cfg.get("workers", 2)), "device": 0,
                "verbose": False, "plots": False,
                "project": str(project_path(f"results/validation/sensitivity/{safe_name}")),
                "name": f"{parameter}_{value}", "exist_ok": False}
        # Override exactly one inference argument for this measurement.
        args[parameter] = value
        metrics = model.val(**args); box = metrics.box
        row = {"parameter": parameter, "value": value, "precision": float(box.mp),
               "recall": float(box.mr), "map50": float(box.map50), "map50_95": float(box.map)}
        rows.append(row)
        LOGGER.info("%s=%s => mAP50-95 %.4f", parameter, value, box.map)
    frame = pd.DataFrame(rows)
    metrics_path = project_path(f"results/metrics/sensitivity_{safe_name}.csv")
    frame.to_csv(metrics_path, index=False)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, parameter in zip(axes, ("conf", "iou", "imgsz")):
        subset = frame[frame.parameter == parameter]
        ax.plot(subset.value, subset.map50_95, marker="o")
        ax.set(title=f"{parameter} vs mAP50-95", xlabel=parameter, ylabel="mAP50-95")
        ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(project_path(f"results/plots/sensitivity_{safe_name}.png"), dpi=160)
    plt.close(fig)
    summary = {p: frame[frame.parameter == p].sort_values("map50_95", ascending=False).iloc[0].to_dict()
               for p in frame.parameter.unique()}
    save_json({"dataset_split_validation": split_validation, "best_by_parameter": summary},
              f"results/reports/sensitivity_{safe_name}.json")
    return frame

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model")
    parser.add_argument("--config", default="configs/experiments.yaml")
    args = parser.parse_args()
    run_sensitivity(args.model, args.config)

