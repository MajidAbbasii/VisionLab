"""Train a YOLO detector; preserve the best checkpoint and training metrics."""
from __future__ import annotations
import argparse
import shutil
from pathlib import Path
import pandas as pd
from ultralytics import YOLO
from .common import configure_logging, load_yaml, project_path, set_seed
from .experiments.dataset_validation import validate_train_val_separation

LOGGER = configure_logging(__name__)

def train(config_path: str = "configs/training.yaml", overrides: dict | None = None) -> Path:
    """Train from YAML settings, then export the best checkpoint and epoch metrics."""
    cfg = load_yaml(config_path)
    cfg.update(overrides or {})
    checkpoint_output = cfg.pop("checkpoint_output", "models/trained/best.pt")
    metrics_output = cfg.pop("metrics_output", "results/metrics/training_metrics.csv")
    split_validation = validate_train_val_separation(project_path(cfg["data"]))
    LOGGER.info("Train/validation split preflight passed: %s", split_validation)
    set_seed(int(cfg.get("seed", 42)))
    model = YOLO(str(cfg["model"]))
    # Pass the seed to Ultralytics as well as seeding Python/NumPy/PyTorch above.
    model.train(**{k: v for k, v in cfg.items() if k != "model"})
    # Ultralytics returns metrics; its trainer owns the actual run directory.
    run_dir = Path(model.trainer.save_dir)
    best = run_dir / "weights" / "best.pt"
    if not best.exists(): raise FileNotFoundError(f"Training checkpoint not found: {best}")
    destination = project_path(checkpoint_output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, destination)
    csv = run_dir / "results.csv"
    if csv.exists():
        metrics_target = project_path(metrics_output)
        metrics_target.parent.mkdir(parents=True, exist_ok=True)
        pd.read_csv(csv).to_csv(metrics_target, index=False)
    LOGGER.info("Best checkpoint copied to %s", destination)
    return destination

def main() -> None:
    """Parse training overrides and start the configured YOLO training run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/training.yaml")
    parser.add_argument("--data")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--model")
    args = parser.parse_args()
    overrides = {k: v for k, v in {"data": args.data, "epochs": args.epochs,
                                    "model": args.model}.items() if v is not None}
    train(args.config, overrides)

if __name__ == "__main__": main()

