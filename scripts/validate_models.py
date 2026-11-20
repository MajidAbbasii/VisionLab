"""Evaluate the baseline and tuned checkpoints on the same COCO 2017 val split."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ultralytics import YOLO


def evaluate(model_path: Path, data: Path, name: str, device: str, batch: int) -> dict[str, Any]:
    if not model_path.is_file():
        raise FileNotFoundError(model_path)

    model = YOLO(str(model_path))
    metrics = model.val(
        data=str(data),
        split="val",
        imgsz=640,
        batch=batch,
        workers=2,
        device=device,
        conf=0.001,
        iou=0.7,
        max_det=300,
        plots=True,
        save_json=False,
        # Resolve against the repository root; Ultralytics otherwise uses its
        # global runs_dir and writes experiment artifacts outside the project.
        project=str(Path("results/validation/coco2017").resolve()),
        name=name,
        exist_ok=False,
        verbose=True,
    )
    return {
        "checkpoint": str(model_path),
        "run_dir": str(metrics.save_dir),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "per_class_map50_95": [float(value) for value in metrics.box.maps],
        "speed_ms_per_image": {key: float(value) for key, value in metrics.speed.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("datasets/coco2017.yaml"))
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--models", nargs=2, type=Path, default=[
        Path("models/trained/best.pt"),
        Path("models/trained/best_tuned_visionlab_yolo11n_coco128_v1.pt"),
    ])
    args = parser.parse_args()

    results = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(args.data),
        "protocol": {
            "split": "val", "imgsz": 640, "batch": args.batch,
            "workers": 2, "device": args.device, "conf": 0.001,
            "iou": 0.7, "max_det": 300,
        },
        "models": {},
    }
    for model_path, name in zip(args.models, ("baseline", "optuna_tuned")):
        print(f"\n=== Evaluating {name}: {model_path} ===", flush=True)
        results["models"][name] = evaluate(model_path, args.data, name, args.device, args.batch)

    baseline = results["models"]["baseline"]["map50_95"]
    tuned = results["models"]["optuna_tuned"]["map50_95"]
    results["comparison"] = {"map50_95_delta_tuned_minus_baseline": tuned - baseline}
    output = Path("results/reports/coco2017_independent_validation.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved comparison report: {output}", flush=True)
    print(json.dumps(results["comparison"], indent=2), flush=True)


if __name__ == "__main__":
    main()
