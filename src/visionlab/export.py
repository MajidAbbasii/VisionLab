"""Export a trained Ultralytics checkpoint to ONNX for the .NET application."""
from __future__ import annotations
import argparse
from pathlib import Path
from ultralytics import YOLO
from .common import project_path

def export_onnx(model_path: str = "models/trained/best.pt", imgsz: int = 640) -> Path:
    """Export a fixed-shape, raw-output ONNX model for the C# detection runner."""
    model = YOLO(str(project_path(model_path)))
    exported = Path(model.export(format="onnx", imgsz=imgsz, dynamic=False, simplify=True,
                                 opset=17, nms=False))
    target = project_path("models/exported") / exported.name
    target.parent.mkdir(parents=True, exist_ok=True)
    if exported.resolve() != target.resolve():
        import shutil
        shutil.copy2(exported, target)
    return target

if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--model", default="models/trained/best.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()
    print(export_onnx(args.model, args.imgsz))

