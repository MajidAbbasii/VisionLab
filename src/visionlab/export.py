"""Export a trained Ultralytics checkpoint to ONNX for the .NET application."""
from __future__ import annotations
import argparse
from pathlib import Path
from ultralytics import YOLO
from .common import project_path

def export_onnx(model_path: str = "models/trained/best.pt", imgsz: int = 640) -> Path:
    """Export a fixed-shape, raw-output ONNX model for the C# detection runner.

    Dynamic axes and embedded NMS are disabled to match the preprocessing and
    post-processing contract implemented by ``YoloOnnxDetector``.
    """
    model = YOLO(str(project_path(model_path)))
    # Keep the graph shape static and leave NMS to the C# runtime, which applies
    # class-aware suppression consistently with the browser demo.
    exported = Path(model.export(format="onnx", imgsz=imgsz, dynamic=False, simplify=True,
                                 opset=17, nms=False))
    target = project_path("models/exported") / exported.name
    target.parent.mkdir(parents=True, exist_ok=True)
    # Ultralytics writes beside the source checkpoint by default; copy the
    # exported artifact into the repository's dedicated distribution folder.
    if exported.resolve() != target.resolve():
        import shutil
        shutil.copy2(exported, target)
    return target

if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--model", default="models/trained/best.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()
    print(export_onnx(args.model, args.imgsz))

