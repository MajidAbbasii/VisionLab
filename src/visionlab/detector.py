"""YOLO image/video inference and webcam capture."""
from __future__ import annotations
from collections.abc import Callable
import math
from pathlib import Path
from typing import Iterator
import cv2
from ultralytics import YOLO
from .common import configure_logging, load_yaml, project_path

LOGGER = configure_logging(__name__)

def load_detector(model_path: str | Path | None = None) -> YOLO:
    """Load the configured Ultralytics detector or an explicitly supplied checkpoint."""
    cfg = load_yaml("configs/model.yaml")
    selected = model_path or cfg["model"]
    # Ultralytics resolves model aliases and downloads official pretrained weights.
    LOGGER.info("Loading model: %s", selected)
    return YOLO(str(selected))

def detect_image(source: str, model_path: str | None = None, conf: float = 0.25,
                 iou: float = 0.45, save: bool = True) -> list:
    """Run fixed-shape YOLO inference on an image and optionally save its annotation."""
    model = load_detector(model_path)
    output = project_path("results/predictions/images")
    output.mkdir(parents=True, exist_ok=True)
    # Use fixed square letterboxing to match the exported 640x640 ONNX contract.
    results = model.predict(source=source, conf=conf, iou=iou, rect=False, save=save,
                            project=str(output), name="run", exist_ok=True)
    LOGGER.info("Image inference completed: %d result(s)", len(results))
    return results

def iter_video_frames(source: str | int, width: int | None = None,
                      height: int | None = None,
                      on_open: Callable[[float], None] | None = None) -> Iterator:
    """Yield frames from a file or camera index; release camera even on exceptions."""
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Cannot open video source: {source}")
    try:
        if on_open is not None:
            on_open(capture.get(cv2.CAP_PROP_FPS))
        if width: capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height: capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        while True:
            ok, frame = capture.read()
            if not ok: break
            yield frame
    finally:
        capture.release()

def detect_video(source: str | int, model_path: str | None = None, conf: float = 0.25,
                 show: bool = False, output_path: str | None = None,
                 max_frames: int | None = None) -> Path | None:
    """Process a video file or camera frame-by-frame and release capture/writer resources."""
    model = load_detector(model_path)
    target = project_path(output_path) if output_path else None
    writer = None
    source_fps = 30.0
    processed_frames = 0

    def set_source_fps(fps: float) -> None:
        nonlocal source_fps
        if math.isfinite(fps) and fps > 0:
            source_fps = fps

    try:
        for frame in iter_video_frames(source, on_open=set_source_fps):
            # Keep preprocessing aligned with the fixed-shape C# ONNX runner.
            annotated = model.predict(frame, conf=conf, rect=False, verbose=False)[0].plot()
            if target and writer is None:
                target.parent.mkdir(parents=True, exist_ok=True)
                writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*"mp4v"),
                                         source_fps, (annotated.shape[1], annotated.shape[0]))
                if not writer.isOpened(): raise RuntimeError(f"Cannot write video: {target}")
            if writer: writer.write(annotated)
            processed_frames += 1
            if max_frames is not None and max_frames > 0 and processed_frames >= max_frames:
                break
            if show:
                cv2.imshow("VisionLab - YOLO", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"): break
    finally:
        if writer: writer.release()
        if show: cv2.destroyAllWindows()
    LOGGER.info("Video inference completed: %s", target or "display only")
    return target

