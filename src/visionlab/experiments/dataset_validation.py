"""Preflight checks that prevent training/validation data leakage."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ultralytics.data.utils import check_det_dataset


IMAGE_SUFFIXES = {".bmp", ".dng", ".jpeg", ".jpg", ".mpo", ".png", ".tif", ".tiff", ".webp", ".pfm"}


def _as_paths(value: Any) -> list[Path]:
    """Normalize Ultralytics' directory/list split values into paths."""
    values = value if isinstance(value, (list, tuple)) else [value]
    return [Path(item).expanduser().resolve() for item in values]


def _images(paths: list[Path]) -> list[Path]:
    """List image files from directory or image-list split definitions."""
    found: list[Path] = []
    for path in paths:
        if path.is_dir():
            found.extend(item.resolve() for item in path.rglob("*")
                         if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES)
        elif path.is_file() and path.suffix.lower() == ".txt":
            for line in path.read_text(encoding="utf-8").splitlines():
                entry = Path(line.strip())
                if not entry.is_absolute():
                    entry = path.parent / entry
                if entry.is_file() and entry.suffix.lower() in IMAGE_SUFFIXES:
                    found.append(entry.resolve())
        elif path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            found.append(path.resolve())
    return found


def validate_train_val_separation(data_yaml: str | Path) -> dict[str, Any]:
    """Resolve a detection YAML and reject identical or overlapping splits.

    Matching filenames are treated as possible leakage even when the files
    live in different directories. COCO image IDs are globally unique across
    the train2017 and val2017 partitions, making this check suitable here.
    """
    dataset = check_det_dataset(str(data_yaml))
    train_paths = _as_paths(dataset["train"])
    val_paths = _as_paths(dataset["val"])
    train_images = _images(train_paths)
    val_images = _images(val_paths)
    if not train_images or not val_images:
        raise ValueError(
            "Both train and val splits must resolve to existing image files; "
            f"found train={len(train_images)}, val={len(val_images)} for {data_yaml}"
        )

    train_resolved = {path for path in train_images}
    val_resolved = {path for path in val_images}
    path_overlap = train_resolved & val_resolved
    train_names = {path.stem for path in train_images}
    val_names = {path.stem for path in val_images}
    name_overlap = train_names & val_names
    if path_overlap or name_overlap:
        examples = sorted(name_overlap)[:5]
        raise ValueError(
            "Training/validation leakage detected: "
            f"{len(path_overlap)} identical paths and {len(name_overlap)} matching filenames. "
            f"Examples: {examples}. Use disjoint image splits and a new Optuna study."
        )

    return {
        "dataset_yaml": str(Path(data_yaml).resolve()),
        "train_images": len(train_images),
        "val_images": len(val_images),
        "path_overlap": 0,
        "filename_overlap": 0,
        "classes": len(dataset.get("names", {})),
        "train_paths": [str(path) for path in train_paths],
        "val_paths": [str(path) for path in val_paths],
    }
