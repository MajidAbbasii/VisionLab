"""Shared paths, YAML configuration, logging and reproducibility helpers."""
from __future__ import annotations
import json
import logging
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]

def project_path(value: str | Path) -> Path:
    """Resolve a path relative to the repository, independent of cwd."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path

def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a project-relative YAML file and require a mapping at its root."""
    with project_path(path).open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream) or {}
    if not isinstance(value, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return value

def ensure_dirs() -> None:
    """Create the standard artifact directories without removing existing files."""
    for folder in ("models/trained", "models/exported", "experiments", "results/metrics",
                   "results/plots", "results/predictions", "results/reports", "logs"):
        project_path(folder).mkdir(parents=True, exist_ok=True)

def configure_logging(name: str = "visionlab") -> logging.Logger:
    """Create a console-and-file logger once for the requested module name."""
    ensure_dirs()
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    file_handler = logging.FileHandler(project_path("logs/visionlab.log"), encoding="utf-8")
    file_handler.setFormatter(formatter)
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console)
    return logger

def set_seed(seed: int = 42) -> None:
    """Seed Python, NumPy and PyTorch (including CUDA when available)."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

def save_json(data: Any, path: str | Path) -> Path:
    """Persist experiment data as UTF-8 JSON with a UTC creation timestamp."""
    target = project_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {"created_utc": datetime.now(timezone.utc).isoformat(), "results": data}
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return target

