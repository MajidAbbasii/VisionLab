"""Write a machine-readable toolchain and artifact manifest for the release candidate."""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_output(*args: str, cwd: Path = ROOT) -> str | None:
    try:
        result = subprocess.run(args, cwd=cwd, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> None:
    packages = {}
    for name in (
        "visionlab", "ultralytics", "torch", "torchvision", "optuna", "opencv-python",
        "numpy", "pandas", "matplotlib", "PyYAML", "scikit-learn", "tabulate",
        "pytest", "onnxruntime",
    ):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None

    cuda = {"available": False, "version": None, "device": None}
    try:
        import torch
        cuda["available"] = bool(torch.cuda.is_available())
        cuda["version"] = torch.version.cuda
        if cuda["available"]:
            cuda["device"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass

    artifacts = {}
    for relative in (
        "models/exported/best.onnx",
        "datasets/classes.txt",
        "artifacts/wheels/visionlab-1.0.0rc1-py3-none-any.whl",
        "artifacts/VisionLab-v1.0.0-rc1-dotnet-win-x64.zip",
    ):
        path = ROOT / relative
        artifacts[relative] = {
            "exists": path.is_file(),
            "size_bytes": path.stat().st_size if path.is_file() else None,
            "sha256": sha256(path) if path.is_file() else None,
        }

    dotnet_projects = {}
    for project in ("src/VisionLab.Core/VisionLab.Core.csproj", "src/VisionLab.App/VisionLab.App.csproj"):
        root = ET.parse(ROOT / project).getroot()
        properties = root.find("PropertyGroup")
        dotnet_projects[project] = {
            "target_framework": properties.findtext("TargetFramework") if properties is not None else None,
            "version": properties.findtext("Version") if properties is not None else None,
            "packages": [
                {"id": item.attrib.get("Include"), "version": item.attrib.get("Version")}
                for item in root.findall(".//PackageReference")
            ],
        }

    manifest = {
        "release": "1.0.0-rc1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "dotnet_sdk": command_output("dotnet", "--version"),
        "dotnet_sdk8_verified": command_output(
            "dotnet", "--version", cwd=ROOT / "artifacts" / "sdk8-validation"
        ),
        "dotnet_runtimes": command_output("dotnet", "--list-runtimes"),
        "dotnet_projects": dotnet_projects,
        "packages": packages,
        "pytorch_cuda": cuda,
        "artifacts": artifacts,
    }

    destination = ROOT / "artifacts" / "release-manifest.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Release manifest written: {destination.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
