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
    """Return a streaming SHA-256 digest without loading the whole file in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_output(*args: str, cwd: Path = ROOT) -> str | None:
    """Run a toolchain probe and return stdout, or None when it is unavailable."""
    try:
        result = subprocess.run(args, cwd=cwd, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> None:
    """Collect release environment, artifact hashes, project metadata, and legal status."""
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

    cuda = {"runtime_probe": "not performed; release verification is CPU-only"}
    try:
        import torch
        cuda["version"] = torch.version.cuda
    except ImportError:
        pass

    audit_path = ROOT / "results" / "reports" / "license-audit.json"
    license_audit = None
    if audit_path.is_file():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        license_audit = {
            "python_package_count": audit.get("python_package_count"),
            "nuget_package_count": audit.get("nuget_package_count"),
            "manual_review_count": audit.get("manual_review_count"),
            "report_sha256": sha256(audit_path),
        }

    # Record expected release payloads even when an artifact is missing; the
    # explicit exists flag makes incomplete packaging visible to release review.
    artifacts = {}
    for relative in (
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

    # Hash all included legal/source notices so the manifest identifies the
    # exact set of attribution and license materials reviewed for this build.
    legal_files = {}
    legal_paths = [ROOT / relative for relative in ("LICENSE", "THIRD-PARTY-NOTICES.md", "DATASETS.md", "MODEL_CARD.md")]
    legal_paths.extend(path for path in (ROOT / "THIRD-PARTY-NOTICES").rglob("*") if path.is_file())
    legal_paths.extend(path for path in (ROOT / "THIRD-PARTY-SOURCES").rglob("*") if path.is_file())
    for path in sorted(set(legal_paths)):
        relative = path.relative_to(ROOT).as_posix()
        legal_files[relative] = {
            "exists": path.is_file(),
            "size_bytes": path.stat().st_size if path.is_file() else None,
            "sha256": sha256(path) if path.is_file() else None,
        }

    # Read target frameworks and package versions from project files rather
    # than duplicating dependency declarations in this script.
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
        "project_license": "AGPL-3.0-only",
        "license_review_status": "python-and-known-nuget-notices-preserved; coco-category-list-cc-by-attributed; onnxruntime-eigen-mpl-and-intel-mkl-component-review-pending; opencv-native-binary-ci-attestation-pending; intel-ipp-static-link-eula-review-pending; ffmpeg-lgpl-source-bundle-staged-but-zip-delivery-and-qualified-review-pending; full-dataset-image-and-model-rights-review-pending",
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
        "legal_files": legal_files,
        "license_audit": license_audit,
        "model_policy": "ONNX model is used for local validation but excluded from the redistributable ZIP pending rights review.",
    }

    destination = ROOT / "artifacts" / "release-manifest.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Release manifest written: {destination.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
