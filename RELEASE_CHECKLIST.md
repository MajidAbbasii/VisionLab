# VisionLab v1.0.0-rc1 Release Checklist

## Scope and release policy

- [x] Python research engine and .NET 8 inference app are in the solution.
- [x] YOLO11n ONNX model and class-name mapping are present in the workspace.
- [x] Image, folder, video and webcam workflows are documented.
- [x] Optuna, sensitivity and ablation workflows and their limitations are documented.
- [x] Python and .NET automated test suites exist.
- [x] Historical experiment limitations are explicit; no training is triggered by release verification.
- [x] Release verification script passes on the current Windows environment.
- [x] Solution also builds and tests successfully with the installed .NET 8 SDK (8.0.101), in addition to the default SDK on PATH.
- [x] Python 1.0.0rc1 wheel built successfully.
- [x] Built Python wheel installed into an isolated target directory and imported with the expected version.
- [x] Windows x64 .NET publish package built and its ZIP extracted and smoke-tested with the ONNX model.
- [x] Release manifest records package/toolchain versions and SHA-256 hashes for model, class map and release artifacts.
- [ ] Live webcam check on the target machine.
- [ ] Validate with user-provided real-world images/video.
- [ ] Review third-party model and dataset license terms before redistribution.
- [ ] User reviews changes and creates the Git commit/tag/push.

## Reproducible verification

Run from the repository root on Windows with Python 3.11 and the .NET 8 SDK/runtime installed:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/verify_release.ps1
```

The script runs dependency checks, Python tests and bytecode compilation, .NET tests/build, Python/.NET CLI help checks and one image inference through the published .NET package. It does not start training, Optuna, sensitivity, ablation, webcam capture or video inference.

## Release artifacts

- `artifacts/wheels/`: Python wheel for `visionlab==1.0.0rc1`.
- `artifacts/publish/dotnet-app/`: framework-dependent .NET 8 application publish output.
- `artifacts/VisionLab-v1.0.0-rc1-dotnet-win-x64.zip`: Windows .NET runtime package, smoke-tested against `bus.jpg`.
- `artifacts/release-manifest.json`: toolchain and package versions observed during packaging.
- `requirements-lock-py311-cu130.txt`: exact dependency snapshot for the verified Windows/Python 3.11/CUDA environment (not a cross-platform lock).
- `results/reports/`: curated experiment and runtime evidence.
- `models/exported/best.onnx`: ONNX model used by the C# application and integration tests.
- `datasets/classes.txt`: model class-index mapping.

## Acceptance criteria

1. [x] `scripts/verify_release.ps1` exits successfully.
2. [x] Python test suite passes and `pip check` reports no broken requirements.
3. [x] .NET test suite passes and Release build has zero warnings/errors.
4. [x] Python wheel and .NET publish outputs are produced and the archive is smoke-tested.
5. [x] Required ONNX model and class-name mapping are present.
6. [x] The final audit and known limitations remain accurate.

Do not mark this release stable until user-dependent checks and license review are complete.
