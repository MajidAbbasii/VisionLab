# VisionLab v1.0.0-rc1 Release Candidate Verification

**Date:** 2026-11-20  
**Status:** Release candidate built and automated checks passed; not yet a stable release.

## Candidate identity

- Python package version: `1.0.0rc1` (PEP 440)
- .NET Core/App assembly version: `1.0.0-rc1`
- Python: 3.11.15
- Ultralytics: 8.4.215
- PyTorch / torchvision: 2.14.1+cu130 / 0.29.1+cu130
- Optuna: 4.9.0
- OpenCV Python: 4.14.0.94
- ONNX Runtime Python: 1.32.0
- PyTorch CUDA availability: true; CUDA runtime 13.0; NVIDIA GeForce GTX 1650 with Max-Q Design
- .NET SDK selected by the current command environment: 10.0.302; application target: `net8.0`
- Explicit .NET 8 SDK verification: SDK 8.0.101 successfully ran the solution tests and Release build (0 warnings, 0 errors).
- .NET 8 runtime installed: 8.0.29
- C# packages: Microsoft.ML.OnnxRuntime 1.20.1; OpenCvSharp4 / OpenCvSharp4.runtime.win 4.10.0.20241108

The CUDA status above belongs to PyTorch/Python. The C# application remains CPU ONNX Runtime; no C# GPU inference claim is made.

## Automated verification

The complete `scripts/verify_release.ps1` run exited with code 0.

| Check | Result |
|---|---|
| `pip check` | No broken requirements |
| Python pytest | 18 passed |
| Python compileall | Passed |
| Python CLI help | Passed |
| .NET integration tests | 9 passed, 0 failed, 0 skipped |
| .NET Release solution build | 0 warnings, 0 errors |
| .NET source CLI help | Passed |
| .NET publish | Passed |
| Published app help | Passed |
| Published app inference on `bus.jpg` | 7 detections; JPEG and JSON written |
| Extracted ZIP inference on `bus.jpg` | 7 detections; JPEG and JSON written |
| Python wheel build | Passed; `visionlab-1.0.0rc1-py3-none-any.whl` |
| Python wheel isolated install/import | Passed; imported version `1.0.0rc1` |
| Release manifest | Written with package/toolchain versions and SHA-256 hashes |

The Windows x64 ZIP retains only the `win-x64` native runtime folder. It was extracted into a temporary verification directory and the executable successfully loaded the packaged ONNX model and class mapping. Non-Windows native runtime payloads were excluded.

## Release artifacts

- `artifacts/wheels/visionlab-1.0.0rc1-py3-none-any.whl`
- `artifacts/publish/dotnet-app/`
- `artifacts/VisionLab-v1.0.0-rc1-dotnet-win-x64.zip`
- `artifacts/release-manifest.json`
- `artifacts/smoke-test/dotnet-published-app-bus.jpg` and `.json`
- `artifacts/smoke-test/dotnet-archive-bus.jpg` and `.json`

The ZIP is a framework-dependent .NET 8 Windows x64 package. The target machine needs the .NET 8 Runtime. The ONNX model and class map are included. The Python wheel contains the Python package code; repository configuration/data assets remain part of the source checkout.

Final artifact sizes and SHA-256 values (also recorded in `artifacts/release-manifest.json`):

| Artifact | Size | SHA-256 |
|---|---:|---|
| .NET Windows x64 ZIP | 49,765,572 bytes (47.46 MiB) | `46ee1df2273e7ac63be3144030f19762cb7906a8192f400180a30fb9a275977c` |
| Python wheel | 28,289 bytes | `f5ff3f35a3705df9c04e4d6eb9f0a5a766753ee114d00d4d709fb34559185090` |

## Source changes for the candidate

- Synchronized Python and .NET project version metadata to `1.0.0-rc1`.
- Added a machine-specific Python 3.11/CUDA environment lock snapshot.
- Added a repeatable non-training release verification script.
- Added Python wheel, .NET publish output, Windows x64 ZIP and machine-readable release manifest generation.
- Added a packaged-app inference smoke test, including extraction and execution from the ZIP itself.
- Updated README, changelog, release checklist and ignore rules for curated reports and the ONNX model.

## Not included / outstanding before stable 1.0.0

1. Live webcam verification on the target computer and its camera permissions.
2. Validation against user-provided real-world images and videos.
3. Review of Ultralytics, pretrained-weight, COCO dataset and dependency license/redistribution terms.
4. User review followed by Git commit, tag and push in the actual Git repository. No commit or tag was created in this workspace.
5. Optional scientific strengthening: repeat augmentation ablation across multiple seeds and consider rerunning experiments on the later expanded training split. These are not required to produce this educational release candidate, and no new training was started.

## Training / experiment safety

No training, Optuna optimization, sensitivity analysis, ablation run, webcam capture or video inference was started during release verification. Existing model checkpoints and the Optuna database were not changed by this release pass.
