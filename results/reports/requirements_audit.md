# VisionLab Requirements Audit

Audit date: 2026-11-20  
Workspace: `F:\Projects\VisionLab`

This audit maps the original educational object-detection requirements to implementation and recorded verification. Historical experiments, current smoke tests and user-dependent checks are deliberately distinguished.

## Requirement checklist

| Requirement | Status | Evidence / limits |
|---|---|---|
| Modern YOLOv8+ detector | Complete | YOLO11n is the primary model; Python/Ultralytics training and ONNX export are implemented. |
| C# / .NET 8 application | Complete | Modular `VisionLab.Core` and `VisionLab.App`, both target `net8.0`; ONNX Runtime + OpenCvSharp. Python remains the research/training engine. |
| Image inference | Verified | Python and C# process individual images; C# also supports batch folders and JSON output. |
| Video-file inference | Verified | Both runtimes were previously exercised on a 393-frame sample; current C# smoke test decoded and processed 60 frames. |
| Webcam capture | Historically verified | The 2026-11-18 audit records successful 120-frame captures in both runtimes at 640×480/30 FPS. A live camera session was not repeated in the 2026-11-20 pass. |
| Hyperparameter tuning | Complete (exploratory) | Leakage-free Optuna study has 20 trials: 13 COMPLETE, 6 PRUNED, 1 FAIL. Best trial gain was small (~0.96% relative mAP50-95 over its baseline); not a strong performance improvement claim. |
| Sensitivity analysis | Complete (exploratory) | 16 inference evaluations over confidence, NMS IoU and image size on 5,000 COCO2017 validation images. One-factor-at-a-time analysis, not internal network-weight sensitivity. |
| Ablation study | Complete (exploratory) | Five augmentation variants, 21 epochs each, same validation set. Single seed; conclusions remain exploratory. |
| Expanded dataset and independent test | Prepared and validated | 20,000 train, 5,000 fresh validation and 5,000 untouched test images; split overlap checks passed. A 100-epoch training run was intentionally stopped at epoch 10 by user request; do not claim full 100-epoch training. |
| Metrics, plots, checkpoints, logs and reports | Complete | Stored under `results/`, `models/`, `experiments/` and `logs/`; JSON/CSV/PNG/Markdown outputs retained. |
| Python/.NET ONNX parity | Verified on sample | Rerun on 10 images: 31/31 detections matched, mean matched box IoU 0.9759132, zero unmatched detections. Not a universal parity guarantee. |
| C# edge-case handling | Verified | Seven image shapes processed; 7/7 JSON outputs valid. Automated tests cover six shapes, box bounds, blank input and invalid thresholds. |
| C# CLI argument handling | Verified | Help/unknown-command paths do not initialize the model; non-positive or malformed `--max-frames` values are rejected. |
| Python automated tests | Verified | Latest run: 18 passed. |
| .NET integration tests | Verified | `VisionLab.Core.Tests`: 9 passed, 0 failed, 0 skipped. Tests use the actual ONNX model artifact. |
| .NET Release build | Verified | `dotnet build VisionLab.sln -c Release --no-restore`: 0 warnings, 0 errors. |
| Release candidate packaging | Verified | `scripts/verify_release.ps1` exits 0; Python wheel and Windows x64 .NET ZIP built; extracted ZIP runs ONNX inference successfully; manifest records versions and SHA-256 hashes. |
| Documentation | Complete and updated | README includes architecture, setup, dataset, training, inference, video/webcam, tuning, sensitivity, ablation, ONNX, tests, project layout and limitations. |
| Free/open-source workflow | Complete with license caveat | No paid APIs or hosted services required. Model weights and datasets retain their own licenses/terms. |
| CUDA inference in C# | Not enabled | Current project references CPU `Microsoft.ML.OnnxRuntime`; do not claim C# CUDA support or GPU speedup. Python training may use CUDA where a compatible PyTorch build is installed. |
| Real personal photos / user's current camera | User-dependent | Synthetic and public COCO samples cannot replace inspection on the user's actual environment and target images. |

## Verification performed on 2026-11-20

- Python test suite: 18 passed.
- .NET integration suite: 9 passed, 0 failed, 0 skipped.
- .NET solution Release build: 0 warnings, 0 errors.
- C# edge-case folder run: 7/7 images processed, 0 skipped, 7/7 JSON reports valid.
- C# video smoke test: 60/60 frames decoded from the output file.
- C# / Python parity rerun: 10 images, 31 detections from each runtime, 31 matched, mean IoU 0.9759132, max confidence difference approximately 0.00000135.
- No model training, Optuna trials, or changes to the Optuna database were performed during this validation pass.

Detailed current runtime evidence: `csharp_runtime_validation_2026-11-20.md`, `csharp_python_parity_2026-11-20.md` and `csharp_edge_case_validation_2026-11-20.md`.

Release candidate packaging evidence: `release_candidate_v1.0.0-rc1.md`.

## Scientific and operational limitations

1. Historical experiments that used the original COCO128 train/validation overlap remain exploratory and must not be presented as independent generalization results.
2. The Optuna, sensitivity and ablation studies used the leakage-free 128-image training / 5,000-image COCO2017 validation protocol; they were not rerun on the later 20,000-image expanded training set.
3. The expanded-dataset 100-epoch run was stopped at epoch 10 at the user's request. Its best epoch-1 checkpoint exists, but a completed 100-epoch expanded training run does not.
4. Ablation has one seed. Repeat across multiple seeds before making strong scientific claims.
5. Sensitivity analysis varies inference parameters; it does not measure internal model-weight sensitivity.
6. Webcam evidence is a historical bounded capture. Current device permissions and camera availability should be confirmed on the user's target machine if needed.
7. C# uses CPU ONNX Runtime. CUDA requires a compatible GPU package and CUDA/cuDNN runtime and has not been tested here.
8. No GitHub Actions or remote deployment is included; neither is required for the local educational project brief.
