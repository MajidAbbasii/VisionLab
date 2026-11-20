# Changelog

All notable project changes are documented here. This project follows a practical pre-release versioning scheme.

## [1.0.0-rc1] - 2026-11-20

### Added
- YOLO11n research workflow for training, Optuna tuning, sensitivity analysis and augmentation ablation.
- .NET 8 ONNX inference application for images, folders, video files and webcam capture.
- Automated Python regression tests and .NET ONNX integration tests.
- C# / Python ONNX parity and edge-case validation reports.
- Release verification script, environment lock snapshot and release checklist.

### Improved
- C# confidence and IoU argument validation.
- C# CLI help, invalid-command handling and positive `--max-frames` validation.
- README installation, experiment provenance, runtime limitations and test instructions.
- Curated Markdown reports and the exported ONNX model are no longer hidden by broad ignore rules.

### Known limitations
- C# inference uses CPU ONNX Runtime; CUDA is not enabled or benchmarked.
- The expanded 100-epoch training plan was intentionally stopped at epoch 10; the saved best checkpoint is from epoch 1.
- Optuna, sensitivity and ablation results are exploratory and were not repeated on the later 20k-image training split.
- Webcam availability depends on the target machine and permissions.
- The Windows OpenCvSharp native runtime package is included; Linux/macOS native deployment requires a matching runtime package and separate validation.

This is a release candidate, not a stable 1.0.0 release.
