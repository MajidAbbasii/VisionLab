# C# Edge-case and Integration-test Validation — 2026-11-20

## Scope

Validate that the .NET 8 ONNX detector remains stable across unusual image dimensions, returns bounded pixel coordinates, handles a blank image, and rejects invalid confidence/IoU parameters. This is a runtime robustness check, not an object-detection accuracy benchmark.

## Seven-image folder smoke test

The C# Release application processed all generated images successfully. All output JPEGs were written and all seven JSON reports parsed successfully.

| Input shape | Result |
|---|---|
| 32 × 32 | Completed; 0 detections |
| 320 × 240 | Completed; 4 detections |
| 1920 × 480 | Completed; 1 detection |
| 480 × 1920 | Completed; 0 detections |
| 2560 × 1440 | Completed; 4 detections |
| 2048 × 2048 | Completed; 5 detections |
| 640 × 640 blank image | Completed; 0 detections |

- Images completed: 7/7
- Skipped: 0
- JSON reports: 7/7 valid
- No inference exceptions or invalid output dimensions observed.

Inputs and outputs are retained under `results/predictions/edge-case-input/` and `results/predictions/edge-case-output/`.

## Automated .NET integration tests

Added `tests/VisionLab.Core.Tests`, referencing the real exported model and the .NET Core detector. The tests cover six image shapes, output coordinate bounds, finite confidence values, blank-image behavior, and invalid confidence/IoU values (including NaN).

Command:

```powershell
dotnet test tests/VisionLab.Core.Tests/VisionLab.Core.Tests.csproj -c Release
```

Result: **9 passed, 0 failed, 0 skipped**.

The detector now validates confidence and IoU at its public API boundary and throws `ArgumentOutOfRangeException` for values outside [0,1] or non-finite values.

## Build and regression status

- `dotnet build VisionLab.sln -c Release --no-restore`: succeeded, 0 warnings, 0 errors.
- CLI help and unknown-command handling were exercised; both return usage without requiring model initialization. `--max-frames` now rejects non-positive or non-integer values.
- Python regression suite: 18 passed (executed before the C#-only changes in this pass).
- C# / Python parity script was rerun: 10 images, 31/31 detections matched, mean matched box IoU 0.9759132, no unmatched detections.

## Limits and next checks

- Synthetic shape tests establish runtime stability and coordinate bounds; they do not establish detection accuracy on every aspect ratio.
- Webcam access and real personal photographs remain environment/user-dependent checks. A historical project audit records bounded 120-frame webcam captures from both runtimes, but this pass did not repeat a live camera session.
- C# inference remains CPU-only with the currently referenced ONNX Runtime package. No CUDA performance claim is made.
- Additional useful educational work: test a broader random sample, measure end-to-end video FPS under sustained load, and inspect predictions on user-provided images.
