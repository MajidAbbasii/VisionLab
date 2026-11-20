# C# ONNX Runtime Validation — 2026-11-20

This report records automated smoke tests performed locally. It does not replace validation on user-provided, real-world images or a live webcam session.

## Environment

- .NET SDK: 10.0.302; application target: .NET 8
- ONNX Runtime: Microsoft.ML.OnnxRuntime 1.20.1 (CPU package)
- Model: `models/exported/best.onnx`
- Input: 640×640; confidence 0.25; NMS IoU 0.45 unless noted
- Execution provider: CPU. The CUDA execution-provider package is not referenced by this project, so a CPU/GPU comparison was not possible with the current dependency set.

## Folder inference smoke test

Ten images were selected from the already-prepared independent COCO test set and passed through the C# folder-inference command. All 10 images completed, with one annotated JPEG and one JSON report per image. The reports contained 31 detections in total and were successfully parsed back as JSON.

| Measurement | Result |
|---|---:|
| Images completed | 10 / 10 |
| Skipped images | 0 |
| Total detections | 31 |
| Mean inference time | 145.50 ms/image |
| Median inference time | 140.93 ms/image |
| Minimum / maximum | 128.97 / 189.73 ms |
| Approx. throughput | 6.87 images/sec |

These timings are a small local smoke-test sample, not a statistically rigorous benchmark. They include the detector's image preprocessing and postprocessing but exclude file loading, annotation rendering and output writing.

## Video-file inference smoke test

The existing `results/predictions/dotnet-video-validation.mp4` was processed headlessly for 60 frames. The output was reopened and fully decoded successfully.

- Output: `results/predictions/dotnet-video-runtime-smoke.mp4`
- Frames decoded: 60 / 60
- Encoded dimensions: 1920×1080
- Container frame rate: 25 FPS
- Video inference uses the same configurable confidence and IoU thresholds as image inference.

The 25 FPS value is the output video's nominal frame rate, not measured real-time inference throughput. The image smoke-test throughput (~6.87 images/sec) indicates that this CPU configuration should not be assumed to sustain 25 FPS at 640×640.

## NMS implementation review

The C# detector performs class-aware Non-Maximum Suppression: proposals are grouped by predicted class, then OpenCV `CvDnn.NMSBoxes` suppresses lower-scoring boxes whose IoU with a higher-scoring box exceeds the configured threshold. This allows overlapping boxes from different classes to remain, which is intentional for class-aware NMS. The confidence threshold filters proposals before NMS and is also passed to the OpenCV NMS call.

## Remaining user-dependent checks

- Run inference on personal photographs from the intended environment and inspect false positives/missed objects.
- Try the webcam on the machine that owns the camera; this workspace cannot establish camera availability or permission.
- Compare CPU and CUDA only after installing and verifying a compatible ONNX Runtime CUDA provider and CUDA/cuDNN runtime. No GPU speedup is claimed here.

