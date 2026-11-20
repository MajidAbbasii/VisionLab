# C# / Python ONNX parity — 2026-11-20

## Purpose

Compare the existing C# ONNX detector with the Python Ultralytics ONNX reference using the exact same exported model and the same ten input images. This isolates runtime/preprocessing/postprocessing parity; it is not a model-accuracy evaluation.

## Protocol

- Model: `models/exported/best.onnx` (same file for both runtimes)
- Inputs: the ten images previously used for the C# batch smoke test in `results/predictions/runtime-batch-smoke-input/`
- Confidence threshold: 0.25
- NMS IoU threshold: 0.45
- Image size: 640 x 640
- Execution provider: CPU for both
- Python reference: Ultralytics 8.4.215 with ONNX Runtime 1.32.0
- C# runtime: Microsoft.ML.OnnxRuntime 1.20.1
- Matching: same class, greedy one-to-one assignment, accepted at box IoU >= 0.50

The ONNX Runtime versions differ, so this is a practical application-parity check rather than a bit-for-bit runtime comparison.

## Results

| Measurement | Result |
|---|---:|
| Images compared | 10 / 10 |
| C# detections | 31 |
| Python detections | 31 |
| Matched detections | 31 / 31 |
| Unmatched C# detections | 0 |
| Unmatched Python detections | 0 |
| Matched fraction of union | 100% |
| Mean per-image matched box IoU | 0.97591 |
| Maximum matched confidence difference | 0.00000135 |

All detections matched one-to-one by class and overlap. The very small confidence differences are consistent with floating-point/runtime variation. The remaining box-coordinate differences are expected to be dominated by C# converting final coordinates to integer pixels while Python retains floating-point coordinates.

## Reproduction

Run from the repository root:

```powershell
.venv\Scripts\python.exe scripts\compare_runtime_parity.py
```

The script writes Python reference JSON files to `results/predictions/python-parity-reference/` and the machine-readable summary to `results/reports/csharp_python_parity_2026-11-20.json`.

## Conclusion and limits

The current C# inference path reproduces the Python ONNX detections on this ten-image sample with no missing or additional detections and high bounding-box overlap. This is strong evidence that the current preprocessing, raw-output decoding and class-aware NMS are aligned for this model/export and these settings. It does not prove parity across all image sizes, unusual aspect ratios, all classes, other exports, or all confidence/IoU thresholds. A larger randomized sample and edge-case tests remain useful next steps.
