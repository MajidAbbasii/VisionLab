# VisionLab â€” YOLO Object Detection & Experiment Workbench

<p align="center"><strong>Release candidate: v1.0.0-rc1</strong></p>

VisionLab is a modular, reproducible object-detection project. It combines a Python research engine (Ultralytics YOLO + PyTorch + Optuna) with a .NET 8 inference application (ONNX Runtime + OpenCvSharp). It supports image detection, video-file/webcam capture, training, hyperparameter tuning, sensitivity analysis, augmentation ablations, metrics, plots, logs and reports.

**Default model:** YOLO11n. You can select another Ultralytics-compatible checkpoint such as `yolov8n.pt`; support for YOLOv9/YOLOv10 model names depends on the installed Ultralytics release and the model's official integration. The Python engine is the reference implementation for training and research. C# consumes exported ONNX models for deployment.

All dependencies are free/open-source. No paid inference API or cloud service is required. Pretrained model weights and datasets may have their own licenses; review them before commercial use. Training speed depends on hardware. A CUDA-capable GPU is recommended, but CPU execution is possible.

## 1. Architecture

```text
Python research engine
  Ultralytics / PyTorch / OpenCV / Optuna
       | train, validate, tune, analyze
       | export best.pt -> ONNX
       v
models/exported/*.onnx (exported locally; not bundled in this source-only repository)
       |
       v
.NET 8 application
  ONNX Runtime + OpenCvSharp
  image inference / video file / webcam
```

The C# ONNX runner is intended for standard YOLO detection exports with a raw rank-3 output tensor (`nms=False`) and fixed 640x640 input. It handles the common `[1, 4+classes, candidates]` and transposed layouts. Models exported with a different preprocessing, output or end-to-end NMS contract may need an adapter; always validate C# predictions against Python before deployment.

### C# inference commands

Run these commands from the repository root. The defaults use `models/exported/best.onnx` and `datasets/classes.txt`:

```powershell
# One image: saves an annotated image and a JSON detection report
dotnet run -c Release --project src/VisionLab.App -- image bus.jpg

# Tune confidence and NMS IoU thresholds, and choose output paths
dotnet run -c Release --project src/VisionLab.App -- image bus.jpg --confidence 0.30 --iou 0.50 --output results/predictions/bus.jpg --json results/predictions/bus.json

# Process every supported image in a folder (top-level only by default)
dotnet run -c Release --project src/VisionLab.App -- image path/to/images --output results/predictions/batch

# Include nested folders; JSON reports are written under the output's json folder
dotnet run -c Release --project src/VisionLab.App -- image path/to/images --output results/predictions/batch --recursive

# Process a video file or webcam (0), with configurable thresholds
dotnet run -c Release --project src/VisionLab.App -- video 0 --confidence 0.30 --iou 0.50 --max-frames 300

# Show CLI help without loading the ONNX model
dotnet run -c Release --project src/VisionLab.App -- --help
```

The JSON report for each image includes source/output paths, image dimensions, thresholds, inference time and detections (class ID, label, confidence and pixel bounding box). Confidence and IoU must each be between 0 and 1. Supported folder image extensions are JPG/JPEG, PNG, BMP, TIFF and WEBP.

## 2. Requirements

- Windows 10/11, Linux or macOS for Python workflows (the included OpenCvSharp native runtime package is Windows-specific).
- Python 3.10â€“3.12 recommended. Python 3.13 may not have compatible wheels for every PyTorch/Ultralytics dependency.
- .NET 8 SDK for the C# application.
- 8 GB RAM minimum; 16 GB+ recommended. GPU memory requirements depend on image size and batch size.
- For the currently verified Windows environment, see `requirements-lock-py311-cu130.txt` (Python 3.11.15 / PyTorch 2.14.1+cu130). This lock is hardware/platform-specific; it is not a universal cross-platform lock.

## 3. Installation

From the repository root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install PyTorch appropriate to your machine first. Use the official selector to choose CPU or the matching CUDA wheel: <https://pytorch.org/get-started/locally/>. Then install project dependencies:

```powershell
pip install -r requirements.txt
pip install -e .
```

For an exact reproduction of the current Windows/CUDA Python environment, first install the matching PyTorch build using the official selector, then use `pip install -r requirements-lock-py311-cu130.txt` followed by `pip install -e . --no-deps`. The general `requirements.txt` intentionally leaves PyTorch platform selection to the user. The lock snapshot is for Python 3.11 on Windows and is not suitable for Linux/macOS or CPU-only systems.

Build the .NET application:

```powershell
dotnet build src/VisionLab.App/VisionLab.App.csproj -c Release
```

## 3.1 Release candidate verification

The current candidate is `1.0.0-rc1` (Python package version `1.0.0rc1`). Run the complete non-training verification from PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/verify_release.ps1
```

This checks required files, Python dependency health/tests/compilation, .NET tests/build and both CLI help paths. It creates a Python wheel, publishes the .NET application with its ONNX model and class mapping, runs real inference through the published app, creates a Windows runtime ZIP, and writes `artifacts/release-manifest.json`. It does **not** train a model, start Optuna, run sensitivity/ablation, or access the webcam.

See [RELEASE_CHECKLIST.md](RELEASE_CHECKLIST.md) and [CHANGELOG.md](CHANGELOG.md) for acceptance criteria and candidate changes. The release candidate is not a stable 1.0.0 release; webcam and user-data checks and third-party license review remain outstanding.

For Linux, replace `OpenCvSharp4.runtime.win` with the matching supported native runtime package and verify the OpenCvSharp runtime deployment instructions.

## 4. Dataset preparation

The included `datasets/coco128.yaml` is a small Ultralytics smoke-test dataset configuration. Ultralytics can download COCO128 automatically. It is not a substitute for a representative training/validation benchmark. **In the original COCO128 experiment configuration, both `train` and `val` pointed to `images/train2017`.** The original Optuna, sensitivity and ablation results therefore remain exploratory and must not be interpreted as generalization evidence.

An official COCO 2017 validation set has since been prepared locally under `datasets/coco2017/`: 5,000 images and 5,000 YOLO label files, covering all 80 classes. The 5,000 validation image IDs have zero filename overlap with the 128 COCO128 fine-tuning images. The conversion script is `scripts/convert_coco_to_yolo.py`; the evaluation YAML is `datasets/coco2017.yaml`. Its `train` key points to the validation directory only to satisfy Ultralytics' dataset-YAML parser; **never use this YAML for training**. Its path is currently absolute for this Windows workspace and must be updated if the repository is moved.

Re-evaluate the two saved checkpoints with the same protocol:

```powershell
python scripts/validate_models.py
```

This evaluates `models/trained/best.pt` and `models/trained/best_tuned_visionlab_yolo11n_coco128_v1.pt` at 640px, batch 4, confidence 0.001, NMS IoU 0.70 and max 300 detections. The metrics are written to `results/reports/coco2017_independent_validation.json`, the interpretation to `results/reports/coco2017_independent_validation.md`, and curves/confusion matrices/sample predictions to `results/validation/coco2017/`. Current results slightly favor the baseline; see the report before selecting a checkpoint. This held-out validation comparison does not retroactively repair the old Optuna study's leaked objective.

### Leakage-free training and experiment protocol

For new experiments, use `configs/experiments_coco128_train_coco2017_val.yaml` and `datasets/coco128_train_coco2017_val.yaml`. The latter resolves to 128 COCO128 training images and 5,000 COCO2017 validation images, with 80 classes and zero matching image filenames. Training, tuning, ablation and sensitivity entry points now run a split preflight and stop before GPU work if train/val image paths or filenames overlap. The old `datasets/coco128.yaml` and `datasets/coco2017.yaml` are intentionally rejected for training because each points train and val at the same image set.

Run the clean baseline first. Its checkpoint and metrics have dedicated filenames, so the historical `best.pt` and old reports are not overwritten:

```powershell
python -m visionlab.train --config configs/training_independent.yaml
```

Then run the new Optuna study (separate study name in the existing SQLite database):

```powershell
python -m visionlab.experiments.tuning --config configs/experiments_coco128_train_coco2017_val.yaml --trials 20
```

After the clean baseline checkpoint exists, run sensitivity analysis:

```powershell
python -m visionlab.experiments.sensitivity --config configs/experiments_coco128_train_coco2017_val.yaml
```

The five-variant augmentation ablation uses the same independent validation set:

```powershell
python -m visionlab.experiments.ablation --config configs/experiments_coco128_train_coco2017_val.yaml
```

These are GPU-intensive jobs. The 5,000-image validation set is evaluated during training, and the complete plan includes a 50-epoch baseline, up to 20 Optuna trials of 50 epochs, and five 50-epoch ablation runs. Start these runs only after reviewing available GPU memory, thermals and time budget. The scripts save each study's artifacts under separate names and do not reset the historical Optuna database.

For a custom dataset, use the YOLO detection layout:

```text
datasets/my_dataset/
  images/train/  images/val/
  labels/train/  labels/val/
```

Each image has a matching `.txt` label file. Each line is `class_id x_center y_center width height`, normalized to 0â€“1. Define every class in the dataset YAML, for example:

```yaml
path: datasets/my_dataset
train: images/train
val: images/val
names:
  0: helmet
  1: person
```

Update `data` in `configs/training.yaml` and `configs/experiments.yaml`. Keep a held-out test set separate; do not tune against it. Ensure train/validation splits do not leak near-duplicate frames from the same video.

## 5. Train a model

Edit `configs/training.yaml` (dataset, epochs, image size, batch, optimizer, augmentation settings), then:

```powershell
python -m visionlab.train
```

For the current leakage-free COCO128/COCO2017 protocol, use `python -m visionlab.train --config configs/training_independent.yaml`. The legacy `configs/training.yaml` points to the old overlapping COCO128 train/val split and is now intentionally blocked by the split preflight.

Or override selected values:

```powershell
python -m visionlab.train --data datasets/my_dataset.yaml --epochs 100 --model yolo11s.pt
```

The best checkpoint is copied to `models/trained/best.pt`; Ultralytics' full run folder and checkpoints remain under `experiments/training/`. Training metrics are copied to `results/metrics/training_metrics.csv`; native Ultralytics plots remain with the run.

## 6. Image detection

```powershell
python -m visionlab.cli image path/to/image.jpg --model models/trained/best.pt --conf 0.25
```

The annotated output is written below `results/predictions/images/`. Python inference uses fixed square letterboxing (`rect=False`) to match the C# runner's fixed 640x640 ONNX preprocessing, which is important when comparing detections across the two runtimes. For the quick pretrained demo, omit `--model` to use `yolo11n.pt` from `configs/model.yaml`.

## 7. Video and webcam capture

Video file:

```powershell
python -m visionlab.cli video path/to/input.mp4 --model models/trained/best.pt --show --output results/predictions/output.mp4
```

Default webcam (camera index 0):

```powershell
python -m visionlab.cli video 0 --model models/trained/best.pt --show --output results/predictions/webcam.mp4
python -m visionlab.cli video 0 --model models/trained/best.pt --output results/predictions/webcam-test.mp4 --max-frames 120
```

Press **Q** in the preview window to stop. The capture is released in a `finally` block. On a headless machine, omit `--show`. The Python video writer preserves the source FPS when it is reported by OpenCV and falls back to 30 FPS when the source does not provide a valid value.
For a bounded capture (useful for webcam smoke tests), pass `--max-frames 120`; the writer is finalized normally when the requested number of frames has been processed. The .NET runner supports the same `--max-frames` option.

## 8. Hyperparameter tuning (Optuna)

Run:

```powershell
python -m visionlab.experiments.tuning --config configs/experiments_coco128_train_coco2017_val.yaml --trials 20
```

The objective maximizes the best validation mAP50-95 recorded in each Ultralytics `results.csv`. Each trial trains a fresh model with sampled final learning-rate factor, weight decay, warmup and HSV/mosaic/mixup augmentation values. The `optimizer: auto` setting is retained from the baseline; `lr0` and `momentum` are deliberately not searched because Ultralytics' automatic optimizer selection can override them. Trial count and ranges are in `configs/experiments.yaml`.

For the leakage-free study, the configuration explicitly uses a seeded Optuna `TPESampler` and `MedianPruner`. The pruner receives validation mAP50-95 after each completed YOLO epoch through Ultralytics' `on_fit_epoch_end` callback. It waits for at least five completed trials and does not consider pruning before epoch 15 (`n_startup_trials: 5`, `n_warmup_steps: 15`). A pruned run is recorded with Optuna state `PRUNED`; it is not counted as a failed trial. Sampler and pruner settings are part of the study fingerprint, so changing them requires a new study name. The historical study `visionlab_yolo11n_coco128_v1` and the earlier `visionlab_yolo11n_coco128_v2` configuration are not the clean study; their old objective used the overlapping COCO128 split. The new clean study has its own name, `visionlab_yolo11n_coco128_train_coco2017_val_v1`, and must remain separate.

Artifacts: Optuna SQLite study in `experiments/tuning/optuna.db`, per-trial run directories, a CSV containing **all** trial states (including failed/pruned trials) in `results/metrics/optuna_<study>_trials.csv`, a JSON summary and environment manifest in `results/reports/`, and optimization-history/parameter-importance plots in `results/plots/optuna/<study>/`. The manifest records Python/library versions, CUDA/GPU, the dataset-YAML SHA-256 and training configuration. The best trial checkpoint is copied to the study-specific path `models/trained/best_tuned_<study>.pt` when its run produced `weights/best.pt`. Parameter-importance plots are only generated after at least two completed trials.

Tuning is compute-intensive. The clean study uses a 50-epoch, 640px, batch-4, workers-2 budget by default. AMP is explicitly disabled for this GTX 1650 setup because Ultralytics' hardware check disabled it during the earlier smoke test. For a short pipeline smoke test, still use the independent split and a distinct study name: `python -m visionlab.experiments.tuning --config configs/experiments_coco128_train_coco2017_val.yaml --study-name visionlab_yolo11n_coco128_train_coco2017_val_smoke_v1 --trials 1 --epochs 3 --batch 2 --workers 0`. A study resumes from SQLite by default. If its training/search configuration changes, the fingerprint check stops the run; create a new `study_name` for a genuinely different experiment. Do not compare trials trained on different data splits or materially different budgets.

## 9. Sensitivity analysis

First train the independent-split baseline using `configs/training_independent.yaml`, then run:

```powershell
python -m visionlab.experiments.sensitivity --config configs/experiments_coco128_train_coco2017_val.yaml
```

This is a one-factor-at-a-time inference analysis: confidence threshold, NMS IoU threshold and inference image size are varied while the independent validation dataset and other settings are held fixed. It records precision, recall, mAP50 and mAP50-95. The clean study uses its own metrics, plots and JSON report names under the study identifier; it does not overwrite the historical sensitivity artifacts.

Interpretation: confidence changes the precision/recall operating point; NMS IoU changes duplicate suppression; image size changes the resolution/accuracy/latency trade-off. This is an inference sensitivity study, not a claim about internal network-weight sensitivity.

## 10. Ablation study

Run:

```powershell
python -m visionlab.experiments.ablation --config configs/experiments_coco128_train_coco2017_val.yaml
```

The controlled variants are baseline, no mosaic, no mixup, no horizontal flip, and reduced augmentation. Each variant starts from the same pretrained model and uses the same configured dataset, seed, epochs, image size and batch size. Only the listed augmentation switches change. Compare validation mAP50-95 and the absolute/percentage difference against baseline.

Artifacts: independent run folders in `experiments/ablation/<study>/`, plus study-specific CSV, PNG and JSON outputs. Each run uses the independent COCO2017 validation split. For publication-quality conclusions, repeat each variant across multiple seeds and report mean and standard deviation; the default one-seed experiment is still exploratory.

## 11. ONNX export and C# inference

Export the trained checkpoint:

```powershell
python -m visionlab.export --model models/trained/best.pt --imgsz 640
```

Copy/confirm the class names in `datasets/classes.txt` in the exact order used during training. Build and run C# inference from the repository root:

```powershell
dotnet build src/VisionLab.App/VisionLab.App.csproj -c Release
dotnet run --project src/VisionLab.App -- image path/to/image.jpg
dotnet run --project src/VisionLab.App -- video path/to/video.mp4 --output results/predictions/dotnet-video.mp4 --no-window
dotnet run --project src/VisionLab.App -- video 0 --output results/predictions/dotnet-webcam.mp4
dotnet run --project src/VisionLab.App -- video 0 --output results/predictions/dotnet-webcam-test.mp4 --max-frames 120 --no-window
```

Put the exported ONNX file at `models/exported/best.onnx` (or pass `--model` to the app). The C# runner expects a 640x640 RGB float NCHW input scaled to [0,1]. It performs class-aware NMS and saves annotated output. The video writer preserves the source FPS when available and falls back to 30 FPS otherwise. GPU execution is opt-in in the C# detector constructor; install a compatible ONNX Runtime GPU package and CUDA/cuDNN runtime before enabling it.

## 12. Consolidated report

After running the experiments:

```powershell
python -m visionlab.report
```

This creates `results/reports/experiment_report.md`, including tables for every available experiment, Optuna study provenance, the configured train/validation paths and an explicit warning when they are identical. CSV is the canonical machine-readable output; PNG plots are generated for visual inspection. A report does not turn a smoke-test dataset into an independent benchmark; keep the validation limitation visible until a separate split has been evaluated.

The requirement-by-requirement acceptance checklist and current verification evidence are maintained in `results/reports/requirements_audit.md`.

## 13. Project layout

```text
VisionLab/
  configs/                    model, training and experiment settings
  CHANGELOG.md                release history
  datasets/                   COCO128 config, COCO2017 val images/labels and class names
  models/                     trained .pt and exported .onnx artifacts
  experiments/                Ultralytics runs and Optuna database
  results/metrics/             CSV metrics and trial tables
  results/plots/               sensitivity, ablation and Optuna plots
  results/predictions/         annotated images and videos
  results/reports/             JSON and Markdown experiment reports
  results/validation/          held-out validation curves and prediction samples
  logs/                        application log
  src/visionlab/               Python training, detection and experiments
  src/VisionLab.Core/          .NET ONNX detector and video processor
  src/VisionLab.App/           .NET command-line application
  scripts/                     data conversion, validation and release verification
  docs/                        runtime package instructions
  tests/                       Python unit tests and .NET ONNX integration tests
  RELEASE_CHECKLIST.md         release candidate acceptance checklist
  requirements-lock-py311-cu130.txt  verified Windows/CUDA dependency snapshot
```

## 14. Tests and reproducibility

```powershell
python -m pytest
python -m compileall src/visionlab
dotnet test tests/VisionLab.Core.Tests/VisionLab.Core.Tests.csproj -c Release
dotnet build VisionLab.sln -c Release
```

The .NET integration tests load the checked-in exported ONNX checkpoint and exercise portrait, landscape, square, small and large images, output-box bounds, blank-image behavior and invalid confidence/IoU values. They are integration tests (they require the model artifact and OpenCV native runtime), not isolated unit tests.

Use a fixed seed, the same dataset version/split, same training budget and documented hardware/software versions when comparing experiments. Save the dataset YAML, class mapping, model checkpoint, Ultralytics run directory and CSV metrics together. Avoid treating COCO128 smoke-test results as evidence of real-world performance. This project does not fabricate experiment outcomes: reports contain only artifacts from runs actually executed.

## 15. Troubleshooting

- **CUDA out of memory:** lower `batch` and/or `imgsz`; close other GPU workloads.
- **No camera:** try camera index `1`, check OS camera permissions, and close other applications using it.
- **ONNX output mismatch:** export with `nms=False`, fixed 640 image size and a supported detection model; verify output dimensions and preprocessing. Segmentation/pose/OBB models are not supported by this C# detection parser.
- **Wrong labels:** `datasets/classes.txt` must exactly match the model's class-index order.
- **Python version/wheel issue:** use Python 3.11 and install the matching PyTorch wheel before requirements.
- **Poor metrics:** validate labels, class balance, train/val split, object size and domain similarity before increasing model complexity.

## License

Project source is provided for learning and research. Ultralytics, PyTorch, OpenCV, ONNX Runtime and model/dataset licenses remain applicable; check their current terms before redistribution or commercial deployment.


## Current educational run (November 2026)

The 10-epoch YOLO11n learning run, checkpoint comparison, visual error analysis, held-out test evaluation, ONNX export and C# smoke test are documented in [results/reports/learning_journey_yolo11n.md](results/reports/learning_journey_yolo11n.md). No further training is required for this learning path.

The C# folder-inference and video-file smoke-test results, measured CPU throughput, NMS review and remaining user-dependent checks are documented in [results/reports/csharp_runtime_validation_2026-11-20.md](results/reports/csharp_runtime_validation_2026-11-20.md).


The ten-image C# versus Python ONNX parity experiment (31/31 matched detections, mean matched box IoU 0.97591) and its reproducible script are documented in [results/reports/csharp_python_parity_2026-11-20.md](results/reports/csharp_python_parity_2026-11-20.md) and scripts/compare_runtime_parity.py.

The additional seven-shape C# edge-case smoke test and automated .NET integration-test results are documented in [results/reports/csharp_edge_case_validation_2026-11-20.md](results/reports/csharp_edge_case_validation_2026-11-20.md).

The current release candidate packaging and known limitations are tracked in [RELEASE_CHECKLIST.md](RELEASE_CHECKLIST.md), [CHANGELOG.md](CHANGELOG.md) and [results/reports/release_candidate_v1.0.0-rc1.md](results/reports/release_candidate_v1.0.0-rc1.md). Run `scripts/verify_release.ps1` before creating a release tag.

