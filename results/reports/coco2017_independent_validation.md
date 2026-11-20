# COCO 2017 Independent Validation

Date: 2026-11-18

## Dataset integrity

- Source: official COCO 2017 `val2017` images and `instances_val2017.json`.
- Images extracted: 5,000; YOLO label files: 5,000.
- Non-crowd annotations converted: 36,335; crowd annotations skipped: 446.
- All 80 COCO classes are represented; invalid label rows: 0; corrupt images: 0.
- Filename overlap with the 128-image COCO128 fine-tuning set: **0**.
- Empty label files: 48 (images with no retained object instances).
- Converter: `scripts/convert_coco_to_yolo.py`; dataset config: `datasets/coco2017.yaml`.

The `train` key in `datasets/coco2017.yaml` exists only because Ultralytics requires it when parsing a detection YAML. It points to the validation directory to satisfy that schema. **Do not use this YAML for training.** The evaluation script explicitly uses `split="val"`.

## Evaluation protocol

Both checkpoints were evaluated on the exact same 5,000 images using Ultralytics 8.4.215 and the same settings:

| Setting | Value |
|---|---:|
| Image size | 640 |
| Batch size | 4 |
| Workers | 2 |
| Device | CUDA 0 (GTX 1650 Max-Q, 4 GB) |
| Confidence threshold | 0.001 |
| NMS IoU | 0.70 |
| Maximum detections/image | 300 |

## Results

| Metric | Baseline (`best.pt`) | Optuna tuned | Tuned − baseline |
|---|---:|---:|---:|
| Precision | 0.59266 | 0.58170 | -0.01096 |
| Recall | 0.42312 | 0.41602 | -0.00709 |
| mAP50 | 0.44697 | 0.44201 | -0.00496 |
| mAP50-95 | **0.30019** | 0.29553 | -0.00466 |

Mean inference time was approximately 5.6 ms/image for both checkpoints on this GPU. See `coco2017_independent_validation.json` for full-precision metrics and per-class mAP50-95.

## Interpretation

On this held-out image split, the baseline checkpoint scores slightly higher than the Optuna-tuned checkpoint on all four aggregate metrics. The tuned checkpoint therefore has **not** demonstrated an improvement in generalization. Keep `models/trained/best.pt` as the current reference checkpoint; do not overwrite it with the tuned model.

The difference is small and this is one evaluation run, not a multi-seed statistical comparison. It is evidence for preferring the baseline among these two existing checkpoints under this protocol, not proof that the baseline is universally superior. The original Optuna objective used a COCO128 train/validation configuration that pointed both paths at the same images, so its search was affected by validation leakage. A clean Optuna study should use the COCO128 training images with a genuinely separate validation set, rather than reuse the old study's scores.

## Artifacts

- Machine-readable summary: `results/reports/coco2017_independent_validation.json`
- Baseline curves, confusion matrices and sample predictions: `results/validation/coco2017/baseline/`
- Tuned curves, confusion matrices and sample predictions: `results/validation/coco2017/optuna_tuned/`
- Re-run command: `python scripts/validate_models.py`

