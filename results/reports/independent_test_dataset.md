# Independent Test Dataset — VisionLab

**Prepared:** 2026-11-19  
**Dataset:** COCO 2017 `train2017` held-out sample  
**Split name:** `test2017`  
**Selection seed:** `20261119`

## Purpose and isolation

This dataset is reserved for the final, one-time evaluation after the final model and all inference settings have been frozen. It must not be used for gradient updates, Optuna, model selection, confidence/IoU threshold tuning, image-size selection, early stopping, or ablation decisions.

The original COCO128 training subset is drawn from COCO `train2017`. To avoid image leakage, the 5,000 selected test images were sampled from the remaining COCO `train2017` images after excluding all 128 existing COCO128 training filenames. The already-used COCO2017 `val2017` split was also checked and has zero filename overlap with this test set.

## Dataset contents

| Item | Count / size |
|---|---:|
| Test images | 5,000 |
| YOLO label files | 5,000 |
| COCO annotations in subset JSON | 36,694 |
| YOLO boxes written (non-crowd) | 36,289 |
| Crowd annotations skipped | 405 |
| Invalid annotations skipped | 0 |
| Classes represented | 80 / 80 |
| Downloaded image payload | 820,132,402 bytes (~782.1 MiB) |

## Split integrity checks

| Comparison | Image count | Filename overlap |
|---|---:|---:|
| COCO128 train vs independent test | 128 vs 5,000 | 0 |
| Existing COCO2017 val vs independent test | 5,000 vs 5,000 | 0 |
| COCO128 train vs existing COCO2017 val | 128 vs 5,000 | 0 |

The project `validate_train_val_separation` preflight was run against `datasets/coco2017_independent_test.yaml` and passed: 128 training images, 5,000 held-out images, zero path overlap, zero filename overlap, 80 classes.

Every downloaded image was checked for a JPEG signature and non-trivial file size. Every YOLO label row was checked for five fields, a class index in `[0, 79]`, normalized coordinates in `[0, 1]`, and positive width/height. A SHA-256 checksum was recorded for each image after download.

## Paths

- Dataset YAML: `datasets/coco2017_independent_test.yaml`
- Images: `datasets/coco2017_independent_test/images/test2017/`
- YOLO labels: `datasets/coco2017_independent_test/labels/test2017/`
- COCO subset annotations: `datasets/coco2017_independent_test/annotations/instances_test2017.json`
- Selection manifest: `datasets/coco2017_independent_test/annotations/selection_manifest.json`
- Per-image checksums: `datasets/coco2017_independent_test/annotations/sha256_manifest.json`
- Preparation summary: `datasets/coco2017_independent_test/annotations/preparation_summary.json`
- Reproducible preparation script: `scripts/prepare_independent_test.py`

## Evaluation protocol

The YAML contains the original COCO128 training directory in its required `train` field and the held-out images in its `val` field. It is intended for the model validation/evaluation API only. **Do not pass this YAML to `model.train()`** because that would expose the test set during training-time validation and model selection.

The final evaluation should be run only after:

1. The final checkpoint has been selected without reference to this test set.
2. Confidence, NMS IoU, image size, and all preprocessing/postprocessing settings have been frozen.
3. The exact checkpoint hash and evaluation command/configuration have been recorded.
4. The evaluation is run once; if results are used to change the model or settings, this set is no longer an untouched test set and a new test set is required for a final unbiased claim.

## Source transport note

The COCO annotation records publish image URLs using HTTP. The download used those official `coco_url` values, then validated JPEG signatures and recorded local SHA-256 checksums. These checksums provide local artifact integrity for subsequent use; they are not publisher-provided source checksums.

**Current status:** Dataset acquisition, annotation conversion, integrity checks, and split-separation preflight are complete. No model training or test-set model evaluation has been run as part of this preparation.
