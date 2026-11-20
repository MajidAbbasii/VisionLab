# VisionLab — Consolidated Experimental Results

**Prepared:** 2026-11-19  
**Model family:** Ultralytics YOLO11n  
**Training data:** COCO128 train2017 (128 images)  
**Validation data:** COCO2017 val2017 (5,000 images; 80 classes)

## 1. Executive summary

The four planned experimental stages are complete: baseline training, 20-trial Optuna tuning, 16 sensitivity evaluations, and a five-variant ablation study. The train and validation splits passed the project leakage preflight (128/5,000 images; zero path overlap; zero filename-stem overlap; 80 classes).

The best Optuna result was trial 14 with mAP50-95 = **0.38271**. Compared with the original baseline best checkpoint (**0.37907**), this is an absolute difference of **+0.00364** (about **+0.96% relative**). This is a small gain and should not be described as a robust generalization improvement.

**Critical interpretation:** COCO2017 val2017 was used repeatedly for epoch validation, Optuna objective selection, sensitivity analysis and ablation comparisons. It is therefore a validation set that has influenced model/configuration selection, not an untouched final test set. These results are exploratory/model-selection evidence only. A genuinely independent test set is still required before making an unbiased final performance claim.

## 2. Dataset integrity and protocol

| Split | Images | Role |
|---|---:|---|
| COCO128 train2017 | 128 | Training / gradient updates |
| COCO2017 val2017 | 5,000 | Validation and experiment selection |

Preflight results: `path_overlap = 0`, `filename_overlap = 0`, `classes = 80`. Class names and index ordering are shared. This confirms image-level separation between these two splits, but does **not** make the validation set independent after repeated use for tuning and comparisons.

## 3. Baseline training

- Run configuration: `configs/training_independent.yaml`
- Checkpoint: `models/trained/best_coco128_train_coco2017_val.pt`
- Metrics: `results/metrics/training_coco128_train_coco2017_val.csv`
- Early stopping: epoch 21; best checkpoint epoch 1.

| Metric at best epoch | Value |
|---|---:|
| Precision | 0.65985 |
| Recall | 0.48913 |
| mAP50 | 0.53795 |
| mAP50-95 | 0.37907 |

The validation metric peaked at the first epoch and declined afterward. This is consistent with severe overfitting risk given only 128 training images; the early-stopping result should be interpreted accordingly.

## 4. Optuna hyperparameter tuning

- Study: `visionlab_yolo11n_coco128_train_coco2017_val_v1`
- Storage: `experiments/tuning/optuna.db`
- Trials: 20 total — 13 COMPLETE, 6 PRUNED, 1 FAIL, 0 RUNNING.
- Sampler: TPE (seed 42); pruner: MedianPruner.
- Best trial: **14**, objective mAP50-95 = **0.38271**.
- Checkpoint: `models/trained/best_tuned_visionlab_yolo11n_coco128_train_coco2017_val_v1.pt`

Top completed trials:

| Trial | mAP50-95 |
|---:|---:|
| 14 | 0.38271 |
| 19 | 0.38262 |
| 16 | 0.38241 |
| 17 | 0.38235 |
| 18 | 0.38199 |

Best trial 14 parameters:

| Parameter | Value |
|---|---:|
| lrf | 0.0028478574 |
| weight_decay | 0.0001123974 |
| warmup_epochs | 1.9016502 |
| hsv_h | 0.0179712 |
| hsv_s | 0.3078842 |
| hsv_v | 0.3897615 |
| mosaic | 0.5173636 |
| mixup | 0.0655424 |

The difference between the best trial and the original baseline is +0.00364 absolute mAP50-95 (approximately +0.96% relative). The margin between the top trials is also very small. In addition, trial 14's best metric occurred at epoch 1 while its last-epoch metric was substantially lower (approximately 0.30909), reinforcing the overfitting/instability concern.

## 5. Sensitivity analysis

Six confidence thresholds, six NMS IoU thresholds and four image sizes were evaluated (16 total runs), using the baseline checkpoint and the 5,000-image validation split.

Best tested setting within each one-dimensional sweep:

| Parameter varied | Best tested value | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|
| confidence | 0.10 | 0.66488 | 0.50564 | 0.50017 | 0.35396 |
| NMS IoU | 0.70 | 0.63627 | 0.49877 | 0.44721 | 0.32946 |
| image size | 768 | 0.67215 | 0.51417 | 0.46350 | 0.33166 |

These are independent one-factor sweeps. The three best values were **not** evaluated together, so no combined optimal inference configuration has been established. The confidence sweep shows the expected precision/recall trade-off: increasing confidence raises precision while reducing recall. These settings are validation-dependent and should be rechecked on an untouched test set.

## 6. Ablation study

Five variants were trained using the Trial 14 hyperparameter configuration as the common base. Each run stopped at epoch 21 and selected epoch 1 as its best checkpoint.

| Variant | Precision | Recall | mAP50 | mAP50-95 | Delta vs tuned baseline |
|---|---:|---:|---:|---:|---:|
| baseline (Trial 14 config) | 0.6541 | 0.4963 | 0.5413 | **0.38271** | 0.00000 |
| no_fliplr | 0.6509 | 0.4995 | 0.5410 | 0.38252 | -0.00019 |
| no_mixup | 0.6399 | 0.5043 | 0.5403 | 0.38244 | -0.00027 |
| reduced_augmentation | 0.6393 | 0.4995 | 0.5372 | 0.37937 | -0.00334 |
| no_mosaic | 0.6473 | 0.4960 | 0.5365 | 0.37871 | -0.00400 |

Interpretation:

- Removing Mosaic produced the largest observed drop (−0.00400 mAP50-95; about −1.05% relative).
- Removing MixUp or horizontal flip had only very small differences in this experiment.
- The reduced-augmentation variant also underperformed the tuned baseline.
- This is a small-data, single-run ablation. Without repeated seeds and an untouched test set, small differences should not be treated as statistically established effects.

**Important comparison note:** the ablation baseline is the Trial 14 configuration, not the original baseline-training run. Its score (0.38271) is therefore not interchangeable with the original baseline score (0.37907).

## 7. Overall conclusions

1. The experimental workflow and the four planned experiment stages have completed successfully.
2. Optuna found a configuration with a modest validation-score gain, but the gain is small and the best epoch behavior indicates instability/overfitting risk.
3. Mosaic appears to be the most consequential of the tested augmentations in this particular ablation; MixUp and horizontal flip had negligible measured effects.
4. The sensitivity sweeps show that inference thresholds affect the precision/recall balance. The best one-dimensional settings have not been combined or independently validated.
5. The 128-image training set is a major limitation. The repeated use of COCO2017 val2017 for model selection prevents it from serving as an unbiased final test set.

## 8. Required next steps (not performed in this report)

1. **Completed 2026-11-19:** Prepared a 5,000-image held-out test sample from COCO train2017, excluding all 128 COCO128 training images and verifying zero overlap with the already-used 5,000-image val2017 split. See `results/reports/independent_test_dataset.md`. Do not use this test set to tune hyperparameters or thresholds.
2. Decide whether the training data should be expanded substantially; 128 images are inadequate for a strong general-purpose 80-class detector.
3. Agree on the final training protocol and evaluation budget before starting final training.
4. Train the final candidate only after explicit approval; then evaluate it once on the untouched test set.
5. Export ONNX and validate numerical/behavioral parity, then proceed to the C# runtime integration and performance tests.

## 9. Source artifacts

- Baseline metrics: `results/metrics/training_coco128_train_coco2017_val.csv`
- Optuna summary: `results/reports/hyperparameter_tuning_visionlab_yolo11n_coco128_train_coco2017_val_v1.json`
- Optuna trials: `results/metrics/optuna_visionlab_yolo11n_coco128_train_coco2017_val_v1_trials.csv`
- Sensitivity summary: `results/reports/sensitivity_visionlab_yolo11n_coco128_train_coco2017_val_v1.json`
- Sensitivity metrics: `results/metrics/sensitivity_visionlab_yolo11n_coco128_train_coco2017_val_v1.csv`
- Ablation summary: `results/metrics/ablation_visionlab_yolo11n_coco128_train_coco2017_val_v1_rerun_20261119.csv`
- Dataset split: `datasets/coco128_train_coco2017_val.yaml`

---

**Status:** Experiment consolidation and independent test dataset preparation complete. No final training or test-set model evaluation has been started.

