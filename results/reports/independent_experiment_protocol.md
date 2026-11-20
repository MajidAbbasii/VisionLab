# Independent Experiment Protocol

Prepared: 2026-11-18

## Dataset split

| Split | Dataset | Images | Usage |
|---|---|---:|---|
| Train | COCO128 `images/train2017` | 128 | Gradient updates only |
| Validation | Official COCO2017 `val2017` | 5,000 | Epoch evaluation, Optuna objective, sensitivity and ablation comparison |

Both splits contain the same 80 class names in the same index order. Preflight resolved both directories, found zero identical image paths and zero filename-stem overlaps. The old COCO128 and COCO2017-only YAMLs both fail this preflight because their train and val definitions overlap.

## Prepared configurations

- Dataset: `datasets/coco128_train_coco2017_val.yaml`
- Full experiment / Optuna: `configs/experiments_coco128_train_coco2017_val.yaml`
- Clean baseline training: `configs/training_independent.yaml`
- Optuna study name: `visionlab_yolo11n_coco128_train_coco2017_val_v1`
- SQLite storage: existing `experiments/tuning/optuna.db`; prior studies remain untouched.
- Baseline checkpoint destination: `models/trained/best_coco128_train_coco2017_val.pt`
- Baseline metrics destination: `results/metrics/training_coco128_train_coco2017_val.csv`

The baseline checkpoint destination is separate from historical `models/trained/best.pt`. The tuning checkpoint is also study-specific. Sensitivity and ablation outputs use study-specific paths so they do not overwrite the historical exploratory outputs.

## Execution order

1. `python -m visionlab.train --config configs/training_independent.yaml`
2. `python -m visionlab.experiments.tuning --config configs/experiments_coco128_train_coco2017_val.yaml --trials 20`
3. `python -m visionlab.experiments.sensitivity --config configs/experiments_coco128_train_coco2017_val.yaml`
4. `python -m visionlab.experiments.ablation --config configs/experiments_coco128_train_coco2017_val.yaml`

All training/evaluation entry points now call a train/val split preflight before model training or validation. It rejects identical image paths, matching filename stems, or empty splits. Sensitivity uses batch size 1 to reduce validation memory pressure on the 4 GB GTX 1650 Max-Q.

## Execution status

- Config YAMLs loaded and resolved successfully.
- Dataset preflight passed: 128 train / 5,000 validation / 80 classes / zero overlap.
- Legacy overlapping dataset YAMLs correctly rejected.
- Python tests: 18 passed; compileall passed; all four CLI help commands passed.
- SQLite inspected read-only: existing study names were preserved; the new study has not been created.
- **No GPU training, Optuna trials, sensitivity runs or ablation runs have been started.** This avoids unexpectedly launching a long workload on the 4 GB laptop GPU before the user confirms the execution budget.

## Compute note

The full configured workload comprises a 50-epoch baseline, up to 20 Optuna trials of 50 epochs each, 16 validation-only sensitivity passes, and five 50-epoch ablation runs. Every training epoch validates against 5,000 images. Run these sequentially and monitor GPU memory, temperature and elapsed time; do not launch all stages concurrently.

