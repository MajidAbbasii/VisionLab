from types import SimpleNamespace

import optuna
import pandas as pd
import pytest

from visionlab.experiments.tuning import (
    _config_fingerprint,
    _legacy_config_fingerprint,
    _make_pruning_callback,
    _read_best_score,
    _validation_map50_95,
)


def test_read_best_score_uses_max_validation_map(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    pd.DataFrame({
        "metrics/precision(B)": [0.7, 0.8],
        "metrics/mAP50-95(B)": [0.51, 0.63],
    }).to_csv(run_dir / "results.csv", index=False)

    assert _read_best_score(run_dir) == 0.63


def test_config_fingerprint_is_stable_and_includes_search_space():
    config = {
        "model": "yolo11n.pt",
        "data": "datasets/coco128.yaml",
        "epochs": 50,
        "tuning": {"trials": 10, "search_space": {"lrf": [0.001, 0.1]}},
    }
    same_config = {
        "tuning": {"search_space": {"lrf": [0.001, 0.1]}, "trials": 10},
        "epochs": 50,
        "data": "datasets/coco128.yaml",
        "model": "yolo11n.pt",
    }
    changed_config = {**config, "tuning": {"trials": 10, "search_space": {"lrf": [0.01, 0.1]}}}

    assert _config_fingerprint(config) == _config_fingerprint(same_config)
    assert _config_fingerprint(config) != _config_fingerprint(changed_config)


def test_legacy_config_fingerprint_excludes_sampler_and_pruner():
    config = {
        "model": "yolo11n.pt",
        "data": "datasets/coco128.yaml",
        "epochs": 50,
        "tuning": {
            "search_space": {"lrf": [0.001, 0.1]},
            "sampler": {"type": "tpe", "seed": 42},
            "pruner": {"type": "median", "n_warmup_steps": 15},
        },
    }
    changed_optimizer = {
        **config,
        "tuning": {
            **config["tuning"],
            "sampler": {"type": "tpe", "seed": 99},
            "pruner": {"type": "median", "n_warmup_steps": 30},
        },
    }

    assert _legacy_config_fingerprint(config) == _legacy_config_fingerprint(changed_optimizer)
    assert _config_fingerprint(config) != _config_fingerprint(changed_optimizer)


def test_config_fingerprint_includes_pruner_configuration():
    config = {
        "model": "yolo11n.pt",
        "data": "datasets/coco128.yaml",
        "epochs": 50,
        "tuning": {
            "search_space": {"lrf": [0.001, 0.1]},
            "pruner": {"type": "median", "n_warmup_steps": 15},
        },
    }
    changed = {
        **config,
        "tuning": {
            **config["tuning"],
            "pruner": {"type": "median", "n_warmup_steps": 20},
        },
    }

    assert _config_fingerprint(config) != _config_fingerprint(changed)


def test_validation_map50_95_extracts_ultralytics_metric():
    assert _validation_map50_95({"metrics/mAP50-95(B)": 0.634}) == 0.634
    assert _validation_map50_95({"metrics/precision(B)": 0.8}) is None


def test_pruning_callback_reports_score_and_prunes_trial():
    class FakeTrial:
        def __init__(self):
            self.reports = []
            self.user_attrs = {}

        def report(self, value, step):
            self.reports.append((value, step))

        def should_prune(self):
            return True

        def set_user_attr(self, key, value):
            self.user_attrs[key] = value

    trial = FakeTrial()
    callback = _make_pruning_callback(trial, total_epochs=50)
    trainer = SimpleNamespace(epoch=14, metrics={"metrics/mAP50-95(B)": 0.42})

    with pytest.raises(optuna.TrialPruned):
        callback(trainer)

    assert trial.reports == [(0.42, 15)]
    assert trial.user_attrs["pruned_epoch"] == 15
    assert trial.user_attrs["pruned_map50_95"] == 0.42


def test_pruning_callback_does_not_prune_on_final_epoch():
    class FakeTrial:
        def report(self, value, step):
            raise AssertionError("Final epoch must not be reported for pruning")

        def should_prune(self):
            raise AssertionError("Final epoch must not trigger pruning")

    callback = _make_pruning_callback(FakeTrial(), total_epochs=50)
    callback(SimpleNamespace(epoch=49, metrics={"metrics/mAP50-95(B)": 0.9}))


def test_optuna_records_callback_pruned_trial_as_pruned():
    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=1,
            n_warmup_steps=0,
        ),
    )
    study.optimize(
        lambda trial: (trial.report(0.8, step=1), 0.8)[1],
        n_trials=1,
    )

    def objective(trial):
        callback = _make_pruning_callback(trial, total_epochs=2)
        callback(SimpleNamespace(epoch=0, metrics={"metrics/mAP50-95(B)": 0.2}))
        return 0.2

    study.optimize(objective, n_trials=1)

    assert study.trials[-1].state == optuna.trial.TrialState.PRUNED
    assert study.trials[-1].user_attrs["pruned_epoch"] == 1
