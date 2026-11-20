import pandas as pd

import visionlab.train as train_module


def test_train_passes_seed_to_ultralytics_and_exports_artifacts(monkeypatch, tmp_path):
    run_dir = tmp_path / "experiments" / "training" / "run1"
    (run_dir / "weights").mkdir(parents=True)
    (run_dir / "weights" / "best.pt").write_bytes(b"test-checkpoint")
    pd.DataFrame({"epoch": [1], "metrics/mAP50-95(B)": [0.5]}).to_csv(
        run_dir / "results.csv", index=False
    )

    class FakeModel:
        def __init__(self, model_name):
            self.model_name = model_name
            self.trainer = type("Trainer", (), {"save_dir": str(run_dir)})()
            self.train_args = None

        def train(self, **kwargs):
            self.train_args = kwargs

    model = FakeModel("yolo11n.pt")
    monkeypatch.setattr(train_module, "YOLO", lambda name: model)
    monkeypatch.setattr(train_module, "load_yaml", lambda path: {
        "model": "yolo11n.pt", "seed": 42, "epochs": 2, "data": "dataset.yaml",
        "save": True, "save_period": 5
    })
    monkeypatch.setattr(train_module, "set_seed", lambda seed: None)
    monkeypatch.setattr(train_module, "validate_train_val_separation", lambda path: {
        "train_images": 1, "val_images": 1, "filename_overlap": 0
    })
    monkeypatch.setattr(train_module, "project_path", lambda value: tmp_path / value)

    destination = train_module.train()

    assert model.train_args["seed"] == 42
    assert model.train_args["epochs"] == 2
    assert model.train_args["save"] is True
    assert model.train_args["save_period"] == 5
    assert destination == tmp_path / "models" / "trained" / "best.pt"
    assert destination.is_file()
    assert (tmp_path / "results" / "metrics" / "training_metrics.csv").is_file()
