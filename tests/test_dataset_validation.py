import pytest

from visionlab.experiments import dataset_validation


def _make_split(tmp_path, train_names, val_names):
    train = tmp_path / "train"
    val = tmp_path / "val"
    train.mkdir()
    val.mkdir()
    for name in train_names:
        (train / name).write_bytes(b"train-image")
    for name in val_names:
        (val / name).write_bytes(b"val-image")
    return train, val


def test_validate_train_val_separation_accepts_disjoint_image_names(monkeypatch, tmp_path):
    train, val = _make_split(tmp_path, ["000001.jpg", "000002.jpg"], ["000003.jpg"])
    monkeypatch.setattr(dataset_validation, "check_det_dataset", lambda _: {
        "train": str(train), "val": str(val), "names": {0: "person"}
    })

    result = dataset_validation.validate_train_val_separation("dataset.yaml")

    assert result["train_images"] == 2
    assert result["val_images"] == 1
    assert result["filename_overlap"] == 0


def test_validate_train_val_separation_rejects_same_filename_in_different_dirs(
    monkeypatch, tmp_path
):
    train, val = _make_split(tmp_path, ["000001.jpg"], ["000001.jpg"])
    monkeypatch.setattr(dataset_validation, "check_det_dataset", lambda _: {
        "train": str(train), "val": str(val), "names": {0: "person"}
    })

    with pytest.raises(ValueError, match="leakage detected"):
        dataset_validation.validate_train_val_separation("dataset.yaml")


def test_validate_train_val_separation_rejects_empty_split(monkeypatch, tmp_path):
    train, val = _make_split(tmp_path, ["000001.jpg"], [])
    monkeypatch.setattr(dataset_validation, "check_det_dataset", lambda _: {
        "train": str(train), "val": str(val), "names": {0: "person"}
    })

    with pytest.raises(ValueError, match="existing image files"):
        dataset_validation.validate_train_val_separation("dataset.yaml")
