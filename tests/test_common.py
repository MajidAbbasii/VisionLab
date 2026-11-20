from visionlab.common import load_yaml, project_path

def test_project_paths_are_rooted():
    """Relative paths must resolve from the repository, not the caller's cwd."""
    assert project_path("configs/model.yaml").is_absolute()

def test_model_config_has_yolo_default():
    """The default model configuration should select an Ultralytics YOLO model."""
    config = load_yaml("configs/model.yaml")
    assert config["model"].startswith("yolo")

