from visionlab.common import load_yaml, project_path

def test_project_paths_are_rooted():
    assert project_path("configs/model.yaml").is_absolute()

def test_model_config_has_yolo_default():
    config = load_yaml("configs/model.yaml")
    assert config["model"].startswith("yolo")

