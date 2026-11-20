import numpy as np
from pathlib import Path

from visionlab import detector


class FakeCapture:
    def __init__(self, source):
        self.source = source
        self.frames = [np.zeros((8, 8, 3), dtype=np.uint8)]
        self.released = False

    def isOpened(self):
        return True

    def get(self, property_id):
        return 25.0

    def set(self, property_id, value):
        return True

    def read(self):
        if self.frames:
            return True, self.frames.pop()
        return False, None

    def release(self):
        self.released = True


class ClosedCapture(FakeCapture):
    def isOpened(self):
        return False


def test_iter_video_frames_reports_source_fps_and_releases_capture(monkeypatch):
    capture = FakeCapture("sample.mp4")
    monkeypatch.setattr(detector.cv2, "VideoCapture", lambda source: capture)
    reported_fps = []

    frames = list(detector.iter_video_frames("sample.mp4", on_open=reported_fps.append))

    assert len(frames) == 1
    assert reported_fps == [25.0]
    assert capture.released


def test_iter_video_frames_releases_capture_when_source_cannot_open(monkeypatch):
    capture = ClosedCapture("missing.mp4")
    monkeypatch.setattr(detector.cv2, "VideoCapture", lambda source: capture)

    try:
        list(detector.iter_video_frames("missing.mp4"))
    except RuntimeError as error:
        assert "Cannot open video source" in str(error)
    else:
        raise AssertionError("An unopened video source must raise RuntimeError")

    assert capture.released


def test_detect_video_uses_source_fps_and_releases_resources(monkeypatch, tmp_path):
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    capture = FakeCapture("sample.mp4")
    writer_state = {}

    class FakeWriter:
        def __init__(self, path, codec, fps, size):
            writer_state.update(path=path, codec=codec, fps=fps, size=size, frames=0, released=False)

        def isOpened(self):
            return True

        def write(self, image):
            writer_state["frames"] += 1

        def release(self):
            writer_state["released"] = True

    class FakeResult:
        def plot(self):
            return frame

    class FakeModel:
        def predict(self, image, **kwargs):
            assert kwargs["rect"] is False
            return [FakeResult()]

    monkeypatch.setattr(detector.cv2, "VideoCapture", lambda source: capture)
    monkeypatch.setattr(detector.cv2, "VideoWriter", FakeWriter)
    monkeypatch.setattr(detector.cv2, "VideoWriter_fourcc", lambda *args: 0)
    monkeypatch.setattr(detector, "load_detector", lambda model_path: FakeModel())
    target = tmp_path / "annotated.mp4"

    result = detector.detect_video("sample.mp4", output_path=str(target))

    assert result == Path(target)
    assert writer_state["fps"] == 25.0
    assert writer_state["frames"] == 1
    assert writer_state["released"]
    assert capture.released


def test_detect_video_stops_after_requested_frame_count(monkeypatch, tmp_path):
    capture = FakeCapture("sample.mp4")
    capture.frames = [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(3)]
    writer_state = {"frames": 0, "released": False}

    class FakeWriter:
        def __init__(self, path, codec, fps, size):
            pass
        def isOpened(self):
            return True
        def write(self, image):
            writer_state["frames"] += 1
        def release(self):
            writer_state["released"] = True

    class FakeResult:
        def plot(self):
            return np.zeros((8, 8, 3), dtype=np.uint8)

    class FakeModel:
        def predict(self, image, **kwargs):
            return [FakeResult()]

    monkeypatch.setattr(detector.cv2, "VideoCapture", lambda source: capture)
    monkeypatch.setattr(detector.cv2, "VideoWriter", FakeWriter)
    monkeypatch.setattr(detector.cv2, "VideoWriter_fourcc", lambda *args: 0)
    monkeypatch.setattr(detector, "load_detector", lambda model_path: FakeModel())

    detector.detect_video("sample.mp4", output_path=str(tmp_path / "bounded.mp4"), max_frames=2)

    assert writer_state["frames"] == 2
    assert writer_state["released"]
    assert capture.released
