from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from sara_rain_drop_detector import inference


def _yolo_result(num_boxes: int | None) -> SimpleNamespace:
    """Build a fake ultralytics Results-like object.

    Pass None to simulate a result with no `boxes` attribute populated.
    """
    boxes = None if num_boxes is None else [object()] * num_boxes
    return SimpleNamespace(boxes=boxes)


@pytest.fixture
def model_file(tmp_path: Path) -> Path:
    path = tmp_path / "model.pt"
    path.write_bytes(b"")
    return path


def test_detect_rain_returns_true_when_detections_present(
    monkeypatch: pytest.MonkeyPatch, model_file: Path, tmp_path: Path
) -> None:
    monkeypatch.setattr(inference.settings, "MODEL_PATH", str(model_file))
    monkeypatch.setattr(inference.settings, "DETECTION_CONFIDENCE_THRESHOLD", 0.25)

    fake_model = MagicMock()
    fake_model.predict.return_value = [_yolo_result(num_boxes=2)]

    with patch.object(inference, "YOLO", return_value=fake_model) as yolo_ctor:
        result = inference.detect_rain(tmp_path / "img.png")

    assert result is True
    yolo_ctor.assert_called_once_with(str(model_file))
    fake_model.predict.assert_called_once_with(
        source=str(tmp_path / "img.png"),
        conf=0.25,
        verbose=False,
    )


def test_detect_rain_returns_false_when_no_detections(
    monkeypatch: pytest.MonkeyPatch, model_file: Path, tmp_path: Path
) -> None:
    monkeypatch.setattr(inference.settings, "MODEL_PATH", str(model_file))

    fake_model = MagicMock()
    fake_model.predict.return_value = [_yolo_result(num_boxes=0)]

    with patch.object(inference, "YOLO", return_value=fake_model):
        assert inference.detect_rain(tmp_path / "img.png") is False


def test_detect_rain_raises_when_model_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        inference.settings, "MODEL_PATH", str(tmp_path / "does-not-exist.pt")
    )

    with pytest.raises(FileNotFoundError, match="YOLO model file not found"):
        inference.detect_rain(tmp_path / "img.png")
