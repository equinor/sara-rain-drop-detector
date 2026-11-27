import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from ultralytics import YOLO

from sara_rain_drop_detector.config.settings import settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RainBoxDetection:
    bbox: tuple[float, float, float, float]
    score: float | None = None


def load_rain_model(model_path: Path | None = None) -> YOLO:
    resolved_path = model_path or Path(settings.MODEL_PATH)
    if not resolved_path.is_file():
        raise FileNotFoundError(f"YOLO model file not found at {resolved_path}")

    logger.info(f"Loading YOLO model from {resolved_path}")
    return YOLO(str(resolved_path))


def extract_box_detections(results: Iterable[Any]) -> list[RainBoxDetection]:
    detections: list[RainBoxDetection] = []
    for result in results:
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue
        xyxy = getattr(boxes, "xyxy", None)
        if xyxy is None:
            continue
        coords = xyxy.tolist()
        raw_conf = getattr(boxes, "conf", None)
        scores = raw_conf.tolist() if raw_conf is not None else [None] * len(coords)
        for coord, score in zip(coords, scores):
            x1, y1, x2, y2 = (float(v) for v in coord)
            width = max(0.0, x2 - x1)
            height = max(0.0, y2 - y1)
            if width <= 0.0 or height <= 0.0:
                continue
            detections.append(
                RainBoxDetection(
                    bbox=(x1, y1, width, height),
                    score=float(score) if score is not None else None,
                )
            )
    return detections


def predict_rain_detections(
    model: YOLO,
    source: Path | NDArray[np.uint8],
    confidence_threshold: float | None = None,
    device: str | None = None,
) -> list[RainBoxDetection]:
    threshold = (
        settings.DETECTION_CONFIDENCE_THRESHOLD
        if confidence_threshold is None
        else confidence_threshold
    )
    predict_source: str | NDArray[np.uint8] = (
        str(source) if isinstance(source, Path) else source
    )
    predict_kwargs: dict[str, Any] = {
        "source": predict_source,
        "conf": threshold,
        "verbose": False,
    }
    if device is not None:
        predict_kwargs["device"] = device
    results = model.predict(**predict_kwargs)
    return extract_box_detections(results)


def detect_rain(image_path: Path) -> bool:
    """Run YOLOv11 inference on the given image and report whether rain was detected.

    Returns True if the model produces at least one detection above the
    configured confidence threshold, otherwise False.
    """
    model = load_rain_model()

    logger.info(f"Running inference on {image_path}")
    results = model.predict(
        source=str(image_path),
        conf=settings.DETECTION_CONFIDENCE_THRESHOLD,
        verbose=False,
    )

    num_detections = 0
    for r in results:
        boxes = getattr(r, "boxes", None)
        if boxes is not None:
            num_detections += len(boxes)
    logger.info(f"Inference produced {num_detections} detection(s)")

    return num_detections > 0
