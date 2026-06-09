import logging
from pathlib import Path

from ultralytics import YOLO

from sara_rain_drop_detector.config.settings import settings

logger = logging.getLogger(__name__)


def detect_rain(image_path: Path) -> bool:
    """Run YOLOv11 inference on the given image and report whether rain was detected.

    Returns True if the model produces at least one detection above the
    configured confidence threshold, otherwise False.
    """
    model_path = Path(settings.MODEL_PATH)
    if not model_path.is_file():
        raise FileNotFoundError(f"YOLO model file not found at {model_path}")

    logger.info(f"Loading YOLO model from {model_path}")
    model = YOLO(str(model_path))

    logger.info(f"Running inference on {image_path}")
    results = model.predict(
        source=str(image_path),
        conf=settings.DETECTION_CONFIDENCE_THRESHOLD,
        verbose=False,
    )

    num_detections = sum(0 if r.boxes is None else len(r.boxes) for r in results)
    logger.info(f"Inference produced {num_detections} detection(s)")

    return num_detections > 0
