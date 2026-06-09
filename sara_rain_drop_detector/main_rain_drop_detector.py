import json
import logging
from pathlib import Path

from sara_rain_drop_detector.blob import download_image_to_local
from sara_rain_drop_detector.inference import detect_rain
from sara_rain_drop_detector.models.blob_storage_location import BlobStorageLocation

logger = logging.getLogger(__name__)


def run_rain_drop_detection_workflow(
    input_location: BlobStorageLocation,
    result_output_file: str,
) -> None:
    """
    Runs the rain drop detection workflow.

    Args:
        input_location: The blob storage location of the image to analyze.
        result_output_file: The path to write the workflow result JSON to. The
            file is created after the workflow succeeds and contains
            'rain' (bool).
    """

    image_path: Path = download_image_to_local(input_location)

    rain_present = detect_rain(image_path)
    logger.info(f"Rain present: {rain_present}")

    result_path = Path(result_output_file)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps({"rain": rain_present}))
    logger.info(f"Wrote workflow result to {result_path}")
