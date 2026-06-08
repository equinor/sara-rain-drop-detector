import json
from pathlib import Path
from unittest.mock import patch

import pytest

from sara_rain_drop_detector import main_rain_drop_detector
from sara_rain_drop_detector.models.blob_storage_location import BlobStorageLocation


@pytest.fixture
def input_location() -> BlobStorageLocation:
    return BlobStorageLocation(blob_container="c", blob_name="img.png")


def test_workflow_writes_true_when_rain_detected(
    tmp_path: Path, input_location: BlobStorageLocation
) -> None:
    result_file = tmp_path / "result.json"

    with (
        patch.object(
            main_rain_drop_detector,
            "download_image_to_local",
            return_value=tmp_path / "img.png",
        ) as download,
        patch.object(
            main_rain_drop_detector, "detect_rain", return_value=True
        ) as detect,
    ):
        main_rain_drop_detector.run_rain_drop_detection_workflow(
            input_location, str(result_file)
        )

    download.assert_called_once_with(input_location)
    detect.assert_called_once_with(tmp_path / "img.png")
    assert json.loads(result_file.read_text()) == {"rain": True}


def test_workflow_writes_false_when_no_rain(
    tmp_path: Path, input_location: BlobStorageLocation
) -> None:
    result_file = tmp_path / "result.json"

    with (
        patch.object(
            main_rain_drop_detector,
            "download_image_to_local",
            return_value=tmp_path / "img.png",
        ),
        patch.object(main_rain_drop_detector, "detect_rain", return_value=False),
    ):
        main_rain_drop_detector.run_rain_drop_detection_workflow(
            input_location, str(result_file)
        )

    assert json.loads(result_file.read_text()) == {"rain": False}
