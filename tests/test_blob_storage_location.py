import pytest
from pydantic import ValidationError

from sara_rain_drop_detector.models.blob_storage_location import BlobStorageLocation


def test_blob_storage_location_accepts_camel_case_alias() -> None:
    loc = BlobStorageLocation.model_validate(
        {"blobContainer": "my-container", "blobName": "path/to/img.png"}
    )

    assert loc.blob_container == "my-container"
    assert loc.blob_name == "path/to/img.png"


def test_blob_storage_location_rejects_empty_container() -> None:
    with pytest.raises(ValidationError):
        BlobStorageLocation(blob_container="", blob_name="img.png")
