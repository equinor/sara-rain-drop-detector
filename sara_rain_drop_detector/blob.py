import logging
from io import BytesIO
from pathlib import Path

import numpy as np
from azure.storage.blob import BlobServiceClient
from numpy.typing import NDArray
from PIL import Image

from sara_rain_drop_detector.config.settings import settings
from sara_rain_drop_detector.models.blob_storage_location import BlobStorageLocation

logger = logging.getLogger(__name__)


def download_blob_to_bytes(
    blob_service_client: BlobServiceClient, blob_storage_location: BlobStorageLocation
) -> bytes:
    blob_client = blob_service_client.get_blob_client(
        container=blob_storage_location.blob_container,
        blob=blob_storage_location.blob_name,
    )
    return blob_client.download_blob().readall()


def download_blob_to_image(
    blob_service_client: BlobServiceClient, blob_storage_location: BlobStorageLocation
) -> NDArray[np.uint8]:
    blob_data = download_blob_to_bytes(blob_service_client, blob_storage_location)
    image = Image.open(BytesIO(blob_data))
    return np.array(image)


def download_image_to_local(
    image_location: BlobStorageLocation,
) -> Path:
    logger.info("Processing new image")

    src_blob_service_client = BlobServiceClient.from_connection_string(
        settings.SOURCE_STORAGE_CONNECTION_STRING
    )

    image_array: NDArray[np.uint8] = download_blob_to_image(
        src_blob_service_client, image_location
    )
    logger.info(
        f"Downloaded image from source storage account, shape: {image_array.shape}"
    )

    image_path = Path(image_location.blob_name)
    image_path.parent.mkdir(parents=True, exist_ok=True)

    image = Image.fromarray(image_array)
    image.save(image_path)

    logger.info(f"Saved image locally at {image_path}")

    return image_path
