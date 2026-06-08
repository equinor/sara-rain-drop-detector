"""Typer callbacks for parsing the generic SARA blob-location CLI contract.

SARA invokes analyzer images with three JSON-encoded options:

* ``--input-blob-storage-locations``  -- a JSON array of blob locations.

sara-rain-drop-detector is a single-image analyzer, so the input array must
contain exactly one element. Anything else is a hard failure surfaced
through Typer.
"""

import json
from typing import List, Type, TypeVar

import typer
from pydantic import BaseModel, ValidationError

from sara_rain_drop_detector.models.blob_storage_location import BlobStorageLocation

ExtrasT = TypeVar("ExtrasT", bound=BaseModel)


def parse_input_blob_storage_locations(value: str) -> List[BlobStorageLocation]:
    try:
        raw = json.loads(value)
    except json.JSONDecodeError as exc:
        raise typer.BadParameter(
            f"--input-blob-storage-locations is not valid JSON: {exc}"
        ) from exc

    if not isinstance(raw, list):
        raise typer.BadParameter(
            "--input-blob-storage-locations must be a JSON array of blob "
            f"locations, got {type(raw).__name__}."
        )

    if len(raw) != 1:
        raise typer.BadParameter(
            "--input-blob-storage-locations must contain exactly one entry "
            f"(sara-rain-drop-detector is a single-image analyzer); got {len(raw)}."
        )

    try:
        return [BlobStorageLocation.model_validate(item) for item in raw]
    except ValidationError as exc:
        raise typer.BadParameter(
            f"--input-blob-storage-locations entry is invalid: {exc}"
        ) from exc
