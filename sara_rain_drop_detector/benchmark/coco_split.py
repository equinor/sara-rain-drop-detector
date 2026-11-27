import json
from collections.abc import Callable
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
from pathlib import PurePosixPath
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image


@dataclass(frozen=True)
class CocoImage:
    image_id: int
    file_name: str
    width: int
    height: int


@dataclass
class CocoDetectionSplit:
    read: Callable[[str], bytes]
    split: str
    manifest_sha256: str
    dataset_id: str | None
    dataset: dict[str, Any]
    images: dict[int, CocoImage]
    boxes_by_image: dict[int, list[tuple[float, float, float, float]]]
    category_id: int

    @classmethod
    def load(cls, read: Callable[[str], bytes], split: str) -> "CocoDetectionSplit":
        manifest_bytes = read("manifest.json")
        dataset_id = parse_manifest_dataset_id(manifest_bytes)
        coco_data = json.loads(read(f"annotations/instances_{split}.json"))
        images, boxes_by_image, category_id = index_coco_detection_dataset(coco_data)
        return cls(
            read=read,
            split=split,
            manifest_sha256=sha256(manifest_bytes).hexdigest(),
            dataset_id=dataset_id,
            dataset=coco_data,
            images=images,
            boxes_by_image=boxes_by_image,
            category_id=category_id,
        )

    def read_image_bgr(self, image_id: int) -> NDArray[np.uint8]:
        entry = self.images[image_id]
        name = PurePosixPath(entry.file_name)
        if not name.parts or name.is_absolute() or ".." in name.parts:
            raise ValueError(f"Invalid COCO file_name for image {image_id}")
        with Image.open(BytesIO(self.read(f"images/{self.split}/{name}"))) as image:
            if image.size != (entry.width, entry.height):
                raise ValueError(f"COCO dimensions do not match image {image_id}")
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
            return rgb[:, :, ::-1].copy()

    def ground_truth_boxes(
        self, image_id: int
    ) -> list[tuple[float, float, float, float]]:
        return self.boxes_by_image.get(image_id, [])


def parse_manifest_dataset_id(manifest_bytes: bytes) -> str | None:
    manifest_data = json.loads(manifest_bytes)
    if not isinstance(manifest_data, dict):
        raise ValueError("Expected a JSON object in manifest.json")
    for key in ("dataset_id", "dataset", "id"):
        value = manifest_data.get(key)
        if value is not None:
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"Expected a non-empty string for {key} in manifest.json"
                )
            return value.strip()
    return None


def index_coco_detection_dataset(
    coco_data: dict[str, Any],
) -> tuple[
    dict[int, CocoImage], dict[int, list[tuple[float, float, float, float]]], int
]:
    raw_images = coco_data.get("images", [])
    raw_categories = coco_data.get("categories", [])
    raw_annotations = coco_data.get("annotations", [])

    images: dict[int, CocoImage] = {}
    for entry in raw_images:
        image_id = int(entry["id"])
        if image_id in images:
            raise ValueError("Duplicate COCO IDs in images")
        images[image_id] = CocoImage(
            image_id=image_id,
            file_name=str(entry["file_name"]),
            width=int(entry["width"]),
            height=int(entry["height"]),
        )

    categories: dict[int, str] = {}
    for entry in raw_categories:
        cat_id = int(entry["id"])
        if cat_id in categories:
            raise ValueError("Duplicate COCO IDs in categories")
        categories[cat_id] = str(entry["name"])

    if not images:
        raise ValueError("Empty COCO split")
    if len(categories) != 1:
        raise ValueError("Expected exactly one COCO category for rain-drop detection")

    category_id = next(iter(categories))
    seen_ann_ids: set[int] = set()
    boxes_by_image: dict[int, list[tuple[float, float, float, float]]] = {}
    for ann in raw_annotations:
        ann_id = int(ann["id"])
        if ann_id in seen_ann_ids:
            raise ValueError("Duplicate COCO IDs in annotations")
        seen_ann_ids.add(ann_id)

        image_id = int(ann["image_id"])
        ann_category_id = int(ann["category_id"])
        if image_id not in images or ann_category_id not in categories:
            raise ValueError("Invalid COCO annotation links")
        if ann.get("ignore"):
            continue
        bbox = ann.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError(f"Expected 4-element bbox for annotation {ann_id}")
        x, y, width, height = (float(v) for v in bbox)
        if width <= 0.0 or height <= 0.0:
            continue
        boxes_by_image.setdefault(image_id, []).append((x, y, width, height))

    return images, boxes_by_image, category_id
