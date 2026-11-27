import argparse
import json
import os
import platform
import re
from pathlib import Path
from time import perf_counter
from typing import Any

from azure.identity import WorkloadIdentityCredential
from azure.storage.blob import ContainerClient, ContentSettings

from sara_rain_drop_detector.benchmark.coco_split import CocoDetectionSplit
from sara_rain_drop_detector.benchmark.rain_metrics import (
    PER_IMAGE_COLUMNS,
    PROTOCOL,
    RainDropMetrics,
    box_to_coco_annotation,
)
from sara_rain_drop_detector.config.settings import settings
from sara_rain_drop_detector.inference import load_rain_model, predict_rain_detections

SCHEMA_VERSION = 3


class Storage:
    def __init__(self, location: str) -> None:
        self.root = Path(location)
        self.blob: ContainerClient | None = None
        if "://" in location:
            if not re.fullmatch(
                r"https://[a-z0-9]{3,24}\.blob\.core\.windows\.net/[a-z0-9-]+/?",
                location,
            ):
                raise ValueError(
                    "Expected an Azure Blob container URL without credentials"
                )
            connection_string = os.environ.get("BENCHMARK_STORAGE_CONNECTION_STRING")
            if connection_string:
                container = location.rstrip("/").rsplit("/", 1)[-1]
                self.blob = ContainerClient.from_connection_string(
                    connection_string, container_name=container
                )
            else:
                self.blob = ContainerClient.from_container_url(
                    location, WorkloadIdentityCredential()
                )

    @property
    def name(self) -> str:
        if self.blob is not None:
            return self.blob.container_name
        return self.root.resolve().name

    def exists(self, name: str) -> bool:
        if self.blob is not None:
            return self.blob.get_blob_client(name).exists()
        return (self.root / name).exists()

    def read(self, name: str) -> bytes:
        if self.blob is not None:
            return self.blob.download_blob(name).readall()
        return (self.root / name).read_bytes()

    def create_json(self, name: str, value: object) -> None:
        content = json.dumps(value, indent=2, allow_nan=False) + "\n"
        if self.blob is not None:
            self.blob.upload_blob(
                name,
                content,
                overwrite=False,
                content_settings=ContentSettings(content_type="application/json"),
            )
        else:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("x", encoding="utf-8") as file:
                file.write(content)


def evaluate_split(
    split: CocoDetectionSplit,
    device: str,
    confidence_threshold: float,
    model_path: Path,
) -> tuple[dict[str, Any], RainDropMetrics, list[dict[str, Any]], list[dict[str, Any]]]:
    model = load_rain_model(model_path)
    metrics = RainDropMetrics()
    per_image_items: list[dict[str, Any]] = []
    predicted_annotations: list[dict[str, Any]] = []
    next_annotation_id = 1
    elapsed = 0.0

    for image_id in sorted(split.images):
        image_bgr = split.read_image_bgr(image_id)
        started = perf_counter()
        detections = predict_rain_detections(
            model,
            image_bgr,
            confidence_threshold=confidence_threshold,
            device=device,
        )
        elapsed += perf_counter() - started

        image_eval = metrics.add_image(detections, split.ground_truth_boxes(image_id))
        entry = split.images[image_id]
        per_image_items.append(
            {
                "image_id": entry.image_id,
                "file_name": entry.file_name,
                "width": entry.width,
                "height": entry.height,
                "passed": image_eval.passed,
                "missed": image_eval.missed,
                "false_positive": image_eval.false_positive,
                "values": {
                    "gt_rain": image_eval.gt_rain,
                    "pred_rain": image_eval.pred_rain,
                    "gt_boxes": image_eval.gt_boxes,
                    "pred_boxes": image_eval.pred_boxes,
                    "max_score": image_eval.max_score,
                    "box_mean_iou": image_eval.box_mean_iou,
                },
            }
        )
        for detection in detections:
            predicted_annotations.append(
                box_to_coco_annotation(
                    next_annotation_id,
                    image_id,
                    split.category_id,
                    detection,
                )
            )
            next_annotation_id += 1

    count = len(split.images)
    processing = {
        "unit": "image",
        "count": count,
        "seconds": elapsed,
        "per_second": count / elapsed if elapsed > 0.0 else 0.0,
    }
    return processing, metrics, per_image_items, predicted_annotations


def hardware_name() -> str:
    cpu_info = Path("/proc/cpuinfo")
    if cpu_info.is_file():
        for line in cpu_info.read_text().splitlines():
            key, _, value = line.partition(":")
            if key.strip() == "model name":
                return value.strip()
    return platform.processor() or platform.machine()


def path_segment(value: str) -> str:
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._-]*", value):
        raise argparse.ArgumentTypeError("Expected a single path segment")
    return value


def parse_optional_float(value: str) -> float | None:
    stripped = value.strip()
    if not stripped:
        return None
    return float(stripped)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark rain drop detection")
    parser.add_argument(
        "--source", required=True, help="Dataset directory or container URL"
    )
    parser.add_argument(
        "--destination", required=True, help="Results directory or container URL"
    )
    parser.add_argument("--version", required=True, type=path_segment)
    parser.add_argument("--split", default="val", type=path_segment)
    parser.add_argument("--run-id", required=True, type=path_segment)
    parser.add_argument("--device", default="cpu", choices=("cpu",))
    parser.add_argument(
        "--artifact", required=True, help="Exact rain-drop-detector image or build ID"
    )
    parser.add_argument(
        "--confidence-threshold",
        default=None,
        type=parse_optional_float,
        help="Detection confidence threshold override",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        type=Path,
        help="Optional override for YOLO model weights path",
    )
    return parser


def write_benchmark_outputs(
    destination: Storage,
    prefix: str,
    args: argparse.Namespace,
    source_name: str,
    split: CocoDetectionSplit,
    model_path: Path,
    confidence_threshold: float,
    processing: dict[str, Any],
    metrics: RainDropMetrics,
    per_image_items: list[dict[str, Any]],
    predicted_annotations: list[dict[str, Any]],
) -> None:
    common = {"schema_version": SCHEMA_VERSION, "run_id": args.run_id}
    destination.create_json(
        f"{prefix}/run.json",
        {
            **common,
            "dataset": {
                "id": split.dataset_id or source_name,
                "version": args.version,
                "split": args.split,
                "manifest_sha256": split.manifest_sha256,
            },
            "artifact": args.artifact,
            "evaluator": args.artifact,
            "protocol": PROTOCOL,
            "configuration": {
                "model": model_path.name,
                "confidence_threshold": confidence_threshold,
            },
            "execution": {
                "hardware": hardware_name(),
                "timing_scope": "inference_only",
            },
        },
    )
    destination.create_json(
        f"{prefix}/metrics.json",
        {
            **common,
            "processing": processing,
            "summary_metrics": metrics.as_summary_metrics(),
            "evaluations": metrics.as_evaluations(),
        },
    )
    destination.create_json(
        f"{prefix}/review/per_image.json",
        {
            "columns": PER_IMAGE_COLUMNS,
            "items": per_image_items,
        },
    )
    destination.create_json(
        f"{prefix}/review/instances_predictions.json",
        {
            "info": {"description": f"Predictions for {args.run_id} ({args.artifact})"},
            "licenses": split.dataset.get("licenses", []),
            "categories": split.dataset["categories"],
            "images": split.dataset["images"],
            "annotations": predicted_annotations,
        },
    )


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    confidence_threshold = (
        settings.DETECTION_CONFIDENCE_THRESHOLD
        if args.confidence_threshold is None
        else args.confidence_threshold
    )
    if not (0.0 < confidence_threshold <= 1.0):
        parser.error("confidence-threshold must be in (0.0, 1.0]")

    model_path = args.model_path or Path(settings.MODEL_PATH)
    source, destination = Storage(args.source), Storage(args.destination)
    prefix = f"{args.version}/benchmark-results/{args.split}/{args.run_id}"
    if any(
        destination.exists(f"{prefix}/{name}") for name in ("run.json", "metrics.json")
    ):
        raise FileExistsError(f"Benchmark run already exists: {prefix}")

    split = CocoDetectionSplit.load(
        lambda name: source.read(f"{args.version}/dataset/{name}"),
        args.split,
    )
    processing, metrics, per_image_items, predicted_annotations = evaluate_split(
        split=split,
        device=args.device,
        confidence_threshold=confidence_threshold,
        model_path=model_path,
    )
    write_benchmark_outputs(
        destination=destination,
        prefix=prefix,
        args=args,
        source_name=source.name,
        split=split,
        model_path=model_path,
        confidence_threshold=confidence_threshold,
        processing=processing,
        metrics=metrics,
        per_image_items=per_image_items,
        predicted_annotations=predicted_annotations,
    )


if __name__ == "__main__":
    main()
