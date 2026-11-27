import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from sara_rain_drop_detector.benchmark.__main__ import (
    SCHEMA_VERSION,
    build_arg_parser,
    main,
)
from sara_rain_drop_detector.benchmark.rain_metrics import (
    PROTOCOL,
    RainDropMetrics,
    compute_box_iou,
    match_boxes_by_iou,
)
from sara_rain_drop_detector.inference import RainBoxDetection


def _write_dataset(
    root: Path,
    images: list[dict],
    annotations: list[dict],
    split: str = "val",
) -> None:
    dataset_root = root / "v001" / "dataset"
    (dataset_root / "annotations").mkdir(parents=True)
    (dataset_root / "images" / split).mkdir(parents=True)

    (dataset_root / "manifest.json").write_text(
        json.dumps({"dataset": "rain-drop-detection", "version": "v001"}),
        encoding="utf-8",
    )
    (dataset_root / "annotations" / f"instances_{split}.json").write_text(
        json.dumps(
            {
                "categories": [{"id": 1, "name": "Rain drop", "supercategory": ""}],
                "images": images,
                "annotations": annotations,
            }
        ),
        encoding="utf-8",
    )
    for img in images:
        pil_img = Image.new("RGB", (img["width"], img["height"]), color=(128, 128, 128))
        pil_img.save(dataset_root / "images" / split / img["file_name"])


def test_rain_metrics_tracks_image_classification_and_box_iou() -> None:
    metrics = RainDropMetrics()

    tp_eval = metrics.add_image(
        predictions=[
            RainBoxDetection(bbox=(10.0, 10.0, 20.0, 20.0), score=0.9),
            RainBoxDetection(bbox=(80.0, 80.0, 10.0, 10.0), score=0.4),
        ],
        gt_boxes=[(10.0, 10.0, 20.0, 20.0)],
    )
    fn_eval = metrics.add_image(
        predictions=[],
        gt_boxes=[(5.0, 5.0, 10.0, 10.0)],
    )
    fp_eval = metrics.add_image(
        predictions=[RainBoxDetection(bbox=(0.0, 0.0, 10.0, 10.0), score=0.55)],
        gt_boxes=[],
    )
    tn_eval = metrics.add_image(predictions=[], gt_boxes=[])

    assert tp_eval.passed is True
    assert tp_eval.max_score == pytest.approx(0.9)
    assert tp_eval.box_mean_iou == pytest.approx(1.0)
    assert fn_eval.missed is True
    assert fn_eval.passed is False
    assert fp_eval.false_positive is True
    assert fp_eval.passed is False
    assert tn_eval.passed is True

    evals = metrics.as_evaluations()
    assert evals["rain_classification"]["labels"] == ["rain", "no_rain"]
    assert evals["rain_classification"]["confusion_matrix"] == [[1, 1], [1, 1]]

    summary = {item["key"]: item for item in metrics.as_summary_metrics()}
    assert summary["precision"]["value"] == pytest.approx(0.5)
    assert summary["recall"]["value"] == pytest.approx(0.5)
    assert summary["f1"]["value"] == pytest.approx(0.5)
    assert summary["fp_rate"]["value"] == pytest.approx(0.5)
    assert summary["box_recall"]["value"] == pytest.approx(0.5)
    assert summary["box_precision"]["value"] == pytest.approx(1 / 3)
    assert summary["box_mean_iou"]["value"] == pytest.approx(1.0)


def test_rain_metrics_handles_all_negative_split_without_zero_division() -> None:
    metrics = RainDropMetrics()
    metrics.add_image(predictions=[], gt_boxes=[])
    metrics.add_image(
        predictions=[RainBoxDetection(bbox=(0.0, 0.0, 10.0, 10.0), score=0.7)],
        gt_boxes=[],
    )

    summary = {item["key"]: item for item in metrics.as_summary_metrics()}
    assert summary["precision"]["value"] == pytest.approx(0.0)
    assert summary["recall"]["value"] is None
    assert summary["f1"]["value"] is None
    assert summary["fp_rate"]["value"] == pytest.approx(0.5)
    assert summary["box_recall"]["value"] is None
    assert summary["box_mean_iou"]["value"] is None


def test_box_iou_and_greedy_matching() -> None:
    assert compute_box_iou(
        (0.0, 0.0, 10.0, 10.0), (0.0, 0.0, 10.0, 10.0)
    ) == pytest.approx(1.0)
    assert compute_box_iou(
        (0.0, 0.0, 10.0, 10.0), (20.0, 20.0, 10.0, 10.0)
    ) == pytest.approx(0.0)
    matched = match_boxes_by_iou(
        pred_boxes=[(0.0, 0.0, 10.0, 10.0), (1.0, 0.0, 10.0, 10.0)],
        gt_boxes=[(0.0, 0.0, 10.0, 10.0)],
        iou_threshold=0.5,
    )
    assert len(matched) == 1
    assert matched[0] == pytest.approx(1.0)


def test_benchmark_cli_accepts_empty_confidence_threshold() -> None:
    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "--source",
            "/tmp/src",
            "--destination",
            "/tmp/dst",
            "--version",
            "v001",
            "--split",
            "val",
            "--run-id",
            "run-1",
            "--device",
            "cpu",
            "--artifact",
            "sara-rain-drop-detector:v0.0.3",
            "--confidence-threshold",
            "",
        ]
    )
    assert args.confidence_threshold is None


def test_benchmark_main_writes_schema_v3_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    model_path = tmp_path / "rain_detection_model.pt"
    model_path.write_bytes(b"weights")

    _write_dataset(
        src,
        images=[
            {"id": 1, "file_name": "rain.jpeg", "width": 40, "height": 30},
            {"id": 2, "file_name": "clear.jpeg", "width": 40, "height": 30},
        ],
        annotations=[
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [4.0, 5.0, 10.0, 10.0],
                "area": 100.0,
                "segmentation": [],
                "iscrowd": 0,
            }
        ],
    )

    fake_model = MagicMock()
    fake_boxes = SimpleNamespace(
        xyxy=np.array([[4.0, 5.0, 14.0, 15.0]], dtype=np.float32),
        conf=np.array([0.88], dtype=np.float32),
    )
    fake_model.predict.side_effect = [
        [SimpleNamespace(boxes=fake_boxes)],
        [SimpleNamespace(boxes=None)],
    ]

    monkeypatch.setattr(
        "sys.argv",
        [
            "python",
            "--source",
            str(src),
            "--destination",
            str(dst),
            "--version",
            "v001",
            "--split",
            "val",
            "--run-id",
            "test-run",
            "--device",
            "cpu",
            "--artifact",
            "sara-rain-drop-detector:v0.0.3",
            "--confidence-threshold",
            "0.35",
            "--model-path",
            str(model_path),
        ],
    )

    with patch(
        "sara_rain_drop_detector.benchmark.__main__.load_rain_model",
        return_value=fake_model,
    ):
        main()

    out = dst / "v001" / "benchmark-results" / "val" / "test-run"
    run_doc = json.loads((out / "run.json").read_text(encoding="utf-8"))
    metrics_doc = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    per_image_doc = json.loads(
        (out / "review" / "per_image.json").read_text(encoding="utf-8")
    )
    preds_doc = json.loads(
        (out / "review" / "instances_predictions.json").read_text(encoding="utf-8")
    )

    assert run_doc["schema_version"] == SCHEMA_VERSION
    assert run_doc["protocol"] == PROTOCOL
    assert run_doc["dataset"]["id"] == "rain-drop-detection"
    assert run_doc["configuration"] == {
        "model": "rain_detection_model.pt",
        "confidence_threshold": 0.35,
    }
    assert metrics_doc["schema_version"] == SCHEMA_VERSION
    assert metrics_doc["evaluations"]["rain_classification"]["confusion_matrix"] == [
        [1, 0],
        [0, 1],
    ]
    summary_by_key = {m["key"]: m for m in metrics_doc["summary_metrics"]}
    assert summary_by_key["precision"]["value"] == pytest.approx(1.0)
    assert summary_by_key["recall"]["value"] == pytest.approx(1.0)
    assert summary_by_key["fp_rate"]["value"] == pytest.approx(0.0)
    assert len(per_image_doc["columns"]) == 6
    assert len(per_image_doc["items"]) == 2
    assert all(item["passed"] for item in per_image_doc["items"])
    assert len(preds_doc["annotations"]) == 1
    assert preds_doc["annotations"][0]["bbox"] == [4.0, 5.0, 10.0, 10.0]
    assert preds_doc["annotations"][0]["score"] == pytest.approx(0.88)
