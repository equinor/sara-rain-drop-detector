from dataclasses import dataclass
from typing import Any

from sara_rain_drop_detector.inference import RainBoxDetection

PROTOCOL = "rain-drop-detection@v001"
BOX_IOU_THRESHOLD = 0.50

PER_IMAGE_COLUMNS: list[dict[str, str]] = [
    {"key": "gt_rain", "label": "GT Rain", "format": "boolean"},
    {"key": "pred_rain", "label": "Pred Rain", "format": "boolean"},
    {"key": "gt_boxes", "label": "GT Boxes", "format": "integer"},
    {"key": "pred_boxes", "label": "Pred Boxes", "format": "integer"},
    {"key": "max_score", "label": "Max Score", "format": "percent"},
    {"key": "box_mean_iou", "label": "Box IoU", "format": "number"},
]

BBox = tuple[float, float, float, float]


@dataclass(frozen=True)
class ImageRainEvaluation:
    gt_rain: bool
    pred_rain: bool
    passed: bool
    missed: bool
    false_positive: bool
    gt_boxes: int
    pred_boxes: int
    matched_boxes: int
    max_score: float | None
    box_mean_iou: float | None


@dataclass
class RainDropMetrics:
    true_positives: int = 0
    false_negatives: int = 0
    false_positives: int = 0
    true_negatives: int = 0
    total_gt_boxes: int = 0
    total_pred_boxes: int = 0
    matched_boxes: int = 0
    matched_box_iou_sum: float = 0.0

    def add_image(
        self,
        predictions: list[RainBoxDetection],
        gt_boxes: list[BBox],
    ) -> ImageRainEvaluation:
        gt_rain = len(gt_boxes) > 0
        pred_rain = len(predictions) > 0

        if gt_rain and pred_rain:
            self.true_positives += 1
        elif gt_rain and not pred_rain:
            self.false_negatives += 1
        elif not gt_rain and pred_rain:
            self.false_positives += 1
        else:
            self.true_negatives += 1

        missed = gt_rain and not pred_rain
        false_positive = not gt_rain and pred_rain
        passed = not missed and not false_positive

        matched_ious = match_boxes_by_iou(
            [p.bbox for p in predictions],
            gt_boxes,
            BOX_IOU_THRESHOLD,
        )
        matched_count = len(matched_ious)
        self.total_gt_boxes += len(gt_boxes)
        self.total_pred_boxes += len(predictions)
        self.matched_boxes += matched_count
        self.matched_box_iou_sum += sum(matched_ious)

        scores = [p.score for p in predictions if p.score is not None]
        max_score = max(scores) if scores else None
        box_mean_iou = sum(matched_ious) / matched_count if matched_count > 0 else None

        return ImageRainEvaluation(
            gt_rain=gt_rain,
            pred_rain=pred_rain,
            passed=passed,
            missed=missed,
            false_positive=false_positive,
            gt_boxes=len(gt_boxes),
            pred_boxes=len(predictions),
            matched_boxes=matched_count,
            max_score=max_score,
            box_mean_iou=box_mean_iou,
        )

    def as_evaluations(self) -> dict[str, dict[str, Any]]:
        return {
            "rain_classification": {
                "labels": ["rain", "no_rain"],
                "confusion_matrix": [
                    [self.true_positives, self.false_negatives],
                    [self.false_positives, self.true_negatives],
                ],
                "matched_count": self.true_positives,
            }
        }

    def as_summary_metrics(self) -> list[dict[str, Any]]:
        tp = self.true_positives
        fn = self.false_negatives
        fp = self.false_positives
        tn = self.true_negatives

        precision = safe_ratio(tp, tp + fp)
        recall = safe_ratio(tp, tp + fn)
        f1 = compute_f1(precision, recall)
        fp_rate = safe_ratio(fp, fp + tn)
        box_recall = safe_ratio(self.matched_boxes, self.total_gt_boxes)
        box_precision = safe_ratio(self.matched_boxes, self.total_pred_boxes)
        box_mean_iou = safe_ratio(self.matched_box_iou_sum, self.matched_boxes)

        return [
            {
                "key": "precision",
                "label": "Precision",
                "value": precision,
                "format": "percent",
                "subtitle": f"TP {tp} / FP {fp}",
                "primary": True,
            },
            {
                "key": "recall",
                "label": "Recall",
                "value": recall,
                "format": "percent",
                "subtitle": f"TP {tp} / FN {fn}",
                "primary": True,
            },
            {
                "key": "f1",
                "label": "F1",
                "value": f1,
                "format": "percent",
                "primary": True,
            },
            {
                "key": "fp_rate",
                "label": "FP Rate",
                "value": fp_rate,
                "format": "percent",
                "subtitle": f"FP {fp} / TN {tn}",
                "primary": True,
            },
            {
                "key": "box_recall",
                "label": "Box Recall",
                "value": box_recall,
                "format": "percent",
                "subtitle": f"{self.matched_boxes} / {self.total_gt_boxes} boxes",
                "primary": False,
            },
            {
                "key": "box_precision",
                "label": "Box Precision",
                "value": box_precision,
                "format": "percent",
                "subtitle": f"{self.matched_boxes} / {self.total_pred_boxes} preds",
                "primary": False,
            },
            {
                "key": "box_mean_iou",
                "label": "Box Mean IoU",
                "value": box_mean_iou,
                "format": "number",
                "subtitle": f"IoU >= {BOX_IOU_THRESHOLD:.2f}",
                "primary": False,
            },
        ]


def compute_box_iou(box_a: BBox, box_b: BBox) -> float:
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    inter_x1 = max(ax, bx)
    inter_y1 = max(ay, by)
    inter_x2 = min(ax + aw, bx + bw)
    inter_y2 = min(ay + ah, by + bh)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h
    union = aw * ah + bw * bh - intersection
    return intersection / union if union > 0.0 else 0.0


def match_boxes_by_iou(
    pred_boxes: list[BBox],
    gt_boxes: list[BBox],
    iou_threshold: float,
) -> list[float]:
    if not pred_boxes or not gt_boxes:
        return []
    candidates: list[tuple[float, int, int]] = []
    for pred_idx, pred_box in enumerate(pred_boxes):
        for gt_idx, gt_box in enumerate(gt_boxes):
            iou = compute_box_iou(pred_box, gt_box)
            if iou >= iou_threshold:
                candidates.append((iou, pred_idx, gt_idx))
    candidates.sort(key=lambda item: item[0], reverse=True)

    matched_preds: set[int] = set()
    matched_gts: set[int] = set()
    matched_ious: list[float] = []
    for iou, pred_idx, gt_idx in candidates:
        if pred_idx in matched_preds or gt_idx in matched_gts:
            continue
        matched_preds.add(pred_idx)
        matched_gts.add(gt_idx)
        matched_ious.append(iou)
    return matched_ious


def box_to_coco_annotation(
    annotation_id: int,
    image_id: int,
    category_id: int,
    detection: RainBoxDetection,
) -> dict[str, Any]:
    x, y, width, height = detection.bbox
    annotation: dict[str, Any] = {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": category_id,
        "bbox": [x, y, width, height],
        "area": width * height,
        "segmentation": [],
        "iscrowd": 0,
    }
    if detection.score is not None:
        annotation["score"] = detection.score
    return annotation


def safe_ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator > 0 else None


def compute_f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None:
        return None
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)
