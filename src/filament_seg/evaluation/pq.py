"""Panoptic Quality evaluation for competition instance masks."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
from pycocotools import mask as mask_utils

from filament_seg.data.coco import annotations_by_image, images_by_id, iter_polygons
from filament_seg.data.folds import FoldRow


RLE = Dict[str, Any]


def json_rle(rle: Mapping[str, Any]) -> RLE:
    """Convert a pycocotools RLE to a JSON-serializable dictionary."""
    counts = rle["counts"]
    if isinstance(counts, bytes):
        counts = counts.decode("ascii")
    return {"size": [int(v) for v in rle["size"]], "counts": counts}


def coco_rle(rle: Mapping[str, Any]) -> RLE:
    """Convert a JSON RLE back to the representation expected by pycocotools."""
    counts = rle["counts"]
    if isinstance(counts, str):
        counts = counts.encode("ascii")
    return {"size": [int(v) for v in rle["size"]], "counts": counts}


def encode_binary_mask(mask: np.ndarray) -> RLE:
    """Encode one 2D binary mask using compressed COCO RLE."""
    if mask.ndim != 2:
        raise ValueError("Expected a 2D mask, got shape {}".format(mask.shape))
    encoded = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    return json_rle(encoded)


def polygons_to_rle(polygons: Sequence[Sequence[float]], height: int, width: int) -> RLE:
    """Convert one instance's polygon list to compressed COCO RLE."""
    if not polygons:
        raise ValueError("Cannot encode an empty polygon list")
    uncompressed = mask_utils.frPyObjects(polygons, int(height), int(width))
    merged = mask_utils.merge(uncompressed)
    return coco_rle(merged)


def read_prediction_jsonl(path: Path) -> Dict[str, Dict[str, Any]]:
    """Load one prediction JSON object per image record."""
    predictions: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            image_id = str(record["image_id"])
            if image_id in predictions:
                raise ValueError("Duplicate prediction for {} at line {}".format(image_id, line_number))
            predictions[image_id] = record
    return predictions


def filter_and_resolve_instances(
    instances: Sequence[Mapping[str, Any]],
    score_threshold: float,
    mask_threshold: float = 0.5,
    min_area: int = 0,
    max_det: Optional[int] = None,
    resolve_overlaps: bool = True,
    small_score_threshold: Optional[float] = None,
    small_min_area: int = 0,
    small_min_aspect_ratio: float = 1.0,
    small_max_fill_ratio: float = 1.0,
) -> List[RLE]:
    """Filter instances and optionally rescue confident thin small masks.

    The default arguments preserve the original score/min-area behavior.  The
    small-instance branch is enabled only when ``small_score_threshold`` is not
    ``None``.  A rescued mask must be below ``min_area`` and satisfy all small
    area, bounding-box aspect-ratio, and fill-ratio constraints.
    """
    del mask_threshold  # Predictions are already binary; retained for a stable evaluator interface.
    if min_area < 0 or small_min_area < 0:
        raise ValueError("Area thresholds must be non-negative")
    if small_min_aspect_ratio < 1.0:
        raise ValueError("small_min_aspect_ratio must be at least 1")
    if not 0.0 <= small_max_fill_ratio <= 1.0:
        raise ValueError("small_max_fill_ratio must be between 0 and 1")

    filtered: List[Tuple[Mapping[str, Any], RLE, bool]] = []
    for item in instances:
        score = float(item.get("score", 0.0))
        rle = coco_rle(item["rle"])
        cached_area = item.get("_shape_area")
        raw_area = float(cached_area) if cached_area is not None else float(mask_utils.area(rle))
        rescued = False
        if small_score_threshold is not None and small_min_area <= raw_area < min_area:
            cached_bbox = item.get("_shape_bbox")
            bbox = cached_bbox if cached_bbox is not None else mask_utils.toBbox(rle)
            x, y, width, height = (float(value) for value in bbox)
            del x, y
            short_side = min(width, height)
            long_side = max(width, height)
            aspect_ratio = long_side / short_side if short_side > 0 else 0.0
            box_area = width * height
            fill_ratio = raw_area / box_area if box_area > 0 else 1.0
            rescued = (
                score >= small_score_threshold
                and raw_area > 0
                and aspect_ratio >= small_min_aspect_ratio
                and fill_ratio <= small_max_fill_ratio
            )
        if score >= score_threshold or rescued:
            filtered.append((item, rle, rescued))
    filtered.sort(key=lambda value: float(value[0].get("score", 0.0)), reverse=True)
    if max_det is not None:
        filtered = filtered[:max_det]
    if not filtered:
        return []

    if not resolve_overlaps:
        return [
            rle
            for _, rle, rescued in filtered
            if int(mask_utils.area(rle)) >= (small_min_area if rescued else min_area)
        ]

    height, width = (int(v) for v in filtered[0][1]["size"])
    occupied = np.zeros((height, width), dtype=bool)
    resolved: List[RLE] = []
    for _, rle, rescued in filtered:
        if [int(v) for v in rle["size"]] != [height, width]:
            raise ValueError("All predictions for one image must have the same size")
        decoded = mask_utils.decode(rle).astype(bool)
        cleaned = np.logical_and(decoded, np.logical_not(occupied))
        area = int(cleaned.sum())
        required_area = small_min_area if rescued else min_area
        if area < required_area or area == 0:
            continue
        occupied |= cleaned
        resolved.append(coco_rle(mask_utils.encode(np.asfortranarray(cleaned.astype(np.uint8)))))
    return resolved


def resolve_instances_by_score_threshold(
    instances: Sequence[Mapping[str, Any]],
    score_thresholds: Sequence[float],
    min_area: int = 0,
    max_det: Optional[int] = None,
) -> Dict[float, List[RLE]]:
    """Resolve masks once while incrementally lowering the score threshold.

    For a fixed minimum area, lower-scored candidates can never alter an
    already accepted higher-scored mask. This makes every threshold result a
    snapshot of one descending-score pass and preserves the exhaustive
    ``filter_and_resolve_instances`` semantics.
    """
    if min_area < 0:
        raise ValueError("Area thresholds must be non-negative")
    thresholds = sorted({float(value) for value in score_thresholds}, reverse=True)
    if not thresholds:
        raise ValueError("score_thresholds must be non-empty")

    ordered = sorted(
        instances,
        key=lambda item: float(item.get("score", 0.0)),
        reverse=True,
    )
    if max_det is not None:
        ordered = ordered[:max_det]
    ordered = [item for item in ordered if float(item.get("score", 0.0)) >= thresholds[-1]]

    if ordered:
        first = coco_rle(ordered[0]["rle"])
        height, width = (int(value) for value in first["size"])
        occupied = np.zeros((height, width), dtype=bool)
    else:
        height = width = 0
        occupied = np.zeros((0, 0), dtype=bool)

    accepted: List[RLE] = []
    snapshots: Dict[float, List[RLE]] = {}
    cursor = 0
    for threshold in thresholds:
        while cursor < len(ordered) and float(ordered[cursor].get("score", 0.0)) >= threshold:
            rle = coco_rle(ordered[cursor]["rle"])
            if [int(value) for value in rle["size"]] != [height, width]:
                raise ValueError("All predictions for one image must have the same size")
            decoded = mask_utils.decode(rle).astype(bool)
            cleaned = np.logical_and(decoded, np.logical_not(occupied))
            area = int(cleaned.sum())
            if area >= min_area and area > 0:
                occupied |= cleaned
                accepted.append(
                    coco_rle(mask_utils.encode(np.asfortranarray(cleaned.astype(np.uint8))))
                )
            cursor += 1
        snapshots[threshold] = list(accepted)
    return snapshots


def _match_instances(iou: np.ndarray, threshold: float) -> List[Tuple[int, int, float]]:
    candidates: List[Tuple[float, int, int]] = []
    for pred_index in range(iou.shape[0]):
        for gt_index in range(iou.shape[1]):
            value = float(iou[pred_index, gt_index])
            if value > threshold:
                candidates.append((value, pred_index, gt_index))
    candidates.sort(reverse=True)
    used_predictions = set()
    used_ground_truth = set()
    matches: List[Tuple[int, int, float]] = []
    for value, pred_index, gt_index in candidates:
        if pred_index in used_predictions or gt_index in used_ground_truth:
            continue
        used_predictions.add(pred_index)
        used_ground_truth.add(gt_index)
        matches.append((pred_index, gt_index, value))
    return matches


def _score_iou_matrix(
    iou: np.ndarray,
    pred_areas: np.ndarray,
    gt_areas: np.ndarray,
    match_iou: float,
    relation_threshold: float,
) -> Dict[str, Any]:
    """Score a cached prediction-by-ground-truth IoU matrix."""
    matches = _match_instances(iou, match_iou)
    true_positive = len(matches)
    false_positive = iou.shape[0] - true_positive
    false_negative = iou.shape[1] - true_positive
    sum_iou = sum(value for _, _, value in matches)
    denominator = true_positive + 0.5 * false_positive + 0.5 * false_negative
    pq = sum_iou / denominator if denominator else 1.0
    sq = sum_iou / true_positive if true_positive else (1.0 if denominator == 0 else 0.0)
    rq = true_positive / denominator if denominator else 1.0

    fragmentation = 0
    over_merging = 0
    if iou.size:
        intersections = np.zeros_like(iou)
        positive = iou > 0
        intersections[positive] = (
            iou[positive]
            * (pred_areas[:, None] + gt_areas[None, :])[positive]
            / (1.0 + iou[positive])
        )
        for gt_index, area in enumerate(gt_areas):
            related = np.sum(intersections[:, gt_index] / max(area, 1.0) >= relation_threshold)
            fragmentation += int(related > 1)
        for pred_index, area in enumerate(pred_areas):
            related = np.sum(intersections[pred_index, :] / max(area, 1.0) >= relation_threshold)
            over_merging += int(related > 1)
    return {
        "pq": pq,
        "sq": sq,
        "rq": rq,
        "tp": true_positive,
        "fp": false_positive,
        "fn": false_negative,
        "sum_iou": sum_iou,
        "num_predictions": iou.shape[0],
        "num_ground_truth": iou.shape[1],
        "fragmentation": fragmentation,
        "over_merging": over_merging,
    }


def _score_iou_matrix_official(iou: np.ndarray, match_iou: float) -> Dict[str, Any]:
    """Apply the official notebook counting rules to a cached P-by-G IoU matrix."""
    official_iou = iou.T
    if official_iou.shape[0] == 0:
        true_positive = 0
        false_positive = official_iou.shape[1]
        false_negative = 0
        sum_iou = 0.0
    elif official_iou.shape[1] == 0:
        true_positive = 0
        false_positive = 0
        false_negative = official_iou.shape[0]
        sum_iou = 0.0
    else:
        hits = official_iou > match_iou
        true_positive = int(hits.sum())
        false_positive = int((hits.sum(axis=0) == 0).sum())
        false_negative = int((hits.sum(axis=1) == 0).sum())
        sum_iou = float(official_iou[hits].sum())
    denominator = true_positive + 0.5 * false_positive + 0.5 * false_negative
    return {
        "pq": sum_iou / denominator if denominator else 0.0,
        "sq": sum_iou / true_positive if true_positive else 0.0,
        "rq": true_positive / denominator if denominator else 0.0,
        "tp": true_positive,
        "fp": false_positive,
        "fn": false_negative,
        "sum_iou": sum_iou,
        "num_predictions": official_iou.shape[1],
        "num_ground_truth": official_iou.shape[0],
    }


def score_image(
    prediction_rles: Sequence[RLE],
    ground_truth_rles: Sequence[RLE],
    match_iou: float = 0.5,
    relation_threshold: float = 0.1,
) -> Dict[str, Any]:
    """Score one annotation record and diagnose split/merge relationships."""
    predictions = [coco_rle(rle) for rle in prediction_rles]
    ground_truth = [coco_rle(rle) for rle in ground_truth_rles]
    if predictions and ground_truth:
        iou = np.asarray(mask_utils.iou(predictions, ground_truth, [0] * len(ground_truth)))
    else:
        iou = np.zeros((len(predictions), len(ground_truth)), dtype=np.float64)

    pred_areas = np.asarray(mask_utils.area(predictions), dtype=np.float64).reshape(-1)
    gt_areas = np.asarray(mask_utils.area(ground_truth), dtype=np.float64).reshape(-1)
    return _score_iou_matrix(iou, pred_areas, gt_areas, match_iou, relation_threshold)


def score_image_official(
    prediction_rles: Sequence[RLE],
    ground_truth_rles: Sequence[RLE],
    match_iou: float = 0.5,
) -> Dict[str, Any]:
    """Score one record exactly as the official Self Evaluation notebook.

    The official implementation counts every pair whose IoU is strictly above
    the threshold as a TP.  A prediction/GT is an FP/FN only when its entire
    IoU column/row has no qualifying hit.  It does not run an explicit greedy
    or Hungarian one-to-one assignment.
    """
    predictions = [coco_rle(rle) for rle in prediction_rles]
    ground_truth = [coco_rle(rle) for rle in ground_truth_rles]
    if predictions and ground_truth:
        # pycocotools returns prediction x ground-truth, while the notebook
        # uses ground-truth x prediction.  Transpose to mirror it literally.
        iou = np.asarray(
            mask_utils.iou(predictions, ground_truth, [0] * len(ground_truth)),
            dtype=np.float64,
        ).T
    else:
        iou = np.zeros((len(ground_truth), len(predictions)), dtype=np.float64)

    return _score_iou_matrix_official(iou.T, match_iou)


def build_ground_truth(data: Mapping[str, Any]) -> Dict[str, List[RLE]]:
    """Build ground-truth RLEs for every annotation-version image id."""
    images = images_by_id(data)
    anns_by_image = annotations_by_image(data)
    result: Dict[str, List[RLE]] = {}
    for image_id, image in images.items():
        instances: List[RLE] = []
        for annotation in anns_by_image.get(image_id, []):
            polygons = list(iter_polygons(annotation))
            if len(polygons) != 1:
                raise ValueError(
                    "Annotation {} has {} polygons; exactly one is expected".format(
                        annotation.get("id"), len(polygons)
                    )
                )
            instances.append(polygons_to_rle(polygons, int(image["height"]), int(image["width"])))
        result[image_id] = instances
    return result


def evaluate_predictions(
    data: Mapping[str, Any],
    fold_rows: Sequence[FoldRow],
    predictions: Mapping[str, Mapping[str, Any]],
    val_fold: int,
    score_threshold: float,
    min_area: int = 0,
    max_det: Optional[int] = None,
    match_iou: float = 0.5,
    relation_threshold: float = 0.1,
    small_score_threshold: Optional[float] = None,
    small_min_area: int = 0,
    small_min_aspect_ratio: float = 1.0,
    small_max_fill_ratio: float = 1.0,
) -> Dict[str, Any]:
    """Evaluate all records in one validation fold."""
    ground_truth = build_ground_truth(data)
    validation_rows = [row for row in fold_rows if int(row["fold"]) == val_fold]
    per_record: List[Dict[str, Any]] = []

    for row in validation_rows:
        image_id = str(row["image_id"])
        prediction_record = predictions.get(image_id, {"instances": []})
        prediction_rles = filter_and_resolve_instances(
            prediction_record.get("instances", []),
            score_threshold=score_threshold,
            min_area=min_area,
            max_det=max_det,
            resolve_overlaps=True,
            small_score_threshold=small_score_threshold,
            small_min_area=small_min_area,
            small_min_aspect_ratio=small_min_aspect_ratio,
            small_max_fill_ratio=small_max_fill_ratio,
        )
        metrics = score_image(
            prediction_rles,
            ground_truth.get(image_id, []),
            match_iou=match_iou,
            relation_threshold=relation_threshold,
        )
        official = score_image_official(
            prediction_rles,
            ground_truth.get(image_id, []),
            match_iou=match_iou,
        )
        metrics.update(
            {
                "official_pq": official["pq"],
                "official_sq": official["sq"],
                "official_rq": official["rq"],
                "official_tp": official["tp"],
                "official_fp": official["fp"],
                "official_fn": official["fn"],
                "official_sum_iou": official["sum_iou"],
            }
        )
        metrics.update(
            {
                "image_id": image_id,
                "file_name": str(row["file_name"]),
                "date": str(row["date"]),
                "station": str(row["station"]),
                "year": str(row["year"]),
                "num_passes": int(row["num_passes"]),
            }
        )
        per_record.append(metrics)

    totals = CounterLike()
    for metrics in per_record:
        for key in ("tp", "fp", "fn", "sum_iou", "fragmentation", "over_merging"):
            totals[key] += float(metrics[key])
    denominator = totals["tp"] + 0.5 * totals["fp"] + 0.5 * totals["fn"]
    global_metrics = {
        "pq": totals["sum_iou"] / denominator if denominator else 1.0,
        "sq": totals["sum_iou"] / totals["tp"] if totals["tp"] else 0.0,
        "rq": totals["tp"] / denominator if denominator else 1.0,
        "tp": int(totals["tp"]),
        "fp": int(totals["fp"]),
        "fn": int(totals["fn"]),
        "fragmentation": int(totals["fragmentation"]),
        "over_merging": int(totals["over_merging"]),
    }

    official_totals = CounterLike()
    for metrics in per_record:
        for key in ("tp", "fp", "fn", "sum_iou"):
            official_totals[key] += float(metrics["official_" + key])
    official_denominator = (
        official_totals["tp"]
        + 0.5 * official_totals["fp"]
        + 0.5 * official_totals["fn"]
    )
    official_global = {
        "pq": (
            official_totals["sum_iou"] / official_denominator
            if official_denominator
            else 0.0
        ),
        "sq": (
            official_totals["sum_iou"] / official_totals["tp"]
            if official_totals["tp"]
            else 0.0
        ),
        "rq": (
            official_totals["tp"] / official_denominator
            if official_denominator
            else 0.0
        ),
        "tp": int(official_totals["tp"]),
        "fp": int(official_totals["fp"]),
        "fn": int(official_totals["fn"]),
    }

    macro_record = {
        key: float(np.mean([record[key] for record in per_record])) if per_record else 0.0
        for key in ("pq", "sq", "rq")
    }
    by_file: MutableMapping[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in per_record:
        by_file[str(record["file_name"])].append(record)
    file_scores = [float(np.mean([item["pq"] for item in items])) for items in by_file.values()]

    return {
        "settings": {
            "val_fold": val_fold,
            "score_threshold": score_threshold,
            "min_area": min_area,
            "max_det": max_det,
            "match_iou": match_iou,
            "relation_threshold": relation_threshold,
            "small_score_threshold": small_score_threshold,
            "small_min_area": small_min_area,
            "small_min_aspect_ratio": small_min_aspect_ratio,
            "small_max_fill_ratio": small_max_fill_ratio,
        },
        "global": global_metrics,
        "official_global": official_global,
        "macro_record": macro_record,
        "macro_physical_image_pq": float(np.mean(file_scores)) if file_scores else 0.0,
        "records": per_record,
    }


def evaluate_postprocess_thresholds_fast(
    data: Mapping[str, Any],
    fold_rows: Sequence[FoldRow],
    predictions: Mapping[str, Mapping[str, Any]],
    val_fold: int,
    score_thresholds: Sequence[float],
    min_area: int = 0,
    max_det: Optional[int] = None,
    match_iou: float = 0.5,
    relation_threshold: float = 0.1,
) -> List[Dict[str, Any]]:
    """Evaluate every score threshold with one mask-resolution pass per image."""
    thresholds = sorted({float(value) for value in score_thresholds}, reverse=True)
    if not thresholds:
        raise ValueError("score_thresholds must be non-empty")
    ground_truth = build_ground_truth(data)
    validation_rows = [row for row in fold_rows if int(row["fold"]) == val_fold]
    cached_images: Dict[str, Dict[str, Any]] = {}
    for row in validation_rows:
        image_id = str(row["image_id"])
        prediction_record = predictions.get(image_id, {"instances": []})
        snapshots = resolve_instances_by_score_threshold(
            prediction_record.get("instances", []),
            score_thresholds=thresholds,
            min_area=min_area,
            max_det=max_det,
        )
        all_predictions = snapshots[thresholds[-1]]
        ground_truth_rles = ground_truth.get(image_id, [])
        if all_predictions and ground_truth_rles:
            full_iou = np.asarray(
                mask_utils.iou(
                    [coco_rle(item) for item in all_predictions],
                    [coco_rle(item) for item in ground_truth_rles],
                    [0] * len(ground_truth_rles),
                ),
                dtype=np.float64,
            )
        else:
            full_iou = np.zeros(
                (len(all_predictions), len(ground_truth_rles)), dtype=np.float64
            )
        cached_images[image_id] = {
            "counts": {threshold: len(snapshots[threshold]) for threshold in thresholds},
            "iou": full_iou,
            "pred_areas": np.asarray(
                mask_utils.area([coco_rle(item) for item in all_predictions]),
                dtype=np.float64,
            ).reshape(-1),
            "gt_areas": np.asarray(
                mask_utils.area([coco_rle(item) for item in ground_truth_rles]),
                dtype=np.float64,
            ).reshape(-1),
        }

    results: List[Dict[str, Any]] = []
    for threshold in thresholds:
        per_record: List[Dict[str, Any]] = []
        for row in validation_rows:
            image_id = str(row["image_id"])
            cached = cached_images[image_id]
            count = int(cached["counts"][threshold])
            iou = cached["iou"][:count]
            pred_areas = cached["pred_areas"][:count]
            metrics = _score_iou_matrix(
                iou,
                pred_areas,
                cached["gt_areas"],
                match_iou,
                relation_threshold,
            )
            official = _score_iou_matrix_official(iou, match_iou)
            metrics.update(
                {
                    "official_pq": official["pq"],
                    "official_sq": official["sq"],
                    "official_rq": official["rq"],
                    "official_tp": official["tp"],
                    "official_fp": official["fp"],
                    "official_fn": official["fn"],
                    "official_sum_iou": official["sum_iou"],
                    "image_id": image_id,
                    "file_name": str(row["file_name"]),
                    "date": str(row["date"]),
                    "station": str(row["station"]),
                    "year": str(row["year"]),
                    "num_passes": int(row["num_passes"]),
                }
            )
            per_record.append(metrics)

        totals = CounterLike()
        for metrics in per_record:
            for key in ("tp", "fp", "fn", "sum_iou", "fragmentation", "over_merging"):
                totals[key] += float(metrics[key])
        denominator = totals["tp"] + 0.5 * totals["fp"] + 0.5 * totals["fn"]
        global_metrics = {
            "pq": totals["sum_iou"] / denominator if denominator else 1.0,
            "sq": totals["sum_iou"] / totals["tp"] if totals["tp"] else 0.0,
            "rq": totals["tp"] / denominator if denominator else 1.0,
            "tp": int(totals["tp"]),
            "fp": int(totals["fp"]),
            "fn": int(totals["fn"]),
            "fragmentation": int(totals["fragmentation"]),
            "over_merging": int(totals["over_merging"]),
        }
        official_totals = CounterLike()
        for metrics in per_record:
            for key in ("tp", "fp", "fn", "sum_iou"):
                official_totals[key] += float(metrics["official_" + key])
        official_denominator = (
            official_totals["tp"]
            + 0.5 * official_totals["fp"]
            + 0.5 * official_totals["fn"]
        )
        official_global = {
            "pq": official_totals["sum_iou"] / official_denominator if official_denominator else 0.0,
            "sq": official_totals["sum_iou"] / official_totals["tp"] if official_totals["tp"] else 0.0,
            "rq": official_totals["tp"] / official_denominator if official_denominator else 0.0,
            "tp": int(official_totals["tp"]),
            "fp": int(official_totals["fp"]),
            "fn": int(official_totals["fn"]),
        }
        macro_record = {
            key: float(np.mean([record[key] for record in per_record])) if per_record else 0.0
            for key in ("pq", "sq", "rq")
        }
        by_file: MutableMapping[str, List[Dict[str, Any]]] = defaultdict(list)
        for record in per_record:
            by_file[str(record["file_name"])].append(record)
        file_scores = [
            float(np.mean([item["pq"] for item in items])) for items in by_file.values()
        ]
        results.append(
            {
                "settings": {
                    "val_fold": val_fold,
                    "score_threshold": threshold,
                    "min_area": min_area,
                    "max_det": max_det,
                    "match_iou": match_iou,
                    "relation_threshold": relation_threshold,
                    "small_score_threshold": None,
                    "small_min_area": 0,
                    "small_min_aspect_ratio": 1.0,
                    "small_max_fill_ratio": 1.0,
                },
                "global": global_metrics,
                "official_global": official_global,
                "macro_record": macro_record,
                "macro_physical_image_pq": (
                    float(np.mean(file_scores)) if file_scores else 0.0
                ),
                "records": per_record,
            }
        )
    return results


class CounterLike(defaultdict):
    """Tiny float counter without importing another public abstraction."""

    def __init__(self) -> None:
        super().__init__(float)
