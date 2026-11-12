"""Utilities for flip TTA and conservative instance-level ensembling."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
from pycocotools import mask as mask_utils

from filament_seg.evaluation.pq import coco_rle, encode_binary_mask


Instance = Dict[str, Any]


def restore_flipped_mask(mask: np.ndarray, view: str) -> np.ndarray:
    """Restore a binary mask predicted from one flipped image view."""
    array = np.asarray(mask).astype(bool)
    if array.ndim != 2:
        raise ValueError("Expected a 2D mask, got shape {}".format(array.shape))
    if view == "original":
        restored = array
    elif view == "hflip":
        restored = np.fliplr(array)
    elif view == "vflip":
        restored = np.flipud(array)
    else:
        raise ValueError("Unsupported TTA view: {}".format(view))
    return np.ascontiguousarray(restored)


def mask_box(mask: np.ndarray) -> List[float]:
    """Return an XYXY box around one non-empty binary mask."""
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise ValueError("Cannot compute a box for an empty mask")
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]


def instance_from_mask(mask: np.ndarray, score: float, view: str = "original") -> Instance:
    """Build the repository's compact prediction instance representation."""
    binary = np.asarray(mask).astype(bool)
    area = int(binary.sum())
    if area == 0:
        raise ValueError("Cannot encode an empty prediction mask")
    return {
        "score": float(score),
        "box": mask_box(binary),
        "area": area,
        "rle": encode_binary_mask(binary),
        "view": view,
    }


def _filtered_instances(
    instances: Sequence[Mapping[str, Any]], score_threshold: float, min_area: int
) -> List[Instance]:
    filtered: List[Instance] = []
    for item in instances:
        if float(item.get("score", 0.0)) < score_threshold:
            continue
        rle = coco_rle(item["rle"])
        area = int(mask_utils.area(rle))
        if area < min_area or area == 0:
            continue
        copied = dict(item)
        copied["area"] = area
        filtered.append(copied)
    filtered.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
    return filtered


def _maximum_iou(candidate: Mapping[str, Any], accepted: Sequence[Mapping[str, Any]]) -> float:
    if not accepted:
        return 0.0
    candidate_rle = coco_rle(candidate["rle"])
    accepted_rles = [coco_rle(item["rle"]) for item in accepted]
    values = np.asarray(mask_utils.iou([candidate_rle], accepted_rles, [0] * len(accepted_rles)))
    return float(values.max()) if values.size else 0.0


def _maximum_iou_index(
    candidate: Mapping[str, Any], accepted: Sequence[Mapping[str, Any]]
) -> tuple:
    if not accepted:
        return 0.0, -1
    candidate_rle = coco_rle(candidate["rle"])
    accepted_rles = [coco_rle(item["rle"]) for item in accepted]
    values = np.asarray(mask_utils.iou([candidate_rle], accepted_rles, [0] * len(accepted_rles)))
    if not values.size:
        return 0.0, -1
    flat_index = int(values.argmax())
    return float(values.reshape(-1)[flat_index]), flat_index


def deduplicate_instances(
    instances: Sequence[Mapping[str, Any]],
    duplicate_iou: float,
    score_threshold: float = 0.0,
    min_area: int = 0,
) -> List[Instance]:
    """Greedily retain the highest-confidence mask from each overlap cluster."""
    if not 0.0 <= duplicate_iou <= 1.0:
        raise ValueError("duplicate_iou must be between 0 and 1")
    accepted: List[Instance] = []
    for candidate in _filtered_instances(instances, score_threshold, min_area):
        maximum_iou, match_index = _maximum_iou_index(candidate, accepted)
        if maximum_iou >= duplicate_iou:
            view = candidate.get("view")
            if view is not None:
                support_views = set(accepted[match_index].get("support_views", []))
                support_views.add(str(view))
                accepted[match_index]["support_views"] = sorted(support_views)
                accepted[match_index]["support_count"] = len(support_views)
            continue
        copied = dict(candidate)
        view = copied.get("view")
        support_views = [str(view)] if view is not None else []
        copied["support_views"] = support_views
        copied["support_count"] = max(1, len(support_views))
        accepted.append(copied)
    return accepted


def fuse_view_instances(
    instances: Sequence[Mapping[str, Any]],
    duplicate_iou: float,
    fusion_mode: str,
    score_threshold: float = 0.0,
    min_area: int = 0,
) -> List[Instance]:
    """Cluster detections from distinct TTA views and fuse their masks.

    A cluster accepts at most one detection from each view.  Singleton clusters
    are kept unchanged, and an empty fused mask falls back to the highest-score
    member.  ``best`` intentionally delegates to the established deduplication
    behavior so existing experiments remain reproducible.
    """
    if fusion_mode == "best":
        return deduplicate_instances(instances, duplicate_iou, score_threshold, min_area)
    if fusion_mode not in {"intersection", "majority", "eroded_majority"}:
        raise ValueError("Unsupported fusion mode: {}".format(fusion_mode))
    if not 0.0 <= duplicate_iou <= 1.0:
        raise ValueError("duplicate_iou must be between 0 and 1")

    clusters: List[List[Instance]] = []
    representatives: List[Instance] = []
    for candidate in _filtered_instances(instances, score_threshold, min_area):
        candidate_view = str(candidate.get("view", "unknown"))
        eligible = [
            index
            for index, cluster in enumerate(clusters)
            if candidate_view not in {str(item.get("view", "unknown")) for item in cluster}
        ]
        if eligible:
            subset = [representatives[index] for index in eligible]
            maximum_iou, subset_index = _maximum_iou_index(candidate, subset)
            match_index = eligible[subset_index] if subset_index >= 0 else -1
        else:
            maximum_iou, match_index = 0.0, -1
        if match_index >= 0 and maximum_iou >= duplicate_iou:
            clusters[match_index].append(dict(candidate))
        else:
            copied = dict(candidate)
            clusters.append([copied])
            representatives.append(copied)

    fused: List[Instance] = []
    for cluster, representative in zip(clusters, representatives):
        masks = [mask_utils.decode(coco_rle(item["rle"])).astype(bool) for item in cluster]
        if len(masks) == 1:
            binary = masks[0]
        elif fusion_mode == "intersection":
            binary = np.logical_and.reduce(masks)
        else:
            votes = np.sum(np.stack(masks, axis=0), axis=0)
            binary = votes >= (len(masks) // 2 + 1)
            if fusion_mode == "eroded_majority":
                import cv2

                binary = cv2.erode(binary.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1).astype(bool)
        if not binary.any():
            binary = masks[0]
        item = instance_from_mask(
            binary,
            float(representative.get("score", 0.0)),
            view=str(representative.get("view", "original")),
        )
        support_views = sorted({str(member.get("view", "unknown")) for member in cluster})
        item["support_views"] = support_views
        item["support_count"] = len(support_views)
        item["fusion_mode"] = fusion_mode
        fused.append(item)
    return fused


def merge_primary_secondary(
    primary: Sequence[Mapping[str, Any]],
    secondary: Sequence[Mapping[str, Any]],
    primary_score_threshold: float,
    secondary_score_threshold: float,
    primary_min_area: int,
    secondary_min_area: int,
    duplicate_iou: float,
) -> List[Instance]:
    """Keep the primary model and add only non-duplicate secondary instances."""
    if not 0.0 <= duplicate_iou <= 1.0:
        raise ValueError("duplicate_iou must be between 0 and 1")
    accepted = _filtered_instances(primary, primary_score_threshold, primary_min_area)
    for item in accepted:
        original_score = float(item.get("score", 0.0))
        item["original_score"] = original_score
        item["score"] = 1.0 + original_score
        item["source"] = "primary"
    for candidate in _filtered_instances(secondary, secondary_score_threshold, secondary_min_area):
        if _maximum_iou(candidate, accepted) >= duplicate_iou:
            continue
        copied = dict(candidate)
        copied["original_score"] = float(copied.get("score", 0.0))
        copied["source"] = "secondary"
        accepted.append(copied)
    return accepted
