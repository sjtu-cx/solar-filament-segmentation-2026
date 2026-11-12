"""Build deterministic instance masks from repeated annotations of one image."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import cv2
import numpy as np
from pycocotools import mask as mask_utils

from filament_seg.evaluation.pq import polygons_to_rle


Mask = np.ndarray


def mean_signed_distance_mask(masks: Sequence[Mask]) -> Mask:
    """Fuse masks at the zero level set of their mean signed distance.

    The result stays between pixelwise intersection and union.  Pixels that
    fall exactly on the averaged zero level set are retained so a one-pixel
    annotator shift does not systematically erode a thin instance.
    """
    if not masks:
        raise ValueError("At least one mask is required")
    binary_masks = [np.asarray(mask).astype(bool) for mask in masks]
    shape = binary_masks[0].shape
    if any(mask.shape != shape for mask in binary_masks):
        raise ValueError("All masks must have the same shape")
    if len(binary_masks) == 1:
        return np.ascontiguousarray(binary_masks[0])

    intersection = np.logical_and.reduce(binary_masks)
    union = np.logical_or.reduce(binary_masks)
    if not union.any():
        return np.zeros(shape, dtype=bool)

    # A filament occupies only a small part of a 2048x2048 image.  Distance
    # transforms over its padded union box are equivalent for the zero-level
    # result and avoid repeatedly processing the full solar disk per instance.
    ys, xs = np.nonzero(union)
    top = max(int(ys.min()) - 1, 0)
    bottom = min(int(ys.max()) + 2, shape[0])
    left = max(int(xs.min()) - 1, 0)
    right = min(int(xs.max()) + 2, shape[1])
    crop = (slice(top, bottom), slice(left, right))

    signed_distances = []
    for mask in binary_masks:
        foreground = mask[crop].astype(np.uint8)
        background = np.logical_not(mask[crop]).astype(np.uint8)
        inside = cv2.distanceTransform(foreground, cv2.DIST_L2, 5)
        outside = cv2.distanceTransform(background, cv2.DIST_L2, 5)
        signed_distances.append(inside - outside)
    mean_distance = np.mean(np.stack(signed_distances, axis=0), axis=0)
    fused_crop = mean_distance >= 0.0

    # These bounds also make the intended behavior explicit if OpenCV's
    # distance-transform implementation changes between environments.
    fused_crop = np.logical_or(fused_crop, intersection[crop])
    fused_crop = np.logical_and(fused_crop, union[crop])
    fused = np.zeros(shape, dtype=bool)
    fused[crop] = fused_crop
    return np.ascontiguousarray(fused.astype(bool))


def annotation_masks(
    annotations: Sequence[Mapping[str, Any]], height: int, width: int
) -> List[Mask]:
    """Rasterize one annotation record's polygon instances."""
    masks: List[Mask] = []
    for annotation in annotations:
        segmentation = annotation.get("segmentation", [])
        polygons = [item for item in segmentation if isinstance(item, list) and item]
        if not polygons:
            continue
        masks.append(mask_utils.decode(polygons_to_rle(polygons, height, width)).astype(bool))
    return masks


def _union_find(size: int) -> Tuple[List[int], Any, Any]:
    parents = list(range(size))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    return parents, find, union


def cluster_instances(
    masks_by_pass: Sequence[Sequence[Mask]],
    iou_threshold: float = 0.3,
    unique_passes: bool = False,
) -> List[List[Tuple[int, Mask]]]:
    """Cluster instances across annotation passes using mask IoU.

    When ``unique_passes`` is enabled, two components are only merged when
    their annotation-pass sets are disjoint. This prevents a broad mask from
    transitively joining two distinct instances drawn in the same pass.
    """
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be between 0 and 1")
    nodes: List[Tuple[int, Mask]] = []
    for pass_index, masks in enumerate(masks_by_pass):
        nodes.extend((pass_index, np.asarray(mask).astype(bool)) for mask in masks)
    if not nodes:
        return []

    parents, find, union = _union_find(len(nodes))
    encoded = [mask_utils.encode(np.asfortranarray(mask.astype(np.uint8))) for _, mask in nodes]
    edges: List[Tuple[float, int, int]] = []
    for left in range(len(nodes)):
        for right in range(left + 1, len(nodes)):
            if nodes[left][0] == nodes[right][0]:
                continue
            value = float(mask_utils.iou([encoded[left]], [encoded[right]], [0])[0, 0])
            if value >= iou_threshold:
                edges.append((value, left, right))

    if unique_passes:
        component_passes = [{pass_index} for pass_index, _ in nodes]
        for _, left, right in sorted(edges, key=lambda item: (-item[0], item[1], item[2])):
            left_root = find(left)
            right_root = find(right)
            if left_root == right_root:
                continue
            if component_passes[left_root] & component_passes[right_root]:
                continue
            merged_passes = component_passes[left_root] | component_passes[right_root]
            union(left_root, right_root)
            component_passes[find(left_root)] = merged_passes
    else:
        for _, left, right in edges:
            union(left, right)

    grouped: Dict[int, List[Tuple[int, Mask]]] = defaultdict(list)
    for index, node in enumerate(nodes):
        grouped[find(index)].append(node)
    clusters = list(grouped.values())
    clusters.sort(
        key=lambda cluster: (
            min(int(np.nonzero(mask)[0].min()) for _, mask in cluster),
            min(int(np.nonzero(mask)[1].min()) for _, mask in cluster),
        )
    )
    return clusters


def consensus_instance_masks(
    masks_by_pass: Sequence[Sequence[Mask]],
    mode: str,
    iou_threshold: float = 0.3,
    unique_passes: bool = False,
) -> Tuple[List[Mask], Dict[str, Any]]:
    """Return masks fused using one of the supported consensus strategies.

    ``count_aware_sdf`` retains strict-majority instance presence.  It changes
    only two-pass boundary fusion, where strict pixel majority would otherwise
    degenerate to intersection; one- and three-pass images remain identical to
    ``majority``.
    """
    if mode not in {
        "union",
        "intersection",
        "majority",
        "majority_union",
        "count_aware_sdf",
    }:
        raise ValueError(
            "mode must be union, intersection, majority, majority_union, "
            "or count_aware_sdf"
        )
    num_passes = len(masks_by_pass)
    if num_passes < 1:
        raise ValueError("At least one annotation pass is required")
    clusters = cluster_instances(
        masks_by_pass,
        iou_threshold=iou_threshold,
        unique_passes=unique_passes,
    )
    required_majority = num_passes // 2 + 1
    outputs: List[Mask] = []
    discarded_unstable = 0
    discarded_empty = 0

    for cluster in clusters:
        present_passes = {pass_index for pass_index, _ in cluster}
        if mode == "intersection" and len(present_passes) != num_passes:
            discarded_unstable += 1
            continue
        if mode in {"majority", "majority_union", "count_aware_sdf"} and len(
            present_passes
        ) < required_majority:
            discarded_unstable += 1
            continue

        # One annotation pass must contribute at most one vote.  Transitive IoU
        # clustering can place two fragments from the same pass in one cluster;
        # summing the raw masks would otherwise let that pass fake a majority.
        masks_by_cluster_pass: Dict[int, List[Mask]] = defaultdict(list)
        for pass_index, mask in cluster:
            masks_by_cluster_pass[pass_index].append(mask)
        pass_masks = []
        for pass_index in sorted(masks_by_cluster_pass):
            pass_masks.append(
                np.logical_or.reduce(masks_by_cluster_pass[pass_index]).astype(np.uint8)
            )
        stack = np.stack(pass_masks, axis=0)
        if mode in {"union", "majority_union"}:
            output = stack.any(axis=0)
        elif mode == "intersection":
            output = stack.sum(axis=0) >= num_passes
        elif mode == "count_aware_sdf" and num_passes == 2:
            output = mean_signed_distance_mask(pass_masks)
        else:
            output = stack.sum(axis=0) >= required_majority
        if not output.any():
            discarded_empty += 1
            continue
        outputs.append(np.ascontiguousarray(output.astype(bool)))

    return outputs, {
        "input_instances": sum(len(items) for items in masks_by_pass),
        "clusters": len(clusters),
        "output_instances": len(outputs),
        "discarded_unstable": discarded_unstable,
        "discarded_empty": discarded_empty,
        "unique_passes": unique_passes,
    }


def mask_components_to_polygons(mask: Mask, min_area: int = 8) -> List[List[float]]:
    """Convert disconnected mask components into YOLO-compatible outer polygons."""
    binary = np.asarray(mask).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    polygons: List[List[float]] = []
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue
        simplified = cv2.approxPolyDP(contour, epsilon=0.5, closed=True).reshape(-1, 2)
        if len(simplified) < 3:
            continue
        polygons.append([float(value) for point in simplified for value in point])
    polygons.sort(key=lambda polygon: (min(polygon[1::2]), min(polygon[0::2])))
    return polygons
