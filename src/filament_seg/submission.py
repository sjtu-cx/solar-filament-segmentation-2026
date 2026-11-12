"""Build and audit Kaggle submission CSV files."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np
from pycocotools import mask as mask_utils

from filament_seg.evaluation.pq import coco_rle, filter_and_resolve_instances


SUBMISSION_COLUMNS = ("filament_id", "segmentation_rle")


def image_stems(image_dir: Path) -> List[str]:
    """Return sorted JPEG stems and reject ambiguous duplicate stems."""
    paths = sorted(
        path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg"}
    )
    stems = [path.stem for path in paths]
    if len(stems) != len(set(stems)):
        raise ValueError("Test image filenames must have unique stems")
    return stems


def build_submission_rows(
    predictions: Mapping[str, Mapping[str, Any]],
    expected_image_stems: Sequence[str],
    score_threshold: float,
    min_area: int = 0,
    max_det: int | None = None,
    small_score_threshold: float | None = None,
    small_min_area: int = 0,
    small_min_aspect_ratio: float = 1.0,
    small_max_fill_ratio: float = 1.0,
) -> List[Dict[str, str]]:
    """Convert raw predictions into non-overlapping compressed COCO RLE rows."""
    expected = set(expected_image_stems)
    unexpected = sorted(set(predictions) - expected)
    if unexpected:
        raise ValueError("Predictions contain unknown test images: {}".format(unexpected[:5]))

    rows: List[Dict[str, str]] = []
    for stem in expected_image_stems:
        record = predictions.get(stem, {"instances": []})
        rles = filter_and_resolve_instances(
            record.get("instances", []),
            score_threshold=score_threshold,
            min_area=min_area,
            max_det=max_det,
            resolve_overlaps=True,
            small_score_threshold=small_score_threshold,
            small_min_area=small_min_area,
            small_min_aspect_ratio=small_min_aspect_ratio,
            small_max_fill_ratio=small_max_fill_ratio,
        )
        for index, rle in enumerate(rles, 1):
            counts = rle["counts"]
            if isinstance(counts, bytes):
                counts = counts.decode("ascii")
            rows.append(
                {
                    "filament_id": "{}_{}".format(stem, index),
                    "segmentation_rle": str(counts),
                }
            )
    return rows


def write_submission(path: Path, rows: Iterable[Mapping[str, str]]) -> None:
    """Write a submission atomically with the exact competition columns."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUBMISSION_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def audit_submission(path: Path, expected_image_stems: Sequence[str], height: int, width: int) -> Dict[str, Any]:
    """Validate columns, IDs, RLE decoding, dimensions, and per-image overlap."""
    expected = set(expected_image_stems)
    errors: List[str] = []
    rows_by_image: Dict[str, List[Dict[str, str]]] = {}
    seen_ids = set()

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != SUBMISSION_COLUMNS:
            errors.append("Columns must be exactly {}".format(SUBMISSION_COLUMNS))
        for line_number, row in enumerate(reader, 2):
            filament_id = str(row.get("filament_id", ""))
            counts = str(row.get("segmentation_rle", ""))
            if filament_id in seen_ids:
                errors.append("Duplicate filament_id at line {}: {}".format(line_number, filament_id))
                continue
            seen_ids.add(filament_id)
            stem, separator, suffix = filament_id.rpartition("_")
            if not separator or not suffix.isdigit() or int(suffix) < 1:
                errors.append("Invalid filament_id at line {}: {}".format(line_number, filament_id))
                continue
            if stem not in expected:
                errors.append("Unknown image stem at line {}: {}".format(line_number, stem))
                continue
            if not counts:
                errors.append("Empty RLE at line {}".format(line_number))
                continue
            rows_by_image.setdefault(stem, []).append(row)

    total_area = 0
    overlap_pixels = 0
    for stem, rows in rows_by_image.items():
        occupied = np.zeros((height, width), dtype=bool)
        expected_indices = list(range(1, len(rows) + 1))
        actual_indices = sorted(int(str(row["filament_id"]).rpartition("_")[2]) for row in rows)
        if actual_indices != expected_indices:
            errors.append("Instance indices are not contiguous for {}".format(stem))
        for row in rows:
            try:
                rle = coco_rle({"size": [height, width], "counts": row["segmentation_rle"]})
                decoded = mask_utils.decode(rle).astype(bool)
            except Exception as exc:
                errors.append("Invalid RLE for {}: {}".format(row["filament_id"], exc))
                continue
            area = int(decoded.sum())
            if area == 0:
                errors.append("Empty mask for {}".format(row["filament_id"]))
            overlap = int(np.logical_and(occupied, decoded).sum())
            overlap_pixels += overlap
            if overlap:
                errors.append("Overlapping mask for {}: {} pixels".format(row["filament_id"], overlap))
            occupied |= decoded
            total_area += area

    images_with_instances = sorted(rows_by_image)
    return {
        "valid": not errors,
        "errors": errors,
        "columns": list(SUBMISSION_COLUMNS),
        "expected_images": len(expected_image_stems),
        "images_with_instances": len(images_with_instances),
        "empty_images": sorted(expected - set(images_with_instances)),
        "instances": len(seen_ids),
        "total_mask_area": total_area,
        "overlap_pixels": overlap_pixels,
        "height": int(height),
        "width": int(width),
    }


def write_audit(path: Path, audit: Mapping[str, Any], settings: Mapping[str, Any]) -> None:
    """Write audit results and generation settings atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump({"audit": audit, "settings": settings}, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)
