#!/usr/bin/env python
"""Prepare one-per-physical-image YOLO datasets with consensus annotations."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import _bootstrap  # noqa: F401
from filament_seg.data.coco import annotations_by_image, load_coco
from filament_seg.data.consensus import (
    annotation_masks,
    consensus_instance_masks,
    mask_components_to_polygons,
)
from filament_seg.data.folds import FoldRow, read_folds
from filament_seg.data.yolo import prepare_yolo_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True, type=Path)
    parser.add_argument("--folds", required=True, type=Path)
    parser.add_argument("--image-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--mode",
        required=True,
        choices=(
            "union",
            "intersection",
            "majority",
            "majority_union",
            "count_aware_sdf",
        ),
    )
    parser.add_argument("--val-fold", type=int, default=0)
    parser.add_argument("--iou-threshold", type=float, default=0.3)
    parser.add_argument(
        "--unique-passes",
        action="store_true",
        help="Prevent one consensus cluster from containing two instances from the same pass",
    )
    parser.add_argument("--link-mode", choices=("symlink", "hardlink", "copy"), default="symlink")
    return parser.parse_args()


def build_consensus_fold(
    data: Mapping[str, Any],
    rows: Sequence[FoldRow],
    mode: str,
    val_fold: int,
    iou_threshold: float,
    unique_passes: bool = False,
) -> Tuple[Dict[str, Any], List[FoldRow], Dict[str, Any]]:
    annotations = annotations_by_image(data)
    rows_by_id = {str(row["image_id"]): row for row in rows}
    by_file: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for image in data["images"]:
        by_file[str(image["file_name"])].append(image)

    output_images: List[Dict[str, Any]] = []
    output_annotations: List[Dict[str, Any]] = []
    output_rows: List[FoldRow] = []
    stats: Dict[str, Any] = {
        "mode": mode,
        "physical_images": len(by_file),
        "train_physical_images": 0,
        "validation_records": 0,
        "input_train_instances": 0,
        "output_train_instances": 0,
        "clusters": 0,
        "discarded_unstable": 0,
        "discarded_empty": 0,
        "unique_passes": unique_passes,
    }
    annotation_index = 1

    for file_name in sorted(by_file):
        records = sorted(by_file[file_name], key=lambda item: str(item["id"]))
        record_folds = {int(rows_by_id[str(item["id"])]["fold"]) for item in records}
        if len(record_folds) != 1:
            raise ValueError("Physical image crosses folds: {}".format(file_name))
        fold = next(iter(record_folds))
        if fold == val_fold:
            for image in records:
                image_id = str(image["id"])
                output_images.append(dict(image))
                output_rows.append(dict(rows_by_id[image_id]))
                for annotation in annotations.get(image_id, []):
                    output_annotations.append(dict(annotation))
                    annotation_index += 1
                stats["validation_records"] += 1
            continue

        representative = dict(records[0])
        representative_id = str(representative["id"])
        height = int(representative["height"])
        width = int(representative["width"])
        passes = [
            annotation_masks(annotations.get(str(image["id"]), []), height, width)
            for image in records
        ]
        masks, summary = consensus_instance_masks(
            passes,
            mode=mode,
            iou_threshold=iou_threshold,
            unique_passes=unique_passes,
        )
        output_images.append(representative)
        output_rows.append(dict(rows_by_id[representative_id]))
        stats["train_physical_images"] += 1
        for key in (
            "input_instances",
            "output_instances",
            "clusters",
            "discarded_unstable",
            "discarded_empty",
        ):
            target = {
                "input_instances": "input_train_instances",
                "output_instances": "output_train_instances",
            }.get(key, key)
            stats[target] += int(summary[key])

        for mask in masks:
            for polygon in mask_components_to_polygons(mask):
                output_annotations.append(
                    {
                        "id": "consensus_{}_{}".format(mode, annotation_index),
                        "image_id": representative["id"],
                        "category_id": 1,
                        "segmentation": [polygon],
                        "area": 0,
                        "bbox": [0, 0, 0, 0],
                        "iscrowd": 0,
                    }
                )
                annotation_index += 1

    output = {
        key: value for key, value in data.items() if key not in {"images", "annotations"}
    }
    output["images"] = output_images
    output["annotations"] = output_annotations
    return output, output_rows, stats


def main() -> int:
    args = parse_args()
    data, rows, stats = build_consensus_fold(
        load_coco(args.json),
        read_folds(args.folds),
        mode=args.mode,
        val_fold=args.val_fold,
        iou_threshold=args.iou_threshold,
        unique_passes=args.unique_passes,
    )
    manifest = prepare_yolo_dataset(
        data,
        rows,
        args.image_dir,
        args.output_dir,
        args.val_fold,
        args.link_mode,
    )
    summary = {"consensus": stats, "dataset": manifest["counts"]}
    (args.output_dir / "consensus_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
