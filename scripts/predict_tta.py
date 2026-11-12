#!/usr/bin/env python
"""Run explicit flip TTA for YOLO segmentation and write fused JSONL predictions."""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from filament_seg.ensembling import (  # noqa: E402
    deduplicate_instances,
    fuse_view_instances,
    instance_from_mask,
    restore_flipped_mask,
)


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
SUPPORTED_VIEWS = ("original", "hflip", "vflip")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--base-predictions",
        type=Path,
        help="Reuse JSONL predictions for the original view instead of inferring it again.",
    )
    parser.add_argument("--views", default="original,hflip,vflip")
    parser.add_argument("--duplicate-iou", type=float, default=0.5)
    parser.add_argument(
        "--fusion-mode",
        choices=("best", "intersection", "majority", "eroded_majority"),
        default="best",
    )
    parser.add_argument("--imgsz", type=int, default=1536)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=100)
    parser.add_argument("--mask-quality", action="store_true")
    parser.add_argument("--mask-quality-alpha", type=float, default=1.0)
    return parser.parse_args()


def image_paths(source: Path) -> List[Path]:
    paths = sorted(path for path in source.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    if not paths:
        raise ValueError("No supported images found in {}".format(source))
    stems = [path.stem for path in paths]
    if len(stems) != len(set(stems)):
        raise ValueError("Image stems must be unique")
    return paths


def parse_views(value: str) -> List[str]:
    views = [item.strip() for item in value.split(",") if item.strip()]
    if not views or len(views) != len(set(views)):
        raise ValueError("views must be a non-empty list without duplicates")
    unsupported = sorted(set(views) - set(SUPPORTED_VIEWS))
    if unsupported:
        raise ValueError("Unsupported views: {}".format(unsupported))
    return views


def chunks(items: Sequence[Path], size: int) -> Iterable[Sequence[Path]]:
    if size < 1:
        raise ValueError("batch must be at least 1")
    for start in range(0, len(items), size):
        yield items[start : start + size]


def transformed(image: np.ndarray, view: str) -> np.ndarray:
    if view == "original":
        result = image
    elif view == "hflip":
        result = np.fliplr(image)
    elif view == "vflip":
        result = np.flipud(image)
    else:
        raise ValueError("Unsupported TTA view: {}".format(view))
    return np.ascontiguousarray(result)


def masks_to_original_size(mask_tensors: Any, height: int, width: int) -> np.ndarray:
    """Move all masks for one result to CPU with a single synchronization."""
    import torch.nn.functional as functional

    tensor = mask_tensors.float()
    if tensor.ndim != 3:
        raise ValueError("Expected NxHxW masks, got shape {}".format(tuple(tensor.shape)))
    if tuple(tensor.shape[-2:]) != (height, width):
        tensor = functional.interpolate(tensor[:, None], size=(height, width), mode="nearest")[:, 0]
    return tensor.detach().cpu().numpy() > 0.5


def read_base_predictions(path: Path) -> Dict[str, Dict[str, Any]]:
    predictions: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            image_id = str(record["image_id"])
            if image_id in predictions:
                raise ValueError("Duplicate base prediction for {} at line {}".format(image_id, line_number))
            predictions[image_id] = record
    return predictions


def append_base_instances(
    target: List[Dict[str, Any]],
    record: Mapping[str, Any],
    height: int,
    width: int,
) -> None:
    if (int(record["height"]), int(record["width"])) != (height, width):
        raise ValueError("Base prediction dimensions do not match the source image")
    for item in record.get("instances", []):
        copied = dict(item)
        copied["view"] = "original"
        target.append(copied)


def main() -> int:
    args = parse_args()
    views = parse_views(args.views)
    paths = image_paths(args.source)
    base_predictions: Optional[Dict[str, Dict[str, Any]]] = None
    inference_views = list(views)
    if args.base_predictions is not None:
        if "original" not in views:
            raise ValueError("base-predictions requires original in --views")
        base_predictions = read_base_predictions(args.base_predictions)
        expected_ids = {path.stem for path in paths}
        missing = sorted(expected_ids - set(base_predictions))
        extra = sorted(set(base_predictions) - expected_ids)
        if missing or extra:
            raise ValueError(
                "Base prediction image ids differ from source: missing={} extra={}".format(
                    missing[:5], extra[:5]
                )
            )
        inference_views = [view for view in views if view != "original"]
    os.environ.setdefault("YOLO_AUTOINSTALL", "false")
    os.environ.setdefault("YOLO_OFFLINE", "true")
    if args.mask_quality_alpha < 0:
        raise ValueError("mask-quality-alpha must be non-negative")
    os.environ["YOLO_MASK_QUALITY_ALPHA"] = str(args.mask_quality_alpha)
    if args.mask_quality:
        from filament_seg.yolo_mask_quality import register_mask_quality_yolo_module

        register_mask_quality_yolo_module()
    from ultralytics import YOLO

    model = YOLO(str(args.model))
    if args.mask_quality:
        from filament_seg.yolo_mask_quality import set_mask_quality_alpha

        quality_heads = set_mask_quality_alpha(model, args.mask_quality_alpha)
        print("mask_quality_heads={}".format(quality_heads))
        print("mask_quality_alpha={}".format(args.mask_quality_alpha))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    processed = 0
    timings = {"decode_transform": 0.0, "model_predict": 0.0, "mask_export": 0.0, "fusion_write": 0.0}

    with temporary.open("w", encoding="utf-8") as handle:
        for path_chunk in chunks(paths, args.batch):
            started = time.perf_counter()
            sources: List[np.ndarray] = []
            metadata: List[Tuple[Path, str, int, int]] = []
            by_image: Dict[str, List[Dict[str, Any]]] = {path.stem: [] for path in path_chunk}
            dimensions: Dict[str, Tuple[int, int]] = {}
            for path in path_chunk:
                image = cv2.imread(str(path), cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError("Could not decode {}".format(path))
                height, width = image.shape[:2]
                dimensions[path.stem] = (height, width)
                if base_predictions is not None:
                    append_base_instances(by_image[path.stem], base_predictions[path.stem], height, width)
                for view in inference_views:
                    sources.append(transformed(image, view))
                    metadata.append((path, view, height, width))
            timings["decode_transform"] += time.perf_counter() - started

            started = time.perf_counter()
            results = (
                model.predict(
                    source=sources,
                    imgsz=args.imgsz,
                    device=args.device,
                    batch=max(1, args.batch * len(inference_views)),
                    conf=args.conf,
                    iou=args.iou,
                    max_det=args.max_det,
                    retina_masks=True,
                    stream=False,
                    save=False,
                    verbose=False,
                )
                if sources
                else []
            )
            timings["model_predict"] += time.perf_counter() - started
            if len(results) != len(metadata):
                raise RuntimeError("YOLO returned an unexpected TTA batch length")

            started = time.perf_counter()
            for result, (path, view, height, width) in zip(results, metadata):
                if result.masks is None or result.boxes is None:
                    continue
                scores = result.boxes.conf.detach().cpu().numpy()
                masks = masks_to_original_size(result.masks.data, height, width)
                for mask, score in zip(masks, scores):
                    restored = restore_flipped_mask(mask, view)
                    if restored.any():
                        by_image[path.stem].append(instance_from_mask(restored, float(score), view=view))
            timings["mask_export"] += time.perf_counter() - started

            started = time.perf_counter()
            for path in path_chunk:
                height, width = dimensions[path.stem]
                if args.fusion_mode == "best":
                    instances = deduplicate_instances(
                        by_image[path.stem], duplicate_iou=args.duplicate_iou, score_threshold=args.conf
                    )
                else:
                    instances = fuse_view_instances(
                        by_image[path.stem],
                        duplicate_iou=args.duplicate_iou,
                        fusion_mode=args.fusion_mode,
                        score_threshold=args.conf,
                    )
                record = {
                    "image_id": path.stem,
                    "height": int(height),
                    "width": int(width),
                    "instances": instances,
                }
                handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                processed += 1
                if processed % 25 == 0 or processed == len(paths):
                    print("images={}/{}".format(processed, len(paths)), flush=True)
            timings["fusion_write"] += time.perf_counter() - started

    temporary.replace(args.output)
    print("views={}".format(",".join(views)))
    print("inference_views={}".format(",".join(inference_views) or "none"))
    print("fusion_mode={}".format(args.fusion_mode))
    print(
        "timings_seconds="
        + json.dumps({key: round(value, 3) for key, value in timings.items()}, separators=(",", ":"))
    )
    print("output={}".format(args.output.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
