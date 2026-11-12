#!/usr/bin/env python
"""Run low-threshold OOF inference and store compact COCO RLE predictions."""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from filament_seg.evaluation.pq import encode_binary_mask  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=100)
    parser.add_argument("--mask-quality", action="store_true")
    parser.add_argument("--mask-quality-alpha", type=float, default=1.0)
    return parser.parse_args()


def _to_original_size(mask_tensor: Any, height: int, width: int) -> np.ndarray:
    import torch
    import torch.nn.functional as functional

    tensor = mask_tensor.float()
    if tuple(tensor.shape[-2:]) != (height, width):
        tensor = functional.interpolate(
            tensor[None, None], size=(height, width), mode="nearest"
        )[0, 0]
    return tensor.detach().cpu().numpy() > 0.5


def main() -> int:
    args = parse_args()
    os.environ.setdefault("YOLO_AUTOINSTALL", "false")
    os.environ.setdefault("YOLO_OFFLINE", "true")
    if args.mask_quality_alpha < 0:
        raise ValueError("mask-quality-alpha must be non-negative")
    os.environ["YOLO_MASK_QUALITY_ALPHA"] = str(args.mask_quality_alpha)
    if args.mask_quality:
        from filament_seg.yolo_mask_quality import register_mask_quality_yolo_module

        register_mask_quality_yolo_module()
    from ultralytics import YOLO

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    model = YOLO(str(args.model))
    if args.mask_quality:
        from filament_seg.yolo_mask_quality import set_mask_quality_alpha

        quality_heads = set_mask_quality_alpha(model, args.mask_quality_alpha)
        print("mask_quality_heads={}".format(quality_heads))
        print("mask_quality_alpha={}".format(args.mask_quality_alpha))
    results = model.predict(
        source=str(args.source),
        imgsz=args.imgsz,
        device=args.device,
        batch=args.batch,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        retina_masks=True,
        stream=True,
        save=False,
        verbose=False,
    )

    count = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for result in results:
            image_id = Path(result.path).stem
            height, width = result.orig_shape
            instances: List[Dict[str, Any]] = []
            if result.masks is not None and result.boxes is not None:
                scores = result.boxes.conf.detach().cpu().numpy().tolist()
                boxes = result.boxes.xyxy.detach().cpu().numpy().tolist()
                for mask_tensor, score, box in zip(result.masks.data, scores, boxes):
                    mask = _to_original_size(mask_tensor, int(height), int(width))
                    instances.append(
                        {
                            "score": float(score),
                            "box": [float(value) for value in box],
                            "area": int(mask.sum()),
                            "rle": encode_binary_mask(mask),
                        }
                    )
            record = {
                "image_id": image_id,
                "height": int(height),
                "width": int(width),
                "instances": instances,
            }
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    temporary.replace(args.output)
    print("images={}".format(count))
    print("output={}".format(args.output.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
