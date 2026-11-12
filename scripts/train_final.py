#!/usr/bin/env python
"""Train either of the two selected final YOLO26 segmentation variants."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict

import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=("y17_v", "qt01_ttah"))
    parser.add_argument("--data", required=True, type=Path, help="Runtime Ultralytics dataset YAML")
    parser.add_argument("--pretrained", required=True, type=Path)
    parser.add_argument("--project", type=Path, default=ROOT / "work/runs")
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int)
    parser.add_argument("--workers", type=int)
    return parser.parse_args()


def load_config(variant: str) -> Dict[str, Any]:
    path = ROOT / "configs/final/{}.yaml".format(variant)
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError("Final configuration must be a YAML mapping")
    return value


def main() -> int:
    args = parse_args()
    config = load_config(args.variant)
    os.environ.setdefault("YOLO_AUTOINSTALL", "false")
    os.environ.setdefault("YOLO_OFFLINE", "true")

    from ultralytics import YOLO, __version__ as ultralytics_version

    train_args = {
        "data": str(args.data.resolve()),
        "project": str(args.project.resolve()),
        "name": str(config["run_name"]),
        "task": "segment",
        "imgsz": int(config["imgsz"]),
        "epochs": int(config["epochs"]),
        "patience": 0,
        "batch": int(args.batch if args.batch is not None else config["batch"]),
        "workers": int(args.workers if args.workers is not None else config["workers"]),
        "device": args.device,
        "seed": int(config["seed"]),
        "deterministic": True,
        "amp": True,
        "cache": False,
        "pretrained": True,
        "single_cls": True,
        "exist_ok": False,
        "save": True,
        "save_period": -1,
        "val": False,
        "plots": True,
        "verbose": True,
        "mosaic": 0.0,
        "mixup": 0.0,
        "copy_paste": 0.0,
        "hsv_h": 0.0,
        "hsv_s": 0.0,
        "hsv_v": 0.15,
        "degrees": 15.0,
        "translate": 0.05,
        "scale": 0.10,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.5,
        "fliplr": 0.5,
    }
    configured_args = config.get("training_args", {})
    if not isinstance(configured_args, dict):
        raise ValueError("training_args must be a YAML mapping")
    train_args.update(configured_args)
    # Runtime paths and explicit CLI overrides must take precedence over YAML.
    train_args.update({
        "data": str(args.data.resolve()),
        "project": str(args.project.resolve()),
        "name": str(config["run_name"]),
        "batch": int(args.batch if args.batch is not None else config["batch"]),
        "workers": int(args.workers if args.workers is not None else config["workers"]),
        "device": args.device,
    })
    print("ultralytics={}".format(ultralytics_version), flush=True)
    print("variant={}".format(args.variant), flush=True)
    model = YOLO(str(args.pretrained.resolve()))
    if config["trainer"] == "quality_target":
        os.environ["YOLO_QUALITY_TARGET_MIX"] = str(config["quality_target_mix"])
        os.environ["YOLO_QUALITY_TARGET_FLOOR"] = str(config["quality_target_floor"])
        os.environ["YOLO_QUALITY_TARGET_SCOPE"] = str(config["quality_target_scope"])
        from filament_seg.yolo_quality_target import QualityTargetSegmentationTrainer

        model.train(trainer=QualityTargetSegmentationTrainer, **train_args)
    elif config["trainer"] == "standard":
        model.train(**train_args)
    else:
        raise ValueError("Unsupported trainer: {}".format(config["trainer"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
