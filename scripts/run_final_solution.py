#!/usr/bin/env python
"""Portable preparation, training, inference, and CSV generation entry point."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml


ROOT = Path(__file__).resolve().parents[1]
PYTHON = str(Path(sys.executable).resolve())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=("y17_v", "qt01_ttah"))
    parser.add_argument("--stage", required=True, choices=("prepare", "train", "predict", "all", "rebuild-csv"))
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/MAGFiLO_1.0_Kaggle_2026")
    parser.add_argument("--work-dir", type=Path, default=ROOT / "work")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--pretrained", type=Path, default=ROOT / "artifacts/weights/yolo26m-seg.pt")
    parser.add_argument("--device", default="0")
    parser.add_argument("--train-batch", type=int)
    parser.add_argument("--infer-batch", type=int, default=1)
    return parser.parse_args()


def load_config(variant: str) -> Dict[str, Any]:
    path = ROOT / "configs/final/{}.yaml".format(variant)
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def run(command: List[str]) -> None:
    print("RUN {}".format(" ".join(command)), flush=True)
    subprocess.run(command, cwd=str(ROOT), check=True)


def dataset_paths(data_root: Path, work_dir: Path) -> Dict[str, Path]:
    return {
        "json": data_root / "train/MAGFiLO_1.0_Annotations_kaggle2026_train.json",
        "train_images": data_root / "train/train_images",
        "test_images": data_root / "test/test_images",
        "folds": work_dir / "splits/folds.csv",
        "fold_summary": work_dir / "splits/folds_summary.json",
        "dataset": work_dir / "yolo_dataset/consensus_majority/full",
        "dataset_yaml": work_dir / "configs/final_full.yaml",
    }


def prepare(data_root: Path, work_dir: Path) -> Path:
    paths = dataset_paths(data_root, work_dir)
    paths["folds"].parent.mkdir(parents=True, exist_ok=True)
    if not paths["folds"].is_file():
        run([
            PYTHON, "scripts/make_folds.py", "--json", str(paths["json"]),
            "--output", str(paths["folds"]), "--summary", str(paths["fold_summary"]),
            "--n-splits", "5", "--seed", "42",
        ])
    if not paths["dataset"].is_dir():
        run([
            PYTHON, "scripts/prepare_consensus_yolo_dataset.py",
            "--json", str(paths["json"]), "--folds", str(paths["folds"]),
            "--image-dir", str(paths["train_images"]), "--output-dir", str(paths["dataset"]),
            "--mode", "majority", "--val-fold", "-1", "--iou-threshold", "0.3",
            "--link-mode", "symlink",
        ])
    paths["dataset_yaml"].parent.mkdir(parents=True, exist_ok=True)
    paths["dataset_yaml"].write_text(
        "path: {}\ntrain: images/train\nval: images/train\nnames:\n  0: filament\n".format(
            paths["dataset"].resolve().as_posix()
        ),
        encoding="utf-8",
    )
    return paths["dataset_yaml"]


def train(args: argparse.Namespace, config: Dict[str, Any]) -> Path:
    dataset_yaml = prepare(args.data_root, args.work_dir)
    command = [
        PYTHON, "scripts/train_final.py", "--variant", args.variant,
        "--data", str(dataset_yaml), "--pretrained", str(args.pretrained),
        "--project", str(args.work_dir / "runs"), "--device", args.device,
    ]
    if args.train_batch is not None:
        command.extend(("--batch", str(args.train_batch)))
    run(command)
    checkpoint = args.work_dir / "runs" / str(config["run_name"]) / "weights/last.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    return checkpoint


def selected_checkpoint(args: argparse.Namespace, config: Dict[str, Any]) -> Path:
    checkpoint = args.checkpoint or (ROOT / str(config["final_checkpoint"]))
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    return checkpoint


def build_csv(args: argparse.Namespace, config: Dict[str, Any], predictions: Path) -> Path:
    paths = dataset_paths(args.data_root, args.work_dir)
    output_dir = args.work_dir / "submissions"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / Path(str(config["final_submission"])).name
    audit = output.with_name(output.stem + "_audit.json")
    run([
        PYTHON, "scripts/build_submission.py", "--predictions", str(predictions),
        "--test-dir", str(paths["test_images"]), "--output", str(output), "--audit", str(audit),
        "--score-threshold", str(config["score_threshold"]), "--min-area", str(config["min_area"]),
        "--max-det", str(config["max_det"]),
    ])
    return output


def predict(args: argparse.Namespace, config: Dict[str, Any], checkpoint: Path) -> Path:
    paths = dataset_paths(args.data_root, args.work_dir)
    predictions = args.work_dir / "predictions/{}_test_tta.jsonl".format(args.variant)
    predictions.parent.mkdir(parents=True, exist_ok=True)
    run([
        PYTHON, "scripts/predict_tta.py", "--model", str(checkpoint),
        "--source", str(paths["test_images"]), "--output", str(predictions),
        "--views", str(config["views"]), "--fusion-mode", str(config["fusion_mode"]),
        "--duplicate-iou", str(config["duplicate_iou"]), "--imgsz", str(config["imgsz"]),
        "--device", args.device, "--batch", str(args.infer_batch), "--conf", "0.001",
        "--iou", "0.70", "--max-det", str(config["max_det"]),
    ])
    return build_csv(args, config, predictions)


def main() -> int:
    args = parse_args()
    config = load_config(args.variant)
    if args.stage == "prepare":
        print(prepare(args.data_root, args.work_dir))
        return 0
    if args.stage == "rebuild-csv":
        cached = ROOT / "artifacts/predictions/{}".format(
            "Y17_v_full_test_support_raw.jsonl" if args.variant == "y17_v" else "QT01_full_test_ttah_raw.jsonl"
        )
        print(build_csv(args, config, cached))
        return 0
    if args.stage in {"train", "all"}:
        checkpoint = train(args, config)
        if args.stage == "train":
            print(checkpoint)
            return 0
    else:
        checkpoint = selected_checkpoint(args, config)
    if args.stage in {"predict", "all"}:
        print(predict(args, config, checkpoint))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
