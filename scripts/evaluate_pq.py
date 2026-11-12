#!/usr/bin/env python
"""Evaluate OOF predictions and optionally sweep score thresholds."""

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import _bootstrap  # noqa: F401
from filament_seg.data.coco import load_coco, write_json
from filament_seg.data.folds import read_folds
from filament_seg.evaluation.pq import evaluate_predictions, read_prediction_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True, type=Path)
    parser.add_argument("--folds", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--val-fold", type=int, default=0)
    parser.add_argument("--score-thresholds", default="0.05,0.10,0.15,0.20,0.25,0.30,0.40,0.50")
    parser.add_argument("--min-area", type=int, default=0)
    parser.add_argument("--max-det", type=int)
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument("--relation-threshold", type=float, default=0.1)
    return parser.parse_args()


def _write_records(path: Path, records: List[Dict[str, Any]]) -> None:
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(records[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def main() -> int:
    args = parse_args()
    thresholds = [float(value.strip()) for value in args.score_thresholds.split(",") if value.strip()]
    if not thresholds:
        raise ValueError("At least one score threshold is required")

    data = load_coco(args.json)
    folds = read_folds(args.folds)
    predictions = read_prediction_jsonl(args.predictions)
    evaluations = []
    for threshold in thresholds:
        result = evaluate_predictions(
            data,
            folds,
            predictions,
            val_fold=args.val_fold,
            score_threshold=threshold,
            min_area=args.min_area,
            max_det=args.max_det,
            match_iou=args.match_iou,
            relation_threshold=args.relation_threshold,
        )
        evaluations.append(result)

    best = max(evaluations, key=lambda item: item["global"]["pq"])
    summary = {
        "best": {key: value for key, value in best.items() if key != "records"},
        "sweep": [
            {
                "score_threshold": item["settings"]["score_threshold"],
                "global": item["global"],
                "official_global": item["official_global"],
                "macro_record": item["macro_record"],
                "macro_physical_image_pq": item["macro_physical_image_pq"],
            }
            for item in evaluations
        ],
    }
    write_json(args.output, summary)
    _write_records(args.output.with_name(args.output.stem + "_records.csv"), best["records"])
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
