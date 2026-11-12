#!/usr/bin/env python
"""Build and audit the competition CSV from raw test predictions."""

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from filament_seg.evaluation.pq import read_prediction_jsonl  # noqa: E402
from filament_seg.submission import (  # noqa: E402
    audit_submission,
    build_submission_rows,
    image_stems,
    write_audit,
    write_submission,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--test-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--score-threshold", required=True, type=float)
    parser.add_argument("--min-area", type=int, default=0)
    parser.add_argument("--max-det", type=int, default=100)
    parser.add_argument("--small-score-threshold", type=float)
    parser.add_argument("--small-min-area", type=int, default=0)
    parser.add_argument("--small-min-aspect-ratio", type=float, default=1.0)
    parser.add_argument("--small-max-fill-ratio", type=float, default=1.0)
    parser.add_argument("--height", type=int, default=2048)
    parser.add_argument("--width", type=int, default=2048)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    stems = image_stems(args.test_dir)
    predictions = read_prediction_jsonl(args.predictions)
    missing_predictions = sorted(set(stems) - set(predictions))
    if missing_predictions:
        raise ValueError("Raw predictions are missing {} test images".format(len(missing_predictions)))

    rows = build_submission_rows(
        predictions,
        stems,
        score_threshold=args.score_threshold,
        min_area=args.min_area,
        max_det=args.max_det,
        small_score_threshold=args.small_score_threshold,
        small_min_area=args.small_min_area,
        small_min_aspect_ratio=args.small_min_aspect_ratio,
        small_max_fill_ratio=args.small_max_fill_ratio,
    )
    write_submission(args.output, rows)
    audit = audit_submission(args.output, stems, height=args.height, width=args.width)
    settings = {
        "predictions": str(args.predictions),
        "test_dir": str(args.test_dir),
        "output": str(args.output),
        "score_threshold": args.score_threshold,
        "min_area": args.min_area,
        "max_det": args.max_det,
        "small_score_threshold": args.small_score_threshold,
        "small_min_area": args.small_min_area,
        "small_min_aspect_ratio": args.small_min_aspect_ratio,
        "small_max_fill_ratio": args.small_max_fill_ratio,
    }
    write_audit(args.audit, audit, settings)
    print(json.dumps({"audit": audit, "settings": settings}, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if audit["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
