#!/usr/bin/env python
"""Create deterministic date-grouped folds."""

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
from filament_seg.data.coco import load_coco, write_json
from filament_seg.data.folds import make_grouped_folds, summarize_folds, write_folds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = make_grouped_folds(load_coco(args.json), n_splits=args.n_splits, seed=args.seed)
    summary = summarize_folds(rows)
    if summary["leakage_files"] or summary["leakage_dates"]:
        raise RuntimeError("Generated folds contain leakage: {}".format(summary))
    write_folds(args.output, rows)
    if args.summary:
        write_json(args.summary, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

