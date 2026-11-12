"""Leakage-safe grouped fold generation."""

from __future__ import annotations

import csv
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

from .coco import annotations_by_image, observation_metadata


FoldRow = Dict[str, Any]


def _date_features(
    images: Sequence[Mapping[str, Any]], annotation_counts: Mapping[str, int]
) -> Dict[str, Counter]:
    physical_passes: MutableMapping[str, int] = Counter(str(i["file_name"]) for i in images)
    by_date: MutableMapping[str, List[Mapping[str, Any]]] = defaultdict(list)
    for image in images:
        date, _, _ = observation_metadata(str(image["file_name"]))
        by_date[date].append(image)

    result: Dict[str, Counter] = {}
    for date, records in by_date.items():
        physical_names = sorted(set(str(r["file_name"]) for r in records))
        features: Counter = Counter()
        features["total:physical"] = len(physical_names)
        features["total:records"] = len(records)
        features["total:annotations"] = sum(annotation_counts.get(str(r["id"]), 0) for r in records)
        features["total:multipass"] = sum(physical_passes[name] > 1 for name in physical_names)
        for name in physical_names:
            _, year, station = observation_metadata(name)
            features["year:{}".format(year)] += 1
            features["station:{}".format(station)] += 1
        result[date] = features
    return result


def _assignment_score(
    fold_features: Counter,
    group_features: Counter,
    targets: Mapping[str, float],
) -> float:
    """Return the increase in normalized imbalance caused by an assignment."""
    score = 0.0
    for key, target in targets.items():
        before = fold_features.get(key, 0.0)
        after = before + group_features.get(key, 0.0)
        weight = 3.0 if key.startswith("total:") else 1.0
        normalizer = max(target, 1.0)
        score += weight * (((after - target) ** 2) - ((before - target) ** 2)) / normalizer
    return score


def make_grouped_folds(data: Mapping[str, Any], n_splits: int = 5, seed: int = 42) -> List[FoldRow]:
    """Assign observation dates to balanced folds with no physical-image leakage."""
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")

    anns_by_image = annotations_by_image(data)
    annotation_counts = {image_id: len(items) for image_id, items in anns_by_image.items()}
    date_features = _date_features(data["images"], annotation_counts)
    totals: Counter = Counter()
    for features in date_features.values():
        totals.update(features)
    targets = {key: value / float(n_splits) for key, value in totals.items()}

    category_frequency = Counter()
    for features in date_features.values():
        for key, value in features.items():
            if not key.startswith("total:") and value:
                category_frequency[key] += 1

    rng = random.Random(seed)
    dates = list(date_features)
    rng.shuffle(dates)

    def priority(date: str) -> Tuple[float, float, str]:
        features = date_features[date]
        rarity = sum(
            float(value) / max(category_frequency[key], 1)
            for key, value in features.items()
            if not key.startswith("total:")
        )
        load = features["total:annotations"] + 5.0 * features["total:records"]
        return (-rarity, -load, date)

    dates.sort(key=priority)
    fold_features = [Counter() for _ in range(n_splits)]
    date_to_fold: Dict[str, int] = {}

    for position, date in enumerate(dates):
        features = date_features[date]
        if position < n_splits:
            # Seed every fold with one group before greedy balancing. Without
            # this guard, rare-feature terms can occasionally leave a fold empty.
            fold = position
        else:
            scores = [_assignment_score(current, features, targets) for current in fold_features]
            best_score = min(scores)
            candidates = [index for index, score in enumerate(scores) if math.isclose(score, best_score)]
            fold = rng.choice(candidates)
        date_to_fold[date] = fold
        fold_features[fold].update(features)

    physical_passes = Counter(str(i["file_name"]) for i in data["images"])
    rows: List[FoldRow] = []
    for image in sorted(data["images"], key=lambda item: str(item["id"])):
        image_id = str(image["id"])
        file_name = str(image["file_name"])
        date, year, station = observation_metadata(file_name)
        rows.append(
            {
                "image_id": image_id,
                "file_name": file_name,
                "date": date,
                "year": year,
                "station": station,
                "num_passes": physical_passes[file_name],
                "num_instances": annotation_counts.get(image_id, 0),
                "fold": date_to_fold[date],
            }
        )
    return rows


FOLD_COLUMNS = [
    "image_id",
    "file_name",
    "date",
    "year",
    "station",
    "num_passes",
    "num_instances",
    "fold",
]


def write_folds(path: Path, rows: Iterable[FoldRow]) -> None:
    """Write fold assignments atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FOLD_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in FOLD_COLUMNS})
    temporary.replace(path)


def read_folds(path: Path) -> List[FoldRow]:
    """Read and type-convert fold assignments."""
    rows: List[FoldRow] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            row["num_passes"] = int(row["num_passes"])
            row["num_instances"] = int(row["num_instances"])
            row["fold"] = int(row["fold"])
            rows.append(row)
    return rows


def summarize_folds(rows: Sequence[FoldRow]) -> Dict[str, Any]:
    """Return fold sizes and leakage checks."""
    by_fold: MutableMapping[int, List[FoldRow]] = defaultdict(list)
    file_folds: MutableMapping[str, set] = defaultdict(set)
    date_folds: MutableMapping[str, set] = defaultdict(set)
    for row in rows:
        fold = int(row["fold"])
        by_fold[fold].append(row)
        file_folds[str(row["file_name"])].add(fold)
        date_folds[str(row["date"])].add(fold)

    leakage_files = sorted(name for name, folds in file_folds.items() if len(folds) > 1)
    leakage_dates = sorted(date for date, folds in date_folds.items() if len(folds) > 1)
    folds = {}
    for fold, items in sorted(by_fold.items()):
        folds[str(fold)] = {
            "image_records": len(items),
            "physical_images": len(set(str(item["file_name"]) for item in items)),
            "dates": len(set(str(item["date"]) for item in items)),
            "annotations": sum(int(item["num_instances"]) for item in items),
            "multipass_records": sum(int(item["num_passes"]) > 1 for item in items),
            "stations": dict(sorted(Counter(str(item["station"]) for item in items).items())),
            "years": dict(sorted(Counter(str(item["year"]) for item in items).items())),
        }
    return {
        "folds": folds,
        "leakage_files": leakage_files,
        "leakage_dates": leakage_dates,
    }
