"""Read and validate the competition's COCO-style annotation file."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple


JsonDict = Dict[str, Any]


def load_coco(path: Path) -> JsonDict:
    """Load a COCO-style JSON file and validate its top-level shape."""
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    for key in ("images", "annotations", "categories"):
        if key not in data or not isinstance(data[key], list):
            raise ValueError("Missing or invalid top-level key: {!r}".format(key))
    return data


def annotations_by_image(data: Mapping[str, Any]) -> Dict[str, List[JsonDict]]:
    """Index annotations by their string image id."""
    result: MutableMapping[str, List[JsonDict]] = defaultdict(list)
    for annotation in data["annotations"]:
        result[str(annotation["image_id"])].append(annotation)
    return dict(result)


def images_by_id(data: Mapping[str, Any]) -> Dict[str, JsonDict]:
    """Index image records by their string id, rejecting duplicates."""
    result: Dict[str, JsonDict] = {}
    for image in data["images"]:
        image_id = str(image["id"])
        if image_id in result:
            raise ValueError("Duplicate image id: {}".format(image_id))
        result[image_id] = image
    return result


def observation_metadata(file_name: str) -> Tuple[str, str, str]:
    """Return (date, year, station) encoded by a competition image name."""
    stem = Path(file_name).stem
    if len(stem) < 16 or not stem[:14].isdigit():
        raise ValueError("Unexpected observation filename: {}".format(file_name))
    return stem[:8], stem[:4], stem[14:16]


def iter_polygons(annotation: Mapping[str, Any]) -> Iterable[List[float]]:
    """Yield non-empty polygons from one annotation."""
    segmentation = annotation.get("segmentation", [])
    if not isinstance(segmentation, list):
        return
    for polygon in segmentation:
        if isinstance(polygon, list) and polygon:
            yield polygon


def audit_coco(
    data: Mapping[str, Any],
    image_dir: Path,
    check_images: bool = False,
) -> JsonDict:
    """Audit references, polygons, dimensions, and optional JPEG decoding."""
    images = images_by_id(data)
    anns_by_image = annotations_by_image(data)
    annotation_ids = Counter(str(a.get("id")) for a in data["annotations"])
    file_name_counts = Counter(str(i["file_name"]) for i in data["images"])

    errors: List[str] = []
    warnings: List[str] = []
    category_counts: Counter = Counter()
    polygon_count = 0
    point_count = 0
    missing_files: List[str] = []
    referenced_files = set()

    duplicate_annotation_ids = sorted(k for k, count in annotation_ids.items() if count > 1)
    if duplicate_annotation_ids:
        errors.append("Duplicate annotation ids: {}".format(len(duplicate_annotation_ids)))

    unknown_image_refs = sorted(set(anns_by_image) - set(images))
    if unknown_image_refs:
        errors.append("Annotations reference {} unknown image ids".format(len(unknown_image_refs)))

    for image_id, image in images.items():
        file_name = str(image["file_name"])
        referenced_files.add(file_name)
        image_path = image_dir / file_name
        if not image_path.is_file():
            missing_files.append(file_name)
            continue

        width = int(image.get("width", 0))
        height = int(image.get("height", 0))
        if width <= 0 or height <= 0:
            errors.append("Invalid dimensions for image record {}".format(image_id))
            continue

        if check_images:
            try:
                from PIL import Image

                with Image.open(str(image_path)) as decoded:
                    decoded.load()
                    if decoded.size != (width, height):
                        errors.append(
                            "Dimension mismatch for {}: JSON={}x{}, JPEG={}x{}".format(
                                file_name, width, height, decoded.size[0], decoded.size[1]
                            )
                        )
            except Exception as exc:  # pragma: no cover - depends on corrupt input
                errors.append("Failed to decode {}: {}".format(file_name, exc))

    for annotation in data["annotations"]:
        annotation_id = str(annotation.get("id"))
        image_id = str(annotation.get("image_id"))
        image = images.get(image_id)
        if image is None:
            continue
        category_counts[str(annotation.get("category_id"))] += 1
        if int(annotation.get("iscrowd", 0)) != 0:
            warnings.append("iscrowd is non-zero for annotation {}".format(annotation_id))

        polygons = list(iter_polygons(annotation))
        if len(polygons) != 1:
            errors.append(
                "Annotation {} has {} polygons; exactly one is expected".format(annotation_id, len(polygons))
            )
            continue

        polygon = polygons[0]
        polygon_count += 1
        if len(polygon) < 6 or len(polygon) % 2:
            errors.append("Invalid polygon length for annotation {}".format(annotation_id))
            continue
        point_count += len(polygon) // 2
        width = float(image["width"])
        height = float(image["height"])
        for index in range(0, len(polygon), 2):
            try:
                x = float(polygon[index])
                y = float(polygon[index + 1])
            except (TypeError, ValueError):
                errors.append("Non-numeric polygon coordinate in annotation {}".format(annotation_id))
                break
            if x < 0 or y < 0 or x > width or y > height:
                errors.append("Out-of-bounds polygon coordinate in annotation {}".format(annotation_id))
                break

    physical_records: MutableMapping[str, List[JsonDict]] = defaultdict(list)
    for image in data["images"]:
        physical_records[str(image["file_name"])].append(image)

    pass_histogram = Counter(len(records) for records in physical_records.values())
    dates = set()
    stations = Counter()
    years = Counter()
    for file_name in physical_records:
        try:
            date, year, station = observation_metadata(file_name)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        dates.add(date)
        stations[station] += 1
        years[year] += 1

    extra_jpegs = sorted(
        path.name
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in (".jpg", ".jpeg") and path.name not in referenced_files
    )

    if missing_files:
        errors.append("Missing {} referenced JPEG files".format(len(set(missing_files))))
    if extra_jpegs:
        warnings.append("Found {} unreferenced JPEG files".format(len(extra_jpegs)))

    return {
        "summary": {
            "image_records": len(data["images"]),
            "physical_images": len(physical_records),
            "unique_dates": len(dates),
            "annotations": len(data["annotations"]),
            "polygons": polygon_count,
            "polygon_points": point_count,
            "categories": len(data["categories"]),
            "missing_jpegs": len(set(missing_files)),
            "extra_jpegs": len(extra_jpegs),
        },
        "pass_histogram": dict(sorted(pass_histogram.items())),
        "category_counts": dict(sorted(category_counts.items())),
        "year_counts": dict(sorted(years.items())),
        "station_counts": dict(sorted(stations.items())),
        "errors": errors,
        "warnings": warnings,
    }


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    """Write JSON atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)

