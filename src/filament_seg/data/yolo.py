"""Convert competition annotations into a leakage-safe YOLO segmentation dataset."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from .coco import annotations_by_image, images_by_id, iter_polygons
from .folds import FoldRow


def _safe_stem(image_id: str) -> str:
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.")
    if not image_id or any(character not in allowed for character in image_id):
        raise ValueError("Unsafe image id for a filename: {!r}".format(image_id))
    return image_id


def polygon_to_yolo(polygon: Sequence[Any], width: int, height: int) -> str:
    """Convert one absolute polygon to a normalized one-class YOLO label line."""
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive")
    if len(polygon) < 6 or len(polygon) % 2:
        raise ValueError("A polygon must contain at least three x/y points")

    values: List[str] = ["0"]
    for index in range(0, len(polygon), 2):
        x = min(max(float(polygon[index]) / float(width), 0.0), 1.0)
        y = min(max(float(polygon[index + 1]) / float(height), 0.0), 1.0)
        values.extend(("{:.8f}".format(x), "{:.8f}".format(y)))
    return " ".join(values)


def _ensure_link(source: Path, destination: Path, mode: str) -> None:
    if destination.exists() or destination.is_symlink():
        if destination.is_symlink() and destination.resolve() == source.resolve():
            return
        raise FileExistsError("Refusing to replace existing path: {}".format(destination))

    destination.parent.mkdir(parents=True, exist_ok=True)
    if mode == "symlink":
        relative_source = os.path.relpath(str(source), str(destination.parent))
        destination.symlink_to(relative_source)
    elif mode == "hardlink":
        os.link(str(source), str(destination))
    elif mode == "copy":
        shutil.copy2(str(source), str(destination))
    else:
        raise ValueError("Unsupported link mode: {}".format(mode))


def prepare_yolo_dataset(
    data: Mapping[str, Any],
    fold_rows: Sequence[FoldRow],
    image_dir: Path,
    output_dir: Path,
    val_fold: int,
    link_mode: str = "symlink",
) -> Dict[str, Any]:
    """Create train/val links, labels, YAML, and a manifest for one fold."""
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("Output directory is not empty: {}".format(output_dir))

    images = images_by_id(data)
    anns_by_image = annotations_by_image(data)
    fold_by_image = {str(row["image_id"]): int(row["fold"]) for row in fold_rows}
    if set(images) != set(fold_by_image):
        missing = sorted(set(images) - set(fold_by_image))
        extra = sorted(set(fold_by_image) - set(images))
        raise ValueError("Fold/image mismatch: missing={}, extra={}".format(len(missing), len(extra)))

    counts = {
        "train_records": 0,
        "val_records": 0,
        "train_instances": 0,
        "val_instances": 0,
    }
    sample_manifest: List[Dict[str, Any]] = []

    for image_id in sorted(images):
        image = images[image_id]
        subset = "val" if fold_by_image[image_id] == val_fold else "train"
        stem = _safe_stem(image_id)
        source = image_dir / str(image["file_name"])
        if not source.is_file():
            raise FileNotFoundError("Missing source image: {}".format(source))

        image_destination = output_dir / "images" / subset / (stem + source.suffix.lower())
        label_destination = output_dir / "labels" / subset / (stem + ".txt")
        _ensure_link(source, image_destination, link_mode)
        label_destination.parent.mkdir(parents=True, exist_ok=True)

        label_lines: List[str] = []
        for annotation in anns_by_image.get(image_id, []):
            polygons = list(iter_polygons(annotation))
            if len(polygons) != 1:
                raise ValueError(
                    "Annotation {} has {} polygons; exactly one is required".format(
                        annotation.get("id"), len(polygons)
                    )
                )
            label_lines.append(
                polygon_to_yolo(polygons[0], int(image["width"]), int(image["height"]))
            )

        label_destination.write_text("\n".join(label_lines) + "\n", encoding="utf-8")
        counts[subset + "_records"] += 1
        counts[subset + "_instances"] += len(label_lines)
        sample_manifest.append(
            {
                "sample_id": image_id,
                "file_name": str(image["file_name"]),
                "subset": subset,
                "fold": fold_by_image[image_id],
                "instances": len(label_lines),
            }
        )

    dataset_yaml = output_dir / "dataset.yaml"
    yaml_text = (
        "path: {}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: filament\n"
    ).format(output_dir.resolve().as_posix())
    dataset_yaml.write_text(yaml_text, encoding="utf-8")

    manifest = {
        "val_fold": val_fold,
        "link_mode": link_mode,
        "dataset_yaml": str(dataset_yaml.resolve()),
        "counts": counts,
        "samples": sample_manifest,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest

