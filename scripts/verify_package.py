#!/usr/bin/env python
"""Verify final artifacts and report publication-only blockers separately."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--publication-ready",
        action="store_true",
        help="Fail on pending URLs, missing license, author placeholders, or private path markers.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_hashed_file(
    entry: Dict[str, Any], path_key: str, hash_key: str, failures: List[str]
) -> None:
    relative = entry.get(path_key)
    expected = entry.get(hash_key)
    if not relative or not expected:
        failures.append("manifest is missing {} or {}".format(path_key, hash_key))
        return
    path = ROOT / str(relative)
    if not path.is_file():
        failures.append("missing {}".format(path.relative_to(ROOT)))
        return
    actual = sha256(path)
    if actual.lower() != str(expected).lower():
        failures.append(
            "hash mismatch {} expected={} actual={}".format(path.relative_to(ROOT), expected, actual)
        )


def publication_scan(blockers: List[str]) -> None:
    private_patterns = {
        "private IP": re.compile(r"10\.(?:75|180)\."),
        "Linux server path": re.compile(r"/mnt/(?:data|data01|caoxin)"),
        "Windows workstation path": re.compile(r"[A-Za-z]:\\(?:Users|Work)\\"),
    }
    scan_paths = [
        ROOT / "README.md",
        ROOT / "FINAL_SUBMISSION_MANIFEST.json",
        ROOT / "FINAL_SUBMISSION_CHECKLIST.md",
        ROOT / "notebooks/final_pipeline.ipynb",
        ROOT / "report/main_draft.tex",
        ROOT / "report/technical_report_draft.md",
        ROOT / "submissions/Y17_V_full_submission_audit.json",
        ROOT / "submissions/QT01TTAH_full_submission_audit.json",
    ]
    for path in scan_paths:
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        for label, pattern in private_patterns.items():
            if pattern.search(text):
                blockers.append("{} contains {}".format(path.relative_to(ROOT), label))
    manifest_text = (ROOT / "FINAL_SUBMISSION_MANIFEST.json").read_text(encoding="utf-8")
    if "PENDING_PUBLIC_RELEASE_URL" in manifest_text:
        blockers.append("manifest still contains pending public release URLs")
    report_text = (ROOT / "report/main_draft.tex").read_text(encoding="utf-8")
    if any(marker in report_text for marker in ("[TEAM NAME]", "[AUTHORS]", "[CONTACT EMAIL]")):
        blockers.append("report author metadata is incomplete")
    if not (ROOT / "LICENSE").is_file():
        blockers.append("OSI-approved LICENSE file is missing")


def main() -> int:
    args = parse_args()
    failures: List[str] = []
    blockers: List[str] = []
    manifest_path = ROOT / "FINAL_SUBMISSION_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    variants = []
    for entry in manifest.get("final_submissions", []):
        variants.append(entry.get("variant"))
        for path_key, hash_key in (
            ("submission", "submission_sha256"),
            ("checkpoint", "checkpoint_sha256"),
            ("prediction", "prediction_sha256"),
        ):
            verify_hashed_file(entry, path_key, hash_key, failures)
        audit_path = ROOT / str(entry.get("audit", ""))
        if not audit_path.is_file():
            failures.append("missing {}".format(audit_path.relative_to(ROOT)))
        else:
            audit = json.loads(audit_path.read_text(encoding="utf-8")).get("audit", {})
            if (
                not audit.get("valid")
                or audit.get("errors")
                or audit.get("overlap_pixels") != 0
                or audit.get("expected_images") != 180
            ):
                failures.append("invalid audit {}".format(audit_path.relative_to(ROOT)))
    verify_hashed_file(
        manifest["pretrained_checkpoint"], "file", "sha256", failures
    )
    notebook_path = ROOT / "notebooks/final_pipeline.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    notebook_source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook.get("cells", [])
    )
    required_notebook_steps = (
        "RUN_TRAINING",
        "RUN_GPU_INFERENCE",
        "rebuild-csv",
        "verify_package.py",
        "('y17_v', 'qt01_ttah')",
    )
    if notebook.get("nbformat") != 4 or not notebook.get("cells"):
        failures.append("invalid notebooks/final_pipeline.ipynb")
    for step in required_notebook_steps:
        if step not in notebook_source:
            failures.append("notebook is missing executable step: {}".format(step))
    for required in (
        "README.md",
        "requirements.txt",
        "configs/final/y17_v.yaml",
        "configs/final/qt01_ttah.yaml",
        "src/filament_seg/data/__init__.py",
        "src/filament_seg/data/coco.py",
        "src/filament_seg/data/folds.py",
        "src/filament_seg/data/consensus.py",
        "src/filament_seg/data/yolo.py",
        "report/main_draft.tex",
        "report/Solar_Filament_Final_Technical_Report.pdf",
    ):
        if not (ROOT / required).is_file():
            failures.append("missing {}".format(required))
    publication_scan(blockers)
    result = {
        "valid": not failures,
        "publication_ready": not failures and not blockers,
        "variants": variants,
        "failures": failures,
        "publication_blockers": blockers,
    }
    print(json.dumps(result, indent=2))
    if failures or (args.publication_ready and blockers):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
