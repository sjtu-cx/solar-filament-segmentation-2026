#!/usr/bin/env python
"""Download missing public weight artifacts declared in the final manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path
from typing import Dict, Iterator, Tuple


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Replace an existing hash-mismatched file")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifacts(manifest: Dict[str, object]) -> Iterator[Tuple[str, str, str]]:
    pretrained = manifest["pretrained_checkpoint"]
    yield pretrained["file"], pretrained["sha256"], pretrained["public_url"]
    for item in manifest["final_submissions"]:
        yield item["checkpoint"], item["checkpoint_sha256"], item["checkpoint_public_url"]


def main() -> int:
    args = parse_args()
    manifest = json.loads((ROOT / "FINAL_SUBMISSION_MANIFEST.json").read_text(encoding="utf-8"))
    for relative, expected, url in artifacts(manifest):
        target = ROOT / relative
        if target.is_file() and sha256(target).lower() == expected.lower():
            print("OK", relative)
            continue
        if url == "PENDING_PUBLIC_RELEASE_URL":
            raise RuntimeError("Public URL has not been filled for {}".format(relative))
        if target.exists() and not args.force:
            raise RuntimeError("Hash mismatch for {}; pass --force to replace it".format(relative))
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".part")
        urllib.request.urlretrieve(url, temporary)
        actual = sha256(temporary)
        if actual.lower() != expected.lower():
            temporary.unlink()
            raise RuntimeError("Downloaded hash mismatch for {}".format(relative))
        temporary.replace(target)
        print("DOWNLOADED", relative)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
