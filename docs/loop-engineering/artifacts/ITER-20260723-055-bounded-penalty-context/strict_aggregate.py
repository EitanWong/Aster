#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ARTIFACT_DIR = Path(__file__).resolve().parent
BASE_ARTIFACT_DIR = ARTIFACT_DIR.parent / "ITER-20260717-051-batch-sampler-sync"
if str(BASE_ARTIFACT_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_ARTIFACT_DIR))

import strict_aggregate as base  # noqa: E402


def aggregate(manifest_path: Path) -> dict[str, object]:
    original_artifact_dir = base.legacy.ARTIFACT_DIR
    base.legacy.ARTIFACT_DIR = ARTIFACT_DIR
    try:
        return base.aggregate(manifest_path)
    finally:
        base.legacy.ARTIFACT_DIR = original_artifact_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = aggregate(args.manifest.resolve())
    rendered = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
