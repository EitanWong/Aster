#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import platform
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import paired_matrix as matrix

ARTIFACT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ARTIFACT_DIR.parents[3]


def _parse_run_id(path: Path) -> int:
    return int(path.stem.rsplit("-", 1)[-1])


def build(output_dir: Path) -> dict[str, Any]:
    paths = sorted(output_dir.glob("run-*.json"), key=_parse_run_id)
    if len(paths) != 3:
        raise ValueError(f"expected 3 focus outputs, found {len(paths)}")
    payloads = [json.loads(path.read_text()) for path in paths]
    first = payloads[0]
    workload = str(first["workload"])
    batch_size = int(first["batch_size"])
    source_hashes = dict(first["source_sha256"])
    for path in (Path(__file__).resolve(), ARTIFACT_DIR / "paired_aggregate.py"):
        source_hashes[str(path.relative_to(PROJECT_ROOT))] = matrix._sha256(path)

    records: list[dict[str, Any]] = []
    for path, payload in zip(paths, payloads, strict=True):
        if (
            str(payload["workload"]) != workload
            or int(payload["batch_size"]) != batch_size
            or payload["source_sha256"] != first["source_sha256"]
        ):
            raise ValueError(f"focus output mismatch: {path}")
        records.append(
            {
                "output": str(path.relative_to(ARTIFACT_DIR)),
                "sha256": matrix._sha256(path),
                "pid": int(payload["pid"]),
                "workload": workload,
                "batch_size": batch_size,
                "run_id": int(payload["run_id"]),
            }
        )

    return {
        "schema_version": 1,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "git_commit": matrix._git("rev-parse", "HEAD"),
            "git_branch": matrix._git("rev-parse", "--abbrev-ref", "HEAD"),
        },
        "matrix": {
            "profile": f"focused-{workload}-b{batch_size}-{first['settings']['steps']}",
            "cells": [{"workload": workload, "batch_size": batch_size}],
            "runs": 3,
            "context_words": int(first["context_words"]),
            "steps": int(first["settings"]["steps"]),
            "pair_warmup_steps": int(first["settings"]["pair_warmup_steps"]),
            "block_size": int(first["settings"]["block_size"]),
            "fresh_processes": True,
            "within_process_adjacent_pairing": True,
            "alternating_ab_ba_order": True,
        },
        "source_sha256": source_hashes,
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    manifest = build(output_dir)
    output = output_dir / "execution-manifest.json"
    output.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"records": len(manifest["records"]), "manifest": str(output)}))


if __name__ == "__main__":
    main()
