#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path
from typing import Any

ARTIFACT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ARTIFACT_DIR.parents[3]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _members(run_root: Path) -> list[Path]:
    names = [
        "legacy-r1.json",
        "runtime-r1.json",
        "compare-r1.json",
        "runtime-r2.json",
        "legacy-r2.json",
        "compare-r2.json",
        "pipeline-pair-aggregate.json",
        *(f"pair-r{index}.json" for index in range(1, 7)),
    ]
    return [run_root / name for name in names]


def _archive(archive_path: Path, run_root: Path, members: list[Path]) -> list[str]:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "w:gz") as archive:
        for member in members:
            archive.add(member, arcname=member.name, recursive=False)
    with tarfile.open(archive_path, "r:gz") as archive:
        names = sorted(member.name for member in archive.getmembers() if member.isfile())
    expected = sorted(member.name for member in members)
    if names != expected:
        raise RuntimeError("screen archive member mismatch")
    return names


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Package the rejected I062 profiling screen without scratch duplication."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    run_root = args.run_dir.resolve()
    members = _members(run_root)
    for member in members:
        if not member.is_file():
            raise RuntimeError(f"missing required screen record: {member.name}")
    work_item_compares = [
        json.loads((run_root / f"compare-r{index}.json").read_text()) for index in (1, 2)
    ]
    pipeline_aggregate = json.loads(
        (run_root / "pipeline-pair-aggregate.json").read_text()
    )
    if not all(compare["comparable"] for compare in work_item_compares):
        raise RuntimeError("work-item screen lost exactness")
    if pipeline_aggregate["decision"] != "inconclusive":
        raise RuntimeError("package only supports the observed rejected pipeline screen")
    archive_path = args.archive.resolve()
    archive_members = _archive(archive_path, run_root, members)
    source_paths = (
        ARTIFACT_DIR / "work_item_profile.py",
        ARTIFACT_DIR / "mlx_pipeline_profile.py",
        ARTIFACT_DIR / "pipeline_pair_aggregate.py",
        Path(__file__).resolve(),
    )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "decision": "reject",
        "reason": (
            "The token-list branch was below the 3% screen floor and the "
            "lookahead-pipeline branch had order-reversing paired results."
        ),
        "work_item_screen": {
            "decode_loop_change_percent": [
                compare["decode_loop_change_percent"] for compare in work_item_compares
            ],
            "legacy_processor_tokens_percent": [
                compare["legacy_processor_tokens_percent"]
                for compare in work_item_compares
            ],
            "runtime_processor_tokens_percent": [
                compare["runtime_processor_tokens_percent"]
                for compare in work_item_compares
            ],
        },
        "pipeline_screen": {
            "decision": pipeline_aggregate["decision"],
            "gates": pipeline_aggregate["gates"],
            "pipeline_elapsed_gain_percent": pipeline_aggregate[
                "pipeline_elapsed_gain_percent"
            ],
            "bootstrap_median_interval": pipeline_aggregate[
                "bootstrap_median_interval"
            ],
            "by_order": pipeline_aggregate["by_order"],
        },
        "archive": {
            "path": str(archive_path.relative_to(PROJECT_ROOT)),
            "sha256": _sha256(archive_path),
            "bytes": archive_path.stat().st_size,
            "members": len(archive_members),
        },
        "source_sha256": {path.name: _sha256(path) for path in source_paths},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
