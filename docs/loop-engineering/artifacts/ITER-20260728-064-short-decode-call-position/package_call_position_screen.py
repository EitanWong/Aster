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
    return sorted(run_root.glob("*-??-r*.json"))


def _archive(archive_path: Path, members: list[Path]) -> list[str]:
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
        description="Package the rejected I064 call-position screen without scratch duplication."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    run_root = args.run_dir.resolve()
    members = _members(run_root)
    if len(members) != 8:
        raise RuntimeError(f"expected eight call-position records, found {len(members)}")
    aggregate = json.loads(args.aggregate.resolve().read_text())
    if aggregate["decision"] != "reject":
        raise RuntimeError("package only supports a valid rejected call-position screen")
    archive_path = args.archive.resolve()
    archive_members = _archive(archive_path, members)
    source_paths = (
        ARTIFACT_DIR / "call_position_probe.py",
        ARTIFACT_DIR / "call_position_aggregate.py",
        ARTIFACT_DIR / "package_call_position_screen.py",
    )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "decision": "reject",
        "reason": aggregate["reason"],
        "screen": {
            name: aggregate[name]
            for name in (
                "gates",
                "second_call_slower_count",
                "second_vs_first_elapsed_gain_percent",
                "by_variant",
                "by_variant_bootstrap_median_interval",
                "by_warmup_terminal",
                "by_cell",
                "paired_variant_deltas",
            )
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
