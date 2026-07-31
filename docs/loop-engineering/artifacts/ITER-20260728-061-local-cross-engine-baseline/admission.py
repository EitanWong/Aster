#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path
from typing import Any

ARTIFACT_DIR = Path(__file__).resolve().parent


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verified_members(run_root: Path, manifest: dict[str, Any]) -> list[Path]:
    members = [run_root / "manifest.json", run_root / "aggregate.json"]
    for scenario in manifest["scenarios"]:
        for pair in scenario["pairs"]:
            descriptors = [*pair["records"].values(), pair["comparison"]]
            for descriptor in descriptors:
                path = run_root / descriptor["path"]
                if _sha256(path) != descriptor["sha256"]:
                    raise RuntimeError(f"evidence hash mismatch: {descriptor['path']}")
                members.append(path)
    return members


def _archive(archive_path: Path, run_root: Path, members: list[Path]) -> list[str]:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "w:gz") as archive:
        for member in members:
            archive.add(member, arcname=str(member.relative_to(run_root)), recursive=False)
    with tarfile.open(archive_path, "r:gz") as archive:
        archive_members = sorted(item.name for item in archive.getmembers() if item.isfile())
    expected = sorted(str(member.relative_to(run_root)) for member in members)
    if archive_members != expected:
        raise RuntimeError("formal archive member set does not match the verified evidence")
    return archive_members


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Package and admit the compact I061 cross-engine evidence."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    run_root = args.run_dir.resolve()
    manifest_path = run_root / "manifest.json"
    aggregate_path = run_root / "aggregate.json"
    manifest = json.loads(manifest_path.read_text())
    aggregate = json.loads(aggregate_path.read_text())
    source_sha256 = {
        path.name: _sha256(path)
        for path in (
            ARTIFACT_DIR / "preflight.py",
            ARTIFACT_DIR / "paired_matrix.py",
            ARTIFACT_DIR / "aggregate.py",
            Path(__file__).resolve(),
        )
    }
    source_matches = (
        manifest["source_sha256"]["preflight.py"] == source_sha256["preflight.py"]
        and manifest["source_sha256"]["paired_matrix.py"] == source_sha256["paired_matrix.py"]
    )
    members = _verified_members(run_root, manifest)
    archive_members = _archive(args.archive.resolve(), run_root, members)
    archive_path = args.archive.resolve()
    archive_under_budget = archive_path.stat().st_size <= 5 * 1024 * 1024
    gates = {
        "aggregate_admit": aggregate["decision"] == "admit",
        "aggregate_gates": all(aggregate["gates"].values()),
        "source_hashes_match": source_matches,
        "record_hashes_verified": True,
        "archive_member_set": len(archive_members) == len(members),
        "archive_under_5_mib": archive_under_budget,
    }
    payload: dict[str, Any] = {
        "schema_version": 1,
        "decision": "admit" if all(gates.values()) else "reject",
        "gates": gates,
        "manifest_sha256": _sha256(manifest_path),
        "aggregate_sha256": _sha256(aggregate_path),
        "source_sha256": source_sha256,
        "archive": {
            "path": str(archive_path.relative_to(ARTIFACT_DIR.parents[3])),
            "sha256": _sha256(archive_path),
            "bytes": archive_path.stat().st_size,
            "members": len(archive_members),
        },
        "scenarios": [
            {
                "name": scenario["name"],
                "aster_p50_tokens_per_second": scenario[
                    "aster_output_tokens_per_second"
                ]["p50"],
                "mlx_lm_p50_tokens_per_second": scenario[
                    "mlx_lm_output_tokens_per_second"
                ]["p50"],
                "paired_change_percent_p50": scenario["paired_change_percent"]["p50"],
                "paired_change_percent_interval": scenario["paired_change_percent"][
                    "bootstrap_median_interval"
                ],
            }
            for scenario in aggregate["scenarios"]
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if payload["decision"] != "admit":
        raise SystemExit("cross-engine evidence rejected")


if __name__ == "__main__":
    main()
