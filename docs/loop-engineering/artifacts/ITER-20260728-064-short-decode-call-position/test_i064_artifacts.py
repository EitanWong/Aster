from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tarfile
from pathlib import Path

ARTIFACT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ARTIFACT_DIR.parents[3]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_rejected_call_position_screen_binds_current_sources_and_archive() -> None:
    summary = json.loads((ARTIFACT_DIR / "screen-summary.json").read_text())
    assert summary["decision"] == "reject"
    assert summary["screen"]["gates"]["both_variant_medians_below_minus_3_percent"]
    assert summary["screen"]["gates"]["at_least_seven_second_calls_slower"]
    for name, digest in summary["source_sha256"].items():
        assert _sha256(ARTIFACT_DIR / name) == digest
    archive = PROJECT_ROOT / summary["archive"]["path"]
    assert _sha256(archive) == summary["archive"]["sha256"]
    with tarfile.open(archive, "r:gz") as bundle:
        assert len([member for member in bundle.getmembers() if member.isfile()]) == 8


def test_archive_only_recomputes_the_rejected_call_position_screen(tmp_path: Path) -> None:
    archive = ARTIFACT_DIR / "call-position-evidence.tar.gz"
    with tarfile.open(archive, "r:gz") as bundle:
        bundle.extractall(tmp_path, filter="data")
    aggregate_path = tmp_path / "aggregate.json"
    command = [
        sys.executable,
        str(ARTIFACT_DIR / "call_position_aggregate.py"),
    ]
    for record in sorted(tmp_path.glob("*-??-r*.json")):
        command.extend(("--record", str(record)))
    command.extend(("--output", str(aggregate_path)))
    subprocess.run(command, check=True, capture_output=True, text=True)
    aggregate = json.loads(aggregate_path.read_text())
    summary = json.loads((ARTIFACT_DIR / "screen-summary.json").read_text())
    assert aggregate["decision"] == "reject"
    for name, expected in summary["screen"].items():
        assert aggregate[name] == expected
