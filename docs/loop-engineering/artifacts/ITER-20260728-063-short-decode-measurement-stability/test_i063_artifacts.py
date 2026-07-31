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


def test_rejected_stability_screen_binds_current_sources_and_archive() -> None:
    summary = json.loads((ARTIFACT_DIR / "screen-summary.json").read_text())
    assert summary["decision"] == "reject"
    assert summary["screen"]["gates"]["all_matched_measurement_order_contrasts_positive"]
    for name, digest in summary["source_sha256"].items():
        assert _sha256(ARTIFACT_DIR / name) == digest
    archive = PROJECT_ROOT / summary["archive"]["path"]
    assert _sha256(archive) == summary["archive"]["sha256"]
    with tarfile.open(archive, "r:gz") as bundle:
        names = {member.name for member in bundle.getmembers() if member.isfile()}
    assert len(names) == 14
    assert {
        "serial-first-no-clear.json",
        "pipeline-first-no-clear.json",
        "serial-first-clear.json",
        "pipeline-first-clear.json",
        "serial-first-clear-gc.json",
        "pipeline-first-clear-gc.json",
    } <= names


def test_archive_only_recomputes_the_rejected_stability_screen(tmp_path: Path) -> None:
    archive = ARTIFACT_DIR / "measurement-stability-evidence.tar.gz"
    with tarfile.open(archive, "r:gz") as bundle:
        bundle.extractall(tmp_path, filter="data")
    aggregate_path = tmp_path / "aggregate.json"
    command = [
        sys.executable,
        str(ARTIFACT_DIR / "warmup_order_aggregate.py"),
    ]
    for record in sorted(tmp_path.glob("warmup-*.json")):
        command.extend(("--record", str(record)))
    command.extend(("--output", str(aggregate_path)))
    subprocess.run(command, check=True, capture_output=True, text=True)
    aggregate = json.loads(aggregate_path.read_text())
    summary = json.loads((ARTIFACT_DIR / "screen-summary.json").read_text())
    assert aggregate["decision"] == "reject"
    for name, expected in summary["screen"].items():
        assert aggregate[name] == expected
