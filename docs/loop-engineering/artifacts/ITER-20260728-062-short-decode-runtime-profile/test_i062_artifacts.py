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


def test_rejected_screen_binds_current_sources_and_archive() -> None:
    summary = json.loads((ARTIFACT_DIR / "screen-summary.json").read_text())
    assert summary["decision"] == "reject"
    for name, digest in summary["source_sha256"].items():
        assert _sha256(ARTIFACT_DIR / name) == digest
    archive = PROJECT_ROOT / summary["archive"]["path"]
    assert _sha256(archive) == summary["archive"]["sha256"]
    with tarfile.open(archive, "r:gz") as bundle:
        assert len([member for member in bundle.getmembers() if member.isfile()]) == 13


def test_archive_only_recomputes_the_inconclusive_pipeline_screen(tmp_path: Path) -> None:
    archive = ARTIFACT_DIR / "rejected-screen-evidence.tar.gz"
    with tarfile.open(archive, "r:gz") as bundle:
        bundle.extractall(tmp_path, filter="data")
    aggregate_path = tmp_path / "aggregate.json"
    command = [
        sys.executable,
        str(ARTIFACT_DIR / "pipeline_pair_aggregate.py"),
    ]
    for index in range(1, 7):
        command.extend(("--record", str(tmp_path / f"pair-r{index}.json")))
    command.extend(("--output", str(aggregate_path)))
    subprocess.run(command, check=True, capture_output=True, text=True)
    aggregate = json.loads(aggregate_path.read_text())
    summary = json.loads((ARTIFACT_DIR / "screen-summary.json").read_text())
    assert aggregate["decision"] == "inconclusive"
    assert aggregate["gates"] == summary["pipeline_screen"]["gates"]
    assert (
        aggregate["bootstrap_median_interval"]
        == summary["pipeline_screen"]["bootstrap_median_interval"]
    )
