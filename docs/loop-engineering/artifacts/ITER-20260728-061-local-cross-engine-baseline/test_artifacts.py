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


def test_admission_binds_the_current_evidence_sources() -> None:
    admission = json.loads((ARTIFACT_DIR / "final-admission.json").read_text())
    assert admission["decision"] == "admit"
    assert all(admission["gates"].values())
    for name, digest in admission["source_sha256"].items():
        assert _sha256(ARTIFACT_DIR / name) == digest
    archive = PROJECT_ROOT / admission["archive"]["path"]
    assert _sha256(archive) == admission["archive"]["sha256"]
    with tarfile.open(archive, "r:gz") as bundle:
        assert len([member for member in bundle.getmembers() if member.isfile()]) == 38


def test_archive_only_recomputes_the_admitted_aggregate(tmp_path: Path) -> None:
    archive = ARTIFACT_DIR / "formal-evidence.tar.gz"
    with tarfile.open(archive, "r:gz") as bundle:
        bundle.extractall(tmp_path, filter="data")
    recomputed_path = tmp_path / "recomputed.json"
    subprocess.run(
        [
            sys.executable,
            str(ARTIFACT_DIR / "aggregate.py"),
            "--manifest",
            str(tmp_path / "manifest.json"),
            "--output",
            str(recomputed_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    recomputed = json.loads(recomputed_path.read_text())
    admission = json.loads((ARTIFACT_DIR / "final-admission.json").read_text())
    assert recomputed["decision"] == "admit"
    assert all(recomputed["gates"].values())
    assert recomputed["manifest_sha256"] == admission["manifest_sha256"]
    assert [
        {
            "name": scenario["name"],
            "p50": scenario["paired_change_percent"]["p50"],
            "interval": scenario["paired_change_percent"]["bootstrap_median_interval"],
        }
        for scenario in recomputed["scenarios"]
    ] == [
        {
            "name": scenario["name"],
            "p50": scenario["paired_change_percent_p50"],
            "interval": scenario["paired_change_percent_interval"],
        }
        for scenario in admission["scenarios"]
    ]
