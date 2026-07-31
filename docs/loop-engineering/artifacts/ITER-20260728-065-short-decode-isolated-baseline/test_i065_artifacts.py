from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ARTIFACT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ARTIFACT_DIR.parents[3]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_isolated_baseline_audit_binds_current_sources_and_i061_archive() -> None:
    audit = json.loads((ARTIFACT_DIR / "isolated-baseline-audit.json").read_text())
    assert audit["decision"] == "admit"
    assert all(audit["gates"].values())
    assert audit["record_inventory"]["record_count"] == 24
    assert audit["record_inventory"]["unique_pid_count"] == 24
    for name, digest in audit["source_sha256"].items():
        source = ARTIFACT_DIR / name if name == "isolated_baseline_audit.py" else (
            PROJECT_ROOT
            / "docs/loop-engineering/artifacts/ITER-20260728-061-local-cross-engine-baseline"
            / name
        )
        assert _sha256(source) == digest
    assert _sha256(PROJECT_ROOT / audit["archive"]["path"]) == audit["archive"]["sha256"]


def test_i061_archive_recomputes_the_isolated_boundary(tmp_path: Path) -> None:
    output = tmp_path / "audit.json"
    command = [
        sys.executable,
        str(ARTIFACT_DIR / "isolated_baseline_audit.py"),
        "--archive",
        str(
            PROJECT_ROOT
            / "docs/loop-engineering/artifacts/ITER-20260728-061-local-cross-engine-baseline/formal-evidence.tar.gz"
        ),
        "--output",
        str(output),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    recomputed = json.loads(output.read_text())
    audit = json.loads((ARTIFACT_DIR / "isolated-baseline-audit.json").read_text())
    assert recomputed["decision"] == "admit"
    assert recomputed["gates"] == audit["gates"]
    assert recomputed["timed_call_plan"] == audit["timed_call_plan"]
    assert recomputed["record_inventory"] == audit["record_inventory"]
