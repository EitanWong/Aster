#!/usr/bin/env python3
"""Report whether the Loop Engineering worktree is bounded and reviewable."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

REPO_ROOT = Path(__file__).resolve().parents[2]
STATE_PATH = Path("docs/loop-engineering/CURRENT.json")
ARTIFACT_PREFIXES = (
    "docs/loop-engineering/artifacts/",
    "docs/loop-engineering/iterations/artifacts/",
)
CACHE_ROOTS = ("aster", "scripts", "tests", "docs/loop-engineering")
ITERATION_PATTERN = re.compile(r"ITER-\d{8}-\d{3}-[a-z0-9-]+")


@dataclass(frozen=True)
class Change:
    path: str
    index_status: str
    worktree_status: str

    @property
    def staged(self) -> bool:
        return self.index_status not in {" ", "?"}

    @property
    def unstaged(self) -> bool:
        return self.worktree_status != " " and self.index_status != "?"

    @property
    def untracked(self) -> bool:
        return self.index_status == "?" and self.worktree_status == "?"


def parse_porcelain(payload: bytes) -> list[Change]:
    """Parse ``git status --porcelain=v1 -z`` without path quoting ambiguity."""
    records = payload.split(b"\0")
    changes: list[Change] = []
    index = 0
    while index < len(records):
        raw = records[index]
        index += 1
        if not raw:
            continue
        record = raw.decode("utf-8", errors="surrogateescape")
        if len(record) < 4 or record[2] != " ":
            raise ValueError(f"Unexpected git status record: {record!r}")
        index_status, worktree_status = record[:2]
        changes.append(Change(record[3:], index_status, worktree_status))
        if index_status in {"R", "C"} and index < len(records):
            index += 1
    return changes


def load_state(repo_root: Path) -> dict[str, Any]:
    state_file = repo_root / STATE_PATH
    state = json.loads(state_file.read_text(encoding="utf-8"))
    required = {"active_iteration", "phase", "workspace_policy"}
    missing = sorted(required - state.keys())
    if missing:
        raise ValueError(f"CURRENT.json is missing required fields: {', '.join(missing)}")
    if not ITERATION_PATTERN.fullmatch(str(state["active_iteration"])):
        raise ValueError("CURRENT.json has an invalid active_iteration")
    return state


def git_output(repo_root: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", *args], cwd=repo_root)


def collect_changes(repo_root: Path) -> list[Change]:
    payload = git_output(
        repo_root,
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
        "--ignore-submodules=none",
    )
    return parse_porcelain(payload)


def find_generated_caches(repo_root: Path) -> list[str]:
    caches: list[str] = []
    for relative_root in CACHE_ROOTS:
        root = repo_root / relative_root
        if not root.exists():
            continue
        for directory, names, _files in os.walk(root):
            if "__pycache__" in names:
                cache_path = Path(directory, "__pycache__")
                caches.append(cache_path.relative_to(repo_root).as_posix())
                names.remove("__pycache__")
            names[:] = [name for name in names if name not in {".git", ".venv", "results"}]
    return sorted(caches)


def path_size(repo_root: Path, relative_path: str) -> int:
    path = repo_root / relative_path
    try:
        return path.stat().st_size if path.is_file() else 0
    except OSError:
        return 0


def is_artifact(path: str) -> bool:
    return path.startswith(ARTIFACT_PREFIXES)


def build_report(
    repo_root: Path,
    state: dict[str, Any],
    changes: list[Change],
    caches: list[str],
) -> dict[str, Any]:
    policy = state["workspace_policy"]
    artifact_changes = [change for change in changes if is_artifact(change.path)]
    artifact_bytes = sum(path_size(repo_root, change.path) for change in artifact_changes)
    largest_artifacts = sorted(
        (
            {"path": change.path, "bytes": path_size(repo_root, change.path)}
            for change in artifact_changes
        ),
        key=lambda item: int(item["bytes"]),
        reverse=True,
    )[:10]
    artifact_iterations = {
        match.group(0)
        for change in artifact_changes
        if (match := ITERATION_PATTERN.search(change.path)) is not None
    }
    active_iteration = str(state["active_iteration"])
    active_artifact_prefixes = tuple(f"{prefix}{active_iteration}/" for prefix in ARTIFACT_PREFIXES)
    active_artifact_changes = [
        change for change in artifact_changes if change.path.startswith(active_artifact_prefixes)
    ]
    active_artifact_bytes = sum(
        path_size(repo_root, change.path) for change in active_artifact_changes
    )
    foreign_iterations = sorted(artifact_iterations - {active_iteration})
    reference_changes = [change.path for change in changes if change.path.startswith("examples/")]
    mixed_changes = [change.path for change in changes if change.staged and change.unstaged]

    issues: list[dict[str, str]] = []

    def add_issue(code: str, severity: str, message: str) -> None:
        issues.append({"code": code, "severity": severity, "message": message})

    debt_baseline = policy.get("debt_baseline")
    changed_growth = 0
    artifact_growth = 0
    if isinstance(debt_baseline, dict):
        typed_baseline = cast(dict[str, Any], debt_baseline)
        baseline_changed = typed_baseline.get("changed_files")
        baseline_artifacts = typed_baseline.get("artifact_files")
        if (
            not isinstance(baseline_changed, int)
            or isinstance(baseline_changed, bool)
            or not isinstance(baseline_artifacts, int)
            or isinstance(baseline_artifacts, bool)
        ):
            raise ValueError("workspace debt baseline counts must be integers")
        changed_growth = max(0, len(changes) - baseline_changed)
        artifact_growth = max(0, len(artifact_changes) - baseline_artifacts)
        if changed_growth > int(policy["max_changed_file_growth"]):
            add_issue(
                "changed-file-growth",
                "blocker",
                (
                    f"{changed_growth} files were added above the inherited debt baseline, "
                    f"exceeding the {policy['max_changed_file_growth']} file growth budget."
                ),
            )
        if artifact_growth > int(policy["max_artifact_file_growth"]):
            add_issue(
                "artifact-file-growth",
                "blocker",
                (
                    f"{artifact_growth} artifacts were added above the inherited debt "
                    f"baseline, exceeding the {policy['max_artifact_file_growth']} file "
                    "growth budget."
                ),
            )
        if len(changes) > int(policy["max_changed_files"]) or len(artifact_changes) > int(
            policy["max_tracked_artifact_files"]
        ):
            add_issue(
                "inherited-workspace-debt",
                "warning",
                (
                    f"Workspace debt remains at {len(changes)} changed files and "
                    f"{len(artifact_changes)} artifact files; growth is checked against "
                    f"the recorded {baseline_changed}/{baseline_artifacts} baseline."
                ),
            )
        if len(active_artifact_changes) > int(policy["max_active_iteration_artifact_files"]):
            add_issue(
                "active-artifact-file-budget",
                "blocker",
                (
                    f"The active iteration owns {len(active_artifact_changes)} artifact "
                    f"files, above its {policy['max_active_iteration_artifact_files']} "
                    "file budget."
                ),
            )
        if active_artifact_bytes > int(policy["max_active_iteration_artifact_bytes"]):
            add_issue(
                "active-artifact-byte-budget",
                "blocker",
                (
                    f"The active iteration owns {active_artifact_bytes} artifact bytes, "
                    f"above its {policy['max_active_iteration_artifact_bytes']} byte "
                    "budget."
                ),
            )
    else:
        if len(changes) > int(policy["max_changed_files"]):
            add_issue(
                "changed-file-budget",
                "blocker",
                (
                    f"{len(changes)} changed files exceed the "
                    f"{policy['max_changed_files']} file budget."
                ),
            )
        if len(artifact_changes) > int(policy["max_tracked_artifact_files"]):
            add_issue(
                "artifact-file-budget",
                "blocker",
                (
                    f"{len(artifact_changes)} changed artifact files exceed the "
                    f"{policy['max_tracked_artifact_files']} file budget."
                ),
            )
    if artifact_bytes > int(policy["max_tracked_artifact_bytes"]):
        add_issue(
            "artifact-byte-budget",
            "blocker",
            (
                f"Changed artifacts use {artifact_bytes} bytes, above the "
                f"{policy['max_tracked_artifact_bytes']} byte budget."
            ),
        )
    if foreign_iterations:
        add_issue(
            "foreign-iteration-artifacts",
            "warning",
            "Changed artifacts span other iterations: " + ", ".join(foreign_iterations),
        )
    if reference_changes:
        add_issue(
            "reference-updates",
            "warning",
            f"{len(reference_changes)} reference project paths are changed.",
        )
    if mixed_changes:
        add_issue(
            "mixed-index-state",
            "warning",
            f"{len(mixed_changes)} paths have both staged and unstaged changes.",
        )
    if caches:
        add_issue(
            "generated-caches",
            "warning",
            f"{len(caches)} generated __pycache__ directories remain in owned paths.",
        )

    blockers = sum(issue["severity"] == "blocker" for issue in issues)
    warnings = sum(issue["severity"] == "warning" for issue in issues)
    health = "fail" if blockers else "warn" if warnings else "pass"
    return {
        "schema_version": 1,
        "health": health,
        "head": git_output(repo_root, "rev-parse", "--short=12", "HEAD").decode("ascii").strip(),
        "active_iteration": active_iteration,
        "phase": state["phase"],
        "counts": {
            "changed": len(changes),
            "staged": sum(change.staged for change in changes),
            "unstaged": sum(change.unstaged for change in changes),
            "untracked": sum(change.untracked for change in changes),
            "mixed": len(mixed_changes),
            "artifact_files": len(artifact_changes),
            "artifact_bytes": artifact_bytes,
            "active_artifact_files": len(active_artifact_changes),
            "active_artifact_bytes": active_artifact_bytes,
            "changed_growth_from_debt_baseline": changed_growth,
            "artifact_growth_from_debt_baseline": artifact_growth,
            "reference_paths": len(reference_changes),
            "generated_caches": len(caches),
        },
        "foreign_artifact_iterations": foreign_iterations,
        "largest_changed_artifacts": largest_artifacts,
        "generated_caches": caches,
        "issues": issues,
        "changes": [asdict(change) for change in changes],
    }


def print_text_report(report: dict[str, Any]) -> None:
    counts = report["counts"]
    print(f"Workspace health: {report['health'].upper()}")
    print(f"HEAD: {report['head']}")
    print(f"Active: {report['active_iteration']} ({report['phase']})")
    print(
        "Changes: "
        f"{counts['changed']} total, {counts['staged']} staged, "
        f"{counts['unstaged']} unstaged, {counts['untracked']} untracked"
    )
    print(
        "Artifacts: "
        f"{counts['artifact_files']} files, "
        f"{counts['artifact_bytes'] / (1024 * 1024):.2f} MiB changed"
    )
    print(
        "Active iteration: "
        f"{counts['active_artifact_files']} artifact files, "
        f"{counts['active_artifact_bytes'] / (1024 * 1024):.2f} MiB; "
        f"debt growth {counts['changed_growth_from_debt_baseline']} files / "
        f"{counts['artifact_growth_from_debt_baseline']} artifacts"
    )
    print(
        "Other: "
        f"{counts['reference_paths']} reference paths, "
        f"{counts['mixed']} mixed-index paths, "
        f"{counts['generated_caches']} generated caches"
    )
    if report["issues"]:
        print("Issues:")
        for issue in report["issues"]:
            print(f"  [{issue['severity']}] {issue['code']}: {issue['message']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=REPO_ROOT)
    parser.add_argument("--json", action="store_true", help="Emit the full JSON report")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return non-zero when the report contains a blocker",
    )
    args = parser.parse_args()
    repo_root = args.repo.resolve()

    try:
        state = load_state(repo_root)
        changes = collect_changes(repo_root)
        report = build_report(repo_root, state, changes, find_generated_caches(repo_root))
    except (OSError, ValueError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        print(f"Workspace check failed: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print_text_report(report)
    return int(args.strict and report["health"] == "fail")


if __name__ == "__main__":
    sys.exit(main())
