from __future__ import annotations

from pathlib import Path

import pytest

from scripts.dev.check_loop_workspace import Change, build_report, parse_porcelain


def test_parse_porcelain_preserves_staged_unstaged_and_untracked_state() -> None:
    changes = parse_porcelain(b"M  staged.py\0 M unstaged.py\0MM mixed.py\0?? new.py\0")

    assert changes == [
        Change("staged.py", "M", " "),
        Change("unstaged.py", " ", "M"),
        Change("mixed.py", "M", "M"),
        Change("new.py", "?", "?"),
    ]
    assert [change.staged for change in changes] == [True, False, True, False]
    assert [change.unstaged for change in changes] == [False, True, True, False]
    assert [change.untracked for change in changes] == [False, False, False, True]


def test_parse_porcelain_skips_rename_source_record() -> None:
    changes = parse_porcelain(b"R  new-name.py\0old-name.py\0")

    assert changes == [Change("new-name.py", "R", " ")]


def test_build_report_enforces_artifact_and_change_budgets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = Path(
        "docs/loop-engineering/artifacts/ITER-20260724-059-structured-residual-profile/run.json"
    )
    artifact_path = tmp_path / artifact
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_bytes(b"12345")
    state = {
        "active_iteration": "ITER-20260724-059-structured-residual-profile",
        "phase": "screen",
        "workspace_policy": {
            "max_changed_files": 1,
            "max_tracked_artifact_files": 0,
            "max_tracked_artifact_bytes": 4,
        },
    }
    changes = [Change(artifact.as_posix(), "?", "?"), Change("aster/runtime.py", "M", " ")]

    def fake_git_output(*_args: object) -> bytes:
        return b"2cb1405\n"

    monkeypatch.setattr("scripts.dev.check_loop_workspace.git_output", fake_git_output)
    report = build_report(tmp_path, state, changes, [])

    assert report["health"] == "fail"
    assert report["counts"]["artifact_bytes"] == 5
    assert {issue["code"] for issue in report["issues"]} == {
        "changed-file-budget",
        "artifact-byte-budget",
        "artifact-file-budget",
    }


def test_build_report_flags_foreign_iterations_and_workspace_noise(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = {
        "active_iteration": "ITER-20260724-059-structured-residual-profile",
        "phase": "screen",
        "workspace_policy": {
            "max_changed_files": 10,
            "max_tracked_artifact_files": 10,
            "max_tracked_artifact_bytes": 1024,
        },
    }
    changes = [
        Change(
            "docs/loop-engineering/artifacts/ITER-20260724-058-structured-mask-cache/a.json",
            "M",
            "M",
        ),
        Change("examples/mlx", " ", "M"),
    ]

    def fake_git_output(*_args: object) -> bytes:
        return b"2cb1405\n"

    monkeypatch.setattr("scripts.dev.check_loop_workspace.git_output", fake_git_output)
    report = build_report(tmp_path, state, changes, ["scripts/dev/__pycache__"])

    assert report["health"] == "warn"
    assert report["foreign_artifact_iterations"] == ["ITER-20260724-058-structured-mask-cache"]
    assert {issue["code"] for issue in report["issues"]} == {
        "foreign-iteration-artifacts",
        "generated-caches",
        "mixed-index-state",
        "reference-updates",
    }


def test_build_report_keeps_inherited_debt_visible_without_blocking_bounded_growth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = Path(
        "docs/loop-engineering/artifacts/ITER-20260724-059-structured-residual-profile/result.json"
    )
    artifact_path = tmp_path / artifact
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_bytes(b"12345")
    state = {
        "active_iteration": "ITER-20260724-059-structured-residual-profile",
        "phase": "consolidate",
        "workspace_policy": {
            "max_changed_files": 1,
            "max_tracked_artifact_files": 0,
            "max_tracked_artifact_bytes": 1024,
            "debt_baseline": {"changed_files": 2, "artifact_files": 1},
            "max_changed_file_growth": 1,
            "max_artifact_file_growth": 1,
            "max_active_iteration_artifact_files": 2,
            "max_active_iteration_artifact_bytes": 10,
        },
    }
    changes = [Change(artifact.as_posix(), "?", "?"), Change("aster/runtime.py", "M", " ")]

    def fake_git_output(*_args: object) -> bytes:
        return b"2cb1405\n"

    monkeypatch.setattr("scripts.dev.check_loop_workspace.git_output", fake_git_output)
    report = build_report(tmp_path, state, changes, [])

    assert report["health"] == "warn"
    assert report["counts"]["changed_growth_from_debt_baseline"] == 0
    assert report["counts"]["artifact_growth_from_debt_baseline"] == 0
    assert report["counts"]["active_artifact_files"] == 1
    assert {issue["code"] for issue in report["issues"]} == {"inherited-workspace-debt"}


def test_build_report_blocks_growth_beyond_inherited_debt_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = Path(
        "docs/loop-engineering/artifacts/ITER-20260724-059-structured-residual-profile/result.json"
    )
    artifact_path = tmp_path / artifact
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_bytes(b"12345")
    state = {
        "active_iteration": "ITER-20260724-059-structured-residual-profile",
        "phase": "consolidate",
        "workspace_policy": {
            "max_changed_files": 1,
            "max_tracked_artifact_files": 0,
            "max_tracked_artifact_bytes": 1024,
            "debt_baseline": {"changed_files": 1, "artifact_files": 0},
            "max_changed_file_growth": 0,
            "max_artifact_file_growth": 0,
            "max_active_iteration_artifact_files": 2,
            "max_active_iteration_artifact_bytes": 10,
        },
    }
    changes = [Change(artifact.as_posix(), "?", "?"), Change("aster/runtime.py", "M", " ")]

    def fake_git_output(*_args: object) -> bytes:
        return b"2cb1405\n"

    monkeypatch.setattr("scripts.dev.check_loop_workspace.git_output", fake_git_output)
    report = build_report(tmp_path, state, changes, [])

    assert report["health"] == "fail"
    assert {issue["code"] for issue in report["issues"]} == {
        "artifact-file-growth",
        "changed-file-growth",
        "inherited-workspace-debt",
    }
