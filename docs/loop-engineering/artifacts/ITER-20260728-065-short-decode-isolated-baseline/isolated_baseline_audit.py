#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import tarfile
from collections import Counter
from pathlib import Path
from typing import Any

ARTIFACT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ARTIFACT_DIR.parents[3]
I061_ARTIFACT_DIR = (
    ARTIFACT_DIR.parent / "ITER-20260728-061-local-cross-engine-baseline"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _call_name(node: ast.AST) -> str | None:
    return node.id if isinstance(node, ast.Name) else None


def _timed_call_plan(preflight: Path) -> dict[str, dict[str, Any]]:
    tree = ast.parse(preflight.read_text(), filename=str(preflight))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "run_engine"
    )
    plan: dict[str, dict[str, Any]] = {}
    for engine, generate_name in (("aster", "_aster_generate"), ("mlx-lm", "_mlx_lm_generate")):
        calls = sorted(
            (
                node
                for node in ast.walk(function)
                if isinstance(node, ast.Call) and _call_name(node.func) == generate_name
            ),
            key=lambda node: node.lineno,
        )
        timed_assignments = sorted(
            (
                node
                for node in ast.walk(function)
                if isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "result"
                and isinstance(node.value, ast.Call)
                and _call_name(node.value.func) == generate_name
            ),
            key=lambda node: node.lineno,
        )
        if len(timed_assignments) != 1:
            raise RuntimeError(f"{engine} must assign exactly one timed result")
        timed_line = timed_assignments[0].lineno
        call_lines = [node.lineno for node in calls]
        if timed_line not in call_lines:
            raise RuntimeError(f"{engine} timed result is not a generation call")
        plan[engine] = {
            "generation_call_lines": call_lines,
            "generation_call_count": len(calls),
            "timed_result_line": timed_line,
            "timed_call_position": call_lines.index(timed_line) + 1,
        }
    return plan


def _read_archive_json(archive: tarfile.TarFile, name: str) -> dict[str, Any]:
    member = archive.getmember(name)
    handle = archive.extractfile(member)
    if handle is None:
        raise RuntimeError(f"archive member is unreadable: {name}")
    return json.loads(handle.read())


def _audit_archive(archive_path: Path) -> dict[str, Any]:
    with tarfile.open(archive_path, "r:gz") as archive:
        member_names = sorted(member.name for member in archive.getmembers() if member.isfile())
        manifest = _read_archive_json(archive, "manifest.json")
        aggregate = _read_archive_json(archive, "aggregate.json")
        records: list[dict[str, Any]] = []
        pair_orders: dict[str, Counter[str]] = {}
        comparisons_comparable = True
        for scenario in manifest["scenarios"]:
            scenario_name = scenario["settings"]["name"]
            orders = Counter()
            for pair in scenario["pairs"]:
                order = pair["order"]
                orders["aster_first" if order[0] == "aster" else "mlx_lm_first"] += 1
                comparison = _read_archive_json(archive, pair["comparison"]["path"])
                comparisons_comparable = comparisons_comparable and bool(comparison["comparable"])
                for engine, descriptor in pair["records"].items():
                    record = _read_archive_json(archive, descriptor["path"])
                    records.append(
                        {
                            "scenario": scenario_name,
                            "pair_number": pair["pair_number"],
                            "engine": engine,
                            "pid": record["pid"],
                            "warmup_tokens": record["settings"]["warmup_tokens"],
                            "max_tokens": record["settings"]["max_tokens"],
                            "decode_seconds": record["result"]["decode_seconds"],
                        }
                    )
                comparisons_comparable = comparisons_comparable and all(pair["gates"].values())
            pair_orders[scenario_name] = orders

    engine_counts = Counter(str(record["engine"]) for record in records)
    scenario_counts = Counter(str(record["scenario"]) for record in records)
    pids = {int(record["pid"]) for record in records}
    return {
        "archive_members": len(member_names),
        "manifest_process_isolation": bool(manifest["matrix"]["process_isolation"]),
        "aggregate_decision": aggregate["decision"],
        "comparisons_comparable": comparisons_comparable,
        "record_count": len(records),
        "unique_pid_count": len(pids),
        "engine_counts": dict(sorted(engine_counts.items())),
        "scenario_counts": dict(sorted(scenario_counts.items())),
        "pair_orders": {
            name: dict(sorted(counts.items())) for name, counts in sorted(pair_orders.items())
        },
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit I061's isolated timed-call boundary after I064 call-position evidence."
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=I061_ARTIFACT_DIR / "formal-evidence.tar.gz",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    preflight = I061_ARTIFACT_DIR / "preflight.py"
    paired_matrix = I061_ARTIFACT_DIR / "paired_matrix.py"
    final_admission = json.loads((I061_ARTIFACT_DIR / "final-admission.json").read_text())
    archive_path = args.archive.resolve()
    call_plan = _timed_call_plan(preflight)
    inventory = _audit_archive(archive_path)
    source_hashes = {
        "isolated_baseline_audit.py": _sha256(Path(__file__).resolve()),
        "preflight.py": _sha256(preflight),
        "paired_matrix.py": _sha256(paired_matrix),
    }
    gates = {
        "archive_hash_matches_i061_admission": (
            _sha256(archive_path) == final_admission["archive"]["sha256"]
        ),
        "source_hashes_match_i061_admission": (
            source_hashes["preflight.py"] == final_admission["source_sha256"]["preflight.py"]
            and source_hashes["paired_matrix.py"]
            == final_admission["source_sha256"]["paired_matrix.py"]
        ),
        "two_generation_calls_per_engine": all(
            item["generation_call_count"] == 2 for item in call_plan.values()
        ),
        "timed_call_is_second_after_warmup": all(
            item["timed_call_position"] == 2 for item in call_plan.values()
        ),
        "process_isolation_manifest": inventory["manifest_process_isolation"],
        "one_record_per_engine_process": (
            inventory["record_count"] == inventory["unique_pid_count"] == 24
        ),
        "balanced_order_per_scenario": all(
            counts == {"aster_first": 3, "mlx_lm_first": 3}
            for counts in inventory["pair_orders"].values()
        ),
        "all_archived_pairs_comparable": inventory["comparisons_comparable"],
        "aggregate_remains_admitted": inventory["aggregate_decision"] == "admit",
    }
    payload = {
        "schema_version": 1,
        "decision": "admit" if all(gates.values()) else "reject",
        "gates": gates,
        "timed_call_plan": call_plan,
        "archive": {
            "path": str(archive_path.relative_to(PROJECT_ROOT)),
            "sha256": _sha256(archive_path),
            "members": inventory["archive_members"],
        },
        "record_inventory": inventory,
        "source_sha256": source_hashes,
    }
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if payload["decision"] != "admit":
        raise SystemExit("isolated baseline audit rejected")


if __name__ == "__main__":
    main()
