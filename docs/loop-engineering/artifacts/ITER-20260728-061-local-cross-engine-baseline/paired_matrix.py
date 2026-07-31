#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ARTIFACT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ARTIFACT_DIR.parents[3]
PREFLIGHT = ARTIFACT_DIR / "preflight.py"

SCENARIOS = (
    {
        "name": "short-decode",
        "context_words": 128,
        "max_tokens": 256,
        "warmup_tokens": 32,
    },
    {
        "name": "long-prefill",
        "context_words": 2048,
        "max_tokens": 64,
        "warmup_tokens": 32,
    },
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _run(command: list[str]) -> None:
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "child benchmark failed: "
            + " ".join(command)
            + "\n"
            + completed.stderr.strip()
        )


def _record_descriptor(path: Path, root: Path) -> dict[str, str]:
    return {
        "path": str(path.relative_to(root)),
        "sha256": _sha256(path),
    }


def _run_engine(
    engine: str,
    scenario: dict[str, Any],
    output: Path,
) -> dict[str, str]:
    _run(
        [
            sys.executable,
            str(PREFLIGHT),
            "run",
            "--engine",
            engine,
            "--context-words",
            str(scenario["context_words"]),
            "--max-tokens",
            str(scenario["max_tokens"]),
            "--warmup-tokens",
            str(scenario["warmup_tokens"]),
            "--output",
            str(output),
        ]
    )
    return _record_descriptor(output, output.parents[1])


def _run_pair(
    run_root: Path,
    scenario: dict[str, Any],
    pair_number: int,
) -> dict[str, Any]:
    scenario_name = str(scenario["name"])
    order = ("aster", "mlx-lm") if pair_number % 2 else ("mlx-lm", "aster")
    records: dict[str, dict[str, str]] = {}
    record_paths: dict[str, Path] = {}
    for engine in order:
        path = run_root / "records" / f"{scenario_name}-pair-{pair_number:02d}-{engine}.json"
        records[engine] = _run_engine(engine, scenario, path)
        record_paths[engine] = path

    comparison_path = run_root / "comparisons" / f"{scenario_name}-pair-{pair_number:02d}.json"
    _run(
        [
            sys.executable,
            str(PREFLIGHT),
            "compare",
            "--aster",
            str(record_paths["aster"]),
            "--reference",
            str(record_paths["mlx-lm"]),
            "--output",
            str(comparison_path),
        ]
    )
    comparison = _record_descriptor(comparison_path, run_root)
    comparison_payload = json.loads(comparison_path.read_text())
    return {
        "pair_number": pair_number,
        "order": list(order),
        "records": records,
        "comparison": comparison,
        "comparable": comparison_payload["comparable"],
        "gates": comparison_payload["gates"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the frozen I061 local Aster/MLX-LM paired baseline matrix."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=6)
    parser.add_argument(
        "--scenario",
        choices=[scenario["name"] for scenario in SCENARIOS],
        help="Run one frozen scenario so each orchestration process remains bounded.",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append a distinct scenario to an existing manifest with the same matrix settings.",
    )
    parser.add_argument(
        "--pair-number",
        type=int,
        help="Run one predeclared AB/BA pair and atomically extend its scenario manifest.",
    )
    args = parser.parse_args()

    if args.repetitions < 4 or args.repetitions % 2:
        raise SystemExit("--repetitions must be an even number of at least four")
    if args.pair_number is not None and not 1 <= args.pair_number <= args.repetitions:
        raise SystemExit("--pair-number must be within the configured repetition range")
    run_root = args.run_dir.resolve()
    manifest_path = run_root / "manifest.json"
    existing_scenarios: list[dict[str, Any]] = []
    existing_failures: list[str] = []
    if run_root.exists() and any(run_root.iterdir()):
        if not args.append or not manifest_path.is_file():
            raise SystemExit(f"run directory must be empty: {run_root}")
        existing_manifest = json.loads(manifest_path.read_text())
        if existing_manifest["matrix"]["repetitions"] != args.repetitions:
            raise SystemExit("appended matrix must retain the original repetition count")
        if existing_manifest["source_sha256"] != {
            "preflight.py": _sha256(PREFLIGHT),
            "paired_matrix.py": _sha256(Path(__file__).resolve()),
        }:
            raise SystemExit("appended matrix source drifted after the first scenario")
        existing_scenarios = existing_manifest["scenarios"]
        existing_failures = existing_manifest["failed_comparisons"]
    else:
        run_root.mkdir(parents=True, exist_ok=True)

    selected_scenarios = tuple(
        scenario for scenario in SCENARIOS if args.scenario in (None, scenario["name"])
    )
    existing_names = {scenario["settings"]["name"] for scenario in existing_scenarios}
    selected_names = {scenario["name"] for scenario in selected_scenarios}
    duplicate_names = existing_names & selected_names
    if duplicate_names and args.pair_number is None:
        raise SystemExit("scenario already present: " + ", ".join(sorted(duplicate_names)))

    scenarios_by_name = {
        scenario["settings"]["name"]: scenario for scenario in existing_scenarios
    }
    failures = list(existing_failures)
    for scenario in selected_scenarios:
        pair_numbers = (
            [args.pair_number]
            if args.pair_number is not None
            else list(range(1, args.repetitions + 1))
        )
        pairs = [
            _run_pair(run_root, scenario, pair_number)
            for pair_number in pair_numbers
        ]
        existing = scenarios_by_name.get(scenario["name"])
        if existing is not None:
            existing_pair_numbers = {pair["pair_number"] for pair in existing["pairs"]}
            repeated = existing_pair_numbers & {pair["pair_number"] for pair in pairs}
            if repeated:
                raise SystemExit(
                    "pair already present: " + ", ".join(map(str, sorted(repeated)))
                )
            pairs = [*existing["pairs"], *pairs]
        for pair in pairs:
            if not pair["comparable"]:
                failure = f"{scenario['name']} pair {pair['pair_number']}"
                if failure not in failures:
                    failures.append(failure)
        scenarios_by_name[scenario["name"]] = {"settings": scenario, "pairs": pairs}

    scenarios = [
        scenarios_by_name[scenario["name"]]
        for scenario in SCENARIOS
        if scenario["name"] in scenarios_by_name
    ]

    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(UTC).isoformat(),
        "project_root": str(PROJECT_ROOT),
        "matrix": {
            "repetitions": args.repetitions,
            "order_balance": {
                "aster_first": args.repetitions // 2,
                "mlx_lm_first": args.repetitions // 2,
            },
            "process_isolation": True,
            "reference_engine": "mlx-lm",
        },
        "source_sha256": {
            "preflight.py": _sha256(PREFLIGHT),
            "paired_matrix.py": _sha256(Path(__file__).resolve()),
        },
        "scenarios": scenarios,
        "failed_comparisons": failures,
    }
    _write_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    if failures:
        raise SystemExit("non-comparable pairs: " + ", ".join(failures))


if __name__ == "__main__":
    main()
