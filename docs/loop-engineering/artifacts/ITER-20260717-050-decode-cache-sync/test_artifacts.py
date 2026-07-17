from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

ARTIFACT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ARTIFACT_DIR.parents[3]
if str(ARTIFACT_DIR) not in sys.path:
    sys.path.insert(0, str(ARTIFACT_DIR))

import aggregate as screen_module  # noqa: E402
import confirm_aggregate as confirmation_module  # noqa: E402
import long_stress_aggregate as long_module  # noqa: E402
import production_aggregate as production_module  # noqa: E402
import token_budget_long_aggregate as token_long_module  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ARTIFACT_DIR / candidate


def _assert_manifest(path: Path) -> None:
    manifest = _load(path)
    records = manifest["records"]
    assert records
    assert len({int(record["pid"]) for record in records}) == len(records)
    for record in records:
        output = _resolve(record["output"])
        assert output.is_file()
        assert _sha256(output) == record["sha256"]
        assert int(_load(output)["pid"]) == int(record["pid"])


def _assert_recomputed(
    output: Path,
    function: Callable[..., dict[str, Any]],
    *args: Any,
    **kwargs: Any,
) -> None:
    archived = _load(output)
    recomputed = function(*args, **kwargs)
    assert recomputed == archived
    json.dumps(recomputed, allow_nan=False)


def test_screen_manifest_is_hash_bound_and_process_isolated() -> None:
    _assert_manifest(ARTIFACT_DIR / "results/screen/execution-manifest.json")


def test_confirmation_manifest_is_hash_bound_and_process_isolated() -> None:
    _assert_manifest(ARTIFACT_DIR / "results/confirmation/execution-manifest.json")


def test_long_stress_manifest_is_hash_bound_and_process_isolated() -> None:
    _assert_manifest(ARTIFACT_DIR / "results/long-stress/execution-manifest.json")


def test_token_budget_confirmation_manifest_is_hash_bound_and_process_isolated() -> None:
    _assert_manifest(
        ARTIFACT_DIR / "results/token-budget-confirmation/execution-manifest.json"
    )


def test_token_budget_long_manifest_is_hash_bound_and_process_isolated() -> None:
    _assert_manifest(ARTIFACT_DIR / "results/token-budget-long/execution-manifest.json")


def test_production_manifest_is_hash_bound_and_process_isolated() -> None:
    _assert_manifest(ARTIFACT_DIR / "results/production/execution-manifest.json")


def test_screen_aggregate_recomputes_exactly() -> None:
    manifest = ARTIFACT_DIR / "results/screen/execution-manifest.json"
    output = ARTIFACT_DIR / "results/screen/aggregate.json"
    _assert_recomputed(output, screen_module.aggregate, manifest)


def test_confirmation_aggregate_recomputes_exactly() -> None:
    manifest = ARTIFACT_DIR / "results/confirmation/execution-manifest.json"
    output = ARTIFACT_DIR / "results/confirmation/aggregate.json"
    _assert_recomputed(output, confirmation_module.aggregate, manifest)


def test_token_budget_confirmation_aggregate_recomputes_exactly() -> None:
    manifest = ARTIFACT_DIR / "results/token-budget-confirmation/execution-manifest.json"
    output = ARTIFACT_DIR / "results/token-budget-confirmation/aggregate.json"
    _assert_recomputed(
        output,
        confirmation_module.aggregate,
        manifest,
        candidate_policy="periodic-token-512",
    )


def test_fixed_step_long_aggregate_preserves_rejected_batch_gate() -> None:
    manifest = ARTIFACT_DIR / "results/long-stress/execution-manifest.json"
    output = ARTIFACT_DIR / "results/long-stress/aggregate.json"
    _assert_recomputed(output, long_module.aggregate, manifest)
    assert _load(output)["admission"]["all_long_stress_cells_passed"] is False


def test_token_budget_long_aggregate_recomputes_exactly() -> None:
    manifest = ARTIFACT_DIR / "results/token-budget-long/execution-manifest.json"
    output = ARTIFACT_DIR / "results/token-budget-long/aggregate.json"
    _assert_recomputed(output, token_long_module.aggregate, manifest)
    assert _load(output)["production_integration_ready"] is True


def test_production_aggregate_recomputes_exactly() -> None:
    production = ARTIFACT_DIR / "results/production/execution-manifest.json"
    confirmation = ARTIFACT_DIR / "results/confirmation/execution-manifest.json"
    token_budget = (
        ARTIFACT_DIR / "results/token-budget-confirmation/execution-manifest.json"
    )
    token_budget_long_manifest = (
        ARTIFACT_DIR / "results/token-budget-long/execution-manifest.json"
    )
    token_budget_long_aggregate = (
        ARTIFACT_DIR / "results/token-budget-long/aggregate.json"
    )
    output = ARTIFACT_DIR / "results/production/aggregate.json"
    _assert_recomputed(
        output,
        production_module.aggregate,
        production,
        confirmation,
        token_budget,
        token_budget_long_manifest,
        token_budget_long_aggregate,
    )
    assert _load(output)["integration_approved"] is True


def test_production_admission_requires_token_budget_long_stress(tmp_path: Path) -> None:
    production = ARTIFACT_DIR / "results/production/execution-manifest.json"
    confirmation = ARTIFACT_DIR / "results/confirmation/execution-manifest.json"
    token_budget = (
        ARTIFACT_DIR / "results/token-budget-confirmation/execution-manifest.json"
    )
    token_budget_long_manifest = (
        ARTIFACT_DIR / "results/token-budget-long/execution-manifest.json"
    )
    archived = _load(ARTIFACT_DIR / "results/token-budget-long/aggregate.json")
    damaged_results = (
        {**archived, "production_integration_ready": False},
        {**archived, "gate": {}},
        {
            **archived,
            "decode_speedup_percent": float(archived["decode_speedup_percent"]) + 1.0,
        },
    )
    for index, damaged in enumerate(damaged_results):
        failed_long = tmp_path / f"failed-long-stress-{index}.json"
        failed_long.write_text(json.dumps(damaged))

        result = production_module.aggregate(
            production,
            confirmation,
            token_budget,
            token_budget_long_manifest,
            failed_long,
        )

        assert result["all_production_bridge_cells_passed"] is True
        assert result["admission"]["token_budget_long_stress_passed"] is False
        assert result["integration_approved"] is False


def test_synthetic_stress_covers_10000_steps_with_exact_parity() -> None:
    payload = _load(ARTIFACT_DIR / "results/synthetic-stress.json")
    assert payload["steps"] == 10_000
    assert payload["all_parity"] is True
    assert payload["swap_after_bytes"] - payload["swap_before_bytes"] == 0
    assert set(payload["results"]) == {
        "native_kv_waw",
        "recurrent_sibling_raw",
        "paged_pool_waw",
    }
    for case in payload["results"].values():
        assert all(case["parity"].values())


def test_production_records_bind_the_current_runtime_source() -> None:
    manifest = _load(ARTIFACT_DIR / "results/production/execution-manifest.json")
    relative = "aster/inference/model_runner.py"
    assert manifest["source_sha256"][relative] == _sha256(PROJECT_ROOT / relative)


def test_production_policy_counts_scale_with_batch_tokens() -> None:
    manifest = _load(ARTIFACT_DIR / "results/production/execution-manifest.json")
    max_tokens = int(manifest["matrix"]["max_tokens"])
    for record in manifest["records"]:
        payload = _load(_resolve(record["output"]))
        expected = max_tokens * int(payload["batch_size"]) // 512
        assert int(payload["policy_metrics"]["clear_executed"]) == expected
        assert int(payload["policy_metrics"]["clear_failures"]) == 0
