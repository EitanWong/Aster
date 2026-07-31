from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts/dev/public_engine_matrix.py"


def load_tool() -> ModuleType:
    spec = importlib.util.spec_from_file_location("public_engine_matrix", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def workload_record(workload_id: str, source: dict[str, Any]) -> dict[str, Any]:
    return {
        "workload_id": workload_id,
        "source": source,
        "prompt": {"sha256": "a" * 64},
        "max_tokens": 8,
    }


def shard_payload(
    tool: ModuleType,
    *,
    engine: str,
    workload_sha256: str,
    source_lock_sha256: str,
    record: dict[str, Any],
    contract: dict[str, Any],
    fingerprint: dict[str, str],
) -> dict[str, Any]:
    return {
        "kind": "public-engine-result-shard",
        "engine": engine,
        "engine_version": "test",
        "workload_sha256": workload_sha256,
        "source_lock_sha256": source_lock_sha256,
        "generation": {"temperature": 0.0},
        "execution": contract,
        "model_fingerprint": fingerprint,
        "shard": {"key": tool._workload_shard_key(record)},
        "records": [{"workload_id": record["workload_id"]}],
    }


def test_shards_preserve_public_task_order_and_official_head_tail_policy() -> None:
    tool = load_tool()
    workload = {
        "records": [
            workload_record("mt-bench:1:turn-1", {"id": "mt-bench-question"}),
            workload_record(
                "longbench:qasper:one",
                {"id": "longbench-v1-data", "dataset": "qasper"},
            ),
            workload_record(
                "longbench:qasper:two",
                {"id": "longbench-v1-data", "dataset": "qasper"},
            ),
            workload_record(
                "longbench:lcc:one",
                {"id": "longbench-v1-data", "dataset": "lcc"},
            ),
        ]
    }

    assert list(tool.workload_shards(workload)) == [
        "mt-bench",
        "longbench-qasper",
        "longbench-lcc",
    ]
    assert tool._trim_input_tokens([0, 1, 2, 3, 4, 5], 4) == ([0, 1, 4, 5], True)
    assert tool._trim_input_tokens([0, 1], 4) == ([0, 1], False)
    with pytest.raises(tool.MatrixError, match="even"):
        tool._require_even_positive(3, "--max-input-tokens")


def test_aggregate_result_restores_manifest_order(tmp_path: Path) -> None:
    tool = load_tool()
    first = workload_record("mt-bench:1:turn-1", {"id": "mt-bench-question"})
    second = workload_record(
        "longbench:qasper:one", {"id": "longbench-v1-data", "dataset": "qasper"}
    )
    workload = {
        "kind": "public-cross-engine-workload",
        "generation": {"temperature": 0.0},
        "records": [first, second],
    }
    workload_path = tmp_path / "workload.json"
    workload_path.write_text(json.dumps(workload))
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    contract = {"input_truncation_policy": tool.TRUNCATION_POLICY}
    fingerprint = {"model_sha256": "m" * 64, "tokenizer_sha256": "t" * 64}
    source_lock_sha256 = "l" * 64
    first_path = output_dir / "first.json"
    second_path = output_dir / "second.json"
    first_payload = shard_payload(
        tool,
        engine="aster",
        workload_sha256=sha256(workload_path),
        source_lock_sha256=source_lock_sha256,
        record=first,
        contract=contract,
        fingerprint=fingerprint,
    )
    second_payload = shard_payload(
        tool,
        engine="aster",
        workload_sha256=sha256(workload_path),
        source_lock_sha256=source_lock_sha256,
        record=second,
        contract=contract,
        fingerprint=fingerprint,
    )
    first_path.write_text(json.dumps(first_payload))
    second_path.write_text(json.dumps(second_payload))

    aggregate = tool._aggregate_engine_result(
        workload,
        workload_path,
        "aster",
        [second_payload, first_payload],
        [second_path, first_path],
        output_dir,
    )

    assert [row["workload_id"] for row in aggregate["records"]] == [
        "mt-bench:1:turn-1",
        "longbench:qasper:one",
    ]
    assert aggregate["source_lock_sha256"] == source_lock_sha256


def test_engine_order_modes_reverse_every_shard() -> None:
    tool = load_tool()

    assert tool.engine_order_for_shard(0, "alternating") == ("aster", "mlx-lm")
    assert tool.engine_order_for_shard(1, "alternating") == ("mlx-lm", "aster")
    assert tool.engine_order_for_shard(0, "reversed") == ("mlx-lm", "aster")
    assert tool.engine_order_for_shard(1, "reversed") == ("aster", "mlx-lm")

    with pytest.raises(tool.MatrixError, match="engine order mode"):
        tool.engine_order_for_shard(0, "unknown")


def test_initial_manifest_pins_selected_engine_order(tmp_path: Path) -> None:
    tool = load_tool()
    workload = {
        "kind": "public-cross-engine-workload",
        "profile": "test",
        "generation": {"temperature": 0.0},
        "records": [
            workload_record("mt-bench:1:turn-1", {"id": "mt-bench-question"}),
            workload_record(
                "longbench:qasper:one",
                {"id": "longbench-v1-data", "dataset": "qasper"},
            ),
        ],
    }
    workload_path = tmp_path / "workload.json"
    workload_path.write_text(json.dumps(workload))
    args = SimpleNamespace(
        model=tmp_path / "model",
        max_input_tokens=32768,
        prefill_step=2048,
        warmup_tokens=8,
        memory_sample_interval=0.05,
        engine_order_mode="reversed",
    )

    manifest = tool._initial_manifest(
        workload_path,
        workload,
        args,
        {"model_sha256": "m" * 64, "tokenizer_sha256": "t" * 64},
    )

    assert manifest["engine_order_mode"] == "reversed"
    assert [entry["order"] for entry in manifest["shards"]] == [
        ["mlx-lm", "aster"],
        ["aster", "mlx-lm"],
    ]


def _matrix_record(
    workload_id: str,
    *,
    prompt_tokens: int,
    output_hash: str,
    metrics: dict[str, float],
) -> dict[str, Any]:
    return {
        "workload_id": workload_id,
        "prompt_sha256": "a" * 64,
        "prompt_token_ids_sha256": "i" * 64,
        "prompt_token_count": prompt_tokens,
        "output_token_ids_sha256": output_hash,
        "output_token_count": 8,
        "text_sha256": "x" * 64,
        "finish_reason": "length",
        "metrics": metrics,
        "resources": {"peak_rss_bytes": int(metrics["peak_rss_bytes"]), "swap_delta_bytes": 0},
    }


def _write_matrix_run(
    tool: ModuleType,
    root: Path,
    workload_path: Path,
    workload: dict[str, Any],
    orders: list[list[str]],
    *,
    aster_decode: float,
    lower_level_decode_trace: bool = False,
) -> None:
    root.mkdir(exist_ok=True)
    workload_sha256 = sha256(workload_path)
    fingerprint = {"model_sha256": "m" * 64, "tokenizer_sha256": "t" * 64}
    execution = {"input_truncation_policy": tool.TRUNCATION_POLICY, "prefill_step_tokens": 2048}
    if lower_level_decode_trace:
        execution["lower_level_decode_trace"] = tool._lower_level_decode_trace_metadata()
    rows_by_engine: dict[str, list[dict[str, Any]]] = {"aster": [], "mlx-lm": []}
    for index, record in enumerate(workload["records"]):
        base_metrics = {
            "ttft_seconds": 1.0,
            "end_to_end_seconds": 2.0,
            "prefill_tokens_per_second": 100.0,
            "decode_tokens_per_second": 100.0,
            "peak_rss_bytes": 1000.0,
            "swap_delta_bytes": 0.0,
        }
        aster_metrics = dict(base_metrics)
        aster_metrics.update(
            {
                "ttft_seconds": 0.9,
                "end_to_end_seconds": 1.8,
                "prefill_tokens_per_second": 110.0,
                "decode_tokens_per_second": aster_decode,
                "peak_rss_bytes": 900.0,
            }
        )
        for engine, metrics in (("aster", aster_metrics), ("mlx-lm", base_metrics)):
            row = _matrix_record(
                record["workload_id"],
                prompt_tokens=256 if index == 0 else 4096,
                output_hash=f"{index:064x}",
                metrics=metrics,
            )
            if lower_level_decode_trace:
                row["lower_level_decode_trace"] = _lower_level_decode_trace_metadata(
                    tool,
                    engine,
                )
            rows_by_engine[engine].append(row)
    for engine, rows in rows_by_engine.items():
        (root / f"{engine}.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "kind": "public-engine-result",
                    "engine": engine,
                    "engine_version": "test",
                    "workload_sha256": workload_sha256,
                    "source_lock_sha256": "l" * 64,
                    "generation": workload["generation"],
                    "execution": execution,
                    "model_fingerprint": fingerprint,
                    "records": rows,
                }
            )
        )
    shards = list(tool.workload_shards(workload).items())
    (root / "matrix-manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "public-engine-matrix",
                "status": "comparable",
                "workload": {"path": str(workload_path), "sha256": workload_sha256},
                "model_path": "test-model",
                "model_fingerprint": fingerprint,
                "execution": execution,
                "engines": ["aster", "mlx-lm"],
                "shards": [
                    {
                        "key": shard,
                        "record_count": len(records),
                        "order": orders[index],
                        "results": {},
                    }
                    for index, (shard, records) in enumerate(shards)
                ],
            }
        )
    )


def _write_trace_noop_shard_result(
    tool: ModuleType,
    path: Path,
    workload_path: Path,
    workload: dict[str, Any],
    *,
    engine: str,
    lower_level_decode_trace: bool,
) -> None:
    execution = {"input_truncation_policy": tool.TRUNCATION_POLICY, "prefill_step_tokens": 2048}
    if lower_level_decode_trace:
        execution["lower_level_decode_trace"] = tool._lower_level_decode_trace_metadata()
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(workload["records"]):
        metrics = {
            "ttft_seconds": 1.0,
            "end_to_end_seconds": 2.0,
            "prefill_tokens_per_second": 100.0,
            "decode_tokens_per_second": 100.0,
            "peak_rss_bytes": 1000.0,
            "swap_delta_bytes": 0.0,
        }
        if engine == "aster":
            metrics["decode_tokens_per_second"] = 110.0
        row = _matrix_record(
            record["workload_id"],
            prompt_tokens=256,
            output_hash=f"{index:064x}",
            metrics=metrics,
        )
        if lower_level_decode_trace:
            row["lower_level_decode_trace"] = _lower_level_decode_trace_metadata(
                tool,
                engine,
            )
        rows.append(row)
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "public-engine-result-shard",
                "engine": engine,
                "workload_sha256": sha256(workload_path),
                "source_lock_sha256": "l" * 64,
                "generation": workload["generation"],
                "execution": execution,
                "model_fingerprint": {"model_sha256": "m" * 64, "tokenizer_sha256": "t" * 64},
                "adapter_source_sha256": {
                    "scripts/dev/public_engine_matrix.py": "s" * 64,
                },
                "shard": {"key": "mt-bench", "record_count": len(rows)},
                "records": rows,
            }
        )
    )


def test_compare_matrices_requires_inverse_order_and_reports_order_strata(tmp_path: Path) -> None:
    tool = load_tool()
    workload = {
        "kind": "public-cross-engine-workload",
        "selection": {"global_cross_engine_claim_eligible": False},
        "generation": {"temperature": 0.0},
        "records": [
            workload_record("mt-bench:1:turn-1", {"id": "mt-bench-question"}),
            workload_record(
                "longbench:qasper:one",
                {"id": "longbench-v1-data", "dataset": "qasper"},
            ),
        ],
    }
    workload_path = tmp_path / "workload.json"
    workload_path.write_text(json.dumps(workload))
    original = tmp_path / "original"
    reversed_run = tmp_path / "reversed"
    _write_matrix_run(
        tool,
        original,
        workload_path,
        workload,
        [["aster", "mlx-lm"], ["mlx-lm", "aster"]],
        aster_decode=110.0,
    )
    _write_matrix_run(
        tool,
        reversed_run,
        workload_path,
        workload,
        [["mlx-lm", "aster"], ["aster", "mlx-lm"]],
        aster_decode=110.0,
    )

    underpowered = tool.compare_matrices(
        original,
        reversed_run,
        bootstrap_samples=100,
    )
    assert underpowered["decision"] == "no-material-order-confirmed-effect"
    underpowered_overall = next(
        group for group in underpowered["groups"] if group["id"] == "overall"
    )
    assert underpowered_overall["metrics"]["decode_tokens_per_second"]["order_agreement"] == "insufficient"

    comparison = tool.compare_matrices(
        original,
        reversed_run,
        bootstrap_samples=100,
        min_order_stratum_records=1,
    )

    assert comparison["decision"] == "order-confirmed-effects-require-component-attribution"
    assert comparison["gates"]["opposite_engine_order_per_shard"] is True
    assert comparison["order_balance"]["aster_first_records"] == 2
    assert comparison["order_balance"]["mlx_lm_first_records"] == 2
    overall = next(group for group in comparison["groups"] if group["id"] == "overall")
    assert overall["metrics"]["decode_tokens_per_second"]["order_agreement"] == "material-agreement"

    _write_matrix_run(
        tool,
        reversed_run,
        workload_path,
        workload,
        [["aster", "mlx-lm"], ["aster", "mlx-lm"]],
        aster_decode=110.0,
    )
    with pytest.raises(tool.MatrixError, match="opposite"):
        tool.compare_matrices(
            original,
            reversed_run,
            bootstrap_samples=100,
            min_order_stratum_records=1,
        )


def test_compare_lower_level_trace_noop_requires_exact_parity_and_no_material_movement(
    tmp_path: Path,
) -> None:
    tool = load_tool()
    workload = {
        "kind": "public-cross-engine-workload",
        "selection": {"global_cross_engine_claim_eligible": False},
        "generation": {"temperature": 0.0},
        "records": [
            workload_record("mt-bench:1:turn-1", {"id": "mt-bench-question"}),
            workload_record("mt-bench:2:turn-1", {"id": "mt-bench-question"}),
        ],
    }
    workload_path = tmp_path / "workload.json"
    workload_path.write_text(json.dumps(workload))
    untraced = tmp_path / "untraced"
    traced = tmp_path / "traced"
    orders = [["aster", "mlx-lm"]]
    _write_matrix_run(
        tool,
        untraced,
        workload_path,
        workload,
        orders,
        aster_decode=110.0,
    )
    _write_matrix_run(
        tool,
        traced,
        workload_path,
        workload,
        orders,
        aster_decode=110.0,
        lower_level_decode_trace=True,
    )

    comparison = tool.compare_lower_level_trace_noop(untraced, traced)

    assert comparison["decision"] == "trace-no-op-admitted"
    assert all(comparison["gates"].values())
    assert comparison["engines"]["aster"]["metrics"]["decode_tokens_per_second"][
        "median_traced_vs_untraced_percent"
    ] == 0.0

    payload_path = traced / "aster.json"
    payload = json.loads(payload_path.read_text())
    for row in payload["records"]:
        row["metrics"]["decode_tokens_per_second"] *= 1.04
    payload_path.write_text(json.dumps(payload))

    rejected = tool.compare_lower_level_trace_noop(untraced, traced)

    assert rejected["decision"] == "trace-no-op-rejected-metric-movement"
    assert rejected["gates"]["no_material_metric_movement"] is False


def test_compare_lower_level_trace_noop_shards_reuses_complete_engine_shards(
    tmp_path: Path,
) -> None:
    tool = load_tool()
    workload = {
        "kind": "public-cross-engine-workload",
        "lock_sha256": "l" * 64,
        "selection": {"global_cross_engine_claim_eligible": False},
        "generation": {"temperature": 0.0},
        "records": [
            workload_record("mt-bench:1:turn-1", {"id": "mt-bench-question"}),
            workload_record("mt-bench:2:turn-1", {"id": "mt-bench-question"}),
        ],
    }
    workload_path = tmp_path / "workload.json"
    workload_path.write_text(json.dumps(workload))
    untraced_results = {
        engine: tmp_path / f"{engine}-untraced.json" for engine in tool.ENGINE_NAMES
    }
    traced_results = {
        engine: tmp_path / f"{engine}-traced.json" for engine in tool.ENGINE_NAMES
    }
    for engine in tool.ENGINE_NAMES:
        _write_trace_noop_shard_result(
            tool,
            untraced_results[engine],
            workload_path,
            workload,
            engine=engine,
            lower_level_decode_trace=False,
        )
        _write_trace_noop_shard_result(
            tool,
            traced_results[engine],
            workload_path,
            workload,
            engine=engine,
            lower_level_decode_trace=True,
        )

    # Direct MLX-LM fingerprints its installed package in addition to the
    # common harness files; that engine-local dependency must not fail a
    # traced/untraced source-pair comparison.
    for payload_path in (untraced_results["mlx-lm"], traced_results["mlx-lm"]):
        payload = json.loads(payload_path.read_text())
        payload["adapter_source_sha256"][".venv/lib/python/site-packages/mlx_lm/__init__.py"] = (
            "p" * 64
        )
        payload_path.write_text(json.dumps(payload))

    comparison = tool.compare_lower_level_trace_noop_shards(
        workload_path,
        "mt-bench",
        untraced_results=untraced_results,
        traced_results=traced_results,
    )

    assert comparison["decision"] == "trace-no-op-admitted"
    assert comparison["gates"]["same_workload"] is True
    assert comparison["engines"]["mlx-lm"]["records"] == 2

    payload_path = traced_results["mlx-lm"]
    payload = json.loads(payload_path.read_text())
    payload["adapter_source_sha256"][".venv/lib/python/site-packages/mlx_lm/__init__.py"] = "n" * 64
    payload_path.write_text(json.dumps(payload))

    rejected = tool.compare_lower_level_trace_noop_shards(
        workload_path,
        "mt-bench",
        untraced_results=untraced_results,
        traced_results=traced_results,
    )

    assert rejected["decision"] == "trace-no-op-rejected-adapter-source-mismatch"
    assert rejected["gates"]["same_adapter_source"] is False


def _state_trace_metadata(tool: ModuleType, spec: dict[str, Any], pid: int) -> dict[str, Any]:
    snapshot = {
        "captured_utc": "2026-07-29T00:00:00+00:00",
        "load_average": [1.0, 1.0, 1.0],
        "cpu_count_logical": 8,
        "available_memory_bytes": 1024,
        "system_swap_used_bytes": 0,
        "process_rss_bytes": 512,
        "process_vms_bytes": 2048,
        "process_cpu_user_seconds": 1.0,
        "process_cpu_system_seconds": 0.5,
    }
    return {
        "schema_version": 1,
        "timing_boundary": tool.STATE_TRACE_TIMING_BOUNDARY,
        "block": {"id": spec["id"], "index": spec["index"]},
        "process": {
            "pid": pid,
            "started_utc": "2026-07-29T00:00:00+00:00",
            "finished_utc": "2026-07-29T00:01:00+00:00",
        },
        "snapshots": {
            "before_model_load": snapshot,
            "before_timed_records": snapshot,
            "after_timed_records": snapshot,
        },
    }


def _component_trace_metadata(
    tool: ModuleType,
    engine: str,
    *,
    output_token_count: int = 8,
) -> dict[str, Any]:
    base = {
        **tool._component_trace_metadata(),
        "decode": {
            "steps": output_token_count,
            "batch_size_min": 1,
            "batch_size_max": 1,
            "batch_size_total_items": output_token_count,
        },
        "cross_engine_comparable_boundary": "decode_driver_seconds",
    }
    if engine == "mlx-lm":
        return {
            **base,
            "engine_boundary": "mlx-lm-stream-generate-next",
            "seconds": {
                "decode_driver_seconds": 8.0,
                "caller_bookkeeping_seconds": 0.1,
                "post_decode_delivery_seconds": 0.01,
                "raw_generation_advance_seconds": 9.0,
            },
        }
    return {
        **base,
        "engine_boundary": "aster-manual-model-runner-single-decode-step",
        "seconds": {
            "decode_driver_seconds": 8.8,
            "caller_bookkeeping_seconds": 0.1,
            "post_decode_delivery_seconds": 0.01,
            "cache_resolution_seconds": 0.2,
            "model_graph_dispatch_seconds": 0.3,
            "processor_graph_dispatch_seconds": 0.1,
            "sampling_completion_seconds": 8.0,
            "result_delivery_seconds": 0.1,
            "unattributed_driver_seconds": 0.1,
        },
        "cache": {
            "decode_mode": "single-request-no-batch-merge",
            "batch_cache_reuses": 0,
            "batch_cache_rebuilds": 0,
            "single_steps": output_token_count,
            "cache_clear_attempts": 0,
            "cache_clears": 0,
            "cache_clear_failures": 0,
        },
    }


def _lower_level_decode_trace_metadata(
    tool: ModuleType,
    engine: str,
    *,
    output_token_count: int = 8,
) -> dict[str, Any]:
    traced_steps = max(output_token_count - 1, 0)
    per_step = (
        {
            "outer_step_seconds": 1.1,
            "model_submit_seconds": 0.1,
            "sampler_submit_seconds": 0.1,
            "completion_residual_seconds": 0.9,
        }
        if engine == "aster"
        else {
            "outer_step_seconds": 1.0,
            "model_submit_seconds": 0.1,
            "sampler_submit_seconds": 0.1,
            "completion_residual_seconds": 0.8,
        }
    )
    return {
        **tool._lower_level_decode_trace_metadata(),
        "engine_boundary": {
            "aster": "aster-manual-single-decode-step",
            "mlx-lm": "mlx-lm-stream-generate-post-first-next",
        }[engine],
        "decode": {
            "generated_output_steps": output_token_count,
            "traced_post_prefill_steps": traced_steps,
            "excluded_initial_output_steps": 1 if output_token_count else 0,
        },
        "seconds": {field: value * traced_steps for field, value in per_step.items()},
        "calls": {
            "model_submit_calls": traced_steps,
            "sampler_submit_calls": traced_steps,
        },
        "cross_engine_comparable_components": [
            "outer_step_seconds",
            *tool.LOWER_LEVEL_COMPONENT_SECONDS,
        ],
    }


def _write_state_trace_matrix_run(
    tool: ModuleType,
    root: Path,
    workload_path: Path,
    workload: dict[str, Any],
    spec: dict[str, Any],
    *,
    aster_decode: float,
    component_trace: bool = False,
    lower_level_decode_trace: bool = False,
) -> None:
    root.mkdir()
    workload_sha256 = sha256(workload_path)
    fingerprint = {"model_sha256": "m" * 64, "tokenizer_sha256": "t" * 64}
    execution = {"input_truncation_policy": tool.TRUNCATION_POLICY, "prefill_step_tokens": 2048}
    if component_trace:
        execution["component_trace"] = tool._component_trace_metadata()
    if lower_level_decode_trace:
        execution["lower_level_decode_trace"] = tool._lower_level_decode_trace_metadata()
    rows_by_engine: dict[str, list[dict[str, Any]]] = {"aster": [], "mlx-lm": []}
    for index, record in enumerate(workload["records"]):
        baseline = {
            "ttft_seconds": 1.0,
            "end_to_end_seconds": 2.0,
            "prefill_tokens_per_second": 100.0,
            "decode_tokens_per_second": 100.0,
            "peak_rss_bytes": 1000.0,
            "swap_delta_bytes": 0.0,
        }
        aster = dict(baseline)
        aster["decode_tokens_per_second"] = aster_decode
        for engine, metrics in (("aster", aster), ("mlx-lm", baseline)):
            row = _matrix_record(
                record["workload_id"],
                prompt_tokens=4096,
                output_hash=f"{index:064x}",
                metrics=metrics,
            )
            if component_trace:
                row["component_trace"] = _component_trace_metadata(tool, engine)
            if lower_level_decode_trace:
                row["lower_level_decode_trace"] = _lower_level_decode_trace_metadata(tool, engine)
            rows_by_engine[engine].append(row)
    for engine, rows in rows_by_engine.items():
        (root / f"{engine}.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "kind": "public-engine-result",
                    "engine": engine,
                    "engine_version": "test",
                    "workload_sha256": workload_sha256,
                    "source_lock_sha256": "l" * 64,
                    "generation": workload["generation"],
                    "execution": execution,
                    "model_fingerprint": fingerprint,
                    "records": rows,
                }
            )
        )
    trace = _state_trace_metadata(tool, spec, 1000 + spec["index"])
    (root / "matrix-manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "public-engine-matrix",
                "status": "comparable",
                "workload": {"path": str(workload_path), "sha256": workload_sha256},
                "model_path": "test-model",
                "model_fingerprint": fingerprint,
                "execution": execution,
                "engines": ["aster", "mlx-lm"],
                "engine_order_mode": spec["engine_order_mode"],
                **({"component_trace": tool._component_trace_metadata()} if component_trace else {}),
                **(
                    {"lower_level_decode_trace": tool._lower_level_decode_trace_metadata()}
                    if lower_level_decode_trace
                    else {}
                ),
                "state_trace": {
                    "schema_version": 1,
                    "timing_boundary": tool.STATE_TRACE_TIMING_BOUNDARY,
                    "block": {"id": spec["id"], "index": spec["index"]},
                },
                "shards": [
                    {
                        "key": tool.STATE_TRACE_QMSUM_SHARD,
                        "record_count": len(workload["records"]),
                        "order": spec["order"],
                        "results": {},
                        "state_traces": {"aster": trace, "mlx-lm": trace},
                    }
                ],
            }
        )
    )


def test_derive_public_shard_workload_retains_all_public_source_descriptors(tmp_path: Path) -> None:
    tool = load_tool()
    parent = {
        "schema_version": 1,
        "kind": "public-cross-engine-workload",
        "lock_sha256": "l" * 64,
        "data_root": "/public-source-root",
        "profile": "cross-engine-core",
        "selection": {"origin": "public-dataset-only"},
        "generation": {"temperature": 0.0},
        "records": [
            workload_record("mt-bench:1:turn-1", {"id": "mt-bench-question"}),
            workload_record(
                "longbench:qmsum:one",
                {"id": "longbench-v1-data", "dataset": "qmsum"},
            ),
            workload_record(
                "longbench:qmsum:two",
                {"id": "longbench-v1-data", "dataset": "qmsum"},
            ),
        ],
    }
    parent_path = tmp_path / "parent.json"
    output_path = tmp_path / "qmsum.json"
    parent_path.write_text(json.dumps(parent))

    derived = tool.derive_public_shard_workload(
        parent_path,
        tool.STATE_TRACE_QMSUM_SHARD,
        output_path,
    )

    assert [record["workload_id"] for record in derived["records"]] == [
        "longbench:qmsum:one",
        "longbench:qmsum:two",
    ]
    assert derived["selection"] == {
        "origin": "public-dataset-only-derived-shard",
        "parent_workload_path": str(parent_path),
        "parent_workload_sha256": sha256(parent_path),
        "parent_profile": "cross-engine-core",
        "shard": "longbench-qmsum",
        "all_parent_shard_records": True,
        "global_cross_engine_claim_eligible": False,
    }
    assert "Public benchmark prompt" not in output_path.read_text()
    assert tool.derive_public_shard_workload(
        parent_path,
        tool.STATE_TRACE_QMSUM_SHARD,
        output_path,
    ) == derived


def test_state_trace_abba_analysis_requires_complete_parity_and_repeats_order_effect(
    tmp_path: Path,
) -> None:
    tool = load_tool()
    specs = tool._state_trace_block_specs()
    assert [spec["order"] for spec in specs] == [
        ["aster", "mlx-lm"],
        ["mlx-lm", "aster"],
        ["mlx-lm", "aster"],
        ["aster", "mlx-lm"],
    ]
    workload = {
        "schema_version": 1,
        "kind": "public-cross-engine-workload",
        "lock_sha256": "l" * 64,
        "profile": "cross-engine-core:longbench-qmsum:order-state-trace",
        "selection": {"global_cross_engine_claim_eligible": False},
        "generation": {"temperature": 0.0},
        "records": [
            workload_record(
                f"longbench:qmsum:{index}",
                {"id": "longbench-v1-data", "dataset": "qmsum"},
            )
            for index in range(200)
        ],
    }
    workload_path = tmp_path / "qmsum-workload.json"
    workload_path.write_text(json.dumps(workload))
    roots: list[Path] = []
    for spec in specs:
        root = tmp_path / spec["id"]
        aster_decode = 110.0 if spec["order"][0] == "aster" else 90.0
        _write_state_trace_matrix_run(
            tool,
            root,
            workload_path,
            workload,
            spec,
            aster_decode=aster_decode,
        )
        roots.append(root)

    analysis = tool.analyze_order_state_trace(
        roots,
        bootstrap_samples=100,
        min_order_stratum_records=8,
    )

    assert analysis["decision"] == "order-interaction-reproduced"
    assert all(analysis["gates"].values())
    decode = analysis["groups"][0]["metrics"]["decode_tokens_per_second"]
    assert decode["aster_first"]["reproducibility"] == "positive-material"
    assert decode["mlx_lm_first"]["reproducibility"] == "negative-material"
    assert decode["order_state"] == "reproduced-directional-disagreement"
    assert len(analysis["block_state_traces"]) == 4

    broken = json.loads((roots[0] / "matrix-manifest.json").read_text())
    del broken["shards"][0]["state_traces"]["aster"]["snapshots"]["after_timed_records"]
    (roots[0] / "matrix-manifest.json").write_text(json.dumps(broken))
    with pytest.raises(tool.MatrixError, match="after_timed_records"):
        tool.analyze_order_state_trace(
            roots,
            bootstrap_samples=100,
            min_order_stratum_records=8,
        )


def test_component_trace_abba_analysis_requires_complete_common_boundary(
    tmp_path: Path,
) -> None:
    tool = load_tool()
    specs = tool._state_trace_block_specs()
    workload = {
        "schema_version": 1,
        "kind": "public-cross-engine-workload",
        "lock_sha256": "l" * 64,
        "profile": "cross-engine-core:longbench-qmsum:component-trace",
        "selection": {"global_cross_engine_claim_eligible": False},
        "generation": {"temperature": 0.0},
        "records": [
            workload_record(
                f"longbench:qmsum:{index}",
                {"id": "longbench-v1-data", "dataset": "qmsum"},
            )
            for index in range(200)
        ],
    }
    workload_path = tmp_path / "qmsum-workload.json"
    workload_path.write_text(json.dumps(workload))
    roots: list[Path] = []
    for spec in specs:
        root = tmp_path / spec["id"]
        _write_state_trace_matrix_run(
            tool,
            root,
            workload_path,
            workload,
            spec,
            aster_decode=90.0,
            component_trace=True,
        )
        roots.append(root)

    analysis = tool.analyze_order_component_trace(
        roots,
        bootstrap_samples=100,
        min_order_stratum_records=8,
    )

    assert analysis["decision"] == "stable-decode-driver-gap-requires-lower-level-boundary"
    assert all(analysis["gates"].values())
    overall = analysis["groups"][0]["decode_driver_seconds_per_output_token"]
    assert overall["order_state"] == "reproduced-order-stable-effect"
    assert overall["aster_first"]["aster_internal_accounting"]["cache_totals"] == {
        "batch_cache_reuses": 0,
        "batch_cache_rebuilds": 0,
        "single_steps": 3200,
        "cache_clear_attempts": 0,
        "cache_clears": 0,
        "cache_clear_failures": 0,
    }

    broken = json.loads((roots[0] / "aster.json").read_text())
    del broken["records"][0]["component_trace"]
    (roots[0] / "aster.json").write_text(json.dumps(broken))
    with pytest.raises(tool.MatrixError, match="no component trace"):
        tool.analyze_order_component_trace(
            roots,
            bootstrap_samples=100,
            min_order_stratum_records=8,
        )


def test_lower_level_decode_trace_abba_analysis_requires_complete_source_boundary(
    tmp_path: Path,
) -> None:
    tool = load_tool()
    specs = tool._state_trace_block_specs()
    workload = {
        "schema_version": 1,
        "kind": "public-cross-engine-workload",
        "lock_sha256": "l" * 64,
        "profile": "cross-engine-core:longbench-qmsum:lower-level-decode-trace",
        "selection": {"global_cross_engine_claim_eligible": False},
        "generation": {"temperature": 0.0},
        "records": [
            workload_record(
                f"longbench:qmsum:{index}",
                {"id": "longbench-v1-data", "dataset": "qmsum"},
            )
            for index in range(200)
        ],
    }
    workload_path = tmp_path / "qmsum-workload.json"
    workload_path.write_text(json.dumps(workload))
    roots: list[Path] = []
    for spec in specs:
        root = tmp_path / spec["id"]
        _write_state_trace_matrix_run(
            tool,
            root,
            workload_path,
            workload,
            spec,
            aster_decode=90.0,
            lower_level_decode_trace=True,
        )
        roots.append(root)

    analysis = tool.analyze_order_lower_level_decode_trace(
        roots,
        bootstrap_samples=100,
        min_order_stratum_records=8,
    )

    assert analysis["decision"] == "stable-lower-level-component-gap-requires-source-candidate"
    assert all(analysis["gates"].values())
    assert analysis["stable_common_components"] == ["completion_residual_seconds"]
    overall = analysis["groups"][0]["post_prefill_decode_step_seconds"]["components"]
    assert overall["outer_step_seconds"]["order_state"] == "reproduced-order-stable-effect"
    assert (
        overall["completion_residual_seconds"]["order_state"]
        == "reproduced-order-stable-effect"
    )
    assert analysis["coverage"] == {
        "output_records": 800,
        "traced_post_prefill_records": 800,
        "initial_only_output_records": 0,
        "traced_post_prefill_steps": 5600,
    }

    broken = json.loads((roots[0] / "aster.json").read_text())
    del broken["records"][0]["lower_level_decode_trace"]
    (roots[0] / "aster.json").write_text(json.dumps(broken))
    with pytest.raises(tool.MatrixError, match="no lower-level trace"):
        tool.analyze_order_lower_level_decode_trace(
            roots,
            bootstrap_samples=100,
            min_order_stratum_records=8,
        )


def test_aster_component_observer_restores_runner_methods() -> None:
    tool = load_tool()

    class FakeModel:
        def __call__(self, value: int) -> int:
            return value + 1

    runner = SimpleNamespace()
    runner._model = FakeModel()
    runner._resolve_decode_cache = lambda value: value
    runner._apply_logits_processors = lambda logits, *, item: logits + item
    runner._sample_token = lambda logprobs, sampler: sampler(logprobs)
    runner._decode_result = lambda **kwargs: kwargs["token"]
    original_model = runner._model
    original_resolve = runner._resolve_decode_cache
    original_processors = runner._apply_logits_processors
    original_sample = runner._sample_token
    original_result = runner._decode_result

    observer = tool._AsterDecodeComponentObserver(runner)
    with observer:
        assert runner._resolve_decode_cache("cache") == "cache"
        assert runner._model(1) == 2
        assert runner._apply_logits_processors(1, item=2) == 3
        assert runner._sample_token(3, lambda value: value + 1) == 4
        assert runner._decode_result(token=5) == 5

    assert runner._model is original_model
    assert runner._resolve_decode_cache is original_resolve
    assert runner._apply_logits_processors is original_processors
    assert runner._sample_token is original_sample
    assert runner._decode_result is original_result
    assert all(value >= 0 for value in observer.seconds.values())
    assert observer.seconds["sampling_completion_seconds"] > 0


def test_aster_lower_level_observer_restores_runner_methods() -> None:
    tool = load_tool()

    class FakeModel:
        def __call__(self, value: int) -> int:
            return value + 1

    runner = SimpleNamespace()
    runner._model = FakeModel()
    runner._sample_token = lambda logprobs, sampler: sampler(logprobs)
    original_model = runner._model
    original_sample = runner._sample_token

    observer = tool._AsterLowerLevelDecodeObserver(runner)
    with observer:
        observer.begin_step()
        assert runner._model(1) == 2
        assert runner._sample_token(3, lambda value: value + 1) == 4
        step = observer.finish_step(1.0)

    assert runner._model is original_model
    assert runner._sample_token is original_sample
    assert step["model_submit_calls"] == 1
    assert step["sampler_submit_calls"] == 1
    assert step["model_submit_seconds"] >= 0
    assert step["sampler_submit_seconds"] >= 0
    assert step["completion_residual_seconds"] >= 0


def test_run_order_state_trace_builds_all_public_qmsum_abba_blocks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool = load_tool()
    parent = {
        "schema_version": 1,
        "kind": "public-cross-engine-workload",
        "lock_sha256": "l" * 64,
        "data_root": "/public-source-root",
        "profile": "cross-engine-core",
        "selection": {"global_cross_engine_claim_eligible": False},
        "generation": {"temperature": 0.0},
        "records": [
            workload_record(
                f"longbench:qmsum:{index}",
                {"id": "longbench-v1-data", "dataset": "qmsum"},
            )
            for index in range(200)
        ],
    }
    parent_path = tmp_path / "cross-engine-core.json"
    parent_path.write_text(json.dumps(parent))
    run_root = tmp_path / "trace"
    calls: list[SimpleNamespace] = []

    def fake_run_matrix(block_args: SimpleNamespace) -> dict[str, Any]:
        calls.append(block_args)
        block_args.run_dir.mkdir(parents=True)
        (block_args.run_dir / "matrix-manifest.json").write_text("{}")
        (block_args.run_dir / "comparison.json").write_text("{}")
        return {"validation": {"decision": "comparable"}}

    def fake_analysis(
        roots: list[Path],
        *,
        shard: str,
        bootstrap_samples: int,
        min_order_stratum_records: int,
    ) -> dict[str, Any]:
        assert roots == [call.run_dir for call in calls]
        assert shard == tool.STATE_TRACE_QMSUM_SHARD
        assert bootstrap_samples == 100
        assert min_order_stratum_records == 8
        return {"decision": "inconclusive-order-state", "production_candidate": "none"}

    monkeypatch.setattr(
        tool,
        "model_fingerprint",
        lambda _path: {"model_sha256": "m" * 64, "tokenizer_sha256": "t" * 64},
    )
    monkeypatch.setattr(tool, "run_matrix", fake_run_matrix)
    monkeypatch.setattr(tool, "analyze_order_state_trace", fake_analysis)
    args = SimpleNamespace(
        workload=parent_path,
        run_dir=run_root,
        shard=tool.STATE_TRACE_QMSUM_SHARD,
        resume=False,
        bootstrap_samples=100,
        min_order_stratum_records=8,
        model=tmp_path / "model",
        max_input_tokens=32768,
        prefill_step=2048,
        warmup_tokens=8,
        memory_sample_interval=0.05,
    )

    result = tool.run_order_state_trace(args)

    assert result["decision"] == "inconclusive-order-state"
    assert [call.engine_order_mode for call in calls] == list(tool.STATE_TRACE_ABBA_MODES)
    assert [call.state_trace_block_id for call in calls] == [
        "01-aster-first",
        "02-mlx-lm-first",
        "03-mlx-lm-first",
        "04-aster-first",
    ]
    assert all(call.state_trace for call in calls)
    derived = json.loads((run_root / "qmsum-workload.json").read_text())
    assert len(derived["records"]) == 200
    manifest = json.loads((run_root / "state-trace-manifest.json").read_text())
    assert manifest["status"] == "inconclusive-order-state"
    assert set(manifest["blocks"]) == {call.state_trace_block_id for call in calls}
