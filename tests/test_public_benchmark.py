from __future__ import annotations

import hashlib
import importlib.util
import json
import zipfile
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts/dev/public_benchmark.py"


def load_tool() -> ModuleType:
    spec = importlib.util.spec_from_file_location("public_benchmark", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def source(
    path: Path, source_id: str, local_path: str, validator: dict[str, Any]
) -> dict[str, Any]:
    return {
        "id": source_id,
        "url": "https://example.invalid/public-data",
        "local_path": local_path,
        "sha256": sha256(path),
        "size_bytes": path.stat().st_size,
        "validator": validator,
    }


def public_fixture(tmp_path: Path) -> tuple[ModuleType, dict[str, Any], Path, Path]:
    tool = load_tool()
    data_root = tmp_path / "public-data"
    mt_bench_path = data_root / "sources/mt-bench/question.jsonl"
    mt_bench_path.parent.mkdir(parents=True)
    mt_bench_path.write_text(
        json.dumps(
            {
                "question_id": 7,
                "category": "reasoning",
                "turns": ["Public benchmark prompt", "A second public turn"],
            }
        )
        + "\n"
    )

    longbench_dir = data_root / "sources/longbench-v1"
    longbench_dir.mkdir(parents=True)
    data_zip = longbench_dir / "data.zip"
    with zipfile.ZipFile(data_zip, "w") as archive:
        archive.writestr(
            "data/qasper.jsonl",
            json.dumps(
                {
                    "_id": "qasper-1",
                    "input": "What does the paper conclude?",
                    "context": "The public paper says the result is stable.",
                    "answers": ["The result is stable."],
                    "length": 9,
                    "dataset": "qasper",
                    "language": "en",
                    "all_classes": None,
                }
            )
            + "\n",
        )
    prompts_path = longbench_dir / "dataset2prompt.json"
    prompts_path.write_text(
        json.dumps({"qasper": "Article: {context}\nQuestion: {input}\nAnswer:"})
    )
    max_output_path = longbench_dir / "dataset2maxlen.json"
    max_output_path.write_text(json.dumps({"qasper": 128}))

    lock = {
        "schema_version": 1,
        "sources": [
            source(
                mt_bench_path,
                "mt-bench-question",
                "sources/mt-bench/question.jsonl",
                {
                    "kind": "jsonl",
                    "record_count": 1,
                    "required_keys": ["question_id", "category", "turns"],
                    "unique_key": "question_id",
                },
            ),
            source(
                data_zip,
                "longbench-v1-data",
                "sources/longbench-v1/data.zip",
                {"kind": "longbench_zip", "task_count": 1, "record_count": 1},
            ),
            source(
                prompts_path,
                "longbench-v1-prompts",
                "sources/longbench-v1/dataset2prompt.json",
                {"kind": "json_mapping", "key_count": 1},
            ),
            source(
                max_output_path,
                "longbench-v1-max-output",
                "sources/longbench-v1/dataset2maxlen.json",
                {"kind": "json_mapping", "key_count": 1},
            ),
        ],
    }
    lock_path = tmp_path / "lock.json"
    lock_path.write_text(json.dumps(lock))
    return tool, lock, lock_path, data_root


def result_payload(tool: ModuleType, workload_path: Path, engine: str) -> dict[str, Any]:
    workload = json.loads(workload_path.read_text())
    records = []
    for record in workload["records"]:
        records.append(
            {
                "workload_id": record["workload_id"],
                "prompt_sha256": record["prompt"]["sha256"],
                "prompt_token_ids_sha256": tool.sha256_text("same effective input tokens"),
                "prompt_token_count": 4,
                "output_token_ids_sha256": tool.sha256_text("same deterministic tokens"),
                "metrics": {
                    "ttft_seconds": 0.1,
                    "end_to_end_seconds": 0.2,
                    "prefill_tokens_per_second": 100.0,
                    "decode_tokens_per_second": 50.0,
                    "peak_rss_bytes": 1024,
                    "swap_delta_bytes": 0,
                },
            }
        )
    return {
        "schema_version": 1,
        "engine": engine,
        "engine_version": "test",
        "workload_sha256": sha256(workload_path),
        "generation": workload["generation"],
        "execution": {
            "input_truncation_policy": "official-half-head-half-tail",
            "max_input_tokens": 32768,
        },
        "model_fingerprint": {
            "model_sha256": tool.sha256_text("model"),
            "tokenizer_sha256": tool.sha256_text("tokenizer"),
        },
        "records": records,
    }


def test_public_fixture_verifies_and_builds_a_full_public_workload(tmp_path: Path) -> None:
    tool, lock, lock_path, data_root = public_fixture(tmp_path)

    verification = tool.verify_install(lock, data_root)
    assert verification["decision"] == "verified"
    assert verification["sources"]["longbench-v1-data"]["record_count"] == 1

    workload = tool.build_workload(lock, lock_path, data_root, "full-public", None)
    assert workload["selection"]["origin"] == "public-dataset-only"
    assert workload["selection"]["global_cross_engine_claim_eligible"] is True
    assert [record["workload_id"] for record in workload["records"]] == [
        "mt-bench:7:turn-1",
        "longbench:qasper:qasper-1",
    ]
    assert workload["records"][1]["prompt"]["renderer"] == "official-longbench-v1-template"


def test_resolver_renders_only_the_hashed_public_source_prompt(tmp_path: Path) -> None:
    tool, lock, lock_path, data_root = public_fixture(tmp_path)
    workload = tool.build_workload(lock, lock_path, data_root, "full-public", None)
    resolver = tool.PublicWorkloadResolver(lock, data_root)

    assert resolver.resolve(workload["records"][0]) == "Public benchmark prompt"
    assert resolver.resolve(workload["records"][1]) == (
        "Article: The public paper says the result is stable.\n"
        "Question: What does the paper conclude?\n"
        "Answer:"
    )

    drifted = json.loads(json.dumps(workload["records"][1]))
    drifted["source"]["record_sha256"] = "0" * 64
    with pytest.raises(tool.PublicBenchmarkError, match="source row drifted"):
        resolver.resolve(drifted)


def test_cross_engine_result_gate_requires_full_public_parity(tmp_path: Path) -> None:
    tool, lock, lock_path, data_root = public_fixture(tmp_path)
    workload = tool.build_workload(lock, lock_path, data_root, "full-public", None)
    workload_path = tmp_path / "workload.json"
    tool.write_json(workload_path, workload)

    aster_path = tmp_path / "aster.json"
    mlx_lm_path = tmp_path / "mlx-lm.json"
    tool.write_json(aster_path, result_payload(tool, workload_path, "aster"))
    tool.write_json(mlx_lm_path, result_payload(tool, workload_path, "mlx-lm"))

    result = tool.validate_engine_results(
        workload_path,
        [aster_path, mlx_lm_path],
        {"aster", "mlx-lm"},
    )
    assert result["decision"] == "comparable"
    assert result["scope"] == "global"
    assert all(result["gates"].values())

    invalid = result_payload(tool, workload_path, "mlx-lm")
    invalid["records"][0]["prompt_sha256"] = "0" * 64
    tool.write_json(mlx_lm_path, invalid)
    rejected = tool.validate_engine_results(
        workload_path,
        [aster_path, mlx_lm_path],
        {"aster", "mlx-lm"},
    )
    assert rejected["decision"] == "incomplete"
    assert any("prompt hash differs" in error for error in rejected["errors"])

    invalid = result_payload(tool, workload_path, "mlx-lm")
    invalid["records"][0]["prompt_token_ids_sha256"] = "0" * 64
    tool.write_json(mlx_lm_path, invalid)
    rejected = tool.validate_engine_results(
        workload_path,
        [aster_path, mlx_lm_path],
        {"aster", "mlx-lm"},
    )
    assert rejected["decision"] == "incomplete"
    assert any("effective input token hashes differ" in error for error in rejected["errors"])


def test_invalid_public_source_checksum_fails_before_workload_build(tmp_path: Path) -> None:
    tool, lock, _lock_path, data_root = public_fixture(tmp_path)
    source_path = data_root / "sources/mt-bench/question.jsonl"
    source_path.write_text(source_path.read_text() + "{}\n")

    with pytest.raises(tool.PublicBenchmarkError, match="size"):
        tool.verify_install(lock, data_root)


def test_inventory_keeps_every_cross_engine_candidate_visible() -> None:
    tool = load_tool()

    inventory = tool.inventory_engines()

    assert set(inventory["engines"]) == set(tool.ENGINE_PROBES)
    assert {
        "aster",
        "mlx-lm",
        "ollama",
        "llama.cpp",
        "vllm",
        "sglang",
        "vllm-mlx",
        "mlc-llm",
        "mistral.rs",
        "lmstudio-mlx-engine",
        "omlx",
        "exo",
    } <= set(inventory["engines"])
