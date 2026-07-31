from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts/dev/cache_ownership_probe.py"
ARTIFACT_PATH = (
    PROJECT_ROOT
    / "docs/loop-engineering/artifacts/ITER-20260731-085-exact-hit-shared-state-feasibility/cache-ownership-probe.json"
)


def load_tool() -> ModuleType:
    spec = importlib.util.spec_from_file_location("cache_ownership_probe", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_batch_size_parser_normalizes_and_rejects_ambiguous_inputs() -> None:
    tool = load_tool()

    assert tool.parse_batch_sizes("8,2,4") == (2, 4, 8)
    with pytest.raises(tool.ProbeError, match="unique"):
        tool.parse_batch_sizes("2,2")
    with pytest.raises(tool.ProbeError, match="at least 2"):
        tool.parse_batch_sizes("1,2")
    with pytest.raises(tool.ProbeError, match="comma-separated integers"):
        tool.parse_batch_sizes("2,eight")


def test_cache_inventory_rejects_unknown_layer_types_by_default() -> None:
    tool = load_tool()
    kv_cache = type("KVCache", (), {})()
    arrays_cache = type("ArraysCache", (), {})()
    unknown_cache = type("UnknownCache", (), {})()

    supported = tool.classify_cache_layers([arrays_cache, kv_cache, arrays_cache])
    rejected = tool.classify_cache_layers([arrays_cache, unknown_cache])

    assert supported == {
        "counts": {"ArraysCache": 2, "KVCache": 1},
        "supported_types": ["ArraysCache", "KVCache"],
        "unsupported_types": [],
        "all_supported": True,
    }
    assert rejected["unsupported_types"] == ["UnknownCache"]
    assert rejected["all_supported"] is False


def test_cache_bytes_are_attributed_by_concrete_layer_type() -> None:
    tool = load_tool()
    KVCache = type("KVCache", (), {"nbytes": 30})
    ArraysCache = type("ArraysCache", (), {"nbytes": 20})

    cache = [ArraysCache(), KVCache(), ArraysCache()]

    assert tool.cache_nbytes_by_type(cache) == {"ArraysCache": 40, "KVCache": 30}


def test_summary_attributes_physical_growth_to_merge_not_clone() -> None:
    tool = load_tool()
    rows = [
        {
            "fanout": 8,
            "clone_seconds": 0.0008,
            "clone_active_delta_bytes": 0,
            "merge_seconds": 0.20,
            "merge_active_delta_bytes": 312,
            "merged_state_bytes": 312,
            "release_active_delta_bytes": 0,
        },
        {
            "fanout": 8,
            "clone_seconds": 0.0007,
            "clone_active_delta_bytes": 0,
            "merge_seconds": 0.19,
            "merge_active_delta_bytes": 314,
            "merged_state_bytes": 314,
            "release_active_delta_bytes": 0,
        },
        {
            "fanout": 8,
            "clone_seconds": 0.0009,
            "clone_active_delta_bytes": 0,
            "merge_seconds": 0.21,
            "merge_active_delta_bytes": 310,
            "merged_state_bytes": 310,
            "release_active_delta_bytes": 0,
        },
    ]

    summary = tool.summarize_measurements(rows)

    assert summary["by_fanout"]["8"]["clone_active_delta_bytes"]["median"] == 0
    assert summary["by_fanout"]["8"]["merge_active_delta_bytes"]["median"] == 312
    assert summary["gates"] == {
        "clone_construction_zero_active_growth": True,
        "merge_growth_matches_materialized_state": True,
        "release_returns_to_baseline": True,
    }
    assert summary["physical_owner"] == "batch_merge_materialization"


def test_prompt_length_is_rejected_before_prefill() -> None:
    tool = load_tool()

    class Runner:
        prefill_calls = 0

        def encode_request(self, request):
            del request
            return SimpleNamespace(prompt_tokens=[1, 2, 3])

        def prefill_to(self, **kwargs):
            del kwargs
            self.prefill_calls += 1
            raise AssertionError("prefill must not run above the token limit")

    runner = Runner()
    with pytest.raises(tool.ProbeError, match="above the configured maximum"):
        tool._prefill_public_record(
            runner,
            "public prompt",
            prefill_step=2,
            max_input_tokens=2,
        )
    assert runner.prefill_calls == 0


def test_mlx_hybrid_cache_deepcopy_isolates_base_siblings_and_extracted_rows() -> None:
    mx = pytest.importorskip("mlx.core")
    cache_module = pytest.importorskip("mlx_lm.models.cache")
    ArraysCache = cache_module.ArraysCache
    KVCache = cache_module.KVCache

    arrays = ArraysCache(size=2)
    arrays[0] = mx.array([[[1.0, 2.0], [3.0, 4.0]]])
    arrays[1] = mx.array([[[5.0], [6.0]]])
    kv = KVCache()
    kv.update_and_fetch(
        mx.array([[[[1.0, 2.0], [3.0, 4.0]]]]),
        mx.array([[[[5.0, 6.0], [7.0, 8.0]]]]),
    )
    mx.eval(arrays.state, kv.state)
    base = [arrays, kv]
    left = copy.deepcopy(base)
    right = copy.deepcopy(base)

    assert base[0][0] is not left[0][0]
    assert base[1].keys is not left[1].keys
    base_arrays = base[0][0].tolist()
    base_keys = base[1].state[0].tolist()

    left[0][0] = left[0][0] + 10
    left[1].update_and_fetch(
        mx.array([[[[9.0, 10.0]]]]),
        mx.array([[[[11.0, 12.0]]]]),
    )
    mx.eval(left[0].state, left[1].state, right[0].state, right[1].state)

    assert base[0][0].tolist() == base_arrays
    assert right[0][0].tolist() == base_arrays
    assert base[1].state[0].tolist() == base_keys
    assert right[1].state[0].tolist() == base_keys
    assert (base[1].offset, left[1].offset, right[1].offset) == (2, 3, 2)

    assert left[1].trim(1) == 1
    assert (base[1].offset, left[1].offset, right[1].offset) == (2, 2, 2)

    merged = [layer.merge([base[index], right[index]]) for index, layer in enumerate(base)]
    mx.eval([layer.state for layer in merged])
    first = [layer.extract(0) for layer in merged]
    second = [layer.extract(1) for layer in merged]
    first[0][0] = first[0][0] + 20
    first[1].update_and_fetch(
        mx.array([[[[13.0, 14.0]]]]),
        mx.array([[[[15.0, 16.0]]]]),
    )
    mx.eval(first[0].state, first[1].state, second[0].state, second[1].state)

    assert second[0][0].tolist() == base_arrays
    assert second[1].state[0].tolist() == base_keys
    assert base[0][0].tolist() == base_arrays
    assert base[1].state[0].tolist() == base_keys
    assert (first[1].offset, second[1].offset, base[1].offset) == (3, 2, 2)


def test_retained_probe_artifact_recomputes_the_ownership_decision() -> None:
    payload = json.loads(ARTIFACT_PATH.read_text())

    assert payload["kind"] == "cache-ownership-probe"
    assert payload["source"]["workload"].startswith("run/loop-engineering/")
    assert payload["source"]["source_lock"].startswith("docs/loop-engineering/")
    assert payload["cache"]["counts"] == {"ArraysCache": 24, "KVCache": 8}
    assert payload["cache"]["unsupported_types"] == []
    rows = payload["measurements"]
    assert len(rows) == 9
    assert sorted(row["fanout"] for row in rows) == [2, 2, 2, 4, 4, 4, 8, 8, 8]
    assert all(row["clone_active_delta_bytes"] == 0 for row in rows)
    assert all(row["release_active_delta_bytes"] == 0 for row in rows)
    assert all(
        abs(row["merge_active_delta_bytes"] - row["merged_state_bytes"])
        <= max(4096, row["merged_state_bytes"] // 1000)
        for row in rows
    )
    assert payload["summary"]["physical_owner"] == "batch_merge_materialization"
    assert all(payload["summary"]["gates"].values())
    assert payload["terminal"]["active_delta_bytes"] == 0
