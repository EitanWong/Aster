from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts/dev/public_arrival_load.py"


def load_tool() -> ModuleType:
    spec = importlib.util.spec_from_file_location("public_arrival_load", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _record(
    workload_id: str,
    *,
    family: str,
    dataset: str | None = None,
    max_tokens: int = 64,
) -> dict[str, object]:
    source: dict[str, object] = {"id": "mt-bench-question"}
    scenario: dict[str, object] = {"family": family}
    if dataset is not None:
        source = {"id": "longbench-v1-data", "dataset": dataset}
        scenario["dataset"] = dataset
    return {
        "workload_id": workload_id,
        "source": source,
        "scenario": scenario,
        "prompt": {"sha256": "a" * 64, "characters": 12},
        "max_tokens": max_tokens,
    }


def _workload() -> dict[str, object]:
    return {
        "kind": "public-cross-engine-workload",
        "generation": {"temperature": 0.0, "top_p": 1.0, "top_k": 0, "min_p": 0.0},
        "records": [
            *[
                _record(f"mt-bench:{index}:turn-1", family="interactive")
                for index in range(81, 89)
            ],
            _record("longbench:qmsum:one", family="long-context", dataset="qmsum"),
            _record("longbench:qmsum:two", family="long-context", dataset="qmsum"),
            _record("longbench:qmsum:three", family="long-context", dataset="qmsum"),
            _record("longbench:qmsum:four", family="long-context", dataset="qmsum"),
        ],
    }


def test_build_arrival_plans_keep_public_record_identity_and_dependencies() -> None:
    tool = load_tool()
    workload = _workload()

    idle = tool.build_arrival_plan(
        workload,
        scenario="idle-lifecycle",
        concurrency=2,
        max_output_tokens=32,
        stagger_delay_seconds=0.05,
    )
    assert idle.entries == ()
    assert idle.cancel_target_key is None
    assert tool._plan_max_output_tokens(idle) is None

    simultaneous = tool.build_arrival_plan(
        workload,
        scenario="simultaneous",
        concurrency=4,
        max_output_tokens=32,
        stagger_delay_seconds=0.05,
    )
    assert [entry.workload_id for entry in simultaneous.entries] == [
        "mt-bench:81:turn-1",
        "mt-bench:82:turn-1",
        "mt-bench:83:turn-1",
        "mt-bench:84:turn-1",
    ]
    assert {entry.release for entry in simultaneous.entries} == {"at-start"}
    assert all(entry.max_tokens == 32 for entry in simultaneous.entries)
    assert tool._plan_max_output_tokens(simultaneous) == 32

    staggered = tool.build_arrival_plan(
        workload,
        scenario="staggered-long-prefill",
        concurrency=4,
        max_output_tokens=32,
        stagger_delay_seconds=0.05,
    )
    assert staggered.entries[0].workload_id == "longbench:qmsum:one"
    assert staggered.entries[0].release == "at-start"
    assert all(entry.release == "after-prefill" for entry in staggered.entries[1:])
    assert all(entry.depends_on == staggered.entries[0].key for entry in staggered.entries[1:])
    assert [entry.delay_seconds for entry in staggered.entries[1:]] == pytest.approx(
        [0.05, 0.1, 0.15]
    )

    shared_prefix = tool.build_arrival_plan(
        workload,
        scenario="shared-prefix",
        concurrency=4,
        max_output_tokens=32,
        stagger_delay_seconds=0.05,
    )
    assert {entry.workload_id for entry in shared_prefix.entries} == {"longbench:qmsum:one"}
    assert shared_prefix.entries[0].release == "at-start"
    assert all(entry.release == "after-completion" for entry in shared_prefix.entries[1:])

    distinct_prefix = tool.build_arrival_plan(
        workload,
        scenario="distinct-prefix",
        concurrency=2,
        max_output_tokens=32,
        stagger_delay_seconds=0.05,
    )
    assert [entry.workload_id for entry in distinct_prefix.entries] == [
        "longbench:qmsum:one",
        "longbench:qmsum:two",
    ]
    assert [entry.release for entry in distinct_prefix.entries] == [
        "at-start",
        "after-completion",
    ]
    assert distinct_prefix.entries[1].depends_on == distinct_prefix.entries[0].key

    capacity_replay = tool.build_arrival_plan(
        workload,
        scenario="capacity-replay",
        concurrency=2,
        max_output_tokens=32,
        stagger_delay_seconds=0.05,
    )
    assert [entry.workload_id for entry in capacity_replay.entries] == [
        "longbench:qmsum:one",
        "longbench:qmsum:two",
        "longbench:qmsum:three",
        "longbench:qmsum:one",
    ]
    assert [entry.release for entry in capacity_replay.entries] == [
        "at-start",
        "after-completion",
        "after-completion",
        "after-completion",
    ]
    assert [entry.depends_on for entry in capacity_replay.entries[1:]] == [
        "long-primary",
        "capacity-distinct-1",
        "capacity-distinct-2",
    ]

    capacity_replay_depth = tool.build_arrival_plan(
        workload,
        scenario="capacity-replay-depth",
        concurrency=2,
        max_output_tokens=32,
        stagger_delay_seconds=0.05,
    )
    assert [entry.workload_id for entry in capacity_replay_depth.entries] == [
        "longbench:qmsum:one",
        "longbench:qmsum:two",
        "longbench:qmsum:three",
        "longbench:qmsum:four",
        "longbench:qmsum:one",
    ]
    assert [entry.release for entry in capacity_replay_depth.entries] == [
        "at-start",
        "after-completion",
        "after-completion",
        "after-completion",
        "after-completion",
    ]
    assert [entry.depends_on for entry in capacity_replay_depth.entries[1:]] == [
        "long-primary",
        "capacity-depth-distinct-1",
        "capacity-depth-distinct-2",
        "capacity-depth-distinct-3",
    ]

    cancellation = tool.build_arrival_plan(
        workload,
        scenario="cancel-during-prefill",
        concurrency=2,
        max_output_tokens=32,
        stagger_delay_seconds=0.05,
    )
    assert cancellation.cancel_target_key == cancellation.entries[0].key
    assert cancellation.entries[1].release == "after-cancellation"


def test_build_arrival_plan_requires_public_records_for_each_scenario() -> None:
    tool = load_tool()
    with pytest.raises(tool.ArrivalLoadError, match="interactive"):
        tool.build_arrival_plan(
            {"kind": "public-cross-engine-workload", "records": []},
            scenario="simultaneous",
            concurrency=1,
            max_output_tokens=32,
            stagger_delay_seconds=0.05,
        )


def test_arrival_baseline_settings_keep_decode_prefill_cap_opt_in() -> None:
    tool = load_tool()
    settings = tool.RuntimeSettings.model_validate({"embeddings": {"enabled": False}})

    default = tool._apply_baseline_settings(
        settings,
        concurrency=4,
        prefix_cache_enabled=True,
        decode_active_prefill_token_budget=None,
        snapshot_budget_bytes=None,
        snapshot_max_entries=None,
    )
    assert default.engine.decode_active_prefill_token_budget is None
    assert default.engine.snapshot_budget_bytes == settings.engine.snapshot_budget_bytes
    assert default.engine.snapshot_max_entries == settings.engine.snapshot_max_entries

    candidate = tool._apply_baseline_settings(
        settings,
        concurrency=4,
        prefix_cache_enabled=True,
        decode_active_prefill_token_budget=512,
        snapshot_budget_bytes=1024,
        snapshot_max_entries=1,
    )
    assert candidate.engine.decode_active_prefill_token_budget == 512
    assert candidate.engine.snapshot_budget_bytes == 1024
    assert candidate.engine.snapshot_max_entries == 1

    with pytest.raises(tool.ArrivalLoadError, match="positive"):
        tool._apply_baseline_settings(
            settings,
            concurrency=4,
            prefix_cache_enabled=True,
            decode_active_prefill_token_budget=0,
            snapshot_budget_bytes=None,
            snapshot_max_entries=None,
        )
    with pytest.raises(tool.ArrivalLoadError, match="positive"):
        tool._apply_baseline_settings(
            settings,
            concurrency=4,
            prefix_cache_enabled=True,
            decode_active_prefill_token_budget=None,
            snapshot_budget_bytes=0,
            snapshot_max_entries=None,
        )
    with pytest.raises(tool.ArrivalLoadError, match="positive"):
        tool._apply_baseline_settings(
            settings,
            concurrency=4,
            prefix_cache_enabled=True,
            decode_active_prefill_token_budget=None,
            snapshot_budget_bytes=None,
            snapshot_max_entries=0,
        )


def test_resource_summary_separates_engine_lifecycle_from_workload() -> None:
    tool = load_tool()
    lifecycle = {
        "before_engine_create": {
            "platform": "test",
            "python": "test",
            "process_rss_bytes": 10,
            "swap_used_bytes": 100,
        },
        "after_engine_create": {
            "platform": "test",
            "python": "test",
            "process_rss_bytes": 15,
            "swap_used_bytes": 100,
        },
        "after_engine_start": {
            "platform": "test",
            "python": "test",
            "process_rss_bytes": 20,
            "swap_used_bytes": 102,
        },
        "after_warmup": {
            "platform": "test",
            "python": "test",
            "process_rss_bytes": 30,
            "swap_used_bytes": 104,
        },
        "before_workload": {
            "platform": "test",
            "python": "test",
            "process_rss_bytes": 31,
            "swap_used_bytes": 105,
        },
        "after_workload": {
            "platform": "test",
            "python": "test",
            "process_rss_bytes": 70,
            "swap_used_bytes": 120,
        },
        "after_close": {
            "platform": "test",
            "python": "test",
            "process_rss_bytes": 25,
            "swap_used_bytes": 111,
        },
    }

    summary = tool._resource_summary(lifecycle=lifecycle, rss_samples=[31, 55, 70])

    assert summary["before"] == lifecycle["before_workload"]
    assert summary["after"] == lifecycle["after_workload"]
    assert summary["peak_rss_bytes"] == 70
    assert summary["rss_delta_bytes"] == 39
    assert summary["swap_delta_bytes"] == 15
    assert summary["stage_deltas"] == {
        "engine_create": {"process_rss_bytes": 5, "swap_used_bytes": 0},
        "engine_start": {"process_rss_bytes": 5, "swap_used_bytes": 2},
        "warmup": {"process_rss_bytes": 10, "swap_used_bytes": 2},
        "workload": {"process_rss_bytes": 39, "swap_used_bytes": 15},
        "close": {"process_rss_bytes": -45, "swap_used_bytes": -9},
        "total": {"process_rss_bytes": 15, "swap_used_bytes": 11},
    }


class _CancelledRequest(Exception):
    code = "request_cancelled"


class _FakeEngine:
    def __init__(self) -> None:
        self.cancelled_aliases: list[str] = []
        self.submissions: list[str] = []
        self._long_active = False
        self._long_cancelled = False
        self._long_released = asyncio.Event()

    async def submit(self, request: Any) -> Any:
        alias = request.request_aliases[0]
        self.submissions.append(alias)
        if alias == "long-primary":
            self._long_active = True
            await self._long_released.wait()
            self._long_active = False
            if self._long_cancelled:
                raise _CancelledRequest("cancelled")
        return SimpleNamespace(
            request_id=f"request-{alias}",
            text=f"response:{alias}",
            prompt_tokens=4,
            completion_tokens=2,
            prefill_cache_hit=False,
            generation_tps=10.0,
            peak_memory_gb=1.0,
            finish_reason="length",
        )

    async def cancel(self, alias: str) -> bool:
        self.cancelled_aliases.append(alias)
        self._long_cancelled = True
        self._long_released.set()
        return True

    def status(self) -> dict[str, object]:
        requests = [{"phase": "prefill"}] if self._long_active else []
        return {"requests": requests, "recent_request_timelines": []}


class _CompletionFakeEngine:
    def __init__(self) -> None:
        self.submissions: list[str] = []
        self._long_active = False

    async def submit(self, request: Any) -> Any:
        alias = request.request_aliases[0]
        self.submissions.append(alias)
        if alias == "long-primary":
            self._long_active = True
            await asyncio.sleep(0.03)
            self._long_active = False
        return SimpleNamespace(
            request_id=f"request-{alias}",
            text=f"response:{alias}",
            completion_tokens=2,
            prefill_cache_hit=alias.startswith("shared-prefix"),
            generation_tps=10.0,
            peak_memory_gb=1.0,
            finish_reason="length",
        )

    def status(self) -> dict[str, object]:
        requests = [{"phase": "prefill"}] if self._long_active else []
        return {"requests": requests, "recent_request_timelines": []}


def test_execute_cancel_plan_waits_for_prefill_then_runs_follow_up() -> None:
    tool = load_tool()
    workload = _workload()
    plan = tool.build_arrival_plan(
        workload,
        scenario="cancel-during-prefill",
        concurrency=2,
        max_output_tokens=8,
        stagger_delay_seconds=0.0,
    )

    async def scenario() -> None:
        engine = _FakeEngine()
        result = await tool.execute_arrival_plan(
            engine,
            plan,
            workload,
            resolve_prompt=lambda record: str(record["workload_id"]),
            timeout_seconds=1.0,
        )

        assert engine.cancelled_aliases == ["long-primary"]
        assert engine.submissions == ["long-primary", "cancel-follow-up"]
        assert result["cancel_accepted"] is True
        assert result["events"][0]["error"]["code"] == "request_cancelled"
        assert result["events"][1]["response"]["finish_reason"] == "length"

    asyncio.run(scenario())
    with pytest.raises(tool.ArrivalLoadError, match="concurrency"):
        tool.build_arrival_plan(
            _workload(),
            scenario="simultaneous",
            concurrency=0,
            max_output_tokens=32,
            stagger_delay_seconds=0.05,
        )


def test_execute_idle_lifecycle_plan_submits_no_request() -> None:
    tool = load_tool()
    workload = _workload()
    plan = tool.build_arrival_plan(
        workload,
        scenario="idle-lifecycle",
        concurrency=2,
        max_output_tokens=8,
        stagger_delay_seconds=0.0,
    )

    async def run_plan() -> None:
        engine = _CompletionFakeEngine()
        result = await tool.execute_arrival_plan(
            engine,
            plan,
            workload,
            resolve_prompt=lambda record: str(record["workload_id"]),
            timeout_seconds=1.0,
        )

        assert engine.submissions == []
        assert result["events"] == []
        assert result["cancel_accepted"] is None
        assert result["engine_status"] == {"requests": [], "recent_request_timelines": []}

    asyncio.run(run_plan())


@pytest.mark.parametrize(
    ("scenario", "concurrency", "expected_submissions"),
    [
        (
            "staggered-long-prefill",
            3,
            ["long-primary", "staggered-short-0", "staggered-short-1"],
        ),
        (
            "shared-prefix",
            3,
            ["long-primary", "shared-prefix-0", "shared-prefix-1"],
        ),
        (
            "distinct-prefix",
            2,
            ["long-primary", "distinct-prefix-0"],
        ),
        (
            "capacity-replay",
            2,
            [
                "long-primary",
                "capacity-distinct-1",
                "capacity-distinct-2",
                "capacity-replay-0",
            ],
        ),
        (
            "capacity-replay-depth",
            2,
            [
                "long-primary",
                "capacity-depth-distinct-1",
                "capacity-depth-distinct-2",
                "capacity-depth-distinct-3",
                "capacity-depth-replay-0",
            ],
        ),
    ],
)
def test_execute_dependency_release_plans_preserve_schedule_order(
    scenario: str,
    concurrency: int,
    expected_submissions: list[str],
) -> None:
    tool = load_tool()
    workload = _workload()
    plan = tool.build_arrival_plan(
        workload,
        scenario=scenario,
        concurrency=concurrency,
        max_output_tokens=8,
        stagger_delay_seconds=0.005,
    )

    async def run_plan() -> None:
        engine = _CompletionFakeEngine()
        result = await tool.execute_arrival_plan(
            engine,
            plan,
            workload,
            resolve_prompt=lambda record: str(record["workload_id"]),
            timeout_seconds=1.0,
        )

        assert engine.submissions == expected_submissions
        assert [event["error"] for event in result["events"]] == [None] * len(plan.entries)
        assert [event["response"]["finish_reason"] for event in result["events"]] == [
            "length"
        ] * len(plan.entries)

    asyncio.run(run_plan())
