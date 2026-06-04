from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path

import pytest

MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "tools/compat/aster_vllm_mlx_compare.py"
)
SPEC = importlib.util.spec_from_file_location("aster_vllm_mlx_compare", MODULE_PATH)
assert SPEC is not None
assert SPEC.loader is not None
compat = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = compat
SPEC.loader.exec_module(compat)


def test_lifecycle_timeout_checks_require_504() -> None:
    passed = compat._check_json_case(
        "lifecycle_chat_timeout",
        {"status_code": 504, "payload": {"detail": "Request timed out"}},
    )
    failed = compat._check_json_case(
        "lifecycle_completion_timeout",
        {"status_code": 400, "payload": {"detail": "bad request"}},
    )

    assert passed["passed"] is True
    assert passed["expected_status_code"] == 504
    assert failed["passed"] is False


def test_lifecycle_unknown_cancel_requires_404() -> None:
    passed = compat._check_json_case(
        "lifecycle_cancel_unknown",
        {"status_code": 404, "payload": {"detail": "Request not found"}},
    )
    failed = compat._check_json_case(
        "lifecycle_cancel_unknown",
        {"status_code": 200, "payload": {"cancelled": True}},
    )

    assert passed["passed"] is True
    assert passed["expected_status_code"] == 404
    assert failed["passed"] is False


def test_stream_closed_early_check_requires_open_stream_without_terminal_done() -> None:
    passed = compat._check_stream_closed_early_case(
        {
            "ok": True,
            "status_code": 200,
            "closed_early": True,
            "done": False,
            "event_count": 2,
            "ttft_s": 0.01,
        }
    )
    terminal_done = compat._check_stream_closed_early_case(
        {
            "ok": True,
            "status_code": 200,
            "closed_early": True,
            "done": True,
            "event_count": 2,
            "ttft_s": 0.01,
        }
    )
    not_closed = compat._check_stream_closed_early_case(
        {
            "ok": True,
            "status_code": 200,
            "closed_early": False,
            "done": False,
            "event_count": 2,
            "ttft_s": 0.01,
        }
    )

    assert passed["passed"] is True
    assert terminal_done["passed"] is False
    assert not_closed["passed"] is False


def test_mixed_scheduling_summary_reports_short_tail_latency() -> None:
    records = [
        {
            "kind": "short",
            "ok": True,
            "elapsed_s": 0.10,
            "completion_tokens": 1,
        },
        {
            "kind": "short",
            "ok": True,
            "elapsed_s": 0.20,
            "completion_tokens": 2,
        },
        {
            "kind": "short",
            "ok": False,
            "elapsed_s": 0.50,
            "completion_tokens": 0,
        },
        {
            "kind": "long",
            "ok": True,
            "elapsed_s": 1.0,
            "completion_tokens": 3,
        },
    ]

    summary = compat._summarize_mixed_scheduling(records)

    assert summary["records"] == 4
    assert summary["short"]["requests"] == 3
    assert summary["short"]["successes"] == 2
    assert summary["short"]["failures"] == 1
    assert summary["short"]["latency_p50_s"] == pytest.approx(0.15)
    assert summary["short"]["latency_p95_s"] is not None
    assert summary["short"]["latency_p99_s"] is not None
    assert summary["short"]["completion_tokens"] == 3
    assert summary["long"]["requests"] == 1


def test_chat_payload_metadata_extracts_finish_usage_and_preview() -> None:
    metadata = compat._chat_payload_metadata(
        {
            "id": "chatcmpl-test",
            "choices": [
                {
                    "finish_reason": "length",
                    "message": {"content": "abcdef"},
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 3,
                "total_tokens": 13,
            },
        }
    )

    assert metadata["response_id"] == "chatcmpl-test"
    assert metadata["finish_reason"] == "length"
    assert metadata["prompt_tokens"] == 10
    assert metadata["completion_tokens"] == 3
    assert metadata["total_tokens"] == 13
    assert metadata["content_preview"] == "abcdef"
    assert metadata["content_length"] == 6


def test_mixed_scheduling_round_records_long_and_short_requests() -> None:
    class FakeProbe:
        async def json_request(self, service, method, path, *, body=None, request_id_header=None):
            await asyncio.sleep(0)
            return {
                "ok": True,
                "status_code": 200,
                "elapsed_s": 0.01,
                "payload": {
                    "id": "chatcmpl-fake",
                    "choices": [
                        {
                            "finish_reason": "length",
                            "message": {"content": "x"},
                        }
                    ],
                    "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
                },
            }

    async def scenario() -> None:
        result = await compat._mixed_scheduling_round(
            FakeProbe(),
            compat.Service("fake", "http://127.0.0.1"),
            model="model",
            run_index=0,
            long_prompt_words=4,
            long_requests=1,
            long_max_tokens=1,
            short_requests=2,
            short_max_tokens=1,
            short_start_delay_s=0.0,
            short_interval_s=0.0,
        )

        assert result["summary"]["long"]["requests"] == 1
        assert result["summary"]["short"]["requests"] == 2
        assert result["summary"]["short"]["successes"] == 2
        assert [record["kind"] for record in result["records"]].count("short") == 2
        assert {record["request_id"] for record in result["records"]} == {
            "mixed-fake-0-long-0",
            "mixed-fake-0-short-0",
            "mixed-fake-0-short-1",
        }
        assert {record["failure_payload"] for record in result["records"]} == {None}
        assert {record["finish_reason"] for record in result["records"]} == {"length"}
        assert {record["content_preview"] for record in result["records"]} == {"x"}

    asyncio.run(scenario())


def test_mixed_scheduling_probe_records_status_snapshots() -> None:
    class FakeProbe:
        def __init__(self) -> None:
            self.status_calls = 0

        async def json_request(self, service, method, path, *, body=None, request_id_header=None):
            await asyncio.sleep(0)
            if method == "GET" and path == "/v1/status":
                self.status_calls += 1
                return {
                    "ok": True,
                    "status_code": 200,
                    "elapsed_s": 0.01,
                    "payload": {
                        "status": "running",
                        "decode_batch_diagnostics": {
                            "batch_attempts": self.status_calls,
                        },
                    },
                }
            return {
                "ok": True,
                "status_code": 200,
                "elapsed_s": 0.01,
                "payload": {
                    "id": "chatcmpl-fake",
                    "choices": [
                        {
                            "finish_reason": "length",
                            "message": {"content": "x"},
                        }
                    ],
                    "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
                },
            }

    async def scenario() -> None:
        probe = FakeProbe()
        result = await compat._mixed_scheduling_probe(
            probe,
            compat.Service("fake", "http://127.0.0.1"),
            model="model",
            runs=1,
            long_prompt_words=4,
            long_requests=1,
            long_max_tokens=1,
            short_requests=1,
            short_max_tokens=1,
            short_start_delay_s=0.0,
            short_interval_s=0.0,
            run_gap_s=0.0,
        )

        assert result["status_before"]["payload"]["decode_batch_diagnostics"][
            "batch_attempts"
        ] == 1
        assert result["status_after"]["payload"]["decode_batch_diagnostics"][
            "batch_attempts"
        ] == 2
        assert result["summary"]["records"] == 2

    asyncio.run(scenario())
