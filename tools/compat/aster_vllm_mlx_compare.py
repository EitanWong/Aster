#!/usr/bin/env python3
"""Black-box compatibility and performance probe for Aster vs examples/vllm-mlx.

The script assumes both services are already running. It intentionally compares
protocol shape, lifecycle markers, and observable metrics instead of exact token
content, because model sampling can diverge between otherwise compatible servers.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import platform
import statistics
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _headers(api_key: str | None) -> dict[str, str]:
    headers = {"content-type": "application/json"}
    if api_key:
        headers["authorization"] = f"Bearer {api_key}"
    return headers


def _jsonable(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        return repr(value)


def _usage_tokens(payload: dict[str, Any]) -> int:
    usage = payload.get("usage")
    if isinstance(usage, dict):
        completion = usage.get("completion_tokens") or usage.get("output_tokens")
        if isinstance(completion, int):
            return completion
    return 0


def _chat_payload_metadata(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    metadata: dict[str, Any] = {
        "response_id": payload.get("id"),
        "finish_reason": None,
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
        "content_preview": None,
        "content_length": None,
    }
    usage = payload.get("usage")
    if isinstance(usage, dict):
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = usage.get(key)
            if isinstance(value, int):
                metadata[key] = value
    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        choice = choices[0]
        if isinstance(choice, dict):
            metadata["finish_reason"] = choice.get("finish_reason")
            message = choice.get("message")
            content: Any = None
            if isinstance(message, dict):
                content = message.get("content")
            elif "text" in choice:
                content = choice.get("text")
            if isinstance(content, str):
                metadata["content_preview"] = content[:240]
                metadata["content_length"] = len(content)
    return metadata


def _p95(values: list[float]) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    return statistics.quantiles(values, n=20, method="inclusive")[18]


def _p99(values: list[float]) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    return statistics.quantiles(values, n=100, method="inclusive")[98]


def _required_keys(payload: Any, keys: set[str]) -> list[str]:
    if not isinstance(payload, dict):
        return sorted(keys)
    return sorted(key for key in keys if key not in payload)


@dataclass(frozen=True)
class Service:
    name: str
    base_url: str


class Probe:
    def __init__(self, *, timeout: float, api_key: str | None) -> None:
        self.timeout = timeout
        self.api_key = api_key

    async def json_request(
        self,
        service: Service,
        method: str,
        path: str,
        *,
        body: dict[str, Any] | None = None,
        request_id_header: str | None = None,
    ) -> dict[str, Any]:
        url = service.base_url.rstrip("/") + path
        headers = _headers(self.api_key)
        if request_id_header:
            headers["x-request-id"] = request_id_header
        started = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.request(
                    method,
                    url,
                    headers=headers,
                    json=body if body is not None else None,
                )
            elapsed = time.perf_counter() - started
            try:
                payload: Any = response.json()
            except Exception:
                payload = response.text
            return {
                "ok": response.status_code < 400,
                "status_code": response.status_code,
                "elapsed_s": elapsed,
                "payload": _jsonable(payload),
            }
        except Exception as exc:
            return {
                "ok": False,
                "status_code": None,
                "elapsed_s": time.perf_counter() - started,
                "error": repr(exc),
            }

    async def stream_request(
        self,
        service: Service,
        path: str,
        *,
        body: dict[str, Any],
        cancel_after_events: int | None = None,
    ) -> dict[str, Any]:
        url = service.base_url.rstrip("/") + path
        started = time.perf_counter()
        events: list[dict[str, Any]] = []
        ttft: float | None = None
        done = False
        closed_early = False
        status_code: int | None = None
        pending_event: str | None = None
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                async with client.stream(
                    "POST",
                    url,
                    headers=_headers(self.api_key),
                    json=body,
                ) as response:
                    status_code = response.status_code
                    async for line in response.aiter_lines():
                        if not line:
                            continue
                        if line.startswith(":"):
                            events.append({"comment": line[1:].strip()})
                            continue
                        if line.startswith("event:"):
                            pending_event = line.removeprefix("event:").strip()
                            continue
                        if not line.startswith("data:"):
                            events.append({"raw": line})
                            continue
                        data = line.removeprefix("data:").strip()
                        if data == "[DONE]":
                            done = True
                            events.append({"data": "[DONE]"})
                            break
                        if ttft is None:
                            ttft = time.perf_counter() - started
                        try:
                            decoded: Any = json.loads(data)
                        except json.JSONDecodeError:
                            decoded = data
                        event_record: dict[str, Any] = {"data": _jsonable(decoded)}
                        if pending_event:
                            event_record["event"] = pending_event
                        events.append(event_record)
                        event_type = decoded.get("type") if isinstance(decoded, dict) else None
                        if pending_event in {"response.completed", "message_stop"} or event_type in {
                            "response.completed",
                            "message_stop",
                        }:
                            done = True
                            break
                        pending_event = None
                        if cancel_after_events and len(events) >= cancel_after_events:
                            closed_early = True
                            break
            return {
                "ok": status_code is not None and status_code < 400,
                "status_code": status_code,
                "elapsed_s": time.perf_counter() - started,
                "ttft_s": ttft,
                "done": done,
                "closed_early": closed_early,
                "event_count": len(events),
                "first_events": events[:5],
                "last_events": events[-3:],
            }
        except Exception as exc:
            return {
                "ok": False,
                "status_code": status_code,
                "elapsed_s": time.perf_counter() - started,
                "ttft_s": ttft,
                "done": done,
                "closed_early": closed_early,
                "event_count": len(events),
                "first_events": events[:5],
                "last_events": events[-3:],
                "error": repr(exc),
            }


def _chat_body(model: str, *, stream: bool, max_tokens: int) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a concise compatibility probe."},
            {"role": "user", "content": "Reply with exactly five words."},
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    if stream:
        body["stream"] = True
        body["stream_options"] = {"include_usage": True}
    return body


def _completion_body(model: str, *, stream: bool, max_tokens: int) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "prompt": "Write a five word compatibility sentence:",
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    if stream:
        body["stream"] = True
        body["stream_options"] = {"include_usage": True}
    return body


def _responses_body(model: str, *, stream: bool, max_tokens: int) -> dict[str, Any]:
    return {
        "model": model,
        "input": "Reply with exactly five words.",
        "temperature": 0,
        "max_output_tokens": max_tokens,
        "stream": stream,
    }


def _timeout_body(body: dict[str, Any], timeout_seconds: float) -> dict[str, Any]:
    clone = json.loads(json.dumps(body))
    clone["timeout"] = timeout_seconds
    return clone


def _mixed_long_chat_body(
    model: str,
    *,
    prompt_words: int,
    max_tokens: int,
) -> dict[str, Any]:
    context = " ".join(f"context{index:05d}" for index in range(prompt_words))
    body = _chat_body(model, stream=False, max_tokens=max_tokens)
    body["messages"] = [
        {"role": "system", "content": "You are a scheduling benchmark target."},
        {
            "role": "user",
            "content": (
                "Read this long context, then answer with exactly one word: done.\n\n"
                f"{context}\n\nAnswer now."
            ),
        },
    ]
    return body


def _mixed_short_chat_body(
    model: str,
    *,
    index: int,
    max_tokens: int,
) -> dict[str, Any]:
    body = _chat_body(model, stream=False, max_tokens=max_tokens)
    body["messages"][-1]["content"] = (
        f"Short scheduling probe {index}. Reply with exactly one word: ok."
    )
    return body


def _check_json_case(name: str, result: dict[str, Any]) -> dict[str, Any]:
    payload = result.get("payload")
    missing: list[str] = []
    expected_object: str | None = None
    if name == "models":
        missing = _required_keys(payload, {"object", "data"})
    elif name == "chat_nonstream":
        missing = _required_keys(payload, {"id", "object", "choices", "usage"})
        expected_object = "chat.completion"
    elif name == "completion_nonstream":
        missing = _required_keys(payload, {"id", "object", "choices", "usage"})
        expected_object = "text_completion"
    elif name == "responses_nonstream":
        missing = _required_keys(payload, {"id", "object", "output", "usage"})
        expected_object = "response"
    elif name == "embeddings":
        missing = _required_keys(payload, {"object", "data", "usage"})
    elif name in {
        "lifecycle_chat_timeout",
        "lifecycle_completion_timeout",
    }:
        return {
            "passed": result.get("status_code") == 504,
            "expected_status_code": 504,
            "payload_shape": type(payload).__name__,
        }
    elif name == "lifecycle_cancel_unknown":
        return {
            "passed": result.get("status_code") == 404,
            "expected_status_code": 404,
            "payload_shape": type(payload).__name__,
        }
    passed = bool(result.get("ok")) and not missing
    if expected_object and isinstance(payload, dict):
        passed = passed and payload.get("object") == expected_object
    return {"passed": passed, "missing_keys": missing, "expected_object": expected_object}


def _check_stream_case(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "passed": bool(result.get("ok"))
        and result.get("status_code") == 200
        and result.get("done") is True
        and result.get("event_count", 0) >= 2,
        "terminal_done": result.get("done"),
        "event_count": result.get("event_count"),
        "ttft_s": result.get("ttft_s"),
    }


def _check_stream_closed_early_case(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "passed": bool(result.get("ok"))
        and result.get("status_code") == 200
        and result.get("closed_early") is True
        and result.get("done") is False
        and result.get("event_count", 0) >= 1,
        "closed_early": result.get("closed_early"),
        "terminal_done": result.get("done"),
        "event_count": result.get("event_count"),
        "ttft_s": result.get("ttft_s"),
    }


async def _run_case(
    probe: Probe,
    services: list[Service],
    name: str,
    method: str,
    path: str,
    *,
    body: dict[str, Any] | None = None,
) -> dict[str, Any]:
    per_service = {
        service.name: await probe.json_request(service, method, path, body=body)
        for service in services
    }
    checks = {name: _check_json_case(name, result) for name, result in per_service.items()}
    return {"name": name, "path": path, "method": method, "services": per_service, "checks": checks}


async def _run_stream_case(
    probe: Probe,
    services: list[Service],
    name: str,
    path: str,
    *,
    body: dict[str, Any],
    cancel_after_events: int | None = None,
) -> dict[str, Any]:
    per_service = {
        service.name: await probe.stream_request(
            service,
            path,
            body=body,
            cancel_after_events=cancel_after_events,
        )
        for service in services
    }
    check_fn = (
        _check_stream_closed_early_case
        if cancel_after_events is not None
        else _check_stream_case
    )
    checks = {name: check_fn(result) for name, result in per_service.items()}
    return {"name": name, "path": path, "method": "POST", "services": per_service, "checks": checks}


async def _performance_round(
    probe: Probe,
    service: Service,
    *,
    model: str,
    max_tokens: int,
    concurrency: int,
    requests: int,
) -> dict[str, Any]:
    semaphore = asyncio.Semaphore(concurrency)
    latencies: list[float] = []
    completion_tokens = 0

    async def one(index: int) -> dict[str, Any]:
        nonlocal completion_tokens
        body = _chat_body(model, stream=False, max_tokens=max_tokens)
        body["messages"][-1]["content"] = f"Reply with exactly five words. Request {index}."
        async with semaphore:
            result = await probe.json_request(service, "POST", "/v1/chat/completions", body=body)
        if result.get("ok"):
            latencies.append(float(result.get("elapsed_s") or 0.0))
            payload = result.get("payload")
            if isinstance(payload, dict):
                completion_tokens += _usage_tokens(payload)
        return result

    started = time.perf_counter()
    results = await asyncio.gather(*(one(index) for index in range(requests)))
    elapsed = time.perf_counter() - started
    successes = sum(1 for result in results if result.get("ok"))
    return {
        "concurrency": concurrency,
        "requests": requests,
        "successes": successes,
        "failures": requests - successes,
        "elapsed_s": elapsed,
        "latency_avg_s": statistics.fmean(latencies) if latencies else None,
        "latency_p50_s": statistics.median(latencies) if latencies else None,
        "latency_p95_s": _p95(latencies),
        "completion_tokens": completion_tokens,
        "completion_tps": completion_tokens / elapsed if elapsed > 0 else None,
    }


def _mixed_latency_summary(records: list[dict[str, Any]], *, kind: str) -> dict[str, Any]:
    selected = [record for record in records if record.get("kind") == kind]
    successful = [record for record in selected if record.get("ok")]
    latencies = [
        float(record["elapsed_s"])
        for record in successful
        if isinstance(record.get("elapsed_s"), int | float)
    ]
    completion_tokens = sum(
        int(record.get("completion_tokens") or 0) for record in successful
    )
    return {
        "requests": len(selected),
        "successes": len(successful),
        "failures": len(selected) - len(successful),
        "latency_avg_s": statistics.fmean(latencies) if latencies else None,
        "latency_p50_s": statistics.median(latencies) if latencies else None,
        "latency_p95_s": _p95(latencies),
        "latency_p99_s": _p99(latencies),
        "completion_tokens": completion_tokens,
    }


def _summarize_mixed_scheduling(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "long": _mixed_latency_summary(records, kind="long"),
        "short": _mixed_latency_summary(records, kind="short"),
        "records": len(records),
    }


async def _mixed_scheduling_round(
    probe: Probe,
    service: Service,
    *,
    model: str,
    run_index: int,
    long_prompt_words: int,
    long_requests: int,
    long_max_tokens: int,
    short_requests: int,
    short_max_tokens: int,
    short_start_delay_s: float,
    short_interval_s: float,
) -> dict[str, Any]:
    started = time.perf_counter()

    async def one(
        *,
        kind: str,
        index: int,
        delay_s: float,
        body: dict[str, Any],
    ) -> dict[str, Any]:
        await asyncio.sleep(max(delay_s, 0.0))
        request_id = f"mixed-{service.name}-{run_index}-{kind}-{index}"
        request_started = time.perf_counter()
        result = await probe.json_request(
            service,
            "POST",
            "/v1/chat/completions",
            body=body,
            request_id_header=request_id,
        )
        finished = time.perf_counter()
        payload = result.get("payload")
        response_metadata = _chat_payload_metadata(payload)
        completion_tokens = int(response_metadata.get("completion_tokens") or 0)
        return {
            "run": run_index,
            "request_id": request_id,
            "kind": kind,
            "index": index,
            "scheduled_delay_s": delay_s,
            "started_at_s": request_started - started,
            "finished_at_s": finished - started,
            "elapsed_s": result.get("elapsed_s"),
            "ok": result.get("ok"),
            "status_code": result.get("status_code"),
            "completion_tokens": completion_tokens,
            "error": result.get("error"),
            "failure_payload": None if result.get("ok") else result.get("payload"),
            **response_metadata,
        }

    tasks = [
        asyncio.create_task(
            one(
                kind="long",
                index=index,
                delay_s=0.0,
                body=_mixed_long_chat_body(
                    model,
                    prompt_words=long_prompt_words,
                    max_tokens=long_max_tokens,
                ),
            )
        )
        for index in range(long_requests)
    ]
    tasks.extend(
        asyncio.create_task(
            one(
                kind="short",
                index=index,
                delay_s=short_start_delay_s + index * short_interval_s,
                body=_mixed_short_chat_body(
                    model,
                    index=index,
                    max_tokens=short_max_tokens,
                ),
            )
        )
        for index in range(short_requests)
    )
    records = await asyncio.gather(*tasks)
    elapsed = time.perf_counter() - started
    return {
        "run": run_index,
        "elapsed_s": elapsed,
        "summary": _summarize_mixed_scheduling(records),
        "records": records,
    }


async def _mixed_scheduling_probe(
    probe: Probe,
    service: Service,
    *,
    model: str,
    runs: int,
    long_prompt_words: int,
    long_requests: int,
    long_max_tokens: int,
    short_requests: int,
    short_max_tokens: int,
    short_start_delay_s: float,
    short_interval_s: float,
    run_gap_s: float,
) -> dict[str, Any]:
    rounds: list[dict[str, Any]] = []
    all_records: list[dict[str, Any]] = []
    started = time.perf_counter()
    status_before = await probe.json_request(service, "GET", "/v1/status")
    for run_index in range(max(runs, 1)):
        round_result = await _mixed_scheduling_round(
            probe,
            service,
            model=model,
            run_index=run_index,
            long_prompt_words=long_prompt_words,
            long_requests=long_requests,
            long_max_tokens=long_max_tokens,
            short_requests=short_requests,
            short_max_tokens=short_max_tokens,
            short_start_delay_s=short_start_delay_s,
            short_interval_s=short_interval_s,
        )
        rounds.append(round_result)
        all_records.extend(round_result["records"])
        if run_index < max(runs, 1) - 1 and run_gap_s > 0:
            await asyncio.sleep(run_gap_s)
    elapsed = time.perf_counter() - started
    status_after = await probe.json_request(service, "GET", "/v1/status")
    return {
        "runs": max(runs, 1),
        "elapsed_s": elapsed,
        "status_before": status_before,
        "status_after": status_after,
        "summary": _summarize_mixed_scheduling(all_records),
        "rounds": rounds,
    }


async def run(args: argparse.Namespace) -> dict[str, Any]:
    services = [
        Service("aster", args.aster_url),
        Service("vllm_mlx", args.vllm_url),
    ]
    probe = Probe(timeout=args.timeout, api_key=args.api_key)
    cases: list[dict[str, Any]] = []

    cases.append(await _run_case(probe, services, "health", "GET", "/health"))
    cases.append(await _run_case(probe, services, "models", "GET", "/v1/models"))
    cases.append(await _run_case(probe, services, "cache_stats", "GET", "/v1/cache/stats"))
    cases.append(
        await _run_case(
            probe,
            services,
            "chat_nonstream",
            "POST",
            "/v1/chat/completions",
            body=_chat_body(args.model, stream=False, max_tokens=args.max_tokens),
        )
    )
    cases.append(
        await _run_stream_case(
            probe,
            services,
            "chat_stream",
            "/v1/chat/completions",
            body=_chat_body(args.model, stream=True, max_tokens=args.max_tokens),
        )
    )
    cases.append(
        await _run_case(
            probe,
            services,
            "completion_nonstream",
            "POST",
            "/v1/completions",
            body=_completion_body(args.model, stream=False, max_tokens=args.max_tokens),
        )
    )
    cases.append(
        await _run_stream_case(
            probe,
            services,
            "completion_stream",
            "/v1/completions",
            body=_completion_body(args.model, stream=True, max_tokens=args.max_tokens),
        )
    )
    cases.append(
        await _run_case(
            probe,
            services,
            "responses_nonstream",
            "POST",
            "/v1/responses",
            body=_responses_body(args.model, stream=False, max_tokens=args.max_tokens),
        )
    )
    cases.append(
        await _run_stream_case(
            probe,
            services,
            "responses_stream",
            "/v1/responses",
            body=_responses_body(args.model, stream=True, max_tokens=args.max_tokens),
        )
    )
    cases.append(
        await _run_case(
            probe,
            services,
            "invalid_excessive_max_tokens",
            "POST",
            "/v1/chat/completions",
            body=_chat_body(args.model, stream=False, max_tokens=999_999_999),
        )
    )
    cases.append(
        await _run_case(
            probe,
            services,
            "rerank_endpoint_presence",
            "POST",
            "/v1/rerank",
            body={
                "model": args.model,
                "query": "compatibility",
                "documents": ["compatible", "unrelated"],
            },
        )
    )
    cases.append(await _run_case(probe, services, "mcp_tools_presence", "GET", "/v1/mcp/tools"))

    if args.include_lifecycle:
        cases.append(
            await _run_case(
                probe,
                services,
                "lifecycle_chat_timeout",
                "POST",
                "/v1/chat/completions",
                body=_timeout_body(
                    _chat_body(args.model, stream=False, max_tokens=args.max_tokens),
                    args.lifecycle_timeout_seconds,
                ),
            )
        )
        cases.append(
            await _run_case(
                probe,
                services,
                "lifecycle_completion_timeout",
                "POST",
                "/v1/completions",
                body=_timeout_body(
                    _completion_body(args.model, stream=False, max_tokens=args.max_tokens),
                    args.lifecycle_timeout_seconds,
                ),
            )
        )
        cases.append(
            await _run_stream_case(
                probe,
                services,
                "lifecycle_chat_stream_early_close",
                "/v1/chat/completions",
                body=_chat_body(
                    args.model,
                    stream=True,
                    max_tokens=max(args.max_tokens, args.lifecycle_stream_max_tokens),
                ),
                cancel_after_events=args.lifecycle_close_after_events,
            )
        )
        cases.append(
            await _run_case(
                probe,
                services,
                "lifecycle_cancel_unknown",
                "POST",
                f"/v1/requests/{args.lifecycle_unknown_request_id}/cancel",
            )
        )

    if not args.skip_embeddings:
        cases.append(
            await _run_case(
                probe,
                services,
                "embeddings",
                "POST",
                "/v1/embeddings",
                body={"model": args.embedding_model or args.model, "input": ["alpha", "beta"]},
            )
        )

    prefix_observation: dict[str, Any] = {}
    if not args.skip_prefix_observation:
        for service in services:
            await probe.json_request(service, "DELETE", "/v1/cache/prefix")
            before = await probe.json_request(service, "GET", "/v1/cache/stats")
            first = await probe.json_request(
                service,
                "POST",
                "/v1/chat/completions",
                body=_chat_body(args.model, stream=False, max_tokens=args.max_tokens),
            )
            second = await probe.json_request(
                service,
                "POST",
                "/v1/chat/completions",
                body=_chat_body(args.model, stream=False, max_tokens=args.max_tokens),
            )
            after = await probe.json_request(service, "GET", "/v1/cache/stats")
            prefix_observation[service.name] = {
                "before": before,
                "first_elapsed_s": first.get("elapsed_s"),
                "second_elapsed_s": second.get("elapsed_s"),
                "after": after,
            }

    performance: dict[str, Any] = {}
    if not args.skip_performance:
        for service in services:
            service_rounds = []
            for concurrency in args.concurrency:
                service_rounds.append(
                    await _performance_round(
                        probe,
                        service,
                        model=args.model,
                        max_tokens=args.max_tokens,
                        concurrency=concurrency,
                        requests=args.requests,
                    )
                )
            performance[service.name] = service_rounds

    mixed_scheduling: dict[str, Any] = {}
    if args.include_mixed_scheduling:
        for service in services:
            mixed_scheduling[service.name] = await _mixed_scheduling_probe(
                probe,
                service,
                model=args.model,
                runs=args.mixed_runs,
                long_prompt_words=args.mixed_long_prompt_words,
                long_requests=args.mixed_long_requests,
                long_max_tokens=args.mixed_long_max_tokens,
                short_requests=args.mixed_short_requests,
                short_max_tokens=args.mixed_short_max_tokens,
                short_start_delay_s=args.mixed_short_start_delay,
                short_interval_s=args.mixed_short_interval,
                run_gap_s=args.mixed_run_gap,
            )

    return {
        "created_at": _now(),
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
        "config": {
            "aster_url": args.aster_url,
            "vllm_url": args.vllm_url,
            "model": args.model,
            "embedding_model": args.embedding_model,
            "max_tokens": args.max_tokens,
            "timeout": args.timeout,
            "requests": args.requests,
            "concurrency": args.concurrency,
            "skip_embeddings": args.skip_embeddings,
            "skip_performance": args.skip_performance,
            "skip_prefix_observation": args.skip_prefix_observation,
            "include_lifecycle": args.include_lifecycle,
            "lifecycle_timeout_seconds": args.lifecycle_timeout_seconds,
            "lifecycle_close_after_events": args.lifecycle_close_after_events,
            "lifecycle_stream_max_tokens": args.lifecycle_stream_max_tokens,
            "lifecycle_unknown_request_id": args.lifecycle_unknown_request_id,
            "include_mixed_scheduling": args.include_mixed_scheduling,
            "mixed_runs": args.mixed_runs,
            "mixed_long_prompt_words": args.mixed_long_prompt_words,
            "mixed_long_requests": args.mixed_long_requests,
            "mixed_long_max_tokens": args.mixed_long_max_tokens,
            "mixed_short_requests": args.mixed_short_requests,
            "mixed_short_max_tokens": args.mixed_short_max_tokens,
            "mixed_short_start_delay": args.mixed_short_start_delay,
            "mixed_short_interval": args.mixed_short_interval,
            "mixed_run_gap": args.mixed_run_gap,
        },
        "cases": cases,
        "prefix_observation": prefix_observation,
        "performance": performance,
        "mixed_scheduling": mixed_scheduling,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare a running Aster service with a running examples/vllm-mlx service."
    )
    parser.add_argument("--aster-url", default="http://127.0.0.1:8080")
    parser.add_argument("--vllm-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--requests", type=int, default=8)
    parser.add_argument(
        "--concurrency",
        type=int,
        nargs="+",
        default=[1, 2, 4],
        help="Concurrency levels for the non-streaming chat performance probe.",
    )
    parser.add_argument("--skip-embeddings", action="store_true")
    parser.add_argument("--skip-performance", action="store_true")
    parser.add_argument("--skip-prefix-observation", action="store_true")
    parser.add_argument(
        "--include-lifecycle",
        action="store_true",
        help="Run timeout, early stream close, and unknown cancel lifecycle probes.",
    )
    parser.add_argument(
        "--lifecycle-timeout-seconds",
        type=float,
        default=0.001,
        help="Per-request timeout used by lifecycle timeout probes.",
    )
    parser.add_argument(
        "--lifecycle-close-after-events",
        type=int,
        default=2,
        help="SSE data events to read before intentionally closing stream probes.",
    )
    parser.add_argument(
        "--lifecycle-stream-max-tokens",
        type=int,
        default=128,
        help="max_tokens for lifecycle early-close stream probes.",
    )
    parser.add_argument(
        "--lifecycle-unknown-request-id",
        default="compat-missing-request",
        help="Request id used for the unknown explicit cancel probe.",
    )
    parser.add_argument(
        "--include-mixed-scheduling",
        action="store_true",
        help="Run long-prefill plus short-request mixed scheduling probes.",
    )
    parser.add_argument(
        "--mixed-runs",
        type=int,
        default=1,
        help="Number of mixed scheduling rounds to run per service.",
    )
    parser.add_argument(
        "--mixed-long-prompt-words",
        type=int,
        default=2048,
        help="Approximate word count for each long-prefill scheduling prompt.",
    )
    parser.add_argument(
        "--mixed-long-requests",
        type=int,
        default=1,
        help="Number of long-prefill requests started at the beginning of each round.",
    )
    parser.add_argument(
        "--mixed-long-max-tokens",
        type=int,
        default=8,
        help="max_tokens for long-prefill scheduling requests.",
    )
    parser.add_argument(
        "--mixed-short-requests",
        type=int,
        default=8,
        help="Number of short requests inserted during each mixed scheduling round.",
    )
    parser.add_argument(
        "--mixed-short-max-tokens",
        type=int,
        default=8,
        help="max_tokens for short mixed scheduling requests.",
    )
    parser.add_argument(
        "--mixed-short-start-delay",
        type=float,
        default=0.05,
        help="Delay before inserting the first short request in a mixed scheduling round.",
    )
    parser.add_argument(
        "--mixed-short-interval",
        type=float,
        default=0.05,
        help="Delay between inserted short requests in a mixed scheduling round.",
    )
    parser.add_argument(
        "--mixed-run-gap",
        type=float,
        default=0.25,
        help="Delay between repeated mixed scheduling rounds.",
    )
    parser.add_argument("--out", default="compat-results/aster-vllm-mlx-compare.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = asyncio.run(run(args))
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"out": str(output), "cases": len(result["cases"])}, indent=2))


if __name__ == "__main__":
    main()
