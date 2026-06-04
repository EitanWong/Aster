"""Tests for API routes and schemas."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Generator
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from aster.core.errors import AsterError
from aster.core.lifecycle import create_application
from aster.inference.decode_engine import DecodeChunk
from aster.inference.tool_parsers import AutoToolParser


@pytest.fixture
def client(tmp_path: Path) -> Generator[TestClient, None, None]:
    """Create a test client with a temporary config."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text("""
api:
  host: 127.0.0.1
  port: 8000
logging:
  level: INFO
model:
  name: dummy-model
  path: "dummy"
  runtime: mlx
audio:
  asr:
    enabled: false
  tts:
    enabled: false
""")
    app = create_application(str(config_path))
    yield TestClient(app)


def _sse_json_payloads(body: str) -> list[dict[str, object]]:
    payloads: list[dict[str, object]] = []
    for line in body.splitlines():
        if not line.startswith("data: "):
            continue
        data = line.removeprefix("data: ")
        if data == "[DONE]":
            continue
        payloads.append(json.loads(data))
    return payloads


def _sse_nonempty_lines(body: str) -> list[str]:
    return [line for line in body.splitlines() if line]


def _metric_line(metrics: str, name: str, **labels: str) -> str:
    matches = [
        line
        for line in metrics.splitlines()
        if line.startswith(f"{name}{{")
        and all(f'{key}="{value}"' in line for key, value in labels.items())
    ]
    assert len(matches) == 1
    return matches[0]


def _sse_events(body: str) -> list[tuple[str | None, dict[str, object]]]:
    events: list[tuple[str | None, dict[str, object]]] = []
    event_name: str | None = None
    data_lines: list[str] = []
    for line in body.splitlines():
        if line.startswith("event: "):
            event_name = line.removeprefix("event: ")
            continue
        if line.startswith("data: "):
            data_lines.append(line.removeprefix("data: "))
            continue
        if line or not data_lines:
            continue
        events.append((event_name, json.loads("\n".join(data_lines))))
        event_name = None
        data_lines = []
    if data_lines:
        events.append((event_name, json.loads("\n".join(data_lines))))
    return events


def _responses_usage(input_tokens: int, output_tokens: int) -> dict[str, object]:
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens_details": {"reasoning_tokens": 0},
    }


async def _post_stream_until_asgi_disconnect(
    app: object,
    path: str,
    payload: dict[str, object],
    *,
    disconnect_after_body_messages: int,
) -> list[dict[str, object]]:
    body = json.dumps(payload).encode()
    cycle = SimpleNamespace(disconnected=False)
    request_sent = False
    sent_messages: list[dict[str, object]] = []
    body_messages = 0
    receive_blocker = asyncio.Event()

    async def receive() -> dict[str, object]:
        nonlocal request_sent
        if not request_sent:
            request_sent = True
            return {"type": "http.request", "body": body, "more_body": False}
        _ = cycle.disconnected
        await receive_blocker.wait()
        return {"type": "http.disconnect"}

    async def send(message: dict[str, object]) -> None:
        nonlocal body_messages
        sent_messages.append(message)
        if message["type"] != "http.response.body" or not message.get("body"):
            return
        body_messages += 1
        if body_messages >= disconnect_after_body_messages:
            cycle.disconnected = True

    headers = [
        (b"host", b"testserver"),
        (b"content-type", b"application/json"),
        (b"content-length", str(len(body)).encode()),
        (b"x-request-id", b"asgi-stream-disconnect"),
    ]
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "headers": headers,
        "client": ("testclient", 50000),
        "server": ("testserver", 80),
        "root_path": "",
    }

    await asyncio.wait_for(app(scope, receive, send), timeout=2)
    return sent_messages


def test_health_endpoint(client: TestClient) -> None:
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    # Health can be either "ok" or "degraded" depending on engine startup state
    assert response.json()["status"] in ("ok", "degraded")


def test_ready_endpoint(client: TestClient) -> None:
    """Test readiness endpoint."""
    response = client.get("/ready")
    assert response.status_code == 200


def test_status_endpoint_reports_scheduler_style_fields(client: TestClient) -> None:
    response = client.get("/v1/status")

    assert response.status_code == 200
    data = response.json()
    assert data["model"] == "dummy-model"
    assert data["status"] in {"running", "stopped"}
    assert data["num_running"] == 0
    assert data["num_waiting"] == 0
    assert data["requests"] == []
    assert "steps_executed" in data
    assert "total_requests_processed" in data
    assert data["admission_rejections"] == 0
    assert data["timed_out_requests"] == 0
    assert data["responses_store"] == {
        "entries": 0,
        "max_entries": 1000,
        "scope": "process/provider",
    }


def test_models_endpoint(client: TestClient) -> None:
    """Test models list endpoint."""
    response = client.get("/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert "data" in data
    assert len(data["data"]) > 0


def test_cache_stats_endpoint_reports_engine_cache(client: TestClient) -> None:
    engine = client.app.state.container.inference_engine
    engine.prefix_store.store(
        model_name="dummy-model",
        prefix_tokens=list(range(64)),
        cache_token_count=63,
        prompt_cache={"cache": True},
        approx_bytes=128,
    )

    response = client.get("/v1/cache/stats")

    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "cache.stats"
    assert data["engine_cache"]["prefix_cache"]["entries"] == 1
    assert data["engine_cache"]["prefix_cache"]["bytes"] == 128
    assert data["engine_cache"]["prefix_cache"]["cached_tokens"] == 63
    assert data["engine_cache"]["prefix_cache"]["evictable_bytes"] == 128
    assert data["engine_cache"]["prefix_cache"]["memory_utilization"] > 0
    assert data["engine_cache"]["prefix_cache"]["tokens_saved"] == 0
    assert data["engine_cache"]["runtime_kernel"] == "manual"


def test_clear_prefix_cache_endpoint_preserves_response_shape(client: TestClient) -> None:
    engine = client.app.state.container.inference_engine
    engine.prefix_store.store(
        model_name="dummy-model",
        prefix_tokens=list(range(64)),
        cache_token_count=63,
        prompt_cache={"cache": True},
        approx_bytes=128,
    )

    response = client.delete("/v1/cache/prefix")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "cleared"
    assert data["engine_cache"]["prefix_cache"]["entries_cleared"] == 1
    assert engine.prefix_store.entry_count == 0


def test_clear_cache_endpoint_clears_prefix_and_runtime_caches(client: TestClient) -> None:
    engine = client.app.state.container.inference_engine
    engine.prefix_store.store(
        model_name="dummy-model",
        prefix_tokens=list(range(64)),
        cache_token_count=63,
        prompt_cache={"cache": True},
        approx_bytes=128,
    )

    response = client.delete("/v1/cache")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "cleared"
    assert data["engine_cache"]["prefix_cache"]["entries_cleared"] == 1
    assert data["engine_cache"]["runtime"] == {
        "mlx_cache_cleared": False,
        "reason": "model_not_loaded",
    }
    assert engine.prefix_store.entry_count == 0


def test_models_endpoint_hides_embedding_model_for_mlx_runtime(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("""
api:
  host: 127.0.0.1
  port: 8000
logging:
  level: INFO
model:
  name: dummy-model
  path: "dummy"
  draft_name: dummy-draft
  draft_path: "dummy-draft"
  runtime: mlx
audio:
  asr_enabled: false
  tts_enabled: false
embeddings:
  model: mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ
""")
    app = create_application(str(config_path))
    with TestClient(app) as mlx_client:
        response = mlx_client.get("/v1/models")

    assert response.status_code == 200
    data = response.json()
    assert [item["id"] for item in data["data"]] == [
        "dummy-model",
        "mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ",
    ]


def test_cancel_request_endpoint_calls_engine_cancel(client: TestClient) -> None:
    captured: list[str] = []

    async def fake_cancel(request_id: str) -> bool:
        captured.append(request_id)
        return True

    client.app.state.container.inference_engine.cancel = fake_cancel

    response = client.post("/v1/requests/chatcmpl-123/cancel")

    assert response.status_code == 200
    assert response.headers["X-Request-Id"] == "chatcmpl-123"
    assert response.json() == {
        "object": "request.cancel",
        "id": "chatcmpl-123",
        "cancelled": True,
        "model": "dummy-model",
    }
    assert captured == ["chatcmpl-123"]


def test_delete_request_endpoint_aliases_cancel(client: TestClient) -> None:
    captured: list[str] = []

    async def fake_cancel(request_id: str) -> bool:
        captured.append(request_id)
        return True

    client.app.state.container.inference_engine.cancel = fake_cancel

    response = client.delete("/v1/requests/chatcmpl-delete")

    assert response.status_code == 200
    assert response.json()["cancelled"] is True
    assert captured == ["chatcmpl-delete"]


def test_cancel_request_endpoint_returns_404_for_unknown_request(client: TestClient) -> None:
    async def fake_cancel(_request_id: str) -> bool:
        return False

    client.app.state.container.inference_engine.cancel = fake_cancel

    response = client.post("/v1/requests/missing-request/cancel")

    assert response.status_code == 404
    assert response.headers["X-Request-Id"] == "missing-request"
    assert response.json()["error"]["type"] == "request_not_found"


def test_metrics_endpoint(client: TestClient) -> None:
    """Test metrics endpoint."""
    response = client.get("/metrics")
    assert response.status_code == 200
    assert b"# HELP" in response.content or b"# TYPE" in response.content


def test_embeddings_endpoint_uses_engine_embedding_backend(client: TestClient) -> None:
    async def fake_embeddings(*, model: str | None, input_data: object) -> dict[str, object]:
        assert model is None
        assert input_data == ["hello", "world"]
        return {
            "object": "list",
            "data": [
                {"object": "embedding", "index": 0, "embedding": [0.1, 0.2]},
                {"object": "embedding", "index": 1, "embedding": [0.3, 0.4]},
            ],
            "model": "dummy-embedding-model",
            "usage": {"prompt_tokens": 2, "total_tokens": 2},
        }

    client.app.state.container.inference_engine.embeddings = fake_embeddings

    response = client.post(
        "/v1/embeddings",
        json={"input": ["hello", "world"]},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "list"
    assert data["model"] == "dummy-embedding-model"
    assert len(data["data"]) == 2


def test_embeddings_endpoint_rejects_unconfigured_request_model(client: TestClient) -> None:
    response = client.post(
        "/v1/embeddings",
        json={"model": "mlx-community/not-configured", "input": "hello"},
    )

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "embedding_model_not_available"


def test_chat_completions_accepts_openai_style_structured_messages(client: TestClient) -> None:
    """Chat Completions should accept common OpenAI-compatible structured message shapes."""

    async def fake_submit(_request: object) -> SimpleNamespace:
        return SimpleNamespace(
            request_id="test-request",
            text="ok",
            prompt_tokens=12,
            completion_tokens=1,
            cache_hit=False,
            speculative_enabled=False,
        )

    client.app.state.container.inference_engine.submit = fake_submit

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "dummy-model",
            "messages": [
                {"role": "developer", "content": "You are helpful."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Hello"},
                    ],
                },
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_123",
                            "type": "function",
                            "function": {
                                "name": "lookup_weather",
                                "arguments": '{"city":"Shanghai"}',
                            },
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_123", "content": "Sunny"},
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "lookup_weather",
                        "description": "Look up weather",
                        "parameters": {"type": "object"},
                    },
                }
            ],
            "tool_choice": "auto",
            "stream": False,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "chat.completion"
    assert data["choices"][0]["message"]["content"] == "ok"


def test_chat_completions_returns_openai_error_payload_on_engine_overload(
    client: TestClient,
) -> None:
    async def fake_submit(_request: object) -> SimpleNamespace:
        raise AsterError(
            code="queue_full",
            message="Inference engine queue is full",
            status_code=503,
            details={"queue_depth": 1},
        )

    client.app.state.container.inference_engine.submit = fake_submit

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "dummy-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 1,
        },
        headers={"X-Request-Id": "queue-full-api"},
    )

    assert response.status_code == 503
    assert response.headers["X-Request-Id"] == "queue-full-api"
    assert response.json()["error"] == {
        "type": "queue_full",
        "code": "queue_full",
        "message": "Inference engine queue is full",
        "details": {"queue_depth": 1},
    }


def test_chat_completions_exposes_reasoning_content(client: TestClient) -> None:
    async def fake_submit(_request: object) -> SimpleNamespace:
        return SimpleNamespace(
            request_id="reasoning_123",
            text="<think>scratch pad</think>final answer",
            prompt_tokens=12,
            completion_tokens=5,
            cache_hit=False,
            speculative_enabled=False,
        )

    client.app.state.container.inference_engine.submit = fake_submit

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "dummy-model",
            "messages": [{"role": "user", "content": "Think then answer"}],
            "stream": False,
        },
    )

    assert response.status_code == 200
    message = response.json()["choices"][0]["message"]
    assert message["reasoning_content"] == "scratch pad"
    assert message["content"] == "final answer"


def test_chat_completions_uses_engine_finish_reason(client: TestClient) -> None:
    async def fake_submit(_request: object) -> SimpleNamespace:
        return SimpleNamespace(
            request_id="length_123",
            text="truncated",
            prompt_tokens=4,
            completion_tokens=2,
            cache_hit=False,
            speculative_enabled=False,
            finish_reason="length",
        )

    client.app.state.container.inference_engine.submit = fake_submit

    response = client.post(
        "/v1/chat/completions",
        headers={"X-Request-Id": "openai-shape-trace"},
        json={
            "model": "dummy-model",
            "messages": [{"role": "user", "content": "Hello"}],
        },
    )

    assert response.status_code == 200
    assert response.json()["choices"][0]["finish_reason"] == "length"


def test_chat_completions_omits_aster_metadata_by_default(client: TestClient) -> None:
    async def fake_submit(_request: object) -> SimpleNamespace:
        return SimpleNamespace(
            request_id="openai_shape_123",
            text="ok",
            prompt_tokens=2,
            completion_tokens=1,
            cache_hit=True,
            speculative_enabled=True,
            finish_reason="stop",
        )

    client.app.state.container.inference_engine.submit = fake_submit

    response = client.post(
        "/v1/chat/completions",
        headers={"X-Request-Id": "openai-shape-trace"},
        json={
            "model": "dummy-model",
            "messages": [{"role": "user", "content": "Hello"}],
        },
    )

    assert response.status_code == 200
    assert response.headers["X-Request-Id"] == "openai-shape-trace"
    data = response.json()
    assert data["id"].startswith("chatcmpl-")
    assert data["id"] != "openai-shape-trace"
    assert "aster" not in data
    assert set(data) == {"id", "object", "created", "model", "choices", "usage"}


def test_chat_completions_debug_header_includes_aster_metadata(client: TestClient) -> None:
    async def fake_submit(_request: object) -> SimpleNamespace:
        return SimpleNamespace(
            request_id="debug_shape_123",
            text="ok",
            prompt_tokens=2,
            completion_tokens=1,
            cache_hit=True,
            speculative_enabled=True,
            finish_reason="stop",
        )

    client.app.state.container.inference_engine.submit = fake_submit

    response = client.post(
        "/v1/chat/completions",
        headers={"X-Aster-Debug": "1"},
        json={
            "model": "dummy-model",
            "messages": [{"role": "user", "content": "Hello"}],
        },
    )

    assert response.status_code == 200
    assert response.json()["aster"] == {
        "cache_hit": True,
        "speculative_enabled": True,
    }


def test_chat_completions_stream_exposes_reasoning_content(client: TestClient) -> None:
    async def fake_stream(_request: object):
        yield DecodeChunk(token="<think>scratch pad</think>", index=0, finished=False)
        yield DecodeChunk(token="final answer", index=1, finished=False)
        yield DecodeChunk(
            token="",
            index=2,
            finished=True,
            stats={"prompt_tokens": 12, "completion_tokens": 5},
        )

    client.app.state.container.inference_engine.stream = fake_stream

    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "dummy-model",
            "messages": [{"role": "user", "content": "Think then answer"}],
            "stream": True,
        },
    ) as response:
        body = b"".join(response.iter_bytes()).decode()

    assert response.status_code == 200
    assert "reasoning_content" in body
    assert "scratch pad" in body
    assert "final answer" in body


def test_chat_completions_stream_uses_openai_chunk_lifecycle(client: TestClient) -> None:
    captured_aliases: list[tuple[str, ...]] = []

    async def fake_stream(_request: object):
        captured_aliases.append(tuple(getattr(_request, "request_aliases", ())))
        yield DecodeChunk(token="hello", index=0, finished=False)
        yield DecodeChunk(
            token="",
            index=1,
            finished=True,
            stats={"prompt_tokens": 3, "completion_tokens": 1},
        )

    client.app.state.container.inference_engine.stream = fake_stream

    with client.stream(
        "POST",
        "/v1/chat/completions",
        headers={"X-Request-Id": "chat-stream-123"},
        json={
            "model": "dummy-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
            "stream_options": {"include_usage": True},
        },
    ) as response:
        body = b"".join(response.iter_bytes()).decode()

    assert response.status_code == 200
    payloads = _sse_json_payloads(body)
    chunks = [item for item in payloads if item.get("object") == "chat.completion.chunk"]

    assert response.headers["X-Request-Id"] == "chat-stream-123"
    assert _sse_nonempty_lines(body)[-1] == "data: [DONE]"
    assert len({chunk["id"] for chunk in chunks}) == 1
    assert chunks[0]["id"].startswith("chatcmpl-")
    assert chunks[0]["id"] != "chat-stream-123"
    assert captured_aliases == [(chunks[0]["id"],)]
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant"}
    assert chunks[1]["choices"][0]["delta"] == {"content": "hello"}
    assert chunks[2]["choices"][0]["delta"] == {}
    assert chunks[2]["choices"][0]["finish_reason"] == "stop"
    assert chunks[3]["choices"] == []
    assert chunks[3]["usage"] == {
        "prompt_tokens": 3,
        "completion_tokens": 1,
        "total_tokens": 4,
    }


def test_chat_completions_asgi_disconnect_closes_engine_stream(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("aster.api.streaming.SSE_DISCONNECT_POLL_SECONDS", 0.001)
    closed = asyncio.Event()

    async def fake_stream(_request: object):
        try:
            yield DecodeChunk(token="hello", index=0, finished=False)
            await asyncio.Event().wait()
        finally:
            closed.set()

    client.app.state.container.inference_engine.stream = fake_stream

    async def scenario() -> None:
        messages = await _post_stream_until_asgi_disconnect(
            client.app,
            "/v1/chat/completions",
            {
                "model": "dummy-model",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": True,
            },
            disconnect_after_body_messages=2,
        )

        body = b"".join(
            message.get("body", b"")
            for message in messages
            if message["type"] == "http.response.body"
        ).decode()

        assert closed.is_set()
        assert "chat.completion.chunk" in body
        assert '"content": "hello"' in body
        assert "data: [DONE]" not in body
        assert (
            'aster_errors_total{code="client_disconnected"} 1.0'
            in client.app.state.container.metrics.render().decode()
        )

    asyncio.run(scenario())


def test_openai_responses_asgi_disconnect_closes_provider_stream(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("aster.api.provider_gateway.SSE_DISCONNECT_POLL_SECONDS", 0.001)
    closed = asyncio.Event()

    async def fake_stream(_request: object):
        try:
            yield DecodeChunk(token="hello", index=0, finished=False)
            await asyncio.Event().wait()
        finally:
            closed.set()

    client.app.state.container.inference_engine.stream = fake_stream

    async def scenario() -> None:
        messages = await _post_stream_until_asgi_disconnect(
            client.app,
            "/v1/responses",
            {
                "model": "gpt-5.1",
                "input": "Hello",
                "stream": True,
            },
            disconnect_after_body_messages=5,
        )
        body = b"".join(
            message.get("body", b"")
            for message in messages
            if message["type"] == "http.response.body"
        ).decode()

        assert closed.is_set()
        assert "response.output_text.delta" in body
        assert "hello" in body
        assert "response.completed" not in body
        assert (
            'aster_errors_total{code="client_disconnected"} 1.0'
            in client.app.state.container.metrics.render().decode()
        )

    asyncio.run(scenario())


def test_openai_chat_live_tool_asgi_disconnect_closes_engine_stream(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("aster.api.interaction_loop.SSE_DISCONNECT_POLL_SECONDS", 0.001)
    closed = asyncio.Event()

    async def fake_stream(_request: object):
        try:
            yield DecodeChunk(token="partial", index=0, finished=False)
            await asyncio.Event().wait()
        finally:
            closed.set()

    client.app.state.container.inference_engine.stream = fake_stream

    async def scenario() -> None:
        messages = await _post_stream_until_asgi_disconnect(
            client.app,
            "/v1/chat/completions",
            {
                "model": "gpt-4.1",
                "stream": True,
                "messages": [{"role": "user", "content": "Use a tool"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "remote_add_numbers",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "a": {"type": "number"},
                                    "b": {"type": "number"},
                                },
                            },
                        },
                    }
                ],
                "tool_choice": "required",
            },
            disconnect_after_body_messages=2,
        )
        body = b"".join(
            message.get("body", b"")
            for message in messages
            if message["type"] == "http.response.body"
        ).decode()

        assert closed.is_set()
        assert "chat.completion.chunk" in body
        assert '"content": "partial"' in body
        assert "data: [DONE]" not in body
        assert (
            'aster_errors_total{code="client_disconnected"} 1.0'
            in client.app.state.container.metrics.render().decode()
        )

    asyncio.run(scenario())


def test_openai_responses_live_tool_asgi_disconnect_does_not_store_partial_replay(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("aster.api.interaction_loop.SSE_DISCONNECT_POLL_SECONDS", 0.001)
    closed = asyncio.Event()

    async def fake_stream(_request: object):
        try:
            yield DecodeChunk(
                token='<tool_call>{"name":"lookup_weather","arguments":{"city":"Shanghai"}}</tool_call>',
                index=0,
                finished=False,
            )
            await asyncio.Event().wait()
        finally:
            closed.set()

    client.app.state.container.inference_engine.stream = fake_stream

    async def scenario() -> None:
        messages = await _post_stream_until_asgi_disconnect(
            client.app,
            "/v1/responses",
            {
                "model": "gpt-5.1",
                "input": "Use a weather tool",
                "stream": True,
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "lookup_weather",
                            "parameters": {
                                "type": "object",
                                "properties": {"city": {"type": "string"}},
                            },
                        },
                    }
                ],
                "tool_choice": "required",
            },
            disconnect_after_body_messages=4,
        )
        body = b"".join(
            message.get("body", b"")
            for message in messages
            if message["type"] == "http.response.body"
        ).decode()
        events = _sse_events(body)
        created = next(payload["response"] for event, payload in events if event == "response.created")
        response_id = created["id"]

        assert closed.is_set()
        assert "response.function_call_arguments.delta" in body
        assert "response.completed" not in body
        assert (
            'aster_errors_total{code="client_disconnected"} 1.0'
            in client.app.state.container.metrics.render().decode()
        )

        follow_up = client.post(
            "/v1/responses",
            json={
                "model": "gpt-5.1",
                "previous_response_id": response_id,
                "input": "Continue.",
            },
        )

        assert follow_up.status_code == 404
        assert follow_up.json()["error"]["code"] == "response_not_found"

    asyncio.run(scenario())


def test_chat_completions_stream_emits_error_payload_on_aster_error(client: TestClient) -> None:
    async def fake_stream(_request: object):
        if False:
            yield DecodeChunk(token="", index=0, finished=False)
        raise AsterError(
            code="request_timeout",
            message="Inference request timed out",
            status_code=504,
            details={"timeout_seconds": 0.01},
        )

    client.app.state.container.inference_engine.stream = fake_stream

    with client.stream(
        "POST",
        "/v1/chat/completions",
        headers={"X-Request-Id": "chat-stream-error"},
        json={
            "model": "dummy-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
        },
    ) as response:
        body = b"".join(response.iter_bytes()).decode()

    assert response.status_code == 200
    payloads = _sse_json_payloads(body)
    assert payloads[-1]["error"] == {
        "message": "Inference request timed out",
        "type": "request_timeout",
        "code": "request_timeout",
        "details": {"timeout_seconds": 0.01},
    }
    assert _sse_nonempty_lines(body)[-1] == "data: [DONE]"


def test_chat_completions_stream_emits_error_payload_on_unhandled_error(client: TestClient) -> None:
    async def fake_stream(_request: object):
        if False:
            yield DecodeChunk(token="", index=0, finished=False)
        raise RuntimeError("runtime exploded")

    client.app.state.container.inference_engine.stream = fake_stream

    with client.stream(
        "POST",
        "/v1/chat/completions",
        headers={"X-Request-Id": "chat-stream-unhandled-error"},
        json={
            "model": "dummy-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
        },
    ) as response:
        body = b"".join(response.iter_bytes()).decode()

    assert response.status_code == 200
    payloads = _sse_json_payloads(body)
    assert payloads[-1]["error"] == {
        "message": "Streaming response failed",
        "type": "stream_failed",
        "code": "stream_failed",
        "details": {},
    }
    assert _sse_nonempty_lines(body)[-1] == "data: [DONE]"


def test_completions_stream_uses_text_completion_chunks(client: TestClient) -> None:
    async def fake_stream(_request: object):
        yield DecodeChunk(token="hello", index=0, finished=False)
        yield DecodeChunk(
            token="",
            index=1,
            finished=True,
            stats={"prompt_tokens": 2, "completion_tokens": 1},
        )

    client.app.state.container.inference_engine.stream = fake_stream

    with client.stream(
        "POST",
        "/v1/completions",
        headers={"X-Request-Id": "completion-stream-123"},
        json={
            "model": "dummy-model",
            "prompt": "Say hello",
            "stream": True,
        },
    ) as response:
        body = b"".join(response.iter_bytes()).decode()

    assert response.status_code == 200
    payloads = _sse_json_payloads(body)

    assert response.headers["X-Request-Id"] == "completion-stream-123"
    assert _sse_nonempty_lines(body)[-1] == "data: [DONE]"
    assert [payload["object"] for payload in payloads] == [
        "text_completion",
        "text_completion",
    ]
    assert len({payload["id"] for payload in payloads}) == 1
    assert payloads[0]["id"].startswith("cmpl-")
    assert payloads[0]["id"] != "completion-stream-123"
    assert payloads[0]["choices"][0]["text"] == "hello"
    assert "delta" not in payloads[0]["choices"][0]
    assert payloads[1]["choices"][0]["finish_reason"] == "stop"
    assert payloads[1]["usage"] == {
        "prompt_tokens": 2,
        "completion_tokens": 1,
        "total_tokens": 3,
    }


def test_completions_stream_emits_error_payload_on_aster_error(client: TestClient) -> None:
    async def fake_stream(_request: object):
        if False:
            yield DecodeChunk(token="", index=0, finished=False)
        raise AsterError(
            code="context_length_exceeded",
            message="Request exceeds the configured model context length",
            status_code=400,
            details={"context_length": 4, "requested_tokens": 5},
        )

    client.app.state.container.inference_engine.stream = fake_stream

    with client.stream(
        "POST",
        "/v1/completions",
        headers={"X-Request-Id": "completion-stream-error"},
        json={
            "model": "dummy-model",
            "prompt": "Hello",
            "stream": True,
        },
    ) as response:
        body = b"".join(response.iter_bytes()).decode()

    assert response.status_code == 200
    payloads = _sse_json_payloads(body)
    assert payloads == [
        {
            "error": {
                "message": "Request exceeds the configured model context length",
                "type": "context_length_exceeded",
                "code": "context_length_exceeded",
                "details": {"context_length": 4, "requested_tokens": 5},
            }
        }
    ]
    assert _sse_nonempty_lines(body)[-1] == "data: [DONE]"


def test_completions_stream_emits_error_payload_on_unhandled_error(client: TestClient) -> None:
    async def fake_stream(_request: object):
        if False:
            yield DecodeChunk(token="", index=0, finished=False)
        raise RuntimeError("runtime exploded")

    client.app.state.container.inference_engine.stream = fake_stream

    with client.stream(
        "POST",
        "/v1/completions",
        headers={"X-Request-Id": "completion-stream-unhandled-error"},
        json={
            "model": "dummy-model",
            "prompt": "Hello",
            "stream": True,
        },
    ) as response:
        body = b"".join(response.iter_bytes()).decode()

    assert response.status_code == 200
    payloads = _sse_json_payloads(body)
    assert payloads == [
        {
            "error": {
                "message": "Streaming response failed",
                "type": "stream_failed",
                "code": "stream_failed",
                "details": {},
            }
        }
    ]
    assert _sse_nonempty_lines(body)[-1] == "data: [DONE]"


def test_completions_uses_engine_finish_reason(client: TestClient) -> None:
    async def fake_submit(_request: object) -> SimpleNamespace:
        return SimpleNamespace(
            request_id="completion_length_123",
            text="truncated",
            prompt_tokens=2,
            completion_tokens=2,
            cache_hit=False,
            speculative_enabled=False,
            finish_reason="length",
        )

    client.app.state.container.inference_engine.submit = fake_submit

    response = client.post(
        "/v1/completions",
        json={"model": "dummy-model", "prompt": "Hello"},
    )

    assert response.status_code == 200
    assert response.json()["choices"][0]["finish_reason"] == "length"


def test_completions_forwards_stop_token_ids(client: TestClient) -> None:
    captured: dict[str, object] = {}

    async def fake_submit(request: object) -> SimpleNamespace:
        captured["stop_token_ids"] = request.stop_token_ids
        captured["timeout_seconds"] = request.timeout_seconds
        return SimpleNamespace(
            request_id="completion_stop_tokens",
            text="ok",
            prompt_tokens=2,
            completion_tokens=1,
            cache_hit=False,
            speculative_enabled=False,
            finish_reason="stop",
        )

    client.app.state.container.inference_engine.submit = fake_submit

    response = client.post(
        "/v1/completions",
        json={
            "model": "dummy-model",
            "prompt": "Hello",
            "stop_token_ids": [42],
            "timeout": 9.5,
        },
    )

    assert response.status_code == 200
    assert captured == {"stop_token_ids": (42,), "timeout_seconds": 9.5}


def test_completions_accepts_prompt_arrays(client: TestClient) -> None:
    captured: list[tuple[str, str]] = []

    async def fake_submit(request: object) -> SimpleNamespace:
        captured.append((request.prompt, request.trace_id))
        index = len(captured)
        return SimpleNamespace(
            request_id=request.trace_id,
            text=f"answer {index}",
            prompt_tokens=index,
            completion_tokens=index + 1,
            cache_hit=index == 2,
            speculative_enabled=False,
            finish_reason="stop",
        )

    client.app.state.container.inference_engine.submit = fake_submit

    response = client.post(
        "/v1/completions",
        headers={"X-Request-Id": "completion-batch", "X-Aster-Debug": "1"},
        json={"model": "dummy-model", "prompt": ["first", "second"]},
    )

    assert response.status_code == 200
    assert response.headers["X-Request-Id"] == "completion-batch"
    data = response.json()
    assert data["id"].startswith("cmpl-")
    assert data["id"] != "completion-batch"
    assert data["choices"] == [
        {"index": 0, "text": "answer 1", "finish_reason": "stop"},
        {"index": 1, "text": "answer 2", "finish_reason": "stop"},
    ]
    assert data["usage"] == {
        "prompt_tokens": 3,
        "completion_tokens": 5,
        "total_tokens": 8,
    }
    assert data["aster"]["cache_hit"] is True
    assert captured == [
        ("first", "completion-batch-0"),
        ("second", "completion-batch-1"),
    ]


def test_chat_completions_forwards_sampling_controls(client: TestClient) -> None:
    captured: dict[str, object] = {}

    async def fake_submit(request: object) -> SimpleNamespace:
        captured["top_k"] = request.top_k
        captured["min_p"] = request.min_p
        captured["presence_penalty"] = request.presence_penalty
        captured["frequency_penalty"] = request.frequency_penalty
        captured["repetition_penalty"] = request.repetition_penalty
        captured["stop"] = request.stop
        captured["stop_token_ids"] = request.stop_token_ids
        captured["timeout_seconds"] = request.timeout_seconds
        return SimpleNamespace(
            request_id="sampling_123",
            text="ok",
            prompt_tokens=4,
            completion_tokens=1,
            cache_hit=False,
            speculative_enabled=False,
        )

    client.app.state.container.inference_engine.submit = fake_submit

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "dummy-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "top_k": 40,
            "min_p": 0.05,
            "presence_penalty": 0.2,
            "frequency_penalty": 0.3,
            "repetition_penalty": 1.1,
            "stop": ["</answer>"],
            "stop_token_ids": [128001, 128009],
            "timeout": 12.5,
        },
    )

    assert response.status_code == 200
    assert captured == {
        "top_k": 40,
        "min_p": 0.05,
        "presence_penalty": 0.2,
        "frequency_penalty": 0.3,
        "repetition_penalty": 1.1,
        "stop": ["</answer>"],
        "stop_token_ids": (128001, 128009),
        "timeout_seconds": 12.5,
    }


def test_chat_completions_accepts_null_max_tokens_as_default(client: TestClient) -> None:
    captured: dict[str, object] = {}

    async def fake_submit(request: object) -> SimpleNamespace:
        captured["max_tokens"] = request.max_tokens
        return SimpleNamespace(
            request_id="nullable-max-tokens",
            text="ok",
            prompt_tokens=3,
            completion_tokens=1,
            cache_hit=False,
            speculative_enabled=False,
        )

    client.app.state.container.inference_engine.submit = fake_submit

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "dummy-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": None,
        },
    )

    assert response.status_code == 200
    assert captured == {"max_tokens": 256}


def test_chat_completions_uses_max_completion_tokens_alias(client: TestClient) -> None:
    captured: dict[str, object] = {}

    async def fake_submit(request: object) -> SimpleNamespace:
        captured["max_tokens"] = request.max_tokens
        return SimpleNamespace(
            request_id="max-completion-tokens",
            text="ok",
            prompt_tokens=3,
            completion_tokens=1,
            cache_hit=False,
            speculative_enabled=False,
        )

    client.app.state.container.inference_engine.submit = fake_submit

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "dummy-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": None,
            "max_completion_tokens": 17,
        },
    )

    assert response.status_code == 200
    assert captured == {"max_tokens": 17}


def test_chat_completions_forwards_chat_template_kwargs(client: TestClient) -> None:
    captured: dict[str, object] = {}

    async def fake_submit(request: object) -> SimpleNamespace:
        captured["enable_thinking"] = request.enable_thinking
        captured["chat_template_kwargs"] = request.chat_template_kwargs
        return SimpleNamespace(
            request_id="chat-template-kwargs",
            text="ok",
            prompt_tokens=3,
            completion_tokens=1,
            cache_hit=False,
            speculative_enabled=False,
        )

    client.app.state.container.inference_engine.submit = fake_submit

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "dummy-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "chat_template_kwargs": {"enable_thinking": True, "custom_flag": "enabled"},
        },
    )

    assert response.status_code == 200
    assert captured == {
        "enable_thinking": True,
        "chat_template_kwargs": {"enable_thinking": True, "custom_flag": "enabled"},
    }


def test_chat_completions_enable_thinking_overrides_chat_template_kwargs(
    client: TestClient,
) -> None:
    captured: dict[str, object] = {}

    async def fake_submit(request: object) -> SimpleNamespace:
        captured["enable_thinking"] = request.enable_thinking
        captured["chat_template_kwargs"] = request.chat_template_kwargs
        return SimpleNamespace(
            request_id="chat-template-enable-thinking",
            text="ok",
            prompt_tokens=3,
            completion_tokens=1,
            cache_hit=False,
            speculative_enabled=False,
        )

    client.app.state.container.inference_engine.submit = fake_submit

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "dummy-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "enable_thinking": False,
            "chat_template_kwargs": {"enable_thinking": True, "custom_flag": "enabled"},
        },
    )

    assert response.status_code == 200
    assert captured == {
        "enable_thinking": False,
        "chat_template_kwargs": {"enable_thinking": False, "custom_flag": "enabled"},
    }


def test_chat_completions_forwards_thinking_token_budget(client: TestClient) -> None:
    captured: dict[str, object] = {}

    async def fake_submit(request: object) -> SimpleNamespace:
        captured["enable_thinking"] = request.enable_thinking
        captured["thinking_token_budget"] = request.thinking_token_budget
        return SimpleNamespace(
            request_id="thinking-token-budget",
            text="ok",
            prompt_tokens=3,
            completion_tokens=1,
            cache_hit=False,
            speculative_enabled=False,
        )

    client.app.state.container.inference_engine.submit = fake_submit

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "dummy-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "enable_thinking": True,
            "thinking_token_budget": 32,
        },
    )

    assert response.status_code == 200
    assert captured == {"enable_thinking": True, "thinking_token_budget": 32}


def test_openai_chat_provider_path_uses_max_completion_tokens_alias(client: TestClient) -> None:
    captured: dict[str, object] = {}

    async def fake_submit(request: object) -> SimpleNamespace:
        captured["max_tokens"] = request.max_tokens
        return SimpleNamespace(
            request_id="provider-max-completion-tokens",
            text='{"answer":"ok"}',
            prompt_tokens=4,
            completion_tokens=2,
            cache_hit=False,
            speculative_enabled=False,
            finish_reason="stop",
        )

    client.app.state.container.inference_engine.submit = fake_submit

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "dummy-model",
            "messages": [{"role": "user", "content": "Return JSON"}],
            "max_completion_tokens": 19,
            "response_format": {"type": "json_object"},
        },
    )

    assert response.status_code == 200
    assert captured == {"max_tokens": 19}


def test_openai_chat_provider_path_forwards_chat_template_kwargs(client: TestClient) -> None:
    captured: dict[str, object] = {}

    async def fake_submit(request: object) -> SimpleNamespace:
        captured["enable_thinking"] = request.enable_thinking
        captured["chat_template_kwargs"] = request.chat_template_kwargs
        return SimpleNamespace(
            request_id="provider-chat-template-kwargs",
            text='{"answer":"ok"}',
            prompt_tokens=4,
            completion_tokens=2,
            cache_hit=False,
            speculative_enabled=False,
            finish_reason="stop",
        )

    client.app.state.container.inference_engine.submit = fake_submit

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "dummy-model",
            "messages": [{"role": "user", "content": "Return JSON"}],
            "response_format": {"type": "json_object"},
            "chat_template_kwargs": {"enable_thinking": True, "custom_flag": "enabled"},
        },
    )

    assert response.status_code == 200
    assert captured == {
        "enable_thinking": True,
        "chat_template_kwargs": {"enable_thinking": True, "custom_flag": "enabled"},
    }


def test_openai_chat_provider_path_uses_model_enable_thinking_default(
    client: TestClient,
) -> None:
    client.app.state.container.settings.model.enable_thinking = True
    captured: dict[str, object] = {}

    async def fake_submit(request: object) -> SimpleNamespace:
        captured["enable_thinking"] = request.enable_thinking
        captured["chat_template_kwargs"] = request.chat_template_kwargs
        return SimpleNamespace(
            request_id="provider-default-thinking",
            text='{"answer":"ok"}',
            prompt_tokens=4,
            completion_tokens=2,
            cache_hit=False,
            speculative_enabled=False,
            finish_reason="stop",
        )

    client.app.state.container.inference_engine.submit = fake_submit

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "dummy-model",
            "messages": [{"role": "user", "content": "Return JSON"}],
            "response_format": {"type": "json_object"},
        },
    )

    assert response.status_code == 200
    assert captured == {
        "enable_thinking": True,
        "chat_template_kwargs": {"enable_thinking": True},
    }


def test_openai_chat_provider_path_preserves_explicit_enable_thinking_false(
    client: TestClient,
) -> None:
    client.app.state.container.settings.model.enable_thinking = True
    captured: dict[str, object] = {}

    async def fake_submit(request: object) -> SimpleNamespace:
        captured["enable_thinking"] = request.enable_thinking
        captured["chat_template_kwargs"] = request.chat_template_kwargs
        return SimpleNamespace(
            request_id="provider-explicit-thinking-false",
            text='{"answer":"ok"}',
            prompt_tokens=4,
            completion_tokens=2,
            cache_hit=False,
            speculative_enabled=False,
            finish_reason="stop",
        )

    client.app.state.container.inference_engine.submit = fake_submit

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "dummy-model",
            "messages": [{"role": "user", "content": "Return JSON"}],
            "response_format": {"type": "json_object"},
            "enable_thinking": False,
        },
    )

    assert response.status_code == 200
    assert captured == {
        "enable_thinking": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def test_openai_chat_provider_path_forwards_thinking_token_budget(client: TestClient) -> None:
    captured: dict[str, object] = {}

    async def fake_submit(request: object) -> SimpleNamespace:
        captured["enable_thinking"] = request.enable_thinking
        captured["thinking_token_budget"] = request.thinking_token_budget
        return SimpleNamespace(
            request_id="provider-thinking-token-budget",
            text='{"answer":"ok"}',
            prompt_tokens=4,
            completion_tokens=2,
            cache_hit=False,
            speculative_enabled=False,
            finish_reason="stop",
        )

    client.app.state.container.inference_engine.submit = fake_submit

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "dummy-model",
            "messages": [{"role": "user", "content": "Return JSON"}],
            "response_format": {"type": "json_object"},
            "enable_thinking": True,
            "thinking_token_budget": 24,
        },
    )

    assert response.status_code == 200
    assert captured == {"enable_thinking": True, "thinking_token_budget": 24}


def test_completions_accepts_null_max_tokens_as_default(client: TestClient) -> None:
    captured: dict[str, object] = {}

    async def fake_submit(request: object) -> SimpleNamespace:
        captured["max_tokens"] = request.max_tokens
        return SimpleNamespace(
            request_id="completion-null-max",
            text="ok",
            prompt_tokens=2,
            completion_tokens=1,
            cache_hit=False,
            speculative_enabled=False,
            finish_reason="stop",
        )

    client.app.state.container.inference_engine.submit = fake_submit

    response = client.post(
        "/v1/completions",
        json={"model": "dummy-model", "prompt": "Hello", "max_tokens": None},
    )

    assert response.status_code == 200
    assert captured == {"max_tokens": 256}


def test_chat_completions_rejects_non_positive_max_tokens(client: TestClient) -> None:
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "dummy-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 0,
        },
    )

    assert response.status_code == 422


def test_chat_completions_rejects_non_positive_thinking_token_budget(
    client: TestClient,
) -> None:
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "dummy-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "thinking_token_budget": 0,
        },
    )

    assert response.status_code == 422


def test_chat_completions_rejects_max_tokens_over_server_limit(client: TestClient) -> None:
    client.app.state.container.settings.api.max_request_tokens = 2

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "dummy-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 3,
        },
    )

    assert response.status_code == 400
    assert response.json()["error"] == {
        "type": "max_tokens_exceeded",
        "code": "max_tokens_exceeded",
        "message": "max_tokens exceeds server limit (2)",
        "details": {"max_tokens": 3, "max_request_tokens": 2},
    }


def test_completions_rejects_non_positive_max_tokens(client: TestClient) -> None:
    response = client.post(
        "/v1/completions",
        json={"model": "dummy-model", "prompt": "Hello", "max_tokens": 0},
    )

    assert response.status_code == 422


def test_completions_rejects_max_tokens_over_server_limit(client: TestClient) -> None:
    client.app.state.container.settings.api.max_request_tokens = 2

    response = client.post(
        "/v1/completions",
        json={"model": "dummy-model", "prompt": "Hello", "max_tokens": 3},
    )

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "max_tokens_exceeded"
    assert response.json()["error"]["details"] == {"max_tokens": 3, "max_request_tokens": 2}


def test_openai_responses_rejects_max_output_tokens_over_server_limit(client: TestClient) -> None:
    client.app.state.container.settings.api.max_request_tokens = 2

    response = client.post(
        "/v1/responses",
        json={"model": "dummy-model", "input": "Hello", "max_output_tokens": 3},
    )

    assert response.status_code == 400
    assert response.json()["error"] == {
        "message": "max_tokens exceeds server limit (2)",
        "type": "max_tokens_exceeded",
        "code": "max_tokens_exceeded",
    }


def test_chat_completions_rejects_media_placeholders(client: TestClient) -> None:
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "dummy-model",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this"},
                        {"type": "image_url", "image_url": {"url": "https://example.com/test.png"}},
                    ],
                }
            ],
            "stream": False,
        },
    )

    assert response.status_code == 400
    data = response.json()
    assert data["error"]["type"] == "multimodal_not_supported"


def test_chat_completions_rejects_input_image_without_placeholder(
    client: TestClient,
) -> None:
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "dummy-model",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "Describe this"},
                        {"type": "input_image", "image_url": "https://example.com/test.png"},
                    ],
                }
            ],
        },
    )

    assert response.status_code == 400
    data = response.json()
    assert data["error"]["type"] == "multimodal_not_supported"


def test_openai_responses_endpoint_uses_local_runtime_and_returns_response_shape(
    client: TestClient,
) -> None:
    async def fake_submit(_request: object) -> SimpleNamespace:
        return SimpleNamespace(
            request_id="resp_123",
            text="hello from local runtime",
            prompt_tokens=10,
            completion_tokens=4,
            cache_hit=False,
            speculative_enabled=False,
        )

    client.app.state.container.inference_engine.submit = fake_submit

    response = client.post(
        "/v1/responses",
        json={
            "model": "gpt-5.1",
            "input": [{"role": "user", "content": [{"type": "input_text", "text": "Hello"}]}],
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["id"].startswith("resp_")
    assert data["object"] == "response"
    assert data["output"][0]["content"][0]["text"] == "hello from local runtime"


def test_openai_responses_endpoint_maps_length_finish_to_incomplete(
    client: TestClient,
) -> None:
    async def fake_submit(_request: object) -> SimpleNamespace:
        return SimpleNamespace(
            request_id="resp_length",
            text="truncated",
            prompt_tokens=10,
            completion_tokens=4,
            cache_hit=False,
            speculative_enabled=False,
            finish_reason="length",
        )

    client.app.state.container.inference_engine.submit = fake_submit

    response = client.post(
        "/v1/responses",
        json={
            "model": "gpt-5.1",
            "input": "Return a short answer",
            "max_output_tokens": 4,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "incomplete"
    assert data["incomplete_details"] == {"reason": "max_output_tokens"}
    assert data["usage"] == _responses_usage(10, 4)


def test_openai_responses_includes_instructions_as_system_message(
    client: TestClient,
) -> None:
    captured_messages: list[dict[str, str]] = []

    async def fake_submit(request: object) -> SimpleNamespace:
        captured_messages.extend(getattr(request, "messages", []) or [])
        return SimpleNamespace(
            request_id="resp_instructions_123",
            text="ok",
            prompt_tokens=12,
            completion_tokens=1,
            cache_hit=False,
            speculative_enabled=False,
        )

    client.app.state.container.inference_engine.submit = fake_submit

    response = client.post(
        "/v1/responses",
        json={
            "model": "gpt-5.1",
            "instructions": "Answer in terse JSON.",
            "max_output_tokens": 7,
            "metadata": {"trace": "abc"},
            "parallel_tool_calls": False,
            "store": False,
            "truncation": "auto",
            "input": "Hello",
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert captured_messages == [
        {"role": "system", "content": "Answer in terse JSON."},
        {"role": "user", "content": "Hello"},
    ]
    assert data["instructions"] == "Answer in terse JSON."
    assert data["max_output_tokens"] == 7
    assert data["metadata"] == {"trace": "abc"}
    assert data["parallel_tool_calls"] is False
    assert data["store"] is False
    assert data["truncation"] == "auto"
    assert data["text"] == {"format": {"type": "text"}}


@pytest.mark.parametrize("endpoint", ["/v1/responses", "/xai/v1/responses"])
def test_responses_previous_response_id_replays_stored_history(
    client: TestClient,
    endpoint: str,
) -> None:
    calls: list[list[dict[str, object]]] = []
    outputs = iter(["First answer", "Second answer"])

    async def fake_submit(request: object) -> SimpleNamespace:
        calls.append(list(getattr(request, "messages", []) or []))
        return SimpleNamespace(
            request_id=getattr(request, "trace_id", "resp"),
            text=next(outputs),
            prompt_tokens=12,
            completion_tokens=2,
            cache_hit=False,
            speculative_enabled=False,
        )

    client.app.state.container.inference_engine.submit = fake_submit

    first = client.post(
        endpoint,
        headers={"X-Request-Id": "resp-first"},
        json={"model": "gpt-5.1", "input": "First"},
    )
    assert first.status_code == 200
    first_id = first.json()["id"]
    second = client.post(
        endpoint,
        headers={"X-Request-Id": "resp-second"},
        json={
            "model": "gpt-5.1",
            "previous_response_id": first_id,
            "instructions": "Only this turn.",
            "input": "Second",
        },
    )

    assert second.status_code == 200
    assert first.headers["X-Request-Id"] == "resp-first"
    assert second.headers["X-Request-Id"] == "resp-second"
    assert first_id.startswith("resp_")
    assert first_id != "resp-first"
    assert second.json()["id"].startswith("resp_")
    assert second.json()["id"] != "resp-second"
    assert calls[1] == [
        {"role": "user", "content": "First"},
        {"role": "assistant", "content": "First answer"},
        {"role": "system", "content": "Only this turn."},
        {"role": "user", "content": "Second"},
    ]
    assert second.json()["previous_response_id"] == first_id


def test_openai_responses_incomplete_response_replays_stored_history(
    client: TestClient,
) -> None:
    calls: list[list[dict[str, object]]] = []
    outputs = iter(
        [
            SimpleNamespace(
                request_id="resp-incomplete-first",
                text="Truncated answer",
                prompt_tokens=10,
                completion_tokens=1,
                cache_hit=False,
                speculative_enabled=False,
                finish_reason="length",
            ),
            SimpleNamespace(
                request_id="resp-incomplete-second",
                text="Follow up",
                prompt_tokens=12,
                completion_tokens=2,
                cache_hit=False,
                speculative_enabled=False,
                finish_reason="stop",
            ),
        ]
    )

    async def fake_submit(request: object) -> SimpleNamespace:
        calls.append(list(getattr(request, "messages", []) or []))
        return next(outputs)

    client.app.state.container.inference_engine.submit = fake_submit

    first = client.post(
        "/v1/responses",
        json={"model": "gpt-5.1", "input": "First", "max_output_tokens": 1},
    )
    assert first.status_code == 200
    first_data = first.json()
    first_id = first_data["id"]
    assert first_data["status"] == "incomplete"
    assert first_data["incomplete_details"] == {"reason": "max_output_tokens"}

    second = client.post(
        "/v1/responses",
        json={
            "model": "gpt-5.1",
            "previous_response_id": first_id,
            "input": "Second",
        },
    )

    assert second.status_code == 200
    assert calls[1] == [
        {"role": "user", "content": "First"},
        {"role": "assistant", "content": "Truncated answer"},
        {"role": "user", "content": "Second"},
    ]
    assert second.json()["previous_response_id"] == first_id


@pytest.mark.parametrize("endpoint", ["/v1/responses", "/xai/v1/responses"])
def test_responses_previous_response_id_returns_404_for_missing_history(
    client: TestClient,
    endpoint: str,
) -> None:
    response = client.post(
        endpoint,
        headers={"X-Request-Id": "missing-response"},
        json={
            "model": "gpt-5.1",
            "previous_response_id": "resp-missing",
            "input": "Second",
        },
    )

    assert response.status_code == 404
    assert response.headers["X-Request-Id"] == "missing-response"
    data = response.json()
    assert data["error"] == {
        "message": "Previous response `resp-missing` not found.",
        "type": "response_not_found",
        "code": "response_not_found",
    }


def test_responses_previous_response_id_is_scoped_to_provider_endpoint(
    client: TestClient,
) -> None:
    async def fake_submit(request: object) -> SimpleNamespace:
        return SimpleNamespace(
            request_id=getattr(request, "trace_id", "resp"),
            text="OpenAI answer",
            prompt_tokens=12,
            completion_tokens=2,
            cache_hit=False,
            speculative_enabled=False,
        )

    client.app.state.container.inference_engine.submit = fake_submit

    first = client.post(
        "/v1/responses",
        headers={"X-Request-Id": "resp-openai-first"},
        json={"model": "gpt-5.1", "input": "First"},
    )
    assert first.status_code == 200
    first_id = first.json()["id"]

    cross_provider = client.post(
        "/xai/v1/responses",
        headers={"X-Request-Id": "resp-xai-cross-provider"},
        json={
            "model": "grok-4",
            "previous_response_id": first_id,
            "input": "Second",
        },
    )

    assert cross_provider.status_code == 404
    assert cross_provider.headers["X-Request-Id"] == "resp-xai-cross-provider"
    assert cross_provider.json()["error"] == {
        "message": f"Previous response `{first_id}` not found.",
        "type": "response_not_found",
        "code": "response_not_found",
    }


@pytest.mark.parametrize("previous_response_id", [False, [], {}])
@pytest.mark.parametrize("endpoint", ["/v1/responses", "/xai/v1/responses"])
def test_responses_previous_response_id_rejects_non_string_values(
    client: TestClient,
    previous_response_id: object,
    endpoint: str,
) -> None:
    response = client.post(
        endpoint,
        headers={"X-Request-Id": "invalid-response-link"},
        json={
            "model": "gpt-5.1",
            "previous_response_id": previous_response_id,
            "input": "Second",
        },
    )

    assert response.status_code == 400
    assert response.headers["X-Request-Id"] == "invalid-response-link"
    assert response.json()["error"] == {
        "message": "previous_response_id must be a string.",
        "type": "invalid_previous_response_id",
        "code": "invalid_previous_response_id",
    }


def test_openai_responses_store_false_does_not_persist_history(
    client: TestClient,
) -> None:
    async def fake_submit(request: object) -> SimpleNamespace:
        return SimpleNamespace(
            request_id=getattr(request, "trace_id", "resp"),
            text="Do not store",
            prompt_tokens=12,
            completion_tokens=2,
            cache_hit=False,
            speculative_enabled=False,
        )

    client.app.state.container.inference_engine.submit = fake_submit

    first = client.post(
        "/v1/responses",
        headers={"X-Request-Id": "resp-nostore"},
        json={"model": "gpt-5.1", "input": "First", "store": False},
    )
    assert first.status_code == 200
    first_id = first.json()["id"]
    second = client.post(
        "/v1/responses",
        json={
            "model": "gpt-5.1",
            "previous_response_id": first_id,
            "input": "Second",
        },
    )

    assert first.headers["X-Request-Id"] == "resp-nostore"
    assert first_id.startswith("resp_")
    assert first_id != "resp-nostore"
    assert second.status_code == 404
    assert second.json()["error"]["type"] == "response_not_found"


def test_openai_responses_store_metrics_track_replay_lifecycle(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("""
api:
  host: 127.0.0.1
  port: 8000
  responses_store_max_entries: 1
logging:
  level: INFO
model:
  name: dummy-model
  path: "dummy"
  runtime: mlx
audio:
  asr_enabled: false
  tts_enabled: false
""")
    app = create_application(str(config_path))
    outputs = iter(["First answer", "Second answer"])

    async def fake_submit(request: object) -> SimpleNamespace:
        return SimpleNamespace(
            request_id=getattr(request, "trace_id", "resp"),
            text=next(outputs),
            prompt_tokens=12,
            completion_tokens=2,
            cache_hit=False,
            speculative_enabled=False,
        )

    with TestClient(app) as store_client:
        store_client.app.state.container.inference_engine.submit = fake_submit

        first = store_client.post(
            "/v1/responses", json={"model": "gpt-5.1", "input": "First"}
        )
        assert first.status_code == 200
        first_id = first.json()["id"]

        second = store_client.post(
            "/v1/responses",
            json={
                "model": "gpt-5.1",
                "previous_response_id": first_id,
                "input": "Second",
            },
        )
        assert second.status_code == 200

        missing = store_client.post(
            "/v1/responses",
            json={
                "model": "gpt-5.1",
                "previous_response_id": first_id,
                "input": "Third",
            },
        )
        assert missing.status_code == 404

        metrics = store_client.get("/metrics").text
        status = store_client.get("/v1/status").json()["responses_store"]

    assert "aster_responses_store_hits_total 1.0" in metrics
    assert "aster_responses_store_misses_total 1.0" in metrics
    assert "aster_responses_store_writes_total 2.0" in metrics
    assert "aster_responses_store_evictions_total 1.0" in metrics
    assert "aster_responses_store_entries 1.0" in metrics
    assert status == {
        "entries": 1,
        "max_entries": 1,
        "scope": "process/provider",
    }


def test_openai_responses_input_replays_function_calls_and_tool_outputs(
    client: TestClient,
) -> None:
    captured_messages: list[dict[str, object]] = []

    async def fake_submit(request: object) -> SimpleNamespace:
        captured_messages.extend(getattr(request, "messages", []) or [])
        return SimpleNamespace(
            request_id="resp_tool_replay",
            text="final answer",
            prompt_tokens=18,
            completion_tokens=2,
            cache_hit=False,
            speculative_enabled=False,
        )

    client.app.state.container.inference_engine.submit = fake_submit

    response = client.post(
        "/v1/responses",
        json={
            "model": "gpt-5.1",
            "input": [
                {
                    "type": "message",
                    "role": "developer",
                    "content": [{"type": "input_text", "text": "Be exact."}],
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Weather?"}],
                },
                {
                    "type": "function_call",
                    "call_id": "call_weather",
                    "name": "lookup_weather",
                    "arguments": "{\"city\":\"Shanghai\"}",
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_weather",
                    "output": "Sunny",
                },
                {
                    "type": "reasoning",
                    "content": [{"type": "reasoning_text", "text": "Tool result is relevant."}],
                },
            ],
        },
    )

    assert response.status_code == 200
    assert captured_messages == [
        {"role": "system", "content": "Be exact."},
        {"role": "user", "content": "Weather?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_weather",
                    "type": "function",
                    "function": {
                        "name": "lookup_weather",
                        "arguments": "{\"city\":\"Shanghai\"}",
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_weather", "content": "Sunny"},
        {"role": "assistant", "content": "Tool result is relevant."},
    ]


def test_openai_responses_rejects_input_image_without_placeholder(
    client: TestClient,
) -> None:
    response = client.post(
        "/v1/responses",
        json={
            "model": "gpt-5.1",
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "Describe this"},
                        {"type": "input_image", "image_url": "https://example.com/test.png"},
                    ],
                }
            ],
        },
    )

    assert response.status_code == 400
    data = response.json()
    assert data["error"]["type"] == "multimodal_not_supported"


def test_openai_responses_endpoint_includes_reasoning_item(client: TestClient) -> None:
    async def fake_submit(_request: object) -> SimpleNamespace:
        return SimpleNamespace(
            request_id="resp_reasoning_123",
            text="<think>scratch</think>hello",
            prompt_tokens=10,
            completion_tokens=4,
            cache_hit=False,
            speculative_enabled=False,
        )

    client.app.state.container.inference_engine.submit = fake_submit

    response = client.post(
        "/v1/responses",
        json={"model": "gpt-5.1", "input": "Hello"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["output"][0]["type"] == "reasoning"
    assert data["output"][0]["content"][0]["text"] == "scratch"
    assert data["output"][1]["content"][0]["text"] == "hello"


def test_anthropic_messages_endpoint_returns_anthropic_shape(client: TestClient) -> None:
    async def fake_submit(_request: object) -> SimpleNamespace:
        return SimpleNamespace(
            request_id="msg_123",
            text="hello anthropic",
            prompt_tokens=12,
            completion_tokens=3,
            cache_hit=False,
            speculative_enabled=False,
        )

    client.app.state.container.inference_engine.submit = fake_submit

    response = client.post(
        "/v1/messages",
        json={
            "model": "claude-sonnet-4-5",
            "system": "Be concise.",
            "messages": [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}],
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["type"] == "message"
    assert data["content"][0]["text"] == "hello anthropic"


def test_anthropic_messages_maps_stop_sequence_finish_reason(client: TestClient) -> None:
    async def fake_submit(_request: object) -> SimpleNamespace:
        return SimpleNamespace(
            request_id="msg_stop_seq_123",
            text="hello anthropic",
            prompt_tokens=12,
            completion_tokens=3,
            finish_reason="stop_sequence",
            stop_sequence="\n\nHuman:",
            cache_hit=False,
            speculative_enabled=False,
        )

    client.app.state.container.inference_engine.submit = fake_submit

    response = client.post(
        "/v1/messages",
        json={
            "model": "claude-sonnet-4-5",
            "messages": [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}],
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["stop_reason"] == "stop_sequence"
    assert data["stop_sequence"] == "\n\nHuman:"


def test_gemini_generate_content_endpoint_returns_candidate_shape(client: TestClient) -> None:
    async def fake_submit(_request: object) -> SimpleNamespace:
        return SimpleNamespace(
            request_id="gem_123",
            text="hello gemini",
            prompt_tokens=8,
            completion_tokens=2,
            cache_hit=False,
            speculative_enabled=False,
        )

    client.app.state.container.inference_engine.submit = fake_submit

    response = client.post(
        "/v1beta/models/gemini-2.5-flash:generateContent",
        json={"contents": [{"role": "user", "parts": [{"text": "Hello"}]}]},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["candidates"][0]["content"]["parts"][0]["text"] == "hello gemini"


def test_cohere_chat_endpoint_returns_cohere_shape(client: TestClient) -> None:
    async def fake_submit(_request: object) -> SimpleNamespace:
        return SimpleNamespace(
            request_id="co_123",
            text="hello cohere",
            prompt_tokens=9,
            completion_tokens=2,
            cache_hit=False,
            speculative_enabled=False,
        )

    client.app.state.container.inference_engine.submit = fake_submit

    response = client.post(
        "/v2/chat",
        json={
            "model": "command-a",
            "messages": [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}],
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["message"]["content"][0]["text"] == "hello cohere"


def test_bedrock_converse_maps_content_filter_finish_reason(client: TestClient) -> None:
    async def fake_submit(_request: object) -> SimpleNamespace:
        return SimpleNamespace(
            request_id="bedrock_123",
            text="blocked",
            prompt_tokens=7,
            completion_tokens=1,
            finish_reason="content_filter",
            cache_hit=False,
            speculative_enabled=False,
        )

    client.app.state.container.inference_engine.submit = fake_submit

    response = client.post(
        "/model/anthropic.claude-3-haiku/converse",
        json={"messages": [{"role": "user", "content": [{"text": "Hello"}]}]},
    )

    assert response.status_code == 200
    assert response.json()["stopReason"] == "content_filtered"


def test_anthropic_stream_maps_length_finish_reason(client: TestClient) -> None:
    async def fake_stream(_request: object):
        yield DecodeChunk(token="hello", index=0, finished=False)
        yield DecodeChunk(
            token="",
            index=1,
            finished=True,
            stats={"prompt_tokens": 5, "completion_tokens": 1, "finish_reason": "length"},
        )

    client.app.state.container.inference_engine.stream = fake_stream

    with client.stream(
        "POST",
        "/v1/messages",
        json={
            "model": "claude-sonnet-4-5",
            "stream": True,
            "messages": [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}],
        },
    ) as response:
        body = b"".join(response.iter_bytes()).decode()

    assert response.status_code == 200
    events = _sse_events(body)
    message_delta = next(payload for name, payload in events if name == "message_delta")
    assert message_delta["delta"] == {"stop_reason": "max_tokens", "stop_sequence": None}


def test_anthropic_stream_maps_stop_sequence_finish_reason(client: TestClient) -> None:
    async def fake_stream(_request: object):
        yield DecodeChunk(token="hello", index=0, finished=False)
        yield DecodeChunk(
            token="",
            index=1,
            finished=True,
            stats={
                "prompt_tokens": 5,
                "completion_tokens": 1,
                "finish_reason": "stop_sequence",
                "stop_sequence": "\n\nHuman:",
            },
        )

    client.app.state.container.inference_engine.stream = fake_stream

    with client.stream(
        "POST",
        "/v1/messages",
        json={
            "model": "claude-sonnet-4-5",
            "stream": True,
            "messages": [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}],
        },
    ) as response:
        body = b"".join(response.iter_bytes()).decode()

    assert response.status_code == 200
    events = _sse_events(body)
    message_delta = next(payload for name, payload in events if name == "message_delta")
    assert message_delta["delta"] == {"stop_reason": "stop_sequence", "stop_sequence": "\n\nHuman:"}


def test_gemini_stream_maps_length_finish_reason(client: TestClient) -> None:
    async def fake_stream(_request: object):
        yield DecodeChunk(token="hello", index=0, finished=False)
        yield DecodeChunk(
            token="",
            index=1,
            finished=True,
            stats={"prompt_tokens": 5, "completion_tokens": 1, "finish_reason": "length"},
        )

    client.app.state.container.inference_engine.stream = fake_stream

    with client.stream(
        "POST",
        "/v1beta/models/gemini-2.5-flash:streamGenerateContent",
        json={"contents": [{"role": "user", "parts": [{"text": "Hello"}]}]},
    ) as response:
        body = b"".join(response.iter_bytes()).decode()

    assert response.status_code == 200
    payloads = _sse_json_payloads(body)
    assert payloads[-1]["candidates"] == [{"index": 0, "finishReason": "MAX_TOKENS"}]
    assert payloads[-1]["usageMetadata"] == {
        "promptTokenCount": 5,
        "candidatesTokenCount": 1,
        "totalTokenCount": 6,
    }


def test_gemini_stream_maps_safety_finish_reason(client: TestClient) -> None:
    async def fake_stream(_request: object):
        yield DecodeChunk(token="hello", index=0, finished=False)
        yield DecodeChunk(
            token="",
            index=1,
            finished=True,
            stats={"prompt_tokens": 5, "completion_tokens": 1, "finish_reason": "content_filter"},
        )

    client.app.state.container.inference_engine.stream = fake_stream

    with client.stream(
        "POST",
        "/v1beta/models/gemini-2.5-flash:streamGenerateContent",
        json={"contents": [{"role": "user", "parts": [{"text": "Hello"}]}]},
    ) as response:
        body = b"".join(response.iter_bytes()).decode()

    assert response.status_code == 200
    payloads = _sse_json_payloads(body)
    assert payloads[-1]["candidates"] == [{"index": 0, "finishReason": "SAFETY"}]


def test_cohere_stream_maps_length_finish_reason(client: TestClient) -> None:
    async def fake_stream(_request: object):
        yield DecodeChunk(token="hello", index=0, finished=False)
        yield DecodeChunk(
            token="",
            index=1,
            finished=True,
            stats={"prompt_tokens": 5, "completion_tokens": 1, "finish_reason": "length"},
        )

    client.app.state.container.inference_engine.stream = fake_stream

    with client.stream(
        "POST",
        "/v2/chat",
        json={
            "model": "command-a",
            "stream": True,
            "messages": [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}],
        },
    ) as response:
        body = b"".join(response.iter_bytes()).decode()

    assert response.status_code == 200
    payloads = _sse_json_payloads(body)
    assert payloads[-1] == {"type": "message-end", "finish_reason": "MAX_TOKENS"}


def test_cohere_stream_maps_stop_sequence_finish_reason(client: TestClient) -> None:
    async def fake_stream(_request: object):
        yield DecodeChunk(token="hello", index=0, finished=False)
        yield DecodeChunk(
            token="",
            index=1,
            finished=True,
            stats={"prompt_tokens": 5, "completion_tokens": 1, "finish_reason": "stop_sequence"},
        )

    client.app.state.container.inference_engine.stream = fake_stream

    with client.stream(
        "POST",
        "/v2/chat",
        json={
            "model": "command-a",
            "stream": True,
            "messages": [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}],
        },
    ) as response:
        body = b"".join(response.iter_bytes()).decode()

    assert response.status_code == 200
    payloads = _sse_json_payloads(body)
    assert payloads[-1] == {"type": "message-end", "finish_reason": "STOP_SEQUENCE"}


def test_openai_responses_stream_endpoint_returns_response_events(client: TestClient) -> None:
    async def fake_stream(_request: object):
        yield DecodeChunk(token="Hel", index=0, finished=False)
        yield DecodeChunk(
            token="",
            index=1,
            finished=True,
            stats={"prompt_tokens": 10, "completion_tokens": 1},
        )

    client.app.state.container.inference_engine.stream = fake_stream

    with client.stream(
        "POST",
        "/v1/responses",
        json={
            "model": "gpt-5.1",
            "stream": True,
            "instructions": "Use short answers.",
            "metadata": {"trace": "stream"},
            "parallel_tool_calls": False,
            "input": [{"role": "user", "content": [{"type": "input_text", "text": "Hello"}]}],
        },
    ) as response:
        body = b"".join(response.iter_bytes()).decode()

    assert response.status_code == 200
    events = _sse_events(body)
    event_names = [name for name, _ in events]

    assert event_names == [
        "response.created",
        "response.in_progress",
        "response.output_item.added",
        "response.content_part.added",
        "response.output_text.delta",
        "response.output_text.done",
        "response.content_part.done",
        "response.output_item.done",
        "response.completed",
    ]
    assert [payload["sequence_number"] for _, payload in events] == list(range(1, len(events) + 1))
    completed = events[-1][1]["response"]
    assert isinstance(completed, dict)
    assert completed["output_text"] == "Hel"
    assert completed["usage"] == _responses_usage(10, 1)
    assert completed["instructions"] == "Use short answers."
    assert completed["metadata"] == {"trace": "stream"}
    assert completed["parallel_tool_calls"] is False


def test_openai_responses_stream_maps_length_finish_to_incomplete(client: TestClient) -> None:
    async def fake_stream(_request: object):
        yield DecodeChunk(token="Hel", index=0, finished=False)
        yield DecodeChunk(
            token="",
            index=1,
            finished=True,
            stats={"prompt_tokens": 10, "completion_tokens": 1, "finish_reason": "length"},
        )

    client.app.state.container.inference_engine.stream = fake_stream

    with client.stream(
        "POST",
        "/v1/responses",
        json={
            "model": "gpt-5.1",
            "stream": True,
            "input": "Hello",
            "max_output_tokens": 1,
        },
    ) as response:
        body = b"".join(response.iter_bytes()).decode()

    assert response.status_code == 200
    events = _sse_events(body)
    assert events[-1][0] == "response.completed"
    completed = events[-1][1]["response"]
    assert completed["status"] == "incomplete"
    assert completed["incomplete_details"] == {"reason": "max_output_tokens"}
    assert completed["usage"] == _responses_usage(10, 1)


def test_openai_responses_stream_persists_history_for_previous_response_id(
    client: TestClient,
) -> None:
    captured_messages: list[list[dict[str, object]]] = []

    async def fake_stream(_request: object):
        yield DecodeChunk(token="streamed ", index=0, finished=False)
        yield DecodeChunk(token="answer", index=1, finished=False)
        yield DecodeChunk(
            token="",
            index=2,
            finished=True,
            stats={"prompt_tokens": 8, "completion_tokens": 2},
        )

    async def fake_submit(request: object) -> SimpleNamespace:
        captured_messages.append(list(getattr(request, "messages", []) or []))
        return SimpleNamespace(
            request_id=getattr(request, "trace_id", "resp"),
            text="follow up",
            prompt_tokens=12,
            completion_tokens=2,
            cache_hit=False,
            speculative_enabled=False,
        )

    client.app.state.container.inference_engine.stream = fake_stream
    client.app.state.container.inference_engine.submit = fake_submit

    with client.stream(
        "POST",
        "/v1/responses",
        headers={"X-Request-Id": "resp-stream"},
        json={"model": "gpt-5.1", "stream": True, "input": "Stream first"},
    ) as response:
        body = b"".join(response.iter_bytes()).decode()

    assert response.status_code == 200
    assert response.headers["X-Request-Id"] == "resp-stream"
    assert "streamed answer" in body
    events = _sse_events(body)
    response_ids = {payload["response_id"] for _, payload in events}
    assert len(response_ids) == 1
    stream_response_id = next(iter(response_ids))
    assert isinstance(stream_response_id, str)
    assert stream_response_id.startswith("resp_")
    assert stream_response_id != "resp-stream"
    assert events[-1][1]["response"]["id"] == stream_response_id

    follow_up = client.post(
        "/v1/responses",
        json={
            "model": "gpt-5.1",
            "previous_response_id": stream_response_id,
            "input": "After stream",
        },
    )

    assert follow_up.status_code == 200
    assert captured_messages == [
        [
            {"role": "user", "content": "Stream first"},
            {"role": "assistant", "content": "streamed answer"},
            {"role": "user", "content": "After stream"},
        ]
    ]


def test_openai_responses_stream_store_false_does_not_persist_history(
    client: TestClient,
) -> None:
    async def fake_stream(_request: object):
        yield DecodeChunk(token="streamed", index=0, finished=False)
        yield DecodeChunk(
            token="",
            index=1,
            finished=True,
            stats={"prompt_tokens": 8, "completion_tokens": 1},
        )

    client.app.state.container.inference_engine.stream = fake_stream

    with client.stream(
        "POST",
        "/v1/responses",
        json={
            "model": "gpt-5.1",
            "stream": True,
            "input": "Stream first",
            "store": False,
        },
    ) as response:
        body = b"".join(response.iter_bytes()).decode()

    assert response.status_code == 200
    events = _sse_events(body)
    completed = events[-1][1]["response"]

    assert isinstance(completed, dict)
    assert completed["store"] is False

    follow_up = client.post(
        "/v1/responses",
        json={
            "model": "gpt-5.1",
            "previous_response_id": completed["id"],
            "input": "After stream",
        },
    )

    assert follow_up.status_code == 404
    assert follow_up.json()["error"]["code"] == "response_not_found"


def test_openai_responses_stream_incomplete_response_replays_stored_history(
    client: TestClient,
) -> None:
    captured_messages: list[list[dict[str, object]]] = []

    async def fake_stream(_request: object):
        yield DecodeChunk(token="streamed ", index=0, finished=False)
        yield DecodeChunk(token="partial", index=1, finished=False)
        yield DecodeChunk(
            token="",
            index=2,
            finished=True,
            stats={"prompt_tokens": 8, "completion_tokens": 1, "finish_reason": "length"},
        )

    async def fake_submit(request: object) -> SimpleNamespace:
        captured_messages.append(list(getattr(request, "messages", []) or []))
        return SimpleNamespace(
            request_id=getattr(request, "trace_id", "resp"),
            text="follow up",
            prompt_tokens=12,
            completion_tokens=2,
            cache_hit=False,
            speculative_enabled=False,
            finish_reason="stop",
        )

    client.app.state.container.inference_engine.stream = fake_stream
    client.app.state.container.inference_engine.submit = fake_submit

    with client.stream(
        "POST",
        "/v1/responses",
        json={
            "model": "gpt-5.1",
            "stream": True,
            "input": "Stream first",
            "max_output_tokens": 1,
        },
    ) as response:
        body = b"".join(response.iter_bytes()).decode()

    assert response.status_code == 200
    events = _sse_events(body)
    completed = events[-1][1]["response"]
    assert completed["status"] == "incomplete"
    assert completed["incomplete_details"] == {"reason": "max_output_tokens"}
    stream_response_id = completed["id"]

    follow_up = client.post(
        "/v1/responses",
        json={
            "model": "gpt-5.1",
            "previous_response_id": stream_response_id,
            "input": "After stream",
        },
    )

    assert follow_up.status_code == 200
    assert captured_messages == [
        [
            {"role": "user", "content": "Stream first"},
            {"role": "assistant", "content": "streamed partial"},
            {"role": "user", "content": "After stream"},
        ]
    ]


def test_openai_responses_stream_endpoint_emits_reasoning_events(client: TestClient) -> None:
    async def fake_stream(_request: object):
        yield DecodeChunk(token="<think>scratch</think>", index=0, finished=False)
        yield DecodeChunk(token="final", index=1, finished=False)
        yield DecodeChunk(
            token="",
            index=2,
            finished=True,
            stats={"prompt_tokens": 10, "completion_tokens": 2},
        )

    client.app.state.container.inference_engine.stream = fake_stream

    with client.stream(
        "POST",
        "/v1/responses",
        json={"model": "gpt-5.1", "stream": True, "input": "Hello"},
    ) as response:
        body = b"".join(response.iter_bytes()).decode()

    assert response.status_code == 200
    events = _sse_events(body)
    event_names = [name for name, _ in events]
    assert "response.reasoning_text.delta" in event_names
    assert "response.reasoning_text.done" in event_names
    assert event_names.count("response.output_item.done") == 2
    assert "scratch" in body
    assert "final" in body


def test_openai_responses_structured_stream_strips_json_fences_and_uses_lifecycle(
    client: TestClient,
) -> None:
    async def fake_stream(_request: object):
        yield DecodeChunk(token="```json\n", index=0, finished=False)
        yield DecodeChunk(token='{"answer":"Sunny","confidence":0.9}', index=1, finished=False)
        yield DecodeChunk(token="\n```", index=2, finished=False)
        yield DecodeChunk(
            token="",
            index=3,
            finished=True,
            stats={"prompt_tokens": 18, "completion_tokens": 8, "finish_reason": "stop"},
        )

    client.app.state.container.inference_engine.stream = fake_stream

    with client.stream(
        "POST",
        "/v1/responses",
        json={
            "model": "gpt-5.1",
            "stream": True,
            "input": "Return structured weather data",
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "weather_response",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "answer": {"type": "string"},
                            "confidence": {"type": "number"},
                        },
                        "required": ["answer", "confidence"],
                        "additionalProperties": False,
                    },
                }
            },
        },
    ) as response:
        body = b"".join(response.iter_bytes()).decode()

    assert response.status_code == 200
    events = _sse_events(body)
    event_names = [name for name, _ in events]
    completed = events[-1][1]["response"]

    assert "```" not in body
    assert "response.output_text.done" in event_names
    assert isinstance(completed, dict)
    assert completed["status"] == "completed"
    assert completed["output_text"] == '{"answer":"Sunny","confidence":0.9}'


def test_openai_responses_structured_stream_emits_failed_event_on_schema_error(
    client: TestClient,
) -> None:
    async def fake_stream(_request: object):
        yield DecodeChunk(token='{"answer":123}', index=0, finished=False)
        yield DecodeChunk(
            token="",
            index=1,
            finished=True,
            stats={"prompt_tokens": 10, "completion_tokens": 4, "finish_reason": "stop"},
        )

    client.app.state.container.inference_engine.stream = fake_stream

    with client.stream(
        "POST",
        "/v1/responses",
        json={
            "model": "gpt-5.1",
            "stream": True,
            "input": "Return structured weather data",
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "weather_response",
                    "schema": {
                        "type": "object",
                        "properties": {"answer": {"type": "string"}},
                        "required": ["answer"],
                        "additionalProperties": False,
                    },
                }
            },
        },
    ) as response:
        body = b"".join(response.iter_bytes()).decode()

    assert response.status_code == 200
    events = _sse_events(body)
    event_names = [name for name, _ in events]
    failed = events[-1][1]["response"]

    assert "response.completed" not in event_names
    assert event_names[-2] == "response.output_item.done"
    assert event_names[-1] == "response.failed"
    assert isinstance(failed, dict)
    assert failed["status"] == "failed"
    assert failed["output"][0]["status"] == "incomplete"
    assert failed["error"]["code"] == "structured_output_invalid"
    assert failed["usage"] == _responses_usage(10, 4)


def test_openai_responses_structured_stream_emits_failed_event_on_bounds_error(
    client: TestClient,
) -> None:
    captured_requests: list[object] = []

    async def fake_stream(request: object):
        captured_requests.append(request)
        yield DecodeChunk(token='{"answer":"Sunny","confidence":1.5}', index=0, finished=False)
        yield DecodeChunk(
            token="",
            index=1,
            finished=True,
            stats={"prompt_tokens": 10, "completion_tokens": 4, "finish_reason": "stop"},
        )

    client.app.state.container.inference_engine.stream = fake_stream

    with client.stream(
        "POST",
        "/v1/responses",
        json={
            "model": "gpt-5.1",
            "stream": True,
            "input": "Return structured weather data",
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "weather_response",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "answer": {"type": "string"},
                            "confidence": {"type": "number", "maximum": 1},
                        },
                        "required": ["answer", "confidence"],
                        "additionalProperties": False,
                    },
                }
            },
        },
    ) as response:
        body = b"".join(response.iter_bytes()).decode()

    assert response.status_code == 200
    events = _sse_events(body)
    event_names = [name for name, _ in events]
    failed = events[-1][1]["response"]

    assert "response.completed" not in event_names
    assert event_names[-1] == "response.failed"
    assert isinstance(failed, dict)
    assert failed["status"] == "failed"
    assert failed["error"] == {
        "code": "structured_output_invalid",
        "message": "$.confidence must be less than or equal to 1.",
        "type": "structured_output_invalid",
    }
    assert failed["usage"] == _responses_usage(10, 4)
    structured_schema = getattr(captured_requests[0], "structured_output_schema", None)
    assert structured_schema["properties"]["confidence"] == {
        "type": "number",
        "maximum": 1,
    }


def test_openai_responses_stream_emits_failed_event_on_aster_error(
    client: TestClient,
) -> None:
    async def fake_stream(_request: object):
        yield DecodeChunk(token="partial", index=0, finished=False)
        raise AsterError(
            code="request_timeout",
            message="Inference request timed out",
            status_code=504,
            details={"timeout_seconds": 0.01},
        )

    client.app.state.container.inference_engine.stream = fake_stream

    with client.stream(
        "POST",
        "/v1/responses",
        json={"model": "gpt-5.1", "stream": True, "input": "Hello"},
    ) as response:
        body = b"".join(response.iter_bytes()).decode()

    assert response.status_code == 200
    events = _sse_events(body)
    event_names = [name for name, _ in events]
    failed = events[-1][1]["response"]

    assert "response.completed" not in event_names
    assert event_names[-1] == "response.failed"
    assert isinstance(failed, dict)
    assert failed["status"] == "failed"
    assert failed["output"][0]["status"] == "incomplete"
    assert failed["output_text"] == "partial"
    assert failed["error"] == {
        "code": "request_timeout",
        "message": "Inference request timed out",
        "type": "request_timeout",
    }


def test_openai_responses_failed_stream_does_not_store_partial_replay(
    client: TestClient,
) -> None:
    async def fake_stream(_request: object):
        yield DecodeChunk(token="partial", index=0, finished=False)
        raise AsterError(
            code="request_timeout",
            message="Inference request timed out",
            status_code=504,
            details={"timeout_seconds": 0.01},
        )

    client.app.state.container.inference_engine.stream = fake_stream

    with client.stream(
        "POST",
        "/v1/responses",
        json={"model": "gpt-5.1", "stream": True, "input": "Hello"},
    ) as response:
        body = b"".join(response.iter_bytes()).decode()

    events = _sse_events(body)
    created = next(payload["response"] for name, payload in events if name == "response.created")

    assert events[-1][0] == "response.failed"
    assert isinstance(created, dict)

    follow_up = client.post(
        "/v1/responses",
        json={
            "model": "gpt-5.1",
            "previous_response_id": created["id"],
            "input": "Continue.",
        },
    )

    assert follow_up.status_code == 404
    assert follow_up.json()["error"]["code"] == "response_not_found"


def test_openai_responses_stream_timeout_emits_failed_event_without_storing_partial_replay(
    client: TestClient,
) -> None:
    async def fake_stream(_request: object):
        yield DecodeChunk(token="partial", index=0, finished=False)
        await asyncio.Event().wait()
        yield DecodeChunk(token="unreachable", index=1, finished=False)

    client.app.state.container.inference_engine.stream = fake_stream

    with client.stream(
        "POST",
        "/v1/responses",
        json={"model": "gpt-5.1", "stream": True, "timeout": 0.001, "input": "Hello"},
    ) as response:
        body = b"".join(response.iter_bytes()).decode()

    assert response.status_code == 200
    events = _sse_events(body)
    event_names = [name for name, _ in events]
    created = next(payload["response"] for name, payload in events if name == "response.created")
    failed = events[-1][1]["response"]

    assert "response.completed" not in event_names
    assert events[-1][0] == "response.failed"
    assert isinstance(failed, dict)
    assert failed["status"] == "failed"
    assert failed["output_text"] == "partial"
    assert failed["error"]["code"] == "request_timeout"
    assert failed["error"]["type"] == "request_timeout"
    assert failed["error"]["message"] == "Inference request timed out"

    assert isinstance(created, dict)
    follow_up = client.post(
        "/v1/responses",
        json={
            "model": "gpt-5.1",
            "previous_response_id": created["id"],
            "input": "Continue.",
        },
    )

    assert follow_up.status_code == 404
    assert follow_up.json()["error"]["code"] == "response_not_found"


def test_openai_responses_structured_stream_preserves_aster_error_on_runtime_failure(
    client: TestClient,
) -> None:
    async def fake_stream(_request: object):
        yield DecodeChunk(token='{"answer":', index=0, finished=False)
        raise AsterError(
            code="request_timeout",
            message="Inference request timed out",
            status_code=504,
            details={"timeout_seconds": 0.01},
        )

    client.app.state.container.inference_engine.stream = fake_stream

    with client.stream(
        "POST",
        "/v1/responses",
        json={
            "model": "gpt-5.1",
            "stream": True,
            "input": "Return structured weather data",
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "weather_response",
                    "schema": {
                        "type": "object",
                        "properties": {"answer": {"type": "string"}},
                        "required": ["answer"],
                    },
                }
            },
        },
    ) as response:
        body = b"".join(response.iter_bytes()).decode()

    assert response.status_code == 200
    events = _sse_events(body)
    event_names = [name for name, _ in events]
    failed = events[-1][1]["response"]

    assert "response.completed" not in event_names
    assert event_names[-1] == "response.failed"
    assert isinstance(failed, dict)
    assert failed["status"] == "failed"
    assert failed["output"][0]["status"] == "incomplete"
    assert failed["output_text"] == '{"ans'
    assert failed["error"] == {
        "code": "request_timeout",
        "message": "Inference request timed out",
        "type": "request_timeout",
    }


def test_openai_like_provider_stream_reuses_chat_sse_lifecycle(client: TestClient) -> None:
    async def fake_stream(_request: object):
        yield DecodeChunk(token="hello", index=0, finished=False)
        yield DecodeChunk(
            token="",
            index=1,
            finished=True,
            stats={"prompt_tokens": 3, "completion_tokens": 1, "finish_reason": "length"},
        )

    client.app.state.container.inference_engine.stream = fake_stream

    with client.stream(
        "POST",
        "/xai/v1/chat/completions",
        headers={"X-Request-Id": "xai-stream-123"},
        json={
            "model": "grok-local",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
            "stream_options": {"include_usage": True},
        },
    ) as response:
        body = b"".join(response.iter_bytes()).decode()

    assert response.status_code == 200
    chunks = [
        item for item in _sse_json_payloads(body) if item.get("object") == "chat.completion.chunk"
    ]

    assert _sse_nonempty_lines(body)[-1] == "data: [DONE]"
    assert response.headers["X-Request-Id"] == "xai-stream-123"
    assert len({chunk["id"] for chunk in chunks}) == 1
    assert chunks[0]["id"].startswith("chatcmpl-")
    assert chunks[0]["id"] != "xai-stream-123"
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant"}
    assert chunks[1]["choices"][0]["delta"] == {"content": "hello"}
    assert chunks[2]["choices"][0]["finish_reason"] == "length"
    assert chunks[3]["choices"] == []
    assert chunks[3]["usage"]["total_tokens"] == 4


def test_openai_chat_completions_supports_tool_calling_with_local_model_json_protocol(
    client: TestClient,
) -> None:
    async def fake_submit(_request: object) -> SimpleNamespace:
        return SimpleNamespace(
            request_id="chat_tool_123",
            text='{"assistant_text": null, "tool_calls": [{"name": "lookup_weather", "arguments": {"city": "Shanghai"}}]}',
            prompt_tokens=20,
            completion_tokens=12,
            cache_hit=False,
            speculative_enabled=False,
        )

    client.app.state.container.inference_engine.submit = fake_submit

    response = client.post(
        "/v1/chat/completions",
        headers={"X-Request-Id": "chat-tool-trace"},
        json={
            "model": "gpt-4.1",
            "messages": [{"role": "user", "content": "What is the weather?"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "lookup_weather",
                        "description": "Look up weather",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                        },
                    },
                }
            ],
            "tool_choice": "required",
        },
    )

    assert response.status_code == 200
    assert response.headers["X-Request-Id"] == "chat-tool-trace"
    data = response.json()
    assert data["id"].startswith("chatcmpl-")
    assert data["id"] != "chat-tool-trace"
    assert data["choices"][0]["finish_reason"] == "tool_calls"
    assert data["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "lookup_weather"


def test_openai_chat_tool_requests_carry_parser_stop_sequences(client: TestClient) -> None:
    captured: dict[str, object] = {}

    async def fake_submit(request: object) -> SimpleNamespace:
        captured["stop"] = getattr(request, "stop", None)
        captured["parser_stop_sequences"] = getattr(request, "parser_stop_sequences", None)
        return SimpleNamespace(
            request_id="chat_tool_stop_123",
            text='{"assistant_text": "No tool needed.", "tool_calls": []}',
            prompt_tokens=20,
            completion_tokens=6,
            cache_hit=False,
            speculative_enabled=False,
            finish_reason="stop",
        )

    client.app.state.container.inference_engine.submit = fake_submit

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-4.1",
            "messages": [{"role": "user", "content": "Say hi"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "lookup_weather",
                        "description": "Look up weather",
                        "parameters": {"type": "object"},
                    },
                }
            ],
            "tool_choice": "none",
            "stop": ["USER_STOP"],
        },
    )

    assert response.status_code == 200
    assert captured["stop"] == ["USER_STOP"]
    assert captured["parser_stop_sequences"] == AutoToolParser.extra_stop_tokens


def test_openai_responses_supports_function_call_output(client: TestClient) -> None:
    async def fake_submit(_request: object) -> SimpleNamespace:
        return SimpleNamespace(
            request_id="resp_tool_123",
            text='{"assistant_text": null, "tool_calls": [{"name": "lookup_weather", "arguments": {"city": "Shanghai"}}]}',
            prompt_tokens=20,
            completion_tokens=10,
            cache_hit=False,
            speculative_enabled=False,
        )

    client.app.state.container.inference_engine.submit = fake_submit

    response = client.post(
        "/v1/responses",
        json={
            "model": "gpt-5.1",
            "input": [
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": "What is the weather?"}],
                }
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "lookup_weather",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                        },
                    },
                }
            ],
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["output"][0]["type"] == "function_call"
    assert data["output"][0]["name"] == "lookup_weather"


def test_anthropic_messages_supports_tool_use_blocks(client: TestClient) -> None:
    async def fake_submit(_request: object) -> SimpleNamespace:
        return SimpleNamespace(
            request_id="anth_tool_123",
            text='{"assistant_text": null, "tool_calls": [{"name": "lookup_weather", "arguments": {"city": "Shanghai"}}]}',
            prompt_tokens=20,
            completion_tokens=10,
            cache_hit=False,
            speculative_enabled=False,
        )

    client.app.state.container.inference_engine.submit = fake_submit

    response = client.post(
        "/v1/messages",
        json={
            "model": "claude-sonnet-4-5",
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "What is the weather?"}]}
            ],
            "tools": [
                {
                    "name": "lookup_weather",
                    "description": "Look up weather",
                    "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}},
                }
            ],
            "tool_choice": {"type": "any"},
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["stop_reason"] == "tool_use"
    assert data["content"][0]["type"] == "tool_use"


def test_openai_chat_completions_supports_structured_outputs(client: TestClient) -> None:
    captured_requests: list[object] = []

    async def fake_submit(request: object) -> SimpleNamespace:
        captured_requests.append(request)
        return SimpleNamespace(
            request_id="chat_struct_123",
            text='{"answer":"Sunny","confidence":0.9}',
            prompt_tokens=18,
            completion_tokens=8,
            cache_hit=False,
            speculative_enabled=False,
        )

    client.app.state.container.inference_engine.submit = fake_submit

    response = client.post(
        "/v1/chat/completions",
        headers={"X-Request-Id": "chat-structured-trace"},
        json={
            "model": "gpt-4.1",
            "messages": [{"role": "user", "content": "Return structured weather data"}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "weather_response",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "answer": {"type": "string"},
                            "confidence": {"type": "number"},
                        },
                        "required": ["answer", "confidence"],
                        "additionalProperties": False,
                    },
                },
            },
        },
    )

    assert response.status_code == 200
    assert response.headers["X-Request-Id"] == "chat-structured-trace"
    data = response.json()
    assert data["id"].startswith("chatcmpl-")
    assert data["id"] != "chat-structured-trace"
    assert data["choices"][0]["message"]["content"] == '{"answer": "Sunny", "confidence": 0.9}'
    structured_schema = getattr(captured_requests[0], "structured_output_schema", None)
    assert structured_schema["properties"]["answer"] == {"type": "string"}
    assert structured_schema["additionalProperties"] is False


def test_openai_chat_completions_structured_output_rejects_bounds_violation(
    client: TestClient,
) -> None:
    async def fake_submit(_request: object) -> SimpleNamespace:
        return SimpleNamespace(
            request_id="chat_struct_bounds_123",
            text='{"answer":"ok","confidence":0.9}',
            prompt_tokens=18,
            completion_tokens=8,
            cache_hit=False,
            speculative_enabled=False,
        )

    client.app.state.container.inference_engine.submit = fake_submit

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-4.1",
            "messages": [{"role": "user", "content": "Return structured weather data"}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "weather_response",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "answer": {"type": "string", "minLength": 3},
                            "confidence": {"type": "number"},
                        },
                        "required": ["answer", "confidence"],
                        "additionalProperties": False,
                    },
                },
            },
        },
    )

    assert response.status_code == 422
    assert response.json()["error"] == {
        "message": "$.answer must contain at least 3 characters.",
        "type": "structured_output_invalid",
        "code": "structured_output_invalid",
    }


def test_openai_responses_structured_output_rejects_bounds_violation(
    client: TestClient,
) -> None:
    captured_requests: list[object] = []

    async def fake_submit(request: object) -> SimpleNamespace:
        captured_requests.append(request)
        return SimpleNamespace(
            request_id="resp_struct_bounds_123",
            text='{"answer":"Sunny","confidence":1.5}',
            prompt_tokens=18,
            completion_tokens=8,
            cache_hit=False,
            speculative_enabled=False,
        )

    client.app.state.container.inference_engine.submit = fake_submit

    response = client.post(
        "/v1/responses",
        json={
            "model": "gpt-5.1",
            "input": "Return structured weather data",
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "weather_response",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "answer": {"type": "string"},
                            "confidence": {"type": "number", "maximum": 1},
                        },
                        "required": ["answer", "confidence"],
                        "additionalProperties": False,
                    },
                }
            },
        },
    )

    assert response.status_code == 422
    assert response.json()["error"] == {
        "message": "$.confidence must be less than or equal to 1.",
        "type": "structured_output_invalid",
        "code": "structured_output_invalid",
    }
    structured_schema = getattr(captured_requests[0], "structured_output_schema", None)
    assert structured_schema["properties"]["confidence"] == {
        "type": "number",
        "maximum": 1,
    }


def test_openai_chat_completions_executes_registered_tools_until_final_answer(
    client: TestClient,
) -> None:
    async def fake_submit(request: object) -> SimpleNamespace:
        messages = getattr(request, "messages", None) or []
        if any(message.get("role") == "tool" for message in messages):
            return SimpleNamespace(
                request_id="chat_tool_exec_123",
                text='{"assistant_text":"The sum is 5.","tool_calls":[]}',
                prompt_tokens=28,
                completion_tokens=6,
                cache_hit=False,
                speculative_enabled=False,
            )
        return SimpleNamespace(
            request_id="chat_tool_exec_123",
            text='{"assistant_text":null,"tool_calls":[{"name":"add_numbers","arguments":{"a":2,"b":3}}]}',
            prompt_tokens=20,
            completion_tokens=10,
            cache_hit=False,
            speculative_enabled=False,
        )

    client.app.state.container.inference_engine.submit = fake_submit

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-4.1",
            "messages": [{"role": "user", "content": "What is 2 + 3?"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "add_numbers",
                        "description": "Add two numbers",
                        "parameters": {
                            "type": "object",
                            "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                            "required": ["a", "b"],
                        },
                    },
                }
            ],
            "tool_choice": "required",
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["choices"][0]["finish_reason"] == "stop"
    assert data["choices"][0]["message"]["content"] == "The sum is 5."


def test_openai_chat_tool_execution_runs_parallel_calls_with_order(
    client: TestClient,
) -> None:
    calls: list[list[dict[str, str]]] = []
    active_tools = 0
    max_active_tools = 0
    both_tools_started = asyncio.Event()

    async def wait_tool(arguments: dict[str, object], _context: object) -> object:
        nonlocal active_tools, max_active_tools
        active_tools += 1
        max_active_tools = max(max_active_tools, active_tools)
        if max_active_tools == 2:
            both_tools_started.set()
        try:
            await asyncio.wait_for(both_tools_started.wait(), timeout=1)
            return {"id": arguments["id"]}
        finally:
            active_tools -= 1

    client.app.state.container.tool_registry.register("wait_tool", wait_tool)

    async def fake_submit(request: object) -> SimpleNamespace:
        messages = list(getattr(request, "messages", []) or [])
        calls.append(messages)
        if any(message.get("role") == "tool" for message in messages):
            return SimpleNamespace(
                request_id="chat_parallel_tools_123",
                text='{"assistant_text":"Both tools finished.","tool_calls":[]}',
                prompt_tokens=30,
                completion_tokens=5,
                cache_hit=False,
                speculative_enabled=False,
            )
        return SimpleNamespace(
            request_id="chat_parallel_tools_123",
            text=(
                '{"assistant_text":null,"tool_calls":['
                '{"name":"wait_tool","arguments":{"id":1}},'
                '{"name":"wait_tool","arguments":{"id":2}}'
                "]}"
            ),
            prompt_tokens=20,
            completion_tokens=10,
            cache_hit=False,
            speculative_enabled=False,
        )

    client.app.state.container.inference_engine.submit = fake_submit

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-4.1",
            "messages": [{"role": "user", "content": "Run both tools"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "wait_tool",
                        "description": "Waits until both calls have started",
                        "parameters": {
                            "type": "object",
                            "properties": {"id": {"type": "number"}},
                            "required": ["id"],
                        },
                    },
                }
            ],
            "tool_choice": "required",
            "parallel_tool_calls": True,
        },
    )

    assert response.status_code == 200
    assert max_active_tools == 2
    tool_messages = [
        json.loads(message["content"])
        for message in calls[1]
        if message.get("role") == "tool"
    ]
    assert [message["result"] for message in tool_messages] == [{"id": 1}, {"id": 2}]
    metrics = client.app.state.container.metrics.render().decode()
    assert _metric_line(
        metrics,
        "aster_tool_executions_total",
        tool_name="wait_tool",
        status="success",
    ).endswith(" 2.0")
    assert _metric_line(
        metrics,
        "aster_tool_execution_latency_seconds_count",
        tool_name="wait_tool",
        status="success",
    ).endswith(" 2.0")
    data = response.json()
    assert data["choices"][0]["message"]["content"] == "Both tools finished."


def test_openai_chat_tool_execution_error_is_returned_to_model(
    client: TestClient,
) -> None:
    calls: list[list[dict[str, str]]] = []

    def failing_tool(_arguments: dict[str, object], _context: object) -> object:
        raise ValueError("backend offline")

    client.app.state.container.tool_registry.register("fail_tool", failing_tool)

    async def fake_submit(request: object) -> SimpleNamespace:
        messages = list(getattr(request, "messages", []) or [])
        calls.append(messages)
        if any(message.get("role") == "tool" for message in messages):
            return SimpleNamespace(
                request_id="chat_tool_error_123",
                text='{"assistant_text":"The tool failed, so I cannot fetch that now.","tool_calls":[]}',
                prompt_tokens=30,
                completion_tokens=8,
                cache_hit=False,
                speculative_enabled=False,
            )
        return SimpleNamespace(
            request_id="chat_tool_error_123",
            text='{"assistant_text":null,"tool_calls":[{"name":"fail_tool","arguments":{"query":"x"}}]}',
            prompt_tokens=20,
            completion_tokens=10,
            cache_hit=False,
            speculative_enabled=False,
        )

    client.app.state.container.inference_engine.submit = fake_submit

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-4.1",
            "messages": [{"role": "user", "content": "Use the failing tool"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "fail_tool",
                        "description": "Always fails",
                        "parameters": {"type": "object"},
                    },
                }
            ],
            "tool_choice": "required",
        },
    )

    assert response.status_code == 200
    assert len(calls) == 2
    tool_message = next(message for message in calls[1] if message.get("role") == "tool")
    assert json.loads(tool_message["content"])["result"] == (
        "Error: Tool 'fail_tool' execution failed: backend offline"
    )
    metrics = client.app.state.container.metrics.render().decode()
    assert _metric_line(
        metrics,
        "aster_tool_executions_total",
        tool_name="fail_tool",
        status="error",
    ).endswith(" 1.0")
    data = response.json()
    assert data["choices"][0]["finish_reason"] == "stop"
    assert data["choices"][0]["message"]["content"] == (
        "The tool failed, so I cannot fetch that now."
    )


def test_openai_chat_tool_argument_validation_error_is_returned_to_model(
    client: TestClient,
) -> None:
    calls: list[list[dict[str, str]]] = []
    handler_called = False

    def schema_tool(_arguments: dict[str, object], _context: object) -> object:
        nonlocal handler_called
        handler_called = True
        return {"ok": True}

    client.app.state.container.tool_registry.register("schema_tool", schema_tool)

    async def fake_submit(request: object) -> SimpleNamespace:
        messages = list(getattr(request, "messages", []) or [])
        calls.append(messages)
        if any(message.get("role") == "tool" for message in messages):
            return SimpleNamespace(
                request_id="chat_tool_validation_123",
                text='{"assistant_text":"The tool arguments were invalid.","tool_calls":[]}',
                prompt_tokens=30,
                completion_tokens=8,
                cache_hit=False,
                speculative_enabled=False,
            )
        return SimpleNamespace(
            request_id="chat_tool_validation_123",
            text='{"assistant_text":null,"tool_calls":[{"name":"schema_tool","arguments":{"a":2}}]}',
            prompt_tokens=20,
            completion_tokens=10,
            cache_hit=False,
            speculative_enabled=False,
        )

    client.app.state.container.inference_engine.submit = fake_submit

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-4.1",
            "messages": [{"role": "user", "content": "Use the schema tool"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "schema_tool",
                        "description": "Requires a and b",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "a": {"type": "number"},
                                "b": {"type": "number"},
                            },
                            "required": ["a", "b"],
                        },
                    },
                }
            ],
            "tool_choice": "required",
        },
    )

    assert response.status_code == 200
    assert handler_called is False
    assert len(calls) == 2
    tool_message = next(message for message in calls[1] if message.get("role") == "tool")
    result = json.loads(tool_message["content"])["result"]
    assert result == (
        "Error: Tool 'schema_tool' execution failed: "
        "Tool 'schema_tool' arguments failed validation: $.b is required."
    )
    metrics = client.app.state.container.metrics.render().decode()
    assert _metric_line(
        metrics,
        "aster_tool_executions_total",
        tool_name="schema_tool",
        status="error",
    ).endswith(" 1.0")
    data = response.json()
    assert data["choices"][0]["message"]["content"] == "The tool arguments were invalid."


def test_openai_chat_tool_bounds_validation_error_skips_handler(
    client: TestClient,
) -> None:
    calls: list[list[dict[str, str]]] = []
    handler_called = False

    def bounded_tool(_arguments: dict[str, object], _context: object) -> object:
        nonlocal handler_called
        handler_called = True
        return {"ok": True}

    client.app.state.container.tool_registry.register("bounded_tool", bounded_tool)

    async def fake_submit(request: object) -> SimpleNamespace:
        messages = list(getattr(request, "messages", []) or [])
        calls.append(messages)
        if any(message.get("role") == "tool" for message in messages):
            return SimpleNamespace(
                request_id="chat_tool_bounds_validation_123",
                text='{"assistant_text":"The bounded tool arguments were invalid.","tool_calls":[]}',
                prompt_tokens=30,
                completion_tokens=8,
                cache_hit=False,
                speculative_enabled=False,
            )
        return SimpleNamespace(
            request_id="chat_tool_bounds_validation_123",
            text='{"assistant_text":null,"tool_calls":[{"name":"bounded_tool","arguments":{"score":11}}]}',
            prompt_tokens=20,
            completion_tokens=10,
            cache_hit=False,
            speculative_enabled=False,
        )

    client.app.state.container.inference_engine.submit = fake_submit

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-4.1",
            "messages": [{"role": "user", "content": "Use the bounded tool"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "bounded_tool",
                        "description": "Requires score within range",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "score": {"type": "number", "maximum": 10},
                            },
                            "required": ["score"],
                        },
                    },
                }
            ],
            "tool_choice": "required",
        },
    )

    assert response.status_code == 200
    assert handler_called is False
    assert len(calls) == 2
    tool_message = next(message for message in calls[1] if message.get("role") == "tool")
    result = json.loads(tool_message["content"])["result"]
    assert result == (
        "Error: Tool 'bounded_tool' execution failed: "
        "Tool 'bounded_tool' arguments failed validation: "
        "$.score must be less than or equal to 10."
    )
    metrics = client.app.state.container.metrics.render().decode()
    assert _metric_line(
        metrics,
        "aster_tool_executions_total",
        tool_name="bounded_tool",
        status="error",
    ).endswith(" 1.0")
    data = response.json()
    assert data["choices"][0]["message"]["content"] == (
        "The bounded tool arguments were invalid."
    )


def test_openai_responses_tool_loop_persists_replayable_history(
    client: TestClient,
) -> None:
    calls: list[list[dict[str, object]]] = []

    async def fake_submit(request: object) -> SimpleNamespace:
        messages = list(getattr(request, "messages", []) or [])
        calls.append(messages)
        if any(message.get("role") == "tool" for message in messages):
            return SimpleNamespace(
                request_id=getattr(request, "trace_id", "resp-tool-loop"),
                text='{"assistant_text":"The sum is 5.","tool_calls":[]}',
                prompt_tokens=28,
                completion_tokens=6,
                cache_hit=False,
                speculative_enabled=False,
            )
        return SimpleNamespace(
            request_id=getattr(request, "trace_id", "resp-tool-loop"),
            text='{"assistant_text":null,"tool_calls":[{"name":"add_numbers","arguments":{"a":2,"b":3}}]}',
            prompt_tokens=20,
            completion_tokens=10,
            cache_hit=False,
            speculative_enabled=False,
        )

    client.app.state.container.inference_engine.submit = fake_submit

    first = client.post(
        "/v1/responses",
        json={
            "model": "gpt-5.1",
            "input": "What is 2 + 3?",
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "add_numbers",
                        "parameters": {
                            "type": "object",
                            "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                            "required": ["a", "b"],
                        },
                    },
                }
            ],
            "tool_choice": "required",
        },
    )

    assert first.status_code == 200
    first_id = first.json()["id"]
    assert first_id.startswith("resp_")

    follow_up = client.post(
        "/v1/responses",
        json={
            "model": "gpt-5.1",
            "previous_response_id": first_id,
            "input": "Continue.",
        },
    )

    assert follow_up.status_code == 200
    replayed_messages = calls[2]
    assert replayed_messages[0] == {"role": "user", "content": "What is 2 + 3?"}
    assert replayed_messages[1]["role"] == "assistant"
    tool_call = replayed_messages[1]["tool_calls"][0]
    assert tool_call["function"]["name"] == "add_numbers"
    assert replayed_messages[2] == {
        "role": "tool",
        "tool_call_id": tool_call["id"],
        "content": '{"sum": 5}',
    }
    assert replayed_messages[3] == {"role": "assistant", "content": "The sum is 5."}
    assert replayed_messages[4] == {"role": "user", "content": "Continue."}


def test_openai_responses_tool_argument_validation_replays_error_result(
    client: TestClient,
) -> None:
    calls: list[list[dict[str, object]]] = []
    handler_called = False

    def schema_tool(_arguments: dict[str, object], _context: object) -> object:
        nonlocal handler_called
        handler_called = True
        return {"ok": True}

    client.app.state.container.tool_registry.register("schema_tool", schema_tool)

    async def fake_submit(request: object) -> SimpleNamespace:
        messages = list(getattr(request, "messages", []) or [])
        calls.append(messages)
        if any(message.get("role") == "tool" for message in messages):
            return SimpleNamespace(
                request_id=getattr(request, "trace_id", "resp-tool-validation"),
                text='{"assistant_text":"The tool arguments were invalid.","tool_calls":[]}',
                prompt_tokens=28,
                completion_tokens=6,
                cache_hit=False,
                speculative_enabled=False,
            )
        return SimpleNamespace(
            request_id=getattr(request, "trace_id", "resp-tool-validation"),
            text='{"assistant_text":null,"tool_calls":[{"name":"schema_tool","arguments":{"a":2}}]}',
            prompt_tokens=20,
            completion_tokens=10,
            cache_hit=False,
            speculative_enabled=False,
        )

    client.app.state.container.inference_engine.submit = fake_submit

    first = client.post(
        "/v1/responses",
        json={
            "model": "gpt-5.1",
            "input": "Use the schema tool.",
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "schema_tool",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "a": {"type": "number"},
                                "b": {"type": "number"},
                            },
                            "required": ["a", "b"],
                        },
                    },
                }
            ],
            "tool_choice": "required",
        },
    )

    assert first.status_code == 200
    assert handler_called is False
    expected_tool_result = (
        "Error: Tool 'schema_tool' execution failed: "
        "Tool 'schema_tool' arguments failed validation: $.b is required."
    )
    internal_tool_message = next(message for message in calls[1] if message.get("role") == "tool")
    assert json.loads(internal_tool_message["content"])["result"] == expected_tool_result

    first_payload = first.json()
    assert first_payload["output_text"] == "The tool arguments were invalid."

    follow_up = client.post(
        "/v1/responses",
        json={
            "model": "gpt-5.1",
            "previous_response_id": first_payload["id"],
            "input": "Continue.",
        },
    )

    assert follow_up.status_code == 200
    replayed_tool_message = next(message for message in calls[2] if message.get("role") == "tool")
    assert replayed_tool_message["content"] == expected_tool_result


def test_openai_chat_completions_accepts_auto_detected_tool_format(client: TestClient) -> None:
    async def fake_submit(request: object) -> SimpleNamespace:
        messages = getattr(request, "messages", None) or []
        if any(message.get("role") == "tool" for message in messages):
            return SimpleNamespace(
                request_id="chat_auto_tool_123",
                text='{"assistant_text":"The sum is 5.","tool_calls":[]}',
                prompt_tokens=28,
                completion_tokens=6,
                cache_hit=False,
                speculative_enabled=False,
            )
        return SimpleNamespace(
            request_id="chat_auto_tool_123",
            text='<tool_call>{"name":"add_numbers","arguments":{"a":2,"b":3}}</tool_call>',
            prompt_tokens=20,
            completion_tokens=10,
            cache_hit=False,
            speculative_enabled=False,
        )

    client.app.state.container.inference_engine.submit = fake_submit

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-4.1",
            "messages": [{"role": "user", "content": "What is 2 + 3?"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "add_numbers",
                        "description": "Add two numbers",
                        "parameters": {"type": "object"},
                    },
                }
            ],
            "tool_choice": "required",
        },
    )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "The sum is 5."


def test_openai_chat_completions_stream_emits_live_tool_call_delta(client: TestClient) -> None:
    async def fake_stream(_request: object):
        yield DecodeChunk(token="Before ", index=0, finished=False)
        yield DecodeChunk(
            token='<tool_call>{"name":"remote_add_numbers","arguments":{"a":',
            index=1,
            finished=False,
        )
        yield DecodeChunk(token='2,"b":3}}</tool_call> After', index=2, finished=False)
        yield DecodeChunk(
            token="",
            index=3,
            finished=True,
            stats={"prompt_tokens": 20, "completion_tokens": 10, "finish_reason": "stop"},
        )

    client.app.state.container.inference_engine.stream = fake_stream

    with client.stream(
        "POST",
        "/v1/chat/completions",
        headers={"X-Request-Id": "chat-tool-stream-trace"},
        json={
            "model": "gpt-4.1",
            "stream": True,
            "messages": [{"role": "user", "content": "What is 2 + 3?"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "remote_add_numbers",
                        "description": "Add two numbers",
                        "parameters": {
                            "type": "object",
                            "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                            "required": ["a", "b"],
                        },
                    },
                }
            ],
            "tool_choice": "required",
        },
    ) as response:
        body = b"".join(response.iter_bytes()).decode()

    assert response.status_code == 200
    assert response.headers["X-Request-Id"] == "chat-tool-stream-trace"
    assert "<tool_call>" not in body
    assert "</tool_call>" not in body
    assert '"tool_calls"' in body
    assert '"remote_add_numbers"' in body
    chunks = [
        item for item in _sse_json_payloads(body) if item.get("object") == "chat.completion.chunk"
    ]
    assert _sse_nonempty_lines(body)[-1] == "data: [DONE]"
    assert len({chunk["id"] for chunk in chunks}) == 1
    assert chunks[0]["id"].startswith("chatcmpl-")
    assert chunks[0]["id"] != "chat-tool-stream-trace"
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant"}
    assert chunks[1]["choices"][0]["delta"] == {"content": "Before "}
    tool_call_deltas = [
        chunk["choices"][0]["delta"]["tool_calls"][0]
        for chunk in chunks
        if chunk["choices"][0]["delta"].get("tool_calls")
    ]
    assert tool_call_deltas[0]["type"] == "function"
    assert tool_call_deltas[0]["function"] == {"name": "remote_add_numbers", "arguments": ""}
    assert json.loads("".join(delta["function"]["arguments"] for delta in tool_call_deltas)) == {
        "a": 2,
        "b": 3,
    }
    assert {"content": " After"} in [chunk["choices"][0]["delta"] for chunk in chunks]
    assert chunks[-1]["choices"][0]["finish_reason"] == "tool_calls"


def test_openai_chat_completions_stream_emits_bracket_tool_argument_deltas(
    client: TestClient,
) -> None:
    async def fake_stream(_request: object):
        yield DecodeChunk(token="Before ", index=0, finished=False)
        yield DecodeChunk(token='[Calling tool: remote_add_numbers({"a":', index=1, finished=False)
        yield DecodeChunk(token='2,"b":3})] After', index=2, finished=False)
        yield DecodeChunk(
            token="",
            index=3,
            finished=True,
            stats={"prompt_tokens": 20, "completion_tokens": 10, "finish_reason": "stop"},
        )

    client.app.state.container.inference_engine.stream = fake_stream

    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "gpt-4.1",
            "stream": True,
            "messages": [{"role": "user", "content": "What is 2 + 3?"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "remote_add_numbers",
                        "parameters": {
                            "type": "object",
                            "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                        },
                    },
                }
            ],
            "tool_choice": "required",
        },
    ) as response:
        body = b"".join(response.iter_bytes()).decode()

    assert response.status_code == 200
    assert "[Calling tool:" not in body
    chunks = [
        item for item in _sse_json_payloads(body) if item.get("object") == "chat.completion.chunk"
    ]
    tool_call_deltas = [
        chunk["choices"][0]["delta"]["tool_calls"][0]
        for chunk in chunks
        if chunk["choices"][0]["delta"].get("tool_calls")
    ]
    assert tool_call_deltas[0]["function"] == {"name": "remote_add_numbers", "arguments": ""}
    assert json.loads("".join(delta["function"]["arguments"] for delta in tool_call_deltas)) == {
        "a": 2,
        "b": 3,
    }
    assert {"content": " After"} in [chunk["choices"][0]["delta"] for chunk in chunks]


def test_openai_chat_completions_stream_emits_mistral_tool_argument_deltas(
    client: TestClient,
) -> None:
    async def fake_stream(_request: object):
        yield DecodeChunk(token="Before ", index=0, finished=False)
        yield DecodeChunk(token='[TOOL_CALLS] remote_add_numbers{"a":', index=1, finished=False)
        yield DecodeChunk(token='2,"b":3} After', index=2, finished=False)
        yield DecodeChunk(
            token="",
            index=3,
            finished=True,
            stats={"prompt_tokens": 20, "completion_tokens": 10, "finish_reason": "stop"},
        )

    client.app.state.container.inference_engine.stream = fake_stream

    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "gpt-4.1",
            "stream": True,
            "messages": [{"role": "user", "content": "What is 2 + 3?"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "remote_add_numbers",
                        "parameters": {
                            "type": "object",
                            "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                        },
                    },
                }
            ],
            "tool_choice": "required",
        },
    ) as response:
        body = b"".join(response.iter_bytes()).decode()

    assert response.status_code == 200
    assert "[TOOL_CALLS]" not in body
    chunks = [
        item for item in _sse_json_payloads(body) if item.get("object") == "chat.completion.chunk"
    ]
    tool_call_deltas = [
        chunk["choices"][0]["delta"]["tool_calls"][0]
        for chunk in chunks
        if chunk["choices"][0]["delta"].get("tool_calls")
    ]
    assert tool_call_deltas[0]["function"] == {"name": "remote_add_numbers", "arguments": ""}
    assert json.loads("".join(delta["function"]["arguments"] for delta in tool_call_deltas)) == {
        "a": 2,
        "b": 3,
    }
    assert {"content": " After"} in [chunk["choices"][0]["delta"] for chunk in chunks]


def test_openai_chat_completions_stream_emits_mistral_array_tool_calls(client: TestClient) -> None:
    async def fake_stream(_request: object):
        yield DecodeChunk(
            token='[TOOL_CALLS] [{"name":"lookup_weather","arguments":{"city":"Shanghai"}},',
            index=0,
            finished=False,
        )
        yield DecodeChunk(
            token='{"name":"remote_add_numbers","arguments":{"a":2,"b":3}}] done',
            index=1,
            finished=False,
        )
        yield DecodeChunk(
            token="",
            index=2,
            finished=True,
            stats={"prompt_tokens": 20, "completion_tokens": 10, "finish_reason": "stop"},
        )

    client.app.state.container.inference_engine.stream = fake_stream

    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "gpt-4.1",
            "stream": True,
            "messages": [{"role": "user", "content": "Use tools"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "lookup_weather",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "remote_add_numbers",
                        "parameters": {
                            "type": "object",
                            "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                        },
                    },
                },
            ],
            "tool_choice": "required",
        },
    ) as response:
        body = b"".join(response.iter_bytes()).decode()

    assert response.status_code == 200
    assert "[TOOL_CALLS]" not in body
    chunks = [
        item for item in _sse_json_payloads(body) if item.get("object") == "chat.completion.chunk"
    ]
    tool_call_deltas = [
        chunk["choices"][0]["delta"]["tool_calls"][0]
        for chunk in chunks
        if chunk["choices"][0]["delta"].get("tool_calls")
    ]
    assert [delta["index"] for delta in tool_call_deltas] == [0, 1]
    assert [delta["function"]["name"] for delta in tool_call_deltas] == [
        "lookup_weather",
        "remote_add_numbers",
    ]
    assert [json.loads(delta["function"]["arguments"]) for delta in tool_call_deltas] == [
        {"city": "Shanghai"},
        {"a": 2, "b": 3},
    ]
    assert {"content": " done"} in [chunk["choices"][0]["delta"] for chunk in chunks]


def test_openai_chat_completions_stream_emits_deepseek_tool_argument_deltas(
    client: TestClient,
) -> None:
    async def fake_stream(_request: object):
        yield DecodeChunk(token="Before ", index=0, finished=False)
        yield DecodeChunk(
            token="<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>lookup_weather\n```json\n",
            index=1,
            finished=False,
        )
        yield DecodeChunk(token='{"city":', index=2, finished=False)
        yield DecodeChunk(token='"Shanghai"}', index=3, finished=False)
        yield DecodeChunk(
            token="\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜> After",
            index=4,
            finished=False,
        )
        yield DecodeChunk(
            token="",
            index=5,
            finished=True,
            stats={"prompt_tokens": 20, "completion_tokens": 10, "finish_reason": "stop"},
        )

    client.app.state.container.inference_engine.stream = fake_stream

    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "gpt-4.1",
            "stream": True,
            "messages": [{"role": "user", "content": "Use a weather tool"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "lookup_weather",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                        },
                    },
                }
            ],
            "tool_choice": "required",
        },
    ) as response:
        body = b"".join(response.iter_bytes()).decode()

    assert response.status_code == 200
    assert "<｜tool▁call" not in body
    chunks = [
        item for item in _sse_json_payloads(body) if item.get("object") == "chat.completion.chunk"
    ]
    tool_call_deltas = [
        chunk["choices"][0]["delta"]["tool_calls"][0]
        for chunk in chunks
        if chunk["choices"][0]["delta"].get("tool_calls")
    ]
    assert tool_call_deltas[0]["function"] == {"name": "lookup_weather", "arguments": ""}
    assert json.loads("".join(delta["function"]["arguments"] for delta in tool_call_deltas)) == {
        "city": "Shanghai"
    }
    assert {"content": " After"} in [chunk["choices"][0]["delta"] for chunk in chunks]


def test_openai_chat_completions_stream_emits_minimax_bare_invoke_tool_deltas(
    client: TestClient,
) -> None:
    async def fake_stream(_request: object):
        yield DecodeChunk(token="Before ", index=0, finished=False)
        yield DecodeChunk(token='<invoke name="lookup_weather">', index=1, finished=False)
        yield DecodeChunk(
            token='<parameter name="city">"Shanghai"</parameter>', index=2, finished=False
        )
        yield DecodeChunk(
            token='<parameter name="days">3</parameter></invoke> After', index=3, finished=False
        )
        yield DecodeChunk(
            token="",
            index=4,
            finished=True,
            stats={"prompt_tokens": 20, "completion_tokens": 10, "finish_reason": "stop"},
        )

    client.app.state.container.inference_engine.stream = fake_stream

    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "gpt-4.1",
            "stream": True,
            "messages": [{"role": "user", "content": "Use a weather tool"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "lookup_weather",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}, "days": {"type": "integer"}},
                        },
                    },
                }
            ],
            "tool_choice": "required",
        },
    ) as response:
        body = b"".join(response.iter_bytes()).decode()

    assert response.status_code == 200
    assert "<invoke" not in body
    chunks = [
        item for item in _sse_json_payloads(body) if item.get("object") == "chat.completion.chunk"
    ]
    tool_call_deltas = [
        chunk["choices"][0]["delta"]["tool_calls"][0]
        for chunk in chunks
        if chunk["choices"][0]["delta"].get("tool_calls")
    ]
    assert tool_call_deltas[0]["function"] == {"name": "lookup_weather", "arguments": ""}
    assert json.loads("".join(delta["function"]["arguments"] for delta in tool_call_deltas)) == {
        "city": "Shanghai",
        "days": 3,
    }
    assert {"content": " After"} in [chunk["choices"][0]["delta"] for chunk in chunks]


def test_openai_chat_completions_live_tool_stream_executes_registered_tools_then_final_answer(
    client: TestClient,
) -> None:
    calls: list[list[dict[str, str]]] = []

    async def fake_stream(request: object):
        messages = list(getattr(request, "messages", []) or [])
        calls.append(messages)
        if any(message.get("role") == "tool" for message in messages):
            yield DecodeChunk(token="The sum is 5.", index=0, finished=False)
            yield DecodeChunk(
                token="",
                index=1,
                finished=True,
                stats={"prompt_tokens": 35, "completion_tokens": 6, "finish_reason": "stop"},
            )
            return

        yield DecodeChunk(token="<tool_call><function=add_numbers>", index=0, finished=False)
        yield DecodeChunk(token="<parameter=a>2</parameter>", index=1, finished=False)
        yield DecodeChunk(
            token="<parameter=b>3</parameter></function></tool_call>", index=2, finished=False
        )
        yield DecodeChunk(
            token="",
            index=3,
            finished=True,
            stats={"prompt_tokens": 20, "completion_tokens": 8, "finish_reason": "stop"},
        )

    client.app.state.container.inference_engine.stream = fake_stream

    with client.stream(
        "POST",
        "/v1/chat/completions",
        headers={"X-Request-Id": "chat-live-tool-stream-trace"},
        json={
            "model": "gpt-4.1",
            "stream": True,
            "stream_options": {"include_usage": True},
            "messages": [{"role": "user", "content": "What is 2 + 3?"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "add_numbers",
                        "description": "Add two numbers",
                        "parameters": {
                            "type": "object",
                            "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                            "required": ["a", "b"],
                        },
                    },
                }
            ],
            "tool_choice": "required",
        },
    ) as response:
        body = b"".join(response.iter_bytes()).decode()

    assert response.status_code == 200
    assert response.headers["X-Request-Id"] == "chat-live-tool-stream-trace"
    assert len(calls) == 2
    assert any(
        message.get("role") == "tool" and '"sum": 5' in message.get("content", "")
        for message in calls[1]
    )
    assert "<tool_call>" not in body
    assert "<function=" not in body
    chunks = [
        item for item in _sse_json_payloads(body) if item.get("object") == "chat.completion.chunk"
    ]
    assert len({chunk["id"] for chunk in chunks}) == 1
    assert chunks[0]["id"].startswith("chatcmpl-")
    assert chunks[0]["id"] != "chat-live-tool-stream-trace"
    deltas = [chunk["choices"][0]["delta"] for chunk in chunks if chunk["choices"]]
    assert deltas[0] == {"role": "assistant"}
    assert sum(1 for delta in deltas if delta.get("role") == "assistant") == 1
    tool_call_deltas = [delta for delta in deltas if delta.get("tool_calls")]
    assert len(tool_call_deltas) >= 4
    assert {"content": "The sum is 5."} in deltas
    finish_reasons = [chunk["choices"][0]["finish_reason"] for chunk in chunks if chunk["choices"]]
    assert "tool_calls" in finish_reasons
    assert finish_reasons[-1] == "stop"
    usage_chunks = [chunk for chunk in chunks if not chunk["choices"] and chunk.get("usage")]
    assert usage_chunks[-1]["usage"] == {
        "prompt_tokens": 35,
        "completion_tokens": 6,
        "total_tokens": 41,
    }
    assert _sse_nonempty_lines(body)[-1] == "data: [DONE]"


def test_openai_chat_live_tool_stream_reports_length_for_partial_tool_call(
    client: TestClient,
) -> None:
    calls: list[list[dict[str, str]]] = []

    async def fake_stream(request: object):
        calls.append(list(getattr(request, "messages", []) or []))
        yield DecodeChunk(
            token='<tool_call>{"name":"lookup_weather","arguments":{"city":',
            index=0,
            finished=False,
        )
        yield DecodeChunk(
            token="",
            index=1,
            finished=True,
            stats={"prompt_tokens": 12, "completion_tokens": 1, "finish_reason": "length"},
        )

    client.app.state.container.inference_engine.stream = fake_stream

    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "gpt-4.1",
            "stream": True,
            "messages": [{"role": "user", "content": "Use the weather tool"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "lookup_weather",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                        },
                    },
                }
            ],
            "tool_choice": "required",
        },
    ) as response:
        body = b"".join(response.iter_bytes()).decode()

    assert response.status_code == 200
    assert len(calls) == 1
    chunks = [
        item for item in _sse_json_payloads(body) if item.get("object") == "chat.completion.chunk"
    ]
    deltas = [chunk["choices"][0]["delta"] for chunk in chunks if chunk["choices"]]
    finish_reasons = [
        chunk["choices"][0]["finish_reason"]
        for chunk in chunks
        if chunk["choices"] and chunk["choices"][0]["finish_reason"] is not None
    ]

    assert "<tool_call>" not in body
    assert any(delta.get("tool_calls") for delta in deltas)
    assert finish_reasons == ["length"]
    assert _sse_nonempty_lines(body)[-1] == "data: [DONE]"


def test_openai_chat_live_tool_stream_executes_bare_function_body_tool_call(
    client: TestClient,
) -> None:
    calls: list[list[dict[str, str]]] = []

    async def fake_stream(request: object):
        messages = list(getattr(request, "messages", []) or [])
        calls.append(messages)
        if any(message.get("role") == "tool" for message in messages):
            yield DecodeChunk(token="Calculation captured.", index=0, finished=False)
            yield DecodeChunk(
                token="",
                index=1,
                finished=True,
                stats={"prompt_tokens": 34, "completion_tokens": 4, "finish_reason": "stop"},
            )
            return

        yield DecodeChunk(token='<function=add_numbers>{"a":', index=0, finished=False)
        yield DecodeChunk(token='2,"b":3}</function>', index=1, finished=False)
        yield DecodeChunk(
            token="",
            index=2,
            finished=True,
            stats={"prompt_tokens": 18, "completion_tokens": 5, "finish_reason": "stop"},
        )

    client.app.state.container.inference_engine.stream = fake_stream

    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "gpt-4.1",
            "stream": True,
            "messages": [{"role": "user", "content": "Use the calculator tool"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "add_numbers",
                        "parameters": {
                            "type": "object",
                            "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                            "required": ["a", "b"],
                        },
                    },
                }
            ],
            "tool_choice": "required",
        },
    ) as response:
        body = b"".join(response.iter_bytes()).decode()

    assert response.status_code == 200
    assert len(calls) == 2
    assert "<function=" not in body
    assert "Calculation captured." in body
    tool_message = next(message for message in calls[1] if message.get("role") == "tool")
    tool_payload = json.loads(tool_message["content"])
    assert tool_payload["tool_name"] == "add_numbers"
    assert tool_payload["result"] == {"sum": 5}


def test_openai_chat_live_tool_stream_executes_raw_json_tool_protocol(
    client: TestClient,
) -> None:
    calls: list[list[dict[str, str]]] = []

    async def fake_stream(request: object):
        messages = list(getattr(request, "messages", []) or [])
        calls.append(messages)
        if any(message.get("role") == "tool" for message in messages):
            yield DecodeChunk(token="Calculation captured.", index=0, finished=False)
            yield DecodeChunk(
                token="",
                index=1,
                finished=True,
                stats={"prompt_tokens": 34, "completion_tokens": 4, "finish_reason": "stop"},
            )
            return

        yield DecodeChunk(
            token='{"assistant_text":null,"tool_calls":[{"name":"add_numbers","arguments":{"a":',
            index=0,
            finished=False,
        )
        yield DecodeChunk(token='2,"b":3}}]}', index=1, finished=False)
        yield DecodeChunk(
            token="",
            index=2,
            finished=True,
            stats={"prompt_tokens": 18, "completion_tokens": 5, "finish_reason": "stop"},
        )

    client.app.state.container.inference_engine.stream = fake_stream

    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "gpt-4.1",
            "stream": True,
            "messages": [{"role": "user", "content": "Use the calculator tool"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "add_numbers",
                        "parameters": {
                            "type": "object",
                            "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                            "required": ["a", "b"],
                        },
                    },
                }
            ],
            "tool_choice": "required",
        },
    ) as response:
        body = b"".join(response.iter_bytes()).decode()

    assert response.status_code == 200
    assert len(calls) == 2
    assert '"assistant_text"' not in body
    assert "Calculation captured." in body
    tool_message = next(message for message in calls[1] if message.get("role") == "tool")
    tool_payload = json.loads(tool_message["content"])
    assert tool_payload["tool_name"] == "add_numbers"
    assert tool_payload["result"] == {"sum": 5}


def test_openai_chat_live_tool_stream_suppresses_partial_raw_json_tool_protocol(
    client: TestClient,
) -> None:
    calls: list[list[dict[str, str]]] = []

    async def fake_stream(request: object):
        calls.append(list(getattr(request, "messages", []) or []))
        yield DecodeChunk(
            token='{"assistant_text":null,"tool_calls":[{"name":"add_numbers","arguments":{"a":',
            index=0,
            finished=False,
        )
        yield DecodeChunk(
            token="",
            index=1,
            finished=True,
            stats={"prompt_tokens": 18, "completion_tokens": 1, "finish_reason": "length"},
        )

    client.app.state.container.inference_engine.stream = fake_stream

    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "gpt-4.1",
            "stream": True,
            "messages": [{"role": "user", "content": "Use the calculator tool"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "add_numbers",
                        "parameters": {
                            "type": "object",
                            "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                            "required": ["a", "b"],
                        },
                    },
                }
            ],
            "tool_choice": "required",
        },
    ) as response:
        body = b"".join(response.iter_bytes()).decode()

    assert response.status_code == 200
    assert len(calls) == 1
    assert '"assistant_text"' not in body
    assert '"tool_calls"' not in body
    chunks = [
        item for item in _sse_json_payloads(body) if item.get("object") == "chat.completion.chunk"
    ]
    deltas = [chunk["choices"][0]["delta"] for chunk in chunks if chunk["choices"]]
    finish_reasons = [
        chunk["choices"][0]["finish_reason"]
        for chunk in chunks
        if chunk["choices"] and chunk["choices"][0]["finish_reason"] is not None
    ]

    assert all("content" not in delta for delta in deltas)
    assert finish_reasons == ["length"]
    assert _sse_nonempty_lines(body)[-1] == "data: [DONE]"


def test_openai_chat_live_tool_stream_returns_tool_error_to_followup_round(
    client: TestClient,
) -> None:
    calls: list[list[dict[str, str]]] = []

    def failing_tool(_arguments: dict[str, object], _context: object) -> object:
        raise RuntimeError("timeout from provider")

    client.app.state.container.tool_registry.register("fail_tool", failing_tool)

    async def fake_stream(request: object):
        messages = list(getattr(request, "messages", []) or [])
        calls.append(messages)
        if any(message.get("role") == "tool" for message in messages):
            yield DecodeChunk(token="The tool timed out.", index=0, finished=False)
            yield DecodeChunk(
                token="",
                index=1,
                finished=True,
                stats={"prompt_tokens": 34, "completion_tokens": 5, "finish_reason": "stop"},
            )
            return

        yield DecodeChunk(token="<tool_call><function=fail_tool>", index=0, finished=False)
        yield DecodeChunk(
            token='<parameter=query>"x"</parameter></function></tool_call>',
            index=1,
            finished=False,
        )
        yield DecodeChunk(
            token="",
            index=2,
            finished=True,
            stats={"prompt_tokens": 20, "completion_tokens": 6, "finish_reason": "stop"},
        )

    client.app.state.container.inference_engine.stream = fake_stream

    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "gpt-4.1",
            "stream": True,
            "messages": [{"role": "user", "content": "Use the failing tool"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "fail_tool",
                        "description": "Always fails",
                        "parameters": {"type": "object"},
                    },
                }
            ],
            "tool_choice": "required",
        },
    ) as response:
        body = b"".join(response.iter_bytes()).decode()

    assert response.status_code == 200
    assert len(calls) == 2
    tool_message = next(message for message in calls[1] if message.get("role") == "tool")
    assert json.loads(tool_message["content"])["result"] == (
        "Error: Tool 'fail_tool' execution failed: timeout from provider"
    )
    payloads = _sse_json_payloads(body)
    assert not any("error" in payload for payload in payloads)
    assert "The tool timed out." in body
    assert _sse_nonempty_lines(body)[-1] == "data: [DONE]"


def test_openai_chat_live_tool_stream_returns_tool_timeout_to_followup_round(
    client: TestClient,
) -> None:
    calls: list[list[dict[str, str]]] = []

    async def slow_tool(_arguments: dict[str, object], _context: object) -> object:
        await asyncio.Event().wait()

    client.app.state.container.tool_registry.register("slow_tool", slow_tool)

    async def fake_stream(request: object):
        messages = list(getattr(request, "messages", []) or [])
        calls.append(messages)
        if any(message.get("role") == "tool" for message in messages):
            yield DecodeChunk(token="Recovered after tool timeout.", index=0, finished=False)
            yield DecodeChunk(
                token="",
                index=1,
                finished=True,
                stats={"prompt_tokens": 34, "completion_tokens": 5, "finish_reason": "stop"},
            )
            return

        yield DecodeChunk(token="<tool_call><function=slow_tool>", index=0, finished=False)
        yield DecodeChunk(
            token='<parameter=query>"x"</parameter></function></tool_call>',
            index=1,
            finished=False,
        )
        yield DecodeChunk(
            token="",
            index=2,
            finished=True,
            stats={"prompt_tokens": 20, "completion_tokens": 6, "finish_reason": "stop"},
        )

    client.app.state.container.inference_engine.stream = fake_stream

    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "gpt-4.1",
            "stream": True,
            "timeout": 0.05,
            "messages": [{"role": "user", "content": "Use the slow tool"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "slow_tool",
                        "description": "Never returns",
                        "parameters": {"type": "object"},
                    },
                }
            ],
            "tool_choice": "required",
        },
    ) as response:
        body = b"".join(response.iter_bytes()).decode()

    assert response.status_code == 200
    assert len(calls) == 2
    tool_message = next(message for message in calls[1] if message.get("role") == "tool")
    assert json.loads(tool_message["content"])["result"] == (
        "Error: Tool 'slow_tool' execution failed: Tool execution timed out"
    )
    metrics = client.app.state.container.metrics.render().decode()
    assert _metric_line(
        metrics,
        "aster_tool_executions_total",
        tool_name="slow_tool",
        status="timeout",
    ).endswith(" 1.0")
    assert "Recovered after tool timeout." in body
    assert _sse_nonempty_lines(body)[-1] == "data: [DONE]"


def test_openai_chat_live_tool_stream_emits_error_payload_on_aster_error(
    client: TestClient,
) -> None:
    async def fake_stream(_request: object):
        yield DecodeChunk(token="partial", index=0, finished=False)
        raise AsterError(
            code="request_timeout",
            message="Inference request timed out",
            status_code=504,
            details={"timeout_seconds": 0.01},
        )

    client.app.state.container.inference_engine.stream = fake_stream

    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "gpt-4.1",
            "stream": True,
            "messages": [{"role": "user", "content": "What is 2 + 3?"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "add_numbers",
                        "parameters": {
                            "type": "object",
                            "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                            "required": ["a", "b"],
                        },
                    },
                }
            ],
            "tool_choice": "required",
        },
    ) as response:
        body = b"".join(response.iter_bytes()).decode()

    assert response.status_code == 200
    payloads = _sse_json_payloads(body)
    error = next(payload["error"] for payload in payloads if "error" in payload)
    assert error == {
        "message": "Inference request timed out",
        "type": "request_timeout",
        "code": "request_timeout",
        "details": {"timeout_seconds": 0.01},
    }
    assert _sse_nonempty_lines(body)[-1] == "data: [DONE]"


def test_openai_chat_live_tool_stream_uses_tool_schema_for_xml_argument_types(
    client: TestClient,
) -> None:
    calls: list[list[dict[str, str]]] = []

    async def fake_stream(request: object):
        messages = list(getattr(request, "messages", []) or [])
        calls.append(messages)
        if any(message.get("role") == "tool" for message in messages):
            yield DecodeChunk(token="Captured.", index=0, finished=False)
            yield DecodeChunk(
                token="",
                index=1,
                finished=True,
                stats={"prompt_tokens": 30, "completion_tokens": 2, "finish_reason": "stop"},
            )
            return

        yield DecodeChunk(token="<tool_call><function=echo>", index=0, finished=False)
        yield DecodeChunk(
            token='<parameter=text>{"raw":1}</parameter></function></tool_call>',
            index=1,
            finished=False,
        )
        yield DecodeChunk(
            token="", index=2, finished=True, stats={"prompt_tokens": 15, "completion_tokens": 5}
        )

    client.app.state.container.inference_engine.stream = fake_stream

    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "gpt-4.1",
            "stream": True,
            "messages": [{"role": "user", "content": "Echo raw JSON as text"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "echo",
                        "parameters": {
                            "type": "object",
                            "properties": {"text": {"type": "string"}},
                            "required": ["text"],
                        },
                    },
                }
            ],
            "tool_choice": "required",
        },
    ) as response:
        body = b"".join(response.iter_bytes()).decode()

    assert response.status_code == 200
    assert "<parameter=" not in body
    assert len(calls) == 2
    tool_message = next(message for message in calls[1] if message.get("role") == "tool")
    assert json.loads(tool_message["content"])["result"] == {"text": '{"raw":1}'}
    assert "Captured." in body


def test_openai_responses_stream_emits_live_function_call_delta(client: TestClient) -> None:
    async def fake_stream(_request: object):
        yield DecodeChunk(
            token='<tool_call>{"name":"lookup_weather","arguments":{"city":',
            index=0,
            finished=False,
        )
        yield DecodeChunk(token='"Shanghai"}}</tool_call>', index=1, finished=False)
        yield DecodeChunk(
            token="",
            index=2,
            finished=True,
            stats={"prompt_tokens": 12, "completion_tokens": 6, "finish_reason": "stop"},
        )

    client.app.state.container.inference_engine.stream = fake_stream

    with client.stream(
        "POST",
        "/v1/responses",
        json={
            "model": "gpt-5.1",
            "stream": True,
            "input": "What is the weather?",
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "lookup_weather",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                        },
                    },
                }
            ],
            "tool_choice": "required",
        },
    ) as response:
        body = b"".join(response.iter_bytes()).decode()

    assert response.status_code == 200
    assert "<tool_call>" not in body
    events = _sse_events(body)
    event_names = [name for name, _ in events]
    assert "response.function_call_arguments.delta" in event_names
    function_arguments = "".join(
        payload["delta"]
        for name, payload in events
        if name == "response.function_call_arguments.delta"
    )
    completed = events[-1][1]["response"]
    assert json.loads(function_arguments) == {"city": "Shanghai"}
    assert isinstance(completed, dict)
    assert completed["output"][0]["type"] == "function_call"
    assert completed["output_text"] == ""


def test_openai_responses_live_tool_stream_reports_length_as_incomplete(
    client: TestClient,
) -> None:
    async def fake_stream(_request: object):
        yield DecodeChunk(token="partial", index=0, finished=False)
        yield DecodeChunk(
            token="",
            index=1,
            finished=True,
            stats={"prompt_tokens": 12, "completion_tokens": 1, "finish_reason": "length"},
        )

    client.app.state.container.inference_engine.stream = fake_stream

    with client.stream(
        "POST",
        "/v1/responses",
        json={
            "model": "gpt-5.1",
            "stream": True,
            "input": "Keep it short.",
            "max_output_tokens": 1,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "lookup_weather",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                        },
                    },
                }
            ],
            "tool_choice": "required",
        },
    ) as response:
        body = b"".join(response.iter_bytes()).decode()

    assert response.status_code == 200
    events = _sse_events(body)
    completed = events[-1][1]["response"]

    assert isinstance(completed, dict)
    assert completed["status"] == "incomplete"
    assert completed["incomplete_details"] == {"reason": "max_output_tokens"}
    assert completed["output_text"] == "partial"
    assert completed["usage"] == _responses_usage(12, 1)


def test_openai_responses_live_tool_stream_marks_partial_tool_call_incomplete(
    client: TestClient,
) -> None:
    async def fake_stream(_request: object):
        yield DecodeChunk(
            token='<tool_call>{"name":"lookup_weather","arguments":{"city":',
            index=0,
            finished=False,
        )
        yield DecodeChunk(
            token="",
            index=1,
            finished=True,
            stats={"prompt_tokens": 12, "completion_tokens": 1, "finish_reason": "length"},
        )

    client.app.state.container.inference_engine.stream = fake_stream

    with client.stream(
        "POST",
        "/v1/responses",
        json={
            "model": "gpt-5.1",
            "stream": True,
            "input": "What is the weather?",
            "max_output_tokens": 1,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "lookup_weather",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                        },
                    },
                }
            ],
            "tool_choice": "required",
        },
    ) as response:
        body = b"".join(response.iter_bytes()).decode()

    assert response.status_code == 200
    events = _sse_events(body)
    completed = events[-1][1]["response"]

    assert isinstance(completed, dict)
    assert completed["status"] == "incomplete"
    assert completed["incomplete_details"] == {"reason": "max_output_tokens"}
    assert completed["output_text"] == ""
    assert [item["type"] for item in completed["output"]] == ["function_call"]
    function_call = completed["output"][0]
    assert function_call["status"] == "incomplete"
    assert function_call["name"] == "lookup_weather"
    assert function_call["arguments"] == '{"city":'


def test_openai_responses_live_tool_stream_suppresses_partial_raw_json_tool_protocol(
    client: TestClient,
) -> None:
    async def fake_stream(_request: object):
        yield DecodeChunk(
            token='{"assistant_text":null,"tool_calls":[{"name":"add_numbers","arguments":{"a":',
            index=0,
            finished=False,
        )
        yield DecodeChunk(
            token="",
            index=1,
            finished=True,
            stats={"prompt_tokens": 12, "completion_tokens": 1, "finish_reason": "length"},
        )

    client.app.state.container.inference_engine.stream = fake_stream

    with client.stream(
        "POST",
        "/v1/responses",
        json={
            "model": "gpt-5.1",
            "stream": True,
            "input": "What is 2 + 3?",
            "max_output_tokens": 1,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "add_numbers",
                        "parameters": {
                            "type": "object",
                            "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                            "required": ["a", "b"],
                        },
                    },
                }
            ],
            "tool_choice": "required",
        },
    ) as response:
        body = b"".join(response.iter_bytes()).decode()

    assert response.status_code == 200
    assert '"assistant_text"' not in body
    assert '"tool_calls"' not in body
    events = _sse_events(body)
    completed = events[-1][1]["response"]

    assert isinstance(completed, dict)
    assert completed["status"] == "incomplete"
    assert completed["incomplete_details"] == {"reason": "max_output_tokens"}
    assert completed["output"] == []
    assert completed["output_text"] == ""
    assert completed["usage"] == _responses_usage(12, 1)


def test_openai_responses_live_tool_stream_emits_failed_event_on_aster_error(
    client: TestClient,
) -> None:
    async def fake_stream(_request: object):
        yield DecodeChunk(token="partial", index=0, finished=False)
        raise AsterError(
            code="request_timeout",
            message="Inference request timed out",
            status_code=504,
            details={"timeout_seconds": 0.01},
        )

    client.app.state.container.inference_engine.stream = fake_stream

    with client.stream(
        "POST",
        "/v1/responses",
        json={
            "model": "gpt-5.1",
            "stream": True,
            "input": "What is 2 + 3?",
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "add_numbers",
                        "parameters": {
                            "type": "object",
                            "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                            "required": ["a", "b"],
                        },
                    },
                }
            ],
            "tool_choice": "required",
        },
    ) as response:
        body = b"".join(response.iter_bytes()).decode()

    assert response.status_code == 200
    events = _sse_events(body)
    event_names = [name for name, _ in events]
    failed = events[-1][1]["response"]

    assert "response.completed" not in event_names
    assert event_names[-1] == "response.failed"
    assert isinstance(failed, dict)
    assert failed["status"] == "failed"
    assert failed["output"][0]["status"] == "incomplete"
    assert failed["output_text"] == "partial"
    assert failed["error"] == {
        "code": "request_timeout",
        "message": "Inference request timed out",
        "type": "request_timeout",
    }


def test_openai_responses_failed_live_tool_stream_does_not_store_partial_replay(
    client: TestClient,
) -> None:
    async def fake_stream(_request: object):
        yield DecodeChunk(token="partial", index=0, finished=False)
        raise AsterError(
            code="request_timeout",
            message="Inference request timed out",
            status_code=504,
            details={"timeout_seconds": 0.01},
        )

    client.app.state.container.inference_engine.stream = fake_stream

    with client.stream(
        "POST",
        "/v1/responses",
        json={
            "model": "gpt-5.1",
            "stream": True,
            "input": "What is 2 + 3?",
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "add_numbers",
                        "parameters": {
                            "type": "object",
                            "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                            "required": ["a", "b"],
                        },
                    },
                }
            ],
            "tool_choice": "required",
        },
    ) as response:
        body = b"".join(response.iter_bytes()).decode()

    events = _sse_events(body)
    created = next(payload["response"] for name, payload in events if name == "response.created")

    assert events[-1][0] == "response.failed"
    assert isinstance(created, dict)

    follow_up = client.post(
        "/v1/responses",
        json={
            "model": "gpt-5.1",
            "previous_response_id": created["id"],
            "input": "Continue.",
        },
    )

    assert follow_up.status_code == 404
    assert follow_up.json()["error"]["code"] == "response_not_found"


def test_openai_responses_live_tool_stream_timeout_does_not_store_partial_replay(
    client: TestClient,
) -> None:
    async def fake_stream(_request: object):
        yield DecodeChunk(token="partial", index=0, finished=False)
        await asyncio.Event().wait()
        yield DecodeChunk(token="unreachable", index=1, finished=False)

    client.app.state.container.inference_engine.stream = fake_stream

    with client.stream(
        "POST",
        "/v1/responses",
        json={
            "model": "gpt-5.1",
            "stream": True,
            "timeout": 0.001,
            "input": "What is 2 + 3?",
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "add_numbers",
                        "parameters": {
                            "type": "object",
                            "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                            "required": ["a", "b"],
                        },
                    },
                }
            ],
            "tool_choice": "required",
        },
    ) as response:
        body = b"".join(response.iter_bytes()).decode()

    assert response.status_code == 200
    events = _sse_events(body)
    event_names = [name for name, _ in events]
    created = next(payload["response"] for name, payload in events if name == "response.created")
    failed = events[-1][1]["response"]

    assert "response.completed" not in event_names
    assert events[-1][0] == "response.failed"
    assert isinstance(failed, dict)
    assert failed["status"] == "failed"
    assert failed["output_text"] == "partial"
    assert failed["error"]["code"] == "request_timeout"
    assert failed["error"]["type"] == "request_timeout"

    assert isinstance(created, dict)
    follow_up = client.post(
        "/v1/responses",
        json={
            "model": "gpt-5.1",
            "previous_response_id": created["id"],
            "input": "Continue.",
        },
    )

    assert follow_up.status_code == 404
    assert follow_up.json()["error"]["code"] == "response_not_found"


def test_openai_responses_live_tool_stream_returns_tool_timeout_to_followup_round(
    client: TestClient,
) -> None:
    calls: list[list[dict[str, str]]] = []
    follow_up_messages: list[dict[str, object]] = []

    async def slow_tool(_arguments: dict[str, object], _context: object) -> object:
        await asyncio.Event().wait()

    client.app.state.container.tool_registry.register("slow_tool", slow_tool)

    async def fake_stream(request: object):
        messages = list(getattr(request, "messages", []) or [])
        calls.append(messages)
        if any(message.get("role") == "tool" for message in messages):
            yield DecodeChunk(token="Recovered after tool timeout.", index=0, finished=False)
            yield DecodeChunk(
                token="",
                index=1,
                finished=True,
                stats={"prompt_tokens": 36, "completion_tokens": 5, "finish_reason": "stop"},
            )
            return

        yield DecodeChunk(token="<tool_call><function=slow_tool>", index=0, finished=False)
        yield DecodeChunk(
            token='<parameter=query>"x"</parameter></function></tool_call>',
            index=1,
            finished=False,
        )
        yield DecodeChunk(
            token="",
            index=2,
            finished=True,
            stats={"prompt_tokens": 20, "completion_tokens": 6, "finish_reason": "stop"},
        )

    client.app.state.container.inference_engine.stream = fake_stream

    async def fake_submit(request: object) -> SimpleNamespace:
        follow_up_messages.extend(getattr(request, "messages", []) or [])
        return SimpleNamespace(
            request_id=getattr(request, "trace_id", "resp-follow-up"),
            text="continued",
            prompt_tokens=20,
            completion_tokens=1,
            cache_hit=False,
            speculative_enabled=False,
        )

    client.app.state.container.inference_engine.submit = fake_submit

    with client.stream(
        "POST",
        "/v1/responses",
        json={
            "model": "gpt-5.1",
            "stream": True,
            "timeout": 0.05,
            "input": "Use the slow tool.",
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "slow_tool",
                        "parameters": {"type": "object"},
                    },
                }
            ],
            "tool_choice": "required",
        },
    ) as response:
        body = b"".join(response.iter_bytes()).decode()

    assert response.status_code == 200
    assert len(calls) == 2
    tool_message = next(message for message in calls[1] if message.get("role") == "tool")
    expected_tool_result = (
        "Error: Tool 'slow_tool' execution failed: Tool execution timed out"
    )
    assert json.loads(tool_message["content"])["result"] == expected_tool_result
    events = _sse_events(body)
    completed = events[-1][1]["response"]

    assert events[-1][0] == "response.completed"
    assert isinstance(completed, dict)
    assert completed["output_text"] == "Recovered after tool timeout."

    follow_up = client.post(
        "/v1/responses",
        json={
            "model": "gpt-5.1",
            "previous_response_id": completed["id"],
            "input": "Continue.",
        },
    )

    assert follow_up.status_code == 200
    replay_tool_message = next(
        message for message in follow_up_messages if message.get("role") == "tool"
    )
    assert replay_tool_message["content"] == expected_tool_result


def test_openai_responses_live_tool_argument_validation_replays_error_result(
    client: TestClient,
) -> None:
    calls: list[list[dict[str, str]]] = []
    follow_up_messages: list[dict[str, object]] = []
    handler_called = False

    def schema_tool(_arguments: dict[str, object], _context: object) -> object:
        nonlocal handler_called
        handler_called = True
        return {"ok": True}

    client.app.state.container.tool_registry.register("schema_tool", schema_tool)

    async def fake_stream(request: object):
        messages = list(getattr(request, "messages", []) or [])
        calls.append(messages)
        if any(message.get("role") == "tool" for message in messages):
            yield DecodeChunk(token="Recovered after invalid tool arguments.", index=0, finished=False)
            yield DecodeChunk(
                token="",
                index=1,
                finished=True,
                stats={"prompt_tokens": 36, "completion_tokens": 5, "finish_reason": "stop"},
            )
            return

        yield DecodeChunk(token="<tool_call><function=schema_tool>", index=0, finished=False)
        yield DecodeChunk(token="<parameter=a>2</parameter>", index=1, finished=False)
        yield DecodeChunk(
            token="</function></tool_call>",
            index=2,
            finished=False,
        )
        yield DecodeChunk(
            token="",
            index=3,
            finished=True,
            stats={"prompt_tokens": 20, "completion_tokens": 6, "finish_reason": "stop"},
        )

    client.app.state.container.inference_engine.stream = fake_stream

    async def fake_submit(request: object) -> SimpleNamespace:
        follow_up_messages.extend(getattr(request, "messages", []) or [])
        return SimpleNamespace(
            request_id=getattr(request, "trace_id", "resp-follow-up"),
            text="continued",
            prompt_tokens=20,
            completion_tokens=1,
            cache_hit=False,
            speculative_enabled=False,
        )

    client.app.state.container.inference_engine.submit = fake_submit

    with client.stream(
        "POST",
        "/v1/responses",
        json={
            "model": "gpt-5.1",
            "stream": True,
            "input": "Use the schema tool.",
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "schema_tool",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "a": {"type": "number"},
                                "b": {"type": "number"},
                            },
                            "required": ["a", "b"],
                        },
                    },
                }
            ],
            "tool_choice": "required",
        },
    ) as response:
        body = b"".join(response.iter_bytes()).decode()

    assert response.status_code == 200
    assert handler_called is False
    assert len(calls) == 2
    tool_message = next(message for message in calls[1] if message.get("role") == "tool")
    expected_tool_result = (
        "Error: Tool 'schema_tool' execution failed: "
        "Tool 'schema_tool' arguments failed validation: $.b is required."
    )
    assert json.loads(tool_message["content"])["result"] == expected_tool_result
    events = _sse_events(body)
    completed = events[-1][1]["response"]

    assert events[-1][0] == "response.completed"
    assert isinstance(completed, dict)
    assert completed["output_text"] == "Recovered after invalid tool arguments."

    metrics = client.app.state.container.metrics.render().decode()
    assert _metric_line(
        metrics,
        "aster_tool_executions_total",
        tool_name="schema_tool",
        status="error",
    ).endswith(" 1.0")

    follow_up = client.post(
        "/v1/responses",
        json={
            "model": "gpt-5.1",
            "previous_response_id": completed["id"],
            "input": "Continue.",
        },
    )

    assert follow_up.status_code == 200
    replay_tool_message = next(
        message for message in follow_up_messages if message.get("role") == "tool"
    )
    assert replay_tool_message["content"] == expected_tool_result


def test_openai_responses_live_tool_stream_executes_registered_tools_then_final_output(
    client: TestClient,
) -> None:
    calls: list[list[dict[str, str]]] = []
    follow_up_messages: list[dict[str, object]] = []

    async def fake_stream(request: object):
        messages = list(getattr(request, "messages", []) or [])
        calls.append(messages)
        if any(message.get("role") == "tool" for message in messages):
            yield DecodeChunk(token="The sum is 5.", index=0, finished=False)
            yield DecodeChunk(
                token="",
                index=1,
                finished=True,
                stats={"prompt_tokens": 36, "completion_tokens": 5, "finish_reason": "stop"},
            )
            return

        yield DecodeChunk(token="<tool_call><function=add_numbers>", index=0, finished=False)
        yield DecodeChunk(token="<parameter=a>2</parameter>", index=1, finished=False)
        yield DecodeChunk(
            token="<parameter=b>3</parameter></function></tool_call>", index=2, finished=False
        )
        yield DecodeChunk(
            token="",
            index=3,
            finished=True,
            stats={"prompt_tokens": 20, "completion_tokens": 8, "finish_reason": "stop"},
        )

    client.app.state.container.inference_engine.stream = fake_stream

    async def fake_submit(request: object) -> SimpleNamespace:
        follow_up_messages.extend(getattr(request, "messages", []) or [])
        return SimpleNamespace(
            request_id=getattr(request, "trace_id", "resp-follow-up"),
            text="continued",
            prompt_tokens=20,
            completion_tokens=1,
            cache_hit=False,
            speculative_enabled=False,
        )

    client.app.state.container.inference_engine.submit = fake_submit

    with client.stream(
        "POST",
        "/v1/responses",
        json={
            "model": "gpt-5.1",
            "stream": True,
            "input": "What is 2 + 3?",
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "add_numbers",
                        "parameters": {
                            "type": "object",
                            "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                            "required": ["a", "b"],
                        },
                    },
                }
            ],
            "tool_choice": "required",
        },
    ) as response:
        body = b"".join(response.iter_bytes()).decode()

    assert response.status_code == 200
    assert len(calls) == 2
    assert any(
        message.get("role") == "tool" and '"sum": 5' in message.get("content", "")
        for message in calls[1]
    )
    assert "<tool_call>" not in body
    assert "<function=" not in body
    events = _sse_events(body)
    event_names = [name for name, _ in events]
    completed = events[-1][1]["response"]

    assert event_names.count("response.function_call_arguments.delta") >= 3
    function_call_added = [
        payload
        for name, payload in events
        if name == "response.output_item.added" and payload["item"]["type"] == "function_call"
    ]
    assert len(function_call_added) == 1
    assert "response.output_text.delta" in event_names
    assert isinstance(completed, dict)
    assert completed["output_text"] == "The sum is 5."
    assert [item["type"] for item in completed["output"]] == ["function_call", "message"]
    assert completed["usage"] == _responses_usage(36, 5)

    follow_up = client.post(
        "/v1/responses",
        json={
            "model": "gpt-5.1",
            "previous_response_id": completed["id"],
            "input": "Continue.",
        },
    )

    assert follow_up.status_code == 200
    assert follow_up_messages[0] == {"role": "user", "content": "What is 2 + 3?"}
    assert follow_up_messages[1]["role"] == "assistant"
    tool_call = follow_up_messages[1]["tool_calls"][0]
    assert tool_call["function"]["name"] == "add_numbers"
    assert follow_up_messages[2] == {
        "role": "tool",
        "tool_call_id": tool_call["id"],
        "content": '{"sum": 5}',
    }
    assert follow_up_messages[3] == {"role": "assistant", "content": "The sum is 5."}
    assert follow_up_messages[4] == {"role": "user", "content": "Continue."}


def test_openai_responses_live_tool_stream_store_false_does_not_persist_history(
    client: TestClient,
) -> None:
    async def fake_stream(request: object):
        messages = list(getattr(request, "messages", []) or [])
        if any(message.get("role") == "tool" for message in messages):
            yield DecodeChunk(token="The sum is 5.", index=0, finished=False)
            yield DecodeChunk(
                token="",
                index=1,
                finished=True,
                stats={"prompt_tokens": 36, "completion_tokens": 5, "finish_reason": "stop"},
            )
            return

        yield DecodeChunk(token="<tool_call><function=add_numbers>", index=0, finished=False)
        yield DecodeChunk(token="<parameter=a>2</parameter>", index=1, finished=False)
        yield DecodeChunk(
            token="<parameter=b>3</parameter></function></tool_call>", index=2, finished=False
        )
        yield DecodeChunk(
            token="",
            index=3,
            finished=True,
            stats={"prompt_tokens": 20, "completion_tokens": 8, "finish_reason": "stop"},
        )

    client.app.state.container.inference_engine.stream = fake_stream

    with client.stream(
        "POST",
        "/v1/responses",
        json={
            "model": "gpt-5.1",
            "stream": True,
            "input": "What is 2 + 3?",
            "store": False,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "add_numbers",
                        "parameters": {
                            "type": "object",
                            "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                            "required": ["a", "b"],
                        },
                    },
                }
            ],
            "tool_choice": "required",
        },
    ) as response:
        body = b"".join(response.iter_bytes()).decode()

    assert response.status_code == 200
    events = _sse_events(body)
    completed = events[-1][1]["response"]

    assert isinstance(completed, dict)
    assert completed["store"] is False
    assert completed["output_text"] == "The sum is 5."

    follow_up = client.post(
        "/v1/responses",
        json={
            "model": "gpt-5.1",
            "previous_response_id": completed["id"],
            "input": "Continue.",
        },
    )

    assert follow_up.status_code == 404
    assert follow_up.json()["error"]["code"] == "response_not_found"


def test_openai_responses_live_tool_stream_executes_raw_json_tool_protocol(
    client: TestClient,
) -> None:
    calls: list[list[dict[str, str]]] = []

    async def fake_stream(request: object):
        messages = list(getattr(request, "messages", []) or [])
        calls.append(messages)
        if any(message.get("role") == "tool" for message in messages):
            yield DecodeChunk(token="The sum is 5.", index=0, finished=False)
            yield DecodeChunk(
                token="",
                index=1,
                finished=True,
                stats={"prompt_tokens": 36, "completion_tokens": 5, "finish_reason": "stop"},
            )
            return

        yield DecodeChunk(
            token='{"assistant_text":null,"tool_calls":[{"name":"add_numbers","arguments":{"a":',
            index=0,
            finished=False,
        )
        yield DecodeChunk(token='2,"b":3}}]}', index=1, finished=False)
        yield DecodeChunk(
            token="",
            index=2,
            finished=True,
            stats={"prompt_tokens": 20, "completion_tokens": 8, "finish_reason": "stop"},
        )

    client.app.state.container.inference_engine.stream = fake_stream

    with client.stream(
        "POST",
        "/v1/responses",
        json={
            "model": "gpt-5.1",
            "stream": True,
            "input": "What is 2 + 3?",
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "add_numbers",
                        "parameters": {
                            "type": "object",
                            "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                            "required": ["a", "b"],
                        },
                    },
                }
            ],
            "tool_choice": "required",
        },
    ) as response:
        body = b"".join(response.iter_bytes()).decode()

    assert response.status_code == 200
    assert len(calls) == 2
    assert '"assistant_text"' not in body
    assert '"tool_calls"' not in body
    events = _sse_events(body)
    event_names = [name for name, _ in events]
    completed = events[-1][1]["response"]

    assert "response.function_call_arguments.delta" in event_names
    assert isinstance(completed, dict)
    assert completed["output_text"] == "The sum is 5."
    assert [item["type"] for item in completed["output"]] == ["function_call", "message"]
    function_call = completed["output"][0]
    assert function_call["name"] == "add_numbers"
    assert json.loads(function_call["arguments"]) == {"a": 2, "b": 3}
    assert completed["usage"] == _responses_usage(36, 5)


def test_openai_responses_live_tool_stream_final_length_replays_tool_history(
    client: TestClient,
) -> None:
    calls: list[list[dict[str, object]]] = []
    follow_up_messages: list[dict[str, object]] = []

    async def fake_stream(request: object):
        messages = list(getattr(request, "messages", []) or [])
        calls.append(messages)
        if any(message.get("role") == "tool" for message in messages):
            yield DecodeChunk(token="The sum", index=0, finished=False)
            yield DecodeChunk(
                token="",
                index=1,
                finished=True,
                stats={"prompt_tokens": 36, "completion_tokens": 1, "finish_reason": "length"},
            )
            return

        yield DecodeChunk(token="<tool_call><function=add_numbers>", index=0, finished=False)
        yield DecodeChunk(token="<parameter=a>2</parameter>", index=1, finished=False)
        yield DecodeChunk(
            token="<parameter=b>3</parameter></function></tool_call>", index=2, finished=False
        )
        yield DecodeChunk(
            token="",
            index=3,
            finished=True,
            stats={"prompt_tokens": 20, "completion_tokens": 8, "finish_reason": "stop"},
        )

    client.app.state.container.inference_engine.stream = fake_stream

    async def fake_submit(request: object) -> SimpleNamespace:
        follow_up_messages.extend(getattr(request, "messages", []) or [])
        return SimpleNamespace(
            request_id=getattr(request, "trace_id", "resp-follow-up"),
            text="continued",
            prompt_tokens=20,
            completion_tokens=1,
            cache_hit=False,
            speculative_enabled=False,
            finish_reason="stop",
        )

    client.app.state.container.inference_engine.submit = fake_submit

    with client.stream(
        "POST",
        "/v1/responses",
        json={
            "model": "gpt-5.1",
            "stream": True,
            "input": "What is 2 + 3?",
            "max_output_tokens": 1,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "add_numbers",
                        "parameters": {
                            "type": "object",
                            "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                            "required": ["a", "b"],
                        },
                    },
                }
            ],
            "tool_choice": "required",
        },
    ) as response:
        body = b"".join(response.iter_bytes()).decode()

    assert response.status_code == 200
    assert len(calls) == 2
    events = _sse_events(body)
    completed = events[-1][1]["response"]

    assert isinstance(completed, dict)
    assert completed["status"] == "incomplete"
    assert completed["incomplete_details"] == {"reason": "max_output_tokens"}
    assert completed["output_text"] == "The sum"
    assert [item["type"] for item in completed["output"]] == ["function_call", "message"]
    assert completed["usage"] == _responses_usage(36, 1)

    follow_up = client.post(
        "/v1/responses",
        json={
            "model": "gpt-5.1",
            "previous_response_id": completed["id"],
            "input": "Continue.",
        },
    )

    assert follow_up.status_code == 200
    assert follow_up_messages[0] == {"role": "user", "content": "What is 2 + 3?"}
    assert follow_up_messages[1]["role"] == "assistant"
    tool_call = follow_up_messages[1]["tool_calls"][0]
    assert tool_call["function"]["name"] == "add_numbers"
    assert follow_up_messages[2] == {
        "role": "tool",
        "tool_call_id": tool_call["id"],
        "content": '{"sum": 5}',
    }
    assert follow_up_messages[3] == {"role": "assistant", "content": "The sum"}
    assert follow_up_messages[4] == {"role": "user", "content": "Continue."}


def test_openai_chat_completions_stream_supports_structured_outputs(client: TestClient) -> None:
    async def fake_stream(_request: object):
        yield DecodeChunk(token="```json\n", index=0, finished=False)
        yield DecodeChunk(token='{"answer":"Sunny","confidence":0.9}', index=1, finished=False)
        yield DecodeChunk(token="\n```", index=2, finished=False)
        yield DecodeChunk(
            token="",
            index=3,
            finished=True,
            stats={"prompt_tokens": 18, "completion_tokens": 8, "finish_reason": "stop"},
        )

    client.app.state.container.inference_engine.stream = fake_stream

    with client.stream(
        "POST",
        "/v1/chat/completions",
        headers={"X-Request-Id": "chat-structured-stream-trace"},
        json={
            "model": "gpt-4.1",
            "stream": True,
            "messages": [{"role": "user", "content": "Return structured weather data"}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "weather_response",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "answer": {"type": "string"},
                            "confidence": {"type": "number"},
                        },
                        "required": ["answer", "confidence"],
                        "additionalProperties": False,
                    },
                },
            },
        },
    ) as response:
        body = b"".join(response.iter_bytes()).decode()

    assert response.status_code == 200
    assert response.headers["X-Request-Id"] == "chat-structured-stream-trace"
    assert "```" not in body
    assert "Sunny" in body
    assert "confidence" in body
    chunks = [
        item for item in _sse_json_payloads(body) if item.get("object") == "chat.completion.chunk"
    ]
    assert _sse_nonempty_lines(body)[-1] == "data: [DONE]"
    assert len({chunk["id"] for chunk in chunks}) == 1
    assert chunks[0]["id"].startswith("chatcmpl-")
    assert chunks[0]["id"] != "chat-structured-stream-trace"
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant"}
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"


def test_openai_chat_completions_structured_stream_emits_error_payload_on_aster_error(
    client: TestClient,
) -> None:
    async def fake_stream(_request: object):
        yield DecodeChunk(token='{"answer":', index=0, finished=False)
        raise AsterError(
            code="request_timeout",
            message="Inference request timed out",
            status_code=504,
            details={"timeout_seconds": 0.01},
        )

    client.app.state.container.inference_engine.stream = fake_stream

    with client.stream(
        "POST",
        "/v1/chat/completions",
        headers={"X-Request-Id": "chat-structured-stream-error"},
        json={
            "model": "gpt-4.1",
            "stream": True,
            "messages": [{"role": "user", "content": "Return structured weather data"}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "weather_response",
                    "schema": {
                        "type": "object",
                        "properties": {"answer": {"type": "string"}},
                        "required": ["answer"],
                        "additionalProperties": False,
                    },
                },
            },
        },
    ) as response:
        body = b"".join(response.iter_bytes()).decode()

    assert response.status_code == 200
    assert response.headers["X-Request-Id"] == "chat-structured-stream-error"
    payloads = _sse_json_payloads(body)
    assert payloads[-1]["error"] == {
        "message": "Inference request timed out",
        "type": "request_timeout",
        "code": "request_timeout",
        "details": {"timeout_seconds": 0.01},
    }
    assert _sse_nonempty_lines(body)[-1] == "data: [DONE]"
