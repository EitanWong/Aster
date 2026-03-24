"""Tests for API routes and schemas."""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from aster.core.lifecycle import create_application
from aster.inference.decode_engine import DecodeChunk


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
  runtime: vllm_mlx
vllm_mlx:
  base_url: http://127.0.0.1:8001
  embedding_model: dummy-embedding-model
audio:
  asr_enabled: false
  tts_enabled: false
""")
    app = create_application(str(config_path))
    yield TestClient(app)


def test_health_endpoint(client: TestClient) -> None:
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    # Health can be either "ok" or "degraded" depending on supervisor state
    assert response.json()["status"] in ("ok", "degraded")


def test_ready_endpoint(client: TestClient) -> None:
    """Test readiness endpoint."""
    response = client.get("/ready")
    assert response.status_code == 200


def test_models_endpoint(client: TestClient) -> None:
    """Test models list endpoint."""
    response = client.get("/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert "data" in data
    assert len(data["data"]) > 0


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


def test_metrics_endpoint(client: TestClient) -> None:
    """Test metrics endpoint."""
    response = client.get("/metrics")
    assert response.status_code == 200
    assert b"# HELP" in response.content or b"# TYPE" in response.content


def test_embeddings_endpoint_proxies_vllm_mlx_embeddings(client: TestClient) -> None:
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

    client.app.state.container.scheduler.submit = fake_submit

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
                        {"type": "image_url", "image_url": {"url": "https://example.com/test.png"}},
                    ],
                },
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_123",
                            "type": "function",
                            "function": {"name": "lookup_weather", "arguments": "{\"city\":\"Shanghai\"}"},
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


def test_openai_responses_endpoint_uses_local_runtime_and_returns_response_shape(client: TestClient) -> None:
    async def fake_submit(_request: object) -> SimpleNamespace:
        return SimpleNamespace(
            request_id="resp_123",
            text="hello from local runtime",
            prompt_tokens=10,
            completion_tokens=4,
            cache_hit=False,
            speculative_enabled=False,
        )

    client.app.state.container.scheduler.submit = fake_submit

    response = client.post(
        "/v1/responses",
        json={
            "model": "gpt-5.1",
            "input": [{"role": "user", "content": [{"type": "input_text", "text": "Hello"}]}],
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "response"
    assert data["output"][0]["content"][0]["text"] == "hello from local runtime"


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

    client.app.state.container.scheduler.submit = fake_submit

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

    client.app.state.container.scheduler.submit = fake_submit

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

    client.app.state.container.scheduler.submit = fake_submit

    response = client.post(
        "/v2/chat",
        json={"model": "command-a", "messages": [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}]},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["message"]["content"][0]["text"] == "hello cohere"


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
        json={"model": "gpt-5.1", "stream": True, "input": [{"role": "user", "content": [{"type": "input_text", "text": "Hello"}]}]},
    ) as response:
        body = b"".join(response.iter_bytes()).decode()

    assert response.status_code == 200
    assert "response.output_text.delta" in body
    assert "response.completed" in body


def test_openai_chat_completions_supports_tool_calling_with_local_model_json_protocol(client: TestClient) -> None:
    async def fake_submit(_request: object) -> SimpleNamespace:
        return SimpleNamespace(
            request_id="chat_tool_123",
            text='{"assistant_text": null, "tool_calls": [{"name": "lookup_weather", "arguments": {"city": "Shanghai"}}]}',
            prompt_tokens=20,
            completion_tokens=12,
            cache_hit=False,
            speculative_enabled=False,
        )

    client.app.state.container.scheduler.submit = fake_submit

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-4.1",
            "messages": [{"role": "user", "content": "What is the weather?"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "lookup_weather",
                        "description": "Look up weather",
                        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
                    },
                }
            ],
            "tool_choice": "required",
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["choices"][0]["finish_reason"] == "tool_calls"
    assert data["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "lookup_weather"


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

    client.app.state.container.scheduler.submit = fake_submit

    response = client.post(
        "/v1/responses",
        json={
            "model": "gpt-5.1",
            "input": [{"role": "user", "content": [{"type": "input_text", "text": "What is the weather?"}]}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "lookup_weather",
                        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
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

    client.app.state.container.scheduler.submit = fake_submit

    response = client.post(
        "/v1/messages",
        json={
            "model": "claude-sonnet-4-5",
            "messages": [{"role": "user", "content": [{"type": "text", "text": "What is the weather?"}]}],
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
    async def fake_submit(_request: object) -> SimpleNamespace:
        return SimpleNamespace(
            request_id="chat_struct_123",
            text='{"answer":"Sunny","confidence":0.9}',
            prompt_tokens=18,
            completion_tokens=8,
            cache_hit=False,
            speculative_enabled=False,
        )

    client.app.state.container.scheduler.submit = fake_submit

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
    data = response.json()
    assert data["choices"][0]["message"]["content"] == '{"answer": "Sunny", "confidence": 0.9}'


def test_openai_chat_completions_executes_registered_tools_until_final_answer(client: TestClient) -> None:
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

    client.app.state.container.scheduler.submit = fake_submit

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


def test_openai_chat_completions_stream_emits_tool_call_then_final_answer(client: TestClient) -> None:
    async def fake_submit(request: object) -> SimpleNamespace:
        messages = getattr(request, "messages", None) or []
        if any(message.get("role") == "tool" for message in messages):
            return SimpleNamespace(
                request_id="chat_tool_stream_123",
                text='{"assistant_text":"The sum is 5.","tool_calls":[]}',
                prompt_tokens=28,
                completion_tokens=6,
                cache_hit=False,
                speculative_enabled=False,
            )
        return SimpleNamespace(
            request_id="chat_tool_stream_123",
            text='{"assistant_text":null,"tool_calls":[{"name":"add_numbers","arguments":{"a":2,"b":3}}]}',
            prompt_tokens=20,
            completion_tokens=10,
            cache_hit=False,
            speculative_enabled=False,
        )

    client.app.state.container.scheduler.submit = fake_submit

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
    assert '"tool_calls"' in body
    assert '"add_numbers"' in body
    assert "The sum is 5." in body
    assert "[DONE]" in body


def test_openai_chat_completions_stream_supports_structured_outputs(client: TestClient) -> None:
    async def fake_submit(_request: object) -> SimpleNamespace:
        return SimpleNamespace(
            request_id="chat_struct_stream_123",
            text='{"answer":"Sunny","confidence":0.9}',
            prompt_tokens=18,
            completion_tokens=8,
            cache_hit=False,
            speculative_enabled=False,
        )

    client.app.state.container.scheduler.submit = fake_submit

    with client.stream(
        "POST",
        "/v1/chat/completions",
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
    assert "Sunny" in body
    assert "confidence" in body
    assert "[DONE]" in body
