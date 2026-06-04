from __future__ import annotations

from collections.abc import Generator
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from aster.core.lifecycle import create_application


@pytest.fixture
def client(tmp_path: Path) -> Generator[TestClient, None, None]:
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
    default_voice: af_heart
""")
    yield TestClient(create_application(str(config_path)))


def test_mcp_endpoints_report_empty_unconfigured_state(client: TestClient) -> None:
    tools = client.get("/v1/mcp/tools")
    servers = client.get("/v1/mcp/servers")
    execute = client.post(
        "/v1/mcp/execute",
        json={"tool_name": "filesystem.read", "arguments": {"path": "/tmp/file"}},
    )

    assert tools.status_code == 200
    assert tools.json() == {"tools": [], "count": 0}
    assert servers.status_code == 200
    assert servers.json() == {"servers": []}
    assert execute.status_code == 503
    assert execute.json() == {
        "detail": "MCP not configured. Start server with --mcp-config"
    }


def test_rerank_endpoint_matches_unconfigured_vllm_mlx_error(client: TestClient) -> None:
    response = client.post(
        "/v1/rerank",
        json={
            "model": "reranker",
            "query": "compatibility",
            "documents": ["Aster", {"text": "vllm-mlx"}],
        },
    )

    assert response.status_code == 404
    assert response.json() == {
        "detail": (
            "No reranker model loaded. Start the server with --rerank-model "
            "to enable the /v1/rerank endpoint."
        )
    }


def test_rerank_endpoint_validates_request_before_unconfigured_error(
    client: TestClient,
) -> None:
    response = client.post(
        "/v1/rerank",
        json={
            "model": "reranker",
            "query": "compatibility",
            "documents": ["Aster"],
            "top_n": 2,
        },
    )

    assert response.status_code == 400
    assert response.json() == {
        "detail": "top_n (2) must not exceed the number of documents (1)"
    }


def test_anthropic_count_tokens_uses_engine_token_counter(client: TestClient) -> None:
    seen: list[str] = []

    async def fake_count_text_tokens(texts: tuple[str, ...]) -> int:
        seen.extend(texts)
        return 321

    client.app.state.container.inference_engine.count_text_tokens = fake_count_text_tokens

    response = client.post(
        "/v1/messages/count_tokens",
        headers={"X-Request-Id": "count-tokens"},
        json={
            "model": "claude-compatible",
            "system": [{"type": "text", "text": "system text"}],
            "messages": [
                {"role": "user", "content": "plain content"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "block text"},
                        {"type": "tool_use", "input": {"city": "Shanghai"}},
                        {
                            "type": "tool_result",
                            "content": [{"type": "text", "text": "tool result"}],
                        },
                    ],
                },
            ],
            "tools": [
                {
                    "name": "weather",
                    "description": "Weather lookup",
                    "input_schema": {"type": "object"},
                }
            ],
        },
    )

    assert response.status_code == 200
    assert response.headers["X-Request-Id"] == "count-tokens"
    assert response.json() == {"input_tokens": 321}
    assert "system text" in seen
    assert "plain content" in seen
    assert "block text" in seen
    assert "tool result" in seen
    assert "weather" in seen
    assert "Weather lookup" in seen
    assert any("Shanghai" in text for text in seen)
    assert any("object" in text for text in seen)


def test_audio_voices_endpoint_reports_default_or_runtime_voices(
    client: TestClient,
) -> None:
    default_response = client.get("/v1/audio/voices")
    assert default_response.status_code == 200
    assert default_response.json() == {"voices": ["af_heart"]}

    client.app.state.container.audio.tts = SimpleNamespace(
        list_voices=lambda: ["voice-a", "voice-b"]
    )
    runtime_response = client.get("/v1/audio/voices?model=custom")

    assert runtime_response.status_code == 200
    assert runtime_response.json() == {"voices": ["voice-a", "voice-b"]}
