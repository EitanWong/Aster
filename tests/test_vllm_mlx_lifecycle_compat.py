from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from aster.core.errors import AsterError
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
""")
    yield TestClient(create_application(str(config_path)))


def _client_disconnected() -> AsterError:
    return AsterError(
        code="client_disconnected",
        message="Client disconnected before inference completed",
        status_code=499,
    )


def test_chat_non_stream_client_disconnect_returns_empty_499(client: TestClient) -> None:
    async def fake_submit(request: object) -> object:
        del request
        raise _client_disconnected()

    client.app.state.container.inference_engine.submit = fake_submit

    response = client.post(
        "/v1/chat/completions",
        headers={"X-Request-Id": "chat-disconnected"},
        json={
            "model": "dummy-model",
            "messages": [{"role": "user", "content": "Hello"}],
        },
    )

    assert response.status_code == 499
    assert response.headers["X-Request-Id"] == "chat-disconnected"
    assert response.content == b""


def test_completion_non_stream_client_disconnect_returns_empty_499(
    client: TestClient,
) -> None:
    async def fake_submit(request: object) -> object:
        del request
        raise _client_disconnected()

    client.app.state.container.inference_engine.submit = fake_submit

    response = client.post(
        "/v1/completions",
        headers={"X-Request-Id": "completion-disconnected"},
        json={"model": "dummy-model", "prompt": "Hello"},
    )

    assert response.status_code == 499
    assert response.headers["X-Request-Id"] == "completion-disconnected"
    assert response.content == b""


def test_provider_non_stream_client_disconnect_returns_empty_499(
    client: TestClient,
) -> None:
    async def fake_submit(request: object) -> object:
        del request
        raise _client_disconnected()

    client.app.state.container.inference_engine.submit = fake_submit

    response = client.post(
        "/v1/responses",
        headers={"X-Request-Id": "responses-disconnected"},
        json={"model": "dummy-model", "input": "Hello"},
    )

    assert response.status_code == 499
    assert response.headers["X-Request-Id"] == "responses-disconnected"
    assert response.content == b""
