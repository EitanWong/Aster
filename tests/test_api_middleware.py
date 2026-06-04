from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from aster.core.lifecycle import create_application


def _client(tmp_path: Path, api_block: str) -> TestClient:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
api:
{api_block}
model:
  name: dummy-model
  path: dummy
audio:
  asr:
    enabled: false
  tts:
    enabled: false
"""
    )
    return TestClient(create_application(str(config_path)))


def test_api_key_protects_non_public_routes(tmp_path: Path) -> None:
    with _client(tmp_path, "  api_key: secret\n") as client:
        public_response = client.get("/health")
        missing_response = client.get("/v1/models")
        invalid_response = client.get("/v1/models", headers={"Authorization": "Bearer wrong"})
        valid_response = client.get("/v1/models", headers={"Authorization": "Bearer secret"})

    assert public_response.status_code == 200
    assert missing_response.status_code == 401
    assert invalid_response.status_code == 401
    assert valid_response.status_code == 200


def test_rate_limit_protects_non_public_routes(tmp_path: Path) -> None:
    with _client(tmp_path, "  rate_limit_per_minute: 1\n") as client:
        public_response = client.get("/health")
        first_response = client.get("/v1/models")
        second_response = client.get("/v1/models")

    assert public_response.status_code == 200
    assert first_response.status_code == 200
    assert second_response.status_code == 429
    assert "Retry-After" in second_response.headers

