"""Tests for API routes and schemas."""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from aster.core.lifecycle import create_application


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
  path: "dummy"
  draft_path: "dummy"
cache:
  max_pages: 100
  page_size: 512
scheduler:
  batch_window_ms: 10
  max_batch_size: 4
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


def test_metrics_endpoint(client: TestClient) -> None:
    """Test metrics endpoint."""
    response = client.get("/metrics")
    assert response.status_code == 200
    assert b"# HELP" in response.content or b"# TYPE" in response.content
