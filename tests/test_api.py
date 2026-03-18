"""Tests for API routes and schemas."""

import pytest
from fastapi.testclient import TestClient

from aster.core.lifecycle import create_application


@pytest.fixture
def client(tmp_path):
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
    return TestClient(app)


def test_health_endpoint(client):
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_ready_endpoint(client):
    """Test readiness endpoint."""
    response = client.get("/ready")
    assert response.status_code == 200


def test_models_endpoint(client):
    """Test models list endpoint."""
    response = client.get("/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert "data" in data
    assert len(data["data"]) > 0


def test_metrics_endpoint(client):
    """Test metrics endpoint."""
    response = client.get("/metrics")
    assert response.status_code == 200
    assert b"# HELP" in response.content or b"# TYPE" in response.content
