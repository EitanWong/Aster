"""Pytest configuration and fixtures."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Generator

import pytest

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def tmp_config(tmp_path: Path) -> Generator[str, None, None]:
    """Create a temporary config file for testing."""
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
    yield str(config_path)


@pytest.fixture
def minimal_config(tmp_path: Path) -> Generator[str, None, None]:
    """Create a minimal config file for testing."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text("""
logging:
  level: DEBUG
""")
    yield str(config_path)
