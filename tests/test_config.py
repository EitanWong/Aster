from __future__ import annotations

from pathlib import Path

from aster.core.config import load_settings


def test_load_settings(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("logging:\n  level: DEBUG\n")
    settings = load_settings(str(path))
    assert settings.logging.level == "DEBUG"
