from __future__ import annotations

import json
from pathlib import Path
from typing import Any


FIXTURES_ROOT = Path(__file__).resolve().parents[2] / "tests" / "fixtures"


def load_fixture(*parts: str) -> dict[str, Any]:
    path = FIXTURES_ROOT.joinpath(*parts)
    return json.loads(path.read_text())
