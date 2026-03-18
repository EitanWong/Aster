from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(slots=True)
class TuningProfile:
    name: str
    speculative_enabled: bool
    draft_tokens: int
    batch_window_ms: float
    max_batch_size: int
    stream_flush_ms: float
    score: float

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_path(cls, path: str) -> TuningProfile | None:
        file = Path(path)
        if not file.exists():
            return None
        return cls(**json.loads(file.read_text()))
