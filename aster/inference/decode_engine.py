from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class DecodeChunk:
    token: str
    index: int
    finished: bool = False
    stats: dict[str, object] | None = None
