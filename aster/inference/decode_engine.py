from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass

from aster.telemetry.metrics import MetricsRegistry


@dataclass(slots=True)
class DecodeChunk:
    token: str
    index: int
    finished: bool = False
    stats: dict[str, object] | None = None


class DecodeEngine:
    def __init__(self, metrics: MetricsRegistry) -> None:
        self.metrics = metrics

    async def generate(self, max_tokens: int, cadence_ms: float) -> AsyncIterator[DecodeChunk]:
        start = time.perf_counter()
        for i in range(max_tokens):
            await asyncio.sleep(max(cadence_ms / 1000.0, 0.002))
            if i == 0:
                self.metrics.first_token_latency.observe(time.perf_counter() - start)
            yield DecodeChunk(token=f"tok_{i}", index=i, finished=False)
        self.metrics.decode_latency.observe(time.perf_counter() - start)
        yield DecodeChunk(token="", index=max_tokens, finished=True)
