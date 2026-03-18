from __future__ import annotations

import asyncio
import time

from aster.inference.engine import InferenceEngine


class InferenceWorker:
    def __init__(self, inference_engine: InferenceEngine) -> None:
        self.inference_engine = inference_engine
        self._task: asyncio.Task[None] | None = None
        self._last_heartbeat = time.monotonic()

    async def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._heartbeat_loop(), name="aster-worker-heartbeat")

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _heartbeat_loop(self) -> None:
        while True:
            self._last_heartbeat = time.monotonic()
            await asyncio.sleep(1.0)

    @property
    def last_heartbeat(self) -> float:
        return self._last_heartbeat
