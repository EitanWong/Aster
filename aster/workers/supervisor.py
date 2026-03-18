from __future__ import annotations

import asyncio
import time

from aster.core.config import RuntimeSettings
from aster.inference.engine import InferenceEngine
from aster.scheduler.scheduler import RequestScheduler
from aster.telemetry.logging import get_logger
from aster.telemetry.metrics import MetricsRegistry
from aster.workers.inference_worker import InferenceWorker


class WorkerSupervisor:
    def __init__(self, settings: RuntimeSettings, metrics: MetricsRegistry, inference_engine: InferenceEngine) -> None:
        self.settings = settings
        self.metrics = metrics
        self.inference_engine = inference_engine
        self.worker = InferenceWorker(inference_engine)
        self.scheduler: RequestScheduler | None = None
        self.logger = get_logger(__name__)
        self._monitor_task: asyncio.Task[None] | None = None
        self._restart_count = 0

    async def attach_scheduler(self, scheduler: RequestScheduler) -> None:
        self.scheduler = scheduler

    async def start(self) -> None:
        await self.worker.start()
        if self.scheduler is not None:
            await self.scheduler.start()
        self._monitor_task = asyncio.create_task(self._monitor(), name="aster-worker-supervisor")

    async def stop(self) -> None:
        if self._monitor_task is not None:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            self._monitor_task = None
        if self.scheduler is not None:
            await self.scheduler.stop()
        await self.worker.stop()

    async def _monitor(self) -> None:
        while True:
            await asyncio.sleep(2.0)
            age = time.monotonic() - self.worker.last_heartbeat
            if age > self.settings.workers.heartbeat_timeout_seconds:
                self._restart_count += 1
                self.metrics.worker_restarts.inc()
                self.logger.error("worker_heartbeat_timeout")
                await self.worker.stop()
                await self.worker.start()
                if self._restart_count >= self.settings.workers.restart_limit:
                    self.logger.error("worker_restart_limit_exceeded")
                    break

    def status(self) -> dict[str, object]:
        heartbeat_age = time.monotonic() - self.worker.last_heartbeat
        scheduler_running = self.scheduler is not None and self.scheduler.is_running()
        monitor_running = self._monitor_task is not None and not self._monitor_task.done()
        worker_healthy = heartbeat_age <= self.settings.workers.heartbeat_timeout_seconds
        degraded = (not worker_healthy) or (not scheduler_running) or self._restart_count >= self.settings.workers.degraded_after_restarts
        return {
            "worker_heartbeat_age_s": round(heartbeat_age, 3),
            "heartbeat_timeout_s": self.settings.workers.heartbeat_timeout_seconds,
            "worker_healthy": worker_healthy,
            "scheduler_running": scheduler_running,
            "monitor_running": monitor_running,
            "restart_count": self._restart_count,
            "restart_limit": self.settings.workers.restart_limit,
            "degraded_after_restarts": self.settings.workers.degraded_after_restarts,
            "degraded": degraded,
        }
