from __future__ import annotations

import asyncio
from dataclasses import dataclass

from aster.core.config import RuntimeSettings
from aster.core.errors import OverloadedError
from aster.inference.engine import InferenceEngine, InferenceRequest, InferenceResponse
from aster.scheduler.adaptive_batcher import AdaptiveBatcher
from aster.scheduler.policy_engine import PolicyEngine
from aster.telemetry.logging import get_logger
from aster.telemetry.metrics import MetricsRegistry


@dataclass(slots=True)
class QueueItem:
    request: InferenceRequest
    future: asyncio.Future[InferenceResponse]


class RequestScheduler:
    def __init__(
        self,
        settings: RuntimeSettings,
        metrics: MetricsRegistry,
        inference_engine: InferenceEngine,
        policy_engine: PolicyEngine,
    ) -> None:
        self.settings = settings
        self.metrics = metrics
        self.inference_engine = inference_engine
        self.policy_engine = policy_engine
        self.batcher = AdaptiveBatcher(settings.batch)
        self._queue: asyncio.Queue[QueueItem] = asyncio.Queue(maxsize=settings.api.max_queue_depth)
        self._task: asyncio.Task[None] | None = None
        self.logger = get_logger(__name__)

    async def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name="aster-scheduler")

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def submit(self, request: InferenceRequest) -> InferenceResponse:
        if self._queue.full():
            self.logger.warning("scheduler_queue_full")
            raise OverloadedError(code="queue_full", message="Scheduler queue is full", status_code=503)
        loop = asyncio.get_running_loop()
        future: asyncio.Future[InferenceResponse] = loop.create_future()
        self.logger.info(
            "scheduler_submit_start",
            extra={
                "request_id": request.trace_id,
                "request_kind": "chat" if request.messages else "prompt",
                "queue_depth_before": self._queue.qsize(),
                "stream": request.stream,
                "max_tokens": request.max_tokens,
            },
        )
        await self._queue.put(QueueItem(request=request, future=future))
        self.metrics.queue_depth.set(self._queue.qsize())
        self.logger.info(
            "scheduler_submit_enqueued",
            extra={"request_id": request.trace_id, "queue_depth_after": self._queue.qsize()},
        )
        return await future

    async def _run(self) -> None:
        while True:
            first = await self._queue.get()
            pending = [first]
            avg_prompt_tokens = self._estimate_request_tokens(first.request)
            decision = self.batcher.decide(self._queue.qsize() + 1, avg_prompt_tokens, self.policy_engine.current())
            self.logger.info(
                "scheduler_batch_window_start",
                extra={
                    "request_id": first.request.trace_id,
                    "queue_depth_visible": self._queue.qsize(),
                    "estimated_prompt_tokens": avg_prompt_tokens,
                    "window_ms": decision.window_ms,
                    "batch_size_target": decision.batch_size,
                },
            )
            if decision.window_ms > 0:
                await asyncio.sleep(decision.window_ms / 1000.0)
            while len(pending) < decision.batch_size and not self._queue.empty():
                pending.append(self._queue.get_nowait())
            self.metrics.batch_size.observe(len(pending))
            self.metrics.queue_depth.set(self._queue.qsize())
            self.logger.info(
                "scheduler_batch_dispatch",
                extra={
                    "request_id": first.request.trace_id,
                    "batch_size": len(pending),
                    "queue_depth_after_drain": self._queue.qsize(),
                    "concurrent_dispatch": self.inference_engine.supports_concurrent_dispatch(),
                },
            )
            if self.inference_engine.supports_concurrent_dispatch():
                await asyncio.gather(*(self._dispatch_item(item) for item in pending))
            else:
                for item in pending:
                    await self._dispatch_item(item)

    def is_running(self) -> bool:
        return self._task is not None and not self._task.done()

    async def _dispatch_item(self, item: QueueItem) -> None:
        try:
            self.logger.info(
                "scheduler_infer_start",
                extra={
                    "request_id": item.request.trace_id,
                    "request_kind": "chat" if item.request.messages else "prompt",
                    "stream": item.request.stream,
                    "max_tokens": item.request.max_tokens,
                },
            )
            result = await self.inference_engine.infer(item.request)
            if not item.future.done():
                item.future.set_result(result)
            self.logger.info(
                "scheduler_infer_finish",
                extra={
                    "request_id": result.request_id,
                    "completion_tokens": result.completion_tokens,
                    "cache_hit": result.cache_hit,
                    "speculative_enabled": result.speculative_enabled,
                },
            )
        except Exception as exc:
            self.logger.exception("scheduler_infer_failed", extra={"request_id": item.request.trace_id})
            if not item.future.done():
                item.future.set_exception(exc)

    def _estimate_request_tokens(self, request: InferenceRequest) -> int:
        if request.prompt:
            return max(1, len(request.prompt.split()))
        if request.messages:
            content = " ".join(str(message.get("content", "")) for message in request.messages)
            return max(1, len(content.split()))
        return 1
