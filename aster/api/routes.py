from __future__ import annotations

import uuid
from time import perf_counter, time

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response

from aster.api.schemas import ChatCompletionRequest, CompletionRequest, HealthResponse, ModelCard
from aster.api.streaming import to_sse
from aster.core.errors import AsterError
from aster.inference.engine import InferenceRequest
from aster.telemetry.logging import get_logger


class RouteBuilder:
    def __init__(self) -> None:
        self.router = APIRouter()
        self.logger = get_logger(__name__)
        self.router.add_api_route("/health", self.health, methods=["GET"])
        self.router.add_api_route("/ready", self.ready, methods=["GET"])
        self.router.add_api_route("/metrics", self.metrics, methods=["GET"])
        self.router.add_api_route("/v1/models", self.models, methods=["GET"])
        self.router.add_api_route("/v1/chat/completions", self.chat_completions, methods=["POST"])
        self.router.add_api_route("/v1/completions", self.completions, methods=["POST"])

    async def health(self, request: Request) -> HealthResponse:
        container = request.app.state.container
        engine_healthy = container.inference_engine.health()
        supervisor_status = container.supervisor.status()
        degraded = (not engine_healthy) or bool(supervisor_status.get("degraded", False))
        details = {
            "engine_healthy": engine_healthy,
            **supervisor_status,
        }
        return HealthResponse(status="ok" if not degraded else "degraded", degraded=degraded, details=details)

    async def ready(self, request: Request) -> HealthResponse:
        container = request.app.state.container
        engine_healthy = container.inference_engine.health()
        supervisor_status = container.supervisor.status()
        ready = engine_healthy and bool(supervisor_status.get("worker_healthy", False)) and bool(supervisor_status.get("scheduler_running", False))
        details = {
            "engine_healthy": engine_healthy,
            **supervisor_status,
        }
        return HealthResponse(status="ready" if ready else "not_ready", degraded=not ready, details=details)

    async def metrics(self, request: Request) -> Response:
        container = request.app.state.container
        return Response(content=container.metrics.render(), media_type="text/plain; version=0.0.4")

    async def models(self, request: Request) -> dict[str, object]:
        container = request.app.state.container
        models = [
            ModelCard(id=container.settings.model.name).model_dump(),
            ModelCard(id=container.settings.model.draft_name).model_dump(),
        ]
        return {"object": "list", "data": models}

    async def chat_completions(self, request: Request, body: ChatCompletionRequest) -> Response:
        container = request.app.state.container
        started = perf_counter()
        request_id = request.headers.get("X-Request-Id") or str(uuid.uuid4())
        debug_summary = request.headers.get("X-Aster-Debug") == "1"
        self.logger.info(
            "chat_request_start",
            extra={
                "request_id": request_id,
                "stream": body.stream,
                "message_count": len(body.messages),
                "max_tokens": body.max_tokens,
                "model": body.model,
                "debug_summary": debug_summary,
            },
        )
        inference_request = InferenceRequest(
            messages=[{"role": m.role, "content": m.content} for m in body.messages],
            max_tokens=body.max_tokens,
            stream=body.stream,
            temperature=body.temperature,
            top_p=body.top_p,
            trace_id=request_id,
        )
        try:
            if body.stream:
                self.logger.info("chat_stream_start", extra={"request_id": request_id, "debug_summary": debug_summary})
                return await to_sse(
                    container.inference_engine.stream(inference_request),
                    body.model,
                    include_debug_summary=debug_summary,
                )
            self.logger.info("chat_scheduler_submit", extra={"request_id": request_id})
            result = await container.scheduler.submit(inference_request)
            payload = {
                "id": result.request_id,
                "object": "chat.completion",
                "created": int(time()),
                "model": body.model,
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": result.text}, "finish_reason": "stop"}
                ],
                "usage": {
                    "prompt_tokens": result.prompt_tokens,
                    "completion_tokens": result.completion_tokens,
                    "total_tokens": result.prompt_tokens + result.completion_tokens,
                },
                "aster": {
                    "cache_hit": result.cache_hit,
                    "speculative_enabled": result.speculative_enabled,
                },
            }
            self.logger.info(
                "chat_non_stream_finish",
                extra={
                    "request_id": result.request_id,
                    "latency_s": round(perf_counter() - started, 4),
                    "completion_tokens": result.completion_tokens,
                },
            )
            return JSONResponse(payload, headers={"X-Request-Id": result.request_id})
        except AsterError as exc:
            self.logger.exception("chat_request_failed", extra={"request_id": request_id})
            return JSONResponse(status_code=exc.status_code, content=exc.to_payload(), headers={"X-Request-Id": request_id})
        except Exception:
            self.logger.exception("chat_request_failed_unhandled", extra={"request_id": request_id})
            raise

    async def completions(self, request: Request, body: CompletionRequest) -> Response:
        container = request.app.state.container
        started = perf_counter()
        request_id = request.headers.get("X-Request-Id") or str(uuid.uuid4())
        debug_summary = request.headers.get("X-Aster-Debug") == "1"
        self.logger.info(
            "completion_request_start",
            extra={
                "request_id": request_id,
                "stream": body.stream,
                "max_tokens": body.max_tokens,
                "model": body.model,
                "debug_summary": debug_summary,
            },
        )
        inference_request = InferenceRequest(
            prompt=body.prompt,
            max_tokens=body.max_tokens,
            stream=body.stream,
            temperature=body.temperature,
            top_p=body.top_p,
            trace_id=request_id,
        )
        try:
            if body.stream:
                self.logger.info(
                    "completion_stream_start",
                    extra={"request_id": request_id, "debug_summary": debug_summary},
                )
                return await to_sse(
                    container.inference_engine.stream(inference_request),
                    body.model,
                    include_debug_summary=debug_summary,
                )
            self.logger.info("completion_scheduler_submit", extra={"request_id": request_id})
            result = await container.scheduler.submit(inference_request)
            payload = {
                "id": result.request_id,
                "object": "text_completion",
                "created": int(time()),
                "model": body.model,
                "choices": [{"index": 0, "text": result.text, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": result.prompt_tokens,
                    "completion_tokens": result.completion_tokens,
                    "total_tokens": result.prompt_tokens + result.completion_tokens,
                },
                "aster": {
                    "cache_hit": result.cache_hit,
                    "speculative_enabled": result.speculative_enabled,
                },
            }
            self.logger.info(
                "completion_non_stream_finish",
                extra={
                    "request_id": result.request_id,
                    "latency_s": round(perf_counter() - started, 4),
                    "completion_tokens": result.completion_tokens,
                },
            )
            return JSONResponse(payload, headers={"X-Request-Id": result.request_id})
        except AsterError as exc:
            self.logger.exception("completion_request_failed", extra={"request_id": request_id})
            return JSONResponse(status_code=exc.status_code, content=exc.to_payload(), headers={"X-Request-Id": request_id})
        except Exception:
            self.logger.exception("completion_request_failed_unhandled", extra={"request_id": request_id})
            raise


def build_router() -> APIRouter:
    return RouteBuilder().router
