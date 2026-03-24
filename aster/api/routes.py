from __future__ import annotations

import uuid
from time import perf_counter, time
from typing import cast

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response

from aster.api.interaction_loop import run_interaction, stream_interaction
from aster.api.provider_gateway import build_provider_request, encode_provider_response, encode_provider_stream, provider_error_response
from aster.api.schemas import (
    ChatCompletionRequest,
    ChatMessage,
    CompletionRequest,
    ContentPart,
    EmbeddingRequest,
    HealthResponse,
    ModelCard,
    TTSRequest,
)
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
        self.router.add_api_route("/v1/responses", self.openai_responses, methods=["POST"])
        self.router.add_api_route("/v1/messages", self.anthropic_messages, methods=["POST"])
        self.router.add_api_route("/v1beta/models/{model_name}:generateContent", self.gemini_generate_content, methods=["POST"])
        self.router.add_api_route("/v1beta/models/{model_name}:streamGenerateContent", self.gemini_stream_generate_content, methods=["POST"])
        self.router.add_api_route("/v2/chat", self.cohere_chat, methods=["POST"])
        self.router.add_api_route("/model/{model_name}/converse", self.bedrock_converse, methods=["POST"])
        self.router.add_api_route("/xai/v1/responses", self.xai_responses, methods=["POST"])
        self.router.add_api_route("/xai/v1/chat/completions", self.xai_chat_completions, methods=["POST"])
        self.router.add_api_route("/mistral/v1/chat/completions", self.mistral_chat_completions, methods=["POST"])
        self.router.add_api_route("/v1/completions", self.completions, methods=["POST"])
        self.router.add_api_route("/v1/embeddings", self.embeddings, methods=["POST"])
        self.router.add_api_route("/v1/audio/transcriptions", self.transcribe, methods=["POST"])
        self.router.add_api_route("/v1/audio/speech", self.synthesize, methods=["POST"])

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
        ]
        embedding_model = container.inference_engine.configured_embedding_model()
        if container.inference_engine.supports_embeddings() and embedding_model:
            models.append(ModelCard(id=embedding_model).model_dump())
        return {"object": "list", "data": models}

    async def chat_completions(self, request: Request, body: ChatCompletionRequest) -> Response:
        if body.tools or body.tool_choice is not None or body.parallel_tool_calls is not None or body.response_format is not None:
            return await self._handle_provider_request(
                request,
                body.model_dump(exclude_none=True),
                provider="openai",
                api_family="chat_completions",
            )
        container = request.app.state.container
        started = perf_counter()
        request_id = request.headers.get("X-Request-Id") or str(uuid.uuid4())
        debug_summary = request.headers.get("X-Aster-Debug") == "1"
        normalized_messages = self._normalize_messages(body.messages)
        self.logger.info(
            "chat_request_start",
            extra={
                "request_id": request_id,
                "stream": body.stream,
                "message_count": len(normalized_messages),
                "max_tokens": body.max_tokens,
                "model": body.model,
                "debug_summary": debug_summary,
            },
        )
        inference_request = InferenceRequest(
            messages=normalized_messages,
            max_tokens=body.max_tokens,
            stream=body.stream,
            temperature=body.temperature,
            top_p=body.top_p,
            trace_id=request_id,
            enable_thinking=body.enable_thinking if body.enable_thinking is not None else container.settings.model.enable_thinking,
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

    async def embeddings(self, request: Request, body: EmbeddingRequest) -> Response:
        container = request.app.state.container
        request_id = request.headers.get("X-Request-Id") or str(uuid.uuid4())
        try:
            payload = await container.inference_engine.embeddings(
                model=body.model,
                input_data=body.input,
            )
            return JSONResponse(payload, headers={"X-Request-Id": request_id})
        except AsterError as exc:
            self.logger.exception("embeddings_request_failed", extra={"request_id": request_id})
            return JSONResponse(status_code=exc.status_code, content=exc.to_payload(), headers={"X-Request-Id": request_id})
        except Exception as exc:
            self.logger.exception("embeddings_request_failed_unhandled", extra={"request_id": request_id})
            return JSONResponse(status_code=500, content={"error": str(exc)}, headers={"X-Request-Id": request_id})

    async def transcribe(self, request: Request) -> dict:
        """Transcribe audio to text (ASR)."""
        container = request.app.state.container
        if not container.audio.asr:
            return JSONResponse(
                status_code=503,
                content={"error": "ASR service not available"},
            )

        try:
            form = await request.form()
            audio_file = form.get("file")
            if not audio_file:
                return JSONResponse(
                    status_code=400,
                    content={"error": "No audio file provided"},
                )

            audio_data = await audio_file.read()
            language = form.get("language")
            prompt = form.get("prompt")

            result = await container.audio.asr.transcribe(
                audio=audio_data,
                language=language,
                prompt=prompt,
            )

            return {
                "text": result.text,
                "language": result.language,
                "duration": result.duration,
                "confidence": result.confidence,
            }
        except Exception as e:
            self.logger.exception("transcribe_failed")
            return JSONResponse(
                status_code=500,
                content={"error": str(e)},
            )

    def _normalize_messages(self, messages: list[ChatMessage]) -> list[dict[str, str]]:
        return [
            {"role": self._normalize_role(message.role), "content": self._flatten_message_content(message)}
            for message in messages
        ]

    def _normalize_role(self, role: str) -> str:
        if role == "developer":
            return "system"
        if role == "function":
            return "tool"
        return role

    def _flatten_message_content(self, message: ChatMessage) -> str:
        content = message.content
        if isinstance(content, str):
            return content
        if content is None:
            return ""
        parts = [self._flatten_content_part(part) for part in content]
        return "\n".join(part for part in parts if part).strip()

    def _flatten_content_part(self, part: ContentPart) -> str:
        if part.text:
            return part.text
        if part.input_text:
            return part.input_text
        if part.content:
            return part.content
        extra = cast(dict[str, object], getattr(part, "__pydantic_extra__", None) or {})
        if isinstance(extra.get("text"), str):
            return cast(str, extra["text"])
        if isinstance(extra.get("input_text"), str):
            return cast(str, extra["input_text"])
        if isinstance(extra.get("content"), str):
            return cast(str, extra["content"])
        if part.type == "image_url":
            return "[image]"
        if part.type == "input_audio":
            return "[audio]"
        if part.type:
            return f"[{part.type}]"
        return ""

    async def openai_responses(self, request: Request) -> Response:
        body = await request.json()
        return await self._handle_provider_request(request, body, provider="openai", api_family="responses")

    async def anthropic_messages(self, request: Request) -> Response:
        body = await request.json()
        return await self._handle_provider_request(request, body, provider="anthropic", api_family="messages")

    async def gemini_generate_content(self, request: Request, model_name: str) -> Response:
        body = await request.json()
        return await self._handle_provider_request(
            request,
            body,
            provider="gemini",
            api_family="generate_content",
            model_name=model_name,
        )

    async def gemini_stream_generate_content(self, request: Request, model_name: str) -> Response:
        body = await request.json()
        return await self._handle_provider_request(
            request,
            body,
            provider="gemini",
            api_family="stream_generate_content",
            model_name=model_name,
        )

    async def cohere_chat(self, request: Request) -> Response:
        body = await request.json()
        return await self._handle_provider_request(request, body, provider="cohere", api_family="chat_v2")

    async def bedrock_converse(self, request: Request, model_name: str) -> Response:
        body = await request.json()
        return await self._handle_provider_request(
            request,
            body,
            provider="bedrock",
            api_family="converse",
            model_name=model_name,
        )

    async def xai_responses(self, request: Request) -> Response:
        body = await request.json()
        return await self._handle_provider_request(request, body, provider="xai", api_family="responses")

    async def xai_chat_completions(self, request: Request) -> Response:
        body = await request.json()
        return await self._handle_provider_request(request, body, provider="xai", api_family="chat_completions")

    async def mistral_chat_completions(self, request: Request) -> Response:
        body = await request.json()
        return await self._handle_provider_request(request, body, provider="mistral", api_family="chat_completions")

    async def _handle_provider_request(
        self,
        request: Request,
        body: dict[str, object],
        *,
        provider: str,
        api_family: str,
        model_name: str | None = None,
    ) -> Response:
        container = request.app.state.container
        request_id = request.headers.get("X-Request-Id") or str(uuid.uuid4())
        try:
            parsed = build_provider_request(
                provider=provider,
                api_family=api_family,
                body=body,
                model=model_name,
                request_id=request_id,
            )
            if parsed.stream:
                if parsed.feature_plan.mode != "plain":
                    return await stream_interaction(container, parsed)
                return encode_provider_stream(parsed, container.inference_engine.stream(parsed.inference_request))
            if parsed.feature_plan.mode == "tools":
                trace = await run_interaction(container, parsed)
                return encode_provider_response(parsed, trace.final_result)
            result = await container.scheduler.submit(parsed.inference_request)
            return encode_provider_response(parsed, result)
        except AsterError as exc:
            self.logger.exception(
                "provider_request_failed",
                extra={"request_id": request_id, "provider": provider, "api_family": api_family},
            )
            return provider_error_response(provider, api_family, exc, request_id=request_id)
        except Exception:
            self.logger.exception(
                "provider_request_failed_unhandled",
                extra={"request_id": request_id, "provider": provider, "api_family": api_family},
            )
            raise

    async def synthesize(self, request: Request, body: TTSRequest) -> Response:
        """Synthesize text to speech (TTS)."""
        container = request.app.state.container
        if not container.audio.tts:
            return JSONResponse(
                status_code=503,
                content={"error": "TTS service not available"},
            )

        try:
            result = await container.audio.tts.synthesize(
                text=body.input,
                voice=body.voice,
                language=body.language,
                speed=body.speed,
                reference_audio=body.reference_audio,
                speaker=body.speaker,
                instruct=body.instruct,
            )

            return Response(
                content=result.audio,
                media_type="audio/wav",
                headers={"Content-Disposition": "attachment; filename=output.wav"},
            )
        except Exception as e:
            self.logger.exception("synthesize_failed")
            return JSONResponse(
                status_code=500,
                content={"error": str(e)},
            )



def build_router() -> APIRouter:
    return RouteBuilder().router
