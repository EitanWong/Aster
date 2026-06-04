from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import AsyncIterator, Awaitable, Mapping
from contextlib import suppress
from pathlib import Path
from time import perf_counter, time
from typing import Any, TypeVar, cast

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, Response

from aster.api.content_parts import is_multimodal_content_type
from aster.api.disconnect import await_with_disconnect
from aster.api.feature_emulation import DecodedLocalOutput, decode_local_output
from aster.api.interaction_loop import (
    responses_replay_messages_from_trace,
    run_interaction,
    stream_interaction,
    stream_live_tool_interaction,
)
from aster.api.model_policies import (
    validate_embedding_model,
    validate_stt_model,
    validate_tts_model,
)
from aster.api.provider_gateway import (
    LocalProviderRequest,
    build_provider_request,
    encode_provider_decoded_response,
    encode_provider_stream,
    provider_error_response,
    responses_replay_output_messages,
    supports_provider_live_tool_stream,
    supports_provider_structured_stream,
)
from aster.api.responses_store import DEFAULT_RESPONSES_STORE_MAX_SIZE, ResponsesStore
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
from aster.api.streaming import to_chat_sse, to_completion_sse
from aster.audio.limits import save_upload_with_limit, validate_tts_input_length
from aster.core.errors import AsterError
from aster.inference.decode_engine import DecodeChunk
from aster.inference.engine import InferenceRequest
from aster.inference.reasoning_parsers import parse_reasoning_output
from aster.telemetry.logging import get_logger
from aster.telemetry.metrics import MetricsRegistry

DEFAULT_MAX_TOKENS = 256
T = TypeVar("T")
CLIENT_DISCONNECTED_CODE = "client_disconnected"


class RouteBuilder:
    def __init__(
        self, *, responses_store_max_entries: int = DEFAULT_RESPONSES_STORE_MAX_SIZE
    ) -> None:
        self.router = APIRouter()
        self.logger = get_logger(__name__)
        self._responses_store = ResponsesStore(max_size=responses_store_max_entries)
        self.router.add_api_route("/health", self.health, methods=["GET"])
        self.router.add_api_route("/ready", self.ready, methods=["GET"])
        self.router.add_api_route("/metrics", self.metrics, methods=["GET"])
        self.router.add_api_route("/v1/status", self.status, methods=["GET"])
        self.router.add_api_route("/v1/models", self.models, methods=["GET"])
        self.router.add_api_route("/v1/cache/stats", self.cache_stats, methods=["GET"])
        self.router.add_api_route("/v1/cache", self.clear_cache, methods=["DELETE"])
        self.router.add_api_route("/v1/cache/prefix", self.clear_prefix_cache, methods=["DELETE"])
        self.router.add_api_route(
            "/v1/requests/{request_id}/cancel", self.cancel_request, methods=["POST"]
        )
        self.router.add_api_route(
            "/v1/requests/{request_id}", self.delete_request, methods=["DELETE"]
        )
        self.router.add_api_route("/v1/chat/completions", self.chat_completions, methods=["POST"])
        self.router.add_api_route("/v1/responses", self.openai_responses, methods=["POST"])
        self.router.add_api_route("/v1/messages", self.anthropic_messages, methods=["POST"])
        self.router.add_api_route(
            "/v1/messages/count_tokens",
            self.anthropic_count_tokens,
            methods=["POST"],
        )
        self.router.add_api_route("/v1/rerank", self.rerank, methods=["POST"])
        self.router.add_api_route("/v1/mcp/tools", self.mcp_tools, methods=["GET"])
        self.router.add_api_route("/v1/mcp/servers", self.mcp_servers, methods=["GET"])
        self.router.add_api_route("/v1/mcp/execute", self.mcp_execute, methods=["POST"])
        self.router.add_api_route(
            "/v1beta/models/{model_name}:generateContent",
            self.gemini_generate_content,
            methods=["POST"],
        )
        self.router.add_api_route(
            "/v1beta/models/{model_name}:streamGenerateContent",
            self.gemini_stream_generate_content,
            methods=["POST"],
        )
        self.router.add_api_route("/v2/chat", self.cohere_chat, methods=["POST"])
        self.router.add_api_route(
            "/model/{model_name}/converse", self.bedrock_converse, methods=["POST"]
        )
        self.router.add_api_route("/xai/v1/responses", self.xai_responses, methods=["POST"])
        self.router.add_api_route(
            "/xai/v1/chat/completions", self.xai_chat_completions, methods=["POST"]
        )
        self.router.add_api_route(
            "/mistral/v1/chat/completions", self.mistral_chat_completions, methods=["POST"]
        )
        self.router.add_api_route("/v1/completions", self.completions, methods=["POST"])
        self.router.add_api_route("/v1/embeddings", self.embeddings, methods=["POST"])
        self.router.add_api_route("/v1/audio/transcriptions", self.transcribe, methods=["POST"])
        self.router.add_api_route("/v1/audio/speech", self.synthesize, methods=["POST"])
        self.router.add_api_route("/v1/audio/voices", self.audio_voices, methods=["GET"])

    async def health(self, request: Request) -> HealthResponse:
        container = request.app.state.container
        engine_healthy = container.inference_engine.health()
        engine_status = container.inference_engine.status()
        degraded = not engine_healthy
        details = {"engine_healthy": engine_healthy, **engine_status}
        return HealthResponse(
            status="ok" if not degraded else "degraded", degraded=degraded, details=details
        )

    async def ready(self, request: Request) -> HealthResponse:
        container = request.app.state.container
        engine_healthy = container.inference_engine.health()
        engine_status = container.inference_engine.status()
        ready = engine_healthy
        details = {"engine_healthy": engine_healthy, **engine_status}
        return HealthResponse(
            status="ready" if ready else "not_ready", degraded=not ready, details=details
        )

    async def metrics(self, request: Request) -> Response:
        container = request.app.state.container
        return Response(content=container.metrics.render(), media_type="text/plain; version=0.0.4")

    async def status(self, request: Request) -> dict[str, object]:
        container = request.app.state.container
        status = container.inference_engine.status()
        status["responses_store"] = {
            "entries": len(self._responses_store),
            "max_entries": self._responses_store.max_size,
            "scope": "process/provider",
        }
        return status

    async def models(self, request: Request) -> dict[str, object]:
        container = request.app.state.container
        models = [
            ModelCard(id=container.settings.model.name).model_dump(),
        ]
        embedding_model = container.inference_engine.configured_embedding_model()
        if container.inference_engine.supports_embeddings() and embedding_model:
            models.append(ModelCard(id=embedding_model).model_dump())
        return {"object": "list", "data": models}

    async def cache_stats(self, request: Request) -> dict[str, object]:
        container = request.app.state.container
        return {
            "object": "cache.stats",
            "engine_cache": container.inference_engine.get_cache_stats(),
        }

    async def clear_cache(self, request: Request) -> dict[str, object]:
        container = request.app.state.container
        return {
            "status": "cleared",
            "engine_cache": await container.inference_engine.clear_runtime_caches(),
        }

    async def clear_prefix_cache(self, request: Request) -> dict[str, object]:
        container = request.app.state.container
        return {
            "status": "cleared",
            "engine_cache": {
                "prefix_cache": container.inference_engine.clear_prefix_cache(),
            },
        }

    async def cancel_request(self, request: Request, request_id: str) -> Response:
        container = request.app.state.container
        try:
            cancelled = await container.inference_engine.cancel(request_id)
        except Exception as exc:
            self.logger.exception("cancel_request_failed", extra={"request_id": request_id})
            return JSONResponse(
                status_code=500,
                content={
                    "error": {
                        "message": f"Failed to cancel request {request_id}: {exc}",
                        "type": "request_cancel_failed",
                        "code": "request_cancel_failed",
                    }
                },
                headers={"X-Request-Id": request_id},
            )
        if not cancelled:
            return JSONResponse(
                status_code=404,
                content={
                    "error": {
                        "message": f"Request not found or cancellation is unsupported: {request_id}",
                        "type": "request_not_found",
                        "code": "request_not_found",
                    }
                },
                headers={"X-Request-Id": request_id},
            )
        self.logger.info("cancel_request_accepted", extra={"request_id": request_id})
        return JSONResponse(
            {
                "object": "request.cancel",
                "id": request_id,
                "cancelled": True,
                "model": container.settings.model.name,
            },
            headers={"X-Request-Id": request_id},
        )

    async def delete_request(self, request: Request, request_id: str) -> Response:
        return await self.cancel_request(request, request_id)

    async def chat_completions(self, request: Request, body: ChatCompletionRequest) -> Response:
        if (
            body.tools
            or body.tool_choice is not None
            or body.parallel_tool_calls is not None
            or body.response_format is not None
        ):
            return await self._handle_provider_request(
                request,
                body.model_dump(exclude_none=True),
                provider="openai",
                api_family="chat_completions",
            )
        container = request.app.state.container
        started = perf_counter()
        request_id = request.headers.get("X-Request-Id") or str(uuid.uuid4())
        response_id = self._chat_response_id()
        debug_summary = request.headers.get("X-Aster-Debug") == "1"
        try:
            normalized_messages = self._normalize_messages(body.messages)
            max_tokens = self._resolve_chat_max_tokens(
                body,
                max_request_tokens=container.settings.api.max_request_tokens,
            )
            enable_thinking, chat_template_kwargs = self._resolve_chat_template_kwargs(
                body,
                default_enable_thinking=container.settings.model.enable_thinking,
            )
            self.logger.info(
                "chat_request_start",
                extra={
                    "request_id": request_id,
                    "stream": body.stream,
                    "message_count": len(normalized_messages),
                    "max_tokens": max_tokens,
                    "model": body.model,
                    "debug_summary": debug_summary,
                },
            )
            inference_request = InferenceRequest(
                messages=normalized_messages,
                max_tokens=max_tokens,
                stream=body.stream,
                temperature=body.temperature,
                top_p=body.top_p,
                top_k=body.top_k,
                min_p=body.min_p,
                presence_penalty=body.presence_penalty,
                frequency_penalty=body.frequency_penalty,
                repetition_penalty=body.repetition_penalty,
                stop=body.stop,
                stop_token_ids=tuple(body.stop_token_ids or ()),
                trace_id=request_id,
                request_aliases=(response_id,) if body.stream else (),
                timeout_seconds=body.timeout,
                enable_thinking=enable_thinking,
                chat_template_kwargs=chat_template_kwargs,
                thinking_token_budget=body.thinking_token_budget,
            )
            if body.stream:
                self.logger.info(
                    "chat_stream_start",
                    extra={"request_id": request_id, "debug_summary": debug_summary},
                )
                return await to_chat_sse(
                    container.inference_engine.stream(inference_request),
                    body.model,
                    response_id=response_id,
                    include_debug_summary=debug_summary,
                    include_usage=self._include_stream_usage(body.stream_options),
                    headers={"X-Request-Id": request_id},
                    raw_request=request,
                    on_error=lambda exc: container.metrics.errors.labels(code=exc.code).inc(),
                )
            self.logger.info("chat_engine_submit", extra={"request_id": request_id})
            result = await self._await_client_request(
                request,
                container.inference_engine.submit(inference_request),
                timeout_seconds=self._effective_timeout_seconds(
                    body.timeout,
                    container.settings.api.request_timeout_seconds,
                ),
            )
            parsed_output = parse_reasoning_output(result.text)
            message = {"role": "assistant", "content": parsed_output.content}
            if parsed_output.reasoning_content:
                message["reasoning_content"] = parsed_output.reasoning_content
            payload = {
                "id": response_id,
                "object": "chat.completion",
                "created": int(time()),
                "model": body.model,
                "choices": [
                    {
                        "index": 0,
                        "message": message,
                        "finish_reason": self._result_finish_reason(result),
                    }
                ],
                "usage": {
                    "prompt_tokens": result.prompt_tokens,
                    "completion_tokens": result.completion_tokens,
                    "total_tokens": result.prompt_tokens + result.completion_tokens,
                },
            }
            if debug_summary:
                payload["aster"] = {
                    "cache_hit": result.cache_hit,
                    "speculative_enabled": result.speculative_enabled,
                }
            self.logger.info(
                "chat_non_stream_finish",
                extra={
                    "request_id": result.request_id,
                    "latency_s": round(perf_counter() - started, 4),
                    "completion_tokens": result.completion_tokens,
                },
            )
            return JSONResponse(payload, headers={"X-Request-Id": request_id})
        except AsterError as exc:
            self.logger.exception("chat_request_failed", extra={"request_id": request_id})
            if self._is_client_disconnected_error(exc):
                return self._client_closed_response(request_id)
            return JSONResponse(
                status_code=exc.status_code,
                content=exc.to_payload(),
                headers={"X-Request-Id": request_id},
            )
        except Exception:
            self.logger.exception("chat_request_failed_unhandled", extra={"request_id": request_id})
            raise

    async def completions(self, request: Request, body: CompletionRequest) -> Response:
        container = request.app.state.container
        started = perf_counter()
        request_id = request.headers.get("X-Request-Id") or str(uuid.uuid4())
        response_id = self._completion_response_id()
        debug_summary = request.headers.get("X-Aster-Debug") == "1"
        prompts = body.prompt if isinstance(body.prompt, list) else [body.prompt]
        try:
            max_tokens = self._resolve_request_max_tokens(
                max_tokens=body.max_tokens,
                max_request_tokens=container.settings.api.max_request_tokens,
            )
            self.logger.info(
                "completion_request_start",
                extra={
                    "request_id": request_id,
                    "stream": body.stream,
                    "max_tokens": max_tokens,
                    "model": body.model,
                    "debug_summary": debug_summary,
                    "prompt_count": len(prompts),
                },
            )
            if body.stream:
                inference_request = self._completion_inference_request(
                    body,
                    prompt=prompts[0] if prompts else "",
                    request_id=request_id,
                    stream=True,
                    max_tokens=max_tokens,
                    request_aliases=(response_id,),
                )
                self.logger.info(
                    "completion_stream_start",
                    extra={"request_id": request_id, "debug_summary": debug_summary},
                )
                return await to_completion_sse(
                    container.inference_engine.stream(inference_request),
                    body.model,
                    response_id=response_id,
                    include_debug_summary=debug_summary,
                    headers={"X-Request-Id": request_id},
                    raw_request=request,
                    on_error=lambda exc: container.metrics.errors.labels(code=exc.code).inc(),
                )
            self.logger.info("completion_engine_submit", extra={"request_id": request_id})
            results = []
            for index, prompt in enumerate(prompts):
                result = await self._await_client_request(
                    request,
                    container.inference_engine.submit(
                        self._completion_inference_request(
                            body,
                            prompt=prompt,
                            request_id=request_id if len(prompts) == 1 else f"{request_id}-{index}",
                            stream=False,
                            max_tokens=max_tokens,
                        )
                    ),
                    timeout_seconds=self._effective_timeout_seconds(
                        body.timeout,
                        container.settings.api.request_timeout_seconds,
                    ),
                )
                results.append(result)
            prompt_tokens = sum(result.prompt_tokens for result in results)
            completion_tokens = sum(result.completion_tokens for result in results)
            payload = {
                "id": response_id,
                "object": "text_completion",
                "created": int(time()),
                "model": body.model,
                "choices": [
                    {
                        "index": index,
                        "text": result.text,
                        "finish_reason": self._result_finish_reason(result),
                    }
                    for index, result in enumerate(results)
                ],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            }
            if debug_summary:
                payload["aster"] = {
                    "cache_hit": any(result.cache_hit for result in results),
                    "speculative_enabled": any(result.speculative_enabled for result in results),
                }
            self.logger.info(
                "completion_non_stream_finish",
                extra={
                    "request_id": request_id,
                    "latency_s": round(perf_counter() - started, 4),
                    "completion_tokens": completion_tokens,
                },
            )
            return JSONResponse(payload, headers={"X-Request-Id": request_id})
        except AsterError as exc:
            self.logger.exception("completion_request_failed", extra={"request_id": request_id})
            if self._is_client_disconnected_error(exc):
                return self._client_closed_response(request_id)
            return JSONResponse(
                status_code=exc.status_code,
                content=exc.to_payload(),
                headers={"X-Request-Id": request_id},
            )
        except Exception:
            self.logger.exception(
                "completion_request_failed_unhandled", extra={"request_id": request_id}
            )
            raise

    async def embeddings(self, request: Request, body: EmbeddingRequest) -> Response:
        container = request.app.state.container
        request_id = request.headers.get("X-Request-Id") or str(uuid.uuid4())
        try:
            resolved_model = validate_embedding_model(body.model, container.settings.embeddings)
            payload = await container.inference_engine.embeddings(
                model=resolved_model,
                input_data=body.input,
            )
            return JSONResponse(payload, headers={"X-Request-Id": request_id})
        except AsterError as exc:
            self.logger.exception("embeddings_request_failed", extra={"request_id": request_id})
            return JSONResponse(
                status_code=exc.status_code,
                content=exc.to_payload(),
                headers={"X-Request-Id": request_id},
            )
        except Exception as exc:
            self.logger.exception(
                "embeddings_request_failed_unhandled", extra={"request_id": request_id}
            )
            return JSONResponse(
                status_code=500, content={"error": str(exc)}, headers={"X-Request-Id": request_id}
            )

    async def rerank(self, request: Request) -> Response:
        body = await request.json()
        query = body.get("query")
        documents = body.get("documents")
        top_n = body.get("top_n")
        if not isinstance(query, str) or not query.strip():
            raise HTTPException(status_code=400, detail="Query must not be empty")
        if not isinstance(documents, list) or not documents:
            raise HTTPException(status_code=400, detail="Documents list must not be empty")
        if top_n is not None:
            if not isinstance(top_n, int):
                raise HTTPException(status_code=400, detail="top_n must be an integer")
            if top_n > len(documents):
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"top_n ({top_n}) must not exceed the number of "
                        f"documents ({len(documents)})"
                    ),
                )
        for document in documents:
            if isinstance(document, str):
                continue
            if isinstance(document, Mapping) and isinstance(document.get("text"), str):
                continue
            raise HTTPException(
                status_code=400,
                detail=(
                    "Each document must be a string or an object with a 'text' field. "
                    f"Got: {type(document).__name__}"
                ),
            )
        raise HTTPException(
            status_code=404,
            detail=(
                "No reranker model loaded. Start the server with --rerank-model "
                "to enable the /v1/rerank endpoint."
            ),
        )

    async def mcp_tools(self) -> dict[str, object]:
        return {"tools": [], "count": 0}

    async def mcp_servers(self) -> dict[str, object]:
        return {"servers": []}

    async def mcp_execute(self, request: Request) -> Response:
        await request.json()
        raise HTTPException(
            status_code=503,
            detail="MCP not configured. Start server with --mcp-config",
        )

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
            if not hasattr(audio_file, "read") or not hasattr(audio_file, "filename"):
                return JSONResponse(
                    status_code=400,
                    content={"error": "Audio file must be uploaded as multipart file data"},
                )
            requested_model = form.get("model")
            validate_stt_model(
                str(requested_model) if requested_model is not None else None,
                container.settings.audio.asr,
            )

            upload_path = await save_upload_with_limit(
                audio_file,
                max_bytes=container.settings.audio.max_audio_upload_mb * 1024 * 1024,
            )
            try:
                audio_data = Path(upload_path).read_bytes()
            finally:
                Path(upload_path).unlink(missing_ok=True)
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
        except AsterError as exc:
            return JSONResponse(status_code=exc.status_code, content=exc.to_payload())
        except HTTPException:
            raise
        except Exception as e:
            self.logger.exception("transcribe_failed")
            return JSONResponse(
                status_code=500,
                content={"error": str(e)},
            )

    async def audio_voices(self, request: Request, model: str = "kokoro") -> dict[str, object]:
        del model
        container = request.app.state.container
        tts = container.audio.tts
        voices: list[str] = []
        list_voices = getattr(tts, "list_voices", None) if tts is not None else None
        if callable(list_voices):
            candidate = list_voices()
            if isinstance(candidate, list):
                voices = [str(voice) for voice in candidate]
        if not voices:
            voices = [container.settings.audio.tts.default_voice or "default"]
        return {"voices": voices}

    def _normalize_messages(self, messages: list[ChatMessage]) -> list[dict[str, str]]:
        return [
            {
                "role": self._normalize_role(message.role),
                "content": self._flatten_message_content(message),
            }
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
        if is_multimodal_content_type(part.type):
            raise AsterError(
                code="multimodal_not_supported",
                message=(
                    "Multimodal content must be handled by the media request model; "
                    "the local text runtime no longer flattens media into placeholders."
                ),
                status_code=400,
            )
        if part.type:
            return f"[{part.type}]"
        return ""

    def _include_stream_usage(self, stream_options: dict[str, object] | None) -> bool:
        return bool(stream_options and stream_options.get("include_usage") is True)

    def _result_finish_reason(self, result: object) -> str:
        finish_reason = getattr(result, "finish_reason", None)
        return finish_reason if isinstance(finish_reason, str) else "stop"

    def _chat_response_id(self) -> str:
        return f"chatcmpl-{uuid.uuid4().hex[:8]}"

    def _completion_response_id(self) -> str:
        return f"cmpl-{uuid.uuid4().hex[:8]}"

    def _completion_inference_request(
        self,
        body: CompletionRequest,
        *,
        prompt: str,
        request_id: str,
        stream: bool,
        max_tokens: int,
        request_aliases: tuple[str, ...] = (),
    ) -> InferenceRequest:
        return InferenceRequest(
            prompt=prompt,
            max_tokens=max_tokens,
            stream=stream,
            temperature=body.temperature,
            top_p=body.top_p,
            top_k=body.top_k,
            min_p=body.min_p,
            presence_penalty=body.presence_penalty,
            frequency_penalty=body.frequency_penalty,
            repetition_penalty=body.repetition_penalty,
            stop=body.stop,
            stop_token_ids=tuple(body.stop_token_ids or ()),
            trace_id=request_id,
            request_aliases=request_aliases,
            timeout_seconds=body.timeout,
        )

    async def openai_responses(self, request: Request) -> Response:
        body = await request.json()
        return await self._handle_provider_request(
            request, body, provider="openai", api_family="responses"
        )

    async def anthropic_messages(self, request: Request) -> Response:
        body = await request.json()
        return await self._handle_provider_request(
            request, body, provider="anthropic", api_family="messages"
        )

    async def anthropic_count_tokens(self, request: Request) -> Response:
        container = request.app.state.container
        request_id = request.headers.get("X-Request-Id") or str(uuid.uuid4())
        try:
            raw_body = await request.json()
            body = raw_body if isinstance(raw_body, Mapping) else {}
            texts = tuple(self._anthropic_count_token_texts(body))
            input_tokens = await container.inference_engine.count_text_tokens(texts)
            return JSONResponse(
                {"input_tokens": input_tokens},
                headers={"X-Request-Id": request_id},
            )
        except AsterError as exc:
            return JSONResponse(
                status_code=exc.status_code,
                content=exc.to_payload(),
                headers={"X-Request-Id": request_id},
            )

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
        return await self._handle_provider_request(
            request, body, provider="cohere", api_family="chat_v2"
        )

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
        return await self._handle_provider_request(
            request, body, provider="xai", api_family="responses"
        )

    async def xai_chat_completions(self, request: Request) -> Response:
        body = await request.json()
        return await self._handle_provider_request(
            request, body, provider="xai", api_family="chat_completions"
        )

    async def mistral_chat_completions(self, request: Request) -> Response:
        body = await request.json()
        return await self._handle_provider_request(
            request, body, provider="mistral", api_family="chat_completions"
        )

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
                previous_response_messages=self._previous_response_messages(
                    body,
                    provider=provider,
                    api_family=api_family,
                    metrics=container.metrics,
                ),
            )
            self._apply_provider_chat_defaults(
                parsed.inference_request,
                body,
                provider=provider,
                api_family=api_family,
                default_enable_thinking=container.settings.model.enable_thinking,
            )
            self._validate_request_max_tokens(
                max_tokens=parsed.inference_request.max_tokens,
                max_request_tokens=container.settings.api.max_request_tokens,
            )
            if parsed.stream:
                stream_timeout_seconds = self._effective_timeout_seconds(
                    parsed.inference_request.timeout_seconds,
                    container.settings.api.request_timeout_seconds,
                )
                if parsed.feature_plan.mode != "plain":
                    if supports_provider_live_tool_stream(parsed):
                        if parsed.api_family in {"chat_completions", "responses"}:
                            return await stream_live_tool_interaction(
                                container,
                                parsed,
                                raw_request=request,
                                timeout_seconds=stream_timeout_seconds,
                                on_responses_complete=lambda replay_messages: self._store_responses_replay_messages(
                                    parsed,
                                    replay_messages,
                                    metrics=container.metrics,
                                ),
                            )
                    if supports_provider_structured_stream(parsed):
                        return encode_provider_stream(
                            parsed,
                            self._store_responses_stream(
                                parsed,
                                self._responses_stream_with_timeout(
                                    parsed,
                                    container.inference_engine.stream(parsed.inference_request),
                                    timeout_seconds=stream_timeout_seconds,
                                ),
                                metrics=container.metrics,
                            ),
                            raw_request=request,
                        )
                    return await self._await_client_request(
                        request,
                        stream_interaction(
                            container,
                            parsed,
                            raw_request=request,
                        ),
                        timeout_seconds=stream_timeout_seconds,
                    )
                return encode_provider_stream(
                    parsed,
                    self._store_responses_stream(
                        parsed,
                        self._responses_stream_with_timeout(
                            parsed,
                            container.inference_engine.stream(parsed.inference_request),
                            timeout_seconds=stream_timeout_seconds,
                        ),
                        metrics=container.metrics,
                    ),
                    raw_request=request,
                )
            if parsed.feature_plan.mode == "tools":
                trace = await self._await_client_request(
                    request,
                    run_interaction(container, parsed),
                    timeout_seconds=self._effective_timeout_seconds(
                        parsed.inference_request.timeout_seconds,
                        container.settings.api.request_timeout_seconds,
                    ),
                )
                self._store_responses_replay_messages(
                    parsed,
                    responses_replay_messages_from_trace(parsed, trace),
                    metrics=container.metrics,
                )
                return encode_provider_decoded_response(
                    parsed, trace.final_result, trace.final_decoded
                )
            result = await self._await_client_request(
                request,
                container.inference_engine.submit(parsed.inference_request),
                timeout_seconds=self._effective_timeout_seconds(
                    parsed.inference_request.timeout_seconds,
                    container.settings.api.request_timeout_seconds,
                ),
            )
            decoded = decode_local_output(result.text, parsed.feature_plan)
            self._store_responses_history(parsed, decoded, metrics=container.metrics)
            return encode_provider_decoded_response(parsed, result, decoded)
        except AsterError as exc:
            self.logger.exception(
                "provider_request_failed",
                extra={"request_id": request_id, "provider": provider, "api_family": api_family},
            )
            if self._is_client_disconnected_error(exc):
                return self._client_closed_response(request_id)
            return provider_error_response(provider, api_family, exc, request_id=request_id)
        except Exception:
            self.logger.exception(
                "provider_request_failed_unhandled",
                extra={"request_id": request_id, "provider": provider, "api_family": api_family},
            )
            raise

    def _is_client_disconnected_error(self, exc: AsterError) -> bool:
        return exc.code == CLIENT_DISCONNECTED_CODE or exc.status_code == 499

    def _client_closed_response(self, request_id: str) -> Response:
        return Response(status_code=499, headers={"X-Request-Id": request_id})

    def _anthropic_count_token_texts(self, body: Mapping[str, object]) -> list[str]:
        texts: list[str] = []
        system = body.get("system", "")
        if isinstance(system, str):
            if system:
                texts.append(system)
        elif isinstance(system, list):
            for block in system:
                if isinstance(block, Mapping):
                    text = block.get("text")
                    if isinstance(text, str) and text:
                        texts.append(text)

        messages = body.get("messages", [])
        if isinstance(messages, list):
            for message in messages:
                if isinstance(message, Mapping):
                    self._append_anthropic_content_texts(texts, message.get("content"))

        tools = body.get("tools", [])
        if isinstance(tools, list):
            for tool in tools:
                if not isinstance(tool, Mapping):
                    continue
                name = tool.get("name")
                if isinstance(name, str) and name:
                    texts.append(name)
                description = tool.get("description")
                if isinstance(description, str) and description:
                    texts.append(description)
                input_schema = tool.get("input_schema")
                if input_schema:
                    texts.append(json.dumps(input_schema))
        return texts

    def _append_anthropic_content_texts(self, texts: list[str], content: object) -> None:
        if isinstance(content, str):
            if content:
                texts.append(content)
            return
        if not isinstance(content, list):
            return
        for block in content:
            if not isinstance(block, Mapping):
                continue
            text = block.get("text")
            if isinstance(text, str) and text:
                texts.append(text)
            tool_input = block.get("input")
            if tool_input:
                texts.append(json.dumps(tool_input))
            sub_content = block.get("content")
            if isinstance(sub_content, str):
                if sub_content:
                    texts.append(sub_content)
            elif isinstance(sub_content, list):
                for item in sub_content:
                    if not isinstance(item, Mapping):
                        continue
                    item_text = item.get("text")
                    if isinstance(item_text, str) and item_text:
                        texts.append(item_text)

    def _previous_response_messages(
        self,
        body: dict[str, object],
        *,
        provider: str,
        api_family: str,
        metrics: MetricsRegistry,
    ) -> list[dict[str, Any]] | None:
        if api_family != "responses" or provider not in {"openai", "xai"}:
            return None
        previous_response_id = body.get("previous_response_id")
        if previous_response_id in (None, ""):
            return None
        if not isinstance(previous_response_id, str):
            raise AsterError(
                code="invalid_previous_response_id",
                message="previous_response_id must be a string.",
                status_code=400,
            )
        messages = self._responses_store.get(
            previous_response_id,
            scope=self._responses_store_scope(provider=provider, api_family=api_family),
        )
        if messages is None:
            metrics.responses_store_misses.inc()
            metrics.responses_store_entries.set(len(self._responses_store))
            raise AsterError(
                code="response_not_found",
                message=f"Previous response `{previous_response_id}` not found.",
                status_code=404,
            )
        metrics.responses_store_hits.inc()
        metrics.responses_store_entries.set(len(self._responses_store))
        return messages

    def _store_responses_history(
        self,
        parsed: LocalProviderRequest,
        decoded: DecodedLocalOutput,
        *,
        metrics: MetricsRegistry,
    ) -> None:
        self._store_responses_replay_messages(
            parsed,
            [
                *parsed.response_replay_messages,
                *responses_replay_output_messages(decoded),
            ],
            metrics=metrics,
        )

    def _store_responses_replay_messages(
        self,
        parsed: LocalProviderRequest,
        replay_messages: list[dict[str, Any]],
        *,
        metrics: MetricsRegistry,
    ) -> None:
        if not self._should_store_response(parsed):
            return
        evictions = self._responses_store.put(
            parsed.response_id,
            replay_messages,
            scope=self._responses_store_scope(
                provider=parsed.provider,
                api_family=parsed.api_family,
            ),
        )
        metrics.responses_store_writes.inc()
        if evictions:
            metrics.responses_store_evictions.inc(evictions)
        metrics.responses_store_entries.set(len(self._responses_store))

    async def _store_responses_stream(
        self,
        parsed: LocalProviderRequest,
        chunks: AsyncIterator[DecodeChunk],
        *,
        metrics: MetricsRegistry,
    ) -> AsyncIterator[DecodeChunk]:
        if not self._should_store_response(parsed):
            async for chunk in chunks:
                yield chunk
            return

        fragments: list[str] = []
        async for chunk in chunks:
            if chunk.finished:
                try:
                    decoded = decode_local_output("".join(fragments), parsed.feature_plan)
                except AsterError:
                    pass
                else:
                    self._store_responses_history(parsed, decoded, metrics=metrics)
            else:
                fragments.append(chunk.token)
            yield chunk

    async def _responses_stream_with_timeout(
        self,
        parsed: LocalProviderRequest,
        chunks: AsyncIterator[DecodeChunk],
        *,
        timeout_seconds: float,
    ) -> AsyncIterator[DecodeChunk]:
        if parsed.api_family != "responses" or parsed.provider not in {"openai", "xai"}:
            async for chunk in chunks:
                yield chunk
            return

        started = perf_counter()
        aiter = chunks.__aiter__()
        try:
            while True:
                elapsed = perf_counter() - started
                remaining = timeout_seconds - elapsed
                if remaining <= 0:
                    raise AsterError(
                        code="request_timeout",
                        message="Inference request timed out",
                        status_code=504,
                        details={
                            "timeout_seconds": timeout_seconds,
                            "elapsed_seconds": round(elapsed, 4),
                        },
                    )
                try:
                    chunk = await asyncio.wait_for(aiter.__anext__(), timeout=remaining)
                except StopAsyncIteration:
                    return
                except TimeoutError:
                    elapsed = perf_counter() - started
                    raise AsterError(
                        code="request_timeout",
                        message="Inference request timed out",
                        status_code=504,
                        details={
                            "timeout_seconds": timeout_seconds,
                            "elapsed_seconds": round(elapsed, 4),
                        },
                    ) from None
                yield chunk
        finally:
            with suppress(Exception):
                await aiter.aclose()

    def _should_store_response(self, parsed: LocalProviderRequest) -> bool:
        return (
            parsed.api_family == "responses"
            and parsed.provider in {"openai", "xai"}
            and parsed.response_metadata.get("store") is not False
        )

    def _responses_store_scope(self, *, provider: str, api_family: str) -> str:
        return f"{provider}:{api_family}"

    def _resolve_chat_max_tokens(
        self,
        body: ChatCompletionRequest,
        *,
        max_request_tokens: int,
    ) -> int:
        requested = (
            body.max_completion_tokens
            if body.max_completion_tokens is not None
            else body.max_tokens
        )
        return self._resolve_request_max_tokens(
            max_tokens=requested,
            max_request_tokens=max_request_tokens,
        )

    async def _await_client_request(
        self,
        request: Request,
        awaitable: Awaitable[T],
        *,
        timeout_seconds: float,
    ) -> T:
        metrics = request.app.state.container.metrics
        return await await_with_disconnect(
            awaitable,
            request,
            timeout_seconds=timeout_seconds,
            on_error=lambda exc: metrics.errors.labels(code=exc.code).inc(),
        )

    def _effective_timeout_seconds(
        self, request_timeout: float | None, server_timeout: float
    ) -> float:
        if request_timeout is not None:
            return max(float(request_timeout), 1e-3)
        return max(float(server_timeout), 1e-3)

    def _apply_provider_chat_defaults(
        self,
        inference_request: InferenceRequest,
        body: dict[str, object],
        *,
        provider: str,
        api_family: str,
        default_enable_thinking: bool,
    ) -> None:
        if api_family != "chat_completions" or provider not in {"openai", "xai", "mistral"}:
            return
        if self._provider_request_has_explicit_thinking(body):
            return
        inference_request.enable_thinking = default_enable_thinking
        inference_request.chat_template_kwargs["enable_thinking"] = default_enable_thinking

    def _provider_request_has_explicit_thinking(self, body: dict[str, object]) -> bool:
        if isinstance(body.get("enable_thinking"), bool):
            return True
        template_kwargs = body.get("chat_template_kwargs")
        return isinstance(template_kwargs, dict) and isinstance(
            template_kwargs.get("enable_thinking"), bool
        )

    def _resolve_chat_template_kwargs(
        self,
        body: ChatCompletionRequest,
        *,
        default_enable_thinking: bool,
    ) -> tuple[bool, dict[str, object]]:
        template_kwargs = dict(body.chat_template_kwargs or {})
        if body.enable_thinking is not None:
            enable_thinking = body.enable_thinking
        else:
            requested_enable_thinking = template_kwargs.get("enable_thinking")
            enable_thinking = (
                requested_enable_thinking
                if isinstance(requested_enable_thinking, bool)
                else default_enable_thinking
            )
        template_kwargs["enable_thinking"] = enable_thinking
        return enable_thinking, template_kwargs

    def _resolve_request_max_tokens(
        self,
        *,
        max_tokens: int | None,
        max_request_tokens: int,
    ) -> int:
        resolved = DEFAULT_MAX_TOKENS if max_tokens is None else max_tokens
        self._validate_request_max_tokens(
            max_tokens=resolved,
            max_request_tokens=max_request_tokens,
        )
        return resolved

    def _validate_request_max_tokens(self, *, max_tokens: int, max_request_tokens: int) -> None:
        limit = max(int(max_request_tokens), 1)
        if max_tokens <= limit:
            return
        raise AsterError(
            code="max_tokens_exceeded",
            message=f"max_tokens exceeds server limit ({limit})",
            status_code=400,
            details={"max_tokens": max_tokens, "max_request_tokens": limit},
        )

    async def synthesize(self, request: Request, body: TTSRequest) -> Response:
        """Synthesize text to speech (TTS)."""
        container = request.app.state.container
        if not container.audio.tts:
            return JSONResponse(
                status_code=503,
                content={"error": "TTS service not available"},
            )

        try:
            validate_tts_model(body.model, container.settings.audio.tts)
            validate_tts_input_length(
                body.input,
                max_chars=container.settings.audio.max_tts_input_chars,
            )
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
        except AsterError as exc:
            return JSONResponse(status_code=exc.status_code, content=exc.to_payload())
        except HTTPException:
            raise
        except Exception as e:
            self.logger.exception("synthesize_failed")
            return JSONResponse(
                status_code=500,
                content={"error": str(e)},
            )


def build_router(
    *, responses_store_max_entries: int = DEFAULT_RESPONSES_STORE_MAX_SIZE
) -> APIRouter:
    return RouteBuilder(responses_store_max_entries=responses_store_max_entries).router
