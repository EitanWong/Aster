from __future__ import annotations

import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import httpx
import orjson

from aster.core.config import VLLMMLXSettings
from aster.core.errors import AsterError

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from aster.core.config import RuntimeSettings


@dataclass(slots=True)
class VLLMMLXResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    total_latency_s: float


@dataclass(slots=True)
class VLLMMLXStreamEvent:
    text: str
    finish_reason: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class VLLMMLXClient:
    def __init__(
        self,
        settings: VLLMMLXSettings,
        *,
        model_path: str = "",
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.settings = settings
        self._model_path = model_path  # from RuntimeSettings.model.path
        headers = {"Content-Type": "application/json"}
        if settings.api_key:
            headers["Authorization"] = f"Bearer {settings.api_key}"
        self._client = httpx.AsyncClient(
            base_url=settings.base_url.rstrip("/"),
            headers=headers,
            timeout=settings.timeout_seconds,
            transport=transport,
        )
        # Lazy-loaded tokenizer for thinking-mode prompt rendering.
        # Only initialised when reasoning_parser is configured.
        self._tokenizer: Any | None = None

    async def aclose(self) -> None:
        await self._client.aclose()

    async def infer(
        self,
        *,
        prompt: str | None,
        messages: list[dict[str, str]] | None,
        max_tokens: int,
        temperature: float,
        top_p: float,
        enable_thinking: bool = False,
    ) -> VLLMMLXResult:
        started = time.perf_counter()

        # When reasoning_parser is active we render the prompt ourselves so we
        # can pass enable_thinking to apply_chat_template, then use the raw
        # /v1/completions endpoint which does not re-apply the chat template.
        if self.settings.reasoning_parser and messages is not None:
            raw_prompt = self._render_prompt(messages, enable_thinking=enable_thinking)
            payload: dict[str, Any] = {
                "model": "default",
                "prompt": raw_prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "stream": False,
            }
            response = await self._client.post("/v1/completions", json=payload)
            self._raise_for_status(response)
            body = response.json()
            usage = body.get("usage", {})
            choice = self._first_choice(body)
            text = self._coerce_text(choice.get("text") if isinstance(choice, dict) else "")
            return VLLMMLXResult(
                text=text,
                prompt_tokens=self._int(usage.get("prompt_tokens")),
                completion_tokens=self._int(usage.get("completion_tokens")),
                total_latency_s=time.perf_counter() - started,
            )

        # Standard /v1/chat/completions path (reasoning_parser disabled, or
        # prompt-only requests that don't have a message list to template).
        payload = self._build_chat_payload(
            prompt=prompt,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=False,
        )
        response = await self._client.post("/v1/chat/completions", json=payload)
        self._raise_for_status(response)
        body = response.json()
        usage = body.get("usage", {})
        return VLLMMLXResult(
            text=self._extract_message_text(body),
            prompt_tokens=self._int(usage.get("prompt_tokens")),
            completion_tokens=self._int(usage.get("completion_tokens")),
            total_latency_s=time.perf_counter() - started,
        )

    async def stream(
        self,
        *,
        prompt: str | None,
        messages: list[dict[str, str]] | None,
        max_tokens: int,
        temperature: float,
        top_p: float,
        enable_thinking: bool = False,
    ) -> AsyncIterator[VLLMMLXStreamEvent]:
        if self.settings.reasoning_parser and messages is not None:
            raw_prompt = self._render_prompt(messages, enable_thinking=enable_thinking)
            payload: dict[str, Any] = {
                "model": "default",
                "prompt": raw_prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "stream": True,
                "stream_options": {"include_usage": True},
            }
            async with self._client.stream("POST", "/v1/completions", json=payload) as response:
                self._raise_for_status(response)
                async for line in response.aiter_lines():
                    event = self._parse_completion_stream_line(line)
                    if event is not None:
                        yield event
            return

        payload = self._build_chat_payload(
            prompt=prompt,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=True,
        )
        async with self._client.stream("POST", "/v1/chat/completions", json=payload) as response:
            self._raise_for_status(response)
            async for line in response.aiter_lines():
                event = self._parse_stream_line(line)
                if event is not None:
                    yield event

    async def embeddings(
        self,
        *,
        model: str,
        input_data: str | list[str],
    ) -> dict[str, Any]:
        response = await self._client.post(
            "/v1/embeddings",
            json={
                "model": model,
                "input": input_data,
            },
        )
        self._raise_for_status(response)
        body: dict[str, Any] = response.json()
        return body

    # ------------------------------------------------------------------
    # Prompt rendering (thinking-mode path)
    # ------------------------------------------------------------------

    def _render_prompt(self, messages: list[dict[str, str]], *, enable_thinking: bool) -> str:
        """Render messages to a raw string using the model's chat template.

        Passes enable_thinking directly to apply_chat_template so the template
        produces the correct <think> / </think> prefix, independent of
        vLLM-MLX's internal template handling.

        The model path comes from RuntimeSettings.model.path — the same path
        the vllm-mlx sidecar is serving, so no extra config field is needed.
        """
        tok = self._ensure_tokenizer()
        try:
            return str(tok.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            ))
        except TypeError:
            # Tokenizer doesn't support enable_thinking; fall back to default.
            return str(tok.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            ))

    def _ensure_tokenizer(self) -> Any:
        if self._tokenizer is None:
            model_path = self._model_path
            if not model_path:
                raise AsterError(
                    code="tokenizer_path_missing",
                    message=(
                        "model.path must be set when reasoning_parser is enabled "
                        "so Aster can apply the chat template for thinking-mode control."
                    ),
                    status_code=500,
                )
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(model_path)
        return self._tokenizer

    # ------------------------------------------------------------------
    # Payload builders (standard chat path)
    # ------------------------------------------------------------------

    def _build_chat_payload(
        self,
        *,
        prompt: str | None,
        messages: list[dict[str, str]] | None,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stream: bool,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": "default",
            "messages": messages if messages else [{"role": "user", "content": prompt or ""}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": stream,
        }
        if stream:
            payload["stream_options"] = {"include_usage": True}
        return payload

    # ------------------------------------------------------------------
    # Stream line parsers
    # ------------------------------------------------------------------

    def _parse_stream_line(self, line: str) -> VLLMMLXStreamEvent | None:
        """Parse a Server-Sent Events line from /v1/chat/completions."""
        if not line or not line.startswith("data:"):
            return None
        data = line[5:].strip()
        if not data or data == "[DONE]":
            return None
        payload = orjson.loads(data)
        usage = payload.get("usage")
        if not isinstance(usage, dict):
            usage = {}
        choice = self._first_choice(payload)
        delta = choice.get("delta", {}) if isinstance(choice, dict) else {}
        content = self._extract_delta_text(delta)
        finish_reason = choice.get("finish_reason") if isinstance(choice, dict) else None
        return VLLMMLXStreamEvent(
            text=content,
            finish_reason=finish_reason if isinstance(finish_reason, str) else None,
            prompt_tokens=self._int_or_none(usage.get("prompt_tokens")),
            completion_tokens=self._int_or_none(usage.get("completion_tokens")),
        )

    def _parse_completion_stream_line(self, line: str) -> VLLMMLXStreamEvent | None:
        """Parse a Server-Sent Events line from /v1/completions."""
        if not line or not line.startswith("data:"):
            return None
        data = line[5:].strip()
        if not data or data == "[DONE]":
            return None
        payload = orjson.loads(data)
        usage = payload.get("usage")
        if not isinstance(usage, dict):
            usage = {}
        choice = self._first_choice(payload)
        text = self._coerce_text(choice.get("text") if isinstance(choice, dict) else "")
        finish_reason = choice.get("finish_reason") if isinstance(choice, dict) else None
        return VLLMMLXStreamEvent(
            text=text,
            finish_reason=finish_reason if isinstance(finish_reason, str) else None,
            prompt_tokens=self._int_or_none(usage.get("prompt_tokens")),
            completion_tokens=self._int_or_none(usage.get("completion_tokens")),
        )

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_message_text(body: dict[str, Any]) -> str:
        choice = VLLMMLXClient._first_choice(body)
        message = choice.get("message", {}) if isinstance(choice, dict) else {}
        return VLLMMLXClient._coerce_text(message.get("content"))

    @staticmethod
    def _extract_delta_text(delta: dict[str, Any]) -> str:
        content = VLLMMLXClient._coerce_text(delta.get("content"))
        if content:
            return content
        reasoning_content = VLLMMLXClient._coerce_text(delta.get("reasoning_content"))
        if reasoning_content:
            return reasoning_content
        return VLLMMLXClient._coerce_text(delta.get("reasoning"))

    @staticmethod
    def _coerce_text(value: Any) -> str:
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            parts: list[str] = []
            for item in value:
                if isinstance(item, str):
                    parts.append(item)
                    continue
                if isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
            return "".join(parts)
        return ""

    @staticmethod
    def _first_choice(payload: dict[str, Any]) -> dict[str, Any]:
        choices = payload.get("choices", [])
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            return choices[0]
        return {}

    @staticmethod
    def _int(value: Any) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _int_or_none(value: Any) -> int | None:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _raise_for_status(response: httpx.Response) -> None:
        if not response.is_error:
            return

        detail = ""
        try:
            payload = response.json()
        except ValueError:
            payload = None

        if isinstance(payload, dict):
            error = payload.get("error")
            if isinstance(error, dict):
                detail = str(error.get("message") or error.get("detail") or "")
            elif isinstance(error, str):
                detail = error
            elif isinstance(payload.get("detail"), str):
                detail = payload["detail"]
        elif isinstance(payload, list):
            detail = str(payload)

        if not detail:
            detail = response.text.strip()

        raise AsterError(
            code="vllm_mlx_upstream_error",
            message=detail or f"vLLM-MLX upstream returned HTTP {response.status_code}",
            status_code=response.status_code,
        )
