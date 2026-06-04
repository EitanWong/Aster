from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NoReturn, Protocol

from aster.core.config import RuntimeSettings
from aster.core.errors import ConfigurationError
from aster.inference.contracts import InferenceRequest
from aster.inference.model_runner import (
    DecodeInit,
    DecodeStepResult,
    DecodeWorkItem,
    ModelRunner,
    PrefillChunkResult,
    PreparedPrompt,
)


@dataclass(frozen=True, slots=True)
class RuntimeKernelCapabilities:
    name: str
    continuous_batching: bool
    experimental: bool = False
    available: bool = True
    notes: tuple[str, ...] = ()


class RuntimeKernel(Protocol):
    @property
    def capabilities(self) -> RuntimeKernelCapabilities: ...

    def warmup(self) -> None: ...

    def encode_request(self, request: InferenceRequest) -> PreparedPrompt: ...

    def estimate_request_bytes(self, prompt_tokens: int, max_tokens: int) -> int: ...

    def model_fingerprint(self) -> str: ...

    def strict_chat_prefix_prompt(
        self,
        messages: list[dict[str, Any]],
        *,
        enable_thinking: bool,
        chat_template_kwargs: dict[str, Any] | None = None,
    ) -> str | None: ...

    def available_memory_bytes(self) -> int: ...

    def clone_cache(
        self,
        prompt_cache: Any | None,
        cache_token_count: int | None = None,
    ) -> Any | None: ...

    def prefill_to(
        self,
        *,
        prompt_tokens: list[int],
        prompt_cache: Any | None,
        cache_token_count: int,
        target_cache_token_count: int,
    ) -> PrefillChunkResult: ...

    def initialize_decode(
        self,
        *,
        prompt_tokens: list[int],
        cache_token_count: int,
        prompt_cache: Any | None,
        request: InferenceRequest,
    ) -> DecodeInit: ...

    def decode_batch_step(self, items: list[DecodeWorkItem]) -> list[DecodeStepResult]: ...

    def decode_diagnostics(self) -> dict[str, object]: ...

    def finalize_detokenizer(self, detokenizer: Any | None) -> str: ...

    def estimate_cache_bytes(self, prompt_cache: Any | None) -> int: ...

    def clear_runtime_caches(self) -> dict[str, object]: ...

    def count_text_tokens(self, texts: tuple[str, ...]) -> int: ...


class ManualRuntimeKernel:
    def __init__(self, runner: ModelRunner) -> None:
        self.runner = runner
        self._capabilities = RuntimeKernelCapabilities(
            name="manual",
            continuous_batching=False,
            notes=(
                "Uses Aster's explicit prefill queue and per-step merge/extract decode batching.",
            ),
        )

    @property
    def capabilities(self) -> RuntimeKernelCapabilities:
        return self._capabilities

    def warmup(self) -> None:
        return self.runner.warmup()

    def encode_request(self, request: InferenceRequest) -> PreparedPrompt:
        return self.runner.encode_request(request)

    def estimate_request_bytes(self, prompt_tokens: int, max_tokens: int) -> int:
        return self.runner.estimate_request_bytes(prompt_tokens, max_tokens)

    def model_fingerprint(self) -> str:
        return self.runner.model_fingerprint()

    def strict_chat_prefix_prompt(
        self,
        messages: list[dict[str, Any]],
        *,
        enable_thinking: bool,
        chat_template_kwargs: dict[str, Any] | None = None,
    ) -> str | None:
        return self.runner.strict_chat_prefix_prompt(
            messages,
            enable_thinking=enable_thinking,
            chat_template_kwargs=chat_template_kwargs,
        )

    def available_memory_bytes(self) -> int:
        return self.runner.available_memory_bytes()

    def clone_cache(
        self,
        prompt_cache: Any | None,
        cache_token_count: int | None = None,
    ) -> Any | None:
        return self.runner.clone_cache(prompt_cache, cache_token_count)

    def prefill_to(
        self,
        *,
        prompt_tokens: list[int],
        prompt_cache: Any | None,
        cache_token_count: int,
        target_cache_token_count: int,
    ) -> PrefillChunkResult:
        return self.runner.prefill_to(
            prompt_tokens=prompt_tokens,
            prompt_cache=prompt_cache,
            cache_token_count=cache_token_count,
            target_cache_token_count=target_cache_token_count,
        )

    def initialize_decode(
        self,
        *,
        prompt_tokens: list[int],
        cache_token_count: int,
        prompt_cache: Any | None,
        request: InferenceRequest,
    ) -> DecodeInit:
        return self.runner.initialize_decode(
            prompt_tokens=prompt_tokens,
            cache_token_count=cache_token_count,
            prompt_cache=prompt_cache,
            request=request,
        )

    def decode_batch_step(self, items: list[DecodeWorkItem]) -> list[DecodeStepResult]:
        return self.runner.decode_batch_step(items)

    def decode_diagnostics(self) -> dict[str, object]:
        diagnostics = getattr(self.runner, "decode_diagnostics", None)
        if callable(diagnostics):
            return dict(diagnostics())
        return {}

    def finalize_detokenizer(self, detokenizer: Any | None) -> str:
        return self.runner.finalize_detokenizer(detokenizer)

    def estimate_cache_bytes(self, prompt_cache: Any | None) -> int:
        return self.runner.estimate_cache_bytes(prompt_cache)

    def clear_runtime_caches(self) -> dict[str, object]:
        return self.runner.clear_runtime_caches()

    def count_text_tokens(self, texts: tuple[str, ...]) -> int:
        return self.runner.count_text_tokens(texts)


class BatchGeneratorRuntimeKernel:
    def __init__(self, settings: RuntimeSettings) -> None:
        self.settings = settings
        self._capabilities = RuntimeKernelCapabilities(
            name="batch_generator",
            continuous_batching=True,
            experimental=True,
            available=False,
            notes=(
                "Boundary-only adapter for mlx_lm.BatchGenerator.",
                "Use benchmark data before enabling it as a serving backend.",
            ),
        )

    @property
    def capabilities(self) -> RuntimeKernelCapabilities:
        return self._capabilities

    def _not_ready(self) -> NoReturn:
        raise ConfigurationError(
            code="batch_generator_kernel_unavailable",
            message=(
                "engine.runtime_kernel=batch_generator is defined as an adapter "
                "boundary, but the serving implementation is not enabled yet. "
                "Use engine.runtime_kernel=manual for production runs."
            ),
            status_code=501,
        )

    def warmup(self) -> None:
        self._not_ready()

    def encode_request(self, request: InferenceRequest) -> PreparedPrompt:
        self._not_ready()

    def estimate_request_bytes(self, prompt_tokens: int, max_tokens: int) -> int:
        self._not_ready()

    def model_fingerprint(self) -> str:
        self._not_ready()

    def strict_chat_prefix_prompt(
        self,
        messages: list[dict[str, Any]],
        *,
        enable_thinking: bool,
        chat_template_kwargs: dict[str, Any] | None = None,
    ) -> str | None:
        del messages, enable_thinking, chat_template_kwargs
        self._not_ready()

    def available_memory_bytes(self) -> int:
        return 0

    def clone_cache(
        self,
        prompt_cache: Any | None,
        cache_token_count: int | None = None,
    ) -> Any | None:
        self._not_ready()

    def prefill_to(
        self,
        *,
        prompt_tokens: list[int],
        prompt_cache: Any | None,
        cache_token_count: int,
        target_cache_token_count: int,
    ) -> PrefillChunkResult:
        self._not_ready()

    def initialize_decode(
        self,
        *,
        prompt_tokens: list[int],
        cache_token_count: int,
        prompt_cache: Any | None,
        request: InferenceRequest,
    ) -> DecodeInit:
        self._not_ready()

    def decode_batch_step(self, items: list[DecodeWorkItem]) -> list[DecodeStepResult]:
        self._not_ready()

    def decode_diagnostics(self) -> dict[str, object]:
        return {
            "batch_attempts": 0,
            "batch_successes": 0,
            "batch_fallbacks": 0,
            "batch_items": 0,
            "batch_fallback_items": 0,
            "single_steps": 0,
            "batch_fallback_rate": 0.0,
            "last_batch_fallback_error": None,
        }

    def finalize_detokenizer(self, detokenizer: Any | None) -> str:
        self._not_ready()

    def estimate_cache_bytes(self, prompt_cache: Any | None) -> int:
        self._not_ready()

    def clear_runtime_caches(self) -> dict[str, object]:
        return {
            "mlx_cache_cleared": False,
            "reason": "batch_generator_kernel_unavailable",
        }

    def count_text_tokens(self, texts: tuple[str, ...]) -> int:
        del texts
        self._not_ready()


def build_runtime_kernel(
    settings: RuntimeSettings,
    runner: ModelRunner | None = None,
) -> RuntimeKernel:
    if settings.engine.runtime_kernel == "manual":
        return ManualRuntimeKernel(runner or ModelRunner(settings))
    if settings.engine.runtime_kernel == "batch_generator":
        return BatchGeneratorRuntimeKernel(settings)
    raise ConfigurationError(
        code="invalid_runtime_kernel",
        message=f"Unsupported engine.runtime_kernel={settings.engine.runtime_kernel!r}",
        status_code=400,
    )
