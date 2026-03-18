from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Literal

from aster.cache.paged_kv_cache import PagedKVCache
from aster.cache.prefix_cache import PrefixCache
from aster.core.config import RuntimeSettings
from aster.inference.decode_engine import DecodeChunk
from aster.inference.mlx_runtime import MLXRuntime
from aster.inference.prefill_engine import PrefillEngine
from aster.inference.speculative import SpeculativeController
from aster.inference.speculative_pipeline import SpeculativePipeline
from aster.scheduler.policy_engine import PolicyEngine
from aster.telemetry.logging import get_logger
from aster.telemetry.metrics import MetricsRegistry


@dataclass(slots=True)
class InferenceRequest:
    prompt: str | None = None
    messages: list[dict[str, str]] | None = None
    max_tokens: int = 256
    stream: bool = False
    temperature: float = 0.7
    top_p: float = 0.95
    request_class: str = "default"
    trace_id: str | None = None


@dataclass(slots=True)
class InferenceResponse:
    request_id: str
    text: str
    prompt_tokens: int
    completion_tokens: int
    cache_hit: bool
    prefill_cache_hit: bool
    generation_cache_reuse: bool
    speculative_enabled: bool
    speculative_path_mode: Literal["disabled", "target_reuse", "full_prompt_no_cache"]
    prompt_tps: float
    generation_tps: float
    peak_memory_gb: float


class InferenceEngine:
    def __init__(
        self,
        settings: RuntimeSettings,
        metrics: MetricsRegistry,
        kv_cache: PagedKVCache,
        prefix_cache: PrefixCache,
        policy_engine: PolicyEngine,
    ) -> None:
        self.settings = settings
        self.metrics = metrics
        self.policy_engine = policy_engine
        self.kv_cache = kv_cache
        self.prefix_cache = prefix_cache
        self.prefill = PrefillEngine(metrics, kv_cache, prefix_cache)
        self.runtime = MLXRuntime(settings)
        self.speculative = SpeculativeController(settings.speculative)
        self.spec_pipeline = SpeculativePipeline(self.speculative)
        self._healthy = True
        self.logger = get_logger(__name__)

    def health(self) -> bool:
        return self._healthy

    async def infer(self, request: InferenceRequest) -> InferenceResponse:
        started = time.perf_counter()
        request_id = request.trace_id or str(uuid.uuid4())
        self.logger.info(
            "infer_start",
            extra={
                "request_id": request_id,
                "request_kind": "chat" if request.messages else "prompt",
                "max_tokens": request.max_tokens,
                "stream": request.stream,
            },
        )
        try:
            prompt_tokens = self._encode_request(request)
            self.logger.info("infer_encoded", extra={"request_id": request_id, "prompt_tokens": len(prompt_tokens)})
            prefill_plan = await self.prefill.run(request_id, self.settings.model.name, prompt_tokens)
            self.logger.info(
                "infer_prefill_ready",
                extra={
                    "request_id": request_id,
                    "prefill_cache_hit": prefill_plan.cache_hit,
                    "page_count": len(prefill_plan.page_ids),
                },
            )
            prefilled = self.runtime.prefill_prompt(prompt_tokens, prefill_plan.backend_cache)
            self._store_prefix_state(prompt_tokens, prefilled.prompt_cache, prefill_plan.page_ids)

            policy_requests_speculative = self.policy_engine.should_use_speculative(request.request_class)
            use_speculative = self.speculative.should_enable(
                request.request_class,
                policy_enabled=policy_requests_speculative,
            )

            stream_prompt_tokens = prompt_tokens[prefilled.matched_prefix_tokens :]
            stream_prompt_cache = prefilled.prompt_cache
            prefill_cache_hit = prefill_plan.cache_hit
            generation_cache_reuse = prefilled.prompt_cache is not None and prefilled.matched_prefix_tokens > 0
            speculative_path_mode: Literal["disabled", "target_reuse", "full_prompt_no_cache"] = "disabled"

            if use_speculative:
                stream_prompt_tokens = prompt_tokens
                stream_prompt_cache = None
                generation_cache_reuse = False
                speculative_path_mode = "full_prompt_no_cache"

            self.logger.info(
                "infer_generation_start",
                extra={
                    "request_id": request_id,
                    "prefill_cache_hit": prefill_cache_hit,
                    "generation_cache_reuse": generation_cache_reuse,
                    "speculative_enabled": use_speculative,
                    "speculative_path_mode": speculative_path_mode,
                    "stream_prompt_tokens": len(stream_prompt_tokens),
                },
            )

            completion: list[str] = []
            generated = 0
            last_tps = 0.0
            peak_memory = 0.0
            first_token_started = time.perf_counter()
            for chunk in self.runtime.stream_tokens(
                stream_prompt_tokens,
                stream_prompt_cache,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                use_speculative=use_speculative,
                num_draft_tokens=self.policy_engine.current().speculative_draft_tokens,
            ):
                if generated == 0:
                    first_token_latency = time.perf_counter() - first_token_started
                    self.metrics.first_token_latency.observe(first_token_latency)
                    self.logger.info(
                        "infer_first_token",
                        extra={"request_id": request_id, "first_token_latency_s": round(first_token_latency, 4)},
                    )
                if chunk.finish_reason is not None:
                    self.logger.info(
                        "infer_finish_reason",
                        extra={"request_id": request_id, "finish_reason": chunk.finish_reason, "generated": generated},
                    )
                    break
                if chunk.text:
                    completion.append(chunk.text)
                generated += 1
                last_tps = chunk.generation_tps
                peak_memory = max(peak_memory, chunk.peak_memory)
                if use_speculative:
                    self.metrics.spec_acceptance.observe(1.0 if chunk.from_draft else 0.0)

            total_latency = time.perf_counter() - started
            self.metrics.request_latency.observe(total_latency)
            self.logger.info(
                "infer_finish",
                extra={
                    "request_id": request_id,
                    "completion_tokens": generated,
                    "total_latency_s": round(total_latency, 4),
                    "generation_tps": round(last_tps, 4),
                    "peak_memory_gb": round(peak_memory, 4),
                },
            )
            return InferenceResponse(
                request_id=request_id,
                text="".join(completion).strip(),
                prompt_tokens=len(prompt_tokens),
                completion_tokens=generated,
                cache_hit=prefill_cache_hit,
                prefill_cache_hit=prefill_cache_hit,
                generation_cache_reuse=generation_cache_reuse,
                speculative_enabled=use_speculative,
                speculative_path_mode=speculative_path_mode,
                prompt_tps=(len(prompt_tokens) / prefilled.prefill_seconds) if prefilled.prefill_seconds > 0 else 0.0,
                generation_tps=last_tps,
                peak_memory_gb=peak_memory,
            )
        except Exception:
            self.logger.exception("infer_failed", extra={"request_id": request_id})
            raise
        finally:
            self.kv_cache.release(request_id)

    async def stream(self, request: InferenceRequest) -> AsyncIterator[DecodeChunk]:
        request_id = request.trace_id or str(uuid.uuid4())
        self.logger.info(
            "stream_start",
            extra={
                "request_id": request_id,
                "request_kind": "chat" if request.messages else "prompt",
                "max_tokens": request.max_tokens,
            },
        )
        try:
            prompt_tokens = self._encode_request(request)
            self.logger.info("stream_encoded", extra={"request_id": request_id, "prompt_tokens": len(prompt_tokens)})
            prefill_plan = await self.prefill.run(request_id, self.settings.model.name, prompt_tokens)
            prefilled = self.runtime.prefill_prompt(prompt_tokens, prefill_plan.backend_cache)
            self._store_prefix_state(prompt_tokens, prefilled.prompt_cache, prefill_plan.page_ids)

            policy_requests_speculative = self.policy_engine.should_use_speculative(request.request_class)
            use_speculative = self.speculative.should_enable(
                request.request_class,
                policy_enabled=policy_requests_speculative,
            )

            stream_prompt_tokens = prompt_tokens[prefilled.matched_prefix_tokens :]
            stream_prompt_cache = prefilled.prompt_cache
            speculative_path_mode: Literal["disabled", "target_reuse", "full_prompt_no_cache"] = "disabled"
            generation_cache_reuse = prefilled.prompt_cache is not None and prefilled.matched_prefix_tokens > 0
            if use_speculative:
                stream_prompt_tokens = prompt_tokens
                stream_prompt_cache = None
                speculative_path_mode = "full_prompt_no_cache"
                generation_cache_reuse = False

            self.logger.info(
                "stream_generation_start",
                extra={
                    "request_id": request_id,
                    "prefill_cache_hit": prefill_plan.cache_hit,
                    "speculative_enabled": use_speculative,
                    "stream_prompt_tokens": len(stream_prompt_tokens),
                },
            )
            index = 0
            generated = 0
            last_tps = 0.0
            peak_memory = 0.0
            for response in self.runtime.stream_tokens(
                stream_prompt_tokens,
                stream_prompt_cache,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                use_speculative=use_speculative,
                num_draft_tokens=self.policy_engine.current().speculative_draft_tokens,
            ):
                if response.finish_reason is not None:
                    self.logger.info(
                        "stream_finish_reason",
                        extra={"request_id": request_id, "finish_reason": response.finish_reason, "chunks": index},
                    )
                    break
                if response.text:
                    if index == 0:
                        self.logger.info("stream_first_chunk", extra={"request_id": request_id})
                    yield DecodeChunk(token=response.text, index=index, finished=False)
                    index += 1
                generated = response.generation_tokens
                last_tps = response.generation_tps
                peak_memory = max(peak_memory, response.peak_memory)
            yield DecodeChunk(
                token="",
                index=index,
                finished=True,
                stats={
                    "request_id": request_id,
                    "prompt_tokens": len(prompt_tokens),
                    "completion_tokens": generated,
                    "cache_hit": prefill_plan.cache_hit,
                    "prefill_cache_hit": prefill_plan.cache_hit,
                    "generation_cache_reuse": generation_cache_reuse,
                    "speculative_enabled": use_speculative,
                    "speculative_path_mode": speculative_path_mode,
                    "prompt_tps": (len(prompt_tokens) / prefilled.prefill_seconds) if prefilled.prefill_seconds > 0 else 0.0,
                    "generation_tps": last_tps,
                    "peak_memory_gb": peak_memory,
                },
            )
            self.logger.info(
                "stream_finish",
                extra={"request_id": request_id, "chunks": index, "completion_tokens": generated},
            )
        except Exception:
            self.logger.exception("stream_failed", extra={"request_id": request_id})
            raise
        finally:
            self.kv_cache.release(request_id)

    def _encode_request(self, request: InferenceRequest) -> list[int]:
        if request.messages:
            return self.runtime.encode_chat(request.messages, enable_thinking=False)
        if request.prompt is None:
            raise ValueError("InferenceRequest requires either prompt or messages")
        return self.runtime.encode(request.prompt)

    def _store_prefix_state(self, prompt_tokens: list[int], prompt_cache: object, page_ids: list[int]) -> None:
        approx_bytes = len(page_ids) * 4096
        self.prefix_cache.store(
            self.settings.model.name,
            prompt_tokens,
            page_ids,
            approx_bytes=approx_bytes,
            backend_cache=prompt_cache,
        )
        reusable_prefix = self._reusable_prefix_tokens(prompt_tokens)
        if reusable_prefix > 0:
            self.prefix_cache.maybe_store_prefix_slice(
                self.settings.model.name,
                prompt_tokens,
                prefix_tokens=reusable_prefix,
                page_ids=page_ids,
                approx_bytes=max(4096, int(approx_bytes * (reusable_prefix / max(len(prompt_tokens), 1)))),
                backend_cache=prompt_cache,
            )

    def _reusable_prefix_tokens(self, prompt_tokens: list[int]) -> int:
        if len(prompt_tokens) < 64:
            return 0
        if len(prompt_tokens) < 256:
            return max(32, len(prompt_tokens) - 16)
        return max(128, int(len(prompt_tokens) * 0.8))
