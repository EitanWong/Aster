from __future__ import annotations

import copy
import hashlib
import json
import math
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

from aster.core.config import RuntimeSettings
from aster.core.errors import ConfigurationError
from aster.inference.constrained import (
    ThinkingAwareJsonLogitsProcessor,
    build_json_logits_processor,
)
from aster.inference.contracts import InferenceRequest
from aster.inference.mlx_cache_utils import prompt_cache_length
from aster.inference.prompt_warmup import build_strict_prefix_string
from aster.inference.thinking_processor import ThinkingAwareLogitsProcessor

_DECODE_CACHE_CLEAR_TOKEN_BUDGET = 512


@dataclass(slots=True)
class PreparedPrompt:
    prompt_tokens: list[int]
    reuse_points: tuple[int, ...] = ()


@dataclass(slots=True)
class PrefillChunkResult:
    prompt_cache: Any
    cache_token_count: int
    elapsed_seconds: float
    peak_memory_gb: float = 0.0
    active_memory_gb: float = 0.0


@dataclass(slots=True)
class DecodeInit:
    prompt_cache: Any
    next_input_token: int
    sampler: Any
    detokenizer: Any
    stop_token_ids: frozenset[int]
    logits_processors: tuple[Any, ...] = ()
    logits_processor_context_size: int | None = None


@dataclass(slots=True)
class DecodeWorkItem:
    prompt_cache: Any
    input_token: int
    sampler: Any
    detokenizer: Any
    stop_token_ids: frozenset[int]
    logits_processors: tuple[Any, ...]
    logits_processor_tokens: list[int]
    completion_tokens: int
    max_tokens: int
    request_id: str | None = None
    logits_processor_context_size: int | None = None


@dataclass(slots=True)
class _DecodeBatchCacheState:
    request_ids: tuple[str, ...]
    merged_cache: list[Any]


@dataclass(frozen=True, slots=True)
class _DecodeCacheRef:
    state: _DecodeBatchCacheState
    index: int


@dataclass(slots=True)
class DecodeResult:
    prompt_cache: Any
    token_id: int | None
    text: str
    completion_tokens: int
    peak_memory_gb: float
    finish_reason: str | None = None


DecodeStepResult = DecodeResult | BaseException


class _BatchPostSampleError(RuntimeError):
    def __init__(self, cause: Exception) -> None:
        super().__init__(str(cause))
        self.cause = cause


@dataclass(frozen=True, slots=True)
class PrefillTransientProfile:
    n_q_heads: int
    head_dim: int
    score_dtype_size: int

    def estimate(self, query_tokens: int, kv_tokens: int) -> int:
        scores = self.n_q_heads * query_tokens * kv_tokens * self.score_dtype_size
        output = self.n_q_heads * query_tokens * self.head_dim * 4
        return int(scores + output)


def _array_memory(arr: Any) -> int:
    shape = getattr(arr, "shape", None)
    dtype = getattr(arr, "dtype", None)
    if shape is not None and dtype is not None and hasattr(dtype, "size"):
        return math.prod(shape) * int(dtype.size)
    nbytes = getattr(arr, "nbytes", None)
    if isinstance(nbytes, int):
        return nbytes
    return 0


class ModelRunner:
    _BUILTIN_PENALTY_CONTEXT_SIZE = 20

    def __init__(self, settings: RuntimeSettings) -> None:
        self.settings = settings
        self._loaded = False
        self._mx: Any | None = None
        self._make_prompt_cache: Any | None = None
        self._prompt_cache_factory: Any | None = None
        self._kv_cache_type: Any | None = None
        self._make_sampler: Any | None = None
        self._make_logits_processors: Any | None = None
        self._model: Any | None = None
        self._tokenizer: Any | None = None
        self._config: dict[str, Any] = {}
        self._model_fingerprint: str | None = None
        self._paged_cache: Any | None = None
        self._chat_prompt_cache_max_entries = settings.engine.chat_prompt_cache_max_entries
        self._chat_prompt_cache: OrderedDict[str, PreparedPrompt] = OrderedDict()
        self._decode_batch_attempts = 0
        self._decode_batch_successes = 0
        self._decode_batch_fallbacks = 0
        self._decode_batch_items = 0
        self._decode_batch_fallback_items = 0
        self._decode_batch_post_sample_failures = 0
        self._decode_single_steps = 0
        self._last_decode_batch_fallback_error: str | None = None
        self._decode_batch_cache_state: _DecodeBatchCacheState | None = None
        self._decode_batch_cache_reuses = 0
        self._decode_batch_cache_rebuilds = 0
        self._decode_tokens_since_cache_clear = 0
        self._decode_cache_clear_attempts = 0
        self._decode_cache_clears = 0
        self._decode_cache_clear_failures = 0

    def warmup(self) -> None:
        self._ensure_loaded()

    def encode_request(self, request: InferenceRequest) -> PreparedPrompt:
        self._ensure_loaded()
        if request.messages:
            cache_key = self._chat_prompt_cache_key(request)
            if cache_key is not None:
                cached = self._chat_prompt_cache.pop(cache_key, None)
                if cached is not None:
                    self._chat_prompt_cache[cache_key] = cached
                    return PreparedPrompt(
                        prompt_tokens=list(cached.prompt_tokens),
                        reuse_points=cached.reuse_points,
                    )
            prompt_tokens = self._encode_chat(
                request.messages,
                enable_thinking=request.enable_thinking,
                chat_template_kwargs=request.chat_template_kwargs,
            )
            reuse_points = ()
            if self.settings.engine.prefix_cache_enabled:
                reuse_points = self._chat_reuse_points(
                    request.messages,
                    full_prompt_tokens=prompt_tokens,
                    enable_thinking=request.enable_thinking,
                    chat_template_kwargs=request.chat_template_kwargs,
                )
            prepared = PreparedPrompt(prompt_tokens=prompt_tokens, reuse_points=reuse_points)
            if cache_key is not None:
                self._chat_prompt_cache[cache_key] = PreparedPrompt(
                    prompt_tokens=list(prompt_tokens),
                    reuse_points=reuse_points,
                )
                while len(self._chat_prompt_cache) > self._chat_prompt_cache_max_entries:
                    self._chat_prompt_cache.popitem(last=False)
            return prepared
        if request.prompt is None:
            raise ConfigurationError(
                code="missing_prompt",
                message="InferenceRequest requires either prompt or messages",
                status_code=400,
            )
        return PreparedPrompt(prompt_tokens=self._encode_text(request.prompt))

    def _chat_prompt_cache_key(self, request: InferenceRequest) -> str | None:
        if self._chat_prompt_cache_max_entries <= 0 or not request.messages:
            return None
        try:
            payload = {
                "messages": request.messages,
                "enable_thinking": request.enable_thinking,
                "chat_template_kwargs": request.chat_template_kwargs,
            }
            encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        except (TypeError, ValueError):
            return None
        return hashlib.sha256(encoded.encode()).hexdigest()

    def clone_cache(
        self,
        prompt_cache: Any | None,
        cache_token_count: int | None = None,
    ) -> Any | None:
        if prompt_cache is None:
            return None
        cloned = copy.deepcopy(prompt_cache)
        if cache_token_count is not None:
            cloned = self._trim_cache_to_token_count(cloned, cache_token_count)
        return cloned

    def prefill_to(
        self,
        *,
        prompt_tokens: list[int],
        prompt_cache: Any | None,
        cache_token_count: int,
        target_cache_token_count: int,
    ) -> PrefillChunkResult:
        self._ensure_loaded()
        if target_cache_token_count <= cache_token_count:
            return PrefillChunkResult(
                prompt_cache=prompt_cache,
                cache_token_count=cache_token_count,
                elapsed_seconds=0.0,
            )

        model = self._model
        mx = self._mx
        make_prompt_cache = self._make_prompt_cache
        assert model is not None and mx is not None and make_prompt_cache is not None

        live_cache = prompt_cache
        if live_cache is None:
            live_cache = make_prompt_cache(model)

        tokens_to_process = prompt_tokens[cache_token_count:target_cache_token_count]
        if not tokens_to_process:
            return PrefillChunkResult(
                prompt_cache=live_cache,
                cache_token_count=cache_token_count,
                elapsed_seconds=0.0,
            )

        reset_peak_memory = getattr(mx, "reset_peak_memory", None)
        if callable(reset_peak_memory):
            reset_peak_memory()
        started = time.perf_counter()
        model(mx.array(tokens_to_process)[None], cache=live_cache)
        self._eval_cache(live_cache)
        return PrefillChunkResult(
            prompt_cache=live_cache,
            cache_token_count=target_cache_token_count,
            elapsed_seconds=time.perf_counter() - started,
            peak_memory_gb=self.current_peak_memory_gb(),
            active_memory_gb=self.current_active_memory_gb(),
        )

    def initialize_decode(
        self,
        *,
        prompt_tokens: list[int],
        cache_token_count: int,
        prompt_cache: Any | None,
        request: InferenceRequest,
    ) -> DecodeInit:
        self._ensure_loaded()
        make_sampler = self._make_sampler
        make_logits_processors = self._make_logits_processors
        tokenizer = self._tokenizer
        make_prompt_cache = self._make_prompt_cache
        model = self._model
        assert tokenizer is not None and make_sampler is not None
        assert make_logits_processors is not None
        assert make_prompt_cache is not None and model is not None

        decode_prompt = prompt_tokens[cache_token_count:] or prompt_tokens[-1:]
        sampler = make_sampler(
            temp=request.temperature,
            top_p=request.top_p,
            min_p=request.min_p,
            top_k=request.top_k,
        )
        logits_processors: list[Any] = []
        json_processor: Any | None = None
        if request.structured_output_schema is not None:
            json_processor = build_json_logits_processor(
                request.structured_output_schema, tokenizer
            )
        thinking_processor = self._thinking_budget_processor(
            request,
            tokenizer=tokenizer,
            inner=json_processor,
        )
        if thinking_processor is not None:
            logits_processors.append(thinking_processor)
        elif json_processor is not None:
            if request.enable_thinking:
                json_processor = ThinkingAwareJsonLogitsProcessor(
                    json_processor,
                    tokenizer=tokenizer,
                    prompt_has_think_tag=True,
                )
            logits_processors.append(json_processor)
        penalty_processors = list(
            make_logits_processors(
                repetition_penalty=(
                    None
                    if request.repetition_penalty == 1.0
                    else request.repetition_penalty
                ),
                repetition_context_size=self._BUILTIN_PENALTY_CONTEXT_SIZE,
                presence_penalty=request.presence_penalty,
                presence_context_size=self._BUILTIN_PENALTY_CONTEXT_SIZE,
                frequency_penalty=request.frequency_penalty,
                frequency_context_size=self._BUILTIN_PENALTY_CONTEXT_SIZE,
            )
        )
        logits_processors.extend(penalty_processors)
        if thinking_processor is not None or json_processor is not None:
            logits_processor_context_size = None
        elif penalty_processors:
            logits_processor_context_size = self._BUILTIN_PENALTY_CONTEXT_SIZE
        else:
            logits_processor_context_size = 0
        detokenizer = tokenizer.detokenizer
        eos_ids = set(getattr(tokenizer, "eos_token_ids", []) or [])
        eos_id = getattr(tokenizer, "eos_token_id", None)
        if eos_id is not None:
            eos_ids.add(int(eos_id))
        eos_ids.update(int(token_id) for token_id in request.stop_token_ids)
        eos_ids.update(self._single_token_stop_ids(request.parser_stop_sequences, tokenizer))
        live_cache = prompt_cache or make_prompt_cache(model)
        for cache in live_cache:
            prepare_direct = getattr(cache, "prepare_direct_attention", None)
            if callable(prepare_direct):
                prepare_direct()
        return DecodeInit(
            prompt_cache=live_cache,
            next_input_token=int(decode_prompt[-1]),
            sampler=sampler,
            detokenizer=detokenizer,
            stop_token_ids=frozenset(int(token_id) for token_id in eos_ids),
            logits_processors=tuple(logits_processors),
            logits_processor_context_size=logits_processor_context_size,
        )

    def decode_batch_step(self, items: list[DecodeWorkItem]) -> list[DecodeStepResult]:
        if not items:
            return []
        if len(items) == 1:
            self._decode_batch_cache_state = None
            self._decode_single_steps += 1
            results = [self._decode_single(items[0])]
            self._maybe_clear_decode_cache(1 if results[0].token_id is not None else 0)
            return results
        self._decode_batch_attempts += 1
        self._decode_batch_items += len(items)
        try:
            results = self._decode_batch(items)
            self._decode_batch_successes += 1
        except MemoryError:
            raise
        except _BatchPostSampleError as exc:
            self._decode_batch_cache_state = None
            self._decode_batch_post_sample_failures += 1
            self._last_decode_batch_fallback_error = (
                f"{exc.cause.__class__.__name__}: {exc.cause}"
            )
            results = [exc.cause for _ in items]
        except Exception as exc:
            self._decode_batch_cache_state = None
            self._decode_batch_fallbacks += 1
            self._decode_batch_fallback_items += len(items)
            self._last_decode_batch_fallback_error = f"{exc.__class__.__name__}: {exc}"
            results = self._decode_items_individually(items)
        self._maybe_clear_decode_cache(
            sum(
                isinstance(result, DecodeResult) and result.token_id is not None
                for result in results
            )
        )
        return results

    def _decode_items_individually(self, items: list[DecodeWorkItem]) -> list[DecodeStepResult]:
        results: list[DecodeStepResult] = []
        for item in items:
            try:
                results.append(self._decode_single(item))
            except MemoryError:
                raise
            except Exception as exc:
                results.append(exc)
        return results

    def finalize_detokenizer(self, detokenizer: Any | None) -> str:
        if detokenizer is None:
            return ""
        try:
            detokenizer.finalize()
            return str(getattr(detokenizer, "last_segment", "") or "")
        except Exception:
            return ""

    def estimate_cache_bytes(self, prompt_cache: Any | None) -> int:
        if prompt_cache is None:
            return 0
        total = 0
        for layer in prompt_cache:
            if getattr(layer, "direct_attention_enabled", False):
                total += int(getattr(layer, "nbytes", 0))
                continue
            if isinstance(layer, dict) and "state" in layer:
                keys, values = layer["state"]
                total += _array_memory(keys)
                total += _array_memory(values)
                continue
            if hasattr(layer, "state"):
                try:
                    keys, values = layer.state
                except (TypeError, ValueError):
                    continue
                total += _array_memory(keys)
                total += _array_memory(values)
                continue
            keys = getattr(layer, "keys", None)
            values = getattr(layer, "values", None)
            if keys is not None and not callable(keys):
                total += _array_memory(keys)
            if values is not None and not callable(values):
                total += _array_memory(values)
        return total

    def clear_runtime_caches(self) -> dict[str, object]:
        self._chat_prompt_cache.clear()
        if self._mx is None:
            return {"mlx_cache_cleared": False, "reason": "model_not_loaded"}
        try:
            self._mx.clear_cache()
        except Exception as exc:
            return {
                "mlx_cache_cleared": False,
                "error": str(exc),
            }
        self._decode_tokens_since_cache_clear = 0
        return {"mlx_cache_cleared": True}

    def count_text_tokens(self, texts: tuple[str, ...]) -> int:
        self._ensure_loaded()
        tokenizer = self._tokenizer
        assert tokenizer is not None
        total = 0
        for text in texts:
            if not text:
                continue
            try:
                token_ids = tokenizer.encode(text, add_special_tokens=False)
            except TypeError:
                token_ids = tokenizer.encode(text)
            total += len(token_ids)
        return total

    def decode_diagnostics(self) -> dict[str, object]:
        fallback_rate = (
            self._decode_batch_fallbacks / self._decode_batch_attempts
            if self._decode_batch_attempts
            else 0.0
        )
        return {
            "batch_attempts": self._decode_batch_attempts,
            "batch_successes": self._decode_batch_successes,
            "batch_fallbacks": self._decode_batch_fallbacks,
            "batch_items": self._decode_batch_items,
            "batch_fallback_items": self._decode_batch_fallback_items,
            "batch_post_sample_failures": self._decode_batch_post_sample_failures,
            "single_steps": self._decode_single_steps,
            "batch_fallback_rate": round(fallback_rate, 6),
            "last_batch_fallback_error": self._last_decode_batch_fallback_error,
            "batch_cache_reuses": self._decode_batch_cache_reuses,
            "batch_cache_rebuilds": self._decode_batch_cache_rebuilds,
            "cache_clear_token_budget": _DECODE_CACHE_CLEAR_TOKEN_BUDGET,
            "cache_clear_attempts": self._decode_cache_clear_attempts,
            "cache_clears": self._decode_cache_clears,
            "cache_clear_failures": self._decode_cache_clear_failures,
        }

    def estimate_request_bytes(self, prompt_tokens: int, max_tokens: int) -> int:
        config = self._config
        text_config = config.get("text_config")
        if isinstance(text_config, dict):
            config = {**config, **text_config}

        layers = int(config.get("num_hidden_layers", config.get("n_layers", 32)))
        hidden_size = int(config.get("hidden_size", config.get("dim", 4096)))
        attn_heads = int(config.get("num_attention_heads", config.get("n_heads", 32)))
        kv_heads = int(config.get("num_key_value_heads", config.get("n_kv_heads", attn_heads)))
        head_dim = int(config.get("head_dim", hidden_size // max(attn_heads, 1)))
        dtype = str(config.get("dtype", "float16")).lower()
        bytes_per_scalar = 4 if "32" in dtype else 2

        layer_types = config.get("layer_types")
        if isinstance(layer_types, list) and layer_types:
            full_attention_layers = sum(layer == "full_attention" for layer in layer_types)
            linear_attention_layers = max(len(layer_types) - full_attention_layers, 0)
        else:
            full_attention_layers = layers
            linear_attention_layers = 0

        sequence_tokens = max(prompt_tokens + max_tokens, 1)
        full_attention_per_token = (
            full_attention_layers
            * kv_heads
            * head_dim
            * 2
            * bytes_per_scalar
        )
        if linear_attention_layers == 0:
            return full_attention_per_token * sequence_tokens

        linear_state_bytes = (
            linear_attention_layers
            * int(config.get("linear_num_value_heads", 0))
            * int(config.get("linear_value_head_dim", 0))
            * int(config.get("linear_key_head_dim", 0))
            * 4
        )
        linear_conv_dim = (
            2
            * int(config.get("linear_num_key_heads", 0))
            * int(config.get("linear_key_head_dim", 0))
            + int(config.get("linear_num_value_heads", 0))
            * int(config.get("linear_value_head_dim", 0))
        )
        linear_conv_bytes = (
            linear_attention_layers
            * max(int(config.get("linear_conv_kernel_dim", 1)) - 1, 0)
            * linear_conv_dim
            * bytes_per_scalar
        )
        return full_attention_per_token * sequence_tokens + linear_state_bytes + linear_conv_bytes

    def estimate_prefill_transient_bytes(self, query_tokens: int, kv_tokens: int) -> int:
        """Estimate transient score/output tensors for one full-attention call."""
        profile = self.prefill_transient_profile()
        if profile is None or query_tokens <= 0 or kv_tokens <= 0:
            return 0
        return profile.estimate(query_tokens, kv_tokens)

    def prefill_transient_profile(self) -> PrefillTransientProfile | None:
        """Return immutable full-attention dimensions for scheduler-side pricing."""

        config = self._config
        text_config = config.get("text_config")
        if isinstance(text_config, dict):
            config = {**config, **text_config}

        layer_types = config.get("layer_types")
        if isinstance(layer_types, list) and layer_types and not any(
            layer == "full_attention" for layer in layer_types
        ):
            return None

        n_q_heads = int(config.get("num_attention_heads", config.get("n_heads", 0)))
        head_dim = int(config.get("head_dim", 0))
        if n_q_heads <= 0 or head_dim <= 0:
            return None

        dtype = str(config.get("dtype", "float16")).lower()
        score_dtype_size = 4 if "32" in dtype else 2
        return PrefillTransientProfile(n_q_heads, head_dim, score_dtype_size)

    def current_peak_memory_gb(self) -> float:
        mx = self._mx
        if mx is None or not hasattr(mx, "get_peak_memory"):
            return 0.0
        try:
            return float(mx.get_peak_memory()) / 1e9
        except Exception:
            return 0.0

    def current_active_memory_gb(self) -> float:
        mx = self._mx
        if mx is None or not hasattr(mx, "get_active_memory"):
            return 0.0
        try:
            return float(mx.get_active_memory()) / 1e9
        except Exception:
            return 0.0

    @staticmethod
    def _configure_prompt_cache_step(
        prompt_cache: list[Any], cache_type: type[Any], step_tokens: int
    ) -> list[Any]:
        for cache in prompt_cache:
            if isinstance(cache, cache_type):
                cache.step = max(int(step_tokens), 1)
        return prompt_cache

    def _make_configured_prompt_cache(self, model: Any) -> list[Any]:
        factory = self._prompt_cache_factory
        cache_type = self._kv_cache_type
        if factory is None or cache_type is None:
            raise RuntimeError("Prompt cache factory is not initialized")
        prompt_cache = factory(model)
        configured = self._configure_prompt_cache_step(
            prompt_cache,
            cache_type,
            self.settings.engine.kv_cache_step_tokens,
        )
        if (
            self.settings.engine.paged_cache_direct_attention_enabled
            and not self.settings.engine.paged_cache_enabled
        ):
            raise ConfigurationError(
                code="paged_direct_attention_requires_paged_cache",
                message="paged_cache_direct_attention_enabled requires paged_cache_enabled",
                status_code=400,
            )
        if not self.settings.engine.paged_cache_enabled:
            return configured
        if self.settings.engine.prefix_cache_enabled:
            raise ConfigurationError(
                code="paged_cache_prefix_cache_conflict",
                message="paged_cache_enabled requires prefix_cache_enabled=false",
                status_code=400,
            )
        if self.settings.engine.max_decode_batch != 1:
            raise ConfigurationError(
                code="paged_cache_batch_conflict",
                message="paged_cache_enabled currently requires max_decode_batch=1",
                status_code=400,
            )
        if self._paged_cache is None:
            raise ConfigurationError(
                code="paged_cache_unavailable",
                message="Paged KV cache manager is unavailable for this model",
                status_code=500,
            )
        from aster.inference.paged_kv_adapter import PagedKVCacheBundle

        bundle = PagedKVCacheBundle.from_prompt_cache(
            configured,
            self._paged_cache,
            kv_cache_type=cache_type,
            request_id=f"model-runner-cache-{id(configured)}",
            enable_block_pool=False,
            enable_direct_attention=self.settings.engine.paged_cache_direct_attention_enabled,
        )
        return bundle.caches

    def model_fingerprint(self) -> str:
        self._ensure_loaded()
        if self._model_fingerprint is None:
            self._model_fingerprint = self._compute_model_fingerprint()
        return self._model_fingerprint

    def strict_chat_prefix_prompt(
        self,
        messages: list[dict[str, Any]],
        *,
        enable_thinking: bool,
        chat_template_kwargs: dict[str, Any] | None = None,
    ) -> str | None:
        self._ensure_loaded()
        tokenizer = self._tokenizer
        if tokenizer is None:
            return None
        return build_strict_prefix_string(
            tokenizer,
            messages,
            enable_thinking=enable_thinking,
            chat_template_kwargs=chat_template_kwargs,
        )

    def available_memory_bytes(self) -> int:
        try:
            import psutil

            return int(psutil.virtual_memory().available)
        except Exception:
            return 8 * 1024 * 1024 * 1024

    def _encode_text(self, prompt: str) -> list[int]:
        tokenizer = self._tokenizer
        assert tokenizer is not None
        add_special_tokens = tokenizer.bos_token is None or not prompt.startswith(
            tokenizer.bos_token or ""
        )
        return list(tokenizer.encode(prompt, add_special_tokens=add_special_tokens))

    def _encode_chat(
        self,
        messages: list[dict[str, str]],
        *,
        enable_thinking: bool,
        chat_template_kwargs: dict[str, Any] | None = None,
    ) -> list[int]:
        return self._chat_tokens(
            messages,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
            chat_template_kwargs=chat_template_kwargs,
        )

    def _chat_tokens(
        self,
        messages: list[dict[str, str]],
        *,
        add_generation_prompt: bool,
        enable_thinking: bool,
        chat_template_kwargs: dict[str, Any] | None = None,
    ) -> list[int]:
        tokenizer = self._tokenizer
        assert tokenizer is not None
        if hasattr(tokenizer, "apply_chat_template"):
            template_kwargs = dict(chat_template_kwargs or {})
            template_kwargs.update(
                {
                    "tokenize": False,
                    "add_generation_prompt": add_generation_prompt,
                    "enable_thinking": enable_thinking,
                }
            )
            rendered = tokenizer.apply_chat_template(
                messages,
                **template_kwargs,
            )
            return list(tokenizer.encode(rendered, add_special_tokens=False))
        fallback_prompt = "\n".join(f"{item['role']}: {item['content']}" for item in messages)
        if add_generation_prompt:
            fallback_prompt = f"{fallback_prompt}\nassistant:"
        return self._encode_text(fallback_prompt)

    def _chat_reuse_points(
        self,
        messages: list[dict[str, str]],
        *,
        full_prompt_tokens: list[int],
        enable_thinking: bool,
        chat_template_kwargs: dict[str, Any] | None = None,
    ) -> tuple[int, ...]:
        if len(messages) < 2:
            return ()
        reuse_points: set[int] = set()
        lcp_boundary = self._chat_lcp_reuse_point(
            messages,
            full_prompt_tokens=full_prompt_tokens,
            enable_thinking=enable_thinking,
            chat_template_kwargs=chat_template_kwargs,
        )
        if lcp_boundary > 0:
            reuse_points.add(lcp_boundary)

        for boundary in range(1, len(messages)):
            try:
                boundary_tokens = self._chat_tokens(
                    messages[:boundary],
                    add_generation_prompt=False,
                    enable_thinking=enable_thinking,
                    chat_template_kwargs=chat_template_kwargs,
                )
            except Exception:
                continue
            if not boundary_tokens:
                continue
            if full_prompt_tokens[: len(boundary_tokens)] != boundary_tokens:
                continue
            reuse_points.add(len(boundary_tokens))
        ordered_points = tuple(sorted(reuse_points))
        max_points = self.settings.engine.snapshot_max_chat_reuse_points
        if max_points <= 0 or len(ordered_points) <= max_points:
            return ordered_points

        selected_points = set(ordered_points[-max_points:])
        sparse_points = self.settings.engine.snapshot_chat_reuse_sparse_points
        sparse_min_tokens = self.settings.engine.snapshot_chat_reuse_sparse_min_tokens
        if sparse_points > 0 and len(full_prompt_tokens) >= sparse_min_tokens:
            # Keep recent boundaries dense and older history logarithmically sparse.
            offset = max_points * 2
            for _ in range(max(sparse_points - 1, 0)):
                if offset > len(ordered_points):
                    break
                selected_points.add(ordered_points[-offset])
                offset *= 2
            for point in ordered_points:
                if point >= self.settings.engine.snapshot_min_prefix_tokens:
                    selected_points.add(point)
                    break
        return tuple(sorted(selected_points))

    def _chat_lcp_reuse_point(
        self,
        messages: list[dict[str, str]],
        *,
        full_prompt_tokens: list[int],
        enable_thinking: bool,
        chat_template_kwargs: dict[str, Any] | None = None,
    ) -> int:
        last_user_idx = None
        for index in range(len(messages) - 1, -1, -1):
            if messages[index].get("role") == "user":
                last_user_idx = index
                break
        if last_user_idx is None or last_user_idx == 0:
            return 0

        dummy_messages = [dict(message) for message in messages]
        dummy_messages[last_user_idx] = {
            **dummy_messages[last_user_idx],
            "content": "XXXXXXXXXX",
        }
        try:
            dummy_tokens = self._chat_tokens(
                dummy_messages,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
                chat_template_kwargs=chat_template_kwargs,
            )
        except Exception:
            return 0

        limit = min(len(full_prompt_tokens), len(dummy_tokens))
        boundary = 0
        for index in range(limit):
            if full_prompt_tokens[index] != dummy_tokens[index]:
                break
            boundary = index + 1
        if boundary >= len(full_prompt_tokens):
            return 0
        return boundary

    def _decode_batch(self, items: list[DecodeWorkItem]) -> list[DecodeResult]:
        self._ensure_loaded()
        mx = self._mx
        model = self._model
        assert mx is not None and model is not None

        merged_cache, batch_cache_state = self._get_decode_batch_cache(items)
        input_tokens = mx.array([[item.input_token] for item in items], dtype=mx.uint32)
        logits = model(input_tokens, cache=merged_cache)
        logits = logits[:, -1, :]
        if self._uses_eager_row_sampling(items):
            return self._decode_batch_eager_rows(
                items,
                logits=logits,
                merged_cache=merged_cache,
                batch_cache_state=batch_cache_state,
            )

        sampled_tokens: list[Any] = []
        for index, item in enumerate(items):
            row = self._apply_logits_processors(
                logits[index : index + 1],
                item=item,
            )
            logprobs = row - mx.logsumexp(row, axis=-1, keepdims=True)
            sampled_tokens.append(item.sampler(logprobs))

        lazy_samples, trusted_array_type = self._mlx_sample_arrays(mx, sampled_tokens)
        if not lazy_samples:
            evaluation_targets: Any = logits
        elif trusted_array_type and len(lazy_samples) == len(sampled_tokens):
            evaluation_targets = lazy_samples
        else:
            evaluation_targets = [logits, *lazy_samples]
        try:
            mx.async_eval(evaluation_targets)
            peak_memory_gb = self.current_peak_memory_gb()
            prompt_caches = [
                (
                    self._decode_cache_ref(batch_cache_state, index)
                    if batch_cache_state is not None
                    else self._extract_prompt_cache(merged_cache, index)
                )
                for index in range(len(items))
            ]
            mx.eval(evaluation_targets)
        except MemoryError:
            raise
        except Exception as exc:
            raise _BatchPostSampleError(exc) from exc

        try:
            results: list[DecodeResult] = []
            for item, sampled, prompt_cache in zip(
                items, sampled_tokens, prompt_caches, strict=True
            ):
                results.append(
                    self._decode_result(
                        item=item,
                        token=self._materialize_sampled_token(sampled),
                        prompt_cache=prompt_cache,
                        peak_memory_gb=peak_memory_gb,
                    )
                )
        except MemoryError:
            raise
        except Exception as exc:
            raise _BatchPostSampleError(exc) from exc
        return results

    def _decode_batch_eager_rows(
        self,
        items: list[DecodeWorkItem],
        *,
        logits: Any,
        merged_cache: list[Any],
        batch_cache_state: _DecodeBatchCacheState | None,
    ) -> list[DecodeResult]:
        mx = self._mx
        assert mx is not None
        mx.eval(logits)
        peak_memory_gb = self.current_peak_memory_gb()
        results: list[DecodeResult] = []
        row_phase_started = False
        try:
            for index, item in enumerate(items):
                row_phase_started = True
                row = self._apply_logits_processors(
                    logits[index : index + 1],
                    item=item,
                )
                logprobs = row - mx.logsumexp(row, axis=-1, keepdims=True)
                prompt_cache = (
                    self._decode_cache_ref(batch_cache_state, index)
                    if batch_cache_state is not None
                    else self._extract_prompt_cache(merged_cache, index)
                )
                results.append(
                    self._decode_result(
                        item=item,
                        token=self._sample_token(logprobs, item.sampler),
                        prompt_cache=prompt_cache,
                        peak_memory_gb=peak_memory_gb,
                    )
                )
        except MemoryError:
            raise
        except Exception as exc:
            if row_phase_started:
                raise _BatchPostSampleError(exc) from exc
            raise
        return results

    def _decode_single(self, item: DecodeWorkItem) -> DecodeResult:
        self._ensure_loaded()
        mx = self._mx
        model = self._model
        assert mx is not None and model is not None

        prompt_cache = self._resolve_decode_cache(item.prompt_cache)
        logits = model(mx.array([[item.input_token]], dtype=mx.uint32), cache=prompt_cache)
        logits = logits[:, -1, :]
        logits = self._apply_logits_processors(logits, item=item)
        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        token = self._sample_token(logprobs, item.sampler)
        return self._decode_result(
            item=item,
            token=token,
            prompt_cache=prompt_cache,
            peak_memory_gb=self.current_peak_memory_gb(),
        )

    def _maybe_clear_decode_cache(self, generated_tokens: int = 1) -> None:
        if generated_tokens <= 0:
            return
        self._decode_tokens_since_cache_clear += generated_tokens
        if self._decode_tokens_since_cache_clear < _DECODE_CACHE_CLEAR_TOKEN_BUDGET:
            return
        mx = self._mx
        if mx is None:
            return
        self._decode_cache_clear_attempts += 1
        try:
            mx.clear_cache()
        except Exception:
            self._decode_cache_clear_failures += 1
        else:
            self._decode_cache_clears += 1
            self._decode_tokens_since_cache_clear %= _DECODE_CACHE_CLEAR_TOKEN_BUDGET

    def _thinking_budget_processor(
        self,
        request: InferenceRequest,
        *,
        tokenizer: Any,
        inner: Any | None,
    ) -> ThinkingAwareLogitsProcessor | None:
        if not request.enable_thinking or request.thinking_token_budget is None:
            return None
        try:
            start_ids = tokenizer.encode("<think>", add_special_tokens=False)
            end_ids = tokenizer.encode("</think>", add_special_tokens=False)
        except Exception:
            return None
        if not start_ids or not end_ids:
            return None
        return ThinkingAwareLogitsProcessor(
            start_token_ids=list(start_ids),
            end_token_ids=list(end_ids),
            thinking_token_budget=request.thinking_token_budget,
            inner=inner,
            prompt_has_think_tag=True,
        )

    @staticmethod
    def _single_token_stop_ids(stop_sequences: tuple[str, ...], tokenizer: Any) -> frozenset[int]:
        token_ids: set[int] = set()
        for stop_sequence in stop_sequences:
            try:
                encoded = tokenizer.encode(stop_sequence, add_special_tokens=False)
            except Exception:
                continue
            if len(encoded) == 1:
                token_ids.add(int(encoded[0]))
        return frozenset(token_ids)

    def _decode_result(
        self,
        *,
        item: DecodeWorkItem,
        token: int,
        prompt_cache: Any,
        peak_memory_gb: float,
    ) -> DecodeResult:
        if token in item.stop_token_ids:
            return DecodeResult(
                prompt_cache=prompt_cache,
                token_id=token,
                text="",
                completion_tokens=item.completion_tokens + 1,
                peak_memory_gb=peak_memory_gb,
                finish_reason="stop",
            )

        item.detokenizer.add_token(token)
        completion_tokens = item.completion_tokens + 1
        finish_reason = "length" if completion_tokens >= item.max_tokens else None
        return DecodeResult(
            prompt_cache=prompt_cache,
            token_id=token,
            text=str(item.detokenizer.last_segment),
            completion_tokens=completion_tokens,
            peak_memory_gb=peak_memory_gb,
            finish_reason=finish_reason,
        )

    def _sample_token(self, logprobs: Any, sampler: Any) -> int:
        return self._materialize_sampled_token(sampler(logprobs))

    @staticmethod
    def _uses_eager_row_sampling(items: list[DecodeWorkItem]) -> bool:
        for item in items:
            for processor in item.logits_processors:
                visited: set[int] = set()
                current = processor
                while current is not None and id(current) not in visited:
                    visited.add(id(current))
                    if getattr(current, "batch_sampling_mode", None) == "eager_rows":
                        return True
                    current = getattr(current, "_inner", None)
        return False

    @staticmethod
    def _mlx_sample_arrays(
        mx: Any,
        sampled_tokens: list[Any],
    ) -> tuple[list[Any], bool]:
        array_type = getattr(mx, "array", None)
        if isinstance(array_type, type):
            return [
                sampled for sampled in sampled_tokens if isinstance(sampled, array_type)
            ], True
        return [
            sampled
            for sampled in sampled_tokens
            if hasattr(sampled, "shape") and hasattr(sampled, "dtype")
        ], False

    @staticmethod
    def _materialize_sampled_token(sampled: Any) -> int:
        if hasattr(sampled, "item"):
            return int(sampled.item())
        if hasattr(sampled, "tolist"):
            values = sampled.tolist()
            if isinstance(values, list):
                return int(values[0])
            return int(values)
        if isinstance(sampled, (list, tuple)):
            return int(sampled[0])
        return int(sampled)

    def _apply_logits_processors(self, logits: Any, *, item: DecodeWorkItem) -> Any:
        if not item.logits_processors:
            return logits
        mx = self._mx
        assert mx is not None
        context_size = item.logits_processor_context_size
        if context_size is None:
            processor_tokens = item.logits_processor_tokens + [item.input_token]
        else:
            preceding = max(context_size - 1, 0)
            processor_tokens = [
                *(
                    item.logits_processor_tokens[-preceding:]
                    if preceding
                    else ()
                ),
                item.input_token,
            ]
        tokens = mx.array(processor_tokens, dtype=mx.uint32)
        for processor in item.logits_processors:
            logits = processor(tokens, logits)
        return logits

    def _get_decode_batch_cache(
        self,
        items: list[DecodeWorkItem],
    ) -> tuple[list[Any], _DecodeBatchCacheState | None]:
        request_ids = tuple(item.request_id for item in items)
        current = self._decode_batch_cache_state
        if (
            current is not None
            and all(request_id is not None for request_id in request_ids)
            and current.request_ids == request_ids
        ):
            self._decode_batch_cache_reuses += 1
            return current.merged_cache, current

        caches = [self._resolve_decode_cache(item.prompt_cache) for item in items]
        merged_cache = self._merge_prompt_caches(caches)
        if (
            request_ids
            and all(request_id is not None for request_id in request_ids)
            and len(set(request_ids)) == len(request_ids)
        ):
            state = _DecodeBatchCacheState(
                request_ids=tuple(str(request_id) for request_id in request_ids),
                merged_cache=merged_cache,
            )
            self._decode_batch_cache_state = state
            self._decode_batch_cache_rebuilds += 1
            return merged_cache, state

        self._decode_batch_cache_state = None
        return merged_cache, None

    @staticmethod
    def _decode_cache_ref(state: _DecodeBatchCacheState, index: int) -> _DecodeCacheRef:
        return _DecodeCacheRef(state=state, index=index)

    def _resolve_decode_cache(self, prompt_cache: Any) -> Any:
        if isinstance(prompt_cache, _DecodeCacheRef):
            return self._extract_prompt_cache(prompt_cache.state.merged_cache, prompt_cache.index)
        return prompt_cache

    def _merge_prompt_caches(self, caches: list[Any]) -> list[Any]:
        if not caches:
            return []
        merged: list[Any] = []
        for index in range(len(caches[0])):
            layer = caches[0][index]
            if not hasattr(layer, "merge"):
                raise ValueError(f"{type(layer)} does not support merge-based batch decode")
            merged.append(layer.merge([cache[index] for cache in caches]))
        return merged

    def _extract_prompt_cache(self, merged_cache: list[Any], index: int) -> list[Any]:
        extracted: list[Any] = []
        for layer in merged_cache:
            if not hasattr(layer, "extract"):
                raise ValueError(f"{type(layer)} does not support extract-based batch decode")
            extracted.append(layer.extract(index))
        return extracted

    def _eval_cache(self, prompt_cache: Any) -> None:
        mx = self._mx
        if mx is None:
            return
        try:
            eval_targets = []
            for cache in prompt_cache:
                if getattr(cache, "direct_attention_enabled", False):
                    if getattr(cache, "_pool_enabled", False):
                        pool_keys, pool_values, _ = cache.attention_view().block_pool()
                        eval_targets.extend((pool_keys, pool_values))
                    else:
                        eval_targets.append(cache.state)
                else:
                    eval_targets.append(cache.state)
            mx.eval(eval_targets)
        except Exception:
            pass
        try:
            mx.clear_cache()
        except Exception:
            pass
        else:
            self._decode_tokens_since_cache_clear = 0

    def _trim_cache_to_token_count(self, prompt_cache: Any, cache_token_count: int) -> Any:
        if not isinstance(prompt_cache, list) or cache_token_count < 0:
            return prompt_cache

        current_offset = self._cache_offset(prompt_cache)
        if current_offset < cache_token_count:
            return prompt_cache
        needs_array_trim = self._cache_has_oversized_arrays(prompt_cache, cache_token_count)
        if current_offset == cache_token_count and not needs_array_trim:
            return prompt_cache
        if not self._cache_can_rewind(prompt_cache):
            if current_offset == cache_token_count:
                return prompt_cache
            raise ValueError("Prompt cache cannot be safely rewound for prefix reuse")

        eval_targets: list[Any] = []
        for layer in prompt_cache:
            keys = getattr(layer, "keys", None)
            values = getattr(layer, "values", None)
            key_shape = getattr(keys, "shape", None)
            value_shape = getattr(values, "shape", None)
            if key_shape is not None and len(key_shape) >= 3 and cache_token_count < key_shape[-2]:
                layer.keys = keys[..., :cache_token_count, :]
                eval_targets.append(layer.keys)
            if (
                value_shape is not None
                and len(value_shape) >= 3
                and cache_token_count < value_shape[-2]
            ):
                layer.values = values[..., :cache_token_count, :]
                eval_targets.append(layer.values)
            layer.offset = cache_token_count

        mx = self._mx
        if mx is not None and eval_targets:
            try:
                mx.eval(*eval_targets)
            except Exception:
                pass
        return prompt_cache

    @staticmethod
    def _cache_has_oversized_arrays(prompt_cache: Any, cache_token_count: int) -> bool:
        if not isinstance(prompt_cache, list) or cache_token_count < 0:
            return False
        for layer in prompt_cache:
            for attr in ("keys", "values"):
                tensor = getattr(layer, attr, None)
                if tensor is None or isinstance(tensor, (list, tuple)):
                    continue
                shape = getattr(tensor, "shape", None)
                if shape is not None and len(shape) >= 3 and cache_token_count < shape[-2]:
                    return True
        return False

    @staticmethod
    def _cache_offset(prompt_cache: Any) -> int:
        if not isinstance(prompt_cache, list) or not prompt_cache:
            return 0
        offsets: list[int] = []
        for layer in prompt_cache:
            offset = getattr(layer, "offset", None)
            if isinstance(offset, int):
                offsets.append(max(offset, 0))
        return min(offsets) if offsets else 0

    @staticmethod
    def _cache_can_rewind(prompt_cache: Any) -> bool:
        if not isinstance(prompt_cache, list) or not prompt_cache:
            return False
        for layer in prompt_cache:
            offset = getattr(layer, "offset", None)
            keys = getattr(layer, "keys", None)
            values = getattr(layer, "values", None)
            if offset is None or keys is None or values is None:
                return False
            if isinstance(keys, (list, tuple)) or isinstance(values, (list, tuple)):
                return False
            if hasattr(layer, "max_size") or hasattr(layer, "_idx"):
                return False
            key_shape = getattr(keys, "shape", None)
            value_shape = getattr(values, "shape", None)
            if key_shape is None or value_shape is None:
                return False
            if len(key_shape) < 3 or len(value_shape) < 3:
                return False
        return True

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        try:
            import mlx.core as mx
            from mlx_lm import load
            from mlx_lm.models import cache as mlx_cache
            from mlx_lm.sample_utils import make_logits_processors, make_sampler
        except Exception as exc:  # pragma: no cover - depends on local runtime
            raise ConfigurationError(
                code="mlx_runtime_unavailable",
                message="MLX runtime dependencies are not installed or importable.",
                status_code=500,
                details={"error": str(exc)},
            ) from exc

        result = load(self.settings.model.path, lazy=False, return_config=True)
        if len(result) == 3:  # type: ignore[arg-type]
            model, tokenizer, config = result  # type: ignore[misc]
        elif len(result) == 2:  # type: ignore[arg-type]
            model, tokenizer = result  # type: ignore[misc]
            config = {}
        else:  # pragma: no cover - defensive
            raise ConfigurationError(
                code="invalid_model_load_result",
                message="Unexpected MLX load() return shape.",
                status_code=500,
            )

        self._mx = mx
        self._prompt_cache_factory = mlx_cache.make_prompt_cache
        self._kv_cache_type = mlx_cache.KVCache
        self._make_prompt_cache = self._make_configured_prompt_cache
        self._make_sampler = make_sampler
        self._make_logits_processors = make_logits_processors
        self._model = model
        self._tokenizer = tokenizer
        self._config = config or {}
        if self.settings.engine.paged_cache_enabled:
            try:
                from aster.inference.paged_cache import PagedCacheManager

                layers = getattr(model, "layers", None) or getattr(
                    getattr(model, "model", None), "layers", None
                )
                self._paged_cache = PagedCacheManager(
                    num_layers=len(layers) if layers else 24,
                    block_size=self.settings.engine.paged_cache_block_size,
                    max_blocks=self.settings.engine.paged_cache_max_blocks,
                )
            except Exception as exc:
                raise ConfigurationError(
                    code="paged_cache_initialization_failed",
                    message="Unable to initialize the opt-in paged KV cache manager",
                    status_code=500,
                    details={"error": str(exc)},
                ) from exc
            if self.settings.engine.paged_cache_direct_attention_enabled:
                if self._config.get("model_type") != "qwen3_5":
                    raise ConfigurationError(
                        code="paged_direct_attention_unsupported_model",
                        message="Direct paged attention currently supports Qwen3.5 only",
                        status_code=400,
                    )
                from aster.inference.paged_attention_bridge import (
                    install_qwen3_next_paged_attention_bridge,
                )

                install_qwen3_next_paged_attention_bridge()
        self._model_fingerprint = self._compute_model_fingerprint()
        self._loaded = True

    def cache_token_count(self, prompt_cache: Any | None) -> int:
        return prompt_cache_length(prompt_cache)

    def _compute_model_fingerprint(self) -> str:
        parts = [
            f"name={self.settings.model.name}",
            f"path={self.settings.model.path}",
        ]
        for key in (
            "model_type",
            "num_hidden_layers",
            "n_layers",
            "hidden_size",
            "dim",
            "vocab_size",
            "num_attention_heads",
            "n_heads",
            "num_key_value_heads",
            "n_kv_heads",
            "head_dim",
            "intermediate_size",
        ):
            value = self._config.get(key)
            if value is not None:
                parts.append(f"{key}={value}")
        return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]
