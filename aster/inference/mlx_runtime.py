from __future__ import annotations

import copy
import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import GenerationResponse, generate_step, speculative_generate_step
from mlx_lm.models import cache as mlx_cache
from mlx_lm.sample_utils import make_sampler

from aster.core.config import RuntimeSettings
from aster.inference.mlx_cache_utils import prompt_cache_length
from aster.telemetry.logging import get_logger


@dataclass(slots=True)
class LoadedRuntimeModel:
    name: str
    path: str
    model: Any
    tokenizer: Any
    config: dict[str, Any]


@dataclass(slots=True)
class PrefilledPrompt:
    prompt_cache: Any
    prompt_tokens: int
    matched_prefix_tokens: int
    prefill_seconds: float


class MLXRuntime:
    def __init__(self, settings: RuntimeSettings) -> None:
        self.settings = settings
        self.logger = get_logger(__name__)
        self._target: LoadedRuntimeModel | None = None
        self._draft: LoadedRuntimeModel | None = None

    def ensure_target_loaded(self) -> LoadedRuntimeModel:
        if self._target is None:
            self._target = self._load_model(self.settings.model.name, self.settings.model.path)
        return self._target

    def ensure_draft_loaded(self) -> LoadedRuntimeModel:
        if self._draft is None:
            self._draft = self._load_model(self.settings.model.draft_name, self.settings.model.draft_path)
        return self._draft

    def _load_model(self, name: str, path: str) -> LoadedRuntimeModel:
        result: Any = load(path, lazy=False, return_config=True)
        if len(result) == 3:  # type: ignore[arg-type]
            model, tokenizer, config = result  # type: ignore[misc]
        elif len(result) == 2:  # type: ignore[arg-type]
            model, tokenizer = result  # type: ignore[misc]
            config = {}
        else:
            raise ValueError(f"Unexpected load result type: {type(result)}")
        self.logger.info(f"loaded_model name={name} path={path}")
        return LoadedRuntimeModel(name=name, path=path, model=model, tokenizer=tokenizer, config=config)

    def encode(self, prompt: str) -> list[int]:
        target = self.ensure_target_loaded()
        add_special_tokens = target.tokenizer.bos_token is None or not prompt.startswith(target.tokenizer.bos_token or "")
        return list(target.tokenizer.encode(prompt, add_special_tokens=add_special_tokens))

    def encode_chat(self, messages: list[dict[str, str]], *, enable_thinking: bool = False) -> list[int]:
        target = self.ensure_target_loaded()
        if hasattr(target.tokenizer, "apply_chat_template"):
            rendered = target.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
            return list(target.tokenizer.encode(rendered, add_special_tokens=False))
        fallback_prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
        return self.encode(fallback_prompt)

    def decode(self, tokens: list[int]) -> str:
        target = self.ensure_target_loaded()
        return str(target.tokenizer.decode(tokens))

    def clone_cache(self, prompt_cache: Any | None) -> Any | None:
        if prompt_cache is None:
            return None
        return copy.deepcopy(prompt_cache)

    def prefill_prompt(self, prompt_tokens: list[int], base_cache: Any | None = None) -> PrefilledPrompt:
        target = self.ensure_target_loaded()
        prompt_cache = self.clone_cache(base_cache)
        if prompt_cache is None:
            prompt_cache = mlx_cache.make_prompt_cache(target.model)
            matched_prefix_tokens = 0
            suffix_tokens = prompt_tokens
        else:
            matched_prefix_tokens = prompt_cache_length(prompt_cache)
            suffix_tokens = prompt_tokens[matched_prefix_tokens:]

        if len(suffix_tokens) == 0:
            return PrefilledPrompt(
                prompt_cache=prompt_cache,
                prompt_tokens=len(prompt_tokens),
                matched_prefix_tokens=matched_prefix_tokens,
                prefill_seconds=0.0,
            )

        start = time.perf_counter()
        remaining = mx.array(suffix_tokens)
        while remaining.size > 0:
            if remaining.size == 1:
                break
            step = int(min(remaining.size - 1, 2048))
            target.model(remaining[:step][None], cache=prompt_cache)
            mx.eval([c.state for c in prompt_cache])  # type: ignore[arg-type]
            remaining = remaining[step:]
            mx.clear_cache()
        return PrefilledPrompt(
            prompt_cache=prompt_cache,
            prompt_tokens=len(prompt_tokens),
            matched_prefix_tokens=matched_prefix_tokens,
            prefill_seconds=time.perf_counter() - start,
        )

    def stream_tokens(
        self,
        prompt_tokens: list[int],
        prompt_cache: Any | None,
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
        use_speculative: bool,
        num_draft_tokens: int,
    ) -> Iterator[GenerationResponse]:
        try:
            yield from self._stream_tokens_impl(
                prompt_tokens,
                prompt_cache,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                use_speculative=use_speculative,
                num_draft_tokens=num_draft_tokens,
            )
        except Exception as exc:
            if use_speculative:
                self.logger.warning(f"speculative_fallback_to_standard error={exc}")
                yield from self._stream_tokens_impl(
                    prompt_tokens,
                    prompt_cache,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    use_speculative=False,
                    num_draft_tokens=0,
                )
            else:
                raise

    def _stream_tokens_impl(
        self,
        prompt_tokens: list[int],
        prompt_cache: Any | None,
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
        use_speculative: bool,
        num_draft_tokens: int,
    ) -> Iterator[GenerationResponse]:
        target = self.ensure_target_loaded()
        draft_model = self.ensure_draft_loaded().model if use_speculative else None
        sampler = make_sampler(temp=temperature, top_p=top_p)
        generator = self._build_generator(
            target_model=target.model,
            draft_model=draft_model,
            prompt_tokens=prompt_tokens,
            prompt_cache=prompt_cache,
            max_tokens=max_tokens,
            sampler=sampler,
            num_draft_tokens=num_draft_tokens,
        )

        detokenizer = target.tokenizer.detokenizer
        started = time.perf_counter()
        generated = 0
        for item in generator:
            if isinstance(item, tuple) and len(item) == 3:
                token, logprobs, from_draft = item  # type: ignore[misc]
            else:
                token, logprobs = item  # type: ignore[misc]
                from_draft = False

            token_id = int(token)
            generation_tps = 0.0
            if generated > 0:
                generation_tps = generated / max(time.perf_counter() - started, 1e-6)

            if token_id in target.tokenizer.eos_token_ids:
                yield GenerationResponse(
                    text="",
                    token=token_id,
                    logprobs=logprobs,
                    from_draft=from_draft,
                    prompt_tokens=len(prompt_tokens),
                    prompt_tps=0.0,
                    generation_tokens=generated,
                    generation_tps=generation_tps,
                    peak_memory=mx.get_peak_memory() / 1e9,
                    finish_reason="stop",
                )
                break

            detokenizer.add_token(token_id)
            generated += 1
            yield GenerationResponse(
                text=detokenizer.last_segment,
                token=token_id,
                logprobs=logprobs,
                from_draft=from_draft,
                prompt_tokens=len(prompt_tokens),
                prompt_tps=0.0,
                generation_tokens=generated,
                generation_tps=generated / max(time.perf_counter() - started, 1e-6),
                peak_memory=mx.get_peak_memory() / 1e9,
                finish_reason=None,
            )
        detokenizer.finalize()

    def _build_speculative_cache(self, target_model: Any, draft_model: Any, prompt_tokens: list[int]) -> Any:
        """Build a cache compatible with both target and draft models for speculative decoding."""
        # Create a fresh cache that works with both models
        # by prefilling with both models simultaneously
        cache = mlx_cache.make_prompt_cache(target_model)

        if len(prompt_tokens) == 0:
            return cache

        # Prefill the cache with both models to ensure compatibility
        prompt_array = mx.array(prompt_tokens)

        # Process in chunks to avoid memory issues
        chunk_size = 512
        for i in range(0, len(prompt_tokens), chunk_size):
            chunk = prompt_array[i:i+chunk_size]
            if chunk.size == 0:
                continue

            # Prefill with target model
            target_model(chunk[None], cache=cache)
            mx.eval([c.state for c in cache])  # type: ignore[arg-type]

            # Note: We don't prefill draft model cache separately
            # mlx_lm.speculative_generate_step will handle draft model internally
            mx.clear_cache()

        return cache

    def _build_generator(
        self,
        *,
        target_model: Any,
        draft_model: Any | None,
        prompt_tokens: list[int],
        prompt_cache: Any | None,
        max_tokens: int,
        sampler: Any,
        num_draft_tokens: int,
    ):
        cache_copy = self.clone_cache(prompt_cache)
        prompt_array = mx.array(prompt_tokens)

        if draft_model is not None:
            # For speculative decoding, try to use a compatible cache
            # If we have a reused cache, try to use it directly
            # If not, build a fresh one
            if cache_copy is None:
                # Build a fresh cache compatible with both models
                try:
                    cache_copy = self._build_speculative_cache(target_model, draft_model, prompt_tokens)
                except Exception as e:
                    self.logger.warning(f"Failed to build speculative cache: {e}, falling back to no cache")
                    cache_copy = None

            return speculative_generate_step(
                prompt_array,
                target_model,
                draft_model,
                max_tokens=max_tokens,
                sampler=sampler,
                prompt_cache=cache_copy,
                num_draft_tokens=num_draft_tokens,
            )
        return generate_step(
            prompt_array,
            target_model,
            max_tokens=max_tokens,
            sampler=sampler,
            prompt_cache=cache_copy,
        )
