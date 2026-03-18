from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from itertools import product
from types import SimpleNamespace
from typing import Literal

from aster.core.config import RuntimeSettings
from aster.inference.engine import InferenceEngine, InferenceRequest
from aster.scheduler.policy_engine import PolicyEngine
from aster.telemetry.logging import get_logger
from aster.telemetry.metrics import MetricsRegistry


@dataclass(slots=True)
class BenchmarkResult:
    speculative_enabled: bool
    draft_tokens: int
    batch_window_ms: float
    max_batch_size: int
    stream_flush_ms: float
    latency_score: float
    throughput_score: float
    stability_score: float
    prompt_tokens: int
    completion_tokens: int
    prompt_tps: float
    generation_tps: float
    elapsed_seconds: float
    second_elapsed_seconds: float
    second_cache_hit: bool
    prefill_cache_hit: bool
    generation_cache_reuse: bool
    speculative_path_mode: str
    mode: str

    @property
    def reuse_gain_ratio(self) -> float:
        if self.second_elapsed_seconds <= 0:
            return 0.0
        return self.elapsed_seconds / self.second_elapsed_seconds

    @property
    def total_score(self) -> float:
        reuse_bonus = 0.5 if self.generation_cache_reuse else 0.0
        return self.throughput_score + self.stability_score + reuse_bonus - self.latency_score


@dataclass(frozen=True, slots=True)
class BenchmarkCandidate:
    speculative_enabled: bool
    draft_tokens: int
    batch_window_ms: float
    max_batch_size: int
    stream_flush_ms: float


class BenchmarkSuite:
    def __init__(self, settings: RuntimeSettings, metrics: MetricsRegistry, policy_engine: PolicyEngine) -> None:
        self.settings = settings
        self.metrics = metrics
        self.policy_engine = policy_engine
        self.logger = get_logger(__name__)

    async def run(
        self,
        *,
        mode: Literal["quick", "full"] = "quick",
        candidate_timeout_seconds: float = 90.0,
        shortlist_size: int = 3,
        progress: bool = False,
    ) -> list[BenchmarkResult]:
        candidates = self._candidates(mode)
        results: list[BenchmarkResult] = []

        for index, candidate in enumerate(candidates, start=1):
            if progress:
                print(
                    f"[benchmark] phase=1 mode={mode} candidate={index}/{len(candidates)} "
                    f"spec={candidate.speculative_enabled} draft={candidate.draft_tokens} "
                    f"window={candidate.batch_window_ms} batch={candidate.max_batch_size} flush={candidate.stream_flush_ms}",
                    flush=True,
                )
            result = await self._measure_with_timeout(
                candidate,
                mode=mode,
                max_tokens=16 if mode == "quick" else 24,
                timeout_seconds=candidate_timeout_seconds,
            )
            if result is not None:
                results.append(result)

        ranked = sorted(results, key=lambda x: x.total_score, reverse=True)
        if mode == "quick":
            return ranked

        finalists = ranked[:shortlist_size]
        refined: list[BenchmarkResult] = []
        for index, prior in enumerate(finalists, start=1):
            candidate = BenchmarkCandidate(
                speculative_enabled=prior.speculative_enabled,
                draft_tokens=prior.draft_tokens,
                batch_window_ms=prior.batch_window_ms,
                max_batch_size=prior.max_batch_size,
                stream_flush_ms=prior.stream_flush_ms,
            )
            if progress:
                print(
                    f"[benchmark] phase=2 finalist={index}/{len(finalists)} "
                    f"spec={candidate.speculative_enabled} draft={candidate.draft_tokens} "
                    f"window={candidate.batch_window_ms} batch={candidate.max_batch_size} flush={candidate.stream_flush_ms}",
                    flush=True,
                )
            result = await self._measure_with_timeout(
                candidate,
                mode="full",
                max_tokens=48,
                timeout_seconds=max(candidate_timeout_seconds, 180.0),
            )
            if result is not None:
                refined.append(result)

        return sorted(refined or ranked, key=lambda x: x.total_score, reverse=True)

    async def _measure_with_timeout(
        self,
        candidate: BenchmarkCandidate,
        *,
        mode: str,
        max_tokens: int,
        timeout_seconds: float,
    ) -> BenchmarkResult | None:
        try:
            return await asyncio.wait_for(
                self._measure_candidate(candidate, mode=mode, max_tokens=max_tokens),
                timeout=timeout_seconds,
            )
        except TimeoutError:
            self.logger.warning(
                "benchmark_candidate_timeout "
                f"spec={candidate.speculative_enabled} draft={candidate.draft_tokens} "
                f"window={candidate.batch_window_ms} batch={candidate.max_batch_size} flush={candidate.stream_flush_ms}"
            )
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            self.logger.error(
                "benchmark_candidate_failed "
                f"spec={candidate.speculative_enabled} draft={candidate.draft_tokens} "
                f"window={candidate.batch_window_ms} batch={candidate.max_batch_size} flush={candidate.stream_flush_ms} "
                f"error={exc}"
            )
        return None

    async def _measure_candidate(
        self,
        candidate: BenchmarkCandidate,
        *,
        mode: str,
        max_tokens: int,
    ) -> BenchmarkResult:
        self.policy_engine.update(
            speculative_enabled=candidate.speculative_enabled,
            speculative_draft_tokens=candidate.draft_tokens,
            batch_window_ms=candidate.batch_window_ms,
            max_batch_size=candidate.max_batch_size,
            stream_flush_ms=candidate.stream_flush_ms,
        )
        deps = self._fresh_engine_dep()
        engine = InferenceEngine(
            self.settings,
            self.metrics,
            kv_cache=deps.kv_cache,
            prefix_cache=deps.prefix_cache,
            policy_engine=self.policy_engine,
        )
        first_prompt, second_prompt = self._benchmark_prompts(mode)

        started = time.perf_counter()
        await engine.infer(
            InferenceRequest(
                prompt=first_prompt,
                max_tokens=max_tokens,
                stream=False,
                temperature=0.0,
                top_p=1.0,
                request_class="benchmark",
            )
        )
        elapsed = max(time.perf_counter() - started, 1e-6)

        second_started = time.perf_counter()
        second = await engine.infer(
            InferenceRequest(
                prompt=second_prompt,
                max_tokens=max_tokens,
                stream=False,
                temperature=0.0,
                top_p=1.0,
                request_class="benchmark",
            )
        )
        second_elapsed = max(time.perf_counter() - second_started, 1e-6)

        throughput = second.completion_tokens / second_elapsed
        latency_penalty = (elapsed + second_elapsed) / 2.0
        throughput_gain = throughput
        stability = 1.0
        if candidate.speculative_enabled and second.generation_tps <= 0:
            stability -= 0.2
        if candidate.speculative_enabled and not second.speculative_enabled:
            stability -= 0.15
        if second.peak_memory_gb > 0 and second.peak_memory_gb > 32.0:
            stability -= 0.1
        if not second.prefill_cache_hit:
            stability -= 0.03
        if candidate.speculative_enabled and not second.generation_cache_reuse:
            stability -= 0.12

        return BenchmarkResult(
            speculative_enabled=candidate.speculative_enabled,
            draft_tokens=candidate.draft_tokens,
            batch_window_ms=candidate.batch_window_ms,
            max_batch_size=candidate.max_batch_size,
            stream_flush_ms=candidate.stream_flush_ms,
            latency_score=latency_penalty,
            throughput_score=throughput_gain,
            stability_score=stability,
            prompt_tokens=second.prompt_tokens,
            completion_tokens=second.completion_tokens,
            prompt_tps=second.prompt_tps,
            generation_tps=second.generation_tps,
            elapsed_seconds=elapsed,
            second_elapsed_seconds=second_elapsed,
            second_cache_hit=second.cache_hit,
            prefill_cache_hit=second.prefill_cache_hit,
            generation_cache_reuse=second.generation_cache_reuse,
            speculative_path_mode=second.speculative_path_mode,
            mode=mode,
        )

    def _candidates(self, mode: str) -> list[BenchmarkCandidate]:
        if mode == "quick":
            speculative_values = [False, True]
            draft_values = sorted({2, min(self.settings.speculative.max_draft_tokens, 4)})
            window_values = sorted({self.settings.batch.min_batch_window_ms, 4.0})
            batch_values = sorted({2, min(4, self.settings.batch.max_batch_size)})
            flush_values = [6.0, 12.0]
        else:
            speculative_values = [False, True]
            draft_values = sorted({2, 4, min(self.settings.speculative.max_draft_tokens, 6)})
            window_values = sorted({self.settings.batch.min_batch_window_ms, 4.0, 8.0})
            batch_values = sorted({2, min(4, self.settings.batch.max_batch_size), self.settings.batch.max_batch_size})
            flush_values = [6.0, 12.0, 20.0]

        candidates = {
            BenchmarkCandidate(*items)
            for items in product(speculative_values, draft_values, window_values, batch_values, flush_values)
        }
        return sorted(
            candidates,
            key=lambda c: (
                c.speculative_enabled,
                c.draft_tokens,
                c.batch_window_ms,
                c.max_batch_size,
                c.stream_flush_ms,
            ),
        )

    def _benchmark_prompts(self, mode: str) -> tuple[str, str]:
        base = (
            "You are an Apple Silicon local inference engine optimized for long-context agent workloads. "
            "System tools, memory, long prefixes, and repeated scaffolding matter. "
        )
        tool_count = 64 if mode == "quick" else 256
        tool_block = "\n".join(
            f"tool_{i}: accepts JSON and returns structured results with deterministic semantics."
            for i in range(tool_count)
        )
        shared = f"{base}\n{tool_block}\nSystem: preserve repeated prefix state for future turns."
        first = f"{shared}\nUser: summarize why prefix caching matters."
        second = f"{shared}\nUser: summarize why prefix caching matters for repeated agent prefixes in one sentence."
        return first, second

    def _fresh_engine_dep(self) -> SimpleNamespace:
        from cache.paged_kv_cache import PagedKVCache
        from cache.prefix_cache import PrefixCache

        return SimpleNamespace(
            kv_cache=PagedKVCache(self.settings.cache, self.metrics),
            prefix_cache=PrefixCache(self.settings.cache, self.metrics),
        )
