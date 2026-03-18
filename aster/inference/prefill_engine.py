from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from aster.cache.paged_kv_cache import PagedKVCache
from aster.cache.prefix_cache import PrefixCache, PrefixEntry
from aster.telemetry.metrics import MetricsRegistry


@dataclass(slots=True)
class PrefillResult:
    request_id: str
    prompt_tokens: int
    cache_hit: bool
    page_ids: list[int]
    matched_prefix_tokens: int
    suffix_tokens: list[int]
    backend_cache: Any | None


class PrefillEngine:
    def __init__(self, metrics: MetricsRegistry, kv_cache: PagedKVCache, prefix_cache: PrefixCache) -> None:
        self.metrics = metrics
        self.kv_cache = kv_cache
        self.prefix_cache = prefix_cache

    async def run(self, request_id: str, model_name: str, prompt_tokens: list[int]) -> PrefillResult:
        start = time.perf_counter()
        entry = self._lookup_best_prefix(model_name, prompt_tokens)
        page_ids = self.kv_cache.allocate(request_id, len(prompt_tokens))
        matched = entry.token_count if entry is not None else 0
        suffix_tokens = prompt_tokens[matched:]
        self.metrics.prefill_latency.observe(time.perf_counter() - start)
        return PrefillResult(
            request_id=request_id,
            prompt_tokens=len(prompt_tokens),
            cache_hit=entry is not None,
            page_ids=page_ids,
            matched_prefix_tokens=matched,
            suffix_tokens=suffix_tokens,
            backend_cache=None if entry is None else entry.backend_cache,
        )

    def _lookup_best_prefix(self, model_name: str, prompt_tokens: list[int]) -> PrefixEntry | None:
        exact = self.prefix_cache.lookup(model_name, prompt_tokens)
        if exact is not None and exact.token_count > 1:
            exact.tokens = exact.tokens[:-1]
            exact.token_count -= 1
            return exact
        return self.prefix_cache.lookup_longest_prefix(
            model_name,
            prompt_tokens,
            min_prefix_tokens=max(32, min(256, len(prompt_tokens) // 4)),
            min_match_ratio=0.5,
        )
