from __future__ import annotations

import copy
from dataclasses import dataclass
from threading import Lock
from time import time
from typing import Any

from aster.cache.cache_keys import prefix_hash
from aster.cache.eviction import LRUEvictionIndex
from aster.core.config import CacheSettings
from aster.telemetry.metrics import MetricsRegistry


@dataclass(slots=True)
class PrefixEntry:
    key: str
    token_count: int
    tokens: list[int]
    page_ids: list[int]
    approx_bytes: int
    backend_cache: Any | None
    created_at: float
    last_used_at: float


class PrefixCache:
    def __init__(self, settings: CacheSettings, metrics: MetricsRegistry) -> None:
        self.settings = settings
        self.metrics = metrics
        self._lock = Lock()
        self._index = LRUEvictionIndex[str, PrefixEntry]()
        self._entries: dict[str, PrefixEntry] = {}
        self._total_bytes = 0

    def lookup(self, model_name: str, tokens: list[int]) -> PrefixEntry | None:
        if not self.settings.prefix_cache_enabled:
            return None
        key = prefix_hash(model_name, tokens)
        with self._lock:
            entry = self._index.get(key)
            if entry is None:
                self.metrics.prefix_cache_misses.inc()
                return None
            entry.last_used_at = time()
            self.metrics.prefix_cache_hits.inc()
            return self._clone_entry(entry)

    def lookup_longest_prefix(
        self,
        model_name: str,
        tokens: list[int],
        *,
        min_prefix_tokens: int = 1,
        require_suffix_token: bool = True,
        min_match_ratio: float = 0.0,
    ) -> PrefixEntry | None:
        if not self.settings.prefix_cache_enabled:
            return None
        best: PrefixEntry | None = None
        best_len = -1
        with self._lock:
            for key, entry in self._entries.items():
                if entry.token_count < min_prefix_tokens:
                    continue
                if require_suffix_token and entry.token_count >= len(tokens):
                    continue
                if entry.token_count <= best_len:
                    continue
                if min_match_ratio > 0 and len(tokens) > 0:
                    if (entry.token_count / len(tokens)) < min_match_ratio:
                        continue
                if entry.tokens == tokens[: entry.token_count]:
                    indexed = self._index.get(key)
                    if indexed is not None:
                        indexed.last_used_at = time()
                    best = entry
                    best_len = entry.token_count
            if best is None:
                self.metrics.prefix_cache_misses.inc()
                return None
            self.metrics.prefix_cache_hits.inc()
            return self._clone_entry(best)

    def store(
        self,
        model_name: str,
        tokens: list[int],
        page_ids: list[int],
        approx_bytes: int,
        backend_cache: Any | None = None,
    ) -> PrefixEntry:
        key = prefix_hash(model_name, tokens)
        entry = PrefixEntry(
            key=key,
            token_count=len(tokens),
            tokens=list(tokens),
            page_ids=list(page_ids),
            approx_bytes=approx_bytes,
            backend_cache=copy.deepcopy(backend_cache),
            created_at=time(),
            last_used_at=time(),
        )
        with self._lock:
            old = self._entries.get(key)
            if old is not None:
                self._total_bytes -= old.approx_bytes
            self._entries[key] = entry
            self._index.put(key, entry)
            self._total_bytes += approx_bytes
            self._evict_if_needed()
        return self._clone_entry(entry)

    def maybe_store_prefix_slice(
        self,
        model_name: str,
        tokens: list[int],
        *,
        prefix_tokens: int,
        page_ids: list[int],
        approx_bytes: int,
        backend_cache: Any | None = None,
    ) -> PrefixEntry | None:
        if prefix_tokens <= 0 or prefix_tokens > len(tokens):
            return None
        return self.store(
            model_name,
            tokens[:prefix_tokens],
            page_ids,
            approx_bytes=approx_bytes,
            backend_cache=backend_cache,
        )

    def _clone_entry(self, entry: PrefixEntry) -> PrefixEntry:
        return PrefixEntry(
            key=entry.key,
            token_count=entry.token_count,
            tokens=list(entry.tokens),
            page_ids=list(entry.page_ids),
            approx_bytes=entry.approx_bytes,
            backend_cache=copy.deepcopy(entry.backend_cache),
            created_at=entry.created_at,
            last_used_at=entry.last_used_at,
        )

    def _evict_if_needed(self) -> None:
        while self._total_bytes > self.settings.prefix_cache_max_bytes or len(self._entries) > self.settings.prefix_cache_max_entries:
            victim = self._index.pop_oldest()
            if victim is None:
                break
            key, entry = victim
            self._entries.pop(key, None)
            self._total_bytes -= entry.approx_bytes
