from __future__ import annotations

import hashlib
import pickle
import time
from bisect import bisect_left, insort
from dataclasses import dataclass, field, replace
from pathlib import Path
from threading import Lock
from typing import Any


@dataclass(slots=True)
class SnapshotEntry:
    key: str
    model_name: str
    model_fingerprint: str | None
    prefix_tokens: tuple[int, ...]
    cache_token_count: int
    prompt_cache: Any
    approx_bytes: int
    created_at: float = field(default_factory=time.monotonic)
    last_used_at: float = field(default_factory=time.monotonic)
    hits: int = 0
    pin_count: int = 0
    match_type: str = "stored"

    @property
    def prefix_token_count(self) -> int:
        return len(self.prefix_tokens)


@dataclass(slots=True)
class PrefixStoreStats:
    lookups: int = 0
    hits: int = 0
    tokens_saved: int = 0
    stores: int = 0
    evictions: int = 0
    evicted_bytes: int = 0
    exact_hits: int = 0
    prefix_hits: int = 0
    lcp_hits: int = 0
    unsafe_lcp_skips: int = 0


class PrefixStore:
    def __init__(
        self,
        *,
        budget_bytes: int,
        max_entries: int,
        min_prefix_tokens: int,
        enabled: bool = True,
    ) -> None:
        self.enabled = enabled
        self.budget_bytes = budget_bytes
        self.max_entries = max_entries
        self.min_prefix_tokens = min_prefix_tokens
        self._lock = Lock()
        self._entries: dict[str, SnapshotEntry] = {}
        self._keys_by_namespace_tokens: dict[tuple[str, str | None, tuple[int, ...]], str] = {}
        self._sorted_tokens_by_namespace: dict[tuple[str, str | None], list[tuple[int, ...]]] = {}
        self._lengths_by_namespace: dict[tuple[str, str | None], list[int]] = {}
        self._length_counts: dict[tuple[str, str | None, int], int] = {}
        self._current_bytes = 0
        self._stats = PrefixStoreStats()
        self._last_match_type: str | None = None

    @property
    def current_bytes(self) -> int:
        return self._current_bytes

    @property
    def entry_count(self) -> int:
        return len(self._entries)

    @property
    def stats(self) -> PrefixStoreStats:
        return PrefixStoreStats(
            lookups=self._stats.lookups,
            hits=self._stats.hits,
            tokens_saved=self._stats.tokens_saved,
            stores=self._stats.stores,
            evictions=self._stats.evictions,
            evicted_bytes=self._stats.evicted_bytes,
            exact_hits=self._stats.exact_hits,
            prefix_hits=self._stats.prefix_hits,
            lcp_hits=self._stats.lcp_hits,
            unsafe_lcp_skips=self._stats.unsafe_lcp_skips,
        )

    @property
    def last_match_type(self) -> str | None:
        return self._last_match_type

    def stats_snapshot(self) -> dict[str, object]:
        with self._lock:
            stats = self.stats
            misses = max(stats.lookups - stats.hits, 0)
            hit_rate = stats.hits / stats.lookups if stats.lookups else 0.0
            pinned_entries = sum(1 for entry in self._entries.values() if entry.pin_count > 0)
            pinned_bytes = sum(
                entry.approx_bytes for entry in self._entries.values() if entry.pin_count > 0
            )
            evictable_entries = len(self._entries) - pinned_entries
            evictable_bytes = max(self._current_bytes - pinned_bytes, 0)
            cached_tokens = sum(entry.cache_token_count for entry in self._entries.values())
            max_entry_bytes = max(
                (entry.approx_bytes for entry in self._entries.values()),
                default=0,
            )
            return {
                "enabled": self.enabled,
                "entries": len(self._entries),
                "pinned_entries": pinned_entries,
                "evictable_entries": evictable_entries,
                "bytes": self._current_bytes,
                "pinned_bytes": pinned_bytes,
                "evictable_bytes": evictable_bytes,
                "budget_bytes": self.budget_bytes,
                "max_entries": self.max_entries,
                "min_prefix_tokens": self.min_prefix_tokens,
                "memory_utilization": (
                    self._current_bytes / self.budget_bytes if self.budget_bytes > 0 else 0.0
                ),
                "cached_tokens": cached_tokens,
                "avg_entry_bytes": (
                    self._current_bytes / len(self._entries) if self._entries else 0.0
                ),
                "max_entry_bytes": max_entry_bytes,
                "lookups": stats.lookups,
                "hits": stats.hits,
                "misses": misses,
                "hit_rate": hit_rate,
                "tokens_saved": stats.tokens_saved,
                "stores": stats.stores,
                "evictions": stats.evictions,
                "evicted_bytes": stats.evicted_bytes,
                "exact_hits": stats.exact_hits,
                "prefix_hits": stats.prefix_hits,
                "lcp_hits": stats.lcp_hits,
                "unsafe_lcp_skips": stats.unsafe_lcp_skips,
                "last_match_type": self._last_match_type,
            }

    def clear(self, *, include_pinned: bool = False) -> dict[str, object]:
        with self._lock:
            before_entries = len(self._entries)
            before_bytes = self._current_bytes
            keys = [
                key
                for key, entry in self._entries.items()
                if include_pinned or entry.pin_count == 0
            ]
            for key in keys:
                self._remove(key)
            return {
                "entries_before": before_entries,
                "bytes_before": before_bytes,
                "entries_cleared": before_entries - len(self._entries),
                "bytes_cleared": max(before_bytes - self._current_bytes, 0),
                "entries_remaining": len(self._entries),
                "bytes_remaining": self._current_bytes,
                "pinned_preserved": sum(1 for entry in self._entries.values() if entry.pin_count > 0),
            }

    def lookup(
        self,
        model_name: str,
        prompt_tokens: list[int],
        *,
        model_fingerprint: str | None = None,
    ) -> SnapshotEntry | None:
        if not self.enabled or len(prompt_tokens) < self.min_prefix_tokens:
            self._last_match_type = None
            return None
        with self._lock:
            self._stats.lookups += 1
            tokens_key = tuple(prompt_tokens)
            namespace = self._namespace(model_name, model_fingerprint)
            sorted_keys = self._sorted_tokens_by_namespace.get(namespace, [])
            if not sorted_keys:
                self._last_match_type = "miss"
                return None

            exact_key = self._keys_by_namespace_tokens.get((*namespace, tokens_key))
            if exact_key is not None:
                entry = self._entries.get(exact_key)
                if entry is not None:
                    self._stats.exact_hits += 1
                    return self._record_hit(entry, "exact")

            index = bisect_left(sorted_keys, tokens_key)
            prefix_entry = self._find_prefix_match(namespace, tokens_key)
            if prefix_entry is not None:
                self._stats.prefix_hits += 1
                return self._record_hit(prefix_entry, "prefix")

            lcp_entry, lcp_len = self._find_lcp_match(namespace, sorted_keys, tokens_key, index)
            if lcp_entry is not None and lcp_len >= self.min_prefix_tokens:
                if self._cache_can_rewind(lcp_entry.prompt_cache):
                    self._stats.lcp_hits += 1
                    matched = replace(
                        lcp_entry,
                        prefix_tokens=tokens_key[:lcp_len],
                        cache_token_count=max(lcp_len - 1, 0),
                    )
                    return self._record_hit(matched, "lcp", touch_key=lcp_entry.key)
                self._stats.unsafe_lcp_skips += 1

            self._last_match_type = "miss"
            return None

    def store(
        self,
        *,
        model_name: str,
        model_fingerprint: str | None = None,
        prefix_tokens: list[int],
        cache_token_count: int,
        prompt_cache: Any,
        approx_bytes: int,
    ) -> SnapshotEntry | None:
        if not self.enabled:
            return None
        if len(prefix_tokens) < self.min_prefix_tokens:
            return None
        key = self._digest(model_name, model_fingerprint, prefix_tokens)
        entry = SnapshotEntry(
            key=key,
            model_name=model_name,
            model_fingerprint=model_fingerprint,
            prefix_tokens=tuple(prefix_tokens),
            cache_token_count=cache_token_count,
            prompt_cache=prompt_cache,
            approx_bytes=max(int(approx_bytes), 0),
        )
        with self._lock:
            previous = self._entries.get(key)
            if previous is not None:
                entry.pin_count = previous.pin_count
                entry.hits = previous.hits
                entry.created_at = previous.created_at
                self._current_bytes -= previous.approx_bytes
                self._remove_length(
                    self._namespace(previous.model_name, previous.model_fingerprint),
                    previous.prefix_token_count,
                )
                self._remove_token_key(
                    self._namespace(previous.model_name, previous.model_fingerprint),
                    previous.prefix_tokens,
                )
            self._entries[key] = entry
            self._current_bytes += entry.approx_bytes
            namespace = self._namespace(model_name, model_fingerprint)
            self._record_length(namespace, entry.prefix_token_count)
            self._record_token_key(namespace, entry.prefix_tokens, key)
            self._stats.stores += 1
            self._evict_if_needed()
            return self._entries.get(key)

    def pin(self, key: str | None) -> None:
        if key is None:
            return
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None:
                entry.pin_count += 1

    def unpin(self, key: str | None) -> None:
        if key is None:
            return
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None and entry.pin_count > 0:
                entry.pin_count -= 1

    def evict_until_below(self, target_bytes: int) -> int:
        evicted = 0
        with self._lock:
            while self._current_bytes > target_bytes:
                victim = self._select_victim()
                if victim is None:
                    break
                self._remove(victim.key)
                evicted += 1
        return evicted

    def save_to_disk(self, path: str | Path) -> int:
        if not self.enabled:
            return 0
        target = Path(path).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        serializable_entries: list[SnapshotEntry] = []
        with self._lock:
            for entry in self._entries.values():
                candidate = replace(entry, pin_count=0)
                try:
                    pickle.dumps(candidate)
                except Exception:
                    continue
                serializable_entries.append(candidate)
        payload = {
            "version": 1,
            "entries": serializable_entries,
        }
        tmp_path = target.with_suffix(f"{target.suffix}.tmp")
        tmp_path.write_bytes(pickle.dumps(payload))
        tmp_path.replace(target)
        return len(serializable_entries)

    def load_from_disk(
        self,
        path: str | Path,
        *,
        model_name: str | None = None,
        model_fingerprint: str | None = None,
        clear: bool = False,
    ) -> int:
        if not self.enabled:
            return 0
        source = Path(path).expanduser()
        if not source.exists():
            return 0
        payload = pickle.loads(source.read_bytes())
        if not isinstance(payload, dict) or payload.get("version") != 1:
            raise ValueError(f"Unsupported prefix cache payload: {source}")
        entries = payload.get("entries")
        if not isinstance(entries, list):
            raise ValueError(f"Invalid prefix cache payload: {source}")

        loaded = 0
        with self._lock:
            if clear:
                self._clear_locked()
            for raw_entry in entries:
                if not isinstance(raw_entry, SnapshotEntry):
                    continue
                if model_name is not None and raw_entry.model_name != model_name:
                    continue
                if model_fingerprint is not None and raw_entry.model_fingerprint != model_fingerprint:
                    continue
                entry = replace(raw_entry, pin_count=0)
                previous = self._entries.get(entry.key)
                if previous is not None:
                    self._remove(previous.key)
                self._entries[entry.key] = entry
                self._current_bytes += entry.approx_bytes
                namespace = self._namespace(entry.model_name, entry.model_fingerprint)
                self._record_length(namespace, entry.prefix_token_count)
                self._record_token_key(namespace, entry.prefix_tokens, entry.key)
                loaded += 1
            self._evict_if_needed()
        return loaded

    def _evict_if_needed(self) -> None:
        while self._current_bytes > self.budget_bytes or len(self._entries) > self.max_entries:
            victim = self._select_victim()
            if victim is None:
                break
            self._remove(victim.key)

    def _select_victim(self) -> SnapshotEntry | None:
        victims = [entry for entry in self._entries.values() if entry.pin_count == 0]
        if not victims:
            return None
        victims.sort(key=lambda item: (item.last_used_at, -item.approx_bytes))
        return victims[0]

    def _remove(self, key: str) -> None:
        entry = self._entries.pop(key, None)
        if entry is not None:
            self._current_bytes -= entry.approx_bytes
            namespace = self._namespace(entry.model_name, entry.model_fingerprint)
            self._remove_length(namespace, entry.prefix_token_count)
            self._remove_token_key(namespace, entry.prefix_tokens)
            self._stats.evictions += 1
            self._stats.evicted_bytes += entry.approx_bytes

    def _clear_locked(self) -> None:
        self._entries.clear()
        self._keys_by_namespace_tokens.clear()
        self._sorted_tokens_by_namespace.clear()
        self._lengths_by_namespace.clear()
        self._length_counts.clear()
        self._current_bytes = 0

    def _record_hit(
        self,
        entry: SnapshotEntry,
        match_type: str,
        *,
        touch_key: str | None = None,
    ) -> SnapshotEntry:
        stored = self._entries.get(touch_key or entry.key)
        if stored is not None:
            stored.hits += 1
            stored.last_used_at = time.monotonic()
        self._stats.hits += 1
        self._stats.tokens_saved += max(entry.cache_token_count, 0)
        self._last_match_type = match_type
        return replace(entry, match_type=match_type)

    def _find_prefix_match(
        self,
        namespace: tuple[str, str | None],
        tokens_key: tuple[int, ...],
    ) -> SnapshotEntry | None:
        lengths = self._lengths_by_namespace.get(namespace, [])
        for prefix_length in reversed(lengths):
            if prefix_length >= len(tokens_key):
                continue
            candidate = tokens_key[:prefix_length]
            key = self._keys_by_namespace_tokens.get((*namespace, candidate))
            if key is not None:
                return self._entries.get(key)
        return None

    def _find_lcp_match(
        self,
        namespace: tuple[str, str | None],
        sorted_keys: list[tuple[int, ...]],
        tokens_key: tuple[int, ...],
        index: int,
    ) -> tuple[SnapshotEntry | None, int]:
        best_entry: SnapshotEntry | None = None
        best_len = 0
        for candidate_index in (index - 1, index, index + 1):
            if candidate_index < 0 or candidate_index >= len(sorted_keys):
                continue
            candidate = sorted_keys[candidate_index]
            if candidate == tokens_key:
                continue
            lcp_len = self._common_prefix_len(candidate, tokens_key)
            if lcp_len <= best_len:
                continue
            key = self._keys_by_namespace_tokens.get((*namespace, candidate))
            if key is None:
                continue
            entry = self._entries.get(key)
            if entry is None or entry.cache_token_count < max(lcp_len - 1, 0):
                continue
            best_entry = entry
            best_len = lcp_len
        return best_entry, best_len

    def _record_length(self, namespace: tuple[str, str | None], prefix_length: int) -> None:
        key = (*namespace, prefix_length)
        count = self._length_counts.get(key, 0)
        self._length_counts[key] = count + 1
        if count > 0:
            return
        lengths = self._lengths_by_namespace.setdefault(namespace, [])
        index = bisect_left(lengths, prefix_length)
        if index >= len(lengths) or lengths[index] != prefix_length:
            insort(lengths, prefix_length)

    def _remove_length(self, namespace: tuple[str, str | None], prefix_length: int) -> None:
        key = (*namespace, prefix_length)
        count = self._length_counts.get(key, 0)
        if count <= 1:
            self._length_counts.pop(key, None)
        else:
            self._length_counts[key] = count - 1
            return
        lengths = self._lengths_by_namespace.get(namespace)
        if not lengths:
            return
        index = bisect_left(lengths, prefix_length)
        if index < len(lengths) and lengths[index] == prefix_length:
            lengths.pop(index)
        if not lengths:
            self._lengths_by_namespace.pop(namespace, None)

    def _record_token_key(
        self,
        namespace: tuple[str, str | None],
        tokens: tuple[int, ...],
        key: str,
    ) -> None:
        self._keys_by_namespace_tokens[(*namespace, tokens)] = key
        sorted_keys = self._sorted_tokens_by_namespace.setdefault(namespace, [])
        index = bisect_left(sorted_keys, tokens)
        if index >= len(sorted_keys) or sorted_keys[index] != tokens:
            sorted_keys.insert(index, tokens)

    def _remove_token_key(
        self,
        namespace: tuple[str, str | None],
        tokens: tuple[int, ...],
    ) -> None:
        self._keys_by_namespace_tokens.pop((*namespace, tokens), None)
        sorted_keys = self._sorted_tokens_by_namespace.get(namespace)
        if not sorted_keys:
            return
        index = bisect_left(sorted_keys, tokens)
        if index < len(sorted_keys) and sorted_keys[index] == tokens:
            sorted_keys.pop(index)
        if not sorted_keys:
            self._sorted_tokens_by_namespace.pop(namespace, None)

    @staticmethod
    def _common_prefix_len(left: tuple[int, ...], right: tuple[int, ...]) -> int:
        limit = min(len(left), len(right))
        index = 0
        while index < limit and left[index] == right[index]:
            index += 1
        return index

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

    @staticmethod
    def _namespace(model_name: str, model_fingerprint: str | None) -> tuple[str, str | None]:
        return (model_name, model_fingerprint)

    @staticmethod
    def _digest(model_name: str, model_fingerprint: str | None, tokens: list[int]) -> str:
        hasher = hashlib.sha256()
        hasher.update(model_name.encode("utf-8"))
        hasher.update(b"\0")
        if model_fingerprint is not None:
            hasher.update(model_fingerprint.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(",".join(str(token) for token in tokens).encode("utf-8"))
        return hasher.hexdigest()
