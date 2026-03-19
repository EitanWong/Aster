from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock

from aster.core.config import CacheSettings
from aster.telemetry.metrics import MetricsRegistry


@dataclass(slots=True)
class KVPage:
    page_id: int
    token_capacity: int
    owner_request_id: str | None = None
    used_tokens: int = 0


@dataclass(slots=True)
class RequestPageMap:
    request_id: str
    page_ids: list[int] = field(default_factory=list)  # type: ignore[assignment]


class PagedKVCache:
    def __init__(self, settings: CacheSettings, metrics: MetricsRegistry) -> None:
        self.settings = settings
        self.metrics = metrics
        self._lock = Lock()
        self._pages = [KVPage(i, settings.kv_page_tokens) for i in range(settings.kv_max_pages)]
        self._free_pages = list(range(settings.kv_max_pages))
        self._request_map: dict[str, RequestPageMap] = {}
        self.metrics.kv_pages_used.set(0)

    def allocate(self, request_id: str, tokens: int) -> list[int]:
        needed = max(1, (tokens + self.settings.kv_page_tokens - 1) // self.settings.kv_page_tokens)
        with self._lock:
            if needed > len(self._free_pages):
                self.evict_unowned(max(needed - len(self._free_pages), 0))
            if needed > len(self._free_pages):
                raise MemoryError("Insufficient KV pages")
            page_ids: list[int] = []
            for _ in range(needed):
                page_id: int = self._free_pages.pop()
                page = self._pages[page_id]
                page.owner_request_id = request_id
                page_ids.append(page_id)
            self._request_map[request_id] = RequestPageMap(request_id, page_ids)
            self.metrics.kv_pages_used.set(self.settings.kv_max_pages - len(self._free_pages))
            return page_ids

    def release(self, request_id: str) -> None:
        with self._lock:
            mapping = self._request_map.pop(request_id, None)
            if mapping is None:
                return
            for page_id in mapping.page_ids:
                page = self._pages[page_id]
                page.owner_request_id = None
                page.used_tokens = 0
                self._free_pages.append(page_id)
            self.metrics.kv_pages_used.set(self.settings.kv_max_pages - len(self._free_pages))

    def evict_unowned(self, count: int) -> None:
        _ = count
