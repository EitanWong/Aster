from __future__ import annotations

import copy
from collections import OrderedDict
from typing import Any

DEFAULT_RESPONSES_STORE_MAX_SIZE = 1000


class ResponsesStore:
    def __init__(self, *, max_size: int = DEFAULT_RESPONSES_STORE_MAX_SIZE) -> None:
        self._max_size = max(max_size, 1)
        self._items: OrderedDict[tuple[str, str], list[dict[str, Any]]] = OrderedDict()

    def __len__(self) -> int:
        return len(self._items)

    @property
    def max_size(self) -> int:
        return self._max_size

    def get(self, response_id: str, *, scope: str = "default") -> list[dict[str, Any]] | None:
        key = (scope, response_id)
        messages = self._items.get(key)
        if messages is None:
            return None
        self._items.move_to_end(key)
        return copy.deepcopy(messages)

    def put(
        self, response_id: str, messages: list[dict[str, Any]], *, scope: str = "default"
    ) -> int:
        key = (scope, response_id)
        self._items[key] = copy.deepcopy(messages)
        self._items.move_to_end(key)
        evictions = 0
        while len(self._items) > self._max_size:
            self._items.popitem(last=False)
            evictions += 1
        return evictions
