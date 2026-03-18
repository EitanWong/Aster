from __future__ import annotations

from collections import OrderedDict
from typing import TypeVar

K = TypeVar("K")
V = TypeVar("V")


class LRUEvictionIndex[K, V]:
    def __init__(self) -> None:
        self._items: OrderedDict[K, V] = OrderedDict()

    def put(self, key: K, value: V) -> None:
        self._items[key] = value
        self._items.move_to_end(key)

    def get(self, key: K) -> V | None:
        value = self._items.get(key)
        if value is not None:
            self._items.move_to_end(key)
        return value

    def pop_oldest(self) -> tuple[K, V] | None:
        if not self._items:
            return None
        return self._items.popitem(last=False)

    def remove(self, key: K) -> None:
        self._items.pop(key, None)

    def __contains__(self, key: K) -> bool:
        return key in self._items
