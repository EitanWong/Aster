from __future__ import annotations

from typing import Any


def prompt_cache_length(prompt_cache: Any | None) -> int:
    if prompt_cache is None:
        return 0

    # Newer/other mlx-lm variants may expose a helper or direct attr on the cache container.
    for attr in ("cache_length", "length", "size"):
        value: Any = getattr(prompt_cache, attr, None)
        if isinstance(value, int):
            return max(0, value)
        if callable(value):
            try:
                result: Any = value()
            except TypeError:
                result = None
            if isinstance(result, int):
                return max(0, result)

    # Common mlx-lm shape: prompt_cache is a list of per-layer cache objects.
    try:
        if len(prompt_cache) == 0:
            return 0
        first: Any = prompt_cache[0]
    except Exception:
        return 0

    lengths: Any = getattr(first, "lengths", None)
    if lengths is not None:
        try:
            if hasattr(lengths, "tolist"):
                values: Any = lengths.tolist()
            else:
                values = list(lengths)
            if values:
                return int(max(values))
        except Exception:
            pass

    meta_state: Any = getattr(first, "meta_state", None)
    if meta_state is not None:
        try:
            if isinstance(meta_state, dict):
                for key in ("length", "lengths", "size"):
                    value = meta_state.get(key)
                    if isinstance(value, int):
                        return max(0, value)
        except Exception:
            pass

    return 0
