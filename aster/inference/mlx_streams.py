from __future__ import annotations

import importlib
import threading
from collections.abc import Iterable
from typing import Any

_STREAM_REBIND_LOCK = threading.Lock()


def bind_generation_streams(
    module_names: Iterable[str] = ("mlx_lm.generate", "mlx_vlm.generate"),
) -> Any | None:
    """Bind mlx-lm/mlx-vlm module streams to the current worker thread."""
    try:
        import mlx.core as mx
    except Exception:
        return None

    with _STREAM_REBIND_LOCK:
        default_stream = mx.new_stream(mx.default_device())
        mx.set_default_stream(default_stream)
        for module_name in module_names:
            try:
                module = importlib.import_module(module_name)
            except ImportError:
                continue
            if hasattr(module, "generation_stream"):
                module.generation_stream = default_stream
        return default_stream
