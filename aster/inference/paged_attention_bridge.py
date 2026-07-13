from __future__ import annotations

from collections.abc import Callable
from typing import Any

from aster.inference.paged_kv_adapter import PagedKVCacheLayer


def _is_causal_mask(mask: Any) -> bool:
    return mask is None or (isinstance(mask, str) and mask == "causal")


def dispatch_paged_attention(
    native_attention: Callable[..., Any],
    queries: Any,
    keys: Any,
    values: Any,
    *,
    cache: Any,
    scale: float,
    mask: Any,
    sinks: Any | None = None,
) -> Any:
    """Select direct pool attention only for a compatible paged cache call."""
    if not isinstance(cache, PagedKVCacheLayer) or not cache.direct_attention_enabled:
        return native_attention(
            queries,
            keys,
            values,
            cache=cache,
            scale=scale,
            mask=mask,
            sinks=sinks,
        )

    if (
        sinks is not None
        or not _is_causal_mask(mask)
        or int(queries.shape[2]) > 8
    ):
        dense_keys, dense_values = cache.state
        return native_attention(
            queries,
            dense_keys,
            dense_values,
            cache=cache,
            scale=scale,
            mask=mask,
            sinks=sinks,
        )

    return cache.attention_view().attention(queries, scale=scale)


def install_qwen3_next_paged_attention_bridge() -> None:
    """Install the opt-in bridge used by Qwen3.5 full-attention layers."""
    import mlx_lm.models.qwen3_next as qwen3_next

    if getattr(qwen3_next, "_aster_paged_attention_bridge", False):
        return

    native_attention = qwen3_next.scaled_dot_product_attention

    def bridged_attention(
        queries: Any,
        keys: Any,
        values: Any,
        *,
        cache: Any,
        scale: float,
        mask: Any,
        sinks: Any | None = None,
    ) -> Any:
        return dispatch_paged_attention(
            native_attention,
            queries,
            keys,
            values,
            cache=cache,
            scale=scale,
            mask=mask,
            sinks=sinks,
        )

    qwen3_next.scaled_dot_product_attention = bridged_attention
    qwen3_next._aster_paged_attention_bridge = True


__all__ = [
    "dispatch_paged_attention",
    "install_qwen3_next_paged_attention_bridge",
]
