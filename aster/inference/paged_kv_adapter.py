from __future__ import annotations

from dataclasses import dataclass
from itertools import count
from typing import Any

from aster.inference.paged_cache import BlockTable, PagedCacheManager


_REQUEST_IDS = count()


def _mlx() -> Any:
    try:
        import mlx.core as mx
    except Exception as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("MLX is required by PagedKVCacheLayer") from exc
    return mx


def _nbytes(value: Any) -> int:
    value_nbytes = getattr(value, "nbytes", None)
    return int(value_nbytes) if isinstance(value_nbytes, int) else 0


@dataclass(frozen=True, slots=True)
class PagedAttentionView:
    """Block metadata plus a contiguous fallback for current MLX attention."""

    layer: PagedKVCacheLayer
    block_ids: tuple[int, ...]
    block_size: int
    sequence_length: int

    def materialize(self) -> tuple[Any, Any]:
        return self.layer._materialize()

    def block_pool(self) -> tuple[Any, Any, Any]:
        """Return a packed pool and logical-to-physical indices for Metal."""
        return self.layer._block_pool()

    def attention(self, queries: Any, *, scale: float) -> Any:
        """Run the experimental block-indexed Metal attention entry point."""
        from aster.inference.metal_paged_attention import paged_block_attention

        key_pool, value_pool, block_indices = self.block_pool()
        query_tokens = int(queries.shape[2])
        return paged_block_attention(
            queries,
            key_pool,
            value_pool,
            block_indices,
            query_offset=self.sequence_length - query_tokens,
            total_kv_tokens=self.sequence_length,
            scale=scale,
        )


class PagedKVCacheLayer:
    """Lossless block-backed cache implementing MLX-LM's KV cache contract.

    The block table is the future custom-attention boundary. Until MLX exposes
    a block-indexed attention API, ``update_and_fetch`` returns a contiguous
    materialization so native model code remains unchanged.
    """

    step = 256

    def __init__(
        self,
        manager: PagedCacheManager,
        *,
        layer_index: int,
        block_table: BlockTable | None = None,
        request_id: str | None = None,
    ) -> None:
        self.manager = manager
        self.layer_index = layer_index
        self.block_table = block_table or manager.create_table(
            request_id or f"paged-layer-{next(_REQUEST_IDS)}"
        )
        self.offset = 0
        self.step = max(manager.block_size, 1)

    @property
    def block_size(self) -> int:
        return self.manager.block_size

    def update_and_fetch(self, keys: Any, values: Any) -> tuple[Any, Any]:
        self._validate_inputs(keys, values)
        token_count = int(keys.shape[2])
        start = self.offset
        end = start + token_count
        position = 0

        while position < token_count:
            logical_index = (start + position) // self.block_size
            block_offset = (start + position) % self.block_size
            self._ensure_table_block(logical_index)
            block_id = self.manager.ensure_writable_block(
                self.block_table, logical_index
            )
            block = self.manager.get_block(block_id)
            take = min(self.block_size - block_offset, token_count - position)
            segment = (
                keys[..., position : position + take, :],
                values[..., position : position + take, :],
            )
            self._write_segment(block, block_offset, segment)
            position += take

        self.offset = end
        self.block_table.num_tokens = max(self.block_table.num_tokens, end)
        return self._materialize()

    def _validate_inputs(self, keys: Any, values: Any) -> None:
        key_shape = getattr(keys, "shape", None)
        value_shape = getattr(values, "shape", None)
        if key_shape is None or value_shape is None or len(key_shape) != 4 or len(value_shape) != 4:
            raise ValueError("Paged KV cache expects rank-4 key and value tensors")
        if key_shape[:3] != value_shape[:3] or key_shape[2] <= 0:
            raise ValueError("Key/value tensors have incompatible shapes")

    def _ensure_table_block(self, logical_index: int) -> None:
        while len(self.block_table.block_ids) <= logical_index:
            self.block_table.append_block(self.manager.allocate_block())

    def _write_segment(
        self,
        block: Any,
        block_offset: int,
        segment: tuple[Any, Any],
    ) -> None:
        if len(block.cache_data) < self.manager.num_layers:
            block.cache_data.extend([None] * (self.manager.num_layers - len(block.cache_data)))
        previous = block.cache_data[self.layer_index]
        if previous is None:
            block.cache_data[self.layer_index] = segment
        else:
            old_keys, old_values = previous
            old_tokens = int(old_keys.shape[2])
            if block_offset == old_tokens:
                block.cache_data[self.layer_index] = (
                    _mlx().concatenate([old_keys, segment[0]], axis=2),
                    _mlx().concatenate([old_values, segment[1]], axis=2),
                )
            else:
                raise ValueError("Paged KV writes must append within a block")
        block.token_count = max(block.token_count, block_offset + int(segment[0].shape[2]))

    def _materialize(self) -> tuple[Any, Any]:
        if self.offset == 0:
            raise ValueError("Cannot materialize an empty paged KV cache")
        keys: list[Any] = []
        values: list[Any] = []
        for block_id in self.block_table.block_ids:
            block = self.manager.get_block(block_id)
            if self.layer_index >= len(block.cache_data):
                raise ValueError(f"Block {block_id} has no layer {self.layer_index} data")
            layer_data = block.cache_data[self.layer_index]
            if layer_data is None:
                raise ValueError(f"Block {block_id} has incomplete layer data")
            keys.append(layer_data[0])
            values.append(layer_data[1])
        mx = _mlx()
        return (
            mx.concatenate(keys, axis=2)[..., : self.offset, :],
            mx.concatenate(values, axis=2)[..., : self.offset, :],
        )

    def _block_pool(self) -> tuple[Any, Any, Any]:
        if self.offset == 0 or not self.block_table.block_ids:
            raise ValueError("Cannot build a block pool from an empty paged cache")
        mx = _mlx()
        keys: list[Any] = []
        values: list[Any] = []
        for block_id in self.block_table.block_ids:
            block = self.manager.get_block(block_id)
            if self.layer_index >= len(block.cache_data):
                raise ValueError(f"Block {block_id} has no layer {self.layer_index} data")
            layer_data = block.cache_data[self.layer_index]
            if layer_data is None:
                raise ValueError(f"Block {block_id} has incomplete layer data")
            block_keys, block_values = layer_data
            padding = self.block_size - int(block_keys.shape[2])
            if padding > 0:
                pad = [(0, 0), (0, 0), (0, padding), (0, 0)]
                block_keys = mx.pad(block_keys, pad)
                block_values = mx.pad(block_values, pad)
            keys.append(block_keys)
            values.append(block_values)
        return (
            mx.stack(keys, axis=0),
            mx.stack(values, axis=0),
            mx.arange(len(keys), dtype=mx.uint32),
        )

    @property
    def state(self) -> tuple[Any, Any]:
        return self._materialize()

    @state.setter
    def state(self, value: tuple[Any, Any]) -> None:
        keys, values = value
        self._reset_owned_blocks()
        self.offset = 0
        self.update_and_fetch(keys, values)

    def _reset_owned_blocks(self) -> None:
        for block_id in self.block_table.block_ids:
            self.manager.free_block(block_id)
        self.block_table.block_ids.clear()
        self.block_table.num_tokens = 0

    def fork(self) -> PagedKVCacheLayer:
        table = self.manager.fork_table(
            self.block_table,
            f"paged-layer-{next(_REQUEST_IDS)}",
        )
        forked = PagedKVCacheLayer(
            self.manager,
            layer_index=self.layer_index,
            block_table=table,
        )
        forked.offset = self.offset
        return forked

    def size(self) -> int:
        return self.offset

    def is_trimmable(self) -> bool:
        return True

    def trim(self, num_tokens: int) -> int:
        trimmed = min(max(int(num_tokens), 0), self.offset)
        self.offset -= trimmed
        return trimmed

    def make_mask(self, *args: Any, **kwargs: Any) -> Any:
        from mlx_lm.models.cache import create_attention_mask

        return create_attention_mask(*args, offset=self.offset, **kwargs)

    @property
    def meta_state(self) -> tuple[str, str, str]:
        return (str(self.offset), str(self.layer_index), str(self.block_size))

    @meta_state.setter
    def meta_state(self, value: tuple[str, str, str]) -> None:
        self.offset = int(value[0])

    @classmethod
    def merge(cls, caches: list[PagedKVCacheLayer]) -> Any:
        """Use MLX-LM's native batch cache until paged batch attention exists."""
        from mlx_lm.models.cache import BatchKVCache, KVCache

        native_caches = []
        for cache in caches:
            native = KVCache()
            native.keys, native.values = cache.state
            native.offset = cache.offset
            native.step = cache.step
            native_caches.append(native)
        return BatchKVCache.merge(native_caches)

    def empty(self) -> bool:
        return self.offset == 0

    @property
    def nbytes(self) -> int:
        total = 0
        for block_id in self.block_table.block_ids:
            block = self.manager.get_block(block_id)
            if self.layer_index < len(block.cache_data):
                layer_data = block.cache_data[self.layer_index]
                if layer_data is not None:
                    total += _nbytes(layer_data[0]) + _nbytes(layer_data[1])
        return total

    def attention_view(self) -> PagedAttentionView:
        return PagedAttentionView(
            layer=self,
            block_ids=tuple(self.block_table.block_ids),
            block_size=self.block_size,
            sequence_length=self.offset,
        )


def replace_kv_cache_layers(
    prompt_cache: list[Any],
    manager: PagedCacheManager,
    *,
    kv_cache_type: type[Any],
    request_id: str,
) -> list[Any]:
    """Replace native sequence KV layers while preserving recurrent layers."""
    table = manager.create_table(request_id)
    return [
        PagedKVCacheLayer(manager, layer_index=index, block_table=table)
        if isinstance(cache, kv_cache_type)
        else cache
        for index, cache in enumerate(prompt_cache)
    ]


__all__ = [
    "PagedAttentionView",
    "PagedKVCacheLayer",
    "replace_kv_cache_layers",
]
