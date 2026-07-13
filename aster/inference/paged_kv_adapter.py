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


class _PagedKVBlockPool:
    """Persistent MLX storage for one full-attention layer."""

    def __init__(self, manager: PagedCacheManager) -> None:
        self.manager = manager
        self.keys: Any | None = None
        self.values: Any | None = None
        self._capacity = 0
        self._shape: tuple[int, int, int, int, int] | None = None
        self._written_tokens: dict[int, int] = {}

    def ensure_capacity(self, block_id: int, keys: Any, values: Any) -> None:
        mx = _mlx()
        B, H, _, key_dim = keys.shape
        block_size = self.manager.block_size
        value_dim = values.shape[3]
        shape = (B, H, block_size, key_dim, value_dim)
        if self._shape is None:
            self._shape = shape
            initial = max(8, 1 << max(block_id, 1).bit_length())
            self._capacity = min(max(initial, block_id + 1), self.manager.max_blocks)
            self.keys = mx.zeros(
                (self._capacity, B, H, block_size, key_dim), dtype=keys.dtype
            )
            self.values = mx.zeros(
                (self._capacity, B, H, block_size, value_dim), dtype=values.dtype
            )
            return

        if shape != self._shape:
            raise ValueError("Paged KV pool shape changed after initialization")
        if block_id < self._capacity:
            return
        if self.keys is None or self.values is None:
            raise RuntimeError("Paged KV pool is not initialized")
        new_capacity = min(
            max(block_id + 1, self._capacity * 2),
            self.manager.max_blocks,
        )
        if new_capacity <= block_id:
            raise MemoryError("Paged KV pool exhausted")
        extra = new_capacity - self._capacity
        self.keys = mx.concatenate(
            [self.keys, mx.zeros((extra, B, H, block_size, key_dim), dtype=keys.dtype)],
            axis=0,
        )
        self.values = mx.concatenate(
            [
                self.values,
                mx.zeros((extra, B, H, block_size, value_dim), dtype=values.dtype),
            ],
            axis=0,
        )
        self._capacity = new_capacity

    def copy_block(self, source_id: int, target_id: int) -> None:
        if self.keys is None or self.values is None:
            raise RuntimeError("Cannot copy an uninitialized paged KV pool")
        self.keys[target_id] = self.keys[source_id]
        self.values[target_id] = self.values[source_id]
        self._written_tokens[target_id] = self._written_tokens.get(source_id, 0)

    def reset_block(self, block_id: int) -> None:
        self._written_tokens.pop(block_id, None)

    def write(self, block_id: int, block_offset: int, keys: Any, values: Any) -> None:
        if self.keys is None or self.values is None:
            raise RuntimeError("Cannot write an uninitialized paged KV pool")
        end = block_offset + int(keys.shape[2])
        self.keys[block_id, ..., block_offset:end, :] = keys
        self.values[block_id, ..., block_offset:end, :] = values
        self._written_tokens[block_id] = end

    def block_token_count(self, block_id: int) -> int:
        return self._written_tokens.get(block_id, 0)

    def block_pool(self) -> tuple[Any, Any]:
        if self.keys is None or self.values is None:
            raise ValueError("Cannot expose an empty paged KV pool")
        return self.keys, self.values

    @property
    def nbytes(self) -> int:
        if self.keys is None or self.values is None:
            return 0
        return _nbytes(self.keys) + _nbytes(self.values)


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
        pool: _PagedKVBlockPool | None = None,
    ) -> None:
        self.manager = manager
        self.layer_index = layer_index
        self.block_table = block_table or manager.create_table(
            request_id or f"paged-layer-{next(_REQUEST_IDS)}"
        )
        self.offset = 0
        self.step = max(manager.block_size, 1)
        self._pool = pool or _PagedKVBlockPool(manager)

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
            self._pool.ensure_capacity(block_id, keys, values)
            cow_source = self.manager.cow_source(block_id)
            if self._pool.block_token_count(block_id) == 0 and cow_source is not None:
                self._pool.copy_block(cow_source, block_id)
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
            block_id = self.manager.allocate_block()
            self._pool.reset_block(block_id)
            self.block_table.append_block(block_id)

    def _write_segment(
        self,
        block: Any,
        block_offset: int,
        segment: tuple[Any, Any],
    ) -> None:
        old_tokens = self._pool.block_token_count(block.block_id)
        if block_offset != old_tokens:
            raise ValueError("Paged KV writes must append within a block")
        self._pool.write(block.block_id, block_offset, segment[0], segment[1])
        if len(block.cache_data) < self.manager.num_layers:
            block.cache_data.extend([None] * (self.manager.num_layers - len(block.cache_data)))
        block.cache_data[self.layer_index] = (self._pool.keys[block.block_id], self._pool.values[block.block_id])
        block.token_count = max(block.token_count, block_offset + int(segment[0].shape[2]))

    def _materialize(self) -> tuple[Any, Any]:
        if self.offset == 0:
            raise ValueError("Cannot materialize an empty paged KV cache")
        pool_keys, pool_values = self._pool.block_pool()
        keys: list[Any] = []
        values: list[Any] = []
        position = 0
        for block_id in self.block_table.block_ids:
            take = min(self.block_size, self.offset - position)
            keys.append(pool_keys[block_id, ..., :take, :])
            values.append(pool_values[block_id, ..., :take, :])
            position += take
        mx = _mlx()
        return (
            mx.concatenate(keys, axis=2)[..., : self.offset, :],
            mx.concatenate(values, axis=2)[..., : self.offset, :],
        )

    def _block_pool(self) -> tuple[Any, Any, Any]:
        if self.offset == 0 or not self.block_table.block_ids:
            raise ValueError("Cannot build a block pool from an empty paged cache")
        mx = _mlx()
        pool_keys, pool_values = self._pool.block_pool()
        return (
            pool_keys,
            pool_values,
            mx.array(self.block_table.block_ids, dtype=mx.uint32),
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
            pool=self._pool,
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
        return self._pool.nbytes

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
