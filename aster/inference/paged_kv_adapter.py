from __future__ import annotations

import copy
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
        self._owners = 1

    def retain(self) -> None:
        if self._owners == 0:
            raise RuntimeError("Cannot retain a released paged KV pool")
        self._owners += 1

    def release(self) -> None:
        if self._owners == 0:
            return
        self._owners -= 1
        if self._owners == 0:
            self.keys = None
            self.values = None
            self._capacity = 0
            self._shape = None
            self._written_tokens.clear()

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
            self.keys = mx.zeros((self._capacity, B, H, block_size, key_dim), dtype=keys.dtype)
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

    def truncate(self, block_id: int, token_count: int) -> None:
        if token_count <= 0:
            self._written_tokens.pop(block_id, None)
        else:
            self._written_tokens[block_id] = token_count

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


@dataclass(frozen=True, slots=True)
class PagedBatchAttentionView:
    """Borrow one physical pool through per-request logical block tables."""

    views: tuple[PagedAttentionView, ...]
    block_ids: tuple[tuple[int, ...], ...]
    block_tables: Any
    sequence_lengths: Any
    block_size: int

    @classmethod
    def from_views(
        cls, views: tuple[PagedAttentionView, ...] | list[PagedAttentionView]
    ) -> PagedBatchAttentionView:
        normalized = tuple(views)
        if not normalized:
            raise ValueError("Paged batch attention requires at least one view")

        first = normalized[0]
        pool = first.layer._pool
        layer_index = first.layer.layer_index
        block_size = first.block_size
        rows: list[tuple[int, ...]] = []
        lengths: list[int] = []
        for view in normalized:
            if view.layer._pool is not pool:
                raise ValueError("Paged batch attention views must share one physical pool")
            if view.layer.layer_index != layer_index or view.block_size != block_size:
                raise ValueError("Paged batch attention views have incompatible layer metadata")
            if view.sequence_length <= 0 or not view.block_ids:
                raise ValueError("Paged batch attention views must contain cached tokens")
            required_blocks = (view.sequence_length + block_size - 1) // block_size
            if len(view.block_ids) != required_blocks:
                raise ValueError("Paged batch attention block table does not cover its sequence")
            rows.append(view.block_ids)
            lengths.append(view.sequence_length)

        max_blocks = max(len(row) for row in rows)
        padded = [row + (0,) * (max_blocks - len(row)) for row in rows]
        mx = _mlx()
        return cls(
            views=normalized,
            block_ids=tuple(rows),
            block_tables=mx.array(padded, dtype=mx.uint32),
            sequence_lengths=mx.array(lengths, dtype=mx.uint32),
            block_size=block_size,
        )

    @property
    def metadata_nbytes(self) -> int:
        return _nbytes(self.block_tables) + _nbytes(self.sequence_lengths)

    def attention(self, queries: Any, *, scale: float) -> Any:
        from aster.inference.metal_paged_attention import paged_batch_block_attention

        query_shape = getattr(queries, "shape", None)
        if query_shape is None or len(query_shape) != 4:
            raise ValueError("Paged batch attention expects rank-4 queries")
        if int(query_shape[0]) != len(self.views):
            raise ValueError("Query batch does not match paged batch metadata")
        query_tokens = int(query_shape[2])
        if query_tokens <= 0 or any(length < query_tokens for length in self._lengths):
            raise ValueError("Query length exceeds a paged sequence")

        pool = self.views[0].layer._pool
        key_pool, value_pool = pool.block_pool()
        for view, block_ids, sequence_length in zip(
            self.views, self.block_ids, self._lengths, strict=True
        ):
            if view.layer._pool is not pool:
                raise RuntimeError("Paged batch attention pool ownership changed")
            if (
                tuple(view.layer.block_table.block_ids) != block_ids
                or view.layer.offset != sequence_length
            ):
                raise RuntimeError("Paged batch attention metadata is stale")

        return paged_batch_block_attention(
            queries,
            key_pool,
            value_pool,
            self.block_tables,
            self.sequence_lengths,
            scale=scale,
        )

    @property
    def _lengths(self) -> tuple[int, ...]:
        return tuple(view.sequence_length for view in self.views)


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
        enable_block_pool: bool = True,
        enable_direct_attention: bool = False,
    ) -> None:
        self.manager = manager
        self.layer_index = layer_index
        self.block_table = block_table or manager.create_table(
            request_id or f"paged-layer-{next(_REQUEST_IDS)}"
        )
        self.offset = 0
        self.step = max(manager.block_size, 1)
        if pool is None:
            self._pool = _PagedKVBlockPool(manager)
        else:
            pool.retain()
            self._pool = pool
        self.direct_attention_enabled = enable_direct_attention
        self._pool_enabled = enable_block_pool
        self._materialized_keys: Any | None = None
        self._materialized_values: Any | None = None
        self._materialized_capacity = 0
        self._storage_written_tokens: dict[int, int] = {}
        self._block_indices: Any | None = None
        self._block_indices_ids: tuple[int, ...] = ()

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
            old_block_id = self.block_table.block_ids[logical_index]
            block_id = self.manager.ensure_writable_block(self.block_table, logical_index)
            if block_id != old_block_id:
                self._block_indices = None
            if not self._pool_enabled:
                cow_source = self.manager.cow_source(block_id)
                if cow_source is not None and block_id not in self._storage_written_tokens:
                    self._storage_written_tokens[block_id] = self._storage_written_tokens.get(
                        cow_source, 0
                    )
            if self._pool_enabled:
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

        if not self.direct_attention_enabled or not self._pool_enabled:
            self._update_materialized(keys, values, start, end)
        self.offset = end
        self.block_table.num_tokens = max(self.block_table.num_tokens, end)
        if self.direct_attention_enabled:
            return keys, values
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
            self._storage_written_tokens.pop(block_id, None)
            self._block_indices = None
            self.block_table.append_block(block_id)

    def _write_segment(
        self,
        block: Any,
        block_offset: int,
        segment: tuple[Any, Any],
    ) -> None:
        old_tokens = (
            self._pool.block_token_count(block.block_id)
            if self._pool_enabled
            else self._storage_written_tokens.get(block.block_id, 0)
        )
        if block_offset != old_tokens:
            raise ValueError("Paged KV writes must append within a block")
        if self._pool_enabled:
            self._pool.write(block.block_id, block_offset, segment[0], segment[1])
        if len(block.cache_data) < self.manager.num_layers:
            block.cache_data.extend([None] * (self.manager.num_layers - len(block.cache_data)))
        # Pool ownership is held by _PagedKVBlockPool; retaining row views here
        # would keep every intermediate MLX write version alive.
        block.cache_data[self.layer_index] = None
        end = block_offset + int(segment[0].shape[2])
        self._storage_written_tokens[block.block_id] = max(
            self._storage_written_tokens.get(block.block_id, 0), end
        )
        block.token_count = max(block.token_count, end)

    def _update_materialized(self, keys: Any, values: Any, start: int, end: int) -> None:
        mx = _mlx()
        if self._materialized_keys is None or end > self._materialized_capacity:
            old_keys = self._materialized_keys
            old_values = self._materialized_values
            old_capacity = self._materialized_capacity
            if old_keys is None and self.offset > 0:
                old_keys, old_values = self._materialize_from_pool()
            growth = max(self.step, end - old_capacity)
            new_capacity = max(end, self.block_size, old_capacity + growth)
            B, H, _, key_dim = keys.shape
            value_dim = values.shape[3]
            new_keys = mx.zeros((B, H, new_capacity, key_dim), dtype=keys.dtype)
            new_values = mx.zeros((B, H, new_capacity, value_dim), dtype=values.dtype)
            if old_keys is not None and old_values is not None and self.offset > 0:
                new_keys[..., : self.offset, :] = old_keys[..., : self.offset, :]
                new_values[..., : self.offset, :] = old_values[..., : self.offset, :]
            self._materialized_keys = new_keys
            self._materialized_values = new_values
            self._materialized_capacity = new_capacity
        if self._materialized_keys is None or self._materialized_values is None:
            raise RuntimeError("Contiguous paged KV fallback is not initialized")
        self._materialized_keys[..., start:end, :] = keys
        self._materialized_values[..., start:end, :] = values

    def _materialize_from_pool(self) -> tuple[Any, Any]:
        if self.offset == 0:
            raise ValueError("Cannot materialize an empty paged KV cache")
        if not self._pool_enabled:
            if self._materialized_keys is None or self._materialized_values is None:
                raise RuntimeError("Storage-only paged KV cache has no materialized state")
            return (
                self._materialized_keys[..., : self.offset, :],
                self._materialized_values[..., : self.offset, :],
            )
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

    def _materialize(self) -> tuple[Any, Any]:
        if self.offset == 0:
            raise ValueError("Cannot materialize an empty paged KV cache")
        if self.direct_attention_enabled and self._pool_enabled:
            return self._materialize_from_pool()
        if self._materialized_keys is None or self._materialized_values is None:
            self._materialized_keys, self._materialized_values = self._materialize_from_pool()
            self._materialized_capacity = self.offset
        return (
            self._materialized_keys[..., : self.offset, :],
            self._materialized_values[..., : self.offset, :],
        )

    def _block_pool(self) -> tuple[Any, Any, Any]:
        if self.offset == 0 or not self.block_table.block_ids:
            raise ValueError("Cannot build a block pool from an empty paged cache")
        self._promote_pool()
        mx = _mlx()
        pool_keys, pool_values = self._pool.block_pool()
        block_ids = tuple(self.block_table.block_ids)
        if self._block_indices is None or self._block_indices_ids != block_ids:
            self._block_indices = mx.array(block_ids, dtype=mx.uint32)
            self._block_indices_ids = block_ids
        return (
            pool_keys,
            pool_values,
            self._block_indices,
        )

    def _promote_pool(self) -> None:
        if self._pool_enabled:
            return
        if self._materialized_keys is None or self._materialized_values is None:
            raise RuntimeError("Cannot promote an empty storage-only paged KV cache")
        writable_block_ids: list[int] = []
        for logical_index in range(len(self.block_table.block_ids)):
            old_block_id = self.block_table.block_ids[logical_index]
            block_id = self.manager.ensure_writable_block(self.block_table, logical_index)
            writable_block_ids.append(block_id)
            if block_id != old_block_id:
                self._block_indices = None
        max_block_id = max(writable_block_ids)
        first_take = min(self.block_size, self.offset)
        self._pool.ensure_capacity(
            max_block_id,
            self._materialized_keys[..., :first_take, :],
            self._materialized_values[..., :first_take, :],
        )
        position = 0
        for block_id in writable_block_ids:
            take = min(self.block_size, self.offset - position)
            segment_keys = self._materialized_keys[..., position : position + take, :]
            segment_values = self._materialized_values[..., position : position + take, :]
            self._pool.write(block_id, 0, segment_keys, segment_values)
            block = self.manager.get_block(block_id)
            if len(block.cache_data) < self.manager.num_layers:
                block.cache_data.extend([None] * (self.manager.num_layers - len(block.cache_data)))
            block.cache_data[self.layer_index] = None
            position += take
        self._pool_enabled = True

    def prepare_direct_attention(self) -> None:
        if not self.direct_attention_enabled:
            return
        if not self._pool_enabled:
            self._promote_pool()
            self._materialized_keys = None
            self._materialized_values = None
            self._materialized_capacity = 0

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
        self._materialized_keys = None
        self._materialized_values = None
        self._materialized_capacity = 0
        self._storage_written_tokens.clear()
        self._block_indices = None
        self._block_indices_ids = ()

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
            enable_block_pool=self._pool_enabled,
            enable_direct_attention=self.direct_attention_enabled,
        )
        forked.offset = self.offset
        forked._storage_written_tokens = dict(self._storage_written_tokens)
        self._copy_materialized_to(forked)
        return forked

    def _copy_materialized_to(self, target: PagedKVCacheLayer) -> None:
        if self._materialized_keys is None or self._materialized_values is None:
            return
        mx = _mlx()
        target._materialized_keys = mx.zeros(
            self._materialized_keys.shape, dtype=self._materialized_keys.dtype
        )
        target._materialized_values = mx.zeros(
            self._materialized_values.shape, dtype=self._materialized_values.dtype
        )
        target._materialized_keys[:, :, :, :] = self._materialized_keys
        target._materialized_values[:, :, :, :] = self._materialized_values
        target._materialized_capacity = self._materialized_capacity
        target._storage_written_tokens = dict(self._storage_written_tokens)

    def size(self) -> int:
        return self.offset

    def is_trimmable(self) -> bool:
        return True

    def trim(self, num_tokens: int) -> int:
        trimmed = min(max(int(num_tokens), 0), self.offset)
        if trimmed == 0:
            return 0
        self.offset -= trimmed
        keep_count = (self.offset + self.block_size - 1) // self.block_size
        removed = self.block_table.block_ids[keep_count:]
        for block_id in removed:
            self.manager.free_block(block_id)
            self._storage_written_tokens.pop(block_id, None)
        if removed:
            self._block_indices = None
            self._block_indices_ids = ()
        del self.block_table.block_ids[keep_count:]
        self.block_table.num_tokens = self.offset
        if keep_count and self.offset % self.block_size:
            block_index = keep_count - 1
            old_block_id = self.block_table.block_ids[block_index]
            token_count = self.offset % self.block_size
            block_id = old_block_id
            if self._pool_enabled:
                block_id = self.manager.ensure_writable_block(self.block_table, block_index)
                if block_id != old_block_id:
                    self._block_indices = None
                    source_id = self.manager.cow_source(block_id)
                    if source_id is not None:
                        self._pool.copy_block(source_id, block_id)
                self._pool.truncate(block_id, token_count)
            block = self.manager.get_block(block_id)
            block.token_count = token_count
            if not self._pool_enabled:
                self._storage_written_tokens[block_id] = token_count
            if self._pool_enabled:
                block.cache_data[self.layer_index] = None
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
        return (
            self._pool.nbytes
            + _nbytes(self._materialized_keys)
            + _nbytes(self._materialized_values)
        )

    def attention_view(self) -> PagedAttentionView:
        return PagedAttentionView(
            layer=self,
            block_ids=tuple(self.block_table.block_ids),
            block_size=self.block_size,
            sequence_length=self.offset,
        )


class PagedKVCacheList(list[Any]):
    """List-compatible hybrid cache container with an explicit owner."""

    def __init__(self, caches: tuple[Any, ...], bundle: PagedKVCacheBundle) -> None:
        super().__init__(caches)
        self.bundle = bundle

    def release(self) -> None:
        self.bundle.release()


class PagedKVCacheBundle:
    """Own hybrid cache layers and release full-attention pools together."""

    def __init__(
        self,
        manager: PagedCacheManager,
        layers: tuple[PagedKVCacheLayer, ...],
        caches: tuple[Any, ...],
        *,
        request_id: str,
        enable_block_pool: bool = True,
        enable_direct_attention: bool = False,
    ) -> None:
        if not layers:
            raise ValueError("Paged KV bundles require at least one layer")
        if not caches or any(layer.layer_index >= len(caches) for layer in layers):
            raise ValueError("Paged KV bundle cache list does not contain all full layers")
        source_table = layers[0].block_table
        if any(
            layer.manager is not manager or layer.block_table is not source_table
            for layer in layers
        ):
            raise ValueError("Paged KV bundle layers must share one manager and block table")
        self.manager = manager
        self.layers = layers
        self.caches = PagedKVCacheList(caches, self)
        self.request_id = request_id
        self._enable_block_pool = enable_block_pool
        self._enable_direct_attention = enable_direct_attention
        self._released = False

    @classmethod
    def from_prompt_cache(
        cls,
        prompt_cache: list[Any],
        manager: PagedCacheManager,
        *,
        kv_cache_type: type[Any],
        request_id: str,
        enable_block_pool: bool = True,
        enable_direct_attention: bool = False,
    ) -> PagedKVCacheBundle:
        if not prompt_cache:
            raise ValueError("Paged KV bundles require at least one cache layer")
        table = manager.create_table(request_id)
        caches = list(prompt_cache)
        layers: list[PagedKVCacheLayer] = []
        for index, cache in enumerate(prompt_cache):
            if isinstance(cache, kv_cache_type):
                layer = PagedKVCacheLayer(
                    manager,
                    layer_index=index,
                    block_table=table,
                    enable_block_pool=enable_block_pool,
                    enable_direct_attention=enable_direct_attention,
                )
                layers.append(layer)
                caches[index] = layer
        if not layers:
            manager.remove_table(request_id)
            raise ValueError("Paged KV bundles require at least one full-attention layer")
        return cls(
            manager,
            tuple(layers),
            tuple(caches),
            request_id=request_id,
            enable_block_pool=enable_block_pool,
            enable_direct_attention=enable_direct_attention,
        )

    def fork(self, request_id: str) -> PagedKVCacheBundle:
        if self._released:
            raise RuntimeError("Cannot fork a released paged KV bundle")
        source_table = self.layers[0].block_table
        source_offset = self.layers[0].offset
        if any(layer.offset != source_offset for layer in self.layers):
            raise ValueError("Paged KV bundle layers must have matching offsets")
        table = self.manager.fork_table(source_table, request_id)
        fork_layers: dict[int, PagedKVCacheLayer] = {}
        for layer in self.layers:
            fork_layers[layer.layer_index] = PagedKVCacheLayer(
                self.manager,
                layer_index=layer.layer_index,
                block_table=table,
                pool=layer._pool,
                enable_block_pool=layer._pool_enabled,
                enable_direct_attention=layer.direct_attention_enabled,
            )
        for layer in fork_layers.values():
            layer.offset = source_offset
        for source_layer in self.layers:
            target_layer = fork_layers[source_layer.layer_index]
            source_layer._copy_materialized_to(target_layer)
        fork_caches = tuple(
            fork_layers[index] if index in fork_layers else copy.deepcopy(cache)
            for index, cache in enumerate(self.caches)
        )
        return PagedKVCacheBundle(
            self.manager,
            tuple(fork_layers.values()),
            fork_caches,
            request_id=request_id,
            enable_block_pool=self._enable_block_pool,
            enable_direct_attention=self._enable_direct_attention,
        )

    def release(self) -> None:
        if self._released:
            return
        self.manager.remove_table(self.request_id, discard_cache_data=True)
        for layer in self.layers:
            layer._pool.release()
        self._released = True


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
    "PagedBatchAttentionView",
    "PagedKVCacheList",
    "PagedKVCacheBundle",
    "PagedKVCacheLayer",
    "replace_kv_cache_layers",
]
