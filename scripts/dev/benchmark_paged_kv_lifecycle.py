"""Measure experimental paged KV pool ownership and release behavior."""

from __future__ import annotations

import gc
import json

import mlx.core as mx
from mlx_lm.models.cache import KVCache

from aster.inference.paged_cache import PagedCacheManager
from aster.inference.paged_kv_adapter import PagedKVCacheBundle


def _memory_snapshot(bundle: PagedKVCacheBundle) -> dict[str, int]:
    return {
        "pool_nbytes": bundle.layers[0]._pool.nbytes,
        "active_memory_bytes": int(mx.get_active_memory()),
        "cache_memory_bytes": int(mx.get_cache_memory()),
    }


def _clear_allocator_cache() -> None:
    gc.collect()
    mx.synchronize()
    mx.clear_cache()
    mx.synchronize()


def main() -> None:
    manager = PagedCacheManager(num_layers=1, block_size=64, max_blocks=32)
    source = PagedKVCacheBundle.from_prompt_cache(
        [KVCache()], manager, kv_cache_type=KVCache, request_id="lifecycle-source"
    )
    keys = mx.random.normal((1, 2, 512, 256)).astype(mx.float16)
    values = mx.random.normal((1, 2, 512, 256)).astype(mx.float16)
    source.layers[0].update_and_fetch(keys, values)
    mx.eval(source.layers[0]._pool.keys, source.layers[0]._pool.values)
    before_fork = _memory_snapshot(source)

    child = source.fork("lifecycle-child")
    child.layers[0].update_and_fetch(
        mx.random.normal((1, 2, 1, 256)).astype(mx.float16),
        mx.random.normal((1, 2, 1, 256)).astype(mx.float16),
    )
    mx.eval(child.layers[0]._pool.keys, child.layers[0]._pool.values)
    after_fork = _memory_snapshot(source)

    child.release()
    _clear_allocator_cache()
    after_child_release = _memory_snapshot(source)

    source.release()
    del keys, values
    _clear_allocator_cache()
    after_source_release = _memory_snapshot(source)

    print(
        json.dumps(
            {
                "shape": [1, 2, 512, 256],
                "block_size": 64,
                "before_fork": before_fork,
                "after_fork": after_fork,
                "after_child_release": after_child_release,
                "after_source_release": after_source_release,
                "manager_allocated_blocks_after_release": manager.stats.allocated_blocks,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
