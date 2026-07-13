# SPDX-License-Identifier: MIT
"""Paged KV Cache for Aster — block-based KV cache allocator.

Design follows vllm-mlx's PagedCacheManager with block-level prefix
sharing via chain hashing and Copy-on-Write.

Architecture:
  PagedCacheManager
    ├── CacheBlock pool (fixed-size blocks, e.g. 64 tokens)
    ├── FreeKVCacheBlockQueue (doubly-linked LRU)
    ├── BlockHashToBlockMap (chain-hash dedup)
    └── BlockTable (per-sequence block mapping)
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NULL_BLOCK_ID = 0  # Sentinel null block (never freed)
DEFAULT_BLOCK_SIZE = 64  # Tokens per block
DEFAULT_MAX_BLOCKS = 1000


# ---------------------------------------------------------------------------
# CacheBlock — the fundamental allocation unit
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class CacheBlock:
    """A single KV cache block.

    Attributes:
        block_id: Unique block ID (0 = null sentinel).
        ref_count: Number of sequences referencing this block.
        block_hash: Chain hash of tokens in this block.
        prev_free: Previous block in free LRU list (None = head).
        next_free: Next block in free LRU list (None = tail).
        cache_data: List of (keys, values) per transformer layer.
        token_count: Number of tokens stored (<= block_size).
        last_access: Monotonic timestamp for LRU eviction.
    """
    block_id: int = 0
    ref_count: int = 0
    block_hash: int = 0
    prev_free: CacheBlock | None = None
    next_free: CacheBlock | None = None
    cache_data: list[tuple[Any, Any]] = field(default_factory=list)
    token_count: int = 0
    last_access: float = 0.0

    def is_shared(self) -> bool:
        return self.ref_count > 1

    def is_free(self) -> bool:
        return self.ref_count == 0 and self.block_id != NULL_BLOCK_ID


# ---------------------------------------------------------------------------
# FreeKVCacheBlockQueue — O(1) doubly-linked LRU list
# ---------------------------------------------------------------------------

class FreeKVCacheBlockQueue:
    """Doubly-linked list for LRU tracking of free blocks.

    Head = LRU (evict next), Tail = MRU (recently returned).
    """

    def __init__(self) -> None:
        self._head: CacheBlock | None = None
        self._tail: CacheBlock | None = None

    def is_empty(self) -> bool:
        return self._head is None

    def popleft(self) -> CacheBlock | None:
        """Pop the LRU block (head)."""
        block = self._head
        if block is None:
            return None
        self._remove(block)
        return block

    def append(self, block: CacheBlock) -> None:
        """Append to tail (MRU)."""
        block.prev_free = self._tail
        block.next_free = None
        if self._tail is not None:
            self._tail.next_free = block
        self._tail = block
        if self._head is None:
            self._head = block

    def remove(self, block: CacheBlock) -> None:
        """Remove a block from anywhere in the list."""
        self._remove(block)

    def _remove(self, block: CacheBlock) -> None:
        prev_b = block.prev_free
        next_b = block.next_free
        if prev_b is not None:
            prev_b.next_free = next_b
        else:
            self._head = next_b
        if next_b is not None:
            next_b.prev_free = prev_b
        else:
            self._tail = prev_b
        block.prev_free = None
        block.next_free = None


# ---------------------------------------------------------------------------
# BlockHasher — chain hash computation
# ---------------------------------------------------------------------------

def compute_block_hash(parent_hash: int, token_ids: list[int]) -> int:
    """Compute chain hash: SHA-256(parent_hash + tokens) → int."""
    h = hashlib.sha256(str(parent_hash).encode())
    for t in token_ids:
        h.update(str(t).encode())
    digest = h.digest()
    return int.from_bytes(digest[:8], "little")


# ---------------------------------------------------------------------------
# BlockTable — per-sequence mapping of logical → physical blocks
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class BlockTable:
    """Maps a sequence's logical token positions to physical cache blocks."""
    request_id: str
    block_ids: list[int] = field(default_factory=list)
    num_tokens: int = 0

    def append_block(self, block_id: int) -> None:
        self.block_ids.append(block_id)

    def get_block_id(self, token_index: int, block_size: int) -> int | None:
        block_idx = token_index // block_size
        if block_idx < len(self.block_ids):
            return self.block_ids[block_idx]
        return None

    def num_blocks(self) -> int:
        return len(self.block_ids)


# ---------------------------------------------------------------------------
# CacheStats — observability
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class CacheStats:
    total_blocks: int = 0
    allocated_blocks: int = 0
    free_blocks: int = 0
    shared_blocks: int = 0
    cache_hits: int = 0
    cow_copies: int = 0
    evictions: int = 0

    def snapshot(self) -> dict[str, int]:
        return {
            "total_blocks": self.total_blocks,
            "allocated_blocks": self.allocated_blocks,
            "free_blocks": self.free_blocks,
            "shared_blocks": self.shared_blocks,
            "cache_hits": self.cache_hits,
            "cow_copies": self.cow_copies,
            "evictions": self.evictions,
        }


# ---------------------------------------------------------------------------
# PagedCacheManager — main block allocator
# ---------------------------------------------------------------------------

class PagedCacheManager:
    """Central block allocator with prefix sharing and LRU eviction."""

    def __init__(
        self,
        num_layers: int,
        block_size: int = DEFAULT_BLOCK_SIZE,
        max_blocks: int = DEFAULT_MAX_BLOCKS,
    ) -> None:
        self.block_size = block_size
        self.max_blocks = max_blocks
        self.num_layers = num_layers
        self.stats = CacheStats(total_blocks=max_blocks)

        # Block pool (indexed by block_id)
        self._blocks: list[CacheBlock] = [CacheBlock(block_id=NULL_BLOCK_ID)]

        # Hash → blocks (multiple blocks can have same hash for hybrid models)
        self._hash_map: dict[int, list[CacheBlock]] = {}

        # Per-sequence block tables
        self._tables: dict[str, BlockTable] = {}
        self._cow_sources: dict[int, int] = {}

        # Free list
        self._free_queue = FreeKVCacheBlockQueue()

        # Initialize the block pool
        for bid in range(1, max_blocks + 1):
            block = CacheBlock(block_id=bid)
            self._blocks.append(block)
            self._free_queue.append(block)
            self.stats.free_blocks += 1

        # The null block is always allocated
        self._blocks[NULL_BLOCK_ID].ref_count = 1

    def allocate_block(self) -> int:
        """Allocate a free block, evicting if necessary. Returns block_id."""
        block = self._free_queue.popleft()
        if block is None:
            self._evict_lru()
            block = self._free_queue.popleft()
        if block is None:
            raise MemoryError("No free cache blocks available")
        block.ref_count = 1
        block.cache_data = []
        block.token_count = 0
        block.block_hash = 0
        self._cow_sources.pop(block.block_id, None)
        self.stats.allocated_blocks += 1
        self.stats.free_blocks -= 1
        return block.block_id

    def free_block(self, block_id: int) -> None:
        """Decrement refcount; return to free pool if zero."""
        if block_id == NULL_BLOCK_ID:
            return
        block = self._blocks[block_id]
        if block.ref_count <= 0:
            return
        was_shared = block.is_shared()
        block.ref_count -= 1
        if was_shared and not block.is_shared():
            self.stats.shared_blocks = max(self.stats.shared_blocks - 1, 0)
        if block.ref_count == 0:
            self._free_queue.append(block)
            self.stats.free_blocks += 1
            self.stats.allocated_blocks -= 1

    def increment_ref(self, block_id: int) -> None:
        """Mark a block as shared by another sequence."""
        if block_id == NULL_BLOCK_ID:
            return
        block = self._blocks[block_id]
        was_shared = block.is_shared()
        block.ref_count += 1
        if not was_shared and block.is_shared():
            self.stats.shared_blocks += 1

    def get_block(self, block_id: int) -> CacheBlock:
        """Return a physical block by ID for an attention adapter."""
        if block_id < 0 or block_id >= len(self._blocks):
            raise IndexError(f"Unknown cache block {block_id}")
        return self._blocks[block_id]

    def fork_table(self, source: BlockTable, request_id: str) -> BlockTable:
        """Fork a block table while retaining one reference to each block."""
        if request_id in self._tables:
            raise ValueError(f"Cache table already exists: {request_id}")
        table = self.create_table(request_id)
        for block_id in source.block_ids:
            self.increment_ref(block_id)
            table.append_block(block_id)
        table.num_tokens = source.num_tokens
        return table

    def ensure_writable_block(self, table: BlockTable, block_index: int) -> int:
        """Return a table block that can be modified without mutating a fork."""
        if block_index < 0 or block_index >= len(table.block_ids):
            raise IndexError(f"Unknown logical cache block {block_index}")
        block_id = table.block_ids[block_index]
        source = self._blocks[block_id]
        if not source.is_shared():
            return block_id

        new_id = self._cow_copy_block(source)
        self._cow_sources[new_id] = block_id
        self.free_block(block_id)
        table.block_ids[block_index] = new_id
        return new_id

    def cow_source(self, block_id: int) -> int | None:
        """Return the source block for a recent COW allocation, if any."""
        return self._cow_sources.get(block_id)

    def get_blocks_for_generation(self, table: BlockTable) -> list[int]:
        """Ensure blocks are not shared before writing (COW)."""
        new_ids: list[int] = []
        for bid in table.block_ids:
            block = self._blocks[bid]
            if block.is_shared():
                # COW: allocate new block, copy cache data
                new_id = self._cow_copy_block(block)
                new_ids.append(new_id)
            else:
                new_ids.append(bid)
        if new_ids != table.block_ids:
            table.block_ids = new_ids
        return new_ids

    def store_block_cache(
        self, block_id: int, layer_caches: list[tuple[Any, Any]], token_count: int
    ) -> None:
        """Store KV cache data in a block."""
        block = self._blocks[block_id]
        block.cache_data = layer_caches
        block.token_count = token_count
        block.last_access = time.monotonic()

    def get_computed_blocks(self, token_ids: list[int], parent_hash: int = 0) -> list[int]:
        """Walk token IDs checking for cached prefix blocks. Stops at first miss."""
        block_ids: list[int] = []
        pos = 0
        hash_val = parent_hash
        while pos < len(token_ids):
            chunk = token_ids[pos:pos + self.block_size]
            hash_val = compute_block_hash(hash_val, chunk)
            blocks = self._hash_map.get(hash_val)
            if blocks:
                block_ids.append(blocks[0].block_id)
            else:
                break
            pos += self.block_size
        return block_ids

    def cache_full_blocks(
        self, token_ids: list[int], block_ids: list[int], parent_hash: int = 0
    ) -> None:
        """Register newly computed blocks in the chain hash index."""
        pos = 0
        hash_val = parent_hash
        for bid in block_ids:
            chunk = token_ids[pos:pos + self.block_size]
            hash_val = compute_block_hash(hash_val, chunk)
            block = self._blocks[bid]
            block.block_hash = hash_val
            self._hash_map.setdefault(hash_val, []).append(block)
            pos += self.block_size

    def create_table(self, request_id: str) -> BlockTable:
        table = BlockTable(request_id=request_id)
        self._tables[request_id] = table
        return table

    def get_table(self, request_id: str) -> BlockTable | None:
        return self._tables.get(request_id)

    def remove_table(self, request_id: str, *, discard_cache_data: bool = False) -> None:
        table = self._tables.pop(request_id, None)
        if table is not None:
            for bid in table.block_ids:
                self.free_block(bid)
                if discard_cache_data and self._blocks[bid].ref_count == 0:
                    self._blocks[bid].cache_data = []

    def _cow_copy_block(self, src: CacheBlock) -> int:
        """Copy-on-write: allocate new block, copy cache data."""
        new_id = self.allocate_block()
        dst = self._blocks[new_id]
        dst.cache_data = list(src.cache_data)  # Shallow copy of layer list
        dst.token_count = src.token_count
        dst.block_hash = src.block_hash
        self.stats.cow_copies += 1
        return new_id

    def _evict_lru(self) -> None:
        """Evict the LRU block from the free queue."""
        block = self._free_queue.popleft()
        if block is None:
            return
        # Remove from hash map
        if block.block_hash:
            blocks = self._hash_map.get(block.block_hash)
            if blocks:
                try:
                    blocks.remove(block)
                    if not blocks:
                        del self._hash_map[block.block_hash]
                except ValueError:
                    pass
        block.cache_data = []
        block.block_hash = 0
        block.token_count = 0
        # Return to free pool
        self._free_queue.append(block)
        self.stats.evictions += 1

    def reset(self) -> None:
        """Release all blocks back to the free pool."""
        self._hash_map.clear()
        self._tables.clear()
        self._cow_sources.clear()
        self._free_queue = FreeKVCacheBlockQueue()
        for block in self._blocks[1:]:
            block.ref_count = 0
            block.cache_data = []
            block.block_hash = 0
            block.token_count = 0
            self._free_queue.append(block)
        self.stats = CacheStats(total_blocks=self.max_blocks)
