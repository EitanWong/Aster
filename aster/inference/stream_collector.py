from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from aster.inference.decode_engine import DecodeChunk


@dataclass(slots=True)
class StreamCollector:
    stream_interval_tokens: int
    pending_parts: list[str] = field(default_factory=list)
    aggregate_pending: bool = True
    _pending_chunk: DecodeChunk | None = None
    _terminal_item: DecodeChunk | BaseException | None = None
    _ready: asyncio.Event = field(default_factory=asyncio.Event)
    _pending_index: int = 0
    _sent_chunks: int = 0
    _closed: bool = False

    @property
    def closed(self) -> bool:
        return self._closed

    async def add_text(self, text: str, *, index: int) -> None:
        if not text or self._closed:
            return
        self.pending_parts.append(text)
        self._pending_index = index
        if self._sent_chunks == 0 or len(self.pending_parts) >= max(self.stream_interval_tokens, 1):
            await self.flush(index=index)

    async def flush(self, *, index: int) -> None:
        if not self.pending_parts or self._closed:
            return
        self._put_chunk(
            DecodeChunk(
                token="".join(self.pending_parts),
                index=index,
                finished=False,
            )
        )
        self.pending_parts.clear()
        self._sent_chunks += 1

    async def finish(self, *, index: int, stats: dict[str, object]) -> None:
        await self.flush(index=max(index - 1, 0))
        if self._terminal_item is not None:
            return
        self._terminal_item = DecodeChunk(
            token="",
            index=index,
            finished=True,
            stats=stats,
        )
        self._closed = True
        self._refresh_ready()

    async def fail(self, exc: BaseException) -> None:
        await self.flush(index=self._pending_index)
        if self._terminal_item is None:
            self._terminal_item = exc
        self._closed = True
        self._refresh_ready()

    async def close(self) -> None:
        self._closed = True
        self._refresh_ready()

    def get_nowait(self) -> DecodeChunk | BaseException | None:
        if self._pending_chunk is not None:
            chunk = self._pending_chunk
            self._pending_chunk = None
            self._refresh_ready()
            return chunk
        if self._terminal_item is not None:
            item = self._terminal_item
            self._terminal_item = None
            self._refresh_ready()
            return item
        return None

    async def get(self) -> DecodeChunk | BaseException | None:
        while True:
            item = self.get_nowait()
            if item is not None or self._closed:
                return item
            await self._ready.wait()

    def _put_chunk(self, chunk: DecodeChunk) -> None:
        if self._closed:
            return
        if (
            self.aggregate_pending
            and self._pending_chunk is not None
            and not self._pending_chunk.finished
            and not chunk.finished
        ):
            self._pending_chunk = DecodeChunk(
                token=f"{self._pending_chunk.token}{chunk.token}",
                index=max(self._pending_chunk.index, chunk.index),
                finished=False,
                stats=self._pending_chunk.stats or chunk.stats,
            )
        else:
            self._pending_chunk = chunk
        self._refresh_ready()

    def _refresh_ready(self) -> None:
        if self._pending_chunk is not None or self._terminal_item is not None or self._closed:
            self._ready.set()
        else:
            self._ready.clear()
