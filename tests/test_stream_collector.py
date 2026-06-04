from __future__ import annotations

import asyncio

from aster.inference.decode_engine import DecodeChunk
from aster.inference.stream_collector import StreamCollector


def test_stream_collector_aggregates_unconsumed_chunks() -> None:
    async def scenario() -> None:
        collector = StreamCollector(stream_interval_tokens=1)

        await collector.add_text("a", index=0)
        await collector.add_text("b", index=1)
        await collector.add_text("c", index=2)

        chunk = collector.get_nowait()
        assert isinstance(chunk, DecodeChunk)
        assert chunk.token == "abc"
        assert chunk.index == 2
        assert collector.get_nowait() is None

    asyncio.run(scenario())


def test_stream_collector_sends_first_token_before_interval() -> None:
    async def scenario() -> None:
        collector = StreamCollector(stream_interval_tokens=4)

        await collector.add_text("a", index=0)

        chunk = collector.get_nowait()
        assert isinstance(chunk, DecodeChunk)
        assert chunk.token == "a"
        assert chunk.index == 0

    asyncio.run(scenario())


def test_stream_collector_finish_sends_terminal_and_close_sentinel() -> None:
    async def scenario() -> None:
        collector = StreamCollector(stream_interval_tokens=4)

        await collector.add_text("a", index=0)
        await collector.add_text("b", index=1)
        await collector.finish(index=2, stats={"completion_tokens": 2})

        first = collector.get_nowait()
        second = collector.get_nowait()
        third = await collector.get()

        assert isinstance(first, DecodeChunk)
        assert first.token == "ab"
        assert isinstance(second, DecodeChunk)
        assert second.finished is True
        assert third is None

    asyncio.run(scenario())


def test_stream_collector_fail_preserves_pending_text_before_error() -> None:
    async def scenario() -> None:
        collector = StreamCollector(stream_interval_tokens=4)
        exc = RuntimeError("decode failed")

        await collector.add_text("a", index=0)
        await collector.add_text("b", index=1)
        await collector.fail(exc)

        first = collector.get_nowait()
        second = collector.get_nowait()
        third = await collector.get()

        assert isinstance(first, DecodeChunk)
        assert first.token == "ab"
        assert second is exc
        assert third is None

    asyncio.run(scenario())
