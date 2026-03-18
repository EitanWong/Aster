from __future__ import annotations

import json
from collections.abc import AsyncIterator

from sse_starlette.sse import EventSourceResponse

from aster.inference.decode_engine import DecodeChunk


async def to_sse(chunks: AsyncIterator[DecodeChunk], model: str, *, include_debug_summary: bool = False) -> EventSourceResponse:
    async def event_generator() -> AsyncIterator[dict[str, str]]:
        async for chunk in chunks:
            if chunk.finished:
                if include_debug_summary and chunk.stats is not None:
                    summary_payload = {
                        "object": "aster.stream.summary",
                        "model": model,
                        "aster": chunk.stats,
                    }
                    yield {"data": json.dumps(summary_payload)}
                yield {"data": "[DONE]"}
                break
            payload = {
                "id": f"chunk-{chunk.index}",
                "object": "chat.completion.chunk",
                "model": model,
                "choices": [{"delta": {"content": chunk.token}, "index": 0, "finish_reason": None}],
            }
            yield {"data": json.dumps(payload)}

    return EventSourceResponse(event_generator())
