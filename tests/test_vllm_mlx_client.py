from __future__ import annotations

import asyncio

import httpx

from aster.core.errors import AsterError
from aster.core.config import VLLMMLXSettings
from aster.inference.vllm_mlx_client import VLLMMLXClient


def test_vllm_mlx_client_infer_parses_openai_response() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "hello from vllm-mlx"}}],
                "usage": {"prompt_tokens": 12, "completion_tokens": 4},
            },
        )

    client = VLLMMLXClient(
        VLLMMLXSettings(),
        transport=httpx.MockTransport(handler),
    )
    async def run_test() -> None:
        try:
            result = await client.infer(
                prompt="hi",
                messages=None,
                max_tokens=16,
                temperature=0.2,
                top_p=0.9,
            )
        finally:
            await client.aclose()

        assert result.text == "hello from vllm-mlx"
        assert result.prompt_tokens == 12
        assert result.completion_tokens == 4

    asyncio.run(run_test())


def test_vllm_mlx_client_stream_parses_sse_chunks() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        payload = b"\n".join(
            [
                b'data: {"choices":[{"delta":{"content":"Hel"},"finish_reason":null}]}',
                b'data: {"choices":[{"delta":{"content":"lo"},"finish_reason":null}]}',
                b'data: {"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":2}}',
                b"data: [DONE]",
            ]
        )
        return httpx.Response(200, content=payload, headers={"content-type": "text/event-stream"})

    client = VLLMMLXClient(
        VLLMMLXSettings(),
        transport=httpx.MockTransport(handler),
    )
    async def run_test() -> None:
        try:
            events = [event async for event in client.stream(prompt="hi", messages=None, max_tokens=16, temperature=0.2, top_p=0.9)]
        finally:
            await client.aclose()

        assert [event.text for event in events] == ["Hel", "lo", ""]
        assert events[-1].finish_reason == "stop"
        assert events[-1].prompt_tokens == 10
        assert events[-1].completion_tokens == 2

    asyncio.run(run_test())


def test_vllm_mlx_client_embeddings_raises_aster_error_on_upstream_400() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/embeddings"
        return httpx.Response(400, json={"detail": "embedding model mismatch"})

    client = VLLMMLXClient(
        VLLMMLXSettings(),
        transport=httpx.MockTransport(handler),
    )

    async def run_test() -> None:
        try:
            await client.embeddings(model="default", input_data="hello")
        except AsterError as exc:
            assert exc.status_code == 400
            assert exc.message == "embedding model mismatch"
        else:
            raise AssertionError("expected AsterError")
        finally:
            await client.aclose()

    asyncio.run(run_test())
