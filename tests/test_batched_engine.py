from __future__ import annotations

from types import SimpleNamespace

from aster.inference.batched_engine import BatchedEngine, _RequestState
from aster.inference.contracts import InferenceRequest
from aster.inference.prefix_store import SnapshotEntry


class _FakeKVCache:
    def __init__(self, offset: int) -> None:
        self.offset = offset
        self.keys = SimpleNamespace(shape=(1, 1, max(offset, 1), 4))
        self.values = SimpleNamespace(shape=(1, 1, max(offset, 1), 4))


def _entry(*, prefix_tokens: tuple[int, ...], cache_token_count: int) -> SnapshotEntry:
    return SnapshotEntry(
        key="prefix-key",
        model_name="test-model",
        model_fingerprint="fingerprint",
        prefix_tokens=prefix_tokens,
        cache_token_count=cache_token_count,
        prompt_cache=[_FakeKVCache(cache_token_count)],
        approx_bytes=128,
    )


def test_batch_generator_prefix_insert_starts_with_first_uncached_token() -> None:
    engine = object.__new__(BatchedEngine)
    entry = _entry(prefix_tokens=(10, 11, 12, 13), cache_token_count=3)

    prepared = engine._prepare_prefix_cache_insert(entry, [10, 11, 12, 13, 14, 15])

    assert prepared is not None
    prompt_cache, prompt_tokens, cached_tokens = prepared
    assert prompt_tokens == [13, 14, 15]
    assert cached_tokens == 4
    assert prompt_cache is not entry.prompt_cache
    assert prompt_cache[0] is not entry.prompt_cache[0]
    assert prompt_cache[0].offset == 3


def test_batch_generator_stores_prompt_boundary_cache() -> None:
    captured: list[dict[str, object]] = []
    engine = object.__new__(BatchedEngine)
    engine._batch_generator = SimpleNamespace(
        extract_cache=lambda uids: {
            uids[0]: ([_FakeKVCache(3)], [10, 11, 12])
        }
    )
    engine._uid_to_rid = {7: "request-7"}
    engine._state = {
        "request-7": _RequestState(
            request_id="request-7",
            request=InferenceRequest(prompt="ignored", trace_id="request-7"),
            prompt_tokens=[10, 11, 12, 13],
        )
    }
    engine.prefix_store = SimpleNamespace(
        enabled=True,
        store=lambda **kwargs: captured.append(kwargs),
    )
    engine._model_fingerprint = "fingerprint"
    engine.settings = SimpleNamespace(model=SimpleNamespace(name="test-model"))

    engine._process_prompt_responses(
        [
            SimpleNamespace(
                uid=7,
                end_of_segment=True,
                end_of_prompt=False,
                progress=(3, 4),
            )
        ]
    )

    assert len(captured) == 1
    assert captured[0]["prefix_tokens"] == [10, 11, 12, 13]
    assert captured[0]["cache_token_count"] == 3
    assert captured[0]["model_fingerprint"] == "fingerprint"
