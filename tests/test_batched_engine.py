from __future__ import annotations

from types import SimpleNamespace

import aster.inference.batched_engine as batched_engine_module
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


def test_batch_generator_rejects_mixed_cached_batch_profiles() -> None:
    engine = object.__new__(BatchedEngine)
    engine._running = {"active"}
    engine._state = {
        "active": _RequestState(
            request_id="active",
            request=InferenceRequest(prompt="ignored", trace_id="active"),
            prompt_tokens=[1, 2, 3, 4],
            prefix_cache_key="active-cache",
        )
    }

    assert engine._prompt_length_compatible(4) is True
    assert engine._prompt_length_compatible(5) is False
    assert engine._cache_restore_compatible(True, 3, 4) is True
    assert engine._cache_restore_compatible(True, 4, 4) is False
    assert engine._cache_restore_compatible(False, 0, 4) is False
    assert engine._cache_restore_compatible(True, 3, 5) is False


def test_batched_engine_passes_structured_schema_before_tokenizer(monkeypatch) -> None:
    calls: list[tuple[object, object]] = []
    tokenizer = object()
    schema = {"type": "object", "properties": {"answer": {"type": "string"}}}

    def fake_builder(received_schema, received_tokenizer):
        calls.append((received_schema, received_tokenizer))
        return object()

    monkeypatch.setattr(batched_engine_module, "build_json_logits_processor", fake_builder)
    engine = object.__new__(BatchedEngine)
    engine._tokenizer = tokenizer

    processors = engine._make_logits_processors_for_request(
        InferenceRequest(prompt="json", structured_output_schema=schema)
    )

    assert len(processors) == 1
    assert calls == [(schema, tokenizer)]


def test_request_state_uses_effective_stop_token_ids() -> None:
    state = _RequestState(
        request_id="stop",
        request=InferenceRequest(prompt="stop", stop_token_ids=(7,)),
    )

    assert state.stop_token_ids == frozenset({7})
    state.effective_stop_token_ids = frozenset({7, 8})
    assert state.stop_token_ids == frozenset({7, 8})


def test_batched_engine_decodes_output_without_special_tokens() -> None:
    class Tokenizer:
        def decode(self, token_ids, *, skip_special_tokens=False):
            assert token_ids == [1, 2]
            assert skip_special_tokens is True
            return "clean output"

    engine = object.__new__(BatchedEngine)
    engine._tokenizer = Tokenizer()

    assert engine._decode_output_text([1, 2]) == "clean output"
