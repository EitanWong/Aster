from __future__ import annotations

from collections import deque
from types import SimpleNamespace

import aster.inference.batched_engine as batched_engine_module
from aster.inference.batched_engine import BatchedEngine, _BatchLane, _RequestState
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
    profile = (4, False, 0)
    lane = _BatchLane(
        profile=profile,
        generator=SimpleNamespace(
        extract_cache=lambda uids: {
            uids[0]: ([_FakeKVCache(3)], [10, 11, 12])
        }
        ),
    )
    lane.uid_to_rid = {7: "request-7"}
    engine._batch_lanes = {profile: lane}
    engine._state = {
        "request-7": _RequestState(
            request_id="request-7",
            request=InferenceRequest(prompt="ignored", trace_id="request-7"),
            prompt_tokens=[10, 11, 12, 13],
            batch_profile=profile,
        )
    }
    engine.prefix_store = SimpleNamespace(
        enabled=True,
        store=lambda **kwargs: captured.append(kwargs),
    )
    engine._model_fingerprint = "fingerprint"
    engine.settings = SimpleNamespace(model=SimpleNamespace(name="test-model"))

    engine._process_prompt_responses(
        lane,
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


def test_batched_engine_creates_bounded_lanes_for_incompatible_profiles() -> None:
    created: list[object] = []
    engine = object.__new__(BatchedEngine)
    engine._batch_lanes = {}
    engine._batch_generator_max_lanes = 2
    engine._create_batch_generator = lambda: created.append(object()) or created[-1]

    first = engine._get_or_create_lane((4, False, 0))
    second = engine._get_or_create_lane((5, False, 0))
    reused = engine._get_or_create_lane((4, False, 0))
    first.request_ids.add("active-first")
    second.request_ids.add("active-second")
    rejected = engine._get_or_create_lane((6, False, 0))
    second.request_ids.clear()
    recycled = engine._get_or_create_lane((6, False, 0))

    assert first is not None
    assert second is not None
    assert reused is first
    assert rejected is None
    assert recycled is not None
    assert recycled is not second
    assert len(created) == 3
    assert len(engine._batch_lanes) == 2


def test_batched_engine_seals_a_lane_after_first_step() -> None:
    engine = object.__new__(BatchedEngine)
    engine._batch_lanes = {}
    engine._batch_generator_max_lanes = 2
    engine._create_batch_generator = lambda: object()

    lane = engine._get_or_create_lane((4, False, 0))
    assert lane is not None
    lane.request_ids.add("active")
    lane.sealed = True

    assert engine._get_or_create_lane((4, False, 0)) is None


def test_batched_engine_admission_window_delays_first_step() -> None:
    engine = object.__new__(BatchedEngine)
    lane = _BatchLane(profile=(4, False, 0), generator=object())
    lane.created_at = 10.0
    lane.admission_window_ms = 100.0
    lane.target_request_count = 3

    assert engine._lane_ready_for_step(lane, now=10.099) is False
    lane.request_ids.update({"one", "two", "three"})
    assert engine._lane_ready_for_step(lane, now=10.099) is True
    assert engine._lane_ready_for_step(lane, now=10.100) is True


def test_batched_engine_applies_cohort_window_only_to_isolated_secondary_lane() -> None:
    engine = object.__new__(BatchedEngine)
    engine._batch_lanes = {}
    engine._batch_generator_max_lanes = 2
    engine._batch_generator_lane_admission_window_ms = 200.0
    engine._create_batch_generator = lambda: object()
    engine._running = {"existing"}
    engine._waiting = deque()

    isolated = engine._get_or_create_lane((4, False, 0), wait_for_cohort=True)

    assert isolated is not None
    assert isolated.admission_window_ms == 200.0

    engine._batch_lanes = {}
    engine._running = {"existing"}
    engine._waiting = deque(["another-request"])
    backlog = engine._get_or_create_lane((5, False, 0), wait_for_cohort=False)

    assert backlog is not None
    assert backlog.admission_window_ms == 0.0


def test_batched_engine_prioritizes_the_longest_prompt_lane() -> None:
    engine = object.__new__(BatchedEngine)
    engine._batch_generator_longest_lane_step_quanta = 2
    short = _BatchLane(profile=(8, False, 0), generator=object())
    long = _BatchLane(profile=(128, False, 0), generator=object())

    assert engine._lane_step_quanta(short, [short, long]) == 1
    assert engine._lane_step_quanta(long, [short, long]) == 2


def test_batched_engine_can_create_a_dedicated_mlx_stream_for_a_lane(monkeypatch) -> None:
    import importlib

    import mlx.core as mx
    generate_module = importlib.import_module("mlx_lm.generate")

    captured: dict[str, object] = {}
    stream = object()

    class BatchGenerator:
        def __init__(self, model: object, **kwargs: object) -> None:
            captured["model"] = model
            captured.update(kwargs)

    monkeypatch.setattr(mx, "new_stream", lambda device: stream)
    monkeypatch.setattr(generate_module, "BatchGenerator", BatchGenerator)

    engine = object.__new__(BatchedEngine)
    engine._model = object()
    engine._batch_generator_max_lanes = 2
    engine._batch_generator_lane_streams = True
    engine.settings = SimpleNamespace(
        engine=SimpleNamespace(
            max_active_requests=4,
            prefill_token_budget=32,
        )
    )

    engine._create_batch_generator()

    assert captured["model"] is engine._model
    assert captured["stream"] is stream


def test_batched_engine_abort_removes_request_from_its_lane() -> None:
    removed: list[list[int]] = []

    class Generator:
        def remove(self, uids: list[int]) -> None:
            removed.append(uids)

    profile = (4, False, 0)
    lane = _BatchLane(profile=profile, generator=Generator())
    lane.request_ids.add("cancel")
    lane.uid_to_rid[41] = "cancel"
    lane.rid_to_uid["cancel"] = 41

    engine = object.__new__(BatchedEngine)
    engine._batch_lanes = {profile: lane}
    engine._pending_aborts = {"cancel"}
    engine._running = {"cancel"}
    engine._state = {
        "cancel": _RequestState(
            request_id="cancel",
            request=InferenceRequest(prompt="cancel", trace_id="cancel"),
            prompt_tokens=[1, 2, 3, 4],
            batch_profile=profile,
        )
    }
    engine._cancelled_requests = 1

    engine._process_aborts()

    assert removed == [[41]]
    assert lane.request_ids == set()
    assert lane.uid_to_rid == {}
    assert lane.rid_to_uid == {}
    assert engine._running == set()
    assert engine._state == {}


def test_batched_engine_extracts_prompt_cache_from_the_owning_lane() -> None:
    captured: list[dict[str, object]] = []

    class Generator:
        def extract_cache(self, uids: list[int]) -> dict[int, object]:
            return {uids[0]: ([_FakeKVCache(3)], [10, 11, 12])}

    profile = (4, False, 0)
    lane = _BatchLane(profile=profile, generator=Generator())
    lane.uid_to_rid[7] = "request-7"

    engine = object.__new__(BatchedEngine)
    engine._batch_lanes = {profile: lane}
    engine._state = {
        "request-7": _RequestState(
            request_id="request-7",
            request=InferenceRequest(prompt="ignored", trace_id="request-7"),
            prompt_tokens=[10, 11, 12, 13],
            batch_profile=profile,
        )
    }
    engine.prefix_store = SimpleNamespace(
        enabled=True,
        store=lambda **kwargs: captured.append(kwargs),
    )
    engine._model_fingerprint = "fingerprint"
    engine.settings = SimpleNamespace(model=SimpleNamespace(name="test-model"))

    engine._process_prompt_responses(
        lane,
        [
            SimpleNamespace(
                uid=7,
                end_of_segment=True,
                end_of_prompt=False,
                progress=(3, 4),
            )
        ],
    )

    assert len(captured) == 1
    assert captured[0]["cache_token_count"] == 3


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
