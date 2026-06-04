from __future__ import annotations

from aster.api.responses_store import DEFAULT_RESPONSES_STORE_MAX_SIZE, ResponsesStore


def test_responses_store_default_capacity_matches_vllm_mlx() -> None:
    assert DEFAULT_RESPONSES_STORE_MAX_SIZE == 1000


def test_responses_store_evicts_least_recently_used_response() -> None:
    store = ResponsesStore(max_size=2)
    assert store.max_size == 2
    assert store.put("resp-a", [{"role": "user", "content": "A"}]) == 0
    assert store.put("resp-b", [{"role": "user", "content": "B"}]) == 0
    assert len(store) == 2

    assert store.get("resp-a") == [{"role": "user", "content": "A"}]

    assert store.put("resp-c", [{"role": "user", "content": "C"}]) == 1

    assert store.get("resp-b") is None
    assert store.get("resp-a") == [{"role": "user", "content": "A"}]
    assert store.get("resp-c") == [{"role": "user", "content": "C"}]
    assert len(store) == 2


def test_responses_store_copies_messages_on_put_and_get() -> None:
    store = ResponsesStore(max_size=2)
    messages = [{"role": "user", "content": "original"}]

    store.put("resp-a", messages)
    messages[0]["content"] = "mutated-after-put"

    stored = store.get("resp-a")
    assert stored == [{"role": "user", "content": "original"}]
    assert stored is not None
    stored[0]["content"] = "mutated-after-get"

    assert store.get("resp-a") == [{"role": "user", "content": "original"}]


def test_responses_store_scopes_response_ids() -> None:
    store = ResponsesStore(max_size=2)

    assert store.put("resp-a", [{"role": "user", "content": "OpenAI"}], scope="openai") == 0
    assert store.get("resp-a", scope="xai") is None
    assert store.get("resp-a", scope="openai") == [{"role": "user", "content": "OpenAI"}]
