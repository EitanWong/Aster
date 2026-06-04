from __future__ import annotations

from aster.inference.prefix_store import PrefixStore


def test_pin_protects_snapshot_from_eviction() -> None:
    store = PrefixStore(
        budget_bytes=128,
        max_entries=4,
        min_prefix_tokens=1,
        enabled=True,
    )
    pinned = store.store(
        model_name="m",
        prefix_tokens=[1, 2, 3],
        cache_token_count=2,
        prompt_cache={"cache": "pinned"},
        approx_bytes=96,
    )
    assert pinned is not None
    store.pin(pinned.key)

    store.store(
        model_name="m",
        prefix_tokens=[4, 5, 6],
        cache_token_count=2,
        prompt_cache={"cache": "other"},
        approx_bytes=96,
    )

    hit = store.lookup("m", [1, 2, 3, 4])
    assert hit is not None
    assert list(hit.prefix_tokens) == [1, 2, 3]
