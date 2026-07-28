from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from aster.inference.constrained import build_json_logits_processor
from aster.inference.constrained.json_schema_processor import (
    JSONSchemaLogitsProcessor,
    ThinkingAwareJsonLogitsProcessor,
)


class AsciiTokenizer:
    vocab_size = 128
    all_special_ids = [0]
    eos_token_id = 0

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [ord(char) for char in text]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(token_id) for token_id in tokens if token_id)


class StaticAllowedTokenEnforcer:
    def __init__(self, allowed_tokens: Any) -> None:
        self._result = SimpleNamespace(allowed_tokens=allowed_tokens)

    def get_allowed_tokens(self, suffix: list[int]) -> Any:
        del suffix
        return self._result


class ContainsCountingList(list[int]):
    def __init__(self, values: list[int]) -> None:
        super().__init__(values)
        self.contains_calls = 0

    def __contains__(self, value: object) -> bool:
        self.contains_calls += 1
        return super().__contains__(value)


def processor_with_allowed_tokens(allowed_tokens: Any) -> JSONSchemaLogitsProcessor:
    processor = object.__new__(JSONSchemaLogitsProcessor)
    processor._enforcer = StaticAllowedTokenEnforcer(allowed_tokens)
    processor._json_context = lambda suffix: "other"
    processor._eos_token_ids = set()
    processor._is_complete_json = lambda suffix: False
    processor._mask_cache_key = None
    processor._mask_cache_allowed = None
    processor._mask_cache_value = None
    processor._mask_cache_contains_eos = None
    processor._pending_mask_allowed = None
    processor._pending_mask_contains_eos = False
    return processor


def test_json_schema_logits_processor_builds_from_schema() -> None:
    processor = build_json_logits_processor(
        {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]},
        AsciiTokenizer(),
    )

    assert processor is not None
    assert processor.name == "json_schema"


def test_json_schema_logits_processor_masks_non_json_start_tokens() -> None:
    mx = pytest.importorskip("mlx.core")
    processor = build_json_logits_processor({"type": "object"}, AsciiTokenizer())
    assert processor is not None

    logits = mx.zeros((AsciiTokenizer.vocab_size,))
    masked = processor(mx.array([ord(" ")], dtype=mx.uint32), logits)
    values = np.array(masked)

    assert math.isfinite(values[ord("{")])
    assert not math.isfinite(values[ord("x")])


def test_json_schema_logits_processor_uses_generated_suffix_after_prompt() -> None:
    processor = build_json_logits_processor({"type": "object"}, AsciiTokenizer())
    assert processor is not None

    assert processor._generated_suffix([1, 2, 3]) == []
    assert processor._generated_suffix([1, 2, 3, ord("{")]) == [ord("{")]


def test_json_schema_logits_processor_reuses_native_allowed_token_list() -> None:
    cached_allowed = [1, 2, 3]
    processor = processor_with_allowed_tokens(cached_allowed)

    assert processor._allowed_tokens([]) is cached_allowed


def test_json_schema_logits_processor_normalizes_non_native_allowed_tokens() -> None:
    processor = processor_with_allowed_tokens((np.int64(1), np.int64(2)))

    allowed = processor._allowed_tokens([])

    assert allowed == [1, 2]
    assert all(type(token_id) is int for token_id in allowed)


def test_json_schema_logits_processor_filters_eos_without_mutating_cached_list() -> None:
    cached_allowed = [1, 2, 3]
    processor = processor_with_allowed_tokens(cached_allowed)
    processor._eos_token_ids = {2}

    assert processor._allowed_tokens([]) == [1, 3]
    assert cached_allowed == [1, 2, 3]

    processor._is_complete_json = lambda suffix: True
    assert processor._allowed_tokens([]) is cached_allowed


def test_json_schema_logits_processor_reuses_cached_eos_membership() -> None:
    mx = pytest.importorskip("mlx.core")
    cached_allowed = ContainsCountingList([1, 2, 3])
    processor = processor_with_allowed_tokens(cached_allowed)
    processor._eos_token_ids = {2}
    processor._is_complete_json = lambda suffix: True

    first_allowed = processor._allowed_tokens([])
    processor._mask(first_allowed, mx.zeros((AsciiTokenizer.vocab_size,)))
    second_allowed = processor._allowed_tokens([])

    assert second_allowed is cached_allowed
    assert cached_allowed.contains_calls == 1


def test_json_schema_logits_processor_rechecks_eos_after_allowed_list_mutation() -> None:
    mx = pytest.importorskip("mlx.core")
    cached_allowed = ContainsCountingList([1, 2, 3, 4, 5])
    processor = processor_with_allowed_tokens(cached_allowed)
    processor._eos_token_ids = {9}

    first_allowed = processor._allowed_tokens([])
    processor._mask(first_allowed, mx.zeros((AsciiTokenizer.vocab_size,)))
    cached_allowed[1] = 9
    cached_allowed[3] = 8

    assert processor._allowed_tokens([]) == [1, 3, 8, 5]
    assert cached_allowed.contains_calls == 2


def test_json_schema_logits_processor_cached_eos_still_checks_json_completion() -> None:
    mx = pytest.importorskip("mlx.core")
    cached_allowed = [1, 2, 3]
    processor = processor_with_allowed_tokens(cached_allowed)
    processor._eos_token_ids = {2}
    processor._is_complete_json = lambda suffix: True

    first_allowed = processor._allowed_tokens([])
    processor._mask(first_allowed, mx.zeros((AsciiTokenizer.vocab_size,)))
    processor._is_complete_json = lambda suffix: False

    assert processor._allowed_tokens([]) == [1, 3]
    assert cached_allowed == [1, 2, 3]


def test_json_schema_logits_processor_filters_key_start_tokens() -> None:
    processor = build_json_logits_processor(
        {"type": "object", "properties": {"answer": {"type": "string"}}},
        AsciiTokenizer(),
    )
    assert processor is not None

    allowed = [ord('"'), ord("a"), ord("x"), ord("}"), ord(" ")]
    filtered = processor._filter_at_key_context("key_start", [ord("{")], allowed)

    assert ord('"') in filtered
    assert ord("}") in filtered
    assert ord(" ") in filtered
    assert ord("a") not in filtered
    assert ord("x") not in filtered


def test_json_schema_logits_processor_filters_in_key_prefix_tokens() -> None:
    processor = build_json_logits_processor(
        {"type": "object", "properties": {"answer": {"type": "string"}}},
        AsciiTokenizer(),
    )
    assert processor is not None

    suffix = AsciiTokenizer().encode('{"an')
    allowed = [ord("s"), ord("x"), ord('"'), ord(" ")]
    filtered = processor._filter_at_key_context("in_key", suffix, allowed)

    assert ord("s") in filtered
    assert ord("x") not in filtered
    assert ord('"') not in filtered
    assert ord(" ") not in filtered


def test_json_schema_logits_processor_distinguishes_object_and_array_commas() -> None:
    processor = build_json_logits_processor(
        {
            "type": "array",
            "items": {"type": "object", "properties": {"answer": {"type": "string"}}},
        },
        AsciiTokenizer(),
    )
    assert processor is not None
    tokenizer = AsciiTokenizer()

    assert processor._json_context(tokenizer.encode('{"answer":"yes",')) == "key_start"
    assert processor._json_context(tokenizer.encode("[1,")) == "other"
    assert processor._json_context(tokenizer.encode('[{"answer":"yes"},')) == "other"
    assert processor._json_context(tokenizer.encode("[{")) == "key_start"


def test_json_schema_logits_processor_disabled_state_forces_eos() -> None:
    mx = pytest.importorskip("mlx.core")
    processor = build_json_logits_processor({"type": "object"}, AsciiTokenizer())
    assert processor is not None
    processor._disabled = True

    logits = mx.zeros((AsciiTokenizer.vocab_size,))
    masked = processor(mx.array([ord("{")], dtype=mx.uint32), logits)
    values = np.array(masked)

    assert math.isfinite(values[AsciiTokenizer.eos_token_id])
    assert not math.isfinite(values[ord("{")])
    assert not math.isfinite(values[ord("x")])


def test_json_schema_logits_processor_reuses_equal_mask() -> None:
    mx = pytest.importorskip("mlx.core")
    processor = build_json_logits_processor({"type": "object"}, AsciiTokenizer())
    assert processor is not None
    logits = mx.zeros((AsciiTokenizer.vocab_size,))

    first = processor._mask([1, 2, 3], logits)
    second = processor._mask([1, 2, 3], logits)

    assert second is first


def test_json_schema_logits_processor_verifies_mask_fingerprint_collisions() -> None:
    mx = pytest.importorskip("mlx.core")
    processor = build_json_logits_processor({"type": "object"}, AsciiTokenizer())
    assert processor is not None
    logits = mx.zeros((AsciiTokenizer.vocab_size,))

    first = processor._mask([1, 2, 3, 4, 5], logits)
    second = processor._mask([1, 9, 3, 8, 5], logits)
    values = np.array(second)

    assert second is not first
    assert math.isfinite(values[9])
    assert not math.isfinite(values[2])


def test_json_schema_logits_processor_detects_in_place_allowed_token_changes() -> None:
    mx = pytest.importorskip("mlx.core")
    processor = build_json_logits_processor({"type": "object"}, AsciiTokenizer())
    assert processor is not None
    logits = mx.zeros((AsciiTokenizer.vocab_size,))
    allowed = [1, 2, 3, 4, 5]

    first = processor._mask(allowed, logits)
    allowed[1] = 9
    second = processor._mask(allowed, logits)
    values = np.array(second)

    assert second is not first
    assert math.isfinite(values[9])
    assert not math.isfinite(values[2])


def test_json_schema_logits_processor_invalidates_mask_for_logit_shape() -> None:
    mx = pytest.importorskip("mlx.core")
    processor = build_json_logits_processor({"type": "object"}, AsciiTokenizer())
    assert processor is not None

    vector = processor._mask([1, 2, 3], mx.zeros((AsciiTokenizer.vocab_size,)))
    row = processor._mask([1, 2, 3], mx.zeros((1, AsciiTokenizer.vocab_size)))

    assert vector.shape == (AsciiTokenizer.vocab_size,)
    assert row.shape == (1, AsciiTokenizer.vocab_size)
    assert row is not vector


class RecordingInner:
    schema = {"type": "object"}
    _disabled = False

    def __init__(self) -> None:
        self._prompt_len = None
        self.calls: list[tuple[list[int], object]] = []

    def __call__(self, tokens, logits):
        self.calls.append((list(tokens), logits))
        return "masked"


def test_thinking_aware_json_processor_waits_until_think_closes_and_json_starts() -> None:
    inner = RecordingInner()
    processor = ThinkingAwareJsonLogitsProcessor(
        inner,
        tokenizer=AsciiTokenizer(),
        prompt_has_think_tag=True,
    )

    logits = object()
    prompt = [1, 2, 3]

    assert processor(prompt, logits) is logits
    assert processor(prompt + AsciiTokenizer().encode("reasoning</think> text"), logits) is logits
    assert processor(prompt + AsciiTokenizer().encode("reasoning</think> text {"), logits) == "masked"

    assert inner.calls
    assert inner._prompt_len == len(prompt + AsciiTokenizer().encode("reasoning</think> text "))


def test_thinking_aware_json_processor_activates_without_think_block() -> None:
    inner = RecordingInner()
    processor = ThinkingAwareJsonLogitsProcessor(inner, tokenizer=AsciiTokenizer())

    logits = object()
    prompt = [1, 2, 3]

    assert processor(prompt, logits) is logits
    assert processor(prompt + AsciiTokenizer().encode("abc{"), logits) == "masked"

    assert inner._prompt_len == len(prompt + AsciiTokenizer().encode("abc"))
