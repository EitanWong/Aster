from __future__ import annotations

import math

import numpy as np
import pytest

from aster.inference.constrained import build_json_logits_processor
from aster.inference.constrained.json_schema_processor import ThinkingAwareJsonLogitsProcessor


class AsciiTokenizer:
    vocab_size = 128
    all_special_ids = [0]
    eos_token_id = 0

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [ord(char) for char in text]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(token_id) for token_id in tokens if token_id)


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
