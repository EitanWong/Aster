from __future__ import annotations

import pytest

from aster.inference.thinking_processor import (
    BoundedSuffixMatcher,
    ThinkingAwareLogitsProcessor,
    ThinkingPhase,
)


def test_bounded_suffix_matcher_detects_overlapping_suffix() -> None:
    matcher = BoundedSuffixMatcher([1, 2, 1, 2])

    assert not matcher.feed(1)
    assert not matcher.feed(2)
    assert not matcher.feed(1)
    assert not matcher.feed(1)
    assert not matcher.feed(2)
    assert not matcher.feed(1)
    assert matcher.feed(2)


def test_bounded_suffix_matcher_requires_target() -> None:
    with pytest.raises(ValueError):
        BoundedSuffixMatcher([])


def test_thinking_processor_transitions_to_content_on_end_tokens() -> None:
    processor = ThinkingAwareLogitsProcessor(
        start_token_ids=[10],
        end_token_ids=[20, 21],
        thinking_token_budget=8,
    )

    processor.observe_token_ids([10, 100, 101, 20, 21])

    assert processor.state is ThinkingPhase.CONTENT
    assert processor.is_retired


def test_thinking_processor_enters_transitioning_when_budget_is_exhausted() -> None:
    processor = ThinkingAwareLogitsProcessor(
        start_token_ids=[10],
        end_token_ids=[20],
        thinking_token_budget=2,
    )

    processor.observe_token_ids([10, 100, 101])

    assert processor.state is ThinkingPhase.TRANSITIONING
    assert processor.thinking_tokens == 2

