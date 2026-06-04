from __future__ import annotations

import enum
from collections import deque
from collections.abc import Callable
from typing import Any


class BoundedSuffixMatcher:
    __slots__ = ("target", "_buf", "_max_len")

    def __init__(self, target_ids: list[int]) -> None:
        if not target_ids:
            raise ValueError("target_ids must be non-empty")
        self.target = tuple(target_ids)
        self._max_len = len(target_ids)
        self._buf: deque[int] = deque(maxlen=self._max_len)

    def feed(self, token_id: int) -> bool:
        self._buf.append(token_id)
        return len(self._buf) == self._max_len and tuple(self._buf) == self.target

    def reset(self) -> None:
        self._buf.clear()

    def snapshot(self) -> tuple[int, ...]:
        return tuple(self._buf)

    def restore(self, state: tuple[int, ...]) -> None:
        self._buf.clear()
        self._buf.extend(state)


class ThinkingPhase(enum.Enum):
    IDLE = "idle"
    THINKING = "thinking"
    TRANSITIONING = "transitioning"
    CONTENT = "content"


class ThinkingAwareLogitsProcessor:
    __slots__ = (
        "_start_matcher",
        "_end_matcher",
        "_end_token_ids",
        "_content_phase_mask_ids",
        "_thinking_token_budget",
        "_inner",
        "_state",
        "_thinking_tokens",
        "_transition_index",
        "_processed_len",
        "_processed_token_ids",
        "_snapshots",
    )

    def __init__(
        self,
        *,
        start_token_ids: list[int],
        end_token_ids: list[int],
        thinking_token_budget: int,
        inner: Callable[[Any, Any], Any] | None = None,
        prompt_has_think_tag: bool = False,
    ) -> None:
        self._start_matcher = BoundedSuffixMatcher(start_token_ids)
        self._end_matcher = BoundedSuffixMatcher(end_token_ids)
        self._end_token_ids = list(end_token_ids)
        self._content_phase_mask_ids = tuple(dict.fromkeys([start_token_ids[0], end_token_ids[0]]))
        self._thinking_token_budget = max(0, thinking_token_budget)
        self._inner = inner
        self._thinking_tokens = 0
        self._transition_index = 0
        self._state = ThinkingPhase.THINKING if prompt_has_think_tag else ThinkingPhase.IDLE
        if prompt_has_think_tag and self._thinking_token_budget == 0:
            self._state = ThinkingPhase.TRANSITIONING
        self._processed_len = 0
        self._processed_token_ids: list[int] = []
        self._snapshots = [self._snapshot_state()]

    @property
    def state(self) -> ThinkingPhase:
        return self._state

    @property
    def thinking_tokens(self) -> int:
        return self._thinking_tokens

    @property
    def is_retired(self) -> bool:
        return self._state == ThinkingPhase.CONTENT and self._inner is None

    def observe_token_ids(self, token_ids: list[int]) -> None:
        self._sync_to_token_ids(token_ids)

    def __call__(self, tokens: Any, logits: Any) -> Any:
        if int(tokens.size) == 0:
            if self._state == ThinkingPhase.TRANSITIONING:
                return self._force_transition(logits)
            if self._state == ThinkingPhase.CONTENT:
                return self._call_inner(tokens, logits)
            return logits

        self._sync_to_token_ids([int(token_id) for token_id in tokens.tolist()])
        if self._state == ThinkingPhase.TRANSITIONING:
            return self._force_transition(logits)
        if self._state == ThinkingPhase.CONTENT:
            return self._call_inner(tokens, logits)
        return logits

    def _force_transition(self, logits: Any) -> Any:
        try:
            import mlx.core as mx
        except Exception as exc:  # pragma: no cover - depends on local runtime
            raise RuntimeError("MLX is required to force thinking transition logits.") from exc

        target_id = self._end_token_ids[self._transition_index]
        masked = mx.full(logits.shape, float("-inf"))
        if masked.ndim == 1:
            masked[target_id] = 0.0
        else:
            masked[..., target_id] = 0.0
        return masked

    def _call_inner(self, tokens: Any, logits: Any) -> Any:
        if self._inner is not None:
            logits = self._inner(tokens, logits)
        for token_id in self._content_phase_mask_ids:
            if logits.ndim == 1:
                logits[token_id] = float("-inf")
            else:
                logits[..., token_id] = float("-inf")
        return logits

    def _snapshot_state(
        self,
    ) -> tuple[ThinkingPhase, int, int, tuple[int, ...], tuple[int, ...]]:
        return (
            self._state,
            self._thinking_tokens,
            self._transition_index,
            self._start_matcher.snapshot(),
            self._end_matcher.snapshot(),
        )

    def _restore_snapshot(self, processed_len: int) -> None:
        snap_idx = min(processed_len, len(self._snapshots) - 1)
        (
            self._state,
            self._thinking_tokens,
            self._transition_index,
            start_state,
            end_state,
        ) = self._snapshots[snap_idx]
        self._start_matcher.restore(start_state)
        self._end_matcher.restore(end_state)
        self._processed_len = processed_len
        self._processed_token_ids = self._processed_token_ids[:processed_len]
        self._snapshots = self._snapshots[: snap_idx + 1]

    def _sync_to_token_ids(self, token_ids: list[int]) -> None:
        target_len = len(token_ids)
        common_len = 0
        max_common = min(target_len, self._processed_len)
        while common_len < max_common and self._processed_token_ids[common_len] == token_ids[common_len]:
            common_len += 1
        if common_len < self._processed_len:
            self._restore_snapshot(common_len)
        if target_len == self._processed_len:
            return
        for token_id in token_ids[self._processed_len :]:
            self._advance_with_token(token_id)
            self._processed_token_ids.append(token_id)
            self._processed_len += 1
            if self._state != ThinkingPhase.CONTENT:
                self._snapshots.append(self._snapshot_state())

    def _advance_with_token(self, token_id: int) -> None:
        if self._state == ThinkingPhase.IDLE:
            if self._start_matcher.feed(token_id):
                self._state = ThinkingPhase.THINKING
                if self._thinking_token_budget == 0:
                    self._state = ThinkingPhase.TRANSITIONING
                    self._transition_index = 0
            return

        if self._state == ThinkingPhase.THINKING:
            if self._end_matcher.feed(token_id):
                self._state = ThinkingPhase.CONTENT
                return
            self._thinking_tokens += 1
            if self._thinking_tokens >= self._thinking_token_budget:
                self._state = ThinkingPhase.TRANSITIONING
                self._transition_index = 0
            return

        if self._state == ThinkingPhase.TRANSITIONING:
            expected = self._end_token_ids[self._transition_index]
            if token_id == expected:
                self._transition_index += 1
                if self._transition_index >= len(self._end_token_ids):
                    self._state = ThinkingPhase.CONTENT
                    self._end_matcher.reset()

