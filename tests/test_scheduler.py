from __future__ import annotations

from aster.inference.request_state import RequestPhase, RequestState
from aster.inference.contracts import InferenceRequest


def test_request_state_starts_submitted() -> None:
    state = RequestState(
        request_id="req-1",
        request=InferenceRequest(prompt="hello"),
    )
    assert state.phase is RequestPhase.SUBMITTED
    assert state.prompt_token_count == 0
    assert state.target_cache_token_count == 0


def test_request_state_target_cache_tokens_tracks_prompt_length() -> None:
    state = RequestState(
        request_id="req-2",
        request=InferenceRequest(prompt="hello"),
        prompt_tokens=[10, 11, 12, 13],
    )
    assert state.prompt_token_count == 4
    assert state.target_cache_token_count == 3


def test_request_state_marks_admission_and_terminal_transition() -> None:
    state = RequestState(
        request_id="req-3",
        request=InferenceRequest(prompt="hello"),
    )

    state.mark_admitted()
    state.mark_terminal(RequestPhase.COMPLETED)

    assert state.admitted_at is not None
    assert state.phase is RequestPhase.COMPLETED
    assert state.completed_at is not None
