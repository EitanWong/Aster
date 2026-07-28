from __future__ import annotations

from typing import Any

import pytest

from aster.core.config import RuntimeSettings
from aster.inference.contracts import InferenceRequest
from aster.inference.engine import InferenceEngine
from aster.inference.model_runner import DecodeWorkItem, ModelRunner
from aster.inference.request_state import RequestState


class _Tokenizer:
    eos_token_ids: list[int] = []
    eos_token_id = None
    detokenizer = object()


def _runner() -> tuple[ModelRunner, list[dict[str, Any]]]:
    runner = ModelRunner(RuntimeSettings.model_validate({"embeddings": {"enabled": False}}))
    runner._loaded = True
    runner._tokenizer = _Tokenizer()
    runner._model = object()
    runner._make_prompt_cache = lambda _model: []
    runner._make_sampler = lambda **_kwargs: "sampler"
    calls: list[dict[str, Any]] = []

    def make_processors(**kwargs: Any) -> list[Any]:
        calls.append(kwargs)
        active = any(
            kwargs[name] not in (None, 0.0)
            for name in (
                "repetition_penalty",
                "presence_penalty",
                "frequency_penalty",
            )
        )
        return [lambda tokens, logits: logits] if active else []

    runner._make_logits_processors = make_processors
    return runner, calls


def test_initialize_decode_omits_neutral_repetition_processor() -> None:
    runner, calls = _runner()

    initialized = runner.initialize_decode(
        prompt_tokens=[1, 2, 3],
        cache_token_count=0,
        prompt_cache=None,
        request=InferenceRequest(prompt="ignored"),
    )

    assert calls == [
        {
            "repetition_penalty": None,
            "repetition_context_size": 20,
            "presence_penalty": 0.0,
            "presence_context_size": 20,
            "frequency_penalty": 0.0,
            "frequency_context_size": 20,
        }
    ]
    assert initialized.logits_processors == ()
    assert initialized.logits_processor_context_size == 0


def test_initialize_decode_bounds_builtin_penalty_context() -> None:
    runner, _calls = _runner()

    initialized = runner.initialize_decode(
        prompt_tokens=[1, 2, 3],
        cache_token_count=0,
        prompt_cache=None,
        request=InferenceRequest(prompt="ignored", repetition_penalty=1.1),
    )

    assert len(initialized.logits_processors) == 1
    assert initialized.logits_processor_context_size == 20


def test_initialize_decode_preserves_full_context_for_structured_processor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, _calls = _runner()
    processor = object()
    monkeypatch.setattr(
        "aster.inference.model_runner.build_json_logits_processor",
        lambda _schema, _tokenizer: processor,
    )

    initialized = runner.initialize_decode(
        prompt_tokens=[1, 2, 3],
        cache_token_count=0,
        prompt_cache=None,
        request=InferenceRequest(
            prompt="ignored",
            structured_output_schema={"type": "object"},
        ),
    )

    assert initialized.logits_processors == (processor,)
    assert initialized.logits_processor_context_size is None


def test_engine_builds_only_preceding_bounded_processor_tokens() -> None:
    state = RequestState(
        request_id="bounded-processor",
        request=InferenceRequest(prompt="ignored"),
        prompt_tokens=list(range(100)),
    )
    state.output_token_ids = list(range(100, 111))
    state.next_input_token = 110
    state.decode_logits_processors = (object(),)
    state.decode_logits_processor_context_size = 20

    tokens = InferenceEngine._logits_processor_tokens(state)

    assert tokens == list(range(91, 110))


def test_engine_bounds_initial_decode_without_duplicating_current_token() -> None:
    state = RequestState(
        request_id="initial-bounded-processor",
        request=InferenceRequest(prompt="ignored"),
        prompt_tokens=list(range(100)),
    )
    state.next_input_token = 99
    state.decode_logits_processors = (object(),)
    state.decode_logits_processor_context_size = 20

    tokens = InferenceEngine._logits_processor_tokens(state)

    assert tokens == list(range(80, 99))


class _MX:
    uint32 = "uint32"

    def __init__(self) -> None:
        self.values: list[int] | None = None

    def array(self, values: list[int], *, dtype: str) -> list[int]:
        assert dtype == self.uint32
        self.values = values
        return values


def test_runner_defensively_bounds_processor_input() -> None:
    runner = ModelRunner(RuntimeSettings.model_validate({"embeddings": {"enabled": False}}))
    mx = _MX()
    runner._mx = mx
    observed: list[int] = []

    def processor(tokens: list[int], logits: object) -> object:
        observed.extend(tokens)
        return logits

    item = DecodeWorkItem(
        prompt_cache=[],
        input_token=30,
        sampler=lambda value: value,
        detokenizer=object(),
        stop_token_ids=frozenset(),
        logits_processors=(processor,),
        logits_processor_tokens=list(range(30)),
        completion_tokens=0,
        max_tokens=1,
        logits_processor_context_size=20,
    )

    logits = object()
    assert runner._apply_logits_processors(logits, item=item) is logits
    assert observed == list(range(11, 31))
    assert mx.values == list(range(11, 31))
