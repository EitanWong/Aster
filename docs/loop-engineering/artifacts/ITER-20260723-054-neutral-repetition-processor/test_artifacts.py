from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace


def _load_module():
    path = Path(__file__).with_name("candidate_benchmark.py")
    spec = importlib.util.spec_from_file_location("iter054_candidate_benchmark", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_candidate_removes_only_leading_neutral_repetition_processor() -> None:
    module = _load_module()
    sampler = SimpleNamespace()
    setattr(sampler, module.NEUTRAL_REPETITION_PROCESSOR, True)
    neutral = object()
    remaining = object()
    item = SimpleNamespace(
        sampler=sampler,
        logits_processors=(neutral, remaining),
    )

    processors, skipped = module._candidate_processors(item)

    assert processors == (remaining,)
    assert skipped is True


def test_candidate_preserves_non_neutral_processors() -> None:
    module = _load_module()
    processor = object()
    item = SimpleNamespace(
        sampler=SimpleNamespace(),
        logits_processors=(processor,),
    )

    processors, skipped = module._candidate_processors(item)

    assert processors == (processor,)
    assert skipped is False
