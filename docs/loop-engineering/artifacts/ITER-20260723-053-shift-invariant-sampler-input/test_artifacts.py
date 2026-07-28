from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    path = Path(__file__).with_name("candidate_benchmark.py")
    spec = importlib.util.spec_from_file_location("iter053_candidate_benchmark", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_temperature_zero_is_shift_invariant_even_with_top_p() -> None:
    module = _load_module()
    assert module.sampler_accepts_raw_logits(temperature=0.0, top_p=0.95)


def test_top_p_requires_normalized_logprobs() -> None:
    module = _load_module()
    assert not module.sampler_accepts_raw_logits(temperature=0.7, top_p=0.95)


def test_top_k_and_min_p_only_sampler_is_shift_invariant() -> None:
    module = _load_module()
    assert module.sampler_accepts_raw_logits(temperature=0.9, top_p=1.0)
    assert module.sampler_accepts_raw_logits(temperature=0.9, top_p=0.0)
