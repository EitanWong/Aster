from __future__ import annotations

import functools
import math
import time
from collections import defaultdict
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class TimingSample:
    name: str
    elapsed_ns: int
    phase: str = "other"
    query_tokens: int | None = None
    kv_tokens: int | None = None


class TimingCollector:
    def __init__(self) -> None:
        self._samples: list[TimingSample] = []
        self._tokens: list[int] = []
        self._enabled = True

    @property
    def samples(self) -> tuple[TimingSample, ...]:
        return tuple(self._samples)

    @property
    def tokens(self) -> tuple[int, ...]:
        return tuple(self._tokens)

    def clear(self) -> None:
        self._samples.clear()
        self._tokens.clear()

    def record(
        self,
        name: str,
        elapsed_ns: int,
        *,
        phase: str = "other",
        query_tokens: int | None = None,
        kv_tokens: int | None = None,
    ) -> None:
        if not self._enabled:
            return
        if not name or elapsed_ns < 0:
            raise ValueError("Timing samples require a name and non-negative duration")
        self._samples.append(
            TimingSample(
                name=name,
                elapsed_ns=int(elapsed_ns),
                phase=phase,
                query_tokens=query_tokens,
                kv_tokens=kv_tokens,
            )
        )

    def record_token(self, token: int) -> None:
        if self._enabled:
            self._tokens.append(int(token))

    @contextmanager
    def paused(self):
        previous = self._enabled
        self._enabled = False
        try:
            yield
        finally:
            self._enabled = previous


def _percentile(values: list[int], quantile: float) -> int:
    if not values:
        return 0
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be between zero and one")
    ordered = sorted(values)
    index = max(math.ceil(quantile * len(ordered)) - 1, 0)
    return ordered[index]


def summarize_samples(samples: tuple[TimingSample, ...]) -> dict[str, dict[str, float | int]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for sample in samples:
        grouped[f"{sample.name}:{sample.phase}"].append(sample.elapsed_ns)

    summary: dict[str, dict[str, float | int]] = {}
    for key, values in sorted(grouped.items()):
        ordered = sorted(values)
        count = len(ordered)
        midpoint = count // 2
        median_ns = (
            ordered[midpoint]
            if count % 2
            else (ordered[midpoint - 1] + ordered[midpoint]) / 2
        )
        summary[key] = {
            "count": count,
            "total_ms": sum(ordered) / 1_000_000,
            "median_ms": median_ns / 1_000_000,
            "p95_ms": _percentile(ordered, 0.95) / 1_000_000,
            "min_ms": ordered[0] / 1_000_000,
            "max_ms": ordered[-1] / 1_000_000,
        }
    return summary


MetadataFactory = Callable[[tuple[Any, ...], dict[str, Any]], Mapping[str, Any]]
ResultCapture = Callable[[Any], None]


def patch_method(
    owner: Any,
    method_name: str,
    collector: TimingCollector,
    event_name: str,
    *,
    metadata: MetadataFactory | None = None,
    capture_result: ResultCapture | None = None,
) -> Callable[[], None]:
    original = getattr(owner, method_name)

    @functools.wraps(original)
    def measured(*args: Any, **kwargs: Any) -> Any:
        details = dict(metadata(args, kwargs)) if metadata is not None else {}
        started = time.perf_counter_ns()
        try:
            result = original(*args, **kwargs)
        finally:
            collector.record(
                event_name,
                time.perf_counter_ns() - started,
                phase=str(details.get("phase", "other")),
                query_tokens=_optional_int(details.get("query_tokens")),
                kv_tokens=_optional_int(details.get("kv_tokens")),
            )
        if capture_result is not None:
            capture_result(result)
        return result

    setattr(owner, method_name, measured)

    def restore() -> None:
        if getattr(owner, method_name) is measured:
            setattr(owner, method_name, original)

    return restore


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def query_tokens(value: Any) -> int | None:
    shape = getattr(value, "shape", None)
    if shape is None or len(shape) < 2:
        return None
    return int(shape[-2])


def phase_for_query(tokens: int | None) -> str:
    if tokens is None:
        return "other"
    return "decode" if tokens <= 8 else "prefill"
