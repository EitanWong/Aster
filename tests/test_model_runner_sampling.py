from __future__ import annotations

from typing import Any

from aster.core.config import RuntimeSettings
from aster.inference.model_runner import DecodeResult, DecodeWorkItem, ModelRunner


def test_batch_sampling_groups_lazy_rows_and_preserves_python_values() -> None:
    events: list[str] = []

    class FakeTensor:
        shape = ()
        dtype = "float16"

        def __init__(self, row: int | None = None) -> None:
            self.row = row

        def __getitem__(self, key: Any) -> FakeTensor:
            if isinstance(key, tuple):
                return self
            if isinstance(key, slice):
                return FakeTensor(int(key.start or 0))
            return self

        def __sub__(self, _other: Any) -> FakeTensor:
            return self

    class FakeSample:
        shape = ()
        dtype = "uint32"

        def __init__(self, token: int) -> None:
            self.token = token
            self.evaluated = False
            self.item_calls = 0

        def item(self) -> int:
            self.item_calls += 1
            events.append(f"item:{self.token}")
            if not self.evaluated:
                raise AssertionError("sample materialized before grouped evaluation")
            return self.token

    class FakeMX:
        uint32 = "uint32"

        def __init__(self) -> None:
            self.eval_calls: list[Any] = []
            self.async_eval_calls: list[Any] = []

        @staticmethod
        def array(_value: Any, *, dtype: Any = None) -> FakeTensor:
            del dtype
            return FakeTensor()

        @staticmethod
        def logsumexp(value: FakeTensor, *, axis: int, keepdims: bool) -> FakeTensor:
            del axis, keepdims
            return value

        def async_eval(self, value: Any) -> None:
            assert all(hasattr(sample, "shape") for sample in value)
            events.append("async_eval")
            self.async_eval_calls.append(value)

        def eval(self, value: Any) -> None:
            assert all(hasattr(sample, "shape") for sample in value)
            events.append("eval")
            self.eval_calls.append(value)
            if isinstance(value, list):
                for sample in value:
                    if isinstance(sample, FakeSample):
                        sample.evaluated = True

        @staticmethod
        def get_peak_memory() -> int:
            return 0

    class FakeLayer:
        state = object()

    class FakeDetokenizer:
        last_segment = ""

        def add_token(self, token: int) -> None:
            self.last_segment = chr(token)

    runner = ModelRunner(RuntimeSettings.model_validate({"embeddings": {"enabled": False}}))
    fake_mx = FakeMX()
    runner._loaded = True
    runner._mx = fake_mx
    model_logits = FakeTensor()
    runner._model = lambda _tokens, *, cache: model_logits  # type: ignore[assignment]
    runner._get_decode_batch_cache = lambda _items: ([FakeLayer()], None)  # type: ignore[method-assign]
    runner._extract_prompt_cache = lambda _cache, index: [index]  # type: ignore[method-assign]

    lazy_sample = FakeSample(65)
    samples: list[Any] = [lazy_sample, [66]]
    processor_rows: list[int] = []
    sampler_rows: list[int] = []

    def processor(_tokens: Any, logits: FakeTensor) -> FakeTensor:
        assert logits.row is not None
        processor_rows.append(logits.row)
        events.append(f"processor:{logits.row}")
        return logits

    def sampler_for(row: int):
        def sample(logprobs: FakeTensor) -> Any:
            assert logprobs.row == row
            sampler_rows.append(row)
            events.append(f"sampler:{row}")
            return samples[row]

        return sample

    items = [
        DecodeWorkItem(
            prompt_cache=[],
            input_token=row + 1,
            sampler=sampler_for(row),
            detokenizer=FakeDetokenizer(),
            stop_token_ids=frozenset(),
            logits_processors=(processor,),
            logits_processor_tokens=[10 + row],
            completion_tokens=0,
            max_tokens=2,
            request_id=f"request-{row}",
        )
        for row in range(2)
    ]

    results = runner.decode_batch_step(items)

    assert all(isinstance(result, DecodeResult) for result in results)
    assert [result.token_id for result in results if isinstance(result, DecodeResult)] == [
        65,
        66,
    ]
    assert processor_rows == [0, 1]
    assert sampler_rows == [0, 1]
    assert len(fake_mx.async_eval_calls) == 1
    assert fake_mx.async_eval_calls[0] == [model_logits, lazy_sample]
    assert len(fake_mx.eval_calls) == 1
    assert fake_mx.eval_calls[0] == [model_logits, lazy_sample]
    assert lazy_sample.item_calls == 1
    assert events.index("eval") < events.index("item:65")


def test_sample_materialization_preserves_supported_sampler_return_shapes() -> None:
    assert ModelRunner._materialize_sampled_token(7) == 7
    assert ModelRunner._materialize_sampled_token([8]) == 8
    assert ModelRunner._materialize_sampled_token((9,)) == 9


def test_mlx_sample_detection_excludes_array_like_non_mlx_values() -> None:
    class MlxArray:
        shape = ()
        dtype = "uint32"

    class ArrayLikeWrapper:
        shape = ()
        dtype = "uint32"

    class RealLikeMX:
        array = MlxArray

    sample = MlxArray()
    wrapper = ArrayLikeWrapper()

    lazy_samples, trusted_type = ModelRunner._mlx_sample_arrays(
        RealLikeMX(), [sample, wrapper, 7]
    )

    assert lazy_samples == [sample]
    assert trusted_type is True


def test_eager_row_mode_is_visible_through_processor_wrappers() -> None:
    class HostDrivenProcessor:
        batch_sampling_mode = "eager_rows"

    class Wrapper:
        def __init__(self, inner: Any) -> None:
            self._inner = inner

    item = DecodeWorkItem(
        prompt_cache=[],
        input_token=1,
        sampler=lambda _logprobs: 1,
        detokenizer=object(),
        stop_token_ids=frozenset(),
        logits_processors=(Wrapper(HostDrivenProcessor()),),
        logits_processor_tokens=[],
        completion_tokens=0,
        max_tokens=2,
    )

    assert ModelRunner._uses_eager_row_sampling([item]) is True


def test_all_python_sampler_values_still_force_model_barrier() -> None:
    class FakeTensor:
        shape = ()
        dtype = "float16"

        def __getitem__(self, _key: Any) -> FakeTensor:
            return self

        def __sub__(self, _other: Any) -> FakeTensor:
            return self

    class FakeMX:
        uint32 = "uint32"

        def __init__(self) -> None:
            self.async_eval_calls: list[Any] = []
            self.eval_calls: list[Any] = []

        @staticmethod
        def array(_value: Any, *, dtype: Any = None) -> FakeTensor:
            del dtype
            return FakeTensor()

        @staticmethod
        def logsumexp(value: FakeTensor, *, axis: int, keepdims: bool) -> FakeTensor:
            del axis, keepdims
            return value

        def async_eval(self, value: Any) -> None:
            self.async_eval_calls.append(value)

        def eval(self, value: Any) -> None:
            self.eval_calls.append(value)

        @staticmethod
        def get_peak_memory() -> int:
            return 0

    class FakeLayer:
        state = object()

    class FakeDetokenizer:
        last_segment = ""

        def add_token(self, token: int) -> None:
            self.last_segment = chr(token)

    runner = ModelRunner(RuntimeSettings.model_validate({"embeddings": {"enabled": False}}))
    fake_mx = FakeMX()
    runner._loaded = True
    runner._mx = fake_mx
    model_logits = FakeTensor()
    runner._model = lambda _tokens, *, cache: model_logits  # type: ignore[assignment]
    runner._get_decode_batch_cache = lambda _items: ([FakeLayer()], None)  # type: ignore[method-assign]
    runner._extract_prompt_cache = lambda _cache, index: [index]  # type: ignore[method-assign]
    items = [
        DecodeWorkItem(
            prompt_cache=[],
            input_token=index,
            sampler=lambda _logprobs, token=token: token,
            detokenizer=FakeDetokenizer(),
            stop_token_ids=frozenset(),
            logits_processors=(),
            logits_processor_tokens=[],
            completion_tokens=0,
            max_tokens=2,
        )
        for index, token in ((1, 67), (2, 68))
    ]

    results = runner.decode_batch_step(items)

    assert [result.token_id for result in results if isinstance(result, DecodeResult)] == [
        67,
        68,
    ]
    assert fake_mx.async_eval_calls == [model_logits]
    assert fake_mx.eval_calls == [model_logits]


def test_non_mlx_shaped_sampler_values_still_force_model_barrier() -> None:
    class FakeTensor:
        shape = ()
        dtype = "float16"

        def __getitem__(self, _key: Any) -> FakeTensor:
            return self

        def __sub__(self, _other: Any) -> FakeTensor:
            return self

    class NonMlxSample:
        shape = ()
        dtype = "uint32"

        def __init__(self, token: int) -> None:
            self.token = token

        def item(self) -> int:
            return self.token

    class FakeMX:
        uint32 = "uint32"

        def __init__(self) -> None:
            self.async_eval_calls: list[Any] = []
            self.eval_calls: list[Any] = []

        @staticmethod
        def array(_value: Any, *, dtype: Any = None) -> FakeTensor:
            del dtype
            return FakeTensor()

        @staticmethod
        def logsumexp(value: FakeTensor, *, axis: int, keepdims: bool) -> FakeTensor:
            del axis, keepdims
            return value

        def async_eval(self, value: Any) -> None:
            self.async_eval_calls.append(value)

        def eval(self, value: Any) -> None:
            self.eval_calls.append(value)

        @staticmethod
        def get_peak_memory() -> int:
            return 0

    class FakeLayer:
        state = object()

    class FakeDetokenizer:
        last_segment = ""

        def add_token(self, token: int) -> None:
            self.last_segment = chr(token)

    runner = ModelRunner(RuntimeSettings.model_validate({"embeddings": {"enabled": False}}))
    fake_mx = FakeMX()
    runner._loaded = True
    runner._mx = fake_mx
    model_logits = FakeTensor()
    runner._model = lambda _tokens, *, cache: model_logits  # type: ignore[assignment]
    runner._get_decode_batch_cache = lambda _items: ([FakeLayer()], None)  # type: ignore[method-assign]
    runner._extract_prompt_cache = lambda _cache, index: [index]  # type: ignore[method-assign]
    samples = [NonMlxSample(69), NonMlxSample(70)]
    items = [
        DecodeWorkItem(
            prompt_cache=[],
            input_token=index,
            sampler=lambda _logprobs, sample=sample: sample,
            detokenizer=FakeDetokenizer(),
            stop_token_ids=frozenset(),
            logits_processors=(),
            logits_processor_tokens=[],
            completion_tokens=0,
            max_tokens=2,
        )
        for index, sample in enumerate(samples)
    ]

    results = runner.decode_batch_step(items)

    assert [result.token_id for result in results if isinstance(result, DecodeResult)] == [
        69,
        70,
    ]
    assert fake_mx.async_eval_calls == [[model_logits, *samples]]
    assert fake_mx.eval_calls == [[model_logits, *samples]]


def test_host_driven_processors_keep_eager_row_sampling() -> None:
    class FakeTensor:
        shape = ()
        dtype = "float16"

        def __init__(self, _value: Any = None, *, dtype: Any = None) -> None:
            del dtype

        def __getitem__(self, _key: Any) -> FakeTensor:
            return self

        def __sub__(self, _other: Any) -> FakeTensor:
            return self

    class FakeSample:
        shape = ()
        dtype = "uint32"

        def __init__(self, token: int) -> None:
            self.token = token

        def item(self) -> int:
            return self.token

    class FakeMX:
        uint32 = "uint32"
        array = FakeTensor

        def __init__(self) -> None:
            self.async_eval_calls: list[Any] = []
            self.eval_calls: list[Any] = []

        @staticmethod
        def logsumexp(value: FakeTensor, *, axis: int, keepdims: bool) -> FakeTensor:
            del axis, keepdims
            return value

        def async_eval(self, value: Any) -> None:
            self.async_eval_calls.append(value)

        def eval(self, value: Any) -> None:
            self.eval_calls.append(value)

        @staticmethod
        def get_peak_memory() -> int:
            return 0

    class HostDrivenProcessor:
        batch_sampling_mode = "eager_rows"

        def __call__(self, _tokens: Any, logits: FakeTensor) -> FakeTensor:
            return logits

    class FakeLayer:
        state = object()

    class FakeDetokenizer:
        last_segment = ""

        def add_token(self, token: int) -> None:
            self.last_segment = chr(token)

    runner = ModelRunner(RuntimeSettings.model_validate({"embeddings": {"enabled": False}}))
    fake_mx = FakeMX()
    runner._loaded = True
    runner._mx = fake_mx
    model_logits = FakeTensor()
    runner._model = lambda _tokens, *, cache: model_logits  # type: ignore[assignment]
    runner._get_decode_batch_cache = lambda _items: ([FakeLayer()], None)  # type: ignore[method-assign]
    runner._extract_prompt_cache = lambda _cache, index: [index]  # type: ignore[method-assign]
    processor = HostDrivenProcessor()
    items = [
        DecodeWorkItem(
            prompt_cache=[],
            input_token=index,
            sampler=lambda _logprobs, token=token: FakeSample(token),
            detokenizer=FakeDetokenizer(),
            stop_token_ids=frozenset(),
            logits_processors=(processor,),
            logits_processor_tokens=[],
            completion_tokens=0,
            max_tokens=2,
        )
        for index, token in ((1, 71), (2, 72))
    ]

    results = runner._decode_batch(items)

    assert [result.token_id for result in results] == [71, 72]
    assert fake_mx.eval_calls == [model_logits]
    assert fake_mx.async_eval_calls == []


def test_eager_row_sampling_failure_does_not_replay_samplers() -> None:
    sampler_calls = [0, 0]

    class FakeTensor:
        def __init__(self, _value: Any = None, *, dtype: Any = None) -> None:
            del dtype

        def __getitem__(self, _key: Any) -> FakeTensor:
            return self

        def __sub__(self, _other: Any) -> FakeTensor:
            return self

    class FakeSample:
        def __init__(self, index: int) -> None:
            self.index = index

        def item(self) -> int:
            if self.index == 1:
                raise RuntimeError("eager row materialization failed")
            return 73

    class FakeMX:
        uint32 = "uint32"
        array = FakeTensor

        @staticmethod
        def logsumexp(value: FakeTensor, *, axis: int, keepdims: bool) -> FakeTensor:
            del axis, keepdims
            return value

        @staticmethod
        def eval(_value: Any) -> None:
            return None

        @staticmethod
        def get_peak_memory() -> int:
            return 0

        @staticmethod
        def clear_cache() -> None:
            return None

    class HostDrivenProcessor:
        batch_sampling_mode = "eager_rows"

        def __call__(self, _tokens: Any, logits: FakeTensor) -> FakeTensor:
            return logits

    class FakeLayer:
        state = object()

    class FakeDetokenizer:
        last_segment = ""

        def add_token(self, token: int) -> None:
            self.last_segment = chr(token)

    runner = ModelRunner(RuntimeSettings.model_validate({"embeddings": {"enabled": False}}))
    runner._loaded = True
    runner._mx = FakeMX()
    runner._model = lambda _tokens, *, cache: FakeTensor()  # type: ignore[assignment]
    runner._get_decode_batch_cache = lambda _items: ([FakeLayer()], None)  # type: ignore[method-assign]
    runner._extract_prompt_cache = lambda _cache, index: [index]  # type: ignore[method-assign]

    def sampler_for(index: int):
        def sampler(_logprobs: Any) -> FakeSample:
            sampler_calls[index] += 1
            return FakeSample(index)

        return sampler

    processor = HostDrivenProcessor()
    items = [
        DecodeWorkItem(
            prompt_cache=[],
            input_token=index,
            sampler=sampler_for(index),
            detokenizer=FakeDetokenizer(),
            stop_token_ids=frozenset(),
            logits_processors=(processor,),
            logits_processor_tokens=[],
            completion_tokens=0,
            max_tokens=2,
        )
        for index in range(2)
    ]

    results = runner.decode_batch_step(items)

    assert all(isinstance(result, RuntimeError) for result in results)
    assert sampler_calls == [1, 1]
    diagnostics = runner.decode_diagnostics()
    assert diagnostics["batch_fallbacks"] == 0
    assert diagnostics["batch_post_sample_failures"] == 1


def test_group_evaluation_failure_does_not_replay_samplers() -> None:
    sampler_calls = [0, 0]

    class FakeTensor:
        shape = ()
        dtype = "float16"

        def __getitem__(self, _key: Any) -> FakeTensor:
            return self

        def __sub__(self, _other: Any) -> FakeTensor:
            return self

    class FakeSample:
        shape = ()
        dtype = "uint32"

        def __init__(self, token: int) -> None:
            self.token = token

        def item(self) -> int:
            raise AssertionError("failed group must not be retried individually")

    class FakeMX:
        uint32 = "uint32"

        @staticmethod
        def array(_value: Any, *, dtype: Any = None) -> FakeTensor:
            del dtype
            return FakeTensor()

        @staticmethod
        def logsumexp(value: FakeTensor, *, axis: int, keepdims: bool) -> FakeTensor:
            del axis, keepdims
            return value

        @staticmethod
        def async_eval(_value: Any) -> None:
            return None

        @staticmethod
        def eval(_value: Any) -> None:
            raise RuntimeError("group evaluation failed")

        @staticmethod
        def get_peak_memory() -> int:
            return 0

    class FakeLayer:
        state = object()

    class FakeDetokenizer:
        last_segment = ""

        def add_token(self, token: int) -> None:
            self.last_segment = chr(token)

    runner = ModelRunner(RuntimeSettings.model_validate({"embeddings": {"enabled": False}}))
    runner._loaded = True
    runner._mx = FakeMX()
    runner._model = lambda _tokens, *, cache: FakeTensor()  # type: ignore[assignment]
    runner._get_decode_batch_cache = lambda _items: ([FakeLayer()], None)  # type: ignore[method-assign]
    runner._extract_prompt_cache = lambda _cache, index: [index]  # type: ignore[method-assign]

    def sampler_for(index: int):
        def sampler(_logprobs: Any) -> FakeSample:
            sampler_calls[index] += 1
            return FakeSample(70 + index)

        return sampler

    items = [
        DecodeWorkItem(
            prompt_cache=[],
            input_token=index,
            sampler=sampler_for(index),
            detokenizer=FakeDetokenizer(),
            stop_token_ids=frozenset(),
            logits_processors=(),
            logits_processor_tokens=[],
            completion_tokens=0,
            max_tokens=2,
        )
        for index in range(2)
    ]

    results = runner.decode_batch_step(items)

    assert all(isinstance(result, RuntimeError) for result in results)
    assert sampler_calls == [1, 1]
    diagnostics = runner.decode_diagnostics()
    assert diagnostics["batch_fallbacks"] == 0
    assert diagnostics["batch_post_sample_failures"] == 1
    assert diagnostics["last_batch_fallback_error"] == (
        "RuntimeError: group evaluation failed"
    )


def test_sample_materialization_failure_does_not_replay_samplers() -> None:
    sampler_calls = [0, 0]

    class FakeTensor:
        shape = ()
        dtype = "float16"

        def __getitem__(self, _key: Any) -> FakeTensor:
            return self

        def __sub__(self, _other: Any) -> FakeTensor:
            return self

    class FakeSample:
        shape = ()
        dtype = "uint32"

        def item(self) -> int:
            raise RuntimeError("sample materialization failed")

    class FakeMX:
        uint32 = "uint32"

        @staticmethod
        def array(_value: Any, *, dtype: Any = None) -> FakeTensor:
            del dtype
            return FakeTensor()

        @staticmethod
        def logsumexp(value: FakeTensor, *, axis: int, keepdims: bool) -> FakeTensor:
            del axis, keepdims
            return value

        @staticmethod
        def async_eval(_value: Any) -> None:
            return None

        @staticmethod
        def eval(_value: Any) -> None:
            return None

        @staticmethod
        def get_peak_memory() -> int:
            return 0

    class FakeLayer:
        state = object()

    class FakeDetokenizer:
        last_segment = ""

        def add_token(self, token: int) -> None:
            self.last_segment = chr(token)

    runner = ModelRunner(RuntimeSettings.model_validate({"embeddings": {"enabled": False}}))
    runner._loaded = True
    runner._mx = FakeMX()
    runner._model = lambda _tokens, *, cache: FakeTensor()  # type: ignore[assignment]
    runner._get_decode_batch_cache = lambda _items: ([FakeLayer()], None)  # type: ignore[method-assign]
    runner._extract_prompt_cache = lambda _cache, index: [index]  # type: ignore[method-assign]

    def sampler_for(index: int):
        def sampler(_logprobs: Any) -> FakeSample:
            sampler_calls[index] += 1
            return FakeSample()

        return sampler

    items = [
        DecodeWorkItem(
            prompt_cache=[],
            input_token=index,
            sampler=sampler_for(index),
            detokenizer=FakeDetokenizer(),
            stop_token_ids=frozenset(),
            logits_processors=(),
            logits_processor_tokens=[],
            completion_tokens=0,
            max_tokens=2,
        )
        for index in range(2)
    ]

    results = runner.decode_batch_step(items)

    assert all(isinstance(result, RuntimeError) for result in results)
    assert sampler_calls == [1, 1]
    diagnostics = runner.decode_diagnostics()
    assert diagnostics["batch_fallbacks"] == 0
    assert diagnostics["batch_post_sample_failures"] == 1


def test_grouped_sampling_keeps_results_and_caches_aligned_after_reorder() -> None:
    class FakeTensor:
        shape = ()
        dtype = "float16"

        def __init__(self, row: int | None = None) -> None:
            self.row = row

        def __getitem__(self, key: Any) -> FakeTensor:
            if isinstance(key, tuple):
                return self
            if isinstance(key, slice):
                return FakeTensor(int(key.start or 0))
            return self

        def __sub__(self, _other: Any) -> FakeTensor:
            return self

    class FakeSample:
        shape = ()
        dtype = "uint32"

        def __init__(self, token: int) -> None:
            self.token = token

        def item(self) -> int:
            return self.token

    class FakeMX:
        uint32 = "uint32"

        @staticmethod
        def array(_value: Any, *, dtype: Any = None) -> FakeTensor:
            del dtype
            return FakeTensor()

        @staticmethod
        def logsumexp(value: FakeTensor, *, axis: int, keepdims: bool) -> FakeTensor:
            del axis, keepdims
            return value

        @staticmethod
        def async_eval(_value: Any) -> None:
            return None

        @staticmethod
        def eval(_value: Any) -> None:
            return None

        @staticmethod
        def get_peak_memory() -> int:
            return 0

    class FakeLayer:
        def __init__(self, labels: tuple[str, ...]) -> None:
            self.labels = labels

        @property
        def state(self) -> object:
            return self.labels

        def merge(self, caches: list[FakeLayer]) -> FakeLayer:
            return FakeLayer(tuple(cache.labels[0] for cache in caches))

        def extract(self, index: int) -> FakeLayer:
            return FakeLayer((self.labels[index],))

    class FakeDetokenizer:
        last_segment = ""

        def add_token(self, token: int) -> None:
            self.last_segment = chr(token)

    def item(
        request_id: str,
        cache: Any,
        token: int,
    ) -> DecodeWorkItem:
        return DecodeWorkItem(
            prompt_cache=cache,
            input_token=token,
            sampler=lambda _logprobs, sampled=token: FakeSample(sampled),
            detokenizer=FakeDetokenizer(),
            stop_token_ids=frozenset(),
            logits_processors=(),
            logits_processor_tokens=[],
            completion_tokens=0,
            max_tokens=4,
            request_id=request_id,
        )

    runner = ModelRunner(RuntimeSettings.model_validate({"embeddings": {"enabled": False}}))
    runner._loaded = True
    runner._mx = FakeMX()
    runner._model = lambda _tokens, *, cache: FakeTensor()  # type: ignore[assignment]

    first = runner.decode_batch_step(
        [
            item("request-a", [FakeLayer(("cache-a",))], 65),
            item("request-b", [FakeLayer(("cache-b",))], 66),
        ]
    )
    assert all(isinstance(result, DecodeResult) for result in first)
    first_results = [result for result in first if isinstance(result, DecodeResult)]

    reordered = runner.decode_batch_step(
        [
            item("request-b", first_results[1].prompt_cache, 80),
            item("request-a", first_results[0].prompt_cache, 81),
        ]
    )

    assert [
        result.token_id for result in reordered if isinstance(result, DecodeResult)
    ] == [80, 81]
    reordered_results = [
        result for result in reordered if isinstance(result, DecodeResult)
    ]
    resolved = [
        runner._resolve_decode_cache(result.prompt_cache) for result in reordered_results
    ]
    assert [cache[0].labels for cache in resolved] == [
        ("cache-b",),
        ("cache-a",),
    ]
