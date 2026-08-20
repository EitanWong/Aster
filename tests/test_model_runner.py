from __future__ import annotations

from typing import Any

import pytest
from mlx_lm.models.cache import ArraysCache, KVCache

from aster.core.config import RuntimeSettings
from aster.core.errors import ConfigurationError
from aster.inference.constrained import ThinkingAwareJsonLogitsProcessor
from aster.inference.contracts import InferenceRequest
from aster.inference.model_runner import (
    DecodeResult,
    DecodeWorkItem,
    ModelRunner,
)
from aster.inference.paged_cache import PagedCacheManager
from aster.inference.thinking_processor import ThinkingAwareLogitsProcessor


class FakeChatTokenizer:
    bos_token = None
    detokenizer = object()

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool,
    ) -> str:
        del tokenize, enable_thinking
        rendered = "".join(
            f"<{message['role']}>{message['content']}</{message['role']}>" for message in messages
        )
        if add_generation_prompt:
            rendered += "<assistant>"
        return rendered

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [ord(char) for char in text]


class CapturingChatTemplateTokenizer:
    bos_token = None
    detokenizer = object()

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def apply_chat_template(
        self,
        messages,
        **kwargs: object,
    ) -> str:
        self.calls.append(dict(kwargs))
        rendered = "".join(
            f"<{message['role']}>{message['content']}</{message['role']}>" for message in messages
        )
        if kwargs.get("add_generation_prompt") is True:
            rendered += "<assistant>"
        if kwargs.get("variant"):
            rendered += f"<variant>{kwargs['variant']}</variant>"
        return rendered

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [ord(char) for char in text]


class ToolStopTokenizer(FakeChatTokenizer):
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        if text == "<|tool_response>":
            return [777]
        if text == "<multi>":
            return [1, 2]
        return [ord(char) for char in text]


class FakeArray:
    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape

    def __getitem__(self, key: Any) -> FakeArray:
        token_count = self.shape[-2]
        if isinstance(key, tuple):
            for item in reversed(key):
                if isinstance(item, slice) and item.stop is not None:
                    token_count = max(int(item.stop), 0)
                    break
        shape = list(self.shape)
        shape[-2] = token_count
        return FakeArray(tuple(shape))


class RewindableCacheLayer:
    def __init__(self, *, offset: int, capacity: int) -> None:
        self.offset = offset
        self.keys = FakeArray((1, 2, capacity, 4))
        self.values = FakeArray((1, 2, capacity, 4))


def test_chat_reuse_points_include_lcp_boundary_for_last_user() -> None:
    runner = ModelRunner(RuntimeSettings.model_validate({"embeddings": {"enabled": False}}))
    runner._loaded = True
    runner._tokenizer = FakeChatTokenizer()

    messages = [
        {"role": "system", "content": "same rules"},
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "first answer"},
        {"role": "user", "content": "second question"},
    ]
    full_tokens = runner._encode_chat(messages, enable_thinking=False)
    old_boundary = len(
        runner._chat_tokens(
            messages[:3],
            add_generation_prompt=False,
            enable_thinking=False,
        )
    )

    reuse_points = runner._chat_reuse_points(
        messages,
        full_prompt_tokens=full_tokens,
        enable_thinking=False,
    )

    assert reuse_points
    assert max(reuse_points) > old_boundary


def test_chat_reuse_points_limit_keeps_most_recent_boundaries() -> None:
    runner = ModelRunner(
        RuntimeSettings.model_validate(
            {
                "embeddings": {"enabled": False},
                "engine": {
                    "snapshot_max_chat_reuse_points": 2,
                    "snapshot_chat_reuse_sparse_points": 0,
                },
            }
        )
    )
    runner._loaded = True
    runner._tokenizer = FakeChatTokenizer()
    messages = [
        {"role": "system", "content": "same rules"},
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "first answer"},
        {"role": "user", "content": "second question"},
        {"role": "assistant", "content": "second answer"},
        {"role": "user", "content": "third question"},
    ]
    full_tokens = runner._encode_chat(messages, enable_thinking=False)

    all_points_runner = ModelRunner(
        RuntimeSettings.model_validate(
            {
                "embeddings": {"enabled": False},
                "engine": {"snapshot_max_chat_reuse_points": 0},
            }
        )
    )
    all_points_runner._loaded = True
    all_points_runner._tokenizer = FakeChatTokenizer()
    all_points = all_points_runner._chat_reuse_points(
        messages,
        full_prompt_tokens=full_tokens,
        enable_thinking=False,
    )

    assert len(all_points) > 2
    assert (
        runner._chat_reuse_points(
            messages,
            full_prompt_tokens=full_tokens,
            enable_thinking=False,
        )
        == all_points[-2:]
    )


def test_chat_reuse_points_keep_sparse_older_boundaries() -> None:
    runner = ModelRunner(
        RuntimeSettings.model_validate(
            {
                "embeddings": {"enabled": False},
                "engine": {
                    "snapshot_max_chat_reuse_points": 2,
                    "snapshot_chat_reuse_sparse_points": 1,
                    "snapshot_chat_reuse_sparse_min_tokens": 0,
                },
            }
        )
    )
    runner._loaded = True
    runner._tokenizer = FakeChatTokenizer()
    messages = [
        {"role": "system", "content": "same rules"},
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "first answer"},
        {"role": "user", "content": "second question"},
        {"role": "assistant", "content": "second answer"},
        {"role": "user", "content": "third question"},
        {"role": "assistant", "content": "third answer"},
        {"role": "user", "content": "fourth question"},
    ]
    full_tokens = runner._encode_chat(messages, enable_thinking=False)

    all_points_runner = ModelRunner(
        RuntimeSettings.model_validate(
            {
                "embeddings": {"enabled": False},
                "engine": {"snapshot_max_chat_reuse_points": 0},
            }
        )
    )
    all_points_runner._loaded = True
    all_points_runner._tokenizer = FakeChatTokenizer()
    all_points = all_points_runner._chat_reuse_points(
        messages,
        full_prompt_tokens=full_tokens,
        enable_thinking=False,
    )
    eligible = [point for point in all_points if point >= 32]
    selected = runner._chat_reuse_points(
        messages,
        full_prompt_tokens=full_tokens,
        enable_thinking=False,
    )

    assert len(all_points) > 2
    assert selected[-2:] == all_points[-2:]
    assert selected[0] == eligible[0]
    assert len(selected) == 3


def test_chat_reuse_points_skip_sparse_retention_below_long_context_threshold() -> None:
    runner = ModelRunner(
        RuntimeSettings.model_validate(
            {
                "embeddings": {"enabled": False},
                "engine": {
                    "snapshot_max_chat_reuse_points": 2,
                    "snapshot_chat_reuse_sparse_points": 1,
                    "snapshot_chat_reuse_sparse_min_tokens": 2048,
                },
            }
        )
    )
    runner._loaded = True
    runner._tokenizer = FakeChatTokenizer()
    messages = [
        {"role": "system", "content": "same rules"},
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "first answer"},
        {"role": "user", "content": "second question"},
        {"role": "assistant", "content": "second answer"},
        {"role": "user", "content": "third question"},
    ]
    full_tokens = runner._encode_chat(messages, enable_thinking=False)

    all_points_runner = ModelRunner(
        RuntimeSettings.model_validate(
            {
                "embeddings": {"enabled": False},
                "engine": {"snapshot_max_chat_reuse_points": 0},
            }
        )
    )
    all_points_runner._loaded = True
    all_points_runner._tokenizer = FakeChatTokenizer()
    all_points = all_points_runner._chat_reuse_points(
        messages,
        full_prompt_tokens=full_tokens,
        enable_thinking=False,
    )

    assert len(full_tokens) < 2048
    assert (
        runner._chat_reuse_points(
            messages,
            full_prompt_tokens=full_tokens,
            enable_thinking=False,
        )
        == all_points[-2:]
    )


def test_encode_request_skips_reuse_analysis_when_prefix_cache_is_disabled() -> None:
    runner = ModelRunner(
        RuntimeSettings.model_validate(
            {"embeddings": {"enabled": False}, "engine": {"prefix_cache_enabled": False}}
        )
    )
    runner._loaded = True
    runner._tokenizer = FakeChatTokenizer()
    runner._chat_reuse_points = lambda *args, **kwargs: (_ for _ in ()).throw(  # type: ignore[method-assign]
        AssertionError("reuse analysis must be skipped when prefix cache is disabled")
    )

    prepared = runner.encode_request(
        InferenceRequest(messages=[{"role": "user", "content": "hello"}])
    )

    assert prepared.prompt_tokens
    assert prepared.reuse_points == ()


def test_encode_request_keeps_reuse_analysis_when_prefix_cache_is_enabled() -> None:
    runner = ModelRunner(
        RuntimeSettings.model_validate(
            {"embeddings": {"enabled": False}, "engine": {"prefix_cache_enabled": True}}
        )
    )
    runner._loaded = True
    runner._tokenizer = FakeChatTokenizer()
    runner._chat_reuse_points = lambda *args, **kwargs: (7,)  # type: ignore[method-assign]

    prepared = runner.encode_request(
        InferenceRequest(messages=[{"role": "user", "content": "hello"}])
    )

    assert prepared.reuse_points == (7,)


def test_encode_request_reuses_bounded_chat_prompt_cache() -> None:
    runner = ModelRunner(
        RuntimeSettings.model_validate(
            {
                "embeddings": {"enabled": False},
                "engine": {"chat_prompt_cache_max_entries": 1},
            }
        )
    )
    runner._loaded = True
    runner._tokenizer = FakeChatTokenizer()
    calls: list[list[dict[str, str]]] = []
    runner._encode_chat = lambda messages, **kwargs: calls.append(messages) or [1, 2, 3]  # type: ignore[method-assign]

    first = runner.encode_request(InferenceRequest(messages=[{"role": "user", "content": "first"}]))
    second = runner.encode_request(
        InferenceRequest(messages=[{"role": "user", "content": "first"}])
    )

    assert first.prompt_tokens == [1, 2, 3]
    assert second.prompt_tokens == [1, 2, 3]
    assert second.prompt_tokens is not first.prompt_tokens
    assert len(calls) == 1


def test_encode_request_chat_prompt_cache_eviction_is_bounded() -> None:
    runner = ModelRunner(
        RuntimeSettings.model_validate(
            {
                "embeddings": {"enabled": False},
                "engine": {"chat_prompt_cache_max_entries": 1},
            }
        )
    )
    runner._loaded = True
    runner._tokenizer = FakeChatTokenizer()
    calls = 0

    def encode_chat(_messages, **_kwargs):
        nonlocal calls
        calls += 1
        return [calls]

    runner._encode_chat = encode_chat  # type: ignore[method-assign]
    first = InferenceRequest(messages=[{"role": "user", "content": "first"}])
    second = InferenceRequest(messages=[{"role": "user", "content": "second"}])

    runner.encode_request(first)
    runner.encode_request(second)
    runner.encode_request(first)

    assert calls == 3


def test_clone_cache_trims_oversized_arrays_at_matching_offset() -> None:
    runner = ModelRunner(RuntimeSettings.model_validate({"embeddings": {"enabled": False}}))
    cache = [RewindableCacheLayer(offset=3, capacity=8)]

    cloned = runner.clone_cache(cache, cache_token_count=3)

    assert cloned is not cache
    assert cloned[0] is not cache[0]
    assert cloned[0].offset == 3
    assert cloned[0].keys.shape == (1, 2, 3, 4)
    assert cloned[0].values.shape == (1, 2, 3, 4)
    assert cache[0].keys.shape == (1, 2, 8, 4)


def test_estimate_request_bytes_accounts_for_nested_hybrid_attention_config() -> None:
    runner = ModelRunner(RuntimeSettings.model_validate({"embeddings": {"enabled": False}}))
    runner._loaded = True
    runner._config = {
        "text_config": {
            "dtype": "bfloat16",
            "hidden_size": 4096,
            "head_dim": 256,
            "num_key_value_heads": 4,
            "num_hidden_layers": 32,
            "layer_types": [
                "linear_attention",
                "full_attention",
                "linear_attention",
                "full_attention",
            ],
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 128,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 32,
            "linear_value_head_dim": 128,
        }
    }

    # The test config has two full-attention and two linear-attention layers.
    full_attention_bytes = 2 * 4 * 256 * 2 * 2 * (100 + 20)
    linear_state_bytes = 2 * 32 * 128 * 128 * 4
    linear_conv_bytes = 2 * 3 * (16 * 128 * 2 + 32 * 128) * 2

    assert runner.estimate_request_bytes(100, 20) == (
        full_attention_bytes + linear_state_bytes + linear_conv_bytes
    )


def test_estimate_prefill_transient_bytes_prices_one_full_attention_call() -> None:
    runner = ModelRunner(RuntimeSettings.model_validate({"embeddings": {"enabled": False}}))
    runner._loaded = True
    runner._config = {
        "text_config": {
            "dtype": "bfloat16",
            "head_dim": 256,
            "num_attention_heads": 8,
            "layer_types": ["linear_attention", "full_attention"],
        }
    }

    query_tokens = 128
    kv_tokens = 1024
    expected = (8 * query_tokens * kv_tokens * 2) + (8 * query_tokens * 256 * 4)

    assert runner.estimate_prefill_transient_bytes(query_tokens, kv_tokens) == expected


def test_configure_prompt_cache_step_only_updates_matching_cache_type() -> None:
    class FakeKVCache:
        def __init__(self) -> None:
            self.step = 256

    runner = ModelRunner(RuntimeSettings.model_validate({"embeddings": {"enabled": False}}))
    kv_cache = FakeKVCache()
    other_cache = object()

    configured = runner._configure_prompt_cache_step([kv_cache, other_cache], FakeKVCache, 2048)

    assert configured == [kv_cache, other_cache]
    assert kv_cache.step == 2048


def test_opt_in_paged_prompt_cache_preserves_hybrid_list_shape_and_owner() -> None:
    settings = RuntimeSettings.model_validate(
        {
            "embeddings": {"enabled": False},
            "engine": {
                "paged_cache_enabled": True,
                "prefix_cache_enabled": False,
                "max_decode_batch": 1,
            },
        }
    )
    runner = ModelRunner(settings)
    runner._prompt_cache_factory = lambda _model: [ArraysCache(1), KVCache()]
    runner._kv_cache_type = KVCache
    runner._paged_cache = PagedCacheManager(num_layers=2, block_size=4, max_blocks=8)

    configured = runner._make_configured_prompt_cache(object())

    assert isinstance(configured, list)
    assert isinstance(configured[0], ArraysCache)
    assert configured[1].__class__.__name__ == "PagedKVCacheLayer"
    assert callable(getattr(configured, "release", None))
    configured.release()


def test_direct_paged_attention_requires_the_paged_cache_boundary() -> None:
    settings = RuntimeSettings.model_validate(
        {
            "embeddings": {"enabled": False},
            "engine": {"paged_cache_direct_attention_enabled": True},
        }
    )
    runner = ModelRunner(settings)
    runner._prompt_cache_factory = lambda _model: [KVCache()]
    runner._kv_cache_type = KVCache

    with pytest.raises(ConfigurationError) as exc_info:
        runner._make_configured_prompt_cache(object())
    assert exc_info.value.code == "paged_direct_attention_requires_paged_cache"


def test_direct_paged_prompt_cache_uses_pool_only_after_prepare() -> None:
    settings = RuntimeSettings.model_validate(
        {
            "embeddings": {"enabled": False},
            "engine": {
                "paged_cache_enabled": True,
                "paged_cache_direct_attention_enabled": True,
                "prefix_cache_enabled": False,
                "max_decode_batch": 1,
            },
        }
    )
    runner = ModelRunner(settings)
    runner._prompt_cache_factory = lambda _model: [KVCache()]
    runner._kv_cache_type = KVCache
    runner._paged_cache = PagedCacheManager(num_layers=1, block_size=4, max_blocks=8)

    configured = runner._make_configured_prompt_cache(object())
    layer = configured[0]

    assert layer.direct_attention_enabled is True
    assert layer._pool_enabled is False
    configured.release()


def test_chat_template_kwargs_reach_tokenizer_and_keep_core_flags_authoritative() -> None:
    runner = ModelRunner(RuntimeSettings.model_validate({"embeddings": {"enabled": False}}))
    runner._loaded = True
    tokenizer = CapturingChatTemplateTokenizer()
    runner._tokenizer = tokenizer

    runner.encode_request(
        InferenceRequest(
            messages=[{"role": "user", "content": "hello"}],
            enable_thinking=False,
            chat_template_kwargs={"enable_thinking": True, "variant": "json"},
        )
    )

    assert tokenizer.calls[0] == {
        "enable_thinking": False,
        "variant": "json",
        "tokenize": False,
        "add_generation_prompt": True,
    }


def test_initialize_decode_adds_structured_logits_processor(monkeypatch) -> None:
    runner = ModelRunner(RuntimeSettings.model_validate({"embeddings": {"enabled": False}}))
    runner._loaded = True
    runner._tokenizer = FakeChatTokenizer()
    runner._model = object()
    runner._make_prompt_cache = lambda _model: {"cache": True}
    runner._make_sampler = lambda **_kwargs: "sampler"
    penalty_processor = object()
    runner._make_logits_processors = lambda **_kwargs: [penalty_processor]

    structured_processor = object()
    captured: dict[str, object] = {}

    def fake_build_json_logits_processor(schema, tokenizer):
        captured["schema"] = schema
        captured["tokenizer"] = tokenizer
        return structured_processor

    monkeypatch.setattr(
        "aster.inference.model_runner.build_json_logits_processor",
        fake_build_json_logits_processor,
    )

    decode_init = runner.initialize_decode(
        prompt_tokens=[1, 2, 3],
        cache_token_count=0,
        prompt_cache=None,
        request=InferenceRequest(
            prompt="ignored",
            structured_output_schema={
                "type": "object",
                "properties": {"answer": {"type": "string"}},
            },
        ),
    )

    assert captured == {
        "schema": {"type": "object", "properties": {"answer": {"type": "string"}}},
        "tokenizer": runner._tokenizer,
    }
    assert decode_init.logits_processors == (structured_processor, penalty_processor)


def test_initialize_decode_wraps_structured_processor_when_thinking_enabled(monkeypatch) -> None:
    runner = ModelRunner(RuntimeSettings.model_validate({"embeddings": {"enabled": False}}))
    runner._loaded = True
    runner._tokenizer = FakeChatTokenizer()
    runner._model = object()
    runner._make_prompt_cache = lambda _model: {"cache": True}
    runner._make_sampler = lambda **_kwargs: "sampler"
    runner._make_logits_processors = lambda **_kwargs: []

    def structured_processor(tokens, logits):
        del tokens
        return logits

    monkeypatch.setattr(
        "aster.inference.model_runner.build_json_logits_processor",
        lambda _schema, _tokenizer: structured_processor,
    )

    decode_init = runner.initialize_decode(
        prompt_tokens=[1, 2, 3],
        cache_token_count=0,
        prompt_cache=None,
        request=InferenceRequest(
            prompt="ignored",
            enable_thinking=True,
            structured_output_schema={"type": "object"},
        ),
    )

    assert len(decode_init.logits_processors) == 1
    assert isinstance(decode_init.logits_processors[0], ThinkingAwareJsonLogitsProcessor)


def test_initialize_decode_adds_thinking_budget_processor() -> None:
    runner = ModelRunner(RuntimeSettings.model_validate({"embeddings": {"enabled": False}}))
    runner._loaded = True
    runner._tokenizer = FakeChatTokenizer()
    runner._model = object()
    runner._make_prompt_cache = lambda _model: {"cache": True}
    runner._make_sampler = lambda **_kwargs: "sampler"
    runner._make_logits_processors = lambda **_kwargs: []

    decode_init = runner.initialize_decode(
        prompt_tokens=[1, 2, 3],
        cache_token_count=0,
        prompt_cache=None,
        request=InferenceRequest(
            prompt="ignored",
            enable_thinking=True,
            thinking_token_budget=4,
        ),
    )

    assert len(decode_init.logits_processors) == 1
    assert isinstance(decode_init.logits_processors[0], ThinkingAwareLogitsProcessor)


def test_initialize_decode_thinking_budget_processor_wraps_structured_processor(
    monkeypatch,
) -> None:
    runner = ModelRunner(RuntimeSettings.model_validate({"embeddings": {"enabled": False}}))
    runner._loaded = True
    runner._tokenizer = FakeChatTokenizer()
    runner._model = object()
    runner._make_prompt_cache = lambda _model: {"cache": True}
    runner._make_sampler = lambda **_kwargs: "sampler"
    runner._make_logits_processors = lambda **_kwargs: []

    def structured_processor(tokens, logits):
        del tokens
        return logits

    monkeypatch.setattr(
        "aster.inference.model_runner.build_json_logits_processor",
        lambda _schema, _tokenizer: structured_processor,
    )

    decode_init = runner.initialize_decode(
        prompt_tokens=[1, 2, 3],
        cache_token_count=0,
        prompt_cache=None,
        request=InferenceRequest(
            prompt="ignored",
            enable_thinking=True,
            thinking_token_budget=4,
            structured_output_schema={"type": "object"},
        ),
    )

    assert len(decode_init.logits_processors) == 1
    processor = decode_init.logits_processors[0]
    assert isinstance(processor, ThinkingAwareLogitsProcessor)
    assert processor._inner is structured_processor


def test_initialize_decode_merges_request_stop_token_ids() -> None:
    runner = ModelRunner(RuntimeSettings.model_validate({"embeddings": {"enabled": False}}))
    runner._loaded = True
    runner._tokenizer = FakeChatTokenizer()
    runner._model = object()
    runner._make_prompt_cache = lambda _model: {"cache": True}
    runner._make_sampler = lambda **_kwargs: "sampler"
    runner._make_logits_processors = lambda **_kwargs: []

    decode_init = runner.initialize_decode(
        prompt_tokens=[1, 2, 3],
        cache_token_count=0,
        prompt_cache=None,
        request=InferenceRequest(prompt="ignored", stop_token_ids=(123, 456)),
    )

    assert decode_init.stop_token_ids == frozenset({123, 456})


def test_initialize_decode_merges_single_token_parser_stop_sequences() -> None:
    runner = ModelRunner(RuntimeSettings.model_validate({"embeddings": {"enabled": False}}))
    runner._loaded = True
    runner._tokenizer = ToolStopTokenizer()
    runner._model = object()
    runner._make_prompt_cache = lambda _model: {"cache": True}
    runner._make_sampler = lambda **_kwargs: "sampler"
    runner._make_logits_processors = lambda **_kwargs: []

    decode_init = runner.initialize_decode(
        prompt_tokens=[1, 2, 3],
        cache_token_count=0,
        prompt_cache=None,
        request=InferenceRequest(
            prompt="ignored",
            stop_token_ids=(123,),
            parser_stop_sequences=("<|tool_response>", "<multi>"),
        ),
    )

    assert decode_init.stop_token_ids == frozenset({123, 777})


def test_apply_logits_processors_prepares_aster_decode_step() -> None:
    class FakeMX:
        uint32 = "uint32"

        @staticmethod
        def array(values, *, dtype):
            del dtype
            return list(values)

    class IncrementalProcessor:
        def __init__(self) -> None:
            self.hints: list[tuple[int, int]] = []
            self.calls: list[tuple[list[int], object]] = []

        def _prepare_aster_decode_step(self, *, input_token: int, completion_tokens: int) -> None:
            self.hints.append((input_token, completion_tokens))

        def __call__(self, tokens, logits):
            self.calls.append((tokens, logits))
            return logits

    runner = ModelRunner(RuntimeSettings.model_validate({"embeddings": {"enabled": False}}))
    runner._mx = FakeMX()
    processor = IncrementalProcessor()
    item = DecodeWorkItem(
        prompt_cache=None,
        input_token=12,
        sampler=lambda logits: logits,
        detokenizer=object(),
        stop_token_ids=frozenset(),
        logits_processors=(processor,),
        logits_processor_tokens=[10, 11],
        completion_tokens=2,
        max_tokens=4,
    )

    assert runner._apply_logits_processors("logits", item=item) == "logits"
    assert processor.hints == [(12, 2)]
    assert processor.calls == [([10, 11, 12], "logits")]


def test_decode_batch_fallback_preserves_per_item_failures() -> None:
    runner = ModelRunner(RuntimeSettings.model_validate({"embeddings": {"enabled": False}}))

    def fail_batch(_items: list[DecodeWorkItem]) -> list[DecodeResult]:
        raise RuntimeError("batched decode unsupported")

    def decode_single(item: DecodeWorkItem) -> DecodeResult:
        if item.max_tokens == 13:
            raise ValueError("bad logits processor")
        return DecodeResult(
            prompt_cache=item.prompt_cache,
            token_id=65,
            text="A",
            completion_tokens=item.completion_tokens + 1,
            peak_memory_gb=0.0,
            finish_reason="length",
        )

    runner._decode_batch = fail_batch  # type: ignore[method-assign]
    runner._decode_single = decode_single  # type: ignore[method-assign]

    ok_item = DecodeWorkItem(
        prompt_cache={"ok": True},
        input_token=1,
        sampler=lambda logits: logits,
        detokenizer=object(),
        stop_token_ids=frozenset(),
        logits_processors=(),
        logits_processor_tokens=[],
        completion_tokens=0,
        max_tokens=1,
    )
    failed_item = DecodeWorkItem(
        prompt_cache={"fail": True},
        input_token=2,
        sampler=lambda logits: logits,
        detokenizer=object(),
        stop_token_ids=frozenset(),
        logits_processors=(),
        logits_processor_tokens=[],
        completion_tokens=0,
        max_tokens=13,
    )

    results = runner.decode_batch_step([ok_item, failed_item])

    assert isinstance(results[0], DecodeResult)
    assert results[0].text == "A"
    assert isinstance(results[1], ValueError)
    assert str(results[1]) == "bad logits processor"
    diagnostics = runner.decode_diagnostics()
    assert diagnostics["batch_attempts"] == 1
    assert diagnostics["batch_successes"] == 0
    assert diagnostics["batch_fallbacks"] == 1
    assert diagnostics["batch_items"] == 2
    assert diagnostics["batch_fallback_items"] == 2
    assert diagnostics["batch_fallback_rate"] == 1.0
    assert diagnostics["last_batch_fallback_error"] == "RuntimeError: batched decode unsupported"


def test_decode_batch_cache_reuses_stable_membership_and_remerges_after_change() -> None:
    runner = ModelRunner(RuntimeSettings.model_validate({"embeddings": {"enabled": False}}))
    merge_inputs: list[tuple[object, ...]] = []
    extracted: list[tuple[object, int]] = []

    def merge(caches: list[object]) -> list[object]:
        merged = [f"merged-{len(merge_inputs) + 1}"]
        merge_inputs.append(tuple(caches))
        return merged

    def extract(merged: list[object], index: int) -> list[object]:
        extracted.append((merged[0], index))
        return [f"extracted-{merged[0]}-{index}"]

    runner._merge_prompt_caches = merge  # type: ignore[method-assign]
    runner._extract_prompt_cache = extract  # type: ignore[method-assign]

    first_items = [
        DecodeWorkItem(
            prompt_cache="cache-a",
            input_token=1,
            sampler=lambda logits: logits,
            detokenizer=object(),
            stop_token_ids=frozenset(),
            logits_processors=(),
            logits_processor_tokens=[],
            completion_tokens=0,
            max_tokens=4,
            request_id="request-a",
        ),
        DecodeWorkItem(
            prompt_cache="cache-b",
            input_token=2,
            sampler=lambda logits: logits,
            detokenizer=object(),
            stop_token_ids=frozenset(),
            logits_processors=(),
            logits_processor_tokens=[],
            completion_tokens=0,
            max_tokens=4,
            request_id="request-b",
        ),
    ]

    first_merged, first_state = runner._get_decode_batch_cache(first_items)
    assert first_state is not None
    assert merge_inputs == [("cache-a", "cache-b")]

    stable_items = [
        DecodeWorkItem(
            prompt_cache=runner._decode_cache_ref(first_state, 0),  # type: ignore[attr-defined]
            input_token=1,
            sampler=lambda logits: logits,
            detokenizer=object(),
            stop_token_ids=frozenset(),
            logits_processors=(),
            logits_processor_tokens=[],
            completion_tokens=0,
            max_tokens=4,
            request_id="request-a",
        ),
        DecodeWorkItem(
            prompt_cache=runner._decode_cache_ref(first_state, 1),  # type: ignore[attr-defined]
            input_token=2,
            sampler=lambda logits: logits,
            detokenizer=object(),
            stop_token_ids=frozenset(),
            logits_processors=(),
            logits_processor_tokens=[],
            completion_tokens=0,
            max_tokens=4,
            request_id="request-b",
        ),
    ]

    stable_merged, stable_state = runner._get_decode_batch_cache(stable_items)
    assert stable_merged is first_merged
    assert stable_state is first_state
    assert merge_inputs == [("cache-a", "cache-b")]
    assert extracted == []

    changed_items = [
        stable_items[0],
        DecodeWorkItem(
            prompt_cache="cache-c",
            input_token=3,
            sampler=lambda logits: logits,
            detokenizer=object(),
            stop_token_ids=frozenset(),
            logits_processors=(),
            logits_processor_tokens=[],
            completion_tokens=0,
            max_tokens=4,
            request_id="request-c",
        ),
    ]

    changed_merged, changed_state = runner._get_decode_batch_cache(changed_items)
    assert changed_state is not None
    assert changed_state is not first_state
    assert changed_merged is not first_merged
    assert merge_inputs == [("cache-a", "cache-b"), (["extracted-merged-1-0"], "cache-c")]
    assert extracted == [("merged-1", 0)]

    reordered_items = [
        DecodeWorkItem(
            prompt_cache=runner._decode_cache_ref(changed_state, 1),
            input_token=3,
            sampler=lambda logits: logits,
            detokenizer=object(),
            stop_token_ids=frozenset(),
            logits_processors=(),
            logits_processor_tokens=[],
            completion_tokens=1,
            max_tokens=4,
            request_id="request-c",
        ),
        DecodeWorkItem(
            prompt_cache=runner._decode_cache_ref(changed_state, 0),
            input_token=1,
            sampler=lambda logits: logits,
            detokenizer=object(),
            stop_token_ids=frozenset(),
            logits_processors=(),
            logits_processor_tokens=[],
            completion_tokens=1,
            max_tokens=4,
            request_id="request-a",
        ),
    ]

    reordered_merged, reordered_state = runner._get_decode_batch_cache(reordered_items)
    assert reordered_state is not None
    assert reordered_state.request_ids == ("request-c", "request-a")
    assert reordered_merged == ["merged-3"]
    assert merge_inputs[-1] == (
        ["extracted-merged-2-1"],
        ["extracted-merged-2-0"],
    )
    assert extracted[-2:] == [("merged-2", 1), ("merged-2", 0)]


def test_decode_result_counts_stop_token_without_emitting_text() -> None:
    runner = ModelRunner(RuntimeSettings.model_validate({"embeddings": {"enabled": False}}))
    item = DecodeWorkItem(
        prompt_cache={"cache": True},
        input_token=1,
        sampler=lambda logits: logits,
        detokenizer=object(),
        stop_token_ids=frozenset({99}),
        logits_processors=(),
        logits_processor_tokens=[],
        completion_tokens=1,
        max_tokens=4,
    )

    result = runner._decode_result(
        item=item,
        token=99,
        prompt_cache={"cache": True},
        peak_memory_gb=0.5,
    )

    assert result.token_id == 99
    assert result.text == ""
    assert result.completion_tokens == 2
    assert result.finish_reason == "stop"


def test_decode_single_relies_on_sample_sync_without_forcing_cache_eval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeTensor:
        def __getitem__(self, _key):
            return self

        def __sub__(self, _other):
            return self

    class FakeScalar:
        shape = ()
        dtype = "uint32"

        def item(self) -> int:
            return 65

    class FakeMX:
        uint32 = "uint32"

        @staticmethod
        def array(_value, *, dtype=None):
            del dtype
            return FakeTensor()

        @staticmethod
        def logsumexp(value, *, axis, keepdims):
            del axis, keepdims
            return value

    class FakeDetokenizer:
        last_segment = ""

        def add_token(self, token: int) -> None:
            self.last_segment = chr(token)

    runner = ModelRunner(
        RuntimeSettings.model_validate(
            {
                "embeddings": {"enabled": False},
                "engine": {"decode_stage_observer_max_events": 1},
            }
        )
    )
    runner._loaded = True
    runner._mx = FakeMX()
    runner._model = lambda _tokens, *, cache: FakeTensor()
    runner._eval_cache = lambda _cache: (_ for _ in ()).throw(  # type: ignore[method-assign]
        AssertionError("decode must not force cache state after sample synchronization")
    )
    timestamps = iter((0.0, 1.0, 3.0, 6.0, 10.0, 15.0))
    monkeypatch.setattr("aster.inference.model_runner.time.perf_counter", lambda: next(timestamps))
    item = DecodeWorkItem(
        prompt_cache=[object()],
        input_token=1,
        sampler=lambda _logprobs: FakeScalar(),
        detokenizer=FakeDetokenizer(),
        stop_token_ids=frozenset(),
        logits_processors=(),
        logits_processor_tokens=[],
        completion_tokens=0,
        max_tokens=2,
        context_tokens=17,
    )

    result = runner._decode_single(item)

    assert result.token_id == 65
    assert result.text == "A"
    observer = runner.decode_diagnostics()["decode_stage_observer"]
    assert observer["batch_steps"] == 0
    assert observer["single_steps"] == 1
    assert observer["events"][0] == {
        "mode": "single",
        "path": "implicit_sample_sync",
        "batch_size": 1,
        "batch_size_squared": 1,
        "context_token_sum": 17,
        "context_token_min": 17,
        "context_token_max": 17,
        "completion_token_sum": 0,
        "processor_rows": 0,
        "cache_mode": "single",
        "seconds": {
            "cache_prepare": 1.0,
            "model_enqueue": 2.0,
            "sampling_enqueue": 3.0,
            "evaluation_window": 4.0,
            "result_delivery": 5.0,
            "eager_completion": 0.0,
            "observed_total": 15.0,
        },
    }


def test_decode_batch_groups_sample_evaluation_without_forcing_cache_state() -> None:
    class FakeTensor:
        def __getitem__(self, _key):
            return self

        def __sub__(self, _other):
            return self

    class FakeScalar:
        shape = ()
        dtype = "uint32"

        def item(self) -> int:
            return 65

    class FakeMX:
        uint32 = "uint32"

        def __init__(self) -> None:
            self.eval_calls: list[object] = []
            self.async_eval_calls: list[object] = []

        @staticmethod
        def array(_value, *, dtype=None):
            del dtype
            return FakeTensor()

        @staticmethod
        def logsumexp(value, *, axis, keepdims):
            del axis, keepdims
            return value

        def eval(self, value) -> None:
            self.eval_calls.append(value)

        def async_eval(self, value) -> None:
            self.async_eval_calls.append(value)

        @staticmethod
        def clear_cache() -> None:
            return None

    class FakeLayer:
        state = object()

    class FakeDetokenizer:
        last_segment = ""

        def add_token(self, token: int) -> None:
            self.last_segment = chr(token)

    runner = ModelRunner(RuntimeSettings.model_validate({"embeddings": {"enabled": False}}))
    runner._loaded = True
    fake_mx = FakeMX()
    runner._mx = fake_mx
    runner._model = lambda _tokens, *, cache: FakeTensor()
    runner._get_decode_batch_cache = lambda _items: ([FakeLayer()], None)  # type: ignore[method-assign]
    runner._extract_prompt_cache = lambda _cache, _index: []  # type: ignore[method-assign]
    items = [
        DecodeWorkItem(
            prompt_cache=[],
            input_token=index,
            sampler=lambda _logprobs: FakeScalar(),
            detokenizer=FakeDetokenizer(),
            stop_token_ids=frozenset(),
            logits_processors=(),
            logits_processor_tokens=[],
            completion_tokens=0,
            max_tokens=2,
        )
        for index in (1, 2)
    ]

    results = runner._decode_batch(items)

    assert len(results) == 2
    assert len(fake_mx.async_eval_calls) == 1
    assert isinstance(fake_mx.async_eval_calls[0], list)
    assert len(fake_mx.eval_calls) == 1
    assert isinstance(fake_mx.eval_calls[0], list)
    assert isinstance(fake_mx.eval_calls[0][0], FakeTensor)
    assert all(isinstance(value, FakeScalar) for value in fake_mx.eval_calls[0][1:])


def test_decode_allocator_cache_clear_is_amortized_over_512_steps() -> None:
    class FakeMX:
        def __init__(self) -> None:
            self.clear_calls = 0

        def clear_cache(self) -> None:
            self.clear_calls += 1

    runner = ModelRunner(RuntimeSettings.model_validate({"embeddings": {"enabled": False}}))
    fake_mx = FakeMX()
    runner._mx = fake_mx

    for _ in range(127):
        runner._maybe_clear_decode_cache(4)
    assert fake_mx.clear_calls == 0

    runner._maybe_clear_decode_cache(4)
    assert fake_mx.clear_calls == 1

    diagnostics = runner.decode_diagnostics()
    assert diagnostics["cache_clear_token_budget"] == 512
    assert diagnostics["cache_clear_attempts"] == 1
    assert diagnostics["cache_clears"] == 1
    assert diagnostics["cache_clear_failures"] == 0


def test_decode_allocator_cache_clear_retries_without_consuming_failed_budget() -> None:
    class FakeMX:
        def __init__(self) -> None:
            self.clear_calls = 0

        def clear_cache(self) -> None:
            self.clear_calls += 1
            if self.clear_calls == 1:
                raise RuntimeError("transient allocator failure")

    runner = ModelRunner(RuntimeSettings.model_validate({"embeddings": {"enabled": False}}))
    fake_mx = FakeMX()
    runner._mx = fake_mx
    runner._decode_tokens_since_cache_clear = 511

    runner._maybe_clear_decode_cache(1)

    assert fake_mx.clear_calls == 1
    assert runner._decode_tokens_since_cache_clear == 512
    assert runner.decode_diagnostics()["cache_clear_failures"] == 1

    runner._maybe_clear_decode_cache(1)

    assert fake_mx.clear_calls == 2
    assert runner._decode_tokens_since_cache_clear == 1
    diagnostics = runner.decode_diagnostics()
    assert diagnostics["cache_clear_attempts"] == 2
    assert diagnostics["cache_clears"] == 1


def test_decode_step_advances_cache_clear_cadence_by_generated_tokens() -> None:
    runner = ModelRunner(RuntimeSettings.model_validate({"embeddings": {"enabled": False}}))
    clear_tokens: list[int] = []

    def record_clear_step(generated_tokens: int = 1) -> None:
        clear_tokens.append(generated_tokens)

    def result(item: DecodeWorkItem) -> DecodeResult:
        return DecodeResult(
            prompt_cache=item.prompt_cache,
            token_id=65,
            text="A",
            completion_tokens=item.completion_tokens + 1,
            peak_memory_gb=0.0,
        )

    runner._maybe_clear_decode_cache = record_clear_step  # type: ignore[method-assign]
    runner._decode_single = result  # type: ignore[method-assign]
    runner._decode_batch = lambda items: [result(item) for item in items]  # type: ignore[method-assign]
    items = [
        DecodeWorkItem(
            prompt_cache=[],
            input_token=index,
            sampler=lambda logits: logits,
            detokenizer=object(),
            stop_token_ids=frozenset(),
            logits_processors=(),
            logits_processor_tokens=[],
            completion_tokens=0,
            max_tokens=2,
        )
        for index in (1, 2)
    ]

    runner.decode_batch_step(items[:1])
    runner.decode_batch_step(items)

    assert clear_tokens == [1, 2]


def test_decode_batch_fallback_counts_only_successful_generated_tokens() -> None:
    runner = ModelRunner(RuntimeSettings.model_validate({"embeddings": {"enabled": False}}))
    generated_counts: list[int] = []

    def fail_batch(_items: list[DecodeWorkItem]) -> list[DecodeResult]:
        raise ValueError("merge unavailable")

    def decode_single(item: DecodeWorkItem) -> DecodeResult:
        if item.input_token == 2:
            raise RuntimeError("row failed")
        return DecodeResult(
            prompt_cache=item.prompt_cache,
            token_id=65,
            text="A",
            completion_tokens=1,
            peak_memory_gb=0.0,
        )

    runner._decode_batch = fail_batch  # type: ignore[method-assign]
    runner._decode_single = decode_single  # type: ignore[method-assign]
    runner._maybe_clear_decode_cache = generated_counts.append  # type: ignore[method-assign]
    items = [
        DecodeWorkItem(
            prompt_cache=[],
            input_token=index,
            sampler=lambda logits: logits,
            detokenizer=object(),
            stop_token_ids=frozenset(),
            logits_processors=(),
            logits_processor_tokens=[],
            completion_tokens=0,
            max_tokens=2,
        )
        for index in (1, 2)
    ]

    results = runner.decode_batch_step(items)

    assert isinstance(results[0], DecodeResult)
    assert isinstance(results[1], RuntimeError)
    assert generated_counts == [1]
    diagnostics = runner.decode_diagnostics()
    assert diagnostics["batch_fallbacks"] == 1
    assert diagnostics["batch_fallback_items"] == 2


def test_prefill_cache_eval_resets_decode_clear_budget() -> None:
    class FakeMX:
        @staticmethod
        def eval(_value) -> None:
            return None

        @staticmethod
        def clear_cache() -> None:
            return None

    class FakeCache:
        state = (object(), object())

    runner = ModelRunner(RuntimeSettings.model_validate({"embeddings": {"enabled": False}}))
    runner._mx = FakeMX()
    runner._decode_tokens_since_cache_clear = 500

    runner._eval_cache([FakeCache()])

    assert runner._decode_tokens_since_cache_clear == 0


def test_explicit_runtime_cache_clear_resets_decode_clear_budget() -> None:
    class FakeMX:
        @staticmethod
        def clear_cache() -> None:
            return None

    runner = ModelRunner(RuntimeSettings.model_validate({"embeddings": {"enabled": False}}))
    runner._mx = FakeMX()
    runner._decode_tokens_since_cache_clear = 500

    result = runner.clear_runtime_caches()

    assert result == {"mlx_cache_cleared": True}
    assert runner._decode_tokens_since_cache_clear == 0
