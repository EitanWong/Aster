from __future__ import annotations

from typing import Any

import pytest

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
from mlx_lm.models.cache import ArraysCache, KVCache


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


def test_configure_prompt_cache_step_only_updates_matching_cache_type() -> None:
    class FakeKVCache:
        def __init__(self) -> None:
            self.step = 256

    runner = ModelRunner(RuntimeSettings.model_validate({"embeddings": {"enabled": False}}))
    kv_cache = FakeKVCache()
    other_cache = object()

    configured = runner._configure_prompt_cache_step(
        [kv_cache, other_cache], FakeKVCache, 2048
    )

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
