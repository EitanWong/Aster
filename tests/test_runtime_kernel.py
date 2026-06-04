from __future__ import annotations

import pytest

from aster.core.config import RuntimeSettings
from aster.core.errors import ConfigurationError
from aster.inference.runtime_kernel import (
    BatchGeneratorRuntimeKernel,
    ManualRuntimeKernel,
    build_runtime_kernel,
)


def test_build_runtime_kernel_defaults_to_manual() -> None:
    settings = RuntimeSettings.model_validate({"embeddings": {"enabled": False}})
    kernel = build_runtime_kernel(settings)

    assert isinstance(kernel, ManualRuntimeKernel)
    assert kernel.capabilities.name == "manual"
    assert kernel.capabilities.continuous_batching is False
    assert kernel.capabilities.available is True


def test_manual_runtime_kernel_forwards_decode_diagnostics() -> None:
    class FakeRunner:
        def decode_diagnostics(self) -> dict[str, object]:
            return {"batch_attempts": 3, "batch_fallbacks": 1}

    kernel = ManualRuntimeKernel(FakeRunner())  # type: ignore[arg-type]

    assert kernel.decode_diagnostics() == {"batch_attempts": 3, "batch_fallbacks": 1}


def test_manual_runtime_kernel_forwards_strict_prefix_kwargs() -> None:
    calls = []

    class FakeRunner:
        def strict_chat_prefix_prompt(
            self,
            messages,
            *,
            enable_thinking,
            chat_template_kwargs=None,
        ):
            calls.append(
                {
                    "messages": messages,
                    "enable_thinking": enable_thinking,
                    "chat_template_kwargs": chat_template_kwargs,
                }
            )
            return "strict prefix"

    kernel = ManualRuntimeKernel(FakeRunner())  # type: ignore[arg-type]
    result = kernel.strict_chat_prefix_prompt(
        [{"role": "system", "content": "shared"}],
        enable_thinking=True,
        chat_template_kwargs={"enable_thinking": True},
    )

    assert result == "strict prefix"
    assert calls == [
        {
            "messages": [{"role": "system", "content": "shared"}],
            "enable_thinking": True,
            "chat_template_kwargs": {"enable_thinking": True},
        }
    ]


def test_batch_generator_kernel_is_boundary_only() -> None:
    settings = RuntimeSettings.model_validate(
        {
            "embeddings": {"enabled": False},
            "engine": {"runtime_kernel": "batch_generator"},
        }
    )
    kernel = build_runtime_kernel(settings)

    assert isinstance(kernel, BatchGeneratorRuntimeKernel)
    assert kernel.capabilities.name == "batch_generator"
    assert kernel.capabilities.continuous_batching is True
    assert kernel.capabilities.experimental is True
    assert kernel.capabilities.available is False

    with pytest.raises(ConfigurationError) as exc_info:
        kernel.warmup()
    assert exc_info.value.code == "batch_generator_kernel_unavailable"
