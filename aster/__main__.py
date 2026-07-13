from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from typing import Any

import uvicorn

from aster.core.config import RuntimeSettings, load_settings
from aster.core.lifecycle import create_application, create_application_from_settings
from aster.core.process_title import build_aster_process_title, set_process_title

TOOL_CALL_PARSERS = [
    "auto",
    "mistral",
    "qwen",
    "qwen3_coder",
    "llama",
    "hermes",
    "harmony",
    "gpt-oss",
    "deepseek",
    "kimi",
    "granite",
    "nemotron",
    "xlam",
    "functionary",
    "gemma4",
    "glm47",
    "minimax",
]


def _json_object(value: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("expected a JSON object")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aster: vllm-mlx-compatible Apple Silicon inference server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  aster serve mlx-community/Llama-3.2-3B-Instruct-4bit --port 8000
  aster --config configs/config.yaml
        """,
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to YAML config file. Kept for existing Aster deployments.",
    )
    subparsers = parser.add_subparsers(dest="command")
    serve_parser = subparsers.add_parser("serve", help="Start OpenAI-compatible server")
    serve_parser.add_argument("model", nargs="?", help="Model path or Hugging Face model id")
    serve_parser.add_argument("--models-config", default=None)
    serve_parser.add_argument("--served-model-name", default=None)
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8000)
    serve_parser.add_argument("--max-num-seqs", type=int, default=256)
    serve_parser.add_argument("--prefill-batch-size", type=int, default=8)
    serve_parser.add_argument("--completion-batch-size", type=int, default=32)
    serve_parser.add_argument("--mllm-prefill-step-size", type=int, default=0)
    serve_parser.add_argument("--enable-prefix-cache", action="store_true", default=True)
    serve_parser.add_argument("--disable-prefix-cache", action="store_true")
    serve_parser.add_argument("--prefix-cache-size", type=int, default=100)
    serve_parser.add_argument("--cache-memory-mb", type=int, default=None)
    serve_parser.add_argument("--cache-memory-percent", type=float, default=0.20)
    serve_parser.add_argument("--no-memory-aware-cache", action="store_true")
    serve_parser.add_argument("--kv-cache-quantization", action="store_true")
    serve_parser.add_argument("--kv-cache-quantization-bits", type=int, choices=[4, 8], default=8)
    serve_parser.add_argument("--kv-cache-quantization-group-size", type=int, default=64)
    serve_parser.add_argument("--kv-cache-min-quantize-tokens", type=int, default=256)
    serve_parser.add_argument("--ssd-cache-dir", default=None)
    serve_parser.add_argument("--ssd-cache-max-gb", type=float, default=10.0)
    serve_parser.add_argument("--warm-prompts", default=None)
    serve_parser.add_argument("--stream-interval", type=int, default=1)
    serve_parser.add_argument("--max-kv-size", type=int, default=None)
    serve_parser.add_argument("--max-tokens", type=int, default=32768)
    serve_parser.add_argument("--max-request-tokens", type=int, default=32768)
    serve_parser.add_argument("--continuous-batching", action="store_true")
    serve_parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    serve_parser.add_argument("--use-paged-cache", action="store_true")
    serve_parser.add_argument("--paged-cache-block-size", type=int, default=64)
    serve_parser.add_argument("--max-cache-blocks", type=int, default=1000)
    serve_parser.add_argument("--chunked-prefill-tokens", type=int, default=0)
    serve_parser.add_argument("--enable-mtp", action="store_true")
    serve_parser.add_argument("--mtp-num-draft-tokens", type=int, default=1)
    serve_parser.add_argument("--mtp-optimistic", action="store_true")
    serve_parser.add_argument("--prefill-step-size", type=int, default=2048)
    serve_parser.add_argument("--specprefill", action="store_true")
    serve_parser.add_argument("--specprefill-threshold", type=int, default=8192)
    serve_parser.add_argument("--specprefill-keep-pct", type=float, default=0.3)
    serve_parser.add_argument("--specprefill-draft-model", default=None)
    serve_parser.add_argument("--mcp-config", default=None)
    serve_parser.add_argument("--api-key", default=None)
    serve_parser.add_argument("--rate-limit", type=int, default=0)
    serve_parser.add_argument("--timeout", type=float, default=300.0)
    serve_parser.add_argument("--enable-metrics", action="store_true")
    serve_parser.add_argument("--auto-unload-idle-seconds", type=float, default=0.0)
    serve_parser.add_argument("--lazy-load-model", action="store_true")
    serve_parser.add_argument("--max-audio-upload-mb", type=int, default=25)
    serve_parser.add_argument("--max-tts-input-chars", type=int, default=4096)
    serve_parser.add_argument("--enable-auto-tool-choice", action="store_true")
    serve_parser.add_argument("--tool-call-parser", choices=TOOL_CALL_PARSERS, default=None)
    serve_parser.add_argument("--reasoning-parser", default=None)
    serve_parser.add_argument("--mllm", action="store_true")
    serve_parser.add_argument("--trust-remote-code", action="store_true")
    serve_parser.add_argument("--default-temperature", type=float, default=None)
    serve_parser.add_argument("--default-top-p", type=float, default=None)
    serve_parser.add_argument("--default-thinking-token-budget", type=int, default=None)
    serve_parser.add_argument("--default-chat-template-kwargs", type=_json_object, default=None)
    serve_parser.add_argument("--embedding-model", default=None)
    serve_parser.add_argument("--rerank-model", default=None)
    serve_parser.add_argument("--download-timeout", type=int, default=300)
    serve_parser.add_argument("--download-retries", type=int, default=3)
    serve_parser.add_argument("--offline", action="store_true")
    return parser


def settings_from_serve_args(args: argparse.Namespace) -> RuntimeSettings:
    if not args.model and not args.models_config:
        raise SystemExit("serve requires MODEL unless --models-config is provided")

    model_id = args.model or args.models_config
    served_name = args.served_model_name or model_id
    prefix_cache_enabled = bool(args.enable_prefix_cache) and not args.disable_prefix_cache
    prefill_token_budget = (
        args.chunked_prefill_tokens
        if args.chunked_prefill_tokens and args.chunked_prefill_tokens > 0
        else 1024
    )
    memory_headroom_ratio = max(0.0, min(1.0, 1.0 - float(args.gpu_memory_utilization)))
    default_chat_template_kwargs = dict(args.default_chat_template_kwargs or {})
    default_enable_thinking = default_chat_template_kwargs.get("enable_thinking", True)
    if not isinstance(default_enable_thinking, bool):
        default_enable_thinking = True
    engine_type = "batched" if getattr(args, "continuous_batching", False) else "manual"
    warnings = _serve_compatibility_warnings(args)

    settings = RuntimeSettings.model_validate(
        {
            "api": {
                "host": args.host,
                "port": args.port,
                "request_timeout_seconds": args.timeout,
                "max_request_tokens": args.max_request_tokens,
                "api_key": args.api_key,
                "rate_limit_per_minute": args.rate_limit,
            },
            "model": {
                "name": served_name,
                "path": model_id,
                "enable_thinking": default_enable_thinking,
            },
            "engine": {
                "engine_type": engine_type,
                "max_active_requests": args.max_num_seqs,
                "max_decode_batch": args.completion_batch_size,
                "prefill_token_budget": prefill_token_budget,
                "snapshot_max_entries": args.prefix_cache_size,
                "prefix_cache_enabled": prefix_cache_enabled,
                "warm_prompts_path": args.warm_prompts,
                "stream_interval_tokens": args.stream_interval,
                "memory_headroom_ratio": memory_headroom_ratio,
            },
            "cache": {
                "prefix_cache_enabled": prefix_cache_enabled,
                "prefix_cache_max_entries": args.prefix_cache_size,
                "kv_page_tokens": args.paged_cache_block_size,
                "kv_max_pages": args.max_cache_blocks,
            },
            "batch": {
                "max_batch_size": args.max_num_seqs,
                "prefill_batch_size": args.prefill_batch_size,
                "decode_batch_size": args.completion_batch_size,
            },
            "embeddings": {
                "enabled": args.embedding_model is not None,
                "model": args.embedding_model or "mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ",
            },
            "audio": {
                "max_audio_upload_mb": args.max_audio_upload_mb,
                "max_tts_input_chars": args.max_tts_input_chars,
            },
        }
    )
    return settings.model_copy(update={"deprecation_warnings": tuple(warnings)})


def _serve_compatibility_warnings(args: argparse.Namespace) -> list[str]:
    warnings: list[str] = []
    unsupported_flags = {
        "models_config": args.models_config,
        "continuous_batching": args.continuous_batching,
        "mllm_prefill_step_size": args.mllm_prefill_step_size,
        "cache_memory_mb": args.cache_memory_mb,
        "no_memory_aware_cache": args.no_memory_aware_cache,
        "kv_cache_quantization": args.kv_cache_quantization,
        "ssd_cache_dir": args.ssd_cache_dir,
        "max_kv_size": args.max_kv_size,
        "use_paged_cache": args.use_paged_cache,
        "enable_mtp": args.enable_mtp,
        "mtp_optimistic": args.mtp_optimistic,
        "specprefill": args.specprefill,
        "specprefill_draft_model": args.specprefill_draft_model,
        "mcp_config": args.mcp_config,
        "enable_metrics": args.enable_metrics,
        "auto_unload_idle_seconds": args.auto_unload_idle_seconds,
        "lazy_load_model": args.lazy_load_model,
        "enable_auto_tool_choice": args.enable_auto_tool_choice,
        "tool_call_parser": args.tool_call_parser,
        "reasoning_parser": args.reasoning_parser,
        "mllm": args.mllm,
        "trust_remote_code": args.trust_remote_code,
        "default_temperature": args.default_temperature,
        "default_top_p": args.default_top_p,
        "default_thinking_token_budget": args.default_thinking_token_budget,
        "rerank_model": args.rerank_model,
        "offline": args.offline,
    }
    requested = sorted(name for name, value in unsupported_flags.items() if value)
    if requested:
        warnings.append(
            "serve accepted vllm-mlx flags that are parsed but not fully implemented yet: "
            + ", ".join(requested)
        )
    return warnings


def settings_from_args(args: argparse.Namespace) -> RuntimeSettings:
    if args.command == "serve":
        return settings_from_serve_args(args)
    if args.config:
        return load_settings(args.config)
    raise SystemExit("expected 'serve MODEL ...' or '--config PATH'")


def app_from_args(args: argparse.Namespace):
    if args.command == "serve":
        return create_application_from_settings(settings_from_serve_args(args))
    if args.config:
        return create_application(args.config)
    raise SystemExit("expected 'serve MODEL ...' or '--config PATH'")


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    bootstrap_settings = settings_from_args(args)
    set_process_title(build_aster_process_title(bootstrap_settings))

    app = app_from_args(args)
    settings = app.state.container.settings
    uvicorn.run(
        app,
        host=settings.api.host,
        port=settings.api.port,
        log_level=settings.logging.level.lower(),
    )


if __name__ == "__main__":
    main(sys.argv[1:])
