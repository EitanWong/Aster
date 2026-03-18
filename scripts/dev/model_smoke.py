from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap
from core.config import load_settings
from inference.mlx_runtime import MLXRuntime


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test Aster MLX model loading and generation")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument(
        "--prompt",
        default="You are a local inference engine. In one sentence, explain why prefix caching helps long-context agents.",
    )
    parser.add_argument("--max-tokens", type=int, default=32)
    args = parser.parse_args()

    settings = load_settings(args.config)
    runtime = MLXRuntime(settings)
    prompt_tokens = runtime.encode(args.prompt)
    prefilled = runtime.prefill_prompt(prompt_tokens)

    print(f"target={settings.model.path}")
    print(f"draft={settings.model.draft_path}")
    print(f"prompt_tokens={len(prompt_tokens)}")
    print(f"prefill_seconds={prefilled.prefill_seconds:.4f}")

    chunks: list[str] = []
    for response in runtime.stream_tokens(
        prompt_tokens,
        prefilled.prompt_cache,
        max_tokens=args.max_tokens,
        temperature=0.0,
        top_p=1.0,
        use_speculative=False,
        num_draft_tokens=0,
    ):
        if response.finish_reason is not None:
            break
        chunks.append(response.text)

    print("completion=")
    print("".join(chunks))


if __name__ == "__main__":
    main()
