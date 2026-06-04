from __future__ import annotations

import argparse

from aster.core.config import load_settings
from aster.inference.contracts import InferenceRequest
from aster.inference.engine import InferenceEngine
from aster.telemetry.metrics import MetricsRegistry


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
    metrics = MetricsRegistry(settings.telemetry.metrics_namespace)
    engine = InferenceEngine(settings, metrics)

    async def _run() -> None:
        await engine.start()
        try:
            result = await engine.infer(
                InferenceRequest(
                    prompt=args.prompt,
                    max_tokens=args.max_tokens,
                    temperature=0.0,
                    top_p=1.0,
                )
            )
        finally:
            await engine.aclose()

        print(f"model={settings.model.path}")
        print(f"prompt_tokens={result.prompt_tokens}")
        print(f"prompt_tps={result.prompt_tps:.4f}")
        print(f"generation_tps={result.generation_tps:.4f}")
        print("completion=")
        print(result.text)

    import asyncio

    asyncio.run(_run())


if __name__ == "__main__":
    main()
