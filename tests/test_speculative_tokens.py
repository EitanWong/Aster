#!/usr/bin/env python3
"""
Manual test for Speculative Decoding token output speed.
Uses Aster's actual initialization logic.
"""

import time
import json
import asyncio
from pathlib import Path
from typing import Dict
import sys

# Add Aster to path
sys.path.insert(0, str(Path(__file__).parent))

from aster.core.lifecycle import create_application
from aster.core.config import load_settings


async def measure_tokens_per_second(
    inference_engine,
    prompt: str,
    max_tokens: int = 100,
    speculative_enabled: bool = False,
) -> Dict:
    """Measure tokens/second for a given prompt."""

    print(f"\n{'='*60}")
    print(f"Mode: {'Speculative' if speculative_enabled else 'Non-Speculative'}")
    print(f"Prompt: {prompt[:50]}...")
    print(f"Max tokens: {max_tokens}")
    print(f"{'='*60}")

    # Prepare settings
    inference_engine.settings.speculative.enabled = speculative_enabled

    # Warm up
    print("Warming up...")
    from inference.engine import InferenceRequest
    warmup_req = InferenceRequest(
        prompt=prompt,
        max_tokens=10,
        temperature=0.7,
        top_p=0.9,
    )
    _ = await inference_engine.infer(warmup_req)

    # Measure
    print("Measuring...")
    start_time = time.time()

    test_req = InferenceRequest(
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=0.7,
        top_p=0.9,
    )

    result = await inference_engine.infer(test_req)

    end_time = time.time()
    elapsed = end_time - start_time

    # Extract metrics
    tokens_per_second = result.generation_tps

    result_dict = {
        "mode": "speculative" if speculative_enabled else "non-speculative",
        "elapsed_seconds": round(elapsed, 3),
        "completion_tokens": result.completion_tokens,
        "tokens_per_second": round(tokens_per_second, 2),
        "generation_tps": round(result.generation_tps, 2),
        "prompt_tps": round(result.prompt_tps, 2),
        "cache_hit": result.cache_hit,
        "prefill_cache_hit": result.prefill_cache_hit,
        "speculative_path_mode": result.speculative_path_mode,
    }

    print(f"\nResults:")
    print(f"  Elapsed: {result_dict['elapsed_seconds']}s")
    print(f"  Completion tokens: {result_dict['completion_tokens']}")
    print(f"  Generation TPS: {result_dict['generation_tps']}")
    print(f"  Prompt TPS: {result_dict['prompt_tps']}")
    print(f"  Cache hit: {result_dict['cache_hit']}")
    print(f"  Prefill cache hit: {result_dict['prefill_cache_hit']}")
    print(f"  Speculative path mode: {result_dict['speculative_path_mode']}")

    return result_dict


async def main():
    """Run the test."""

    print("\n" + "="*60)
    print("  Aster Speculative Decoding Token Speed Test")
    print("="*60)

    # Load config
    config_path = Path(__file__).parent / "configs" / "config.yaml"
    print(f"\nLoading config from: {config_path}")

    if not config_path.exists():
        print(f"Error: Config file not found at {config_path}")
        sys.exit(1)

    try:
        settings = load_settings(str(config_path))
        print(f"✅ Config loaded successfully")
        print(f"   Model: {settings.model.name}")
        print(f"   Draft model: {settings.model.draft_name}")
        print(f"   Speculative enabled: {settings.speculative.enabled}")
    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)

    # Initialize application
    print("\nInitializing inference engine...")
    try:
        app = create_application(str(config_path))
        container = app.state.container
        inference_engine = container.inference_engine
        print("✅ Inference engine initialized")
    except Exception as e:
        print(f"Error initializing engine: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Test prompts
    test_prompts = [
        "Explain quantum computing in simple terms:",
        "Write a Python function to calculate fibonacci numbers:",
        "What are the benefits of machine learning?",
    ]

    results = []

    # Test non-speculative mode
    print("\n" + "="*60)
    print("PHASE 1: Non-Speculative Mode")
    print("="*60)

    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n[{i}/{len(test_prompts)}]")
        try:
            result = await measure_tokens_per_second(
                inference_engine,
                prompt,
                max_tokens=100,
                speculative_enabled=False,
            )
            results.append(result)
        except Exception as e:
            print(f"Error in test: {e}")
            import traceback
            traceback.print_exc()

    # Test speculative mode
    print("\n" + "="*60)
    print("PHASE 2: Speculative Mode")
    print("="*60)

    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n[{i}/{len(test_prompts)}]")
        try:
            result = await measure_tokens_per_second(
                inference_engine,
                prompt,
                max_tokens=100,
                speculative_enabled=True,
            )
            results.append(result)
        except Exception as e:
            print(f"Error in test: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "="*60)
    print("  SUMMARY")
    print("="*60)

    non_spec_results = [r for r in results if r["mode"] == "non-speculative"]
    spec_results = [r for r in results if r["mode"] == "speculative"]

    if non_spec_results:
        avg_non_spec = sum(r["tokens_per_second"] for r in non_spec_results) / len(non_spec_results)
        print(f"\nNon-Speculative Average: {avg_non_spec:.2f} tokens/second")

    if spec_results:
        avg_spec = sum(r["tokens_per_second"] for r in spec_results) / len(spec_results)
        print(f"Speculative Average: {avg_spec:.2f} tokens/second")

    if non_spec_results and spec_results:
        speedup = avg_spec / avg_non_spec if avg_non_spec > 0 else 0
        print(f"\nSpeedup: {speedup:.2f}x")
        if speedup > 1:
            print(f"✅ Speculative mode is {(speedup-1)*100:.1f}% faster")
        else:
            print(f"⚠️ Speculative mode is {(1-speedup)*100:.1f}% slower")

    # Save results
    results_file = Path(__file__).parent / "reports" / "speculative_tokens_test.json"
    results_file.parent.mkdir(parents=True, exist_ok=True)

    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {results_file}")
    print("="*60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
