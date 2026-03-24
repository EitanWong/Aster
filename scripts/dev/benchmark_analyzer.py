#!/usr/bin/env python3
"""
Aster Benchmark Analyzer - Analyze and compare benchmark results

Provides:
- Performance summary
- Trend analysis
- Comparison between runs
- Performance recommendations
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "benchmark_results"


class BenchmarkAnalyzer:
    """Analyze benchmark results."""

    def __init__(self, results_file: Path) -> None:
        self.results_file = results_file
        self.results = self._load_results()

    def _load_results(self) -> dict[str, Any]:
        """Load benchmark results."""
        if not self.results_file.exists():
            print(f"✗ Results file not found: {self.results_file}")
            sys.exit(1)

        with open(self.results_file) as f:
            return json.load(f)

    def print_summary(self) -> None:
        """Print performance summary."""
        print("\n" + "=" * 70)
        print("  Aster Benchmark Analysis")
        print("=" * 70 + "\n")

        # System info
        sys_info = self.results.get("system_info", {})
        print("System Information:")
        print(f"  CPU Cores:        {sys_info.get('cpu_count', 'N/A')}")
        print(f"  Memory:           {sys_info.get('memory_gb', 'N/A'):.1f} GB")
        print(f"  Available:        {sys_info.get('memory_available_gb', 'N/A'):.1f} GB")
        print(f"  Platform:         {sys_info.get('platform', 'N/A')}")
        print(f"  Timestamp:        {sys_info.get('timestamp', 'N/A')}")
        print()

        # LLM Analysis
        llm_results = self.results.get("llm_benchmarks", [])
        if llm_results:
            print("LLM Performance Analysis:")
            print()
            for result in llm_results:
                if result["success"]:
                    print(f"  {result['test_name']}:")
                    print(f"    Throughput:       {result['tokens_per_second']:.2f} tok/s")
                    print(f"    TTFT:             {result['time_to_first_token_ms']:.1f} ms")
                    print(f"    Total Time:       {result['total_time_ms']:.1f} ms")
                    print(f"    Tokens Generated: {result['completion_tokens']}")
                    print(f"    Memory Peak:      {result['memory_peak_mb']:.1f} MB")
                    print(f"    CPU Avg:          {result['cpu_percent_avg']:.1f}%")
                    print()
                else:
                    print(f"  {result['test_name']}: FAILED - {result['error']}")
                    print()

            # LLM Recommendations
            print("  LLM Recommendations:")
            avg_tps = sum(r["tokens_per_second"] for r in llm_results if r["success"]) / len([r for r in llm_results if r["success"]])
            if avg_tps < 10:
                print("    ⚠️  Low throughput. Consider:")
                print("       - Reducing model size")
                print("       - Enabling speculative decoding")
                print("       - Increasing batch size")
            elif avg_tps < 30:
                print("    ℹ️  Moderate throughput. Performance is acceptable.")
            else:
                print("    ✓ Good throughput. Performance is excellent.")
            print()

        # ASR Analysis
        asr_results = self.results.get("asr_benchmarks", [])
        if asr_results:
            print("ASR Performance Analysis:")
            print()
            for result in asr_results:
                if result["success"]:
                    print(f"  {result['test_name']}:")
                    print(f"    Audio Duration:   {result['audio_duration_sec']:.2f} sec")
                    print(f"    Processing Time:  {result['transcription_time_ms']:.1f} ms")
                    print(f"    Real-Time Factor: {result['real_time_factor']:.2f}x")
                    print(f"    Memory Peak:      {result['memory_peak_mb']:.1f} MB")
                    print(f"    CPU Avg:          {result['cpu_percent_avg']:.1f}%")
                    print()
                else:
                    print(f"  {result['test_name']}: FAILED - {result['error']}")
                    print()

            # ASR Recommendations
            print("  ASR Recommendations:")
            avg_rtf = sum(r["real_time_factor"] for r in asr_results if r["success"]) / len([r for r in asr_results if r["success"]])
            if avg_rtf > 10:
                print("    ⚠️  Slow processing. Consider:")
                print("       - Reducing audio quality")
                print("       - Using smaller model")
                print("       - Disabling other services")
            elif avg_rtf > 1:
                print("    ℹ️  Slower than real-time. Performance is acceptable.")
            else:
                print("    ✓ Faster than real-time. Performance is excellent.")
            print()

        # TTS Analysis
        tts_results = self.results.get("tts_benchmarks", [])
        if tts_results:
            print("TTS Performance Analysis:")
            print()
            for result in tts_results:
                if result["success"]:
                    print(f"  {result['test_name']}:")
                    print(f"    Text Length:      {result['text_length']} chars")
                    print(f"    Synthesis Time:   {result['synthesis_time_ms']:.1f} ms")
                    print(f"    Throughput:       {result['characters_per_second']:.2f} char/s")
                    print(f"    Audio Duration:   {result['audio_duration_sec']:.2f} sec")
                    print(f"    Memory Peak:      {result['memory_peak_mb']:.1f} MB")
                    print(f"    CPU Avg:          {result['cpu_percent_avg']:.1f}%")
                    print()
                else:
                    print(f"  {result['test_name']}: FAILED - {result['error']}")
                    print()

            # TTS Recommendations
            print("  TTS Recommendations:")
            avg_cps = sum(r["characters_per_second"] for r in tts_results if r["success"]) / len([r for r in tts_results if r["success"]])
            if avg_cps < 30:
                print("    ⚠️  Slow synthesis. Consider:")
                print("       - Reducing text length")
                print("       - Using lower quality audio")
                print("       - Disabling other services")
            elif avg_cps < 100:
                print("    ℹ️  Moderate synthesis speed. Performance is acceptable.")
            else:
                print("    ✓ Fast synthesis. Performance is excellent.")
            print()

    def compare_with_previous(self) -> None:
        """Compare with previous benchmark run."""
        # Find all result files
        result_files = sorted(RESULTS_DIR.glob("*.json"))

        if len(result_files) < 2:
            print("✗ Need at least 2 benchmark runs to compare")
            return

        # Load previous result
        with open(result_files[-2]) as f:
            previous = json.load(f)

        print("\n" + "=" * 70)
        print("  Benchmark Comparison (Previous vs Current)")
        print("=" * 70 + "\n")

        # Compare LLM
        current_llm = self.results.get("llm_benchmarks", [])
        previous_llm = previous.get("llm_benchmarks", [])

        if current_llm and previous_llm:
            print("LLM Performance Change:")
            for curr, prev in zip(current_llm, previous_llm):
                if curr["success"] and prev["success"]:
                    change = ((curr["tokens_per_second"] - prev["tokens_per_second"]) / prev["tokens_per_second"]) * 100
                    symbol = "↑" if change > 0 else "↓" if change < 0 else "→"
                    print(f"  {curr['test_name']:20} {symbol} {abs(change):6.1f}%")
            print()

        # Compare ASR
        current_asr = self.results.get("asr_benchmarks", [])
        previous_asr = previous.get("asr_benchmarks", [])

        if current_asr and previous_asr:
            print("ASR Performance Change:")
            for curr, prev in zip(current_asr, previous_asr):
                if curr["success"] and prev["success"]:
                    # For RTF, lower is better, so invert the comparison
                    change = ((prev["real_time_factor"] - curr["real_time_factor"]) / prev["real_time_factor"]) * 100
                    symbol = "↑" if change > 0 else "↓" if change < 0 else "→"
                    print(f"  {curr['test_name']:20} {symbol} {abs(change):6.1f}%")
            print()

        # Compare TTS
        current_tts = self.results.get("tts_benchmarks", [])
        previous_tts = previous.get("tts_benchmarks", [])

        if current_tts and previous_tts:
            print("TTS Performance Change:")
            for curr, prev in zip(current_tts, previous_tts):
                if curr["success"] and prev["success"]:
                    change = ((curr["characters_per_second"] - prev["characters_per_second"]) / prev["characters_per_second"]) * 100
                    symbol = "↑" if change > 0 else "↓" if change < 0 else "→"
                    print(f"  {curr['test_name']:20} {symbol} {abs(change):6.1f}%")
            print()


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Aster Benchmark Analyzer - Analyze benchmark results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze latest benchmark
  python benchmark_analyzer.py

  # Analyze specific benchmark
  python benchmark_analyzer.py --file benchmark_results/20260319_174937.json

  # Compare with previous run
  python benchmark_analyzer.py --compare
        """,
    )

    parser.add_argument(
        "--file",
        type=Path,
        help="Benchmark results file (default: latest)",
    )

    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare with previous benchmark run",
    )

    args = parser.parse_args()

    # Find results file
    if args.file:
        results_file = args.file
    else:
        # Find latest results file
        result_files = sorted(RESULTS_DIR.glob("*.json"))
        if not result_files:
            print(f"✗ No benchmark results found in {RESULTS_DIR}")
            return 1
        results_file = result_files[-1]

    print(f"Analyzing: {results_file}")

    analyzer = BenchmarkAnalyzer(results_file)
    analyzer.print_summary()

    if args.compare:
        analyzer.compare_with_previous()

    return 0


if __name__ == "__main__":
    sys.exit(main())
