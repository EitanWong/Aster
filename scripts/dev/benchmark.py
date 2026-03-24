#!/usr/bin/env python3
"""
Aster Benchmark Suite - Comprehensive performance testing

Tests performance of:
- LLM (Large Language Model) inference
- ASR (Automatic Speech Recognition)
- TTS (Text-to-Speech synthesis)

Measures:
- Throughput (tokens/sec, samples/sec)
- Latency (first token, end-to-end)
- Memory usage
- CPU usage
- Power efficiency
"""

from __future__ import annotations

import argparse
import asyncio
import json
import psutil
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx


PROJECT_ROOT = Path(__file__).parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "benchmark_results"


@dataclass
class SystemInfo:
    """System information."""
    cpu_count: int
    cpu_freq_ghz: float
    memory_gb: int
    memory_available_gb: float
    platform: str
    timestamp: str


@dataclass
class LLMBenchmarkResult:
    """LLM benchmark result."""
    test_name: str
    prompt: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    time_to_first_token_ms: float
    total_time_ms: float
    tokens_per_second: float
    memory_peak_mb: float
    memory_avg_mb: float
    cpu_percent_avg: float
    success: bool
    error: str | None = None


@dataclass
class ASRBenchmarkResult:
    """ASR benchmark result."""
    test_name: str
    audio_duration_sec: float
    audio_size_kb: float
    transcription_time_ms: float
    real_time_factor: float  # audio_duration / transcription_time
    memory_peak_mb: float
    memory_avg_mb: float
    cpu_percent_avg: float
    success: bool
    error: str | None = None


@dataclass
class TTSBenchmarkResult:
    """TTS benchmark result."""
    test_name: str
    text_length: int
    audio_duration_sec: float
    synthesis_time_ms: float
    characters_per_second: float
    memory_peak_mb: float
    memory_avg_mb: float
    cpu_percent_avg: float
    success: bool
    error: str | None = None


class SystemMonitor:
    """Monitor system resources during benchmark."""

    def __init__(self, interval: float = 0.1) -> None:
        self.interval = interval
        self.process = psutil.Process()
        self.measurements: list[dict[str, float]] = []
        self.monitoring = False

    async def monitor(self) -> None:
        """Monitor system resources."""
        self.measurements = []
        self.monitoring = True

        try:
            while self.monitoring:
                measurement = {
                    "timestamp": time.time(),
                    "memory_mb": self.process.memory_info().rss / 1024 / 1024,
                    "cpu_percent": self.process.cpu_percent(interval=0.01),
                }
                self.measurements.append(measurement)
                await asyncio.sleep(self.interval)
        except Exception:
            pass

    def stop(self) -> None:
        """Stop monitoring."""
        self.monitoring = False

    def get_stats(self) -> dict[str, float]:
        """Get monitoring statistics."""
        if not self.measurements:
            return {
                "memory_peak_mb": 0,
                "memory_avg_mb": 0,
                "cpu_percent_avg": 0,
            }

        memory_values = [m["memory_mb"] for m in self.measurements]
        cpu_values = [m["cpu_percent"] for m in self.measurements if m["cpu_percent"] >= 0]

        return {
            "memory_peak_mb": max(memory_values) if memory_values else 0,
            "memory_avg_mb": sum(memory_values) / len(memory_values) if memory_values else 0,
            "cpu_percent_avg": sum(cpu_values) / len(cpu_values) if cpu_values else 0,
        }


class AsterBenchmark:
    """Aster benchmark suite."""

    def __init__(self, api_url: str = "http://127.0.0.1:8080") -> None:
        self.api_url = api_url
        self.client = httpx.AsyncClient(timeout=300.0)
        self.system_info = self._get_system_info()

    def _get_system_info(self) -> SystemInfo:
        """Get system information."""
        cpu_freq = psutil.cpu_freq()
        memory = psutil.virtual_memory()

        return SystemInfo(
            cpu_count=psutil.cpu_count(),
            cpu_freq_ghz=cpu_freq.max / 1000 if cpu_freq else 0,
            memory_gb=memory.total / 1024 / 1024 / 1024,
            memory_available_gb=memory.available / 1024 / 1024 / 1024,
            platform=sys.platform,
            timestamp=datetime.now().isoformat(),
        )

    async def benchmark_llm(
        self,
        prompt: str,
        model: str = "Qwen3.5-9B",
        max_tokens: int = 100,
        test_name: str = "LLM Inference",
    ) -> LLMBenchmarkResult:
        """Benchmark LLM inference."""
        monitor = SystemMonitor()
        monitor_task = asyncio.create_task(monitor.monitor())

        try:
            start_time = time.time()
            first_token_time = None
            completion_tokens = 0

            async with self.client.stream(
                "POST",
                f"{self.api_url}/v1/chat/completions",
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "stream": True,
                },
            ) as response:
                async for line in response.aiter_lines():
                    if first_token_time is None:
                        first_token_time = time.time() - start_time

                    if line.startswith("data: "):
                        try:
                            data = json.loads(line[6:])
                            if data.get("choices"):
                                delta = data["choices"][0].get("delta", {})
                                if "content" in delta:
                                    completion_tokens += 1
                        except json.JSONDecodeError:
                            pass

            total_time = time.time() - start_time
            monitor.stop()
            await monitor_task

            stats = monitor.get_stats()

            # Estimate token counts (rough approximation)
            prompt_tokens = len(prompt.split())
            total_tokens = prompt_tokens + completion_tokens

            return LLMBenchmarkResult(
                test_name=test_name,
                prompt=prompt[:100],
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                time_to_first_token_ms=(first_token_time or total_time) * 1000,
                total_time_ms=total_time * 1000,
                tokens_per_second=completion_tokens / total_time if total_time > 0 else 0,
                memory_peak_mb=stats["memory_peak_mb"],
                memory_avg_mb=stats["memory_avg_mb"],
                cpu_percent_avg=stats["cpu_percent_avg"],
                success=True,
            )
        except Exception as e:
            monitor.stop()
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass

            return LLMBenchmarkResult(
                test_name=test_name,
                prompt=prompt[:100],
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                time_to_first_token_ms=0,
                total_time_ms=0,
                tokens_per_second=0,
                memory_peak_mb=0,
                memory_avg_mb=0,
                cpu_percent_avg=0,
                success=False,
                error=str(e),
            )

    async def benchmark_asr(
        self,
        audio_file: Path,
        test_name: str = "ASR Transcription",
    ) -> ASRBenchmarkResult:
        """Benchmark ASR."""
        monitor = SystemMonitor()
        monitor_task = asyncio.create_task(monitor.monitor())

        try:
            if not audio_file.exists():
                raise FileNotFoundError(f"Audio file not found: {audio_file}")

            audio_data = audio_file.read_bytes()
            audio_size_kb = len(audio_data) / 1024

            # Get audio duration (rough estimate: 16-bit mono at 16kHz)
            # This is approximate; ideally use librosa or similar
            audio_duration_sec = len(audio_data) / (2 * 16000)

            start_time = time.time()

            response = await self.client.post(
                f"{self.api_url}/v1/audio/transcriptions",
                files={"file": ("audio.wav", audio_data, "audio/wav")},
            )

            transcription_time = time.time() - start_time
            monitor.stop()
            await monitor_task

            if response.status_code != 200:
                raise Exception(f"ASR failed with status {response.status_code}")

            data = response.json()
            stats = monitor.get_stats()

            return ASRBenchmarkResult(
                test_name=test_name,
                audio_duration_sec=audio_duration_sec,
                audio_size_kb=audio_size_kb,
                transcription_time_ms=transcription_time * 1000,
                real_time_factor=audio_duration_sec / transcription_time if transcription_time > 0 else 0,
                memory_peak_mb=stats["memory_peak_mb"],
                memory_avg_mb=stats["memory_avg_mb"],
                cpu_percent_avg=stats["cpu_percent_avg"],
                success=True,
            )
        except Exception as e:
            monitor.stop()
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass

            return ASRBenchmarkResult(
                test_name=test_name,
                audio_duration_sec=0,
                audio_size_kb=0,
                transcription_time_ms=0,
                real_time_factor=0,
                memory_peak_mb=0,
                memory_avg_mb=0,
                cpu_percent_avg=0,
                success=False,
                error=str(e),
            )

    async def benchmark_tts(
        self,
        text: str,
        test_name: str = "TTS Synthesis",
    ) -> TTSBenchmarkResult:
        """Benchmark TTS."""
        monitor = SystemMonitor()
        monitor_task = asyncio.create_task(monitor.monitor())

        try:
            start_time = time.time()

            response = await self.client.post(
                f"{self.api_url}/v1/audio/speech",
                json={
                    "model": "Qwen3-TTS",
                    "input": text,
                    "voice": "default",
                    "response_format": "wav",
                },
            )

            synthesis_time = time.time() - start_time
            monitor.stop()
            await monitor_task

            if response.status_code != 200:
                error_msg = response.text if response.text else f"HTTP {response.status_code}"
                raise Exception(f"TTS failed: {error_msg}")

            # Estimate audio duration from response size
            # Assuming 16-bit mono at 22050 Hz
            audio_size = len(response.content)
            audio_duration_sec = audio_size / (2 * 22050)

            stats = monitor.get_stats()

            return TTSBenchmarkResult(
                test_name=test_name,
                text_length=len(text),
                audio_duration_sec=audio_duration_sec,
                synthesis_time_ms=synthesis_time * 1000,
                characters_per_second=len(text) / synthesis_time if synthesis_time > 0 else 0,
                memory_peak_mb=stats["memory_peak_mb"],
                memory_avg_mb=stats["memory_avg_mb"],
                cpu_percent_avg=stats["cpu_percent_avg"],
                success=True,
            )
        except Exception as e:
            monitor.stop()
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass

            return TTSBenchmarkResult(
                test_name=test_name,
                text_length=len(text),
                audio_duration_sec=0,
                synthesis_time_ms=0,
                characters_per_second=0,
                memory_peak_mb=0,
                memory_avg_mb=0,
                cpu_percent_avg=0,
                success=False,
                error=str(e),
            )

    async def run_full_benchmark(self) -> dict[str, Any]:
        """Run full benchmark suite."""
        print("\n" + "=" * 70)
        print("  Aster Comprehensive Benchmark Suite")
        print("=" * 70 + "\n")

        # System info
        print("System Information:")
        print(f"  CPU Cores:        {self.system_info.cpu_count}")
        print(f"  CPU Freq:         {self.system_info.cpu_freq_ghz:.2f} GHz")
        print(f"  Memory:           {self.system_info.memory_gb:.2f} GB")
        print(f"  Available:        {self.system_info.memory_available_gb:.2f} GB")
        print(f"  Platform:         {self.system_info.platform}")
        print()

        results = {
            "system_info": asdict(self.system_info),
            "llm_benchmarks": [],
            "asr_benchmarks": [],
            "tts_benchmarks": [],
        }

        # LLM Benchmarks
        print("Running LLM Benchmarks...")
        llm_tests = [
            ("Short prompt", "What is 2+2?", 50),
            ("Medium prompt", "Explain quantum computing in simple terms.", 100),
            ("Long prompt", "Write a detailed explanation of machine learning, including supervised and unsupervised learning, neural networks, and practical applications.", 150),
        ]

        for test_name, prompt, max_tokens in llm_tests:
            print(f"  {test_name}...", end=" ", flush=True)
            result = await self.benchmark_llm(prompt, test_name=test_name, max_tokens=max_tokens)
            results["llm_benchmarks"].append(asdict(result))
            if result.success:
                print(f"✓ {result.tokens_per_second:.2f} tok/s")
            else:
                print(f"✗ {result.error}")

        print()

        # ASR Benchmarks
        print("Running ASR Benchmarks...")
        audio_file = Path("/Users/eitan/Desktop/PromptAudio.wav")
        if audio_file.exists():
            print(f"  Testing with {audio_file.name}...", end=" ", flush=True)
            result = await self.benchmark_asr(audio_file, test_name="ASR Test")
            results["asr_benchmarks"].append(asdict(result))
            if result.success:
                print(f"✓ RTF: {result.real_time_factor:.2f}x")
            else:
                print(f"✗ {result.error}")
        else:
            print(f"  ✗ Audio file not found: {audio_file}")

        print()

        # TTS Benchmarks
        print("Running TTS Benchmarks...")
        tts_tests = [
            ("Short text", "Hello world"),
            ("Medium text", "The quick brown fox jumps over the lazy dog. This is a test of the text-to-speech system."),
            ("Long text", "Artificial intelligence is transforming the world. From healthcare to transportation, AI is revolutionizing how we live and work. Machine learning models can now understand language, recognize images, and make predictions with remarkable accuracy."),
        ]

        for test_name, text in tts_tests:
            print(f"  {test_name}...", end=" ", flush=True)
            result = await self.benchmark_tts(text, test_name=test_name)
            results["tts_benchmarks"].append(asdict(result))
            if result.success:
                print(f"✓ {result.characters_per_second:.2f} char/s")
            else:
                print(f"✗ {result.error}")

        print()

        return results

    async def close(self) -> None:
        """Close HTTP client."""
        await self.client.aclose()


async def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Aster Benchmark Suite - Comprehensive performance testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full benchmark suite
  python benchmark.py

  # Run with custom API URL
  python benchmark.py --api-url http://localhost:8080

  # Save results to file
  python benchmark.py --output results.json
        """,
    )

    parser.add_argument(
        "--api-url",
        default="http://127.0.0.1:8080",
        help="API URL (default: http://127.0.0.1:8080)",
    )

    parser.add_argument(
        "--output",
        type=Path,
        help="Save results to JSON file",
    )

    args = parser.parse_args()

    benchmark = AsterBenchmark(api_url=args.api_url)

    try:
        results = await benchmark.run_full_benchmark()

        # Print summary
        print("=" * 70)
        print("  Benchmark Summary")
        print("=" * 70 + "\n")

        if results["llm_benchmarks"]:
            print("LLM Performance:")
            for result in results["llm_benchmarks"]:
                if result["success"]:
                    print(f"  {result['test_name']:20} {result['tokens_per_second']:8.2f} tok/s  "
                          f"TTFT: {result['time_to_first_token_ms']:6.1f}ms  "
                          f"Memory: {result['memory_peak_mb']:6.1f}MB")
                else:
                    print(f"  {result['test_name']:20} FAILED: {result['error']}")
            print()

        if results["asr_benchmarks"]:
            print("ASR Performance:")
            for result in results["asr_benchmarks"]:
                if result["success"]:
                    print(f"  {result['test_name']:20} RTF: {result['real_time_factor']:6.2f}x  "
                          f"Time: {result['transcription_time_ms']:8.1f}ms  "
                          f"Memory: {result['memory_peak_mb']:6.1f}MB")
                else:
                    print(f"  {result['test_name']:20} FAILED: {result['error']}")
            print()

        if results["tts_benchmarks"]:
            print("TTS Performance:")
            for result in results["tts_benchmarks"]:
                if result["success"]:
                    print(f"  {result['test_name']:20} {result['characters_per_second']:8.2f} char/s  "
                          f"Time: {result['synthesis_time_ms']:8.1f}ms  "
                          f"Memory: {result['memory_peak_mb']:6.1f}MB")
                else:
                    print(f"  {result['test_name']:20} FAILED: {result['error']}")
            print()

        # Save results if requested
        if args.output:
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(results, indent=2))
            print(f"Results saved to: {args.output}")

        return 0
    except Exception as e:
        print(f"✗ Benchmark failed: {e}", file=sys.stderr)
        return 1
    finally:
        await benchmark.close()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
