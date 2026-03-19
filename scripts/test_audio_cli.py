#!/usr/bin/env python3
"""Simple CLI test script for ASR and TTS services.

Usage:
    python test_audio_cli.py --health
    python test_audio_cli.py --tts "Hello world"
    python test_audio_cli.py --asr audio.wav
    python test_audio_cli.py --pipeline "Test message"
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import httpx


API_BASE_URL = "http://127.0.0.1:8080"
TIMEOUT = 30.0


def print_header(text: str) -> None:
    """Print a formatted header."""
    print(f"\n{'=' * 60}")
    print(f"  {text}")
    print(f"{'=' * 60}\n")


def print_success(text: str) -> None:
    """Print success message."""
    print(f"✓ {text}")


def print_error(text: str) -> None:
    """Print error message."""
    print(f"✗ {text}", file=sys.stderr)


def check_health() -> bool:
    """Check if API is healthy."""
    try:
        response = httpx.get(f"{API_BASE_URL}/health", timeout=TIMEOUT)
        if response.status_code == 200:
            data = response.json()
            status = data.get("status", "unknown")
            print_success(f"API is {status}")
            return True
        else:
            print_error(f"API returned status {response.status_code}")
            return False
    except Exception as e:
        print_error(f"Failed to connect to API: {e}")
        return False


def test_tts(text: str, output_file: str | None = None) -> bool:
    """Test TTS service."""
    print_header("Testing TTS (Text-to-Speech)")
    print(f"Input text: {text}\n")

    try:
        response = httpx.post(
            f"{API_BASE_URL}/v1/audio/speech",
            json={
                "model": "Qwen3-TTS-0.6B",
                "input": text,
                "voice": "default",
            },
            timeout=TIMEOUT,
        )

        if response.status_code != 200:
            print_error(f"TTS failed with status {response.status_code}")
            print(f"Response: {response.text}")
            return False

        audio_data = response.content
        print_success(f"Generated audio: {len(audio_data)} bytes")

        if output_file:
            Path(output_file).write_bytes(audio_data)
            print_success(f"Saved to: {output_file}")

        return True
    except Exception as e:
        print_error(f"TTS request failed: {e}")
        return False


def test_asr(audio_file: str) -> bool:
    """Test ASR service."""
    print_header("Testing ASR (Speech-to-Text)")
    print(f"Input audio: {audio_file}\n")

    try:
        audio_path = Path(audio_file)
        if not audio_path.exists():
            print_error(f"Audio file not found: {audio_file}")
            return False

        audio_data = audio_path.read_bytes()
        print_success(f"Loaded audio: {len(audio_data)} bytes")

        response = httpx.post(
            f"{API_BASE_URL}/v1/audio/transcriptions",
            files={"file": (audio_path.name, audio_data, "audio/wav")},
            data={"model": "Qwen3-ASR-0.6B"},
            timeout=TIMEOUT,
        )

        if response.status_code != 200:
            print_error(f"ASR failed with status {response.status_code}")
            print(f"Response: {response.text}")
            return False

        result = response.json()
        transcribed_text = result.get("text", "")
        print_success(f"Transcribed text: {transcribed_text}")

        return True
    except Exception as e:
        print_error(f"ASR request failed: {e}")
        return False


def test_pipeline(text: str) -> bool:
    """Test end-to-end pipeline: TTS -> ASR."""
    print_header("Testing Pipeline (TTS -> ASR)")
    print(f"Original text: {text}\n")

    try:
        # Step 1: TTS
        print("Step 1: Synthesizing speech...")
        tts_response = httpx.post(
            f"{API_BASE_URL}/v1/audio/speech",
            json={
                "model": "Qwen3-TTS-0.6B",
                "input": text,
                "voice": "default",
            },
            timeout=TIMEOUT,
        )

        if tts_response.status_code != 200:
            print_error(f"TTS failed: {tts_response.status_code}")
            return False

        audio_data = tts_response.content
        print_success(f"Generated audio: {len(audio_data)} bytes")

        # Step 2: ASR
        print("\nStep 2: Transcribing audio...")
        asr_response = httpx.post(
            f"{API_BASE_URL}/v1/audio/transcriptions",
            files={"file": ("generated.wav", audio_data, "audio/wav")},
            data={"model": "Qwen3-ASR-0.6B"},
            timeout=TIMEOUT,
        )

        if asr_response.status_code != 200:
            print_error(f"ASR failed: {asr_response.status_code}")
            return False

        result = asr_response.json()
        transcribed_text = result.get("text", "")
        print_success(f"Transcribed text: {transcribed_text}")

        # Compare
        print(f"\nComparison:")
        print(f"  Original:    {text}")
        print(f"  Transcribed: {transcribed_text}")

        return True
    except Exception as e:
        print_error(f"Pipeline test failed: {e}")
        return False


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Test ASR and TTS services",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check API health
  python test_audio_cli.py --health

  # Test TTS
  python test_audio_cli.py --tts "Hello world" --output output.wav

  # Test ASR
  python test_audio_cli.py --asr input.wav

  # Test end-to-end pipeline
  python test_audio_cli.py --pipeline "Test message"
        """,
    )

    parser.add_argument("--health", action="store_true", help="Check API health")
    parser.add_argument("--tts", metavar="TEXT", help="Test TTS with given text")
    parser.add_argument("--asr", metavar="FILE", help="Test ASR with audio file")
    parser.add_argument("--pipeline", metavar="TEXT", help="Test TTS->ASR pipeline")
    parser.add_argument("--output", metavar="FILE", help="Output file for TTS audio")
    parser.add_argument("--api-url", default=API_BASE_URL, help="API base URL")

    args = parser.parse_args()

    if not args.health and not args.tts and not args.asr and not args.pipeline:
        parser.print_help()
        return 1

    # Check health first
    if not check_health():
        print_error("API is not responding. Make sure the server is running:")
        print_error("  python -m aster --config configs/config.yaml")
        return 1

    success = True

    if args.health:
        pass  # Already checked above

    if args.tts:
        output = args.output or "output.wav"
        success = test_tts(args.tts, output) and success

    if args.asr:
        success = test_asr(args.asr) and success

    if args.pipeline:
        success = test_pipeline(args.pipeline) and success

    print_header("Test Summary")
    if success:
        print_success("All tests passed!")
        return 0
    else:
        print_error("Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
