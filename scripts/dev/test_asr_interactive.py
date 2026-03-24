#!/usr/bin/env python3
"""Interactive ASR testing script with audio recording.

Usage:
    # Record and transcribe
    python test_asr_interactive.py --record 5 --transcribe
    
    # Transcribe existing file
    python test_asr_interactive.py --file audio.wav
    
    # List available audio devices
    python test_asr_interactive.py --list-devices
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import httpx


API_BASE_URL = "http://127.0.0.1:8080"
TIMEOUT = 60.0


def print_header(text: str) -> None:
    """Print formatted header."""
    print(f"\n{'=' * 60}")
    print(f"  {text}")
    print(f"{'=' * 60}\n")


def print_success(text: str) -> None:
    """Print success message."""
    print(f"✓ {text}")


def print_error(text: str) -> None:
    """Print error message."""
    print(f"✗ {text}", file=sys.stderr)


def record_audio(duration: int, output_file: str = "recorded_audio.wav") -> bool:
    """Record audio from microphone."""
    print_header(f"Recording Audio ({duration}s)")
    
    try:
        import sounddevice as sd
        import soundfile as sf
        import numpy as np
        
        print(f"Recording for {duration} seconds...")
        print("Speak now...")
        
        # Record audio
        sample_rate = 16000
        audio_data = sd.rec(int(sample_rate * duration), samplerate=sample_rate, channels=1, dtype=np.int16)
        sd.wait()
        
        # Save to file
        sf.write(output_file, audio_data, sample_rate)
        print_success(f"Audio saved to: {output_file}")
        
        return True
    except ImportError:
        print_error("sounddevice or soundfile not installed")
        print("Install with: pip install sounddevice soundfile")
        return False
    except Exception as e:
        print_error(f"Recording failed: {e}")
        return False


def list_audio_devices() -> None:
    """List available audio devices."""
    print_header("Available Audio Devices")
    
    try:
        import sounddevice as sd
        
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            print(f"{i}: {device['name']}")
            print(f"   Channels: {device['max_input_channels']} in, {device['max_output_channels']} out")
            if device['default_samplerate']:
                print(f"   Sample rate: {device['default_samplerate']}")
            print()
    except ImportError:
        print_error("sounddevice not installed")
        print("Install with: pip install sounddevice")
    except Exception as e:
        print_error(f"Failed to list devices: {e}")


def transcribe_audio(audio_file: str, language: str | None = None) -> bool:
    """Transcribe audio file using ASR API."""
    print_header("Transcribing Audio")
    
    audio_path = Path(audio_file)
    if not audio_path.exists():
        print_error(f"Audio file not found: {audio_file}")
        return False
    
    print(f"File: {audio_file}")
    print(f"Size: {audio_path.stat().st_size / 1024:.1f} KB")
    
    try:
        audio_data = audio_path.read_bytes()
        
        print("\nSending to ASR API...")
        response = httpx.post(
            f"{API_BASE_URL}/v1/audio/transcriptions",
            files={"file": (audio_path.name, audio_data, "audio/wav")},
            data={"model": "Qwen3-ASR-0.6B", **({"language": language} if language else {})},
            timeout=TIMEOUT,
        )
        
        if response.status_code != 200:
            print_error(f"ASR failed with status {response.status_code}")
            print(f"Response: {response.text}")
            return False
        
        result = response.json()
        transcribed_text = result.get("text", "")
        
        print_success(f"Transcription complete!")
        print(f"\nTranscribed text:")
        print(f"  {transcribed_text}")
        
        if result.get("duration"):
            print(f"\nDuration: {result['duration']:.2f}s")
        if result.get("language"):
            print(f"Language: {result['language']}")
        
        return True
    except Exception as e:
        print_error(f"Transcription failed: {e}")
        return False


def check_api_health() -> bool:
    """Check if API is running."""
    try:
        response = httpx.get(f"{API_BASE_URL}/health", timeout=5)
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
        print_error("Make sure the server is running:")
        print_error("  python -m aster --config configs/config.yaml")
        return False


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Interactive ASR testing with audio recording",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Record 5 seconds and transcribe
  python test_asr_interactive.py --record 5 --transcribe
  
  # Transcribe existing file
  python test_asr_interactive.py --file audio.wav
  
  # List audio devices
  python test_asr_interactive.py --list-devices
  
  # Record with custom output file
  python test_asr_interactive.py --record 10 --output my_audio.wav --transcribe
        """,
    )
    
    parser.add_argument("--record", type=int, metavar="SECONDS", help="Record audio for N seconds")
    parser.add_argument("--file", metavar="FILE", help="Transcribe audio file")
    parser.add_argument("--output", default="recorded_audio.wav", help="Output file for recording")
    parser.add_argument("--transcribe", action="store_true", help="Transcribe after recording")
    parser.add_argument("--language", help="Language hint (e.g., en, zh, ja)")
    parser.add_argument("--list-devices", action="store_true", help="List audio devices")
    parser.add_argument("--api-url", default=API_BASE_URL, help="API base URL")
    
    args = parser.parse_args()
    
    if not args.record and not args.file and not args.list_devices:
        parser.print_help()
        return 1
    
    # Check API health first
    if args.record or args.file or args.transcribe:
        if not check_api_health():
            return 1
    
    if args.list_devices:
        list_audio_devices()
        return 0
    
    if args.record:
        if not record_audio(args.record, args.output):
            return 1
        
        if args.transcribe:
            if not transcribe_audio(args.output, args.language):
                return 1
    
    if args.file:
        if not transcribe_audio(args.file, args.language):
            return 1
    
    print_header("Test Complete")
    print_success("ASR test completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
