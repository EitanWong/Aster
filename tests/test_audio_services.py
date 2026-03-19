"""Integration tests for ASR and TTS services."""

from __future__ import annotations

import asyncio
import io
import sys
from pathlib import Path

import httpx
import pytest
from pydub import AudioSegment


# Test configuration
API_BASE_URL = "http://127.0.0.1:8080"
TIMEOUT = 30.0


def generate_test_audio(duration_ms: int = 2000, sample_rate: int = 16000) -> bytes:
    """Generate a simple test audio file (sine wave)."""
    import numpy as np

    # Generate sine wave
    t = np.linspace(0, duration_ms / 1000, int(sample_rate * duration_ms / 1000))
    frequency = 440  # A4 note
    audio_data = np.sin(2 * np.pi * frequency * t) * 0.3

    # Convert to 16-bit PCM
    audio_data = (audio_data * 32767).astype(np.int16)

    # Create WAV file
    audio = AudioSegment(
        audio_data.tobytes(),
        frame_rate=sample_rate,
        sample_width=2,
        channels=1,
    )

    # Export to bytes
    wav_buffer = io.BytesIO()
    audio.export(wav_buffer, format="wav")
    return wav_buffer.getvalue()


class TestASRService:
    """Test ASR (speech-to-text) service."""

    @pytest.mark.asyncio
    async def test_transcribe_audio(self) -> None:
        """Test basic audio transcription."""
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            # Generate test audio
            audio_data = generate_test_audio(duration_ms=3000)

            # Send transcription request
            response = await client.post(
                f"{API_BASE_URL}/v1/audio/transcriptions",
                files={"file": ("test.wav", audio_data, "audio/wav")},
                data={"model": "Qwen3-ASR-0.6B"},
            )

            assert response.status_code == 200, f"Error: {response.text}"
            result = response.json()
            assert "text" in result
            print(f"✓ Transcription result: {result['text']}")

    @pytest.mark.asyncio
    async def test_transcribe_with_language(self) -> None:
        """Test transcription with language hint."""
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            audio_data = generate_test_audio()

            response = await client.post(
                f"{API_BASE_URL}/v1/audio/transcriptions",
                files={"file": ("test.wav", audio_data, "audio/wav")},
                data={
                    "model": "Qwen3-ASR-0.6B",
                    "language": "en",
                },
            )

            assert response.status_code == 200
            result = response.json()
            assert "text" in result
            print(f"✓ Transcription with language: {result['text']}")


class TestTTSService:
    """Test TTS (text-to-speech) service."""

    @pytest.mark.asyncio
    async def test_synthesize_speech(self) -> None:
        """Test basic text-to-speech synthesis."""
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            response = await client.post(
                f"{API_BASE_URL}/v1/audio/speech",
                json={
                    "model": "Qwen3-TTS-0.6B",
                    "input": "Hello, this is a test of the text to speech system.",
                    "voice": "default",
                },
            )

            assert response.status_code == 200, f"Error: {response.text}"
            assert response.headers["content-type"] == "audio/wav"
            assert len(response.content) > 0
            print(f"✓ Generated audio: {len(response.content)} bytes")

    @pytest.mark.asyncio
    async def test_synthesize_with_speed(self) -> None:
        """Test TTS with custom speed."""
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            response = await client.post(
                f"{API_BASE_URL}/v1/audio/speech",
                json={
                    "model": "Qwen3-TTS-0.6B",
                    "input": "Speaking at a faster pace.",
                    "voice": "default",
                    "speed": 1.5,
                },
            )

            assert response.status_code == 200
            assert response.headers["content-type"] == "audio/wav"
            print(f"✓ Generated audio with speed: {len(response.content)} bytes")

    @pytest.mark.asyncio
    async def test_synthesize_with_reference_audio(self) -> None:
        """Test TTS with voice cloning (reference audio)."""
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            # Generate reference audio
            reference_audio = generate_test_audio(duration_ms=3000)

            response = await client.post(
                f"{API_BASE_URL}/v1/audio/speech",
                json={
                    "model": "Qwen3-TTS-0.6B",
                    "input": "This should sound like the reference voice.",
                    "voice": "cloned",
                    "reference_audio": reference_audio.hex(),  # Send as hex string
                },
            )

            assert response.status_code == 200
            assert response.headers["content-type"] == "audio/wav"
            print(f"✓ Generated cloned voice audio: {len(response.content)} bytes")


class TestAudioPipeline:
    """Test end-to-end audio pipeline (TTS -> ASR)."""

    @pytest.mark.asyncio
    async def test_tts_to_asr_pipeline(self) -> None:
        """Test synthesizing speech and then transcribing it."""
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            original_text = "Hello world, this is a test message."

            # Step 1: Synthesize speech
            tts_response = await client.post(
                f"{API_BASE_URL}/v1/audio/speech",
                json={
                    "model": "Qwen3-TTS-0.6B",
                    "input": original_text,
                    "voice": "default",
                },
            )
            assert tts_response.status_code == 200
            audio_data = tts_response.content

            # Step 2: Transcribe the generated audio
            asr_response = await client.post(
                f"{API_BASE_URL}/v1/audio/transcriptions",
                files={"file": ("generated.wav", audio_data, "audio/wav")},
                data={"model": "Qwen3-ASR-0.6B"},
            )
            assert asr_response.status_code == 200
            result = asr_response.json()

            print(f"✓ Original text: {original_text}")
            print(f"✓ Transcribed text: {result['text']}")
            print(f"✓ Pipeline test passed!")


class TestHealthAndMetrics:
    """Test health checks and metrics."""

    @pytest.mark.asyncio
    async def test_health_check(self) -> None:
        """Test health endpoint."""
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            response = await client.get(f"{API_BASE_URL}/health")
            assert response.status_code == 200
            data = response.json()
            assert "status" in data
            print(f"✓ Health status: {data['status']}")

    @pytest.mark.asyncio
    async def test_ready_check(self) -> None:
        """Test readiness endpoint."""
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            response = await client.get(f"{API_BASE_URL}/ready")
            assert response.status_code == 200
            data = response.json()
            assert "status" in data
            print(f"✓ Ready status: {data['status']}")

    @pytest.mark.asyncio
    async def test_metrics(self) -> None:
        """Test metrics endpoint."""
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            response = await client.get(f"{API_BASE_URL}/metrics")
            assert response.status_code == 200
            assert b"# HELP" in response.content or b"# TYPE" in response.content
            print(f"✓ Metrics available: {len(response.content)} bytes")


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s"])
