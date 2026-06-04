from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from aster.audio.service import ASRResult, TTSResult
from aster.core.lifecycle import create_application


class FakeASR:
    async def transcribe(
        self,
        audio: bytes,
        language: str | None = None,
        prompt: str | None = None,
    ) -> ASRResult:
        del audio, language, prompt
        return ASRResult(text="ok")

    def health(self) -> bool:
        return True


class FakeTTS:
    async def synthesize(
        self,
        text: str,
        voice: str = "default",
        language: str | None = None,
        speed: float = 1.0,
        reference_audio: bytes | None = None,
        speaker: str | None = None,
        instruct: str | None = None,
    ) -> TTSResult:
        del text, voice, language, speed, reference_audio, speaker, instruct
        return TTSResult(audio=b"RIFF")

    def health(self) -> bool:
        return True


def _client(tmp_path: Path, *, max_audio_upload_mb: int = 25, max_tts_input_chars: int = 4096) -> TestClient:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
embeddings:
  enabled: false
audio:
  max_audio_upload_mb: {max_audio_upload_mb}
  max_tts_input_chars: {max_tts_input_chars}
  asr:
    enabled: false
  tts:
    enabled: false
"""
    )
    return TestClient(create_application(str(config_path)))


def test_audio_transcription_rejects_oversized_upload(tmp_path: Path) -> None:
    with _client(tmp_path, max_audio_upload_mb=0) as client:
        client.app.state.container.audio.asr = FakeASR()

        response = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("sample.wav", b"too-large", "audio/wav")},
        )

    assert response.status_code == 413


def test_tts_rejects_oversized_input(tmp_path: Path) -> None:
    with _client(tmp_path, max_tts_input_chars=3) as client:
        client.app.state.container.audio.tts = FakeTTS()

        response = client.post(
            "/v1/audio/speech",
            json={"model": "tts", "input": "four", "voice": "default"},
        )

    assert response.status_code == 413


def test_audio_transcription_rejects_unconfigured_request_model(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        client.app.state.container.audio.asr = FakeASR()

        response = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("sample.wav", b"RIFF", "audio/wav")},
            data={"model": "unknown-asr"},
        )

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "audio_model_not_available"


def test_tts_rejects_unconfigured_request_model(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        client.app.state.container.audio.tts = FakeTTS()

        response = client.post(
            "/v1/audio/speech",
            json={"model": "unknown-tts", "input": "hello", "voice": "default"},
        )

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "audio_model_not_available"
