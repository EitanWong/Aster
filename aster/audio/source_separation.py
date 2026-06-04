from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class SeparationResult:
    target_audio: Any
    residual_audio: Any
    sample_rate: int | None = None
    peak_memory_gb: float = 0.0


class SourceSeparationService:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model: Any | None = None
        self._processor: Any | None = None
        self._sample_rate = 44100

    async def separate(
        self,
        audio_path: str | Path,
        *,
        description: str = "speech",
        chunk_seconds: float | None = None,
    ) -> SeparationResult:
        return await asyncio.to_thread(
            self._separate_sync,
            Path(audio_path),
            description,
            chunk_seconds,
        )

    def _separate_sync(
        self,
        audio_path: Path,
        description: str,
        chunk_seconds: float | None,
    ) -> SeparationResult:
        model = self._ensure_model()
        processor = self._ensure_processor()

        batch = processor(
            descriptions=[description],
            audios=[str(audio_path)],
        )
        if chunk_seconds is not None and hasattr(model, "separate_long"):
            output = model.separate_long(
                audios=batch.audios,
                descriptions=batch.descriptions,
                chunk_seconds=chunk_seconds,
                overlap_seconds=chunk_seconds / 3,
                anchor_ids=getattr(batch, "anchor_ids", None),
                anchor_alignment=getattr(batch, "anchor_alignment", None),
            )
        else:
            output = model.separate(
                audios=batch.audios,
                descriptions=batch.descriptions,
                sizes=getattr(batch, "sizes", None),
                anchor_ids=getattr(batch, "anchor_ids", None),
                anchor_alignment=getattr(batch, "anchor_alignment", None),
            )
        return SeparationResult(
            target_audio=_to_numpy(output.target[0]),
            residual_audio=_to_numpy(output.residual[0]),
            sample_rate=getattr(output, "sample_rate", None) or self._sample_rate,
            peak_memory_gb=float(getattr(output, "peak_memory", 0.0) or 0.0),
        )

    def _ensure_model(self) -> Any:
        if self._model is None:
            try:
                from mlx_audio.sts import SAMAudio
            except Exception as exc:  # pragma: no cover - optional dependency
                raise RuntimeError("mlx-audio is required for source separation.") from exc
            self._model = SAMAudio.from_pretrained(self.model_name)
            if hasattr(self._model, "sample_rate"):
                self._sample_rate = int(self._model.sample_rate)
        return self._model

    def _ensure_processor(self) -> Any:
        if self._processor is None:
            try:
                from mlx_audio.sts import SAMAudioProcessor
            except Exception as exc:  # pragma: no cover - optional dependency
                raise RuntimeError("mlx-audio is required for source separation.") from exc
            self._processor = SAMAudioProcessor.from_pretrained(self.model_name)
        return self._processor


def _to_numpy(audio: Any) -> Any:
    try:
        import numpy as np
    except Exception:  # pragma: no cover - optional dependency
        return audio.tolist() if hasattr(audio, "tolist") else audio
    if hasattr(audio, "tolist"):
        return np.array(audio.tolist(), dtype=np.float32)
    return np.array(audio, dtype=np.float32)
