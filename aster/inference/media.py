from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Literal

MediaKind = Literal["image", "audio", "video"]
MediaSource = Literal["url", "base64", "bytes", "path"]


@dataclass(frozen=True, slots=True)
class MediaRef:
    kind: MediaKind
    source: MediaSource
    value: str | bytes
    mime_type: str | None = None

    @property
    def fingerprint(self) -> str:
        hasher = hashlib.sha256()
        hasher.update(self.kind.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(self.source.encode("utf-8"))
        hasher.update(b"\0")
        if self.mime_type:
            hasher.update(self.mime_type.encode("utf-8"))
        hasher.update(b"\0")
        value = self.value if isinstance(self.value, bytes) else self.value.encode("utf-8")
        hasher.update(value)
        return hasher.hexdigest()


@dataclass(frozen=True, slots=True)
class MessagePart:
    type: Literal["text", "media"]
    text: str = ""
    media: MediaRef | None = None


@dataclass(frozen=True, slots=True)
class MediaProcessorResult:
    text_tokens: tuple[int, ...] = ()
    media_refs: tuple[MediaRef, ...] = ()
    processed_payload: object | None = None
