from __future__ import annotations

MULTIMODAL_CONTENT_TYPES = frozenset(
    {
        "audio",
        "audio_url",
        "file",
        "image",
        "image_url",
        "input_audio",
        "input_file",
        "input_image",
        "video",
        "video_url",
    }
)


def is_multimodal_content_type(part_type: str | None) -> bool:
    return part_type in MULTIMODAL_CONTENT_TYPES
