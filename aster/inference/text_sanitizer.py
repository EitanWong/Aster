from __future__ import annotations

import re

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_LEADING_ASSISTANT_RE = re.compile(r"^(assistant\s*:\s*)", re.IGNORECASE)


def sanitize_assistant_text(text: str) -> str:
    cleaned = _THINK_BLOCK_RE.sub("", text)
    cleaned = _LEADING_ASSISTANT_RE.sub("", cleaned.strip())
    cleaned = cleaned.replace("<|im_end|>", "").replace("<|im_start|>", "")
    return cleaned.strip()
