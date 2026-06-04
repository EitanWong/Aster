from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_warmup_file(path: str | Path) -> list[list[dict[str, Any]]]:
    source = Path(path).expanduser()
    data = json.loads(source.read_text())
    if not isinstance(data, list):
        raise ValueError(f"Warm prompt file must contain a JSON list: {source}")
    prompts: list[list[dict[str, Any]]] = []
    for entry_index, entry in enumerate(data):
        if not isinstance(entry, list) or not entry:
            raise ValueError(f"Warm prompt entry {entry_index} must be a non-empty message list")
        messages: list[dict[str, Any]] = []
        for message_index, message in enumerate(entry):
            if not isinstance(message, dict):
                raise ValueError(
                    f"Warm prompt entry {entry_index} message {message_index} must be an object"
                )
            if "role" not in message or "content" not in message:
                raise ValueError(
                    f"Warm prompt entry {entry_index} message {message_index} misses role/content"
                )
            messages.append(dict(message))
        prompts.append(messages)
    return prompts


def ensure_user_terminator(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if messages and messages[-1].get("role") in {"user", "assistant"}:
        return [dict(message) for message in messages]
    return [*(dict(message) for message in messages), {"role": "user", "content": " "}]


def build_strict_prefix_string(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    *,
    enable_thinking: bool,
    chat_template_kwargs: dict[str, Any] | None = None,
) -> str | None:
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if apply_chat_template is None:
        return None

    def with_user(content: str) -> list[dict[str, Any]]:
        patched = [dict(message) for message in messages]
        if patched and patched[-1].get("role") == "user":
            patched[-1] = {**patched[-1], "content": content}
        else:
            patched.append({"role": "user", "content": content})
        return patched

    kwargs = dict(chat_template_kwargs or {})
    kwargs.update(
        {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": enable_thinking,
        }
    )
    try:
        left = apply_chat_template(with_user("Alpha"), **kwargs)
        right = apply_chat_template(with_user("Bravo"), **kwargs)
    except Exception:
        kwargs.pop("enable_thinking", None)
        try:
            left = apply_chat_template(with_user("Alpha"), **kwargs)
            right = apply_chat_template(with_user("Bravo"), **kwargs)
        except Exception:
            return None

    if not isinstance(left, str) or not isinstance(right, str):
        return None
    boundary = 0
    diverged = False
    for index in range(min(len(left), len(right))):
        if left[index] != right[index]:
            diverged = True
            break
        boundary = index + 1
    if not diverged or boundary < 16:
        return None
    return left[:boundary]
