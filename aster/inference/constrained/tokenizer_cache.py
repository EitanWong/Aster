from __future__ import annotations

import functools
import threading
from typing import Any

from aster.telemetry.logging import get_logger

_logger = get_logger(__name__)
_CACHE: dict[int, Any] = {}
_CACHE_LOCK = threading.Lock()


def get_tokenizer_data(tokenizer: Any) -> Any | None:
    try:
        from lmformatenforcer.tokenenforcer import TokenEnforcerTokenizerData
    except ImportError:
        return None

    inner = _resolve_inner_tokenizer(tokenizer)
    key = id(inner)
    with _CACHE_LOCK:
        cached = _CACHE.get(key)
        if cached is not None:
            return cached

    try:
        vocab_size = _get_vocab_size(inner)
        regular_tokens = _regular_tokens(inner, vocab_size)
        data = TokenEnforcerTokenizerData(
            regular_tokens,
            functools.partial(_decode_tokens, inner),
            _eos_token_id(inner),
            use_bitmask=False,
            vocab_size=vocab_size,
        )
    except Exception as exc:
        _logger.warning("constrained_tokenizer_data_build_failed", extra={"error": str(exc)})
        return None

    with _CACHE_LOCK:
        _CACHE[key] = data
    return data


def clear_tokenizer_cache() -> None:
    with _CACHE_LOCK:
        _CACHE.clear()


def _resolve_inner_tokenizer(tokenizer: Any) -> Any:
    for attr in ("tokenizer", "_tokenizer"):
        inner = getattr(tokenizer, attr, None)
        if inner is not None and inner is not tokenizer and hasattr(inner, "all_special_ids"):
            tokenizer = inner
    return tokenizer


def _regular_tokens(tokenizer: Any, vocab_size: int) -> list[tuple[int, str, bool]]:
    try:
        special_ids = set(tokenizer.all_special_ids)
    except AttributeError:
        special_ids = set()
    try:
        zero_token = tokenizer.encode("0")[-1]
    except Exception:
        zero_token = None

    tokens: list[tuple[int, str, bool]] = []
    for token_id in range(vocab_size):
        if token_id in special_ids:
            continue
        try:
            decoded = tokenizer.decode([token_id])
        except Exception:
            continue
        if not isinstance(decoded, str):
            continue
        if zero_token is None:
            decoded_after_zero = decoded
        else:
            try:
                decoded_after_zero = tokenizer.decode([zero_token, token_id])[1:]
            except Exception:
                decoded_after_zero = decoded
        tokens.append((token_id, decoded_after_zero, len(decoded_after_zero) > len(decoded)))
    return tokens


def _eos_token_id(tokenizer: Any) -> int | list[int]:
    eos_ids = getattr(tokenizer, "eos_token_ids", None)
    if isinstance(eos_ids, (list, tuple, set)) and eos_ids:
        return [int(token_id) for token_id in eos_ids]
    eos_id = getattr(tokenizer, "eos_token_id", None)
    return int(eos_id) if eos_id is not None else 0


def _get_vocab_size(tokenizer: Any) -> int:
    vocab_size = getattr(tokenizer, "vocab_size", None)
    if isinstance(vocab_size, int) and vocab_size > 0:
        return vocab_size
    try:
        return len(tokenizer)
    except TypeError:
        pass
    get_vocab = getattr(tokenizer, "get_vocab", None)
    if callable(get_vocab):
        return len(get_vocab())
    raise ValueError("Cannot determine tokenizer vocab size")


def _decode_tokens(tokenizer: Any, tokens: list[int]) -> str:
    try:
        decoded = tokenizer.decode(tokens)
    except Exception:
        return ""
    return decoded.rstrip("\ufffd") if isinstance(decoded, str) else ""
