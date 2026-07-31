from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

import numpy as np

from aster.inference.constrained.tokenizer_cache import clear_tokenizer_cache, get_tokenizer_data
from aster.inference.structured_schema import GENERIC_JSON_SCHEMA, normalize_json_schema
from aster.telemetry.logging import get_logger

_logger = get_logger(__name__)
_PARSER_CACHE: dict[str, tuple[dict[str, Any], Any]] = {}


class LMFormatEnforcerNotAvailableError(RuntimeError):
    pass


def is_available() -> bool:
    try:
        import lmformatenforcer  # noqa: F401
    except ImportError:
        return False
    return True


def build_json_logits_processor(schema: dict[str, Any] | None, tokenizer: Any) -> JSONSchemaLogitsProcessor | None:
    if not is_available():
        return None
    try:
        return JSONSchemaLogitsProcessor(schema=schema, tokenizer=tokenizer)
    except LMFormatEnforcerNotAvailableError:
        return None
    except Exception as exc:
        _logger.warning("json_logits_processor_build_failed", extra={"error": str(exc)})
        return None


def clear_constrained_caches() -> None:
    _PARSER_CACHE.clear()
    clear_tokenizer_cache()


def _new_token_enforcer(
    tokenizer_data: Any,
    parser: Any,
    *,
    reuse_freetext_token_lists: bool,
) -> Any:
    from lmformatenforcer import TokenEnforcer

    if not reuse_freetext_token_lists:
        return TokenEnforcer(tokenizer_data, parser)

    from lmformatenforcer.exceptions import LMFormatEnforcerException
    from lmformatenforcer.tokenlist import TokenList

    class ReusingFreetextTokenEnforcer(TokenEnforcer):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self._aster_working_freetext_lists: dict[int, tuple[Any, int]] = {}
            self._aster_working_freetext_ids: set[int] = set()

        def _aster_prepare_for_next_state(self) -> None:
            for token_list, static_length in self._aster_working_freetext_lists.values():
                del token_list.allowed_tokens[static_length:]

        def _working_freetext_list(self, static_tokens: Any) -> Any:
            key = id(static_tokens.allowed_tokens)
            cached = self._aster_working_freetext_lists.get(key)
            if cached is not None:
                return cached[0]
            working = TokenList(self.use_bitmask, self.vocab_size)
            working.extend(static_tokens.allowed_tokens)
            self._aster_working_freetext_lists[key] = (working, len(working.allowed_tokens))
            self._aster_working_freetext_ids.add(id(working))
            return working

        def _compute_allowed_tokens(self, state_tokens: tuple[int, ...], state: Any) -> None:
            try:
                allowed_tokens = TokenList(self.use_bitmask, self.vocab_size)
                cache_key = state.parser.cache_key()
                if cache_key is not None and cache_key in self.allowed_token_cache:
                    state.allowed_tokens = self.allowed_token_cache[cache_key]
                    return
                shortcut_key = state.parser.shortcut_key()
                allowed_tokens = self._collect_allowed_tokens(
                    state.parser,
                    self.tokenizer_tree.root,
                    allowed_tokens,
                    shortcut_key,
                )
                if state.parser.can_end():
                    if isinstance(self.eos_token_id, list):
                        allowed_tokens.extend(self.eos_token_id)
                    else:
                        allowed_tokens.append(self.eos_token_id)
                if not allowed_tokens:
                    raise ValueError("Parser reached state with no allowed tokens")
                state.allowed_tokens = allowed_tokens
                if cache_key is not None and id(allowed_tokens) not in self._aster_working_freetext_ids:
                    self.allowed_token_cache[cache_key] = allowed_tokens
            except LMFormatEnforcerException:
                raise
            except Exception:
                logging.basicConfig(level=logging.ERROR)
                prefix = self.decoder(list(state_tokens))
                logging.exception("Unknown LMFormatEnforcer Problem. Prefix: '%s'", prefix)
                state.allowed_tokens = TokenList(self.use_bitmask, self.vocab_size)
                if isinstance(self.eos_token_id, list):
                    state.allowed_tokens.extend(self.eos_token_id)
                else:
                    state.allowed_tokens.append(self.eos_token_id)

        def _collect_allowed_tokens(
            self,
            parser: Any,
            tree_node: Any,
            allowed_tokens: Any,
            shortcut_key: Any,
        ) -> Any:
            allowed_characters = parser.get_allowed_characters()
            characters_to_explore = set(tree_node.children.keys()).intersection(allowed_characters)
            if isinstance(shortcut_key, tuple) and shortcut_key[0] == "json_freetext":
                _, cur_len, min_len, max_len = shortcut_key
                cache = self.tokenizer_tree.json_freetext_tokens
                min_remaining = min(cache.max_token_len, max(0, min_len - cur_len))
                max_allowed_len = min(cache.max_token_len, max_len - cur_len)
                static_tokens = cache.lookup_allowed_tokens(
                    min_remaining,
                    max_allowed_len,
                )
                if self.use_bitmask:
                    allowed_tokens.extend(static_tokens.allowed_tokens)
                else:
                    allowed_tokens = self._working_freetext_list(static_tokens)
                allowed_tokens.extend(tree_node.tokens)
                characters_to_explore = characters_to_explore.intersection(['"'])
            else:
                allowed_tokens.extend(tree_node.tokens)
            for character in characters_to_explore:
                allowed_tokens = self._collect_allowed_tokens(
                    parser.add_character(character),
                    tree_node.children[character],
                    allowed_tokens,
                    None,
                )
            return allowed_tokens

    return ReusingFreetextTokenEnforcer(tokenizer_data, parser)


class JSONSchemaLogitsProcessor:
    name = "json_schema"
    batch_sampling_mode = "eager_rows"
    _DECODE_PREFIX_STATE_WINDOW = 1

    def __init__(self, *, schema: dict[str, Any] | None, tokenizer: Any) -> None:
        if not is_available():
            raise LMFormatEnforcerNotAvailableError("lm-format-enforcer is not installed")

        tokenizer_data = get_tokenizer_data(tokenizer)
        if tokenizer_data is None:
            raise LMFormatEnforcerNotAvailableError("Could not adapt tokenizer for constrained decoding")

        parser_schema, parser = _get_or_build_parser(schema)
        self.schema = parser_schema
        self._tokenizer = tokenizer
        self._tokenizer_data = tokenizer_data
        self._reuse_freetext_token_lists = True
        self._enforcer = _new_token_enforcer(
            tokenizer_data,
            parser,
            reuse_freetext_token_lists=self._reuse_freetext_token_lists,
        )
        self._bounded_prefix_states = True
        self._enforcer_last_suffix: tuple[int, ...] | None = ()
        self._decode_step_hint: tuple[int, int] | None = None
        self._prompt_len: int | None = None
        self._disabled = False
        self._eos_token_ids = _eos_token_ids(tokenizer_data)
        self._decode_cache: dict[tuple[int, ...], str] = {}
        self._token_decode_cache: dict[int, str | None] = {}
        self._mask_cache_key: tuple[object, ...] | None = None
        self._mask_cache_allowed: list[int] | None = None
        self._mask_cache_value: Any | None = None
        self._mask_cache_contains_eos: bool | None = None
        self._pending_mask_allowed: list[int] | None = None
        self._pending_mask_contains_eos = False
        self._valid_key_names = _collect_property_names(parser_schema)
        self._valid_key_first_chars = {name[0] for name in self._valid_key_names if name}

        try:
            self._enforcer.get_allowed_tokens([])
        except Exception as exc:
            _logger.warning("json_logits_processor_bootstrap_failed", extra={"error": str(exc)})
            self._disabled = True

    def __call__(self, tokens: Any, logits: Any) -> Any:
        if self._disabled:
            if self._eos_token_ids:
                return logits + self._mask(sorted(self._eos_token_ids), logits)
            return logits
        try:
            token_ids = _token_list(tokens)
            suffix = self._generated_suffix(token_ids)
            allowed = self._allowed_tokens(suffix)
            if not allowed:
                if self._is_complete_json(suffix) and self._eos_token_ids:
                    allowed = sorted(self._eos_token_ids)
                else:
                    self._disabled = True
                    return logits
            return logits + self._mask(allowed, logits)
        except Exception as exc:
            _logger.warning("json_logits_processor_failed", extra={"error": str(exc)})
            self._disabled = True
            return logits

    def _generated_suffix(self, token_ids: list[int]) -> list[int]:
        if self._prompt_len is None:
            self._prompt_len = len(token_ids)
        return token_ids[self._prompt_len :]

    def _prepare_aster_decode_step(self, *, input_token: int, completion_tokens: int) -> None:
        self._decode_step_hint = (int(input_token), int(completion_tokens))

    def _enforcer_allowed_tokens(self, suffix: list[int]) -> Any:
        if not getattr(self, "_bounded_prefix_states", False):
            return self._enforcer.get_allowed_tokens(suffix)

        decode_step_hint = getattr(self, "_decode_step_hint", None)
        self._decode_step_hint = None
        if decode_step_hint is not None:
            input_token, completion_tokens = decode_step_hint
            previous_suffix = getattr(self, "_enforcer_last_suffix", None)
            if completion_tokens == 0 and not suffix:
                return self._enforcer_allowed_tokens_for_decode_step(suffix)
            if (
                previous_suffix is not None
                and completion_tokens == len(previous_suffix) + 1
                and len(suffix) == completion_tokens
                and suffix[-1] == input_token
            ):
                return self._enforcer_allowed_tokens_for_decode_step(suffix)

        suffix_key = tuple(suffix)
        previous_suffix = getattr(self, "_enforcer_last_suffix", None)
        is_new_suffix = suffix_key != previous_suffix
        if (
            previous_suffix is not None
            and is_new_suffix
            and (
                len(suffix_key) != len(previous_suffix) + 1
                or suffix_key[:-1] != previous_suffix
            )
        ):
            return self._rebuild_enforcer_at_suffix(suffix_key)

        if is_new_suffix:
            self._prepare_enforcer_for_next_state()
        allowed = self._enforcer.get_allowed_tokens(suffix_key)
        self._enforcer_last_suffix = suffix_key
        self._retain_current_enforcer_state(suffix_key)
        return allowed

    def _enforcer_allowed_tokens_for_decode_step(self, suffix: list[int]) -> Any:
        # ModelRunner supplies a strictly append-only request sequence. Let LMFE
        # advance its current state directly, then discard the predecessor state.
        self._prepare_enforcer_for_next_state()
        allowed = self._enforcer.get_allowed_tokens(suffix)
        prefix_states = self._enforcer.prefix_states
        if len(prefix_states) > self._DECODE_PREFIX_STATE_WINDOW:
            current_suffix = next(reversed(prefix_states))
            current_state = prefix_states[current_suffix]
            prefix_states.clear()
            prefix_states[current_suffix] = current_state
        self._enforcer_last_suffix = next(reversed(prefix_states))
        return allowed

    def _prepare_enforcer_for_next_state(self) -> None:
        prepare_next_state = getattr(self._enforcer, "_aster_prepare_for_next_state", None)
        if callable(prepare_next_state):
            prepare_next_state()

    def _rebuild_enforcer_at_suffix(self, suffix: tuple[int, ...]) -> Any:
        self._enforcer = _new_token_enforcer(
            self._tokenizer_data,
            self._enforcer.root_parser,
            reuse_freetext_token_lists=getattr(self, "_reuse_freetext_token_lists", False),
        )
        allowed = self._enforcer.get_allowed_tokens(())
        for length in range(1, len(suffix) + 1):
            self._prepare_enforcer_for_next_state()
            allowed = self._enforcer.get_allowed_tokens(suffix[:length])
        self._enforcer_last_suffix = suffix
        self._retain_current_enforcer_state(suffix)
        return allowed

    def _retain_current_enforcer_state(self, suffix: tuple[int, ...]) -> None:
        state = self._enforcer.prefix_states.get(suffix)
        if state is not None:
            self._enforcer.prefix_states = {suffix: state}

    def _allowed_tokens(self, suffix: list[int]) -> list[int]:
        self._pending_mask_allowed = None
        allowed_result = self._enforcer_allowed_tokens(suffix)
        allowed = getattr(allowed_result, "allowed_tokens", allowed_result)
        if allowed is None:
            return []
        if isinstance(allowed, list) and (not allowed or isinstance(allowed[0], int)):
            allowed_tokens = allowed
        else:
            allowed_tokens = [int(token_id) for token_id in allowed]
        context = self._json_context(suffix)
        if context in {"key_start", "in_key"}:
            allowed_tokens = self._filter_at_key_context(context, suffix, allowed_tokens)
        contains_eos = self._allowed_contains_eos(allowed_tokens)
        if contains_eos and not self._is_complete_json(suffix):
            allowed_tokens = [
                token_id for token_id in allowed_tokens if token_id not in self._eos_token_ids
            ]
            contains_eos = False
        self._pending_mask_allowed = allowed_tokens
        self._pending_mask_contains_eos = contains_eos
        return allowed_tokens

    def _allowed_contains_eos(self, allowed: list[int]) -> bool:
        if not self._eos_token_ids:
            return False
        cached_allowed = self._mask_cache_allowed
        cached_contains_eos = self._mask_cache_contains_eos
        if (
            cached_allowed is not None
            and cached_contains_eos is not None
            and len(cached_allowed) == len(allowed)
        ):
            if not allowed or (
                cached_allowed[0] == allowed[0]
                and cached_allowed[len(allowed) // 2] == allowed[len(allowed) // 2]
                and cached_allowed[-1] == allowed[-1]
            ):
                if cached_allowed == allowed:
                    return cached_contains_eos
        return any(token_id in allowed for token_id in self._eos_token_ids)

    def _is_complete_json(self, suffix: list[int]) -> bool:
        if not suffix:
            return False
        text = self._decode_suffix(suffix)
        if text is None:
            return False
        scan = _scan_json_text(text)
        if scan.in_string or scan.brace_depth or scan.bracket_depth:
            return False
        try:
            json.loads(text.strip())
        except (TypeError, ValueError, json.JSONDecodeError):
            return False
        return True

    def _decode_suffix(self, suffix: list[int]) -> str | None:
        key = tuple(suffix)
        text = self._decode_cache.get(key)
        if text is None:
            try:
                decoded = self._tokenizer.decode(suffix)
            except Exception:
                return None
            text = decoded if isinstance(decoded, str) else ""
            self._decode_cache[key] = text
        return text

    def _decode_token_cached(self, token_id: int) -> str | None:
        cached = self._token_decode_cache.get(token_id)
        if cached is not None or token_id in self._token_decode_cache:
            return cached
        try:
            decoded = self._tokenizer.decode([token_id])
        except Exception:
            decoded = None
        text = decoded if isinstance(decoded, str) else None
        self._token_decode_cache[token_id] = text
        return text

    def _json_context(self, suffix: list[int]) -> str:
        text = self._decode_suffix(suffix)
        if text is None:
            return "other"
        scan = _scan_json_text(text)
        if scan.in_string:
            before_key = text[: scan.last_quote_pos].rstrip()
            if (not before_key or before_key[-1] in {"{", ","}) and scan.container_stack:
                return "in_key" if scan.container_stack[-1] == "object" else "other"
            return "other"

        stripped = text.rstrip()
        if stripped and stripped[-1] in {"{", ","} and scan.container_stack:
            return "key_start" if scan.container_stack[-1] == "object" else "other"
        return "other"

    def _get_json_context(self, suffix: list[int]) -> str:
        return self._json_context(suffix)

    def _filter_at_key_context(self, context: str, suffix: list[int], allowed: list[int]) -> list[int]:
        if not self._valid_key_names:
            return allowed
        if context == "key_start":
            return self._filter_key_start_tokens(allowed)
        if context == "in_key":
            return self._filter_in_key_tokens(suffix, allowed)
        return allowed

    def _filter_key_start_tokens(self, allowed: list[int]) -> list[int]:
        filtered: list[int] = []
        for token_id in allowed:
            if token_id in self._eos_token_ids:
                continue
            token_text = self._decode_token_cached(token_id)
            if token_text is None:
                filtered.append(token_id)
                continue
            stripped = token_text.lstrip()
            if not stripped or stripped[0] == "}":
                filtered.append(token_id)
                continue
            if stripped[0] != '"':
                continue
            rest = stripped[1:]
            if not rest:
                filtered.append(token_id)
                continue
            close_index = rest.find('"')
            if close_index >= 0:
                if rest[:close_index] in self._valid_key_names:
                    filtered.append(token_id)
                continue
            if rest[0] in self._valid_key_first_chars and self._is_valid_key_prefix(rest):
                filtered.append(token_id)
        return filtered or allowed

    def _filter_in_key_tokens(self, suffix: list[int], allowed: list[int]) -> list[int]:
        text = self._decode_suffix(suffix)
        if text is None:
            return allowed
        quote_index = text.rfind('"')
        if quote_index < 0:
            return allowed
        key_so_far = text[quote_index + 1 :]
        filtered: list[int] = []
        for token_id in allowed:
            token_text = self._decode_token_cached(token_id)
            if token_text is None:
                filtered.append(token_id)
                continue
            if token_text and token_text[0] in {" ", "\t", "\n", "\r"}:
                continue
            close_index = token_text.find('"')
            if close_index >= 0:
                if key_so_far + token_text[:close_index] in self._valid_key_names:
                    filtered.append(token_id)
                continue
            if self._is_valid_key_prefix(key_so_far + token_text):
                filtered.append(token_id)
        return filtered or allowed

    def _is_valid_key_prefix(self, prefix: str) -> bool:
        return any(name.startswith(prefix) for name in self._valid_key_names)

    def _mask(self, allowed: list[int], logits: Any) -> Any:
        try:
            import mlx.core as mx
        except Exception as exc:  # pragma: no cover - runtime dependency guard
            raise LMFormatEnforcerNotAvailableError("mlx is required for constrained decoding") from exc

        shape = tuple(int(dimension) for dimension in logits.shape)
        pending_contains_eos = (
            self._pending_mask_contains_eos if allowed is self._pending_mask_allowed else None
        )
        self._pending_mask_allowed = None
        key: tuple[object, ...] = (shape, len(allowed))
        if allowed:
            key += (allowed[0], allowed[len(allowed) // 2], allowed[-1])
        cached_allowed = self._mask_cache_allowed
        cached_value = self._mask_cache_value
        if (
            key == self._mask_cache_key
            and cached_allowed is not None
            and cached_value is not None
            and cached_allowed == allowed
        ):
            return cached_value

        vocab_size = int(logits.shape[-1])
        allowed_clamped = [token_id for token_id in allowed if 0 <= token_id < vocab_size]
        mask = np.full(vocab_size, -np.inf, dtype=np.float32)
        mask[allowed_clamped] = 0.0
        mask_array = mx.array(mask)
        if getattr(logits, "ndim", 1) == 2 and logits.shape[0] == 1:
            mask_array = mask_array[None, :]
        self._mask_cache_key = key
        self._mask_cache_allowed = allowed.copy()
        self._mask_cache_value = mask_array
        self._mask_cache_contains_eos = (
            pending_contains_eos
            if pending_contains_eos is not None
            else any(token_id in allowed for token_id in self._eos_token_ids)
        )
        return mask_array


def _get_or_build_parser(schema: dict[str, Any] | None) -> tuple[dict[str, Any], Any]:
    from lmformatenforcer import JsonSchemaParser

    parser_schema = normalize_json_schema(schema or GENERIC_JSON_SCHEMA)
    key = _canonical_schema_key(parser_schema)
    cached = _PARSER_CACHE.get(key)
    if cached is not None:
        return cached
    parser = JsonSchemaParser(parser_schema)
    cached = (parser_schema, parser)
    _PARSER_CACHE[key] = cached
    return cached


def _canonical_schema_key(schema: dict[str, Any]) -> str:
    blob = json.dumps(schema, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def _eos_token_ids(tokenizer_data: Any) -> set[int]:
    eos = getattr(tokenizer_data, "eos_token_id", None)
    if isinstance(eos, (list, tuple, set)):
        return {int(token_id) for token_id in eos}
    if eos is None:
        return set()
    return {int(eos)}


def _token_list(tokens: Any) -> list[int]:
    values = tokens.tolist() if hasattr(tokens, "tolist") else list(tokens)
    if isinstance(values, int):
        return [values]
    if values and isinstance(values[0], list):
        values = values[0]
    return [int(token_id) for token_id in values]


class _JsonScanState:
    def __init__(
        self,
        *,
        in_string: bool,
        last_quote_pos: int,
        brace_depth: int,
        bracket_depth: int,
        container_stack: list[str],
    ) -> None:
        self.in_string = in_string
        self.last_quote_pos = last_quote_pos
        self.brace_depth = brace_depth
        self.bracket_depth = bracket_depth
        self.container_stack = container_stack


def _scan_json_text(text: str) -> _JsonScanState:
    in_string = False
    last_quote_pos = -1
    brace_depth = 0
    bracket_depth = 0
    container_stack: list[str] = []
    index = 0
    while index < len(text):
        char = text[index]
        if in_string:
            if char == "\\" and index + 1 < len(text):
                index += 2
                continue
            if char == '"':
                in_string = False
        else:
            if char == '"':
                in_string = True
                last_quote_pos = index
            elif char == "{":
                brace_depth += 1
                container_stack.append("object")
            elif char == "}":
                brace_depth -= 1
                if container_stack and container_stack[-1] == "object":
                    container_stack.pop()
            elif char == "[":
                bracket_depth += 1
                container_stack.append("array")
            elif char == "]":
                bracket_depth -= 1
                if container_stack and container_stack[-1] == "array":
                    container_stack.pop()
        index += 1
    return _JsonScanState(
        in_string=in_string,
        last_quote_pos=last_quote_pos,
        brace_depth=brace_depth,
        bracket_depth=bracket_depth,
        container_stack=container_stack,
    )


def _collect_property_names(schema: dict[str, Any] | None) -> set[str]:
    names: set[str] = set()
    _walk_property_names(schema, names)
    return names


def _walk_property_names(node: Any, names: set[str]) -> None:
    if not isinstance(node, dict):
        return
    properties = node.get("properties")
    if isinstance(properties, dict):
        names.update(str(name) for name in properties)
        for value in properties.values():
            _walk_property_names(value, names)
    for key in ("items", "additionalProperties", "not"):
        value = node.get(key)
        if isinstance(value, dict):
            _walk_property_names(value, names)
    for key in ("allOf", "anyOf", "oneOf"):
        values = node.get(key)
        if isinstance(values, list):
            for value in values:
                _walk_property_names(value, names)


class ThinkingAwareJsonLogitsProcessor:
    name = "thinking_aware_json_schema"
    batch_sampling_mode = "eager_rows"

    def __init__(
        self,
        inner: Any,
        *,
        tokenizer: Any,
        prompt_has_think_tag: bool = False,
        json_start_scan_limit: int = 50,
    ) -> None:
        self._inner = inner
        self._tokenizer = tokenizer
        self._prompt_has_think_tag = prompt_has_think_tag
        self._json_start_scan_limit = max(1, json_start_scan_limit)
        self._active = False
        self._in_thinking: bool | None = True if prompt_has_think_tag else None
        self._waiting_for_json = False
        self._base_prompt_len: int | None = None
        self._json_scan_offset: int | None = None

    @property
    def schema(self) -> Any:
        return getattr(self._inner, "schema", None)

    @property
    def is_active(self) -> bool:
        return self._active

    def __call__(self, tokens: Any, logits: Any) -> Any:
        if self._active:
            return self._inner(tokens, logits)

        token_ids = _token_list(tokens)
        if self._base_prompt_len is None:
            self._base_prompt_len = len(token_ids)

        if self._waiting_for_json:
            return self._scan_for_json_start(token_ids, tokens, logits)

        generated = token_ids[self._base_prompt_len :]
        if not generated:
            return logits

        if self._in_thinking is None:
            prefix = self._decode(generated[: min(3, len(generated))])
            if "<think>" in prefix:
                self._in_thinking = True
            elif len(generated) >= 3:
                self._waiting_for_json = True
                self._json_scan_offset = self._base_prompt_len
                return self._scan_for_json_start(token_ids, tokens, logits)
            else:
                return logits

        if self._in_thinking:
            end_offset = self._think_end_offset(generated)
            if end_offset is not None:
                self._waiting_for_json = True
                self._json_scan_offset = (self._base_prompt_len or 0) + end_offset
                return self._scan_for_json_start(token_ids, tokens, logits)

        return logits

    def _scan_for_json_start(self, token_ids: list[int], tokens: Any, logits: Any) -> Any:
        start = self._json_scan_offset if self._json_scan_offset is not None else self._base_prompt_len or 0
        scan_tokens = token_ids[start:]
        for index, token_id in enumerate(scan_tokens):
            if any(marker in self._decode([token_id]) for marker in ("{", "[")):
                self._active = True
                if hasattr(self._inner, "_prompt_len"):
                    self._inner._prompt_len = start + index
                return self._inner(tokens, logits)
        if len(scan_tokens) > self._json_start_scan_limit:
            self._active = True
            if hasattr(self._inner, "_prompt_len"):
                self._inner._prompt_len = len(token_ids)
            return self._inner(tokens, logits)
        return logits

    def _decode(self, token_ids: list[int]) -> str:
        try:
            decoded = self._tokenizer.decode(token_ids)
        except Exception:
            return ""
        return decoded if isinstance(decoded, str) else ""

    def _think_end_offset(self, generated: list[int]) -> int | None:
        recent_window = min(8, len(generated))
        if "</think>" in self._decode(generated[-recent_window:]):
            return len(generated)
        if "</think>" not in self._decode(generated):
            return None
        for index in range(len(generated)):
            if "</think>" in self._decode(generated[: index + 1]):
                return index + 1
        return len(generated)
