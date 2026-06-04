# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import copy
import hashlib
import json
from typing import Any

GENERIC_JSON_SCHEMA: dict[str, Any] = {
    "anyOf": [
        {"type": "object"},
        {"type": "array"},
    ]
}

_NORMALIZED_SCHEMA_CACHE: dict[str, dict[str, Any]] = {}


def canonical_schema_key(schema: dict[str, Any] | None) -> str:
    if schema is None:
        return "__none__"
    blob = json.dumps(schema, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def normalize_json_schema(schema: dict[str, Any] | None) -> dict[str, Any]:
    key = canonical_schema_key(schema)
    cached = _NORMALIZED_SCHEMA_CACHE.get(key)
    if cached is not None:
        return copy.deepcopy(cached)

    normalized = copy.deepcopy(GENERIC_JSON_SCHEMA if schema is None else schema)
    normalized = simplify_json_schema(normalized)
    normalized = force_no_additional_properties(normalized)
    _NORMALIZED_SCHEMA_CACHE[key] = normalized
    return copy.deepcopy(normalized)


def clear_schema_cache() -> None:
    _NORMALIZED_SCHEMA_CACHE.clear()


def simplify_json_schema(schema: dict[str, Any]) -> dict[str, Any]:
    schema = copy.deepcopy(schema)
    definitions: dict[str, Any] = {}
    definitions.update(schema.pop("definitions", {}))
    definitions.update(schema.pop("$defs", {}))
    resolving: set[str] = set()

    def resolve(node: Any, depth: int = 0) -> Any:
        if depth > 12 or not isinstance(node, dict):
            return node

        if "$ref" in node:
            ref = node["$ref"]
            parts = ref.split("/")
            if (
                isinstance(ref, str)
                and len(parts) == 3
                and parts[0] == "#"
                and parts[1] in {"definitions", "$defs"}
            ):
                name = parts[2]
                if name in definitions and ref not in resolving:
                    resolving.add(ref)
                    resolved = copy.deepcopy(definitions[name])
                    for key, value in node.items():
                        if key != "$ref" and key not in resolved:
                            resolved[key] = value
                    result = resolve(resolved, depth + 1)
                    resolving.discard(ref)
                    return result
            return {}

        for unsupported in (
            "not",
            "$schema",
            "$id",
            "default",
            "examples",
            "title",
            "description",
        ):
            node.pop(unsupported, None)

        if isinstance(node.get("type"), list):
            types = node.pop("type")
            items_schema = node.pop("items", None)
            branches: list[dict[str, Any]] = []
            for schema_type in types:
                branch: dict[str, Any] = {"type": schema_type}
                if schema_type == "array" and items_schema is not None:
                    branch["items"] = resolve(copy.deepcopy(items_schema), depth + 1)
                branches.append(branch)
            existing = node.pop("anyOf", [])
            node["anyOf"] = (existing if isinstance(existing, list) else []) + branches

        properties = node.get("properties")
        if isinstance(properties, dict):
            for key in list(properties):
                properties[key] = resolve(properties[key], depth + 1)

        for key in ("items", "additionalProperties"):
            if isinstance(node.get(key), dict):
                node[key] = resolve(node[key], depth + 1)

        for key in ("allOf", "anyOf", "oneOf"):
            if isinstance(node.get(key), list):
                resolved_items = [resolve(item, depth + 1) for item in node[key]]
                node[key] = [item for item in resolved_items if item != {}]
                if not node[key]:
                    del node[key]

        for key in ("anyOf", "oneOf"):
            if isinstance(node.get(key), list):
                flattened: list[Any] = []
                for item in node[key]:
                    if isinstance(item, dict) and key in item and len(item) == 1:
                        flattened.extend(item[key])
                    else:
                        flattened.append(item)
                node[key] = flattened

        return node

    return resolve(schema)


def force_no_additional_properties(schema: dict[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(schema)
    _inject_no_additional_properties(normalized)
    return normalized


def _inject_no_additional_properties(node: Any) -> None:
    if not isinstance(node, dict):
        return
    if "properties" in node and "additionalProperties" not in node:
        node["additionalProperties"] = False
    for value in node.values():
        if isinstance(value, dict):
            _inject_no_additional_properties(value)
        elif isinstance(value, list):
            for item in value:
                _inject_no_additional_properties(item)
