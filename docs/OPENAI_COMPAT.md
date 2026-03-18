# Aster OpenAI Compatibility Contract

This document defines Aster's compatibility boundary for its OpenAI-style API.

The goal is simple:

> **Default API behavior must remain OpenAI-compatible.**

Aster may expose additional diagnostics and operator tooling, but those must be **opt-in** and must not pollute default responses used by standard OpenAI clients.

---

## Compatibility principle

By default, Aster should behave like an OpenAI-compatible chat/completions backend for the endpoints it implements.

That means:

- standard clients should be able to call Aster without special handling
- default request and response shapes should remain compatible
- Aster-specific diagnostics must not appear unless explicitly requested

---

## Implemented endpoints

Aster currently implements these API endpoints:

- `GET /v1/models`
- `POST /v1/chat/completions`
- `POST /v1/completions`

Operational endpoints also exist, but they are **not part of the OpenAI API surface**:

- `GET /health`
- `GET /ready`
- `GET /metrics`

These operational endpoints are intentionally Aster-specific.

---

## Default streaming behavior

For streaming chat/completions requests, the default SSE contract must remain:

- zero or more `chat.completion.chunk`-style data events
- final `data: [DONE]`

When the client does **not** opt into Aster debug metadata, Aster must **not** append any extra SSE event types after normal chunks.

### Required default rule

Without explicit debug opt-in, this is allowed:

```text
data: {"object":"chat.completion.chunk", ...}
data: {"object":"chat.completion.chunk", ...}
data: [DONE]
```

This is **not** allowed by default:

```text
data: {"object":"chat.completion.chunk", ...}
data: {"object":"aster.stream.summary", ...}
data: [DONE]
```

---

## Aster-specific debug extension

Aster exposes richer streaming diagnostics only when explicitly requested.

### Debug opt-in header

```http
X-Aster-Debug: 1
```

When this header is present on a streaming request, Aster may append an additional diagnostic SSE payload before `[DONE]`:

```json
{
  "object": "aster.stream.summary",
  "model": "Qwen3.5-9B",
  "aster": {
    "request_id": "...",
    "prompt_tokens": 19,
    "completion_tokens": 4,
    "cache_hit": false,
    "prefill_cache_hit": false,
    "generation_cache_reuse": false,
    "speculative_enabled": false,
    "speculative_path_mode": "disabled",
    "prompt_tps": 194.097,
    "generation_tps": 14.433,
    "peak_memory_gb": 5.198
  }
}
```

### Compatibility rule

- `X-Aster-Debug: 1` is an **Aster extension**, not part of OpenAI semantics
- standard clients are not expected to send it
- only Aster-aware tooling such as `scripts/chat.sh` / `scripts/chat_cli.py` should rely on it

---

## Request tracing extension

Aster supports an optional request tracing header:

```http
X-Request-Id: <client-supplied-id>
```

Behavior:

- if present, Aster propagates it through route → scheduler → inference → logs
- for non-stream responses, Aster echoes it back as the `X-Request-Id` response header
- if absent, Aster generates a request id internally

This is a safe transport-level extension and does not change the JSON response contract.

---

## Current response shape notes

### `/v1/chat/completions`
Aster currently returns an OpenAI-style response body with:

- `id`
- `object`
- `created`
- `model`
- `choices`
- `usage`

It also currently includes an additional top-level object:

```json
"aster": {
  "cache_hit": false,
  "speculative_enabled": false
}
```

### `/v1/completions`
Likewise, Aster returns an OpenAI-style completion body plus the same `aster` metadata object.

### Practical status

This means Aster is currently:

- **OpenAI-compatible in practice for many clients**
- but **not strictly byte-for-byte identical** to OpenAI responses because of the extra `aster` object

This is acceptable only if the extra fields do not break clients.

If stricter compatibility becomes necessary, Aster should move these fields behind an explicit opt-in mechanism similar to `X-Aster-Debug: 1`.

---

## Rules for future changes

When adding new features, follow these rules:

### Allowed by default

- improving internals without changing public response shape
- adding logs
- adding internal metrics
- adding request tracing headers
- changing implementation details behind stable outputs

### Not allowed by default

- adding extra SSE event objects to default streams
- changing `object` types away from OpenAI-style values
- changing the meaning of `choices`, `usage`, or `[DONE]`
- requiring custom headers for ordinary OpenAI-compatible requests
- returning Aster-only required fields in default requests

### Allowed only as opt-in

- debug metadata events in streaming
- Aster-specific response enrichments
- custom diagnostic headers
- extra introspection payloads for local tooling

---

## Guidance for Aster tooling

Aster's local tools may use extensions, but they must do so explicitly.

### `chat_cli.py` / `chat.sh`
These tools are allowed to:

- send `X-Aster-Debug: 1`
- consume `aster.stream.summary`
- display real service-side `generation_tps`, `completion_tokens`, and related debug data

They should not assume that ordinary OpenAI clients will ever see these fields.

---

## Recommended future tightening

If strict compatibility matters more over time, the next tightening step should be:

1. keep default JSON bodies limited to OpenAI fields only
2. move the current top-level `aster` JSON field behind explicit debug opt-in
3. reserve all Aster-specific metadata for:
   - debug headers
   - operational endpoints
   - local tooling

---

## Summary

The compatibility contract is:

- **default behavior stays OpenAI-compatible**
- **Aster diagnostics are opt-in**
- **operator tooling may use Aster extensions explicitly**
- **future changes must not silently pollute default client flows**
