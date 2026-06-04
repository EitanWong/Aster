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
- `POST /v1/responses`
- `POST /v1/chat/completions`
- `POST /v1/completions`

Operational endpoints also exist, but they are **not part of the OpenAI API surface**:

- `GET /health`
- `GET /ready`
- `GET /metrics`
- `GET /v1/status`

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
- Aster echoes it back as the `X-Request-Id` response header on local OpenAI-compatible responses, including Chat/Completions and Responses API tools or structured-output paths
- if absent, Aster generates a request id internally

This is a safe transport-level extension and does not change the JSON response contract. JSON response `id` values remain OpenAI-style response identifiers such as `chatcmpl-...`, `cmpl-...`, or `resp_...`; they are not the request tracing id. For `/v1/responses`, clients must pass the returned JSON `id` as `previous_response_id`; `X-Request-Id` is only for tracing.

---

## Responses replay history

`/v1/responses` supports `previous_response_id` using an in-memory replay
store. The store is process-local and provider-scoped: if Aster is run with
multiple worker processes, a follow-up request must hit the same process and
provider-facing endpoint that created the original `resp_...` id or it will
return `response_not_found`. For example, a response created through
`/v1/responses` is not replayable through `/xai/v1/responses`.

`previous_response_id` must be a string when provided. Non-string values are
rejected with `invalid_previous_response_id` instead of being treated as a new
conversation.

Operators can inspect the current process store via `GET /v1/status`:

```json
"responses_store": {
  "entries": 1,
  "max_entries": 1000,
  "scope": "process/provider"
}
```

The capacity is controlled by `api.responses_store_max_entries`; the default is
`1000`, matching vllm-mlx's in-process store size.

---

## Current response shape notes

### `/v1/chat/completions`
Aster returns an OpenAI-style response body with:

- `id`
- `object`
- `created`
- `model`
- `choices`
- `usage`

When `X-Aster-Debug: 1` is present, Aster adds an extension object:

```json
"aster": {
  "cache_hit": false,
  "speculative_enabled": false
}
```

### `/v1/completions`
Likewise, Aster returns an OpenAI-style completion body by default and adds the same `aster` metadata object only when `X-Aster-Debug: 1` is present.

### Practical status

This means Aster is currently:

- OpenAI-compatible by default for these response shapes
- able to expose Aster-specific debug metadata through explicit opt-in

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
2. keep `/v1/responses` history replay keyed by returned response IDs, not request trace headers
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
