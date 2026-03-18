#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import yaml


@dataclass
class RunResult:
    text: str
    metrics: dict[str, Any]


def load_config(config_path: str) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_base_url(config: dict[str, Any]) -> str:
    api = config.get("api", {})
    host = api.get("host", "127.0.0.1")
    port = api.get("port", 8080)
    return f"http://{host}:{port}"


def default_model(config: dict[str, Any]) -> str:
    model = config.get("model", {})
    return model.get("name", "Qwen3.5-9B")


def now() -> float:
    return time.perf_counter()


def round3(value: float) -> float:
    return round(value, 3)


def approx_token_count(text: str) -> int:
    stripped = text.strip()
    if not stripped:
        return 0
    return max(1, math.ceil(len(stripped) / 4))


def parse_prometheus_metrics(text: str) -> dict[str, float]:
    values: dict[str, float] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 2:
            continue
        name, raw_value = parts
        try:
            values[name] = float(raw_value)
        except ValueError:
            continue
    return values


def fetch_metrics_snapshot(client: httpx.Client, base_url: str, timeout: float) -> dict[str, float]:
    try:
        resp = client.get(f"{base_url}/metrics", timeout=timeout)
        resp.raise_for_status()
        return parse_prometheus_metrics(resp.text)
    except Exception:
        return {}


def diff_metric(before: dict[str, float], after: dict[str, float], key: str) -> str:
    if key not in before and key not in after:
        return "n/a"
    return str(round3(after.get(key, 0.0) - before.get(key, 0.0)))


def first_present(snapshot: dict[str, float], suffixes: list[str]) -> str:
    for key in sorted(snapshot.keys()):
        for suffix in suffixes:
            if key.endswith(suffix):
                return key
    return ""


def parse_stream_line(line: str) -> tuple[str, dict[str, Any] | None]:
    if not line.startswith("data: "):
        return "ignore", None
    data = line[6:].strip()
    if data == "[DONE]":
        return "done", None
    try:
        payload = json.loads(data)
    except json.JSONDecodeError:
        return "ignore", None
    obj = payload.get("object")
    if obj == "aster.stream.summary":
        return "summary", payload.get("aster") or {}
    choices = payload.get("choices") or []
    if not choices:
        return "ignore", None
    delta = choices[0].get("delta") or {}
    text = delta.get("content")
    if text is None:
        return "ignore", None
    return "text", {"text": str(text)}


def print_metrics(metrics: dict[str, Any]) -> None:
    print("\n--- Performance Summary ---")
    for key, value in metrics.items():
        print(f"{key:24} {value}")
    print()


def build_common_summary(
    *,
    mode: str,
    text: str,
    prompt_chars: int,
    elapsed: float,
    started: float,
    finished: float,
    before_metrics: dict[str, float],
    after_metrics: dict[str, float],
    http_status: int,
) -> dict[str, Any]:
    request_latency_key = first_present(after_metrics, ["_request_latency_seconds_sum"])
    first_token_key = first_present(after_metrics, ["_first_token_latency_seconds_sum"])
    cache_hit_key = first_present(after_metrics, ["_prefix_cache_hits_total"])
    cache_miss_key = first_present(after_metrics, ["_prefix_cache_misses_total"])
    error_key = first_present(after_metrics, ["_errors_total"])
    queue_depth_key = first_present(after_metrics, ["_queue_depth"])

    approx_tokens = approx_token_count(text)
    tok_s = round3(approx_tokens / elapsed) if elapsed > 0 and approx_tokens > 0 else "n/a"
    return {
        "mode": mode,
        "started_at_perf_s": round3(started),
        "finished_at_perf_s": round3(finished),
        "latency_s": round3(elapsed),
        "output_chars": len(text),
        "output_lines": len(text.splitlines()) if text else 0,
        "approx_output_tokens": approx_tokens,
        "approx_output_tok_s": tok_s,
        "prompt_chars": prompt_chars,
        "http_status": http_status,
        "metrics_request_latency_delta_s": diff_metric(before_metrics, after_metrics, request_latency_key) if request_latency_key else "n/a",
        "metrics_first_token_delta_s": diff_metric(before_metrics, after_metrics, first_token_key) if first_token_key else "n/a",
        "metrics_cache_hits_delta": diff_metric(before_metrics, after_metrics, cache_hit_key) if cache_hit_key else "n/a",
        "metrics_cache_misses_delta": diff_metric(before_metrics, after_metrics, cache_miss_key) if cache_miss_key else "n/a",
        "metrics_errors_delta": diff_metric(before_metrics, after_metrics, error_key) if error_key else "n/a",
        "queue_depth_after": round3(after_metrics.get(queue_depth_key, 0.0)) if queue_depth_key else "n/a",
    }


def run_non_stream(
    client: httpx.Client,
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float,
    top_p: float,
    timeout: float,
) -> RunResult:
    before = fetch_metrics_snapshot(client, base_url, min(timeout, 10.0))
    started = now()
    resp = client.post(
        f"{base_url}/v1/chat/completions",
        json={
            "model": model,
            "stream": False,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "messages": messages,
        },
        timeout=timeout,
    )
    finished = now()
    elapsed = finished - started
    resp.raise_for_status()
    payload = resp.json()
    text = payload["choices"][0]["message"]["content"]
    usage = payload.get("usage", {})
    aster = payload.get("aster", {})
    after = fetch_metrics_snapshot(client, base_url, min(timeout, 10.0))
    summary = build_common_summary(
        mode="non-stream",
        text=text,
        prompt_chars=sum(len(m.get("content", "")) for m in messages),
        elapsed=elapsed,
        started=started,
        finished=finished,
        before_metrics=before,
        after_metrics=after,
        http_status=resp.status_code,
    )
    completion_tokens = usage.get("completion_tokens", 0)
    exact_tok_s = round3(completion_tokens / elapsed) if elapsed > 0 and completion_tokens else "n/a"
    summary.update(
        {
            "request_id": resp.headers.get("X-Request-Id", payload.get("id", "n/a")),
            "prompt_tokens": usage.get("prompt_tokens", "n/a"),
            "completion_tokens": completion_tokens,
            "total_tokens": usage.get("total_tokens", "n/a"),
            "exact_output_tok_s": exact_tok_s,
            "cache_hit": aster.get("cache_hit", "n/a"),
            "speculative": aster.get("speculative_enabled", "n/a"),
        }
    )
    return RunResult(text=text, metrics=summary)


def run_stream(
    client: httpx.Client,
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float,
    top_p: float,
    timeout: float,
) -> RunResult:
    before = fetch_metrics_snapshot(client, base_url, min(timeout, 10.0))
    pieces: list[str] = []
    chunk_count = 0
    started = now()
    first_piece_at: float | None = None
    service_summary: dict[str, Any] = {}
    with client.stream(
        "POST",
        f"{base_url}/v1/chat/completions",
        json={
            "model": model,
            "stream": True,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "messages": messages,
        },
        headers={"X-Aster-Debug": "1"},
        timeout=timeout,
    ) as resp:
        resp.raise_for_status()
        print("Assistant: ", end="", flush=True)
        for line in resp.iter_lines():
            if not line:
                continue
            kind, payload = parse_stream_line(line)
            if kind == "text" and payload is not None:
                text = str(payload["text"])
                if first_piece_at is None:
                    first_piece_at = now()
                pieces.append(text)
                chunk_count += 1
                print(text, end="", flush=True)
            elif kind == "summary" and payload is not None:
                service_summary = payload
            elif kind == "done":
                break
        print()
        http_status = resp.status_code
    finished = now()
    elapsed = finished - started
    text = "".join(pieces)
    after = fetch_metrics_snapshot(client, base_url, min(timeout, 10.0))
    summary = build_common_summary(
        mode="stream",
        text=text,
        prompt_chars=sum(len(m.get("content", "")) for m in messages),
        elapsed=elapsed,
        started=started,
        finished=finished,
        before_metrics=before,
        after_metrics=after,
        http_status=http_status,
    )
    ttft = round3(first_piece_at - started) if first_piece_at is not None else "n/a"
    gen_elapsed = finished - first_piece_at if first_piece_at is not None else 0.0
    approx_piece_s = round3(chunk_count / gen_elapsed) if gen_elapsed > 0 and chunk_count else "n/a"
    summary.update({"ttft_s": ttft, "streamed_pieces": chunk_count, "approx_pieces_s": approx_piece_s})
    if service_summary:
        generation_tps = service_summary.get("generation_tps")
        prompt_tps = service_summary.get("prompt_tps")
        completion_tokens = service_summary.get("completion_tokens")
        summary.update(
            {
                "request_id": service_summary.get("request_id", "n/a"),
                "prompt_tokens": service_summary.get("prompt_tokens", "n/a"),
                "completion_tokens": completion_tokens if completion_tokens is not None else "n/a",
                "exact_output_tok_s": round3(float(generation_tps)) if generation_tps is not None else "n/a",
                "prompt_tps": round3(float(prompt_tps)) if prompt_tps is not None else "n/a",
                "cache_hit": service_summary.get("cache_hit", "n/a"),
                "prefill_cache_hit": service_summary.get("prefill_cache_hit", "n/a"),
                "generation_cache_reuse": service_summary.get("generation_cache_reuse", "n/a"),
                "speculative": service_summary.get("speculative_enabled", "n/a"),
                "speculative_path_mode": service_summary.get("speculative_path_mode", "n/a"),
                "peak_memory_gb": round3(float(service_summary.get("peak_memory_gb", 0.0))),
            }
        )
    else:
        summary["note"] = "Streaming summary payload missing; exact service-side tok/s unavailable."
    return RunResult(text=text, metrics=summary)


def fetch_and_print_metrics(client: httpx.Client, base_url: str, timeout: float) -> None:
    try:
        resp = client.get(f"{base_url}/metrics", timeout=timeout)
        resp.raise_for_status()
        print(resp.text[:4000])
    except Exception as exc:
        print(f"Failed to fetch metrics: {exc}")


def transcript_payload(model: str, messages: list[dict[str, str]]) -> dict[str, Any]:
    return {"model": model, "messages": messages}


def save_transcript(path: str, model: str, messages: list[dict[str, str]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(transcript_payload(model, messages), ensure_ascii=False, indent=2), encoding="utf-8")


def load_transcript(path: str) -> tuple[str | None, list[dict[str, str]]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    model = payload.get("model")
    messages = payload.get("messages") or []
    if not isinstance(messages, list):
        raise ValueError("Invalid transcript: messages must be a list")
    cleaned: list[dict[str, str]] = []
    for item in messages:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", "user"))
        content = str(item.get("content", ""))
        cleaned.append({"role": role, "content": content})
    return model, cleaned


def aggregate_stats(results: list[RunResult]) -> dict[str, Any]:
    if not results:
        return {"turns": 0}
    latencies = [float(r.metrics["latency_s"]) for r in results if isinstance(r.metrics.get("latency_s"), (int, float))]
    ttfts = [float(r.metrics["ttft_s"]) for r in results if isinstance(r.metrics.get("ttft_s"), (int, float))]
    exact_tps = [float(r.metrics["exact_output_tok_s"]) for r in results if isinstance(r.metrics.get("exact_output_tok_s"), (int, float))]
    completion_tokens = [int(r.metrics["completion_tokens"]) for r in results if isinstance(r.metrics.get("completion_tokens"), int)]
    return {
        "turns": len(results),
        "latency_avg_s": round3(sum(latencies) / len(latencies)) if latencies else "n/a",
        "ttft_avg_s": round3(sum(ttfts) / len(ttfts)) if ttfts else "n/a",
        "exact_output_tok_s_avg": round3(sum(exact_tps) / len(exact_tps)) if exact_tps else "n/a",
        "completion_tokens_total": sum(completion_tokens) if completion_tokens else 0,
    }


def emit_result(result: RunResult, *, output_format: str, stats_only: bool) -> None:
    if output_format == "json":
        payload = {"text": result.text, "metrics": result.metrics}
        if stats_only:
            payload = {"metrics": result.metrics}
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    if not stats_only:
        print_metrics(result.metrics)
    else:
        print(json.dumps(result.metrics, ensure_ascii=False, indent=2))


def repl(args: argparse.Namespace, client: httpx.Client, base_url: str, model: str, messages: list[dict[str, str]]) -> int:
    session_results: list[RunResult] = []
    print(f"Aster chat CLI connected to {base_url} using model {model}")
    print("Commands: /quit /reset /messages /metrics /stats /save <path> /load <path>")
    print(f"Streaming mode: {'on' if args.stream else 'off'}")
    while True:
        try:
            user_text = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            return 0
        if not user_text:
            continue
        if user_text == "/quit":
            print("Bye.")
            return 0
        if user_text == "/reset":
            messages[:] = ([{"role": "system", "content": args.system}] if args.system else [])
            session_results.clear()
            print("Conversation reset.")
            continue
        if user_text == "/messages":
            print(json.dumps(messages, ensure_ascii=False, indent=2))
            continue
        if user_text == "/metrics":
            fetch_and_print_metrics(client, base_url, args.timeout)
            continue
        if user_text == "/stats":
            print(json.dumps(aggregate_stats(session_results), ensure_ascii=False, indent=2))
            continue
        if user_text.startswith("/save "):
            path = user_text[6:].strip()
            save_transcript(path, model, messages)
            print(f"Saved transcript to {path}")
            continue
        if user_text.startswith("/load "):
            path = user_text[6:].strip()
            loaded_model, loaded_messages = load_transcript(path)
            messages[:] = loaded_messages
            if loaded_model:
                model = loaded_model
            session_results.clear()
            print(f"Loaded transcript from {path}")
            continue

        messages.append({"role": "user", "content": user_text})
        try:
            if args.stream:
                result = run_stream(client, base_url, model, messages, args.max_tokens, args.temperature, args.top_p, args.timeout)
            else:
                result = run_non_stream(client, base_url, model, messages, args.max_tokens, args.temperature, args.top_p, args.timeout)
                if args.format == "text" and not args.stats_only:
                    print(f"Assistant: {result.text}")
        except httpx.TimeoutException:
            print(f"Request timed out after {args.timeout}s.")
            messages.pop()
            continue
        except Exception as exc:
            print(f"Request failed: {exc}")
            messages.pop()
            continue

        session_results.append(result)
        emit_result(result, output_format=args.format, stats_only=args.stats_only)
        messages.append({"role": "assistant", "content": result.text})
        if args.save_transcript:
            save_transcript(args.save_transcript, model, messages)


def main() -> int:
    parser = argparse.ArgumentParser(description="Interactive CLI for local Aster chat testing")
    parser.add_argument("prompt", nargs="?", help="Optional one-shot prompt")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config YAML")
    parser.add_argument("--model", default=None, help="Override model name")
    parser.add_argument("--system", default=None, help="Optional system prompt")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--stream", action="store_true", help="Use streaming mode")
    parser.add_argument("--repl", action="store_true", help="Force interactive chat mode")
    parser.add_argument("--save-transcript", default=None, help="Save transcript JSON after each turn")
    parser.add_argument("--load-transcript", default=None, help="Load transcript JSON before chatting")
    parser.add_argument("--stats-only", action="store_true", help="Only emit metrics/stats, not assistant text summary block")
    parser.add_argument("--format", choices=["text", "json"], default="text", help="Output format")
    args = parser.parse_args()

    config = load_config(args.config)
    base_url = build_base_url(config)
    model = args.model or default_model(config)
    messages: list[dict[str, str]] = []
    if args.load_transcript:
        loaded_model, loaded_messages = load_transcript(args.load_transcript)
        messages = loaded_messages
        if loaded_model and args.model is None:
            model = loaded_model
    elif args.system:
        messages.append({"role": "system", "content": args.system})

    try:
        with httpx.Client() as client:
            if args.prompt and not args.repl:
                local_messages = list(messages)
                local_messages.append({"role": "user", "content": args.prompt})
                try:
                    if args.stream:
                        result = run_stream(client, base_url, model, local_messages, args.max_tokens, args.temperature, args.top_p, args.timeout)
                    else:
                        result = run_non_stream(client, base_url, model, local_messages, args.max_tokens, args.temperature, args.top_p, args.timeout)
                        if args.format == "text" and not args.stats_only:
                            print(f"Assistant: {result.text}")
                except httpx.TimeoutException:
                    print(f"Fatal error: timed out after {args.timeout}s.", file=sys.stderr)
                    return 1
                emit_result(result, output_format=args.format, stats_only=args.stats_only)
                local_messages.append({"role": "assistant", "content": result.text})
                if args.save_transcript:
                    save_transcript(args.save_transcript, model, local_messages)
                return 0
            return repl(args, client, base_url, model, messages)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130
    except Exception as exc:
        print(f"Fatal error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
