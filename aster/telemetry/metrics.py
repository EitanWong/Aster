from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest


class MetricsRegistry:
    def __init__(self, namespace: str) -> None:
        self.registry = CollectorRegistry()
        self.request_latency = Histogram(
            f"{namespace}_request_latency_seconds", "End-to-end request latency", registry=self.registry
        )
        self.first_token_latency = Histogram(
            f"{namespace}_first_token_latency_seconds", "First token latency", registry=self.registry
        )
        self.prefill_latency = Histogram(
            f"{namespace}_prefill_latency_seconds", "Prefill latency", registry=self.registry
        )
        self.decode_latency = Histogram(
            f"{namespace}_decode_latency_seconds", "Decode latency", registry=self.registry
        )
        self.queue_depth = Gauge(f"{namespace}_queue_depth", "Scheduler queue depth", registry=self.registry)
        self.batch_size = Histogram(f"{namespace}_batch_size", "Observed batch size", registry=self.registry)
        self.prefix_cache_hits = Counter(
            f"{namespace}_prefix_cache_hits_total", "Prefix cache hits", registry=self.registry
        )
        self.prefix_cache_misses = Counter(
            f"{namespace}_prefix_cache_misses_total", "Prefix cache misses", registry=self.registry
        )
        self.kv_pages_used = Gauge(f"{namespace}_kv_pages_used", "KV pages used", registry=self.registry)
        self.spec_acceptance = Histogram(
            f"{namespace}_speculative_acceptance_ratio", "Speculative acceptance ratio", registry=self.registry
        )
        self.worker_restarts = Counter(
            f"{namespace}_worker_restarts_total", "Inference worker restarts", registry=self.registry
        )
        self.errors = Counter(f"{namespace}_errors_total", "Error count", ["code"], registry=self.registry)

    def render(self) -> bytes:
        return generate_latest(self.registry)
