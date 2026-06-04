from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest


class MetricsRegistry:
    def __init__(self, namespace: str) -> None:
        self.registry = CollectorRegistry()
        self.request_latency = Histogram(
            f"{namespace}_request_latency_seconds",
            "End-to-end request latency",
            registry=self.registry,
        )
        self.first_token_latency = Histogram(
            f"{namespace}_first_token_latency_seconds",
            "First token latency",
            registry=self.registry,
        )
        self.prefill_latency = Histogram(
            f"{namespace}_prefill_latency_seconds",
            "Prefill chunk latency",
            registry=self.registry,
        )
        self.decode_latency = Histogram(
            f"{namespace}_decode_latency_seconds",
            "Decode latency",
            registry=self.registry,
        )
        self.queue_depth = Gauge(
            f"{namespace}_queue_depth",
            "Submission queue depth",
            registry=self.registry,
        )
        self.active_requests = Gauge(
            f"{namespace}_active_requests",
            "Requests owned by the engine loop",
            registry=self.registry,
        )
        self.prefill_active = Gauge(
            f"{namespace}_prefill_active_requests",
            "Requests waiting for or running prefill",
            registry=self.registry,
        )
        self.decode_active = Gauge(
            f"{namespace}_decode_active_requests",
            "Requests eligible for decode scheduling",
            registry=self.registry,
        )
        self.decode_batch = Histogram(
            f"{namespace}_decode_batch_size",
            "Decode requests processed per scheduler step",
            registry=self.registry,
        )
        self.prefill_steps = Counter(
            f"{namespace}_prefill_steps_total",
            "Prefill chunks executed by the engine",
            registry=self.registry,
        )
        self.decode_steps = Counter(
            f"{namespace}_decode_steps_total",
            "Decode scheduler steps executed by the engine",
            registry=self.registry,
        )
        self.decode_tokens = Counter(
            f"{namespace}_decode_tokens_total",
            "Generated decode tokens emitted by the engine",
            registry=self.registry,
        )
        self.prefix_reuse_attempts = Counter(
            f"{namespace}_prefix_reuse_attempts_total",
            "Requests that attempted prefix reuse lookup",
            registry=self.registry,
        )
        self.prefix_cache_hits = Counter(
            f"{namespace}_prefix_cache_hits_total",
            "Prefix snapshot cache hits",
            registry=self.registry,
        )
        self.prefix_cache_misses = Counter(
            f"{namespace}_prefix_cache_misses_total",
            "Prefix snapshot cache misses",
            registry=self.registry,
        )
        self.prefix_tokens_reused = Counter(
            f"{namespace}_prefix_tokens_reused_total",
            "Prompt tokens skipped due to prefix reuse",
            registry=self.registry,
        )
        self.snapshot_bytes = Gauge(
            f"{namespace}_snapshot_bytes",
            "Bytes retained by the prefix snapshot store",
            registry=self.registry,
        )
        self.snapshot_entries = Gauge(
            f"{namespace}_snapshot_entries",
            "Entries retained by the prefix snapshot store",
            registry=self.registry,
        )
        self.cancellations = Counter(
            f"{namespace}_request_cancellations_total",
            "Cancelled requests",
            registry=self.registry,
        )
        self.admission_rejections = Counter(
            f"{namespace}_admission_rejections_total",
            "Requests rejected during admission control",
            registry=self.registry,
        )
        self.queue_wait_latency = Histogram(
            f"{namespace}_queue_wait_latency_seconds",
            "Time from submission to admission by the engine",
            registry=self.registry,
        )
        self.worker_restarts = Counter(
            f"{namespace}_worker_restarts_total",
            "Legacy worker restart metric retained for compatibility",
            registry=self.registry,
        )
        self.errors = Counter(
            f"{namespace}_errors_total",
            "Error count",
            ["code"],
            registry=self.registry,
        )
        self.responses_store_hits = Counter(
            f"{namespace}_responses_store_hits_total",
            "Responses replay store hits for previous_response_id lookups",
            registry=self.registry,
        )
        self.responses_store_misses = Counter(
            f"{namespace}_responses_store_misses_total",
            "Responses replay store misses for previous_response_id lookups",
            registry=self.registry,
        )
        self.responses_store_writes = Counter(
            f"{namespace}_responses_store_writes_total",
            "Responses replay histories written to the in-memory store",
            registry=self.registry,
        )
        self.responses_store_evictions = Counter(
            f"{namespace}_responses_store_evictions_total",
            "Responses replay histories evicted from the in-memory store",
            registry=self.registry,
        )
        self.responses_store_entries = Gauge(
            f"{namespace}_responses_store_entries",
            "Responses replay histories currently retained in the in-memory store",
            registry=self.registry,
        )
        self.tool_executions = Counter(
            f"{namespace}_tool_executions_total",
            "Tool calls executed by the interaction loop",
            ["tool_name", "status"],
            registry=self.registry,
        )
        self.tool_execution_latency = Histogram(
            f"{namespace}_tool_execution_latency_seconds",
            "Tool execution latency observed by the interaction loop",
            ["tool_name", "status"],
            registry=self.registry,
        )

    def render(self) -> bytes:
        return generate_latest(self.registry)
