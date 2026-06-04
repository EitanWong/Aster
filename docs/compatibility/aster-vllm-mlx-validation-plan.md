# Aster/vllm-mlx Validation Plan

目标：每轮 Aster 改造后，用同一套命令重新确认源码测试、黑盒协议一致性、性能和稳定性是否真正缩小差距。

## 1. 源码与单元测试复跑

Aster 当前验证命令：

```bash
.venv/bin/pytest \
  tests/test_cli.py \
  tests/test_api.py \
  tests/test_engine_runtime.py \
  tests/test_cache.py \
  tests/test_prefix_cache.py \
  tests/test_embedding_backends.py \
  tests/test_streaming.py \
  tests/test_tool_parsers.py \
  tests/test_structured_schema.py \
  tests/test_constrained_decoding.py -q
```

本轮结果：`252 passed, 2 warnings in 10.35s`。

调度切面回归：

```bash
.venv/bin/pytest \
  tests/test_engine_runtime.py::test_scheduler_step_prioritizes_decode_before_new_admissions \
  tests/test_engine_runtime.py::test_drain_submissions_respects_prefill_batch_size \
  tests/test_engine_runtime.py::test_engine_batches_decode_steps_for_concurrent_requests -q
```

该组验证 Aster 目前的部分 continuous scheduling 能力：decode-first step、`batch.prefill_batch_size` admission throttle、decode batch 合并。

vllm-mlx 参考测试命令：

```bash
PYTHONPATH=examples/vllm-mlx .venv/bin/pytest \
  examples/vllm-mlx/tests/test_responses_api.py \
  examples/vllm-mlx/tests/test_embeddings.py \
  examples/vllm-mlx/tests/test_rerank.py \
  examples/vllm-mlx/tests/test_metrics.py \
  examples/vllm-mlx/tests/test_lifecycle_cli.py -q
```

本轮结果：`84 passed, 1 deselected in 40.64s`。

## 2. 双服务黑盒启动

推荐使用 vllm-mlx 风格启动 Aster。轻量 smoke 的等价启动命令：

```bash
PYTHONPATH=. .venv/bin/python -m aster \
  serve /Users/eitan/Documents/Projects/Python/Aster/models/qwen3.5-0.8b-mlx/Qwen3.5-0.8B-4bit \
  --host 127.0.0.1 \
  --port 18080 \
  --max-num-seqs 8 \
  --prefill-batch-size 2 \
  --completion-batch-size 4 \
  --max-request-tokens 2048
```

旧的 `--config compat-results/aster-compat-0.8b.yaml` 路径仍可用于回归旧部署，但不再作为替代 vllm-mlx 的首选启动方式。

```bash
PYTHONPATH=examples/vllm-mlx .venv/bin/python -m vllm_mlx.cli serve \
  /Users/eitan/Documents/Projects/Python/Aster/models/qwen3.5-0.8b-mlx/Qwen3.5-0.8B-4bit \
  --host 127.0.0.1 \
  --port 18000 \
  --max-tokens 64 \
  --max-request-tokens 2048 \
  --continuous-batching \
  --max-num-seqs 8 \
  --prefill-batch-size 2 \
  --completion-batch-size 4 \
  --enable-metrics
```

正式 9B 对照应使用同一模型、同一 max request tokens、同一请求集，并打开 vllm-mlx 的目标能力开关，例如 prefix cache、metrics、rerank/embedding/audio/MCP 按场景逐项启用。

## 3. 黑盒一致性 harness

新增脚本：

```bash
.venv/bin/python tools/compat/aster_vllm_mlx_compare.py --help
```

本轮实际执行：

```bash
.venv/bin/python tools/compat/aster_vllm_mlx_compare.py \
  --aster-url http://127.0.0.1:18080 \
  --vllm-url http://127.0.0.1:18000 \
  --model /Users/eitan/Documents/Projects/Python/Aster/models/qwen3.5-0.8b-mlx/Qwen3.5-0.8B-4bit \
  --max-tokens 8 \
  --requests 3 \
  --concurrency 1 2 \
  --skip-embeddings \
  --out compat-results/aster-vllm-mlx-compare-0.8b.json
```

输出：`compat-results/aster-vllm-mlx-compare-0.8b.json`。

覆盖项：

- `/health`
- `/v1/models`
- `/v1/cache/stats`
- `/v1/chat/completions` non-stream/stream
- `/v1/completions` non-stream/stream
- `/v1/responses` non-stream/stream
- excessive `max_tokens` 错误行为
- `/v1/rerank` 端点探测
- `/v1/mcp/tools` 端点探测
- prefix cache 可观察 stats
- concurrency 1/2 的短请求吞吐

新增 lifecycle 扩展（本轮完成 harness 与单元保护，仍需在双服务真实运行时执行）：

```bash
.venv/bin/python tools/compat/aster_vllm_mlx_compare.py \
  --aster-url http://127.0.0.1:18080 \
  --vllm-url http://127.0.0.1:18000 \
  --model /path/to/the/same/model \
  --max-tokens 32 \
  --requests 30 \
  --concurrency 1 2 4 8 16 \
  --include-lifecycle \
  --out compat-results/aster-vllm-mlx-compare-lifecycle.json
```

`--include-lifecycle` 追加验证：chat/completion request timeout 504、stream 提前断开、未知请求 cancel 404。

Continuous scheduling / 长 prefill 混合短请求验证：

```bash
.venv/bin/python tools/compat/aster_vllm_mlx_compare.py \
  --aster-url http://127.0.0.1:18080 \
  --vllm-url http://127.0.0.1:18000 \
  --model /path/to/the/same/model \
  --max-tokens 32 \
  --skip-embeddings \
  --include-mixed-scheduling \
  --mixed-runs 3 \
  --mixed-long-prompt-words 4096 \
  --mixed-long-requests 1 \
  --mixed-short-requests 16 \
  --mixed-short-start-delay 0.05 \
  --mixed-short-interval 0.05 \
  --out compat-results/aster-vllm-mlx-mixed-scheduling.json
```

该 workload 在每轮中先发长 prompt 请求，再插入短请求，记录 short request p50/p95/p99、成功率和 completion tokens，用于发现长 prefill 对短 decode 的阻塞。

本轮实际 smoke 结果：

```bash
.venv/bin/python tools/compat/aster_vllm_mlx_compare.py \
  --aster-url http://127.0.0.1:18080 \
  --vllm-url http://127.0.0.1:18000 \
  --model /Users/eitan/Documents/Projects/Python/Aster/models/qwen3.5-0.8b-mlx/Qwen3.5-0.8B-4bit \
  --timeout 180 \
  --max-tokens 8 \
  --requests 2 \
  --concurrency 1 \
  --skip-embeddings \
  --skip-performance \
  --skip-prefix-observation \
  --include-mixed-scheduling \
  --mixed-runs 2 \
  --mixed-long-prompt-words 512 \
  --mixed-long-requests 1 \
  --mixed-short-requests 6 \
  --mixed-short-start-delay 0.05 \
  --mixed-short-interval 0.05 \
  --out compat-results/aster-vllm-mlx-mixed-scheduling-0.8b-thinking-default.json
```

结果摘要：Aster long 2/2 成功，short p95/p99 `3.037s/3.045s`；vllm-mlx long 2/2 成功，short p95/p99 `1.761s/1.783s`。Aster 已修复 `--max-request-tokens` 被误映射为 context length 导致 long prompt 400 的问题，并把 `aster serve` 的默认 thinking 行为对齐到 vllm-mlx；新的 mixed records 包含 `finish_reason`、`prompt_tokens`、`completion_tokens` 和 `content_preview`，可证明本轮性能差距是在同 token/finish 负载下发生的。

Prefill continuation yielding 改造后复跑：

```bash
.venv/bin/python tools/compat/aster_vllm_mlx_compare.py \
  --aster-url http://127.0.0.1:18080 \
  --vllm-url http://127.0.0.1:18000 \
  --model /Users/eitan/Documents/Projects/Python/Aster/models/qwen3.5-0.8b-mlx/Qwen3.5-0.8B-4bit \
  --timeout 180 \
  --max-tokens 8 \
  --requests 2 \
  --concurrency 1 \
  --skip-embeddings \
  --skip-performance \
  --skip-prefix-observation \
  --include-mixed-scheduling \
  --mixed-runs 2 \
  --mixed-long-prompt-words 512 \
  --mixed-long-requests 1 \
  --mixed-short-requests 6 \
  --mixed-short-start-delay 0.05 \
  --mixed-short-interval 0.05 \
  --out compat-results/aster-vllm-mlx-mixed-scheduling-0.8b-prefill-yield.json
```

结果摘要：Aster long/short completion tokens `16/96`，short p95/p99 `3.107s/3.116s`；vllm-mlx `16/96`，short p95/p99 `1.968s/1.976s`。该改造关闭了一个单元层队列公平性问题，但没有证明真实 mixed workload 尾延迟改善。

Decode batch 诊断扩展后复跑：

```bash
.venv/bin/python tools/compat/aster_vllm_mlx_compare.py \
  --aster-url http://127.0.0.1:18080 \
  --vllm-url http://127.0.0.1:18000 \
  --model /Users/eitan/Documents/Projects/Python/Aster/models/qwen3.5-0.8b-mlx/Qwen3.5-0.8B-4bit \
  --timeout 180 \
  --max-tokens 8 \
  --requests 2 \
  --concurrency 1 \
  --skip-embeddings \
  --skip-performance \
  --skip-prefix-observation \
  --include-mixed-scheduling \
  --mixed-runs 1 \
  --mixed-long-prompt-words 512 \
  --mixed-long-requests 1 \
  --mixed-long-max-tokens 8 \
  --mixed-short-requests 4 \
  --mixed-short-max-tokens 8 \
  --mixed-short-start-delay 0.05 \
  --mixed-short-interval 0.05 \
  --mixed-run-gap 0.25 \
  --out compat-results/aster-vllm-mlx-mixed-scheduling-0.8b-decode-diagnostics.json
```

结果摘要：Aster long/short `1/4` 全成功，short p95/p99 `3.129s/3.134s`；vllm-mlx `1/4` 全成功，short p95/p99 `1.712s/1.713s`。Aster status 快照显示 `decode_batch_diagnostics.batch_attempts=17`、`batch_successes=17`、`batch_fallbacks=0`，因此本轮 mixed workload 的尾延迟差距不能归因于 batch decode 失败后回退到逐请求执行。

Engine timing 诊断扩展后复跑：

```bash
.venv/bin/python tools/compat/aster_vllm_mlx_compare.py \
  --aster-url http://127.0.0.1:18080 \
  --vllm-url http://127.0.0.1:18000 \
  --model /Users/eitan/Documents/Projects/Python/Aster/models/qwen3.5-0.8b-mlx/Qwen3.5-0.8B-4bit \
  --timeout 180 \
  --max-tokens 8 \
  --requests 2 \
  --concurrency 1 \
  --skip-embeddings \
  --skip-performance \
  --skip-prefix-observation \
  --include-mixed-scheduling \
  --mixed-runs 1 \
  --mixed-long-prompt-words 512 \
  --mixed-long-requests 1 \
  --mixed-long-max-tokens 8 \
  --mixed-short-requests 4 \
  --mixed-short-max-tokens 8 \
  --mixed-short-start-delay 0.05 \
  --mixed-short-interval 0.05 \
  --mixed-run-gap 0.25 \
  --out compat-results/aster-vllm-mlx-mixed-scheduling-0.8b-engine-timing.json
```

结果摘要：Aster long/short `1/4` 全成功，short p95/p99 `2.519s/2.521s`；vllm-mlx `1/4` 全成功，short p95/p99 `1.631s/1.636s`。Aster status 快照显示 `prompt_tps=2741.634`、`generation_tps=92.88`、`avg_decode_batch_size=1.246`、`max_prefill_step_seconds=0.882226`；vllm-mlx status 显示 `prompt_tps=1110.51`、`generation_tps=96.41`。因此该轮差距不能简单归因于 Aster 纯 runner TPS 低，下一步需要记录 admission/decode/complete 的请求级时间线。

Request timeline 诊断扩展后复跑：

```bash
.venv/bin/python tools/compat/aster_vllm_mlx_compare.py \
  --aster-url http://127.0.0.1:18080 \
  --vllm-url http://127.0.0.1:18000 \
  --model /Users/eitan/Documents/Projects/Python/Aster/models/qwen3.5-0.8b-mlx/Qwen3.5-0.8B-4bit \
  --timeout 180 \
  --max-tokens 8 \
  --requests 2 \
  --concurrency 1 \
  --skip-embeddings \
  --skip-performance \
  --skip-prefix-observation \
  --include-mixed-scheduling \
  --mixed-runs 1 \
  --mixed-long-prompt-words 512 \
  --mixed-long-requests 1 \
  --mixed-long-max-tokens 8 \
  --mixed-short-requests 4 \
  --mixed-short-max-tokens 8 \
  --mixed-short-start-delay 0.05 \
  --mixed-short-interval 0.05 \
  --mixed-run-gap 0.25 \
  --out compat-results/aster-vllm-mlx-mixed-scheduling-0.8b-request-timeline.json
```

结果摘要：Aster long/short `1/4` 全成功，short p95/p99 `2.707s/2.710s`；vllm-mlx `1/4` 全成功，short p95/p99 `2.399s/2.405s`。Aster status 快照的 `recent_request_timelines` 显示短请求 `queue_wait_s` 为 `1.329s/1.533s/1.722s/2.102s`，短请求 prefill wall 仅 `0.027s/0.026s/0.030s/0.064s`。因此下一轮验证应优先比较 admission/active slot policy 改造前后的 short `queue_wait_s` 和 p95/p99。

Admission fill 修正后复跑：

```bash
.venv/bin/python tools/compat/aster_vllm_mlx_compare.py \
  --aster-url http://127.0.0.1:18080 \
  --vllm-url http://127.0.0.1:18000 \
  --model /Users/eitan/Documents/Projects/Python/Aster/models/qwen3.5-0.8b-mlx/Qwen3.5-0.8B-4bit \
  --timeout 180 \
  --max-tokens 8 \
  --requests 2 \
  --concurrency 1 \
  --skip-embeddings \
  --skip-performance \
  --skip-prefix-observation \
  --include-mixed-scheduling \
  --mixed-runs 1 \
  --mixed-long-prompt-words 512 \
  --mixed-long-requests 1 \
  --mixed-long-max-tokens 8 \
  --mixed-short-requests 4 \
  --mixed-short-max-tokens 8 \
  --mixed-short-start-delay 0.05 \
  --mixed-short-interval 0.05 \
  --mixed-run-gap 0.25 \
  --out compat-results/aster-vllm-mlx-mixed-scheduling-0.8b-admission-fill.json
```

结果摘要：Aster long/short `1/4` 全成功，short p95/p99 `2.475s/2.477s`；vllm-mlx `1/4` 全成功，short p95/p99 `1.455s/1.461s`。Aster status 快照显示 `admission_policy=fill_available_active_slots`，前三个短请求 `queue_wait_s` 降到约 `1.01-1.12s`，第 4 个短请求仍等待 `1.925s`。这证明 admission policy 修正有效，但 continuous batching 差距未关闭。

## 4. 必须补充的黑盒用例

当前 harness 是 smoke 级别，还需要扩展这些用例后才可用于 production verdict：

- chat/completion/responses 的所有 sampling 参数：`top_p/top_k/min_p/presence_penalty/frequency_penalty/repetition_penalty/stop/stop_token_ids`。
- 工具调用：每个 vllm-mlx parser 至少一个 positive 和一个 malformed case。
- structured output：`json_object`、`json_schema`、schema violation 修复/失败路径。
- Anthropic：`/v1/messages` stream/non-stream、thinking/tool_use block、`/v1/messages/count_tokens`。
- Embeddings：batch 顺序、empty input、model hot-swap/locked model。
- Rerank：加载 reranker 后验证排序、`top_n`、`return_documents=false`、非法 top_n。
- Audio：STT multipart、TTS bytes、voices、大小限制、禁用服务错误。
- Multimodal：image_url、video_url、audio_url、unsafe URL、预处理失败。
- 调度：长 prefill + 短请求混合 workload 已有 harness 探针，但尚未跑完整双服务结果。
- 生命周期：真实 non-stream client disconnect 499、stream heartbeat、异常后恢复；timeout 504、stream 提前断开、未知 cancel 404 已有 harness 探针但尚未跑完整双服务结果。
- Cache：warm prompts、prefix cache hit/miss、paged cache、SSD tier、KV quantization、clear cache 后恢复。
- Model registry：`--models-config`、lazy load、lease、eviction、idle unload。

## 5. 性能与稳定性验收

最低复跑矩阵：

- 模型：0.8B smoke、9B 目标模型；如目标替代 35B，也必须加入 35B。
- 并发：1、2、4、8、16。
- 请求：每档至少 30 次，报告 p50/p95/p99，不使用单次结果。
- 上下文：短 prompt、4K、8K、16K、重复前缀、长上下文混合流量。
- 混合调度：长 prefill 与短 decode 混合流量下，短 decode p95/p99 不能被新 admission/prefill 明显饿死。
- 流式：TTFT、chunk gap p95、是否断流。
- 稳定性：30 分钟循环，包含取消、非法请求、超时和服务恢复。
- 内存：启动后、峰值、清 cache 后、停止后 RSS/Metal memory。

建议记录：

```bash
ps -o pid,rss,vsz,pcpu,pmem,command -p <aster_pid>,<vllm_pid>
curl -s http://127.0.0.1:18080/metrics > compat-results/aster.metrics.txt
curl -s http://127.0.0.1:18000/metrics > compat-results/vllm-mlx.metrics.txt
```

关闭标准：所有 P0/P1 gap 必须有对应黑盒或性能测试，且结果文件可提交复核。
