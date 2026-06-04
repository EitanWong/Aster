# Aster vs vllm-mlx Compatibility Matrix

状态定义：

- `PASS`：本轮源码审查和黑盒运行都证明该子能力足以替代参考实现的对应行为。
- `PARTIAL`：存在实现或部分验证，但能力范围、边界、性能或协议细节不足。
- `MISSING`：不存在可用实现或端点。
- `INCOMPATIBLE`：存在实现但行为/响应格式与参考实现不兼容。
- `UNKNOWN`：仍缺少足够运行验证。
- `OUT_OF_SCOPE`：确认不需要替代的参考能力。本轮没有把任何 vllm-mlx 核心能力判为 out of scope。

## 执行证据摘要

本轮执行了三类验证：

- Aster 单元/接口测试：`252 passed, 2 warnings in 10.35s`。
- vllm-mlx 参考测试：`84 passed, 1 deselected in 40.64s`。
- 双服务黑盒轻量对照：`compat-results/aster-vllm-mlx-compare-0.8b.json`，0.8B 本地模型，Aster `18080`，vllm-mlx `18000`，vllm-mlx 以 `--continuous-batching --enable-metrics` 启动。
- 本轮追加验收体系改造：`tools/compat/aster_vllm_mlx_compare.py --include-lifecycle` 支持 timeout、stream 提前断开、未知 cancel 探针；`tests/test_compat_harness.py` 给检查逻辑加单元保护。该项是 harness 证据，不等价于真实双服务 lifecycle 已通过。
- 启动入口修正：Aster 现在支持 `aster serve MODEL` 风格启动，并解析 vllm-mlx serve 的主要参数；`--config` 仍作为旧部署兼容路径保留。验证：`tests/test_cli.py`，`python -m aster serve --help`。
- 本轮调度改造：Aster scheduler step 改为 cancellation/decode/prefill/admission 顺序；后续按 vllm-mlx 语义修正 admission policy，waiting 请求会填满可用 active slots，`prefill_batch_size` 只作为 prefill 批量/预算相关配置暴露；验证：`tests/test_engine_runtime.py::test_scheduler_step_prioritizes_decode_before_new_admissions`、`tests/test_engine_runtime.py::test_drain_submissions_fills_available_active_slots`。
- 本轮验收工具扩展：`tools/compat/aster_vllm_mlx_compare.py --include-mixed-scheduling` 新增长 prefill + 短请求混合 workload；`tests/test_compat_harness.py` 保护 short request p50/p95/p99 汇总。该项是验收能力，不等价于真实双服务调度已通过。
- 本轮真实双服务 mixed scheduling smoke：`compat-results/aster-vllm-mlx-mixed-scheduling-0.8b-fixed.json`。Aster long 2/2 成功，short p95/p99 `2.473s/2.487s`；vllm-mlx long 2/2 成功，short p95/p99 `1.721s/1.735s`。
- 本轮默认 thinking 对齐后真实双服务 mixed scheduling smoke：`compat-results/aster-vllm-mlx-mixed-scheduling-0.8b-thinking-default.json`。Aster 与 vllm-mlx 的 basic chat 均为 prompt/completion/total `28/8/36`、finish_reason `length`；mixed workload 中 long/short completion tokens 也对齐为 `16/96`。Aster short p95/p99 `3.037s/3.045s`，vllm-mlx `1.761s/1.783s`。
- 本轮 prefill continuation fairness 改造：长 prefill chunk 未完成且有新 admission 时，continuation 会让出一个 prefill turn，并通过 `prefill_yield_rotations` 暴露；单测通过，但真实 smoke `compat-results/aster-vllm-mlx-mixed-scheduling-0.8b-prefill-yield.json` 未改善尾延迟，Aster short p95/p99 `3.107s/3.116s`，vllm-mlx `1.968s/1.976s`。
- 本轮 decode batch 诊断：Aster `/v1/status` 暴露 `decode_batch_diagnostics`，mixed scheduling harness 记录 status 前后快照。真实 smoke `compat-results/aster-vllm-mlx-mixed-scheduling-0.8b-decode-diagnostics.json` 显示 mixed 后 Aster `batch_attempts=17`、`batch_successes=17`、`batch_fallbacks=0`，但 short p95/p99 仍为 `3.129s/3.134s`，vllm-mlx 为 `1.712s/1.713s`。
- 本轮 engine timing 诊断：Aster `/v1/status` 新增 `engine_timing`，记录服务级 prefill/decode 秒数、tokens、TPS、平均 decode batch size。真实 smoke `compat-results/aster-vllm-mlx-mixed-scheduling-0.8b-engine-timing.json` 显示 Aster `prompt_tps=2741.634`、`generation_tps=92.88`、`avg_decode_batch_size=1.246`，vllm-mlx status 为 `prompt_tps=1110.51`、`generation_tps=96.41`；但 Aster short p95/p99 仍 `2.519s/2.521s`，vllm-mlx `1.631s/1.636s`。
- 本轮请求级 timeline 诊断：Aster `/v1/status` 新增 `recent_request_timelines`，mixed scheduling harness 对 mixed 请求设置稳定 `X-Request-Id`。真实 smoke `compat-results/aster-vllm-mlx-mixed-scheduling-0.8b-request-timeline.json` 显示 Aster long/short `1/4` 全成功，short p95/p99 `2.707s/2.710s`，vllm-mlx `2.399s/2.405s`；Aster 短请求 `queue_wait_s` 为 `1.329s/1.533s/1.722s/2.102s`，而短请求 prefill wall 仅 `0.027s/0.026s/0.030s/0.064s`，decode duration 为 `1.038s/0.652s/0.666s/0.204s`。当前瓶颈更明确地落在 admission 前等待/active slot 占用，而不是单次 short prefill 或 batch fallback。
- 本轮 admission policy 修正：vllm-mlx scheduler 每步将 waiting 请求填入 running 直到 `max_num_seqs`，Aster 对齐为 `admission_policy=fill_available_active_slots`。真实 smoke `compat-results/aster-vllm-mlx-mixed-scheduling-0.8b-admission-fill.json` 显示 Aster short p95/p99 从上一轮 `2.707s/2.710s` 降到 `2.475s/2.477s`，前三个短请求 `queue_wait_s` 降到约 `1.01-1.12s`，但第 4 个短请求仍因 active slot 上限等待 `1.925s`；同轮 vllm-mlx short p95/p99 `1.455s/1.461s`，所以 continuous batching 仍未关闭。

## 矩阵

| 基准能力 | Aster 当前状态 | 状态 | 验证证据 | 替代性影响 |
|---|---|---:|---|---|
| 服务启动方式与配置 | Aster 已支持 `aster serve MODEL`，并映射 `--host/--port/--served-model-name/--max-num-seqs/--prefill-batch-size/--completion-batch-size/--max-request-tokens/--timeout/--api-key/--rate-limit/--stream-interval/--disable-prefix-cache/--embedding-model` 等常用 vllm-mlx serve 参数；`--max-request-tokens` 不再误降模型 context length；`aster serve` 默认启用 thinking，且可用 `--default-chat-template-kwargs '{"enable_thinking": false}'` 显式关闭；`--config` 保留为旧路径。部分 vllm-mlx 参数目前只解析并记录 warning，真实能力仍由对应 gap 跟踪。 | PARTIAL | `aster/__main__.py`；`tests/test_cli.py`；`python -m aster serve --help`；mixed scheduling long prompt 修复结果；默认 thinking 对齐结果；vllm CLI 见 `examples/vllm-mlx/vllm_mlx/cli.py:959-1388` | 基础启动命令已能按 vllm-mlx 方式迁移；`models-config`、lazy load、continuous batching、paged/SSD/KV quant、MCP/rerank/multimodal 等启动参数仍需真实后端能力闭合。 |
| 基础 health/models/cache stats | 端点存在；轻量黑盒均 200；但 health/cache stats 字段结构不同。 | PARTIAL | `aster/api/routes.py:69-76`；黑盒 `health/models/cache_stats` 通过；health payload 为 `status=ok`，vllm 为 `status=healthy`。 | 简单探活可用，依赖 vllm 字段的监控不兼容。 |
| `/v1/chat/completions` 基础文本 non-stream | 短文本请求可返回 OpenAI chat shape、usage、choices；vllm 风格 `aster serve` 默认 thinking 已与 vllm-mlx 对齐。 | PASS | `aster/api/routes.py:222-340`；黑盒 `chat_nonstream` 双方 200 且必需字段通过；`compat-results/aster-vllm-mlx-mixed-scheduling-0.8b-thinking-default.json` 中双方 prompt/completion/total 为 `28/8/36`、finish_reason `length`。 | 基础文本 chat 客户端可初步替代。 |
| `/v1/chat/completions` 基础文本 stream | SSE 可输出 chunk 并以 `[DONE]` 结束。 | PASS | `aster/api/routes.py:281-295`；黑盒 `chat_stream` 双方 200、terminal done、Aster event_count=10、vllm=11。 | 基础流式 chat 可初步替代。 |
| `/v1/chat/completions` 高级参数 | tools/response_format 会走 provider emulation；多模态内容在本地 runtime 报 `multimodal_not_supported`。 | PARTIAL | `aster/api/routes.py:223-234`；`aster/api/routes.py:591-599`；tool/structured 单测通过。 | 工具/结构化有实现，但不是 vllm-mlx 原生 parser/约束解码等价。 |
| `/v1/completions` 基础文本 non-stream/stream | 基础 prompt、stream、usage 可用；黑盒通过。 | PASS | `aster/api/routes.py:107`；黑盒 `completion_nonstream` 和 `completion_stream` 双方 200、stream `[DONE]`。 | 传统 completions 基础场景可替代。 |
| `/v1/responses` 基础文本 | 端点存在，黑盒 non-stream/stream lifecycle 均通过；但 parser/previous_response/tool/reasoning 子集仍需更深覆盖。 | PARTIAL | `aster/api/routes.py:645-649`；黑盒 `responses_nonstream`、`responses_stream` 双方 200 且 terminal `response.completed`。 | 基础 responses 可用，不能证明完整替代。 |
| Anthropic `/v1/messages` | 端点存在，经 provider gateway 转换；`/v1/messages/count_tokens` 已补充，并复用 engine tokenizer 计数。 | PARTIAL | `aster/api/routes.py`；`tests/test_vllm_mlx_surface_compat.py`。 | Anthropic 客户端部分可用；messages 深层 thinking/tool_use 兼容性仍需黑盒覆盖。 |
| `/v1/embeddings` | 端点存在；Aster 本轮临时黑盒配置关闭 embeddings；单测覆盖 backend。没有 vllm-mlx hot-swap/locked model 等全部语义证明。 | PARTIAL | `aster/api/routes.py:465-488`；`tests/test_embedding_backends.py` passed；vllm baseline `tests/test_embeddings.py`。 | embedding 可作为候选能力，未达到替代证明。 |
| `/v1/rerank` | 路由已存在，校验 query/documents/top_n，并在未配置 reranker 时返回 vllm-mlx 风格 404；尚无 reranker scoring backend。 | PARTIAL | `tests/test_vllm_mlx_surface_compat.py`。 | 端点探测不再是 Not Found；真实 rerank 仍不能替代。 |
| MCP endpoints | `/v1/mcp/tools`、`/v1/mcp/servers`、`/v1/mcp/execute` 已存在；未配置时 tools/servers 返回空状态，execute 返回 503；尚无 MCP manager/executor。 | PARTIAL | `tests/test_vllm_mlx_surface_compat.py`；vllm source `server.py:3664-3735`。 | MCP 发现路径兼容性改善；真实工具生态仍不能替代。 |
| Audio transcription/speech/voices | Aster 有 transcriptions/speech/voices；voices 返回默认或 runtime voices；默认配置关闭；音频生成/识别兼容性未完整证明。 | PARTIAL | `tests/test_vllm_mlx_surface_compat.py`；`aster/api/routes.py`。 | 音频仍不是完整替代，但 voices 端点不再缺失。 |
| 多模态 text/image/video/audio | Aster 本地文本 runtime 明确拒绝 multimodal content；ModelRunner 仅 `mlx_lm.load` 文本模型。 | MISSING | `aster/api/routes.py:591-599`；`aster/inference/model_runner.py:765-803`；vllm content parts `api/models.py:42-60`。 | vllm-mlx 的 MLLM 使用方式无法替代。 |
| 请求取消/超时/断连 | Aster 有 cancel endpoint、stream timeout/disconnect 单测；非流式 `client_disconnected` 已对齐 vllm-mlx 空 499；harness 已支持 timeout 504、stream 提前断开、未知 cancel 404 探针；尚未做完整真实双服务 lifecycle 矩阵。 | PARTIAL | `aster/api/routes.py`；`tests/test_vllm_mlx_lifecycle_compat.py`；`tests/test_api_disconnect.py`；`tests/test_compat_harness.py`；`tools/compat/aster_vllm_mlx_compare.py --include-lifecycle`。 | 断连语义和验收覆盖改善；生产生命周期兼容性仍未完整证明。 |
| 并发调度 | Aster 自研 prefill/decode queue；decode 可 merge/extract batch；调度顺序为 cancellation/decode/prefill/admission；waiting 请求现在会填满可用 active slots；长 prefill continuation 在新 admission 后让出一个 prefill turn，并暴露 `prefill_yield_rotations`；runner 现在暴露 batch decode attempts/success/fallback、服务级 prefill/decode timing 和最近完成请求 timeline。 | PARTIAL | `aster/inference/engine.py`；`aster/inference/model_runner.py`；`tests/test_engine_runtime.py::test_drain_submissions_fills_available_active_slots`；`tests/test_engine_runtime.py::test_prefill_continuation_yields_to_new_admissions`；`tests/test_engine_runtime.py::test_engine_batches_decode_steps_for_concurrent_requests`；`tests/test_model_runner.py::test_decode_batch_fallback_preserves_per_item_failures`；`compat-results/aster-vllm-mlx-mixed-scheduling-0.8b-admission-fill.json`。 | Admission policy 已按 vllm-mlx 语义修正，并实测降低 Aster short p95；但 vllm-mlx 同轮仍明显更快，说明 active slot 上限、长 prefill/short decode 交错和 BatchGenerator 架构差距仍未关闭。 |
| Continuous batching | Aster 已有 decode batching、chunked prefill、decode-first admission、prefill continuation yielding 和状态可观测 scheduler 字段；默认 thinking 对齐后，mixed workload 的 long/short completion tokens 已与 vllm-mlx 对齐为 `16/96`，但 Aster short p95/p99 `3.037s/3.045s`，慢于 vllm-mlx `1.761s/1.783s`；prefill-yield 改造后 smoke 为 Aster `3.107s/3.116s` vs vllm-mlx `1.968s/1.976s`，未证明改善；decode diagnostics smoke 显示 Aster `batch_attempts=17`、`batch_successes=17`、`batch_fallbacks=0`；engine timing smoke 显示 Aster `prompt_tps=2741.634`、`generation_tps=92.88` 接近 vllm-mlx `generation_tps=96.41`，但 Aster short p95/p99 仍 `2.519s/2.521s` vs vllm-mlx `1.631s/1.636s`；request timeline smoke 显示 Aster short `queue_wait_s` 高达 `1.329s` 到 `2.102s`；admission-fill 修正后 Aster short p95/p99 降到 `2.475s/2.477s`，但同轮 vllm-mlx 为 `1.455s/1.461s`，manual kernel 仍标记 `runtime_kernel_continuous_batching=False`，`batch_generator` adapter 仍 `available=False`。 | PARTIAL | `aster/inference/engine.py`；`aster/inference/runtime_kernel.py`；`tests/test_engine_runtime.py`；`tests/test_model_runner.py`；`tests/test_compat_harness.py`；`compat-results/aster-vllm-mlx-mixed-scheduling-0.8b-thinking-default.json`；`compat-results/aster-vllm-mlx-mixed-scheduling-0.8b-prefill-yield.json`；`compat-results/aster-vllm-mlx-mixed-scheduling-0.8b-decode-diagnostics.json`；`compat-results/aster-vllm-mlx-mixed-scheduling-0.8b-engine-timing.json`；`compat-results/aster-vllm-mlx-mixed-scheduling-0.8b-request-timeline.json`；`compat-results/aster-vllm-mlx-mixed-scheduling-0.8b-admission-fill.json`。 | P0 阻塞项已排除“batch decode 直接回退到逐请求执行”和“纯 runner TPS 明显不足”两个假设，并关闭了一个 admission policy 误差；真实同 token 负载下的尾延迟差距仍未关闭，仍不能判定为 vllm-mlx 等价 continuous batching。 |
| Prefix cache | Aster PrefixStore 支持 exact/prefix/LCP stats、持久化；黑盒重复前缀有 stats；但不是 vllm memory-aware/paged/SSD cache 栈。 | PARTIAL | `aster/inference/prefix_store.py:97-261`；黑盒 `cache_stats` Aster hit_rate=0.5。 | 有前缀复用基础，不能替代完整 KV cache 行为。 |
| Paged KV cache | 未发现 Aster paged KV block cache 实现。 | MISSING | Aster `CacheSettings` 有 `kv_page_tokens` 字段但 runtime 未实现等价 block manager；vllm `paged_cache.py:1-225`。 | 长上下文/高并发内存效率缺口。 |
| SSD tiered KV cache | 未发现 Aster SSD KV tier；只有 prefix cache persist path。 | MISSING | `aster/core/config.py:54-58`；vllm `ssd_cache.py:1-211`。 | 冷热分层缓存能力缺失。 |
| KV cache quantization | 未发现 Aster KV cache 量化存储路径。 | MISSING | vllm CLI `--kv-cache-quantization` at `cli.py:1030-1054`；Aster config 无对应 runtime。 | 大上下文内存控制缺失。 |
| 模型注册/lazy/multi-model/idle unload | Aster 单主模型配置；无 vllm registry lease/eviction/lazy 多模型服务。 | MISSING | `aster/core/config.py:25-30`；vllm `model_registry.py:1-220`。 | 多模型替代不可用。 |
| Sampling/stop/usage | Aster schema 和 runner 支持 temperature/top_p/top_k/min_p、penalty、stop、usage；基础黑盒使用 temperature/max_tokens/usage 通过。 | PARTIAL | `aster/api/schemas.py:28-76`；`aster/inference/model_runner.py:175-243`；黑盒基础 usage 通过。 | 常用参数可用，边界语义需补齐对照测试。 |
| Tool calling | Aster 有 provider emulation、tool parser 单测；未覆盖 vllm-mlx 全 parser 列表与原生 auto tool choice 行为。 | PARTIAL | `aster/api/routes.py:752-818`；`tests/test_tool_parsers.py` passed；vllm CLI parser list `cli.py:1261-1297`。 | agent 工具调用兼容性不足。 |
| Structured output | Aster 有 schema/constrained decoding 单测；未与 vllm-mlx json_schema/json_object 逐项黑盒比对。 | PARTIAL | `tests/test_structured_schema.py`、`tests/test_constrained_decoding.py` passed；vllm `api/models.py:118-142`。 | 可用但未证明等价。 |
| Reasoning/thinking | Aster 支持 `enable_thinking`、thinking budget 和 reasoning parsing；`aster serve` 默认 thinking 已对齐 vllm-mlx 非 coder 模型默认；未覆盖 vllm reasoning parser registry。 | PARTIAL | `aster/api/routes.py:246-279`；`tests/test_cli.py`；`compat-results/aster-vllm-mlx-mixed-scheduling-0.8b-thinking-default.json`；vllm CLI `cli.py:1298-1312`。 | 默认 chat template 行为改善；reasoning parser 字段兼容性仍需补黑盒。 |
| Error format | 状态码可相同，但 body 不兼容：Aster 是 OpenAI-like `error.type/code/message/details`；vllm-mlx FastAPI `detail`。 | INCOMPATIBLE | 黑盒 excessive max_tokens：双方 400；Aster `{"error":...}`，vllm `{"detail":"max_tokens exceeds..."}`。 | 错误处理客户端会出现兼容风险。 |
| Metrics | Aster `/metrics` 存在，指标命名/开关语义不同；vllm 默认 disabled，`--enable-metrics` 开启。 | INCOMPATIBLE | `aster/api/routes.py:71`；`aster/api/middleware.py:14`；vllm `cli.py:1233-1237`。 | Prometheus dashboard/alerts 不能直接替代。 |
| Benchmark/验收工具 | Aster 有自有 benchmark scripts；黑盒 compare harness 已覆盖基础协议、性能 smoke、prefix observation，并新增 lifecycle 探针；没有 vllm-mlx `bench-serve` 等价。 | PARTIAL | `scripts/dev/benchmark_live.py`；`tools/compat/aster_vllm_mlx_compare.py`；`tests/test_compat_harness.py`。 | 可开始验收，但体系仍未覆盖全部基准能力，且 lifecycle 探针仍需真实双服务结果。 |

## 本轮轻量性能对照

配置：本地 `Qwen3.5-0.8B-4bit`，`max_tokens=8`，每档 3 请求，concurrency 1/2。该结果只用于发现明显差距，不足以作为最终性能验收。

| 实现 | concurrency | 成功/总数 | avg latency | p95 latency | completion tps |
|---|---:|---:|---:|---:|---:|
| Aster | 1 | 3/3 | 0.317s | 0.326s | 22.08 |
| Aster | 2 | 3/3 | 0.487s | 0.586s | 24.02 |
| vllm-mlx | 1 | 3/3 | 0.122s | 0.134s | 65.57 |
| vllm-mlx | 2 | 3/3 | 0.107s | 0.118s | 115.06 |

进程 RSS 快照：Aster 约 `1009792 KB`，vllm-mlx 约 `1245472 KB`。
