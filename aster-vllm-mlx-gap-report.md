# Aster/vllm-mlx Gap Report

最终判定见 `aster-vllm-mlx-replaceability-verdict.md`。本文件只记录影响真实替代能力的未关闭差距。

## P0: 阻塞完整替代

| Gap | 为什么影响替代性 | vllm-mlx 依据 | Aster 当前状态 | 关闭标准 |
|---|---|---|---|---|
| Continuous batching 未达到 vllm-mlx 等价 | vllm-mlx 的主要 serving 模式支持多请求持续调度、prefill/completion batch、abort 和 stats；Aster 已有部分等价调度结构，但真实 smoke 在同 token 负载下仍显示短请求尾延迟落后。 | `vllm_mlx/engine/batched.py:1-12`；`vllm_mlx/scheduler.py:1961-2115`；`vllm_mlx/mllm_scheduler.py:514-565` | Aster 已有 decode batching、chunked prefill、decode-first scheduler step、prefill continuation yielding 和 scheduler status；默认 thinking 对齐后 mixed scheduling smoke：Aster/vllm-mlx long/short completion tokens 均为 `16/96`；Aster short p95/p99 `3.037s/3.045s`，vllm-mlx `1.761s/1.783s`。prefill-yield 改造后 smoke Aster `3.107s/3.116s`，vllm-mlx `1.968s/1.976s`，未证明改善。decode diagnostics 后，真实 smoke 显示 Aster `batch_attempts=17`、`batch_successes=17`、`batch_fallbacks=0`。engine timing smoke 显示 Aster `prompt_tps=2741.634`、`generation_tps=92.88`、`avg_decode_batch_size=1.246`，vllm-mlx `prompt_tps=1110.51`、`generation_tps=96.41`，但 Aster short p95/p99 仍 `2.519s/2.521s`，vllm-mlx `1.631s/1.636s`。request timeline smoke 显示 Aster short p95/p99 `2.707s/2.710s`，vllm-mlx `2.399s/2.405s`；Aster 短请求 `queue_wait_s` 为 `1.329s/1.533s/1.722s/2.102s`。本轮按 vllm-mlx 语义修正 admission fill 后，Aster short p95/p99 降到 `2.475s/2.477s`，前三个短请求 `queue_wait_s` 降到约 `1.01-1.12s`，但第 4 个仍等待 `1.925s`，同轮 vllm-mlx short p95/p99 `1.455s/1.461s`。`ManualRuntimeKernel.continuous_batching=False`，`BatchGeneratorRuntimeKernel.available=False`。 | Aster 在真实 MLX 服务中完成 vllm-mlx 目标范围的 continuous batching 等价能力；并发 1/2/4/8/16、长 prefill 混合短 decode、取消和异常恢复对照通过；状态/metrics 可观察，且 request timeline 证明短请求不会因 active slot/admission 策略长时间排队。 |
| Paged/memory-aware/SSD/KV quant cache 栈缺失 | vllm-mlx 的长上下文和高并发内存管理依赖多层 KV cache；Aster 只有 PrefixStore，不能替代 paged/SSD/quant 行为。 | `paged_cache.py:1-225`；`memory_cache.py:1-240`；`ssd_cache.py:1-211` | PrefixStore 支持 exact/prefix/LCP 和 persist；无 paged block manager、SSD tier、KV quantization。 | 对照测试覆盖重复前缀、长上下文、cache clear、内存上限、eviction；Aster stats 字段可解释，并达到目标内存/TTFT收益。 |
| 多模态能力缺失 | vllm-mlx 支持 text/image/video/audio content 和 MLLM scheduler；Aster 本地 runtime 明确拒绝 multimodal。 | `api/models.py:42-60`；`engine/batched.py:285-414` | `multimodal_not_supported`；ModelRunner 使用 `mlx_lm.load` 文本模型。 | image/video/audio 黑盒请求与 vllm-mlx 成功路径和错误路径对齐，含 stream、unsafe URL、预处理异常。 |
| API 表面仍未具备真实后端：MCP、rerank、audio | 完整替代不仅要求端点存在，还要求真实 MCP 执行、reranker scoring、音频模型行为可用且可验证。 | `server.py:3534-3735`；`server.py:5181-5266`；`server.py:3875-3885` | Aster 已补齐 `/v1/rerank` 未配置语义、`/v1/mcp/*` 空状态、`/v1/messages/count_tokens`、`/v1/audio/voices`；但 MCP/rerank 后端仍缺失，audio 未完整黑盒验证。 | 加载对应模型/配置后 positive/negative 黑盒通过；MCP tools/servers/execute 与 rerank scoring 结果结构对齐。 |
| 模型注册、lazy serving、idle unload 缺失 | vllm-mlx 可通过 `--models-config` 多模型服务、lazy load、lease/eviction；Aster 是单主模型配置。 | `model_registry.py:1-220`；`cli.py:959-966`；`cli.py:1238-1248` | 单 `model.name/path`；无 registry lease/eviction。 | 支持 registry 配置、模型列表、并发 lease、内存预算和 idle unload；黑盒验证模型切换/卸载/错误恢复。 |

## P1: 阻塞主要场景替代

| Gap | 为什么影响替代性 | vllm-mlx 依据 | Aster 当前状态 | 关闭标准 |
|---|---|---|---|---|
| 错误响应格式不兼容 | 状态码相同不足以兼容；客户端可能读取 `detail` 或 OpenAI `error`。 | 黑盒 vllm excessive max_tokens 返回 `{"detail":"..."}` | Aster 返回 `{"error":{"type","code","message","details"}}`。 | 为 vllm-mlx compatibility mode 明确错误 schema；所有非法请求黑盒断言通过。 |
| Request lifecycle 未完整黑盒验证 | cancel、disconnect、timeout、heartbeat、异常恢复是 serving 稳定性核心。 | `server.py:3893-4269` | Aster 已把非流式 `client_disconnected` 对齐为空 499，并有内部实现和单测；harness 已新增 timeout 504、stream 提前断开、未知 cancel 404 探针；仍缺真实双服务断连/504/恢复结果。 | `tools/compat/aster_vllm_mlx_compare.py --include-lifecycle` 覆盖显式 cancel、断开 stream、non-stream client disconnect、timeout、异常后继续请求，并在 Aster/vllm-mlx 双服务上通过。 |
| Tool calling/structured/reasoning 不等价 | vllm-mlx 支持大量 parser、auto tool choice、reasoning parser registry 和 JSON schema 输出。 | `cli.py:1261-1312`；`api/models.py:118-142` | Aster 有 provider emulation 和单测，但没有全 parser/真实生成闭环对照。 | 每个目标 parser 和 schema 类型都有 positive/malformed/stream 用例，响应字段与 vllm-mlx 对齐。 |
| 性能差距明显 | 轻量对照中 vllm-mlx 短请求吞吐约为 Aster 3x-4.8x；concurrency 2 下 vllm latency 下降而 Aster 上升。 | 本轮黑盒 `compat-results/aster-vllm-mlx-compare-0.8b.json` | Aster concurrency 1/2 completion tps 22.08/24.02；vllm-mlx 65.57/115.06。 | 目标模型下多轮基准达到替代门槛：吞吐、TTFT、p95 latency、内存、长跑稳定均达标。 |
| Metrics/monitoring 不兼容 | 生产替代需要 dashboard/alert 能复用或可明确迁移。 | `metrics.py:80-288` | Aster metrics namespace/字段不同，`/metrics` 公开语义不同。 | 兼容指标映射或迁移文档；Prometheus scrape 与核心告警测试通过。 |

## 本轮关闭/降级的差距

| Gap | 结果 | 证据 |
|---|---|---|
| `aster serve --max-request-tokens` 误降低模型 context length | 已修复。该参数现在只映射到 API 允许的生成 token 上限，不再覆盖 `model.context_length`。 | `tests/test_cli.py::test_serve_cli_max_request_tokens_does_not_reduce_context_length`；修复后 mixed long prompt Aster 2/2 成功。 |
| `aster serve` 默认 thinking 与 vllm-mlx 不一致 | 已修复。vllm 风格启动路径现在默认启用 thinking，并支持通过 `--default-chat-template-kwargs '{"enable_thinking": false}'` 显式关闭。 | `tests/test_cli.py::test_serve_cli_default_chat_template_kwargs_can_disable_thinking`；`compat-results/aster-vllm-mlx-mixed-scheduling-0.8b-thinking-default.json` 中 basic chat 双方 prompt/completion/total `28/8/36`、finish_reason `length`。 |
| 长 prefill continuation 在 admission 后继续排在短请求前 | 单元层已修复并暴露 `prefill_yield_rotations`；真实 smoke 未显示尾延迟收益，因此不关闭 P0 continuous batching gap。 | `tests/test_engine_runtime.py::test_prefill_continuation_yields_to_new_admissions`；`compat-results/aster-vllm-mlx-mixed-scheduling-0.8b-prefill-yield.json`。 |
| 真实 mixed workload 是否发生 batch decode fallback | 本轮已证伪该假设。Aster 现在在 `/v1/status` 暴露 `decode_batch_diagnostics`，harness 会记录 mixed 前后快照；真实 smoke 中 Aster `batch_attempts=17`、`batch_successes=17`、`batch_fallbacks=0`。 | `tests/test_model_runner.py::test_decode_batch_fallback_preserves_per_item_failures`；`tests/test_compat_harness.py::test_mixed_scheduling_probe_records_status_snapshots`；`compat-results/aster-vllm-mlx-mixed-scheduling-0.8b-decode-diagnostics.json`。 |
| 真实 mixed workload 是否由纯 runner TPS 不足造成 | 本轮证据显示不是主要解释。Aster status 新增 `engine_timing`；真实 smoke 中 Aster `prompt_tps=2741.634`、`generation_tps=92.88`，vllm-mlx status `generation_tps=96.41`，但 Aster 端到端 short p95 仍慢约 0.89s。 | `tests/test_engine_runtime.py::test_engine_batches_decode_steps_for_concurrent_requests`；`compat-results/aster-vllm-mlx-mixed-scheduling-0.8b-engine-timing.json`。 |
| mixed workload 的主要等待阶段未知 | 本轮已降级为更具体的 admission/active slot 问题。Aster status 新增 `recent_request_timelines`，harness mixed 请求带稳定 `X-Request-Id`；真实 smoke 显示短请求主要等待在 admission 前，`queue_wait_s` 最高 `2.102s`。 | `tests/test_engine_runtime.py::test_engine_batches_decode_steps_for_concurrent_requests`；`tests/test_compat_harness.py::test_mixed_scheduling_round_records_long_and_short_requests`；`compat-results/aster-vllm-mlx-mixed-scheduling-0.8b-request-timeline.json`。 |
| `prefill_batch_size` 被误用为 admission throttle | 已修复并部分改善 mixed p95。vllm-mlx scheduler 把 waiting 请求填入 running 直到 `max_num_seqs`；Aster 现在同样填满 available active slots，并在 status 暴露 `admission_policy=fill_available_active_slots`。真实 smoke 中 Aster short p95 从 `2.707s` 降到 `2.475s`，但仍慢于 vllm-mlx `1.455s`，所以只关闭该局部误差，不关闭 P0 continuous batching。 | `tests/test_engine_runtime.py::test_drain_submissions_fills_available_active_slots`；`compat-results/aster-vllm-mlx-mixed-scheduling-0.8b-admission-fill.json`。 |

## P2: 影响覆盖面和运维一致性

| Gap | 影响 | 当前状态 | 关闭标准 |
|---|---|---|---|
| CLI 参数仍未完整闭合到真实能力 | 基础 `serve MODEL` 命令已可替换，但部分 vllm-mlx 参数目前只是被解析并记录 warning；运维命令可迁移，能力语义仍取决于 continuous batching、cache、registry、MCP/rerank/multimodal 等 gap。 | Aster 支持 `aster serve MODEL` 和常用 serving 参数；`--config` 仅作为旧部署兼容路径保留。 | 每个保留参数都映射到真实实现或被从目标范围明确移除；`python -m aster serve --help` 与目标 vllm-mlx 参数集对照通过。 |
| Embeddings 未完成替代证明 | endpoint 存在但未跑真实黑盒；vllm hot-swap/locked model 语义未覆盖。 | Aster 单测通过，smoke 配置关闭 embeddings。 | 启用 embedding 模型后跑 batch/order/error/hot-swap 对照。 |
| Audio 未完成替代证明 | STT/TTS/voices/格式限制未完整覆盖。 | Aster 有部分服务，默认关闭。 | 启用 ASR/TTS 后跑 multipart、bytes、voice、limit、禁用错误测试。 |
| Benchmark 工具不同 | 后续验收需要统一结果格式。 | compare harness 已有基础协议、性能 smoke、prefix observation 和 lifecycle 探针；Aster 自有 benchmark 不等价 vllm bench-serve。 | harness 扩展为固定 workload + JSON schema + 趋势比较。 |

## 推荐改造顺序

1. 明确 compatibility mode：health/status/error/metrics response shape 是否跟随 vllm-mlx，避免每个端点单独漂移。
2. 继续定位同 token mixed workload 下的真实瓶颈；prefill continuation yielding 未改善 p95/p99，decode diagnostics 已证明本轮 mixed workload 未发生 batch fallback，engine timing 又显示纯 runner TPS 并非主要解释，request timeline 进一步显示短请求主要卡在 admission 前等待。Admission fill 修正已改善但未闭合差距；下一步应比较 vllm-mlx BatchGenerator 的 long+short prefill/decode 交错方式，重点看第 4 个短请求受 `max_active_requests=4` 影响时如何更快进入运行，以及是否需要真正接入 BatchGenerator runtime kernel。
3. 实现或明确等价替代 vllm-mlx cache stack：prefix cache stats 先对齐，再进入 paged/SSD/KV quant。
4. 补 MCP manager/executor 与 reranker backend，让本轮新增的兼容端点具备真实 positive path。
5. 补多模态和模型 registry；这两项代码面大，但都是完整替代的硬门槛。
6. 最后扩展 tool/structured/reasoning/parser 矩阵和 30 分钟稳定性测试。
