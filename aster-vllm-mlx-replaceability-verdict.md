# Aster/vllm-mlx Replaceability Verdict

## Verdict: Not Replaceable

截至本轮审查，Aster 不能完整替代 `examples/vllm-mlx`。

它已经能覆盖一部分基础文本 serving：OpenAI chat/completions 的短文本 non-stream 和 stream 在轻量黑盒中通过；Responses API 的基础生命周期也能跑通；Aster 自身相关测试 `252 passed`。但 vllm-mlx 的核心替代目标包括 continuous batching、MLLM、多层 KV cache、真实 MCP 执行、rerank scoring、模型 registry、完整音频能力、完整工具/结构化/reasoning parser 与生产级 lifecycle/metrics。Aster 对这些能力要么缺失，要么只实现了部分形态。

## 已验证通过的范围

本轮轻量黑盒：`compat-results/aster-vllm-mlx-compare-0.8b.json`。
本轮 mixed scheduling 双服务黑盒：`compat-results/aster-vllm-mlx-mixed-scheduling-0.8b-fixed.json`。
本轮默认 thinking 对齐后 mixed scheduling 双服务黑盒：`compat-results/aster-vllm-mlx-mixed-scheduling-0.8b-thinking-default.json`。
本轮 prefill continuation fairness 改造后 mixed scheduling 双服务黑盒：`compat-results/aster-vllm-mlx-mixed-scheduling-0.8b-prefill-yield.json`。
本轮 decode batch 诊断扩展后 mixed scheduling 双服务黑盒：`compat-results/aster-vllm-mlx-mixed-scheduling-0.8b-decode-diagnostics.json`。
本轮 engine timing 诊断扩展后 mixed scheduling 双服务黑盒：`compat-results/aster-vllm-mlx-mixed-scheduling-0.8b-engine-timing.json`。
本轮请求级 timeline 诊断扩展后 mixed scheduling 双服务黑盒：`compat-results/aster-vllm-mlx-mixed-scheduling-0.8b-request-timeline.json`。
本轮 admission policy 修正后 mixed scheduling 双服务黑盒：`compat-results/aster-vllm-mlx-mixed-scheduling-0.8b-admission-fill.json`。

- `/v1/chat/completions` 基础文本 non-stream：双方 200，必需字段通过。
- `/v1/chat/completions` 基础文本 stream：双方 200，均有 terminal done。
- `/v1/completions` 基础文本 non-stream/stream：双方 200，stream 以 `[DONE]` 结束。
- `/v1/responses` 基础文本 non-stream/stream：双方 200，stream 以 `response.completed` 结束。
- excessive `max_tokens`：双方状态码 400，但 body 不兼容。
- `/health`、`/v1/models`、`/v1/cache/stats`：双方 200，但字段结构不兼容或只可部分迁移。
- `aster serve MODEL --max-request-tokens 2048` 不再把模型 context length 错误降到 2048；修复后 512-word long prompt mixed workload 中 Aster long requests 2/2 成功，vllm-mlx 2/2 成功。
- `aster serve MODEL` 默认 thinking 已对齐 vllm-mlx：basic chat 双方 prompt/completion/total `28/8/36`、finish_reason `length`；显式关闭路径由 `--default-chat-template-kwargs '{"enable_thinking": false}'` 覆盖。
- Aster `/v1/status` 现在暴露 `decode_batch_diagnostics`；本轮 mixed smoke 中 Aster `batch_attempts=17`、`batch_successes=17`、`batch_fallbacks=0`，证明当前 mixed 尾延迟差距不是由 batch decode fallback 直接造成。
- Aster `/v1/status` 现在暴露 `engine_timing`；本轮 mixed smoke 中 Aster `prompt_tps=2741.634`、`generation_tps=92.88`，vllm-mlx `generation_tps=96.41`，证明当前尾延迟差距也不能简单归因于纯 runner TPS 明显不足。
- Aster `/v1/status` 现在暴露 `recent_request_timelines`；本轮 mixed smoke 中 Aster 短请求 `queue_wait_s` 为 `1.329s/1.533s/1.722s/2.102s`，短请求 prefill wall 只有 `0.027s/0.026s/0.030s/0.064s`，把当前差距定位到 admission/active slot 等待，而不是短请求 prefill 本身。
- Aster admission policy 已按 vllm-mlx scheduler 语义修正为填满可用 active slots；本轮 mixed smoke 中 Aster short p95 从 `2.707s` 降到 `2.475s`，但同轮 vllm-mlx short p95 为 `1.455s`，所以 continuous batching 仍未达到替代。

## 最严重的未关闭差距

- P0 continuous batching：Aster 已有 decode batching、chunked prefill 和本轮新增 decode-first admission throttle，但尚未证明与 vllm-mlx continuous batching 等价；`batch_generator` adapter 仍 `available=False`。
- P0 cache stack：Aster 没有 vllm-mlx 的 paged KV cache、SSD tier、KV quantization、memory-aware cache 等价实现。
- P0 多模态：Aster 本地 runtime 对 multimodal content 返回 `multimodal_not_supported`。
- P0 API 后端缺失：`/v1/mcp/*`、`/v1/rerank`、`/v1/messages/count_tokens`、`/v1/audio/voices` 的基础端点已补齐，但 MCP 执行、rerank scoring 和音频完整行为仍未达到 vllm-mlx。
- P0 模型管理：缺少 `--models-config`、registry、lazy multi-model serving、lease/eviction、idle unload。
- P1 兼容性：错误格式、metrics 字段、health/status 字段与 vllm-mlx 不一致；启动 CLI surface 已改善，但部分 serve 参数仍未闭合到真实能力。

## 本轮验收体系变化

`tools/compat/aster_vllm_mlx_compare.py` 新增 `--include-lifecycle`，可对 chat/completion timeout 504、stream 提前断开和未知请求 cancel 404 做双服务探测；`tests/test_compat_harness.py` 已保护这些判定规则。该变化提升了后续验收可重复性，但尚未证明 Aster 的真实双服务 lifecycle 全面通过。

启动入口已从“必须 `--config`”修正为 vllm-mlx 风格 `aster serve MODEL`，并解析主要 serving 参数。该项不再判为 `INCOMPATIBLE`；仍保留为 `PARTIAL`，因为 `--continuous-batching`、`--models-config`、paged/SSD/KV quant、MCP、rerank、多模态等参数虽然可解析，但真实能力仍由对应 P0/P1 gap 阻塞。本轮进一步修复了 `--max-request-tokens` 语义，并将 vllm 风格启动路径的默认 thinking 行为对齐到 vllm-mlx。

调度侧新增多项可验证改造：engine loop 现在先处理 cancellation/decode/prefill，再处理新 admission，并用 `batch.prefill_batch_size` 限制每轮 admission 数量；本轮又让长 prefill continuation 在新 admission 后让出一个 prefill turn，并暴露 `prefill_yield_rotations`。这些改动降低了局部公平性风险，但还不是 vllm-mlx BatchGenerator/AsyncEngineCore 等价实现。

验收工具新增 `--include-mixed-scheduling`：可在双服务上运行长 prefill + 短请求混合 workload，记录短请求 p50/p95/p99。本轮还增加了 mixed records 的 finish_reason、usage 和 content preview，能区分“生成语义不一致”和“同 token 负载下调度慢”。随后又加入 mixed 前后 `/v1/status` 快照，用于记录 Aster decode batch attempts/success/fallback 和 engine timing。默认 thinking 对齐后，Aster 和 vllm-mlx 的 token/finish 已对齐；decode diagnostics 证明本轮没有 batch fallback，engine timing 证明纯 runner TPS 不是主要解释，但 Aster 短请求 p95/p99 仍慢于 vllm-mlx，因此 continuous batching 仍为 P0/PARTIAL。

本轮继续扩展 mixed scheduling 证据链：harness 会给 mixed 请求设置稳定 `X-Request-Id`，Aster status 保留最近完成请求的 bounded timeline。新的黑盒结果证明，Aster 短请求主要在 admission 前等待，后续 prefill/decode 阶段并不是主要增量来源。该改动没有关闭 continuous batching gap，但把下一轮改造目标从“泛化调度慢”收敛到 active slot/admission policy。

随后对照 vllm-mlx scheduler 源码修正了一个局部误差：`prefill_batch_size` 不应限制 admission，waiting 请求应尽量调入 running 直到 `max_num_seqs`。Aster 现在暴露 `admission_policy=fill_available_active_slots`，实测 short p95 有改善；但第 4 个短请求仍受 active slot 上限影响明显，且 vllm-mlx 同轮仍更快，因此 verdict 不变。

## 性能证据

本轮只做了 smoke 级性能，不作为最终性能结论，但足以证明不能判定为可替代。

| 实现 | concurrency | avg latency | p95 latency | completion tps |
|---|---:|---:|---:|---:|
| Aster | 1 | 0.317s | 0.326s | 22.08 |
| Aster | 2 | 0.487s | 0.586s | 24.02 |
| vllm-mlx | 1 | 0.122s | 0.134s | 65.57 |
| vllm-mlx | 2 | 0.107s | 0.118s | 115.06 |

同一 0.8B 模型、短请求、3 次/档下，vllm-mlx 已明显更快；Aster 在 concurrency 2 时 latency 上升，vllm-mlx 因 continuous batching 吞吐上升。

Mixed scheduling smoke 使用同一 0.8B 模型、1 个 long request + 6 个 short requests、重复 2 轮。修复前 Aster long requests 因错误 context limit 0/2 成功，vllm-mlx 2/2 成功；修复后：

| 实现 | long success | short success | short p95 | short p99 |
|---|---:|---:|---:|---:|
| Aster | 2/2 | 12/12 | 2.473s | 2.487s |
| vllm-mlx | 2/2 | 12/12 | 1.721s | 1.735s |

Aster 已能完成该混合 workload，但短请求尾延迟仍明显落后，continuous batching 不能判定为已替代。

默认 thinking 对齐后，同一 workload 的生成 token 和 finish_reason 已对齐，新的性能差距更直接反映调度/执行差异：

| 实现 | long completion tokens | short completion tokens | short p95 | short p99 |
|---|---:|---:|---:|---:|
| Aster | 16 | 96 | 3.037s | 3.045s |
| vllm-mlx | 16 | 96 | 1.761s | 1.783s |

prefill continuation yielding 改造后再次运行同一 smoke，没有证明 p95/p99 改善：

| 实现 | long completion tokens | short completion tokens | short p95 | short p99 |
|---|---:|---:|---:|---:|
| Aster | 16 | 96 | 3.107s | 3.116s |
| vllm-mlx | 16 | 96 | 1.968s | 1.976s |

因此本轮只关闭了一个局部公平性问题，P0 continuous batching 差距仍然存在。

decode batch 诊断扩展后，小规模 mixed smoke 进一步缩小了原因范围：

| 实现 | long success | short success | short p95 | short p99 | Aster batch fallback |
|---|---:|---:|---:|---:|---:|
| Aster | 1/1 | 4/4 | 3.129s | 3.134s | 0/17 |
| vllm-mlx | 1/1 | 4/4 | 1.712s | 1.713s | n/a |

这说明 Aster 的 manual runner 在该负载下确实走了 batch decode，下一轮应转向 prefill/decode 时间占比、scheduler tick 粒度、runner 单线程阻塞和 BatchGenerator 架构差异。

engine timing 诊断扩展后，小规模 mixed smoke 继续缩小原因范围：

| 实现 | short p95 | short p99 | prompt TPS | generation TPS |
|---|---:|---:|---:|---:|
| Aster | 2.519s | 2.521s | 2741.634 | 92.88 |
| vllm-mlx | 1.631s | 1.636s | 1110.51 | 96.41 |

Aster runner 侧吞吐不低，但端到端短请求仍慢，且短请求完成时间集中在 long request 之后。这把下一轮重点从 runner batch fallback/TPS 转向 scheduler tick 粒度、请求级状态时间线和 HTTP/non-stream 交付链路。

request timeline 诊断扩展后，小规模 mixed smoke 进一步定位到 admission/active slot 等待：

| 实现 | long success | short success | short p95 | short p99 | 关键阶段 |
|---|---:|---:|---:|---:|---|
| Aster | 1/1 | 4/4 | 2.707s | 2.710s | short `queue_wait_s` 最高 `2.102s` |
| vllm-mlx | 1/1 | 4/4 | 2.399s | 2.405s | status 无 completed timeline |

Aster 的短请求 prefill wall 仅 `0.027s/0.026s/0.030s/0.064s`，decode duration 为 `1.038s/0.652s/0.666s/0.204s`，但 admission 前等待已经占据主要延迟。因此下一轮应验证更早 admission、active slot 预留或把 waiting short request 直接纳入 decode-first/chunked-prefill 调度池是否能缩短 p95。

admission fill 修正后，同一小规模 mixed smoke 有阶段性改善，但仍未达到 vllm-mlx：

| 实现 | long success | short success | short p95 | short p99 | 关键阶段 |
|---|---:|---:|---:|---:|---|
| Aster | 1/1 | 4/4 | 2.475s | 2.477s | 前三 short `queue_wait_s` 约 `1.01-1.12s`，第 4 个 `1.925s` |
| vllm-mlx | 1/1 | 4/4 | 1.455s | 1.461s | BatchGenerator/MLLM scheduler |

这关闭了“prefill_batch_size 被误用为 admission throttle”这一局部问题，但没有关闭 P0 continuous batching gap。

## 尚未被证明的范围

- 9B/35B 目标模型下的完整黑盒对照。
- 长上下文、重复前缀、warm prompts、paged/SSD/KV quant cache 收益。
- 真实双服务 client disconnect、cancel、timeout、heartbeat、异常后恢复；非流式 `client_disconnected` 空 499 已有单测覆盖，timeout/stream 提前断开/未知 cancel 已有 harness 探针但未跑完整双服务结果。
- 长 prefill + 短请求混合调度已有 smoke 结果，但缺 concurrency 1/2/4/8/16、多轮重复、取消压力和长跑稳定性数据。
- embeddings、rerank、audio、MCP、多模态 positive cases。
- 工具调用、structured output、reasoning parser 的 parser-by-parser 兼容性。
- 30 分钟以上持续运行和内存释放行为。

## 下一步优先事项

1. 继续推进 continuous batching 等价能力：prefill continuation yielding 未改善 mixed smoke，decode diagnostics 已排除 batch fallback，engine timing 已排除纯 runner TPS 明显不足，request timeline 已定位到短请求 admission 前等待；admission fill 修正有收益但仍落后。下一轮应比较 vllm-mlx BatchGenerator 的 long+short 交错方式，验证 active slot 上限下第 4 个短请求为何能更快完成，并以 concurrency 1/2/4/8/16、多轮重复和取消压力重跑 harness。
2. 对齐 cache stats 与 prefix cache 行为，再实现 paged/SSD/KV quant 的可观察替代。
3. 给本轮新增的 MCP/rerank 兼容端点接入真实 backend，并补 positive-path 黑盒测试。
4. 扩展 harness 覆盖 tool/structured/reasoning、Anthropic、embedding、rerank、audio、多模态和 lifecycle。
5. 用目标 9B/35B 模型跑多轮性能和稳定性，结果写入 `compat-results/`。

## 重新执行审查

```bash
.venv/bin/pytest \
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

```bash
PYTHONPATH=examples/vllm-mlx .venv/bin/pytest \
  examples/vllm-mlx/tests/test_responses_api.py \
  examples/vllm-mlx/tests/test_embeddings.py \
  examples/vllm-mlx/tests/test_rerank.py \
  examples/vllm-mlx/tests/test_metrics.py \
  examples/vllm-mlx/tests/test_lifecycle_cli.py -q
```

```bash
.venv/bin/python tools/compat/aster_vllm_mlx_compare.py \
  --aster-url http://127.0.0.1:18080 \
  --vllm-url http://127.0.0.1:18000 \
  --model /path/to/the/same/model \
  --max-tokens 32 \
  --requests 30 \
  --concurrency 1 2 4 8 16 \
  --include-lifecycle \
  --out compat-results/aster-vllm-mlx-compare.json
```
