# Aster → vllm-mlx Full Replacement Roadmap

Generated: 2026-06-04
Based on: vllm-mlx commit `015e080` (v0.4.0rc1), Aster commit `44d364d`

## 基线性能 (2026-06-04, Qwen3.5-0.8B-4bit)

| Concurrency | Aster avg/p95 | Aster TPS | vllm-mlx avg | vllm-mlx TPS |
|---|---|---|---|---|
| 1 | 0.76s / 0.87s | 42.2 | 0.015s | 0 (error) |
| 2 | 1.47s / 2.02s | 43.3 | 0.024s | 0 (error) |
| 4 | 2.13s / 2.36s | 59.6 | 0.032s | 0 (error) |

⚠️ 本轮 vllm-mlx non-stream 全部返回 `finish_reason: "error"`，completion_tokens=0。
Aster non-stream 全部正常 200/32 tokens。
需要修复 vllm-mlx 的兼容性问题后重跑。

---

## 总体架构对比

```
vllm-mlx                              Aster
─────────                              ─────
server.py (6688行)                     api/routes.py (1322行)
    FastAPI + lifespan                     FastAPI + lifespan
    OpenAI + Anthropic endpoints           OpenAI + Anthropic + Gemini + Cohere + ...

engine_core.py (794行)                  inference/engine.py (1526行)
    EngineCore._engine_loop()               InferenceEngine._engine_loop()
    async loop → executor thread            async loop → ThreadPoolExecutor(1)
    scheduler.step() → BatchGenerator       _step_prefill + _step_decode (manual)

scheduler.py (3007行)                    scheduler/adaptive_batcher.py (120行)
    Scheduler class                         [thin — mostly stubs]
    SchedulerConfig                         
    BatchGenerator integration              
    chunked prefill monkey-patch            
    abort/cancel/reschedule                 

model_runner.py (476行)                  inference/model_runner.py (876行)
    MLXModelRunner                          ModelRunner
    vLLM V1 plugin interface                Manual runtime kernel

prefix_cache.py (2 impls)                inference/prefix_store.py (552行)
    PrefixCacheManager (trie)               PrefixStore (bisect + SHA-256)
    BlockAwarePrefixCache (paged)           

paged_cache.py                           [MISSING]
    PagedCacheManager
    CacheBlock / BlockTable / COW

memory_cache.py                          [MISSING]
    MemoryAwarePrefixCache
    LRU eviction by RAM bytes

ssd_cache.py                             [MISSING]
    SSDCacheTier
    SQLite index + async spill/promote

model_registry.py (220行)               inference/model_registry.py (138行)
    multi-model serve                       single-model (registry exists but unused)
    lazy load / lease / eviction            ModelRegistry + ModelLease (infra ready)

optimizations.py                         [MISSING]
    mx.compile() fusion
    memory bandwidth detection
    Metal kernel optimization
```

---

## 阶段路线图

### Phase 0: 让连续批处理真正工作 (P0)

**目标**: Aster 的 continuous batching 功能与 vllm-mlx 等价，同负载下性能差距 <10%

| 任务 | vllm-mlx 参考 | Aster 现状 | 工作量 |
|------|------------|---------|--------|
| 0.1 修复 `BatchGeneratorRuntimeKernel` 使其可用 | `scheduler.py` Scheduler + BatchGenerator | `runtime_kernel.py` BatchGeneratorRuntimeKernel 全部 stub，`available=False` | 5-7天 |
| 0.2 实现 chunked prefill | `scheduler.py` `_install_chunked_prefill()` | 现有 prefill 是同步分块的，但缺少与 decode 的交错 | 3-4天 |
| 0.3 实现 prefill/decode 交错 | `_chunked_next()` + `_generation_step()` | `_step_prefill` → `_step_decode` 是顺序的，没有单步交错 | 2-3天 |
| 0.4 修复 abort/cancel 路径 | `_process_pending_aborts()` | 已有 `cancel()` 方法，但缺少从 `BatchGenerator.remove()` 清理 | 1-2天 |
| 0.5 对齐调度器配置 | `SchedulerConfig` + CLI 参数 | `RuntimeSettings` 已有对应字段，但映射不全 | 1天 |

**关键实现思路**:
- 不要重新实现 BatchGenerator 的轮子。vllm-mlx 的核心优势是直接用 `mlx_lm.generate.BatchGenerator` 做 token 级连续批处理
- Aster 的 `BatchGeneratorRuntimeKernel` 需要从 Protocol 桩变成真正的 `mlx_lm.BatchGenerator` 封装
- engine.py 的 `_step_prefill` + `_step_decode` 顺序执行模式需要改成单步 `batch_generator.next()` 调用
- vllm-mlx 的 chunked prefill 是 monkey-patch 进去的，Aster 可以直接做成原生支持

**验收标准**:
- mixed workload (1 long + 6 short) short p95 差距 ≤10%
- concurrency 1/2/4/8 吞吐量差距 ≤15%
- 无 `batch_fallbacks`
- 30 分钟稳定性运行

---

### Phase 1: 补齐 Cache 栈 (P0)

**目标**: 实现与 vllm-mlx 等价的四层缓存体系

| 任务 | vllm-mlx 参考 | Aster 现状 | 工作量 |
|------|------------|---------|--------|
| 1.1 Paged KV cache | `paged_cache.py` (600行) | **缺失** | 5-6天 |
| 1.2 Memory-aware prefix cache | `memory_cache.py` (700行) | **缺失** — 现有 PrefixStore 是文件名哈希的，非内存感知 | 4-5天 |
| 1.3 SSD tiering | `ssd_cache.py` (500行) | **缺失** | 4-5天 |
| 1.4 KV cache quantization | `memory_cache.py` `_QuantizedCacheWrapper` | **缺失** — CLI 已解析 `--kv-cache-quantization` 但未实现 | 3-4天 |
| 1.5 Block-aware prefix sharing | `prefix_cache.py` BlockAwarePrefixCache | **缺失** | 3-4天 |
| 1.6 缓存指标对齐 | 所有 cache 文件 | PrefixStore 已有 stats 但格式不同 | 1-2天 |

**关键实现思路**:
- vllm-mlx 的 `PagedCacheManager` 是核心基础设施: 基于 block 的内存管理、chain hash 前缀匹配、COW 写时复制
- `MemoryAwarePrefixCache` 基于实际 RAM 字节驱逐，比 Aster 的 `PrefixStore` (基于 entry 数量) 更精确
- Aster 已有的 `PrefixStore` 的 LCP 匹配和 checkpoint 机制是独特优势，可以与 paged cache 结合
- SSD tiering 通过 SQLite + safetensors 异步 spill/promote，Aster 可以直接复用类似设计

**验收标准**:
- 重复前缀请求 TTFT 降低 ≥50%
- 长上下文 (8K/16K/32K) 内存使用 ≤vllm-mlx
- 缓存指标格式与 vllm-mlx 兼容
- SSD 回填后缓存依然可用

---

### Phase 2: 补齐 API 后端 (P0)

**目标**: 所有 vllm-mlx 端点在 Aster 上有可工作的真实后端

| 任务 | vllm-mlx 参考 | Aster 现状 | 工作量 |
|------|------------|---------|--------|
| 2.1 MCP 执行引擎 | `mcp/*` 完整体系 | 端点存在，全部 stub 返回空 | 3-4天 |
| 2.2 Rerank 后端 | `rerank.py` + `rerank_forward.py` | 端点存在，返回 404 | 2-3天 |
| 2.3 多模型服务 | `model_registry.py` + lazy load | `ModelRegistry` 基础设施已建，但未接入实际服务流程 | 3-4天 |
| 2.4 多模态 | `models/mllm.py` + `multimodal_processor.py` | 显式返回 `multimodal_not_supported` | 大量 |
| 2.5 Embedding hot-swap | `embedding.py` | 已有 embedding 后端，需验证 hot-swap | 1-2天 |
| 2.6 Audio 完整验证 | TTS/STT/multilingual | 已有 ASR/TTS 服务，需完整黑盒验证 | 2-3天 |

**注意**: 2.4 (多模态) 是最大工作量，建议放在最后，因为:
- 需要 `mlx-vlm` 模型加载
- 需要图像/视频/音频 content 解析
- 需要 MLLM scheduler
- 但 2.4 不是文本 LLM 替代的核心场景

---

### Phase 3: API 兼容性打磨 (P1)

**目标**: 所有 API 响应格式与 vllm-mlx 兼容

| 任务 | 具体内容 | 工作量 |
|------|---------|--------|
| 3.1 错误格式对齐 | vllm-mlx 返回 `{"detail": "..."}`, Aster 返回 `{"error": {...}}` | 1天 |
| 3.2 `/health` 响应格式 | vllm-mlx `{"status", "model_loaded", "model_type", "engine_type"}` vs Aster `{"status", "degraded", "details"}` | 1天 |
| 3.3 `/v1/models` 格式 | vllm-mlx 有 `created` 字段，Aster 没有 | 0.5天 |
| 3.4 Cache stats 格式 | 两者字段不同 | 1天 |
| 3.5 Metrics 命名空间 | Prometheus 指标兼容 | 2-3天 |
| 3.6 Lifecycle (cancel/timeout/disconnect) | 已有部分实现，需全量验证 | 2-3天 |

---

### Phase 4: 性能优化 (P2)

**目标**: 在连续批处理等价后，做差异化性能超越

| 任务 | 说明 | 优先级 | 工作量 |
|------|------|--------|--------|
| 4.1 `mx.compile()` kernel fusion | vllm-mlx `optimizations.py` 已实现，Aster 缺失 | P2 | 3-4天 |
| 4.2 Tokenizer 下沉 Rust/C | 减少 Python server 开销 | P3 | 大量 |
| 4.3 Segment-level prefix cache | 差异化亮点 — Agent 场景 | P2 | 5-7天 |
| 4.4 Latency-first scheduler | Aster 场景独特价值 | P2 | 5-7天 |
| 4.5 Speculative decoding | `specprefill.py` + MTP | P3 | 5-7天 |
| 4.6 MoE fast path | `--moe-top-k`, 权重预取 | P3 | 3-5天 |

---

## 精确到文件的差距清单

### 缺失文件 (vllm-mlx 有, Aster 无)

| 文件 | 行数 | 功能 | 必要性 |
|------|------|------|--------|
| `vllm_mlx/paged_cache.py` | ~600 | Block-based KV cache allocator | P0 |
| `vllm_mlx/memory_cache.py` | ~700 | Memory-aware prefix cache w/ eviction | P0 |
| `vllm_mlx/ssd_cache.py` | ~500 | SSD tiering for cold cache | P0 |
| `vllm_mlx/prefix_cache.py` | ~400 | Trie + BlockAware prefix cache | P0 (需重构) |
| `vllm_mlx/optimizations.py` | ~200 | Metal kernel fusion + bandwidth opt | P2 |
| `vllm_mlx/specprefill.py` | ~300 | Attention-based sparse prefill | P3 |
| `vllm_mlx/prompt_warmup.py` | ~150 | Pre-load popular prefixes | P2 |
| `vllm_mlx/rerank.py` | ~150 | Reranker backend | P1 |
| `vllm_mlx/multimodal_processor.py` | ~400 | Image/video/audio processing | P0 (多模态) |
| `vllm_mlx/models/mllm.py` | ~300 | Multimodal model loading | P0 (多模态) |
| `vllm_mlx/mcp/` (4 files) | ~600 | MCP manager + executor + security | P1 |
| `vllm_mlx/gradio_app.py` | ~200 | Web UI (非核心) | P2 |

### 需重大改造的文件

| Aster 文件 | 问题 | 改造方向 | 工作量 |
|-----------|------|---------|--------|
| `aster/inference/runtime_kernel.py` | BatchGeneratorRuntimeKernel 全部 stub | 实现真正的 mlx-lm BatchGenerator 适配 | 5天 |
| `aster/inference/engine.py` | _step_prefill + _step_decode 顺序执行 | 改为单步 batch_generator.next() + chunked prefill 交错 | 5天 |
| `aster/inference/model_runner.py` | 没有 BatchGenerator 封装 | 新增 `_run_batch_generator_step()`, `_ensure_batch_generator()` | 3天 |
| `aster/inference/prefix_store.py` | 不是 paged/memory-aware | 重构为 MemoryAwarePrefixCache + PagedCacheManager 下层 | 5天 |
| `aster/inference/model_registry.py` | 基础设施已建，未接入服务 | 接入 lifecycle + lazy load + 多模型 serve | 3天 |
| `aster/api/feature_emulation.py` | 部分 emulation 逻辑可能不再需要 | 简化或移除 | 1天 |

### 可直接复用或微调的文件

| Aster 文件 | 状态 | 操作 |
|-----------|------|------|
| `aster/api/routes.py` | 端点基本完整 | 加 rerank/MCP backend 适配 |
| `aster/api/streaming.py` | 功能完整 | 微调 Anthropic response 格式 |
| `aster/api/schemas.py` | 功能完整 | 对齐 vllm-mlx 错误格式 |
| `aster/core/config.py` | 功能完整 | 加缺失的 cache 参数映射 |
| `aster/core/lifecycle.py` | 功能完整 | 微调 |
| `aster/inference/thinking_processor.py` | 功能完整 | 微调 |
| `aster/inference/reasoning_parsers.py` | 功能完整 | 加缺失 parser |
| `aster/inference/tool_parsers.py` | 功能完整 (18 parsers) | 加 missing vllm-mlx parser |
| `aster/audio/` | 服务存在 | 完整黑盒验证 |

---

## 推荐执行顺序 (按优先级)

```
Week 1-2:  Phase 0 (Continuous Batching)
  Day 1-2   0.5 对齐调度器配置 + CLI 映射
  Day 3-7   0.1 BatchGeneratorRuntimeKernel 实现
  Day 8-11  0.2-0.3 chunked prefill + prefill/decode 交错
  Day 12    0.4 abort/cancel 路径修复
  Day 13    0.5 验收: mixed workload + 30min 稳定性

Week 3-4:  Phase 1 (Cache Stack)
  Day 14-17 1.1 Paged KV cache (核心)
  Day 18-21 1.2 Memory-aware prefix cache
  Day 22-23 1.5 Block-aware prefix sharing
  Day 24-25 1.6 缓存指标对齐
  Day 26-28 1.3 SSD tiering (可与 1.2 并行)

Week 5:    Phase 2 (API Backends)
  Day 29-31 2.1 MCP 执行引擎
  Day 32-33 2.2 Rerank 后端
  Day 34-35 2.3 多模型服务接入

Week 5-6:  Phase 3 (API Compatibility)
  Day 36    3.1-3.4 错误/health/models/cache 格式对齐
  Day 37-38 3.5 Metrics 命名空间对齐
  Day 39-40 3.6 Lifecycle 全量验证

Week 7:    Phase 4 (差异化优化, 可选)
  Day 41-43 4.1 mx.compile() kernel fusion
  Day 44-47 4.3 Segment-level prefix cache (Agent 场景)
  Day 48-49 4.4 Latency-first scheduler
```

---

## 验证框架

每次合入后跑:

```bash
# 双服务模式验证
.venv/bin/python tools/compat/aster_vllm_mlx_compare.py \
  --aster-url http://127.0.0.1:18080 \
  --vllm-url http://127.0.0.1:18000 \
  --model /path/to/model \
  --max-tokens 32 \
  --requests 30 \
  --concurrency 1 2 4 8 16 \
  --include-lifecycle \
  --include-mixed-scheduling \
  --out compat-results/phase-validate.json

# 单元测试
.venv/bin/pytest tests/ -q

# vllm-mlx 自身测试 (验证 Aster 没引入 regression)
.venv/bin/pytest examples/vllm-mlx/tests/ -q
```

---

## 风险

1. **mlx-lm BatchGenerator API 稳定性**: vllm-mlx 大量 monkey-patch 了 BatchGenerator 内部方法 (chunked prefill, MTP, prompt cache save)。如果 mlx-lm 更新内部 API，这些 patch 会断。Aster 需要做同样的 patch。
2. **内存压力**: 连续批处理 + paged cache + SSD tiering 同时启用时，内存管理逻辑复杂。建议逐层开启。
3. **多模态工作量过大**: MLLM 需要 mlx-vlm 集成、图像预处理、MLLM scheduler。建议 Phase 2 最后做或延后。
4. **vllm-mlx 自身 bug**: 本轮测试发现 vllm-mlx non-stream 请求返回 finish_reason="error"，可能是特定模型版本的兼容性问题。需要确认。
