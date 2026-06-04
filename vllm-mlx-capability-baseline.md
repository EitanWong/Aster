# vllm-mlx Capability Baseline

审查对象：`examples/vllm-mlx`。本文件只记录参考实现已经在源码、测试或可运行服务中体现的能力；README 只作为提示，不作为单独依据。

## 基准结论

`examples/vllm-mlx` 不是一个单纯的 OpenAI chat wrapper。它包含完整 FastAPI 服务、CLI 启动配置、simple/batched 两套 engine、连续批处理、MLLM 调度、prefix/memory-aware/SSD/paged KV cache、模型注册与 lazy serving、多协议 API、工具调用、结构化输出、reasoning parser、音频、embedding、rerank、MCP、Prometheus metrics 与 benchmark 工具。

## 能力清单

| 能力 | vllm-mlx 行为 | 源码或测试依据 |
|---|---|---|
| 服务入口与 CLI | `vllm-mlx serve [MODEL]`，支持 `--models-config`、`--served-model-name`、host/port、continuous batching、prefix cache、paged cache、KV quantization、SSD cache、warm prompts、MTP、SpecPrefill、MCP、auth、rate limit、timeout、metrics、lazy load、auto unload、audio limits、tool/reasoning parsers、embedding/rerank model。 | `examples/vllm-mlx/pyproject.toml` scripts；`vllm_mlx/cli.py:22-120`；`vllm_mlx/cli.py:959-1388` |
| 启动生命周期 | FastAPI lifespan 在启动时加载默认模型或模型注册表、加载 prefix cache、执行 warm prompts、初始化 MCP、启动 idle unload；关闭时保存 cache、停止 MCP/registry/engine。 | `vllm_mlx/server.py:1255-1385` |
| 安全与运行配置 | 可配置 API key、rate limit、默认 timeout、max request tokens、默认 temperature/top_p、metrics 开关。 | `vllm_mlx/server.py:125-145`；`vllm_mlx/cli.py:71-80`；`vllm_mlx/cli.py:1215-1237` |
| 健康、状态、模型列表 | `/health`、`/v1/status`、`/v1/models` 暴露模型加载状态、engine 类型、scheduler stats、cache stats、registry/MCP 信息；models 可包含主模型、embedding、reranker。 | `vllm_mlx/server.py:3106-3215`；`vllm_mlx/server.py:3377-3393` |
| Prometheus metrics | `/metrics` 在 `--enable-metrics` 时开放，覆盖 HTTP、inference、TTFT、token、scheduler、Metal memory、cache、registry、MCP 等指标。 | `vllm_mlx/server.py:3093-3103`；`vllm_mlx/metrics.py:80-288`；`tests/test_metrics.py` |
| OpenAI chat completions | `/v1/chat/completions` 支持 non-stream 与 SSE stream、OpenAI choices/usage、tools/tool_choice、response_format、chat_template_kwargs、multimodal content、reasoning、thinking budget、stop/sampling 参数；`enable_thinking` 请求默认是 `None`，BatchedEngine 对非 coder 模型默认启用 thinking。 | `vllm_mlx/api/models.py:42-193`；`vllm_mlx/engine/batched.py:617-626`；`vllm_mlx/server.py:4517-4715`；`tests/test_server.py:2236-2702` |
| OpenAI completions | `/v1/completions` 支持 string 或 list prompt；non-stream 可处理多 prompt；stream 处理首 prompt；返回 text_completion、usage、finish_reason。 | `vllm_mlx/api/models.py:258-278`；`vllm_mlx/server.py:4374-4515` |
| OpenAI Responses API | `/v1/responses` 实现本地 subset，支持 input/instructions、stream、tools/tool_choice、parallel tool calls、previous_response_id、text/reasoning 字段、response lifecycle events。 | `vllm_mlx/api/responses_models.py:1-8`；`vllm_mlx/api/responses_models.py:150-226`；`vllm_mlx/server.py:4791-4880`；`tests/test_responses_api.py:118-340` |
| Anthropic Messages API | `/v1/messages` 支持 lenient JSON、system/messages/tools/tool_choice、thinking/tool_use block 映射、stream/non-stream；`/v1/messages/count_tokens` 估算 tokens。 | `vllm_mlx/api/anthropic_models.py:53-113`；`vllm_mlx/server.py:4955-5266`；`tests/test_server.py:2326-2527` |
| Embeddings | `/v1/embeddings` 支持 batch input、token usage、embedding model lazy load/hot swap、模型锁定拒绝。 | `vllm_mlx/server.py:3401-3527`；`tests/test_embeddings.py:146-243` |
| Rerank | `/v1/rerank` 支持 query/documents、`top_n`、排序、可选返回 documents；未加载 reranker 时返回明确错误。 | `vllm_mlx/server.py:3534-3657`；`tests/test_rerank.py:489-587` |
| MCP | `/v1/mcp/tools`、`/v1/mcp/servers`、`/v1/mcp/execute`；可通过 CLI `--mcp-config` 初始化。 | `vllm_mlx/server.py:3664-3735`；`vllm_mlx/cli.py:1207-1213`；`tests/test_mcp_security.py` |
| Audio | `/v1/audio/transcriptions`、`/v1/audio/speech`、`/v1/audio/voices`，支持上传大小限制、TTS 输入长度限制、JSON/text/audio 响应。 | `vllm_mlx/server.py:3747-3885`；`tests/test_audio.py`；`tests/test_audio_limits.py` |
| 多模态 | Message content 支持 text、image_url、video/video_url、audio_url；MLLM engine 支持 image/video/audio 预处理和流式生成。 | `vllm_mlx/api/models.py:42-60`；`vllm_mlx/engine/batched.py:285-414`；`tests/test_mllm_continuous_batching.py` |
| request lifecycle | 支持显式 cancel endpoint、client disconnect guard、stream heartbeat、absolute timeout、non-stream disconnect 返回 499、timeout 返回 504、cleanup/release lease。 | `vllm_mlx/server.py:3334-3374`；`vllm_mlx/server.py:3893-4269` |
| simple engine | 单请求路径基于 MLXLanguageModel/MLXMultimodalLM，带锁保护、KV cache snapshot、MTP/SpecPrefill 入口。 | `vllm_mlx/engine/base.py:90-275`；`vllm_mlx/engine/simple.py:107-360` |
| continuous batching | `BatchedEngine` 基于 `AsyncEngineCore`/LLM scheduler，支持 prefill/completion batch、max seqs、chunked prefill、abort、stats；scheduler 每步会把 waiting 队列尽量调入 running 直到 `max_num_seqs`，`prefill_batch_size` 是 BatchGenerator prefill 批量参数，不是 admission 限流。 | `vllm_mlx/engine/batched.py:1-12`；`vllm_mlx/scheduler.py:1961-2115`；`vllm_mlx/scheduler.py:2473-2495`；`tests/test_batching_deterministic.py` |
| MLLM continuous batching | `MLLMScheduler` 与 `MLLMBatchGenerator` 支持 MLLM 请求队列、abort、stream outputs、prefix cache stats；MLLM scheduler 同样会把 waiting 请求调入 running 直到 `max_num_seqs`，`prefill_batch_size` 在 batch generator 内控制 prefill batch。 | `vllm_mlx/mllm_scheduler.py:514-565`；`vllm_mlx/mllm_batch_generator.py:393-452`；`tests/test_mllm_continuous_batching.py` |
| Prefix cache | 支持 prefix cache、warm prompts、cache stats、clear prefix、持久化 load/save。 | `vllm_mlx/server.py:3218-3331`；`vllm_mlx/cli.py:995-1087`；`tests/test_prefix_cache.py` |
| Memory-aware KV cache | 自动内存限制、LRU、准确内存估算、KV quantization 配置。 | `vllm_mlx/memory_cache.py:1-240`；`tests/test_memory_cache_mlx.py`；`tests/test_kv_cache_quantization.py` |
| Paged KV cache | vLLM-style block cache，refcount、copy-on-write、LRU、chain hashing。 | `vllm_mlx/paged_cache.py:1-225`；`tests/test_paged_cache.py`；`tests/test_paged_cache_real_inference.py` |
| SSD tiered cache | SQLite index、异步 writer、atomic writes、SSD 容量限制、stats。 | `vllm_mlx/ssd_cache.py:1-211`；`tests/test_ssd_cache.py` |
| 模型注册与资源释放 | registry-backed multi-model serving，lazy load/preload/eviction、memory budget、lease；CLI 支持 `--models-config`、lazy load、idle unload。 | `vllm_mlx/model_registry.py:1-220`；`vllm_mlx/cli.py:959-966`；`vllm_mlx/cli.py:1238-1248`；`tests/test_lifecycle_manager.py` |
| Sampling/stop/usage | temperature、top_p、top_k、min_p、presence/frequency/repetition penalty、stop、stop_token_ids、usage accounting。 | `vllm_mlx/api/models.py:155-193`；`vllm_mlx/server.py:4374-4715` |
| Tool calling | 多 parser：mistral/qwen/qwen3_coder/llama/hermes/harmony/gpt-oss/deepseek/kimi/granite/nemotron/xlam/functionary/gemma4/glm47/minimax；auto tool choice 需 parser。 | `vllm_mlx/cli.py:1261-1297`；`tests/test_server.py:2236-2324` |
| Structured output | `response_format` 支持 json_object/json_schema，依赖 lm-format-enforcer。 | `vllm_mlx/api/models.py:118-142`；`tests/test_server.py`；`tests/test_responses_api.py` |
| Reasoning | reasoning parser 可配置；可输出 `reasoning_content`；支持 thinking token budget。 | `vllm_mlx/cli.py:1298-1312`；`vllm_mlx/server.py:101-113`；`tests/test_server.py:2236-2411` |
| Benchmark | `vllm-mlx bench`、`bench-kv-cache`、`bench-serve` 等命令覆盖 engine 和 serve benchmark。 | `vllm_mlx/cli.py:458-577`；`vllm_mlx/cli.py:695+`；`tests/test_bench_serve.py` |

## 本轮参考测试

已执行：

```bash
PYTHONPATH=examples/vllm-mlx .venv/bin/pytest \
  examples/vllm-mlx/tests/test_responses_api.py \
  examples/vllm-mlx/tests/test_embeddings.py \
  examples/vllm-mlx/tests/test_rerank.py \
  examples/vllm-mlx/tests/test_metrics.py \
  examples/vllm-mlx/tests/test_lifecycle_cli.py -q
```

结果：`84 passed, 1 deselected in 40.64s`。
