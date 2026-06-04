# BatchedEngine 实现方案

## 目标

新建 `aster/inference/batched_engine.py`，直接包装 `mlx_lm.BatchGenerator`，
实现 vllm-mlx 级别的连续批处理性能。

## 接口

必须实现 InferenceEngine 的 public methods（routes.py 使用的）:

- `health() -> dict`
- `status() -> dict`
- `start()` / `aclose()`
- `submit(request: InferenceRequest) -> InferenceResponse`
- `stream(request: InferenceRequest) -> AsyncIterator[str]`
- `cancel(request_id: str) -> bool`
- `configured_embedding_model() -> str | None`
- `supports_embeddings() -> bool`
- `get_cache_stats() -> dict`
- `clear_runtime_caches() -> dict`
- `clear_prefix_cache() -> dict`
- `embeddings(...)` — 可先抛 not implemented
- `count_text_tokens(texts) -> int`
- `warmup()`

## 架构设计

```
BatchedEngine
  ├── _state: dict[str, RequestState]     # 活跃请求
  ├── _waiting: deque[str]                 # 等待队列
  ├── _running: set[str]                   # 运行中
  ├── _finished: dict[str, InferenceResponse]  # 已完成
  ├── _batch_generator: BatchGenerator | None  # mlx-lm 核心
  ├── _scheduler: Scheduler                # 自定义调度器
  ├── model_runner: ModelRunner            # tokenize/sample
  ├── prefix_store: PrefixStore            # KV 缓存
  └── _engine_loop_task: asyncio.Task      # 异步步进循环
```

## 核心循环 (`_engine_loop`)

```
while running:
    # 1. 处理取消请求
    for req_id in _pending_aborts:
        batch_generator.remove(uid)
        cleanup

    # 2. 调度等待队列
    while len(_running) < max_active and _waiting:
        req = _waiting.popleft()
        # 前缀缓存查找
        # tokenize
        batch_generator.insert(tokens, max_tokens=...)
        _running.add(req_id)

    # 3. 单步生成
    if _running and batch_generator:
        results = batch_generator.step()  # 一次 MLX 前向
        # prefill + decode 交错由 BatchGenerator 内部处理

    # 4. 分发结果
    for result in results:
        if finished:
            push to _finished + notify event
            _running.remove
        else:
            update tokens, push to stream

    await asyncio.sleep(0)
```

## 集成点

1. `lifecycle.py`: 新增 `engine_type` 配置，`batch_generator` 时创建 BatchedEngine
2. `config.py`: EngineSettings 加 `engine_type: Literal["manual", "batched"]`
3. routes.py: 不动——BatchedEngine 实现同一接口

## 实现顺序

1. 文件框架 + 接口方法签名
2. 核心循环 + BatchGenerator 包装
3. 前缀缓存集成
4. submit/stream 方法
5. 配置接入
