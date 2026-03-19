# Aster ASR & TTS 部署指南

## 快速开始

### 1. 启动服务器

```bash
cd /Users/eitan/Documents/Projects/Python/Aster
source .venv/bin/activate
python -m aster --config configs/config.yaml
```

服务器会在 `http://127.0.0.1:8080` 启动。

### 2. 验证服务健康

```bash
curl http://127.0.0.1:8080/health
curl http://127.0.0.1:8080/ready
```

## 测试 ASR 和 TTS

### 方式一：使用 CLI 测试脚本（推荐）

```bash
# 检查 API 健康状态
python scripts/test_audio_cli.py --health

# 测试 TTS（文本转语音）
python scripts/test_audio_cli.py --tts "Hello world" --output output.wav

# 测试 ASR（语音转文本）
python scripts/test_audio_cli.py --asr output.wav

# 测试端到端管道（TTS -> ASR）
python scripts/test_audio_cli.py --pipeline "This is a test message"
```

### 方式二：使用 pytest（完整测试套件）

```bash
# 运行所有音频服务测试
pytest tests/test_audio_services.py -v -s

# 运行特定测试
pytest tests/test_audio_services.py::TestTTSService::test_synthesize_speech -v -s
pytest tests/test_audio_services.py::TestASRService::test_transcribe_audio -v -s
```

### 方式三：使用 curl 直接调用 API

**TTS 示例：**
```bash
curl -X POST http://127.0.0.1:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-TTS-0.6B",
    "input": "Hello, this is a test",
    "voice": "default"
  }' \
  --output output.wav
```

**ASR 示例：**
```bash
curl -X POST http://127.0.0.1:8080/v1/audio/transcriptions \
  -F "file=@output.wav" \
  -F "model=Qwen3-ASR-0.6B"
```

## API 端点

### 健康检查
- `GET /health` - 服务健康状态
- `GET /ready` - 服务就绪状态
- `GET /metrics` - Prometheus 指标

### 音频服务
- `POST /v1/audio/speech` - TTS（文本转语音）
- `POST /v1/audio/transcriptions` - ASR（语音转文本）

### LLM 推理
- `POST /v1/chat/completions` - 聊天补全
- `POST /v1/completions` - 文本补全

## TTS 参数

```json
{
  "model": "Qwen3-TTS-0.6B",
  "input": "要合成的文本",
  "voice": "default",           // 声音类型
  "speed": 1.0,                 // 语速（0.5-2.0）
  "language": "en",             // 语言（可选）
  "reference_audio": "...",     // 参考音频（用于声音克隆，可选）
  "speaker": "...",             // 说话人（可选）
  "instruct": "..."             // 指令（可选）
}
```

## ASR 参数

```json
{
  "model": "Qwen3-ASR-0.6B",
  "language": "en",             // 语言（可选）
  "prompt": "..."               // 提示词（可选）
}
```

## 配置文件

主配置文件：`configs/config.yaml`

关键配置项：
- `api.host` - API 监听地址（默认 127.0.0.1）
- `api.port` - API 监听端口（默认 8080）
- `model.path` - LLM 模型路径
- `cache.prefix_cache_enabled` - 启用前缀缓存
- `speculative.enabled` - 启用推测解码（默认关闭）

## 模型信息

已下载的模型：

| 模型 | 大小 | 用途 |
|------|------|------|
| Qwen3-ASR-0.6B | 0.66GB | 语音转文本 |
| Qwen3.5-9B | 6.22GB | 主推理模型 |
| Qwen3-TTS-0.6B | 1.59GB | 文本转语音 |

## 故障排除

### 连接被拒绝
```
✗ Failed to connect to API: Connection refused
```
**解决方案：** 确保服务器正在运行
```bash
python -m aster --config configs/config.yaml
```

### 模型加载失败
```
[Aster][error] Failed to load ASR model
```
**解决方案：** 检查模型文件是否完整
```bash
python scripts/test_audio_cli.py --health
bash scripts/setup/download_models.sh --verify-only
```

### 超时错误
```
✗ ASR request failed: Request timeout
```
**解决方案：** 增加超时时间或检查系统资源
```bash
# 在 test_audio_cli.py 中修改 TIMEOUT 值
TIMEOUT = 60.0  # 增加到 60 秒
```

## 性能优化

1. **启用前缀缓存** - 减少重复计算
   ```yaml
   cache:
     prefix_cache_enabled: true
   ```

2. **调整批处理大小** - 平衡吞吐量和延迟
   ```yaml
   batch:
     max_batch_size: 4
     prefill_batch_size: 4
   ```

3. **启用自动调优** - 自动优化性能
   ```yaml
   autotune:
     enabled: true
   ```

## 下一步

- 集成到你的应用中
- 调整模型参数以获得最佳性能
- 监控 `/metrics` 端点的性能指标
- 考虑部署到生产环境（使用 Gunicorn/Uvicorn）
