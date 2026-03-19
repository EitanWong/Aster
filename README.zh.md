<div align="center">
  <img src="assets/logo.svg" alt="Aster Logo" width="200" height="200">

  # Aster

  **面向生产的 Apple Silicon 本地 LLM 推理运行时**

  [English](README.md) | [中文](README.zh.md) | [日本語](README.ja.md) | [Español](README.es.md) | [Français](README.fr.md) | [Deutsch](README.de.md) | [한국어](README.ko.md)
</div>

Aster 是一个面向生产的 Apple Silicon 本地 LLM 推理运行时，专为长上下文、OpenClaw 风格的 Agent 工作负载优化。

## 为什么选择 Aster

Aster 针对以下场景进行了优化：

- 超长提示词和重复的长前缀
- 工具密集型 Agent 提示词
- 长对话
- 连续本地后台服务
- 基准测试验证的运行时策略选择
- Apple Silicon + MLX 部署

它提供 OpenAI 兼容的 API，并将高级优化视为候选策略而非教条。推测解码、前缀缓存、批处理、调度和流式传输节奏都经过基准测试，并根据本地测量的性能和稳定性进行选择。

## 核心理念

- OpenAI 兼容的 API，支持流式和非流式端点
- 显式的预填充/解码分离
- 具有队列感知批处理的自适应调度器
- 分页 KV 管理器抽象
- 具有确定性哈希的自动前缀缓存
- 具有自动禁用回退的推测解码控制器
- 保持最快稳定配置文件的基准测试/自动调优子系统
- 结构化日志、指标、监督和就绪/健康报告

## 快速开始

```bash
cd /Users/eitan/Documents/Projects/Python/Aster

# 创建虚拟环境
/opt/homebrew/bin/python3.13 -m venv .venv
source .venv/bin/activate

# 安装依赖（包括用于 ASR/TTS 的 mlx-audio）
python -m pip install -r requirements.txt

# 下载模型（ASR、LLM、TTS）
bash scripts/setup/download_models.sh

# 启动服务器
python -m aster --config configs/config.yaml
```

API 将在 `http://127.0.0.1:8080` 上可用

### 验证安装

```bash
# 检查健康状态
curl http://127.0.0.1:8080/health

# 测试 LLM 推理
curl -X POST http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3.5-9B",
    "messages": [{"role": "user", "content": "你好"}],
    "max_tokens": 100
  }'

# 测试 ASR（语音转文本）
python scripts/test_audio_cli.py --tts "你好世界" --output test.wav
python scripts/test_audio_cli.py --asr test.wav

# 测试端到端管道
python scripts/test_audio_cli.py --pipeline "这是一个测试"
```

## Python 版本

Aster 针对现代 Python，应在 Python 3.13.x（如果可用）上运行（最低 3.12+）。macOS 系统 Python 被认为不支持此项目。

## API

- `GET /health`
- `GET /ready`
- `GET /metrics`
- `GET /v1/models`
- `POST /v1/chat/completions` — LLM 聊天推理
- `POST /v1/completions` — LLM 文本完成
- `POST /v1/audio/transcriptions` — ASR（语音转文本）
- `POST /v1/audio/speech` — TTS（文本转语音）

兼容性说明：
- 参见 `docs/api/OPENAI_COMPAT.md` 了解 Aster 的默认兼容性契约和可选调试扩展。

## 音频服务（ASR & TTS）

Aster 包含由 Qwen3 模型驱动的集成语音识别和合成：

### ASR（语音转文本）
- 模型：Qwen3-ASR-0.6B（0.66GB）
- 支持多种语言
- 快速本地转录

### TTS（文本转语音）
- 基础模型：Qwen3-TTS-0.6B（1.59GB）
- CustomVoice 模型：Qwen3-TTS-CustomVoice-0.6B（可选，用于声音克隆）
- 可调节的语速
- 使用参考音频进行声音克隆

### 音频 API 示例

**TTS（文本转语音）：**
```bash
curl -X POST http://127.0.0.1:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-TTS-0.6B",
    "input": "你好，这是一个测试",
    "voice": "default",
    "speed": 1.0
  }' \
  --output output.wav
```

**ASR（语音转文本）：**
```bash
curl -X POST http://127.0.0.1:8080/v1/audio/transcriptions \
  -F "file=@audio.wav" \
  -F "model=Qwen3-ASR-0.6B"
```

### 音频测试

使用提供的 CLI 测试工具：
```bash
# 测试 TTS
python scripts/test_audio_cli.py --tts "你好世界" --output output.wav

# 测试 ASR
python scripts/test_audio_cli.py --asr output.wav

# 测试端到端管道（TTS -> ASR）
python scripts/test_audio_cli.py --pipeline "测试消息"

# 运行完整测试套件
pytest tests/test_audio_services.py -v -s
```

参见 `docs/guides/DEPLOYMENT.md` 了解详细的音频服务文档。

## 基准测试哲学

启动自动调优可以运行短预热基准测试来选择最快的稳定策略。基准测试子系统比较：

- 推测解码开/关
- 草稿令牌计数
- 前缀缓存开/关
- 批处理窗口
- 批处理上限
- 页面大小
- 调度模式
- 流式传输刷新节奏

配置文件被保留并在后续启动时使用。

## Apple Silicon 调优说明

- 优先使用预分配和页面池而不是重复的动态分配
- 小心使用 MLX 模型驻留以避免统一内存抖动
- 为每台机器基准测试前缀缓存和推测解码
- 保持 Python 热路径小；将协调移到稳定循环中
- 优先考虑长提示下的一致首令牌延迟

## 动态优化哲学

Aster 仅启用在本地机器上证明有益的优化：

- 推测解码可以全局禁用或按请求类禁用
- 当命中率低或内存压力上升时，前缀缓存可以减少或禁用
- 当延迟上升时，批处理窗口自动缩小
- 当检测到不稳定或回归时，选择回退配置文件

## 模型设置

使用 hfd + aria2 加速的一键式模型下载：

```bash
# 下载所有必需的模型（ASR、LLM、TTS）
bash scripts/setup/download_models.sh

# 或直接使用 Python 获得更多控制
python scripts/download_models.py --all
python scripts/download_models.py --group llm
python scripts/download_models.py --list
```

参见 `scripts/setup/README-model-download.md` 了解详细说明。

## 模型路径

`model.path` 和 `model.draft_path` 可以是：
- MLX 转换模型目录的绝对本地路径
- 可由 `mlx-lm` 加载的兼容 Hugging Face 仓库 ID

对于生产环境，优先使用本地 MLX 转换目录。更新 `configs/config.yaml`：

```yaml
model:
  path: models/qwen3.5-9b-mlx
  draft_path: models/qwen3.5-0.8b-mlx

audio:
  asr_model_path: models/qwen3-asr-0.6b
  tts_model_path: models/qwen3-tts-0.6b-base
```

## OpenClaw 集成

将 OpenClaw 指向 Aster 的 OpenAI 兼容基础 URL 和模型 ID。Aster 为重复的系统/工具前缀和长期 Agent 会话而构建，因此应特别受益于具有稳定脚手架和长上下文重用的工作负载。

## 项目指导文档

- `docs/guides/QUICK_START_MODELS.md` — 模型下载快速指南
- `docs/reference/MODEL_SETUP.md` — 详细设置和故障排除
- `docs/development/MODEL_DOWNLOAD_ARCHITECTURE.md` — 系统设计
- `docs/reference/ROADMAP.md` — 长期架构演进计划
- `docs/api/OPENAI_COMPAT.md` — 兼容性边界和调试扩展
- `docs/development/DEBUGGING.md` — 操作员调试指南
- `docs/operations/OPERATIONS.md` — 日常服务运维
- `docs/guides/BENCHMARK_GUIDE.md` — 性能基准测试指南
- `docs/guides/BACKGROUND_SERVICE_SETUP.md` — 后台服务设置
- `DOCS.md` — 完整文档导航
