<div align="center">
  <img src="assets/logo.svg" alt="Aster Logo" width="200" height="200">
  
  # Aster
  
  **为 Apple Silicon 优化的本地 LLM 推理运行时**
  
  [English](README.md) | [中文](README.zh.md) | [日本語](README.ja.md) | [Español](README.es.md) | [Français](README.fr.md) | [Deutsch](README.de.md) | [한국어](README.ko.md)
</div>

Aster 是一个为 Apple Silicon 优化的本地 LLM 推理运行时，专为长上下文和 OpenClaw 风格的 Agent 工作负载设计。

## 为什么选择 Aster

Aster 针对以下场景进行了优化：

- 超长提示词和重复的长前缀
- 工具密集型 Agent 提示词
- 长对话
- 连续本地后台服务
- 基准测试验证的运行时策略选择
- Apple Silicon + MLX 部署

它提供了 OpenAI 兼容的 API，并将高级优化视为候选策略而非教条。推测解码、前缀缓存、批处理、调度和流式传输速率都经过基准测试，并根据本地性能和稳定性进行选择。

## 核心特性

- OpenAI 兼容的 API，支持流式和非流式端点
- 显式的 prefill/decode 分离
- 具有队列感知的自适应调度器
- 分页 KV 管理器抽象
- 具有确定性哈希的自动前缀缓存
- 具有自动禁用回退的推测解码控制器
- 基准测试/自调优子系统，保持最快的稳定配置
- 结构化日志、指标、监督和就绪/健康报告

## 快速开始

```bash
cd /Users/eitan/Documents/Projects/Python/Aster
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp configs/config.yaml.example configs/config.yaml
python -m aster --config configs/config.yaml
```

## Python 版本

Aster 针对现代 Python 进行了优化，应在 Python 3.13.x（如果可用）或 3.12+ 上运行。不支持 macOS 系统 Python。

## API 端点

- `GET /health` - 健康检查
- `GET /ready` - 就绪检查
- `GET /metrics` - Prometheus 指标
- `GET /v1/models` - 模型列表
- `POST /v1/chat/completions` - 聊天完成
- `POST /v1/completions` - 文本完成

兼容性说明：
- 查看 `docs/OPENAI_COMPAT.md` 了解 Aster 的默认兼容性契约和可选的调试扩展。

## 基准测试理念

启动自调优可以运行短暂的预热基准测试来选择最快的稳定策略。基准测试子系统比较：

- 推测解码开/关
- 草稿令牌数量
- 前缀缓存开/关
- 批处理窗口
- 批处理上限
- 页面大小
- 调度模式
- 流式传输刷新速率

配置文件被保存并在后续启动时使用。

## Apple Silicon 调优说明

- 优先使用预分配和页面池而不是重复的动态分配
- 小心使用 MLX 模型驻留以避免统一内存抖动
- 为每台机器基准测试前缀缓存和推测解码
- 保持 Python 热路径小；将协调移到稳定循环中
- 优先考虑长提示词下的一致首令牌延迟

## 动态优化理念

Aster 仅启用在本地机器上证明有益的优化：

- 推测解码可以全局禁用或按请求类禁用
- 当命中率低或内存压力上升时，前缀缓存可以减少或禁用
- 当延迟上升时，批处理窗口自动缩小
- 当检测到不稳定或回归时，选择回退配置文件

## 模型路径

`model.path` 和 `model.draft_path` 可以是：
- MLX 转换模型目录的绝对本地路径
- 可由 `mlx-lm` 加载的兼容 Hugging Face 仓库 ID

对于预期的生产设置，优先使用本地 MLX 转换目录，同时用于 9B 目标和 0.8B 草稿模型。

有用的设置和验证命令：

```bash
bash scripts/setup/download_models.sh
# 或者，使用更具下载弹性的路径：
USE_HFD=1 bash scripts/setup/download_models.sh
source .venv/bin/activate
python scripts/dev/model_smoke.py --config configs/config.yaml
python scripts/dev/benchmark_live.py --config configs/config.yaml
```

## OpenClaw 集成

将 OpenClaw 指向 Aster 的 OpenAI 兼容基础 URL 和模型 ID。Aster 为重复的系统/工具前缀和长期 Agent 会话而构建，因此应特别受益于具有稳定脚手架和长上下文重用的工作负载。

## 项目文档

- `docs/ROADMAP.md` — 长期架构演进计划
- `docs/OPENAI_COMPAT.md` — 兼容性边界和调试扩展规则
- `docs/DEBUGGING.md` — 操作员调试指南
- `docs/OPERATIONS.md` — 日常服务操作
- `docs/DEVELOPMENT.md` — 开发指南

## 许可证

MIT License - 详见 [LICENSE](LICENSE)

## 贡献

欢迎贡献！请查看 [CONTRIBUTING.zh.md](CONTRIBUTING.zh.md) 了解贡献指南。
