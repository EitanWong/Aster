# 为 Aster 做贡献

感谢您对 Aster 的贡献兴趣！我们欢迎来自社区的贡献。

## 快速开始

### 前置条件

- Python 3.12 或更高版本（推荐 3.13）
- macOS with Apple Silicon (M1/M2/M3 或更新)
- MLX 框架已安装

### 开发环境设置

1. 克隆仓库：
```bash
git clone https://github.com/yourusername/aster.git
cd aster
```

2. 创建虚拟环境：
```bash
python3.13 -m venv .venv
source .venv/bin/activate
```

3. 安装开发依赖：
```bash
pip install -e ".[dev]"
```

4. 下载模型（可选）：
```bash
bash scripts/setup/download_models.sh
```

## 开发工作流

### 代码风格

我们使用：
- **Ruff** 用于代码检查和导入排序
- **Pyright** 用于类型检查（严格模式）
- **Black** 用于代码格式化（通过 Ruff）

提交前运行检查：
```bash
make lint
make type-check
```

### 测试

运行测试：
```bash
make test
```

运行测试并生成覆盖率报告：
```bash
make test-cov
```

### 项目结构

```
aster/
├── api/              # OpenAI 兼容 API 路由
├── scheduler/        # 请求调度和批处理
├── inference/        # 推理引擎（prefill、decode）
├── cache/            # KV 缓存管理
├── autotune/         # 策略选择和基准测试
├── core/             # 配置和生命周期
├── telemetry/        # 日志和指标
└── workers/          # 工作进程监督
```

## 进行更改

1. 创建特性分支：
```bash
git checkout -b feature/your-feature-name
```

2. 进行更改并确保测试通过：
```bash
make test
make lint
make type-check
```

3. 使用清晰的消息提交：
```bash
git commit -m "feat: add new feature"
git commit -m "fix: resolve issue with X"
git commit -m "docs: update README"
```

4. 推送到您的分支并创建 Pull Request

## 提交消息约定

我们遵循常规提交：
- `feat:` - 新功能
- `fix:` - 错误修复
- `docs:` - 文档
- `test:` - 测试添加/更改
- `refactor:` - 代码重构
- `perf:` - 性能改进
- `chore:` - 构建、依赖等

示例：
```
feat: add continuous batching support

- Implement token-level batch formation
- Add batch resizing logic
- Update scheduler to manage active batches

Closes #123
```

## Pull Request 流程

1. 确保所有测试通过
2. 如需要，更新文档
3. 为新功能添加测试
4. 遵循代码风格指南
5. 编写清晰的 PR 描述说明更改

## 报告问题

报告错误时，请包括：
- Python 版本和操作系统
- 复现步骤
- 预期行为 vs 实际行为
- 相关日志或错误消息
- 您的硬件信息（M1/M2/M3、RAM 等）

## 性能考虑

贡献优化时：
1. 使用基准测试测量前后性能
2. 如果可能，在多个硬件配置上测试
3. 确保稳定性不受影响
4. 记录性能影响

## 问题？

如有贡献相关问题，请随时：
- 在 GitHub Issues 中提出
- 提交讨论
- 联系项目维护者

感谢您帮助改进 Aster！
