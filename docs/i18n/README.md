# 多语言文档指南 / Multilingual Documentation Guide

## 支持的语言 / Supported Languages

Aster 项目支持以下语言的文档：

- **English** (en) - [README](../../README.md)
- **中文** (zh) - [README](../../README.zh.md)
- **日本語** (ja) - [README](../../README.ja.md)
- **Español** (es) - [README](../../README.es.md)
- **Français** (fr) - [README](../../README.fr.md)
- **Deutsch** (de) - [README](../../README.de.md)
- **한국어** (ko) - [README](../../README.ko.md)

## 文档结构 / Documentation Structure

### 根目录文件 / Root Directory Files

主要文档以多语言形式提供在项目根目录：

- `README.md` - English
- `README.zh.md` - 中文
- `README.ja.md` - 日本語
- `README.es.md` - Español
- `README.fr.md` - Français
- `README.de.md` - Deutsch
- `README.ko.md` - 한국어

- `CONTRIBUTING.md` - English
- `CONTRIBUTING.zh.md` - 中文
- `CONTRIBUTING.ja.md` - 日本語
- (其他语言的贡献指南)

### 文档目录 / Docs Directory

`docs/` 目录包含详细的技术文档：

- `ROADMAP.md` - 架构演进计划 (English)
- `ARCHITECTURE.md` - 架构概述 (English)
- `DEBUGGING.md` - 调试指南 (English)
- `OPENAI_COMPAT.md` - OpenAI 兼容性 (English)
- `OPERATIONS.md` - 运维指南 (English)
- `DEVELOPMENT.md` - 开发指南 (English)

## 如何贡献翻译 / How to Contribute Translations

### 添加新语言 / Adding a New Language

1. 在项目根目录创建新的 README 文件：
   ```
   README.{language_code}.md
   ```

2. 复制 `README.md` 的内容并翻译

3. 在所有 README 文件的顶部添加语言链接

4. 创建对应的 `CONTRIBUTING.{language_code}.md`

5. 提交 Pull Request

### 翻译指南 / Translation Guidelines

- 保持原文的结构和格式
- 保留代码块和命令不变
- 保留文件路径和 URL 不变
- 翻译应该准确且自然
- 使用该语言的技术术语标准

### 语言代码 / Language Codes

使用 ISO 639-1 语言代码：

- `en` - English
- `zh` - 中文 (Simplified Chinese)
- `ja` - 日本語
- `es` - Español
- `fr` - Français
- `de` - Deutsch
- `ko` - 한국어

## 当前翻译状态 / Current Translation Status

| 文档 | English | 中文 | 日本語 | Español | Français | Deutsch | 한국어 |
|------|---------|------|--------|---------|----------|---------|--------|
| README | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| CONTRIBUTING | ✅ | ✅ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| DEVELOPMENT | ✅ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |

✅ = 已完成 / Completed
⏳ = 进行中 / In Progress
❌ = 需要翻译 / Needs Translation

## 翻译工具 / Translation Tools

推荐使用以下工具辅助翻译：

- [DeepL](https://www.deepl.com/) - 高质量的机器翻译
- [Google Translate](https://translate.google.com/) - 快速翻译
- [Crowdin](https://crowdin.com/) - 协作翻译平台

## 联系方式 / Contact

如有翻译相关问题，请：

1. 在 GitHub Issues 中提出
2. 提交 Pull Request
3. 联系项目维护者

感谢您的贡献！/ Thank you for your contribution!
