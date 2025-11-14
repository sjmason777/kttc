# 快速入门指南

欢迎使用 KTTC！本教程将在 5 分钟内指导您完成第一次翻译质量检查。

## 您将学到什么

- 如何安装 KTTC
- 如何设置您的 API 密钥
- 如何检查翻译质量
- 如何解读结果

## 前置要求

- Python 3.11 或更高版本
- OpenAI 或 Anthropic API 密钥

## 步骤 1：安装

使用 pip 安装 KTTC：

```bash
pip install kttc
```

这将安装核心包（约 50MB）。针对特定语言增强功能：

```bash
# 英语语法检查（需要 Java 17+）
pip install kttc[english]

# 中文 NLP 功能
pip install kttc[chinese]

# 所有语言助手
pip install kttc[all-languages]
```

## 步骤 2：设置您的 API 密钥

设置您的 LLM 提供商 API 密钥：

```bash
# OpenAI（推荐初学者使用）
export KTTC_OPENAI_API_KEY="sk-..."

# 或 Anthropic
export KTTC_ANTHROPIC_API_KEY="sk-ant-..."
```

**提示：** 将此添加到您的 `~/.bashrc` 或 `~/.zshrc` 中以使其永久生效。

## 步骤 3：创建测试文件

创建源文本文件：

```bash
echo "Hello, world! This is a test." > source.txt
```

创建翻译文件：

```bash
echo "¡Hola, mundo! Esto es una prueba." > translation.txt
```

## 步骤 4：运行您的第一次质量检查

运行 KTTC 的智能检查命令：

```bash
kttc check source.txt translation.txt \
  --source-lang en \
  --target-lang es
```

**注意：** `kttc check` 自动检测操作模式：
- 单个文件 → 质量检查
- 多个翻译 → 自动比较
- CSV/JSON → 批量处理

## 步骤 5：理解结果

KTTC 将输出：

```
✓ Translation Quality Check
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✓ Step 1/3: Linguistic analysis complete
✓ Step 2/3: Quality assessment complete
✓ Step 3/3: Report ready

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 Quality Assessment Results
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ MQM Score: 96.5 (PASS - Excellent Quality)

📊 5 agents analyzed translation
⚠️  Found 2 minor issues, 0 major, 0 critical
✓ Quality threshold met (≥95.0)
```

### 理解 MQM 分数

- **95-100：** 优秀（可用于生产）
- **90-94：** 良好（需要轻微修复）
- **80-89：** 可接受（需要修订）
- **<80：** 较差（需要大幅返工）

## 下一步

现在您已经运行了第一次质量检查，可以探索：

- [批量处理](../guides/batch-processing.md) - 处理多个翻译
- [自动纠错](../guides/auto-correction.md) - 自动修复检测到的错误
- [术语表](../guides/glossary-management.md) - 使用自定义术语
- [智能路由](../guides/smart-routing.md) - 通过智能模型选择优化成本

## 故障排除

### "找不到 API 密钥" 错误

确保您已设置环境变量：

```bash
echo $KTTC_OPENAI_API_KEY
```

如果为空，请重新设置并在同一终端会话中重试。

### "找不到模块" 错误

确保您已安装 KTTC：

```bash
pip install kttc
```

对于特定语言功能，请安装额外包：

```bash
pip install kttc[english]  # 用于 LanguageTool
```

### Python 版本错误

KTTC 需要 Python 3.11+。检查您的版本：

```bash
python3 --version
```

如果您已安装 3.11，请明确使用它：

```bash
python3.11 -m pip install kttc
```

## 演示模式

想在不调用 API 的情况下试用 KTTC？

```bash
kttc check source.txt translation.txt \
  --source-lang en \
  --target-lang es \
  --demo
```

这使用模拟响应，让您可以在不产生费用的情况下探索 CLI。