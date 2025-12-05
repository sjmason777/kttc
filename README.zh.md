<p align="center">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="docs/content/assets/img/kttc.logo-dark.png">
      <source media="(prefers-color-scheme: light)" srcset="docs/content/assets/img/kttc.logo.png">
      <img alt="KTTC" title="KTTC" src="docs/content/assets/img/kttc.logo.png">
    </picture>
</p>

[English](README.md) · [Русский](README.ru.md) · **中文** · [हिन्दी](README.hi.md) · [فارسی](README.fa.md)

# KTTC - Knowledge Translation Transmutation Core

> **📖 完整中文文档:** [docs/zh/README.md](docs/zh/README.md)

---

**基于人工智能的自主翻译质量保证**

KTTC 使用专门的多智能体系统，根据行业标准 MQM（多维质量指标，Multidimensional Quality Metrics）框架自动检测、分析和修复翻译质量问题。在几秒钟内获得生产就绪的翻译质量。

---

## 主要特性

### 核心分析
- **多智能体质量保证系统** - 专门的智能体分析准确性、流畅性、术语、风格、幻觉和上下文
- **MQM 评分** - WMT 基准测试中使用的行业标准质量指标
- **翻译指标** - BLEU、TER（翻译编辑率）、chrF（字符 F 分数）通过 sacrebleu
- **特定语言智能体** - 5 个母语级流畅度智能体：英语、中文、俄语、印地语、波斯语
- **60+ 领域词汇表** - 汽车、海关、金融、法律、物流、医疗术语 + 语言陷阱

### 智能体编排
- **加权共识** - 具有不同信任权重的智能体，置信度评分，一致性指标
- **多智能体辩论** - 智能体间交叉验证，减少 30-50% 误报
- **动态智能体选择** - 根据文本复杂度自动选择 2-5 个智能体（节省 30-50% 成本）
- **智能体预设** - `minimal`（2 个）、`default`（3 个）、`full`（5 个）快速配置
- **自我评估重试** - 智能体评估自己的置信度，在低确定性时重试

### MQM 配置文件系统
- **内置配置文件** - default、strict、minimal、legal、medical、marketing、literary、technical
- **YAML 自定义配置** - 定义智能体选择、权重、严重度乘数、阈值
- **领域适配** - 医疗：98% 阈值 + 幻觉智能体；法律：术语聚焦
- **质量门控** - 按领域可配置的通过/失败阈值（88-98%）

### 文学风格分析
- **StyleFingerprint** - 自动检测文学风格模式（Burrows Delta 方法）
- **风格保持智能体** - 评估文学翻译中作者声音的保持
- **国家特定模式** - 列斯科夫民间叙事、普拉托诺夫冗余、乔伊斯意识流、哈菲兹加扎尔、恰亚瓦德诗歌
- **流畅度容差** - 可调整的流畅度权重，用于有意的风格偏离

### 语言智能
- **语言陷阱检测** - 60+ 词汇表检测同音词、假朋友、成语、短语动词、近义词
- **自我检查 / 校对** - 基于学校课程标准的语法、拼写、标点检查
- **学校课程** - 部编版（中国）、ФГОС（俄罗斯）、UK GPS（英国）、NCERT（印度）、伊朗语法
- **快速检查模式** - 无需 LLM 的规则检查，适合 CI/CD 和 pre-commit 钩子
- **自动修正** - 基于大语言模型的错误修复与迭代优化（TEaR 循环）

### 企业基础设施
- **智能路由** - 基于文本复杂度自动选择最优模型（节省 60% 成本）
- **XLSX 导出** - 带有 Summary、Errors、Breakdown 工作表的 Excel 报告，用于企业报告
- **翻译记忆库** - 具有质量跟踪和重用的语义搜索
- **术语表管理** - 自定义术语 + 60+ 内置多语言词汇表
- **批量处理** - 并行处理数千个翻译，支持 CSV/JSON 过滤
- **CI/CD 就绪** - GitHub Actions 集成、退出代码、JSON/Markdown/HTML/XLSX 输出
- **多 LLM 支持** - OpenAI、Anthropic、Google Gemini、GigaChat、YandexGPT，集成 LanguageTool
- **机器翻译** - DeepL API 集成，提供高质量翻译
- **RAG 上下文** - 基于 BM25 的术语表检索（轻量级，仅 CPU，默认禁用）
- **仲裁工作流** - AI 辅助或人工参与的 QA 错误争议解决
- **QA 触发器** - 文件更改、阈值违规、计划任务时的主动检查（CI/CD 就绪）
- **使用分析** - 报告中包含令牌计数、API 成本和调用统计

**性能：** 相比人工审核降低 90% 成本 • 快 100-1000 倍 • MQM 质量目标 95+

---

## 在线试用 KTTC

无需安装即可体验 KTTC：

[![Open in Colab](https://img.shields.io/badge/Open_in_Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/github/kttc-ai/kttc/blob/main/examples/kttc_demo.ipynb)
[![Streamlit Demo](https://img.shields.io/badge/Streamlit_Demo-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://kttc-demo.streamlit.app)
[![Open in Codespaces](https://img.shields.io/badge/Open_in_Codespaces-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/codespaces/new?repo=kttc-ai/kttc)

- **Google Colab** - 带有示例的交互式教程（5 分钟，无需设置）
- **Streamlit 演示** - 测试您自己的翻译的网页界面（无需编码）
- **GitHub Codespaces** - 浏览器中的完整开发环境（适合贡献者）

---

## 快速开始

### 1. 安装

```bash
pip install kttc
```

可选的语言增强功能：

```bash
pip install kttc[english]        # 英语: LanguageTool（5,000+ 语法规则）
pip install kttc[chinese]        # 中文: HanLP（量词、语气助词）
pip install kttc[hindi]          # 印地语: Indic NLP + Stanza + Spello
pip install kttc[persian]        # 波斯语: DadmaTools（基于 spaCy）
pip install kttc[all-languages]  # 所有语言助手
```

### 2. 设置 API 密钥

```bash
export KTTC_OPENAI_API_KEY="sk-..."
# 或
export KTTC_ANTHROPIC_API_KEY="sk-ant-..."
# 或
export KTTC_GEMINI_API_KEY="AIza..."
```

**可选（用于机器翻译和 RAG）：**
```bash
export KTTC_DEEPL_API_KEY="..."        # DeepL 翻译
export KTTC_RAG_ENABLED="true"         # 启用 BM25 上下文检索
```

### 3. 检查翻译质量

```bash
kttc check source.txt translation.txt --source-lang en --target-lang zh
```

**输出：**

```
✅ MQM Score: 96.5 (PASS - 优秀质量)
📊 5 个智能体分析了翻译
⚠️  发现 2 个轻微问题，0 个主要问题，0 个严重问题
✓ 达到质量阈值 (≥95.0)
```

就是这样！KTTC 开箱即用，具有智能默认设置：
- ✅ 智能路由（自动为简单文本选择更便宜的模型）
- ✅ 自动术语表（如果存在则使用 'base' 术语表）
- ✅ 自动格式（从文件扩展名检测输出格式）

---

## 命令

```bash
kttc check source.txt translation.txt          # 单个质量检查
kttc check source.txt t1.txt t2.txt t3.txt     # 自动比较多个翻译
kttc check translations.csv                     # 自动检测批处理模式
kttc check source_dir/ trans_dir/              # 目录批处理

kttc batch --file translations.csv              # 显式批处理
kttc compare --source src.txt -t t1 -t t2      # 并排比较翻译
kttc translate --text "Hello" --source-lang en --target-lang zh  # 带 QA 的翻译
kttc benchmark --source text.txt --providers openai,anthropic    # LLM 基准测试

# 术语表管理（项目 + 用户全局存储）
kttc glossary list                              # 列出所有术语表
kttc glossary create tech --from-csv terms.csv  # 创建项目术语表
kttc glossary create personal --from-csv my.csv --user  # 创建用户术语表

# 🥚 自我检查 / 校对（新功能！）
kttc check article.md --self --lang zh          # 无需翻译的校对
kttc proofread article.md --lang zh             # 同上（别名）
kttc lint article.md --lang zh --fix            # 快速规则检查（无 LLM）
```

**查看完整命令参考：** [docs/zh/reference/cli-commands.md](docs/zh/reference/cli-commands.md)

---

## Python API

```python
import asyncio
from kttc.agents import AgentOrchestrator
from kttc.llm import OpenAIProvider
from kttc.core import TranslationTask

async def check_quality():
    llm = OpenAIProvider(api_key="your-key")
    orchestrator = AgentOrchestrator(llm)

    task = TranslationTask(
        source_text="Hello, world!",
        translation="你好，世界！",
        source_lang="en",
        target_lang="zh",
    )

    report = await orchestrator.evaluate(task)
    print(f"MQM Score: {report.mqm_score}")
    print(f"Status: {report.status}")

asyncio.run(check_quality())
```

**查看完整 API 参考：** [docs/zh/reference/api-reference.md](docs/zh/reference/api-reference.md)

---

## 📚 文档

**完整的中文文档：** [docs/zh/README.md](docs/zh/README.md)

### 快速链接

- **[快速入门指南](docs/zh/tutorials/README.md)** - 5 分钟入门
- **[安装指南](docs/zh/guides/README.md)** - 详细安装说明
- **[CLI 参考](docs/zh/reference/README.md)** - 所有命令和选项
- **[架构](docs/zh/explanation/README.md)** - KTTC 工作原理

### 文档结构

遵循 [Diátaxis](https://diataxis.fr/) 文档框架：

- 📚 **[教程](docs/zh/tutorials/README.md)** - 边做边学（分步指南）
- 📖 **[指南](docs/zh/guides/README.md)** - 解决具体问题（操作指南）
- 📋 **[参考](docs/zh/reference/README.md)** - 查找技术细节（API、CLI）
- 💡 **[说明](docs/zh/explanation/README.md)** - 理解概念（架构、设计）

---

## 贡献

我们欢迎贡献！请参阅 [CONTRIBUTING.md](CONTRIBUTING.md) 了解指南。

---

## 许可证

采用 Apache License 2.0 许可。详见 [LICENSE](LICENSE)。

Copyright 2025 KTTC AI (https://github.com/kttc-ai)

---

## 链接

- 📦 [PyPI 包](https://pypi.org/project/kttc/)
- 📖 [文档](docs/zh/)
- 🐛 [问题跟踪器](https://github.com/kttc-ai/kttc/issues)
- 💬 [讨论](https://github.com/kttc-ai/kttc/discussions)
- 🇺🇸 [English Version](README.md)
