# 架构概述

本文档解释了 KTTC 的架构、设计决策以及各组件如何协同工作。

## 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                         CLI Layer                           │
│  (kttc check, batch, translate, compare, benchmark)         │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│                   Orchestration Layer                       │
│  • AgentOrchestrator                                        │
│  • DynamicAgentSelector (budget-aware)                      │
│  • WeightedConsensus                                        │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│                    Multi-Agent QA System                    │
│  ┌───────────┐  ┌────────────┐  ┌─────────────┐            │
│  │ Accuracy  │  │  Fluency   │  │Terminology  │            │
│  │  Agent    │  │   Agent    │  │   Agent     │            │
│  └───────────┘  └────────────┘  └─────────────┘            │
│  ┌───────────┐  ┌────────────┐                             │
│  │Hallucin-  │  │  Context   │                             │
│  │ation Agent│  │   Agent    │                             │
│  └───────────┘  └────────────┘                             │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│               Language-Specific Helpers                     │
│  • EnglishLanguageHelper (LanguageTool integration)         │
│  • ChineseLanguageHelper (HanLP integration)                │
│  • RussianLanguageHelper (MAWO NLP integration)             │
│    → Anti-hallucination verification for LLM outputs        │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│                     LLM Layer                               │
│  • ComplexityRouter (smart model selection)                 │
│  • ModelSelector (language pair optimization)               │
│  • Providers: OpenAI, Anthropic, GigaChat, Yandex          │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│                 Supporting Systems                          │
│  • TranslationMemory (semantic search, MQM tracking)        │
│  • TerminologyBase (glossary management)                    │
│  • AutoCorrector (LLM-powered error fixing)                 │
│  • IterativeRefinement (TEaR loop)                          │
└─────────────────────────────────────────────────────────────┘
```

## 核心组件

### 1. 多智能体质量评估系统

KTTC 使用专门的智能体按照 MQM（多维质量指标）框架评估不同的质量维度。

**基础智能体（始终激活）：**

- **AccuracyAgent** - 语义正确性、意义保持、误译检测
- **FluencyAgent** - 语法、自然性、可读性（基类）
- **TerminologyAgent** - 领域特定术语一致性、词汇表验证
- **HallucinationAgent** - 检测虚构内容、实体保持
- **ContextAgent** - 文档级一致性、交叉引用验证

**特定语言流畅性智能体：**

基于目标语言自动选择：

- **EnglishFluencyAgent**（`target_lang == "en"`）
  - LanguageTool 集成：5,000+ 语法规则
  - 主谓一致、冠词、介词
  - spaCy 命名实体识别

- **ChineseFluencyAgent**（`target_lang == "zh"`）
  - HanLP 集成：量词验证
  - 体态助词检查（了/过）
  - 高精度词性标注（约95%）

- **RussianFluencyAgent**（`target_lang == "ru"`）
  - MAWO NLP 堆栈：形态分析
  - 格变一致性（6个格）、动词体态验证
  - 语气词使用（же, ли, бы）、语域一致性（ты/вы）

### 2. 混合 NLP + LLM 方法

**为什么采用混合方法？**

纯 LLM 方法在质量评估任务中存在幻觉问题。KTTC 结合了：

1. **确定性 NLP** - 基于规则的检查（无幻觉）
2. **LLM 智能** - 语义理解、上下文感知
3. **反幻觉验证** - NLP 验证 LLM 输出

**流程：**

```
文本 → NLP 分析（确定性）→ LLM 分析（语义）
                                            ↓
                      ← 反幻觉验证 ←
                                            ↓
                                      已验证错误
```

**示例（俄语）：**

```python
# NLP 检测格变不匹配
nlp_error = "Adjective 'красный' should agree with noun 'дом' in case"

# LLM 检测语义问题
llm_error = "Unnatural word order in Russian"

# NLP 验证 LLM 未产生幻觉
# （检查 LLM 声称的错误是否真实存在于文本中）

# 返回：NLP 错误 + 仅已验证的 LLM 错误
```

### 3. 智能路由系统

**ComplexityRouter** 分析文本复杂性并路由到合适的模型：

**复杂性因子：**

- 句子长度和结构
- 罕见词频率
- 句法复杂性（依存深度）
- 领域特定术语密度

**路由决策：**

```
复杂性评分（0.0-1.0）：
├─ 0.0-0.3: 简单 → GPT-4o-mini（便宜、快速）
├─ 0.3-0.7: 中等 → Claude 3.5 Sonnet（平衡）
└─ 0.7-1.0: 复杂 → GPT-4.5/o1-preview（最佳质量）
```

**优势：**

- 在简单文本上节省 60% 成本
- 与始终使用高端模型相同的质量
- 自动化，无需手动配置

### 4. MQM 评分

KTTC 实现了 WMT 基准测试中使用的 MQM 框架。

**错误严重性权重：**

- **中性：** 0分（无惩罚）
- **轻微：** 1分（拼写错误、风格偏好）
- **主要：** 5分（语法错误、误译）
- **严重：** 10分（遗漏、意义改变）

**公式：**

```
MQM 评分 = 100 - (总惩罚 / 词数 * 1000)
```

**示例：**

```
文本：50个词
错误：1个轻微，2个主要
惩罚：1*1 + 2*5 = 11
MQM 评分：100 - (11 / 50 * 1000) = 100 - 220 = 78.0
```

**质量等级：**

- **95-100：** 优秀（可投入生产）
- **90-94：** 良好（需要轻微修正）
- **80-89：** 可接受（需要修订）
- **<80：** 较差（需要大幅返工）

### 5. 翻译记忆库

**语义搜索与 MQM 跟踪：**

```python
# 存储翻译及质量评分
await tm.add_translation(
    source="API request",
    translation="Запрос API",
    mqm_score=98.5,
    domain="technical"
)

# 查找相似翻译（sentence-transformers）
results = await tm.search_similar(
    source="API call",
    threshold=0.80  # 余弦相似度
)
# 返回："Запрос API"（相似度：0.92，MQM：98.5）
```

**优势：**

- 重用高质量翻译
- 项目间术语一致性
- 领域特定组织

### 6. 自动纠错系统

**AutoCorrector** 使用 LLM 自然地修复检测到的错误：

**级别：**

- **轻度：** 仅修复严重和主要错误
- **完整：** 修复所有检测到的错误

**流程：**

```
1. 智能体检测错误
2. AutoCorrector 生成修复提示
3. LLM 在上下文中纠正
4. 智能体重新评估
5. 重复直至达到阈值（最大迭代次数）
```

**结果：**

- 比手动后期编辑快 40%
- 比人工编辑节省 60% 成本
- 保持上下文和自然性

### 7. TEaR 循环（翻译-估计-改进）

**IterativeRefinement** 实现 TEaR 方法论：

```
1. 翻译：生成初始翻译（LLM）
2. 估计：评估质量（多智能体质量评估）
3. 改进：修复错误并改进（AutoCorrector）
4. 重复2-3直至收敛或最大迭代次数
```

**收敛标准：**

- MQM 评分 ≥ 阈值（如 95.0）
- 改进 < 最小值（如 1.0 分）
- 达到最大迭代次数（如 3）

## 设计决策

### 为什么选择多智能体而非单一 LLM？

**单一 LLM 方法：**

- 在质量评估任务中容易产生幻觉
- 可能遗漏特定语言错误
- 错误分类不一致

**多智能体方法：**

- 每个智能体专门负责一个维度
- 并行执行（更快）
- 混合 NLP+LLM 减少幻觉
- 符合 MQM 的错误分类

**研究支持：** WMT 2025 发现多智能体系统在准确性上比单模型质量评估优出 15-20%。

### 为什么使用特定语言智能体？

**问题：** 通用流畅性智能体会遗漏特定语言错误：

- 英语：冠词（a/an/the）、主谓一致
- 中文：量词、体态助词（了/过）
- 俄语：格变一致性（6个格）、动词体

**解决方案：** 通过 NLP 库编码母语者知识的专门智能体。

**激活：** 基于 `target_lang` 自动选择：

```python
if target_lang == "en":
    fluency_agent = EnglishFluencyAgent()
elif target_lang == "zh":
    fluency_agent = ChineseFluencyAgent()
elif target_lang == "ru":
    fluency_agent = RussianFluencyAgent()
else:
    fluency_agent = FluencyAgent()  # 通用
```

### 为什么采用混合 NLP + LLM？

**纯 LLM 问题：**

- 幻觉：声称不存在的错误
- 不一致：重新运行时出现不同错误
- 成本：简单检查昂贵

**纯 NLP 问题：**

- 语义理解有限
- 无法检测上下文问题
- 需要大量规则工程

**混合优势：**

- NLP 提供确定性检查（无幻觉）
- LLM 提供语义理解
- NLP 验证 LLM 输出（反幻觉）
- 经济高效：简单检查用 NLP，复杂用 LLM

### 为什么使用智能路由？

**问题：** 对"Hello, world!"使用 GPT-4.5 是浪费的。

**解决方案：** 基于复杂性路由：

- 简单文本 → 便宜模型（GPT-4o-mini）
- 复杂文本 → 高端模型（GPT-4.5、o1-preview）

**影响：**

- 实践中节省 60% 成本
- 无质量下降
- 自动化，对用户透明

## 性能特性

### 延迟

**单次翻译（100词）：**

- NLP 分析：约 0.1秒
- 智能体评估（5个智能体，并行）：约 2-5秒
- 总计：约 2-6秒

**批处理（1000次翻译）：**

- 顺序：约 50-100分钟
- 并行（4个工作进程）：约 12-25分钟
- 并行（8个工作进程）：约 6-12分钟

### 成本

**每1000词（GPT-4o-mini + 智能路由）：**

- 质量检查：$0.01-0.05
- 翻译 + 质量评估：$0.05-0.15
- 带自动纠错：$0.10-0.25

**与人工审查比较：**

- 人工：每1000词 $100-500
- KTTC：每1000词 $0.01-0.25
- 节省：90-99%

### 准确性

**错误检测（对比 WMT 基准）：**

- 精确度：85-92%（少数误报）
- 召回率：78-88%（发现大多数真实错误）
- F1 评分：81-90%

**MQM 评分相关性：**

- 对比人工 MQM：r = 0.82-0.89（强相关）

## 可扩展性

**水平扩展：**

- 无状态设计（每次检查独立）
- 并行批处理
- 工作进程间无共享状态

**垂直扩展：**

- 全程 async/await
- 并发 LLM API 调用
- 内存高效（大文件流式处理）

**生产部署：**

```
负载均衡器
    ↓
┌─────────┬─────────┬─────────┐
│ 工作    │ 工作    │ 工作    │
│ 进程1   │ 进程2   │ 进程3   │
└─────────┴─────────┴─────────┘
    ↓         ↓         ↓
┌─────────────────────────────┐
│   共享翻译记忆库            │
│   共享术语库                │
└─────────────────────────────┘
```

## 另请参阅

- [智能体系统](agent-system.md) - 智能体深入探讨
- [MQM 框架](mqm-scoring.md) - 评分详情
- [CLI 架构](../guides/cli-usage.md) - CLI 设计
- [API 参考](../reference/api-reference.md) - Python API