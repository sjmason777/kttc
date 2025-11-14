# 说明

**理解导向**的概念指南和架构说明。

说明文档帮助您理解 KTTC 的工作原理以及这样设计的原因。阅读本节内容可以加深您对翻译质量评估概念和 KTTC 架构的理解。

## 架构与设计

- **[架构概览](architecture.md)** ⭐ *从这里开始！*
  - 系统架构图
  - 组件概览
  - 设计决策
  - 性能特征

## 核心概念

- **[智能体系统](agent-system.md)** - 多智能体质量评估解释
  - 为什么使用多智能体？
  - 智能体类型和职责
  - 混合 NLP + LLM 方法

- **[MQM 框架](mqm-scoring.md)** - 质量指标解释
  - 什么是 MQM？
  - 错误分类
  - 评分计算

- **[智能路由](smart-routing-explained.md)** - 智能模型选择
  - 复杂度分析
  - 路由决策逻辑
  - 成本优化

## 高级主题

- **[翻译记忆](translation-memory-explained.md)** - TM 工作原理
  - 语义搜索
  - MQM 跟踪
  - 领域组织

- **[语言特定功能](language-features-explained.md)** - 为什么需要专门的智能体？
  - 英语：LanguageTool 集成
  - 中文：HanLP 和量词
  - 俄语：MAWO NLP 栈

- **[自动修正](auto-correction-explained.md)** - 自动修复的工作原理
  - TEaR 循环（翻译-评估-改进）
  - 迭代改进
  - 收敛条件

## 研究背景

- **[MQM 和 WMT](mqm-wmt.md)** - 行业标准
- **[用于翻译质量评估的 LLM](llm-qa.md)** - LLM 方法和挑战
- **[幻觉缓解](hallucination.md)** - 反幻觉技术

## 需要实用信息？

- **如何操作？** → 参见[指南](../guides/README.md)
- **API 详情？** → 参见[参考](../reference/README.md)
- **入门教程？** → 参见[教程](../tutorials/README.md)