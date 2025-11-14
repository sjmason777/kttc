# Explanation

**Understanding-oriented** conceptual guides and architecture.

Explanation documentation helps you understand how KTTC works and why it was designed this way. Read this section to deepen your knowledge of translation QA concepts and KTTC's architecture.

## Architecture & Design

- **[Architecture Overview](architecture.md)** ⭐ *Start here!*
  - System architecture diagram
  - Component overview
  - Design decisions
  - Performance characteristics

## Core Concepts

- **[Agent System](agent-system.md)** - Multi-agent QA explained
  - Why multi-agent?
  - Agent types and responsibilities
  - Hybrid NLP + LLM approach

- **[MQM Framework](mqm-scoring.md)** - Quality metrics explained
  - What is MQM?
  - Error categorization
  - Scoring calculation

- **[Smart Routing](smart-routing-explained.md)** - Intelligent model selection
  - Complexity analysis
  - Routing decision logic
  - Cost optimization

## Advanced Topics

- **[Translation Memory](translation-memory-explained.md)** - How TM works
  - Semantic search
  - MQM tracking
  - Domain organization

- **[Language-Specific Features](language-features-explained.md)** - Why specialized agents?
  - English: LanguageTool integration
  - Chinese: HanLP and measure words
  - Russian: MAWO NLP stack

- **[Auto-Correction](auto-correction-explained.md)** - How automatic fixing works
  - TEaR loop (Translate-Estimate-Refine)
  - Iterative refinement
  - Convergence criteria

## Research Background

- **[MQM and WMT](mqm-wmt.md)** - Industry standards
- **[LLM for Translation QA](llm-qa.md)** - LLM approaches and challenges
- **[Hallucination Mitigation](hallucination.md)** - Anti-hallucination techniques

## Need Practical Information?

- **How to do something?** → See [Guides](../guides/README.md)
- **API details?** → See [Reference](../reference/README.md)
- **Getting started?** → See [Tutorials](../tutorials/README.md)
