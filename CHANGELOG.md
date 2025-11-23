# Changelog

## [0.4.0] - 2025-11-23

### New Features

- **Language Traps Detection** - 30+ glossaries for detecting false friends and translation pitfalls
- **Proofreading Mode** - School curriculum glossaries for educational content validation
- **Style Preservation** - Literary translation evaluation with tone/register checking
- **CLI in 5 Languages** - Interface localized to English, Russian, Chinese, Hindi, Persian

### Interactive Demos

- Google Colab, Streamlit web app, GitHub Codespaces support

### Breaking Changes

- spaCy moved to optional `[nlp]` extra (pydantic conflict)
- jieba moved to `[chinese]` extra

## [0.3.0] - 2025-11-20

### New Features

- **Hindi language support** - Full NLP pipeline with Indic NLP, Stanza, Spello
- **Persian language support** - Full NLP pipeline with DadmaTools
- **Evaluation module** - Lightweight metrics (chrF, BLEU, TER) and rule-based error detection
- **FLORES200 benchmarks** - Test datasets for Hindi and Persian translation pairs

## [0.2.0] - 2025-11-14

### New Features

- **Russian morphological disambiguation** - 6-layer system eliminating false positives
- **Documentation in 3 languages** - Complete Russian and Chinese translations
- **Smart CLI** - Auto-detection of check/compare/batch modes

### Improvements

- MQM accuracy: 73.96 → 95.15 for Russian

## [0.1.0] - 2025-11-01

- Initial release with core MQM evaluation pipeline
- Support for English, Russian, Chinese language pairs
- Multi-agent architecture with specialized fluency agents
- GigaChat, OpenAI, Anthropic LLM providers

---

**Full Changelog**: https://github.com/xopoiii/kttc/compare/v0.3.0...v0.4.0
