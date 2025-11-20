# Changelog

## [0.3.0] - 2025-11-20

### New Language Support

- **Hindi language support**
  - New `HindiFluencyAgent` for Hindi-specific quality checks
  - Hindi language helper with comprehensive NLP capabilities
  - Indic NLP library for tokenization and normalization
  - Stanza integration for POS tagging, NER, and lemmatization
  - Spello for spell checking
  - Language detection and auto-routing support

- **Persian (Farsi) language support**
  - New `PersianFluencyAgent` for Persian-specific quality checks
  - Persian language helper with comprehensive NLP capabilities
  - DadmaTools integration for complete Persian NLP toolkit
  - Support for tokenization, POS tagging, NER, and spell checking
  - Language detection and auto-routing support

### Evaluation Infrastructure

- **New evaluation module** (`src/kttc/evaluation/`)
  - Lightweight metrics (chrF, BLEU, TER) for CPU-based evaluation
  - Rule-based error detection framework
  - Metrics module with comprehensive scoring capabilities
  - Error detection module for linguistic analysis

- **Benchmark datasets**
  - FLORES200 integration: 8 new benchmark files (dev/devtest for en-hi, hi-en, en-fa, fa-en)
  - Quality benchmark datasets for Hindi and Persian
  - Critical bad translation datasets for error detection testing
  - Synthetic bad translation datasets for validation

### CLI Improvements

- **Enhanced output formatting**
  - New `print_lightweight_metrics()` function for displaying translation metrics
  - New `print_rule_based_errors()` function for error visualization
  - Color-coded score display with quality thresholds
  - Improved readability for metrics and errors

### Development Tools

- **Benchmark generation and validation**
  - `generate_hindi_persian_benchmarks.py` - Generate benchmark data from FLORES200
  - `validate_benchmark_data.py` - Validate benchmark file integrity
  - `clean_quality_files.py` - Clean up and format benchmark data
  - Comprehensive unit tests for Hindi and Persian components
  - Integration test documentation

### Configuration Updates

- **Enhanced language registry**
  - Hindi and Persian added to language registry with metadata
  - Recommended LLM models for Hindi and Persian
  - ISO 639-3 codes and native names
  - Resource level classification

- **Updated dependencies**
  - Optional `hindi` extra for Hindi-specific NLP libraries
  - Optional `persian` extra for Persian-specific NLP libraries
  - Optional `metrics` extra for semantic similarity tools
  - Optional `all-languages` extra for all language helpers
  - Added MANIFEST.in for proper package distribution

### Core Improvements

- **Orchestrator updates**
  - Support for Hindi and Persian language pairs
  - Enhanced language detection for new languages
  - Improved routing logic for specialized agents

- **Benchmark data quality**
  - Reformatted existing benchmark files (en-ru, ru-en, en-zh, zh-en, ru-zh, zh-ru)
  - Consistent JSON formatting across all benchmark files
  - Validated data integrity

## [0.2.0] - 2025-11-14

### Internationalization

- **Complete Russian and Chinese documentation translations**
  - Added Russian translations: 13 documentation files (docs/ru/*, *.ru.md)
  - Added Chinese translations: 13 documentation files (docs/zh/*, *.zh.md)
  - Translated: CODE_OF_CONDUCT, SECURITY, CONTRIBUTING, all guides, tutorials, and references
  - All translations validated with KTTC CLI (MQM scores 91-96)

### Major Improvements - Russian NLP

- **Implemented 6-layer morphological disambiguation system**
  - Layer 0: Custom dictionary for technical compound adjectives
  - Layer 1: POS priority for 30+ function words (prepositions/conjunctions)
  - Layer 2: Preposition-driven case selection (22+ prepositions with grammar rules)
  - Layer 3A: Adjective look-back for ADJ+ADJ patterns
  - Layer 3B: Adjective look-ahead for PREP+ADJ+NOUN patterns
  - Layer 3C: Direct ADJ+NOUN look-ahead with genitive detection
  - Layer 4: Adjective-noun agreement with animate/inanimate preference
  - Layer 5: Compound adjective heuristic detection

- **Improved Russian translation quality**
  - MQM score improvement: 73.96 → 95.15 (from FAIL to near-PASS)
  - Eliminated 100% of false positives in README.ru.md (7/7 fixed)
  - Context-aware morphological analysis with look-ahead and look-back
  - Support for substantivized adjectives with genitive complements

### Bug Fixes

- **Critical**: Fixed substantivized adjectives with genitive complements
  - Example: "переменную окружения" (environment variable) no longer triggers false positive
  - Added genitive parse detection in Layer 3C look-ahead
  - Added substantivized adjective skip in grammar checks

- **Minor**: Filtered empty linguistic errors in CLI output
  - Skip reporting errors with empty adjective/noun data
  - Cleaner CLI output without "Case mismatch: '' and ''" noise

- **Tests**: Fixed CLI test assertions with ANSI escape codes
  - Added `strip_ansi()` helper to handle Rich-formatted output
  - Fixed `test_check_help` and `test_translate_help` test failures

### CLI Improvements

- Intelligent command-line interface with auto-detection
  - Single `check` command handles check/compare/batch modes automatically
  - Smart defaults: routing enabled, auto-glossary detection, auto-format
  - Backward compatible: legacy commands still work

### Code Quality

- All code passes strict quality checks
- Fixed type annotations in tests
- Organized imports across the codebase

### Documentation

- Complete trilingual documentation (English, Russian, Chinese)
- Improved installation guides
- Updated CLI command references
- Architecture documentation in all languages

## [0.1.0] - Previous releases

See git history for earlier changes.

---

**Full Changelog**: https://github.com/xopoiii/kttc/compare/v0.1.0...v0.2.0
