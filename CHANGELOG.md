# Changelog

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
