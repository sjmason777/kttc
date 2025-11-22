# KTTC Multi-Lingual Glossaries

Comprehensive linguistic glossaries for translation quality assessment in 5 languages.

## 📚 Overview

This directory contains authoritative glossaries that power KTTC's quality agents:
- **MQM (Multidimensional Quality Metrics)** - W3C standard error taxonomy for translation quality
- **NLP/Computational Linguistics** - Natural language processing terminology
- **Language-Specific Grammar** - Grammatical features and rules for each language

These glossaries are automatically loaded by quality agents and used for error detection, validation, and enrichment.

## 🌍 Supported Languages

| Language | Code | Grammar Files | School Curriculum | Status |
|----------|------|---------------|-------------------|--------|
| English | `en` | mqm_core.json, nlp_terms.json, grammar_advanced.json, **mqm_error_taxonomy.json**, **translation_metrics.json**, **transformer_nlp_terms.json**, **llm_terminology.json** | 🆕 `school_curriculum/spelling_uk_gps.json` | ✅ Complete + **Enhanced** |
| Russian | `ru` | mqm_core_ru.json, nlp_terms_ru.json, morphology_ru.json | 🆕 `school_curriculum/orthography_fgos.json`, `punctuation_fgos.json` | ✅ Complete + **Enhanced** |
| Chinese | `zh` | mqm_core_zh.json, nlp_terms_zh.json, classifiers_zh.json, **idioms_expressions_zh.json** | 🆕 `school_curriculum/grammar_pep.json` | ✅ Complete + **Enhanced** |
| Persian | `fa` | mqm_core_fa.json, nlp_terms_fa.json, grammar_fa.json | 🆕 `school_curriculum/grammar_iranian.json` | ✅ Complete + **Enhanced** |
| Hindi | `hi` | mqm_core_hi.json, nlp_terms_hi.json, cases_hi.json | 🆕 `school_curriculum/grammar_ncert.json` | ✅ Complete + **Enhanced** |

**NEW (2025-11-21):** Added 7 new comprehensive glossaries with 402+ terms for translation quality assessment and modern NLP/LLM terminology.

## 📁 Directory Structure

```
glossaries/
├── README.md                           # This file
├── base.json                           # Base glossary (shared terms)
├── technical.json                      # Technical terminology
├── en/                                 # English glossaries
│   ├── mqm_core.json                   # MQM error types & metrics
│   ├── nlp_terms.json                  # NLP/CL terminology
│   ├── grammar_advanced.json           # Advanced English grammar
│   ├── 🆕 mqm_error_taxonomy.json       # Complete MQM error classification (47 terms)
│   ├── 🆕 translation_metrics.json      # BLEU, COMET, TER, evaluation methods (32 terms)
│   ├── 🆕 transformer_nlp_terms.json    # Attention, embeddings, tokenization (58 terms)
│   ├── 🆕 llm_terminology.json          # Hallucinations, RLHF, alignment, RAG (45 terms)
│   └── 🆕 school_curriculum/            # UK GPS spelling & grammar rules
│       └── spelling_uk_gps.json        # Homophones, common errors, apostrophes
├── ru/                                 # Russian glossaries
│   ├── mqm_core_ru.json                # MQM на русском
│   ├── nlp_terms_ru.json               # NLP терминология
│   ├── morphology_ru.json              # Морфология (cases, aspects, gender, 70+ terms)
│   └── 🆕 school_curriculum/            # ФГОС orthography & punctuation
│       ├── orthography_fgos.json       # НЕ/НИ, hyphens, ЖИ-ШИ rules
│       └── punctuation_fgos.json       # Commas, colons, quotation marks
├── zh/                                 # Chinese glossaries
│   ├── mqm_core_zh.json                # MQM 中文
│   ├── nlp_terms_zh.json               # NLP 术语
│   ├── classifiers_zh.json             # 量词 (measure words, 150+ terms)
│   ├── 🆕 idioms_expressions_zh.json    # 成语、惯用语、歇后语、谚语 (120 terms)
│   └── 🆕 school_curriculum/            # 部编版 PEP grammar rules
│       └── grammar_pep.json            # 量词, 的/地/得, punctuation
├── fa/                                 # Persian glossaries
│   ├── mqm_core_fa.json                # MQM فارسی
│   ├── nlp_terms_fa.json               # اصطلاحات NLP
│   ├── grammar_fa.json                 # دستور (ezafe, compound verbs, 80+ terms)
│   └── 🆕 school_curriculum/            # Iranian curriculum grammar
│       └── grammar_iranian.json        # Ezafe, nim-fasele, verb conjugation
└── hi/                                 # Hindi glossaries
    ├── mqm_core_hi.json                # MQM हिन्दी
    ├── nlp_terms_hi.json               # NLP शब्दावली
    ├── cases_hi.json                   # कारक (8 cases, 60+ terms)
    └── 🆕 school_curriculum/            # NCERT grammar rules
        └── grammar_ncert.json          # Sandhi, samas, spelling rules
```

**Total Glossary Files:** 31 (7 NEW in 2025-11-21 expansion + 7 school_curriculum files)
**Total Terms:** 1200+ across all languages

## Usage

### For Users

See user-facing documentation:
- **[English Guide](../docs/en/guides/glossary-management.md)**
- **[Russian Guide](../docs/ru/guides/glossary-management.md)**
- **[Chinese Guide](../docs/zh/guides/glossary-management.md)**
- **[Persian Guide](../docs/fa/guides/glossary-management.md)**
- **[Hindi Guide](../docs/hi/guides/glossary-management.md)**

### For Developers

#### Python API

```python
from kttc.terminology import GlossaryManager

# Initialize glossary manager
manager = GlossaryManager()

# Load a specific glossary
mqm_glossary = manager.load_glossary("en", "mqm_core")
# Returns: Dict[str, Any] with glossary contents

# Load Russian morphology
morphology = manager.load_glossary("ru", "morphology_ru")

# Load Chinese classifiers
classifiers = manager.load_glossary("zh", "classifiers_zh")

# Load all glossaries for a language
all_russian = manager.load_all_for_language("ru")
# Returns: Dict[str, Dict[str, Any]]
# Keys: "mqm_core_ru", "nlp_terms_ru", "morphology_ru"

# Search across glossaries
results = manager.search_in_glossaries(
    language="en",
    query="mistranslation",
    case_sensitive=False
)
# Returns: List[Tuple[str, str]] - [(glossary_type, path)]
```

#### Validation API

```python
from kttc.terminology import TermValidator

# Initialize validator
validator = TermValidator()

# Validate MQM error type
is_valid, info = validator.validate_mqm_error_type("mistranslation", "en")
# Returns: (bool, Dict[str, Any])
# info contains: definition, severity, category, etc.

# Get category weights (for MQM scoring)
from kttc.core.mqm import MQMScorer

scorer = MQMScorer(use_glossary_weights=True)
# Loads weights from glossaries/en/mqm_core.json
# Example weights: accuracy=1.5, fluency=1.0, style=0.8

# Custom weights
custom_weights = {"accuracy": 2.0, "fluency": 1.5}
score = scorer.calculate_score(errors, word_count=100, custom_weights=custom_weights)
```

#### Language-Specific Validators

```python
from kttc.terminology import (
    RussianCaseAspectValidator,
    ChineseMeasureWordValidator,
    HindiPostpositionValidator,
    PersianEzafeValidator
)

# Russian validator (uses morphology_ru.json)
russian_validator = RussianCaseAspectValidator()
case_info = russian_validator.get_case_info("genitive")
aspect_rules = russian_validator.get_aspect_usage_rules("perfective")

# Chinese validator (uses classifiers_zh.json)
chinese_validator = ChineseMeasureWordValidator()
classifiers = chinese_validator.get_classifier_by_category("individual_classifiers")
common = chinese_validator.get_most_common_classifiers(limit=10)

# Hindi validator (uses cases_hi.json)
hindi_validator = HindiPostpositionValidator()
case_info = hindi_validator.get_case_info(1)  # Karta case
oblique_rules = hindi_validator.get_oblique_form_rule()

# Persian validator (uses grammar_fa.json)
persian_validator = PersianEzafeValidator()
ezafe_rules = persian_validator.get_ezafe_rules()
verb_info = persian_validator.get_compound_verb_info("کردن")
```

### Command Line

```bash
# List available glossaries
kttc glossary list

# View specific glossary
kttc glossary view --language ru --name morphology_ru

# Search in glossaries
kttc glossary search --language zh --query "量词"

# Export glossary to JSON
kttc glossary view --language en --name mqm_core --format json > mqm_export.json

# Validate error type
kttc terminology validate --error-type "mistranslation" --language en

# List MQM categories
kttc terminology list-categories

# Get error definition
kttc terminology define --error-type "grammar" --language ru
```

## 📊 Glossary Format

All glossaries follow this JSON structure:

```json
{
  "metadata": {
    "name": "Glossary Name",
    "version": "1.0",
    "language": "en",
    "description": "Description of this glossary"
  },
  "categories": {
    "category_name": {
      "weight": 1.5,
      "subcategories": {
        "term_name": {
          "definition": "Clear definition",
          "examples": ["example1", "example2"],
          "related_terms": ["term1", "term2"],
          "severity_guidelines": "Major or Critical"
        }
      }
    }
  }
}
```

### Language-Specific Formats

Each language has specialized structures:

**Russian (morphology_ru.json)**:
```json
{
  "cases": {
    "nominative": {
      "name": "Именительный",
      "usage": ["Subject", "Predicate nominative"],
      "question": "кто? что?",
      "examples": ["студент читает"]
    }
  },
  "aspects": {
    "perfective": {
      "usage": "Completed actions",
      "markers": ["prefixes: по-, на-, с-"]
    }
  }
}
```

**Chinese (classifiers_zh.json)**:
```json
{
  "individual_classifiers": {
    "个": {
      "usage": "General classifier",
      "examples": ["一个人", "三个苹果"]
    },
    "本": {
      "usage": "Books, volumes",
      "examples": ["一本书", "两本杂志"]
    }
  }
}
```

**Hindi (cases_hi.json)**:
```json
{
  "cases": {
    "1": {
      "name": "कर्ता कारक",
      "english_name": "Nominative",
      "postposition": null,
      "usage": ["Subject of sentence"]
    },
    "2": {
      "name": "कर्म कारक",
      "english_name": "Accusative",
      "postposition": "को",
      "usage": ["Direct object"]
    }
  }
}
```

**Persian (grammar_fa.json)**:
```json
{
  "ezafe": {
    "rules": {
      "usage": "Connecting noun to adjective or noun to noun",
      "examples": ["کتاب خوب", "خانه پدر"]
    }
  },
  "compound_verbs": {
    "کردن": {
      "usage": "Common light verb",
      "examples": ["کار کردن", "صحبت کردن"]
    }
  }
}
```

## 🔍 Key Features by Language

### English (en)
- **MQM Core**: Standard error taxonomy, severity levels, category weights
- **NLP Terms**: Computational linguistics terminology, MT concepts
- **Grammar**: Advanced syntax, articles, tense/aspect system

### Russian (ru)
- **MQM Core**: Translated MQM with Russian-specific adaptations
- **NLP Terms**: Russian NLP tools and concepts
- **Morphology**: 6 cases, 2 aspects, 3 genders, suppletion, verb conjugation

### Chinese (zh)
- **MQM Core**: MQM adapted for Chinese-specific issues
- **NLP Terms**: Word segmentation, Chinese NLP challenges
- **Classifiers**: 6 categories of measure words (個/本/只/條/張/次 etc.)

### Persian (fa)
- **MQM Core**: MQM for Persian with script considerations
- **NLP Terms**: Persian NLP approaches
- **Grammar**: Ezafe construction, compound verbs, formal/informal register

### Hindi (hi)
- **MQM Core**: MQM with Devanagari script support
- **NLP Terms**: Indic NLP terminology
- **Cases**: 8 कारक with postpositions, ergative marking, oblique forms

## 🎯 Integration with KTTC Agents

### Automatic Loading

Glossaries are automatically loaded when agents initialize:

```python
from kttc.agents import RussianFluencyAgent

# Agent automatically loads morphology_ru.json
agent = RussianFluencyAgent(llm_provider)

# Access validator
case_info = agent.case_validator.get_case_info("genitive")
aspect_rules = agent.case_validator.get_aspect_usage_rules("perfective")
```

### Error Enrichment

When agents detect errors, ErrorParser enriches them with glossary definitions:

```python
from kttc.agents.parser import ErrorParser

# Parse with enrichment (default)
errors = ErrorParser.parse_errors(
    response=llm_response,
    enrich_with_glossary=True,  # Default
    language="en"
)

# Each error gets enriched with MQM definition
# error.description includes: original + MQM definition + examples
```

### MQM Scoring

MQM scorer uses category weights from glossaries:

```python
from kttc.core.mqm import MQMScorer

# Load weights from glossary
scorer = MQMScorer(use_glossary_weights=True)

# scorer.category_weights loaded from glossaries/en/mqm_core.json:
# {
#   "accuracy": 1.5,
#   "terminology": 1.2,
#   "fluency": 1.0,
#   "style": 0.8
# }

score = scorer.calculate_score(errors, word_count=100)
```

### Agent-Specific Usage

**RussianFluencyAgent**:
- Loads `morphology_ru.json` via `RussianCaseAspectValidator`
- Validates case agreement, verb aspect, gender
- Used in `_glossary_check()` method

**ChineseFluencyAgent**:
- Loads `classifiers_zh.json` via `ChineseMeasureWordValidator`
- Validates measure word usage
- Used in `_glossary_check()` method

**HindiFluencyAgent**:
- Loads `cases_hi.json` via `HindiPostpositionValidator`
- Validates postposition usage, case markers
- Used in `_glossary_check()` method

**PersianFluencyAgent**:
- Loads `grammar_fa.json` via `PersianEzafeValidator`
- Validates ezafe construction, compound verbs
- Used in `_glossary_check()` method

## 📝 Adding New Terms

To add terms to an existing glossary:

1. **Edit the JSON file** in the appropriate language directory
2. **Follow the existing structure** (see formats above)
3. **Update `metadata.total_terms`** if present
4. **Add sources** in metadata.sources array
5. **Test with GlossaryManager**:

```python
from kttc.terminology import GlossaryManager

manager = GlossaryManager()
glossary = manager.load_glossary("en", "mqm_core")

# Verify new term is loaded
assert "your_new_term" in glossary["categories"]["your_category"]
```

6. **Run tests** to ensure integration works:

```bash
python3.11 -m pytest tests/unit/test_agents_glossary_integration.py -v
python3.11 -m pytest tests/unit/test_mqm_glossary_integration.py -v
```

### Creating New Glossaries

To add a new glossary file:

1. Create JSON file in appropriate language directory: `glossaries/{lang}/{name}.json`
2. Follow the standard format (see Glossary Format section)
3. Update this README with the new file
4. Add support in `GlossaryManager.load_glossary()` if needed
5. Document usage in user-facing guides

## 🔗 Architecture Integration

Glossaries integrate with KTTC at multiple levels:

```
┌─────────────────────────────────────────────────────┐
│                  User Input                         │
│          (kttc check, batch, translate)             │
└───────────────────┬─────────────────────────────────┘
                    │
┌───────────────────▼─────────────────────────────────┐
│            AgentOrchestrator                        │
│   • Coordinates all quality agents                  │
│   • Collects errors from each agent                 │
└───────────────────┬─────────────────────────────────┘
                    │
    ┌───────────────┼───────────────┐
    │               │               │
┌───▼────┐   ┌─────▼──────┐   ┌───▼────┐
│Accuracy│   │  Fluency   │   │  Term  │
│ Agent  │   │   Agents   │   │ Agent  │
└───┬────┘   └─────┬──────┘   └───┬────┘
    │              │              │
    │         ┌────┴────┐         │
    │         │ Russian │         │
    │         │ Chinese │         │
    │         │ Hindi   │         │
    │         │ Persian │         │
    │         └────┬────┘         │
    │              │              │
    │    ┌─────────▼────────┐    │
    │    │ Language         │    │
    │    │ Validators       │    │
    │    │ • Russian        │    │
    │    │ • Chinese        │    │
    │    │ • Hindi          │    │
    │    │ • Persian        │    │
    │    └─────────┬────────┘    │
    │              │              │
┌───▼──────────────▼──────────────▼───┐
│        GlossaryManager               │
│  • load_glossary(lang, type)         │
│  • Caches loaded glossaries          │
└───────────────┬──────────────────────┘
                │
┌───────────────▼──────────────────────┐
│         Glossary Files               │
│  glossaries/{lang}/{type}.json       │
│  • mqm_core*.json                    │
│  • morphology_ru.json                │
│  • classifiers_zh.json               │
│  • cases_hi.json                     │
│  • grammar_fa.json                   │
└──────────────────────────────────────┘
```

### Data Flow

1. **Agent Initialization**: Language-specific agents load glossaries via validators
2. **Error Detection**: Agents use glossary data to validate language-specific patterns
3. **Error Enrichment**: ErrorParser enriches error descriptions with MQM definitions
4. **MQM Scoring**: MQMScorer applies category weights from glossaries
5. **Output**: User receives enriched errors with explanations

## 📖 References

### Standards & Frameworks
- **MQM**: W3C Multidimensional Quality Metrics Community Group ([themqm.org](http://www.themqm.org))
- **ISO 5060:2024**: Translation quality evaluation standard

### Linguistic Resources
- **English**: Cambridge Grammar, Oxford English Corpus
- **Russian**: Russian National Corpus, OpenCorpora
- **Chinese**: CCL Corpus (Peking University), HSK vocabulary
- **Persian**: Dehkhoda Dictionary, Persian Wikipedia
- **Hindi**: CSTT (Commission for Scientific & Technical Terminology, Govt of India)

### NLP Libraries Referenced
- **Russian**: pymorphy2, spaCy, stanza
- **Chinese**: jieba, HanLP, pkuseg
- **Hindi**: indic-nlp-library, stanza
- **Persian**: hazm, parsivar, stanza
- **Multilingual**: spaCy, stanza, Hugging Face Transformers

## 📄 License

These glossaries are compiled from public domain sources and academic publications:
- **MQM framework**: W3C Community Group (open standard, public domain)
- **NLP terminology**: Academic consensus terminology (public domain)
- **Language-specific**: Standard linguistic reference works (fair use for educational purposes)

All glossary content is provided for educational and research purposes under fair use principles.

## 🤝 Contributing

To contribute new terms or glossaries:

1. **Research authoritative sources** (academic, standards bodies, linguistic references)
2. **Follow the JSON format** (see Glossary Format section)
3. **Add metadata** with sources and version information
4. **Update this README** with new glossary information
5. **Test integration**:
   ```bash
   python3.11 -m pytest tests/unit/test_agents_glossary_integration.py -v
   ```
6. **Update user documentation** in `docs/*/guides/glossary-management.md`
7. **Submit pull request** with clear description of additions

### Quality Guidelines

When adding terms:
- Use authoritative sources (academic papers, standards, linguistic references)
- Provide clear, concise definitions
- Include practical examples
- Add related terms for cross-referencing
- Cite sources in metadata
- Ensure JSON is valid and properly formatted

## 📧 Contact

For questions about glossaries or terminology:
- **GitHub Issues**: [kttc-ai/kttc/issues](https://github.com/kttc-ai/kttc/issues)
- **User Documentation**: See `docs/*/guides/glossary-management.md` for language-specific guides

---

## 🆕 Recent Updates

### School Curriculum Glossaries (2025-11-22)

Added proofreading support with school curriculum rules for all 5 languages:

**Russian (ФГОС):**
- `school_curriculum/orthography_fgos.json` - НЕ/НИ with verbs, hyphens, ЖИ-ШИ rules
- `school_curriculum/punctuation_fgos.json` - Comma rules, colons, quotation marks

**English (UK GPS):**
- `school_curriculum/spelling_uk_gps.json` - Homophones, apostrophes, common errors

**Chinese (部编版 PEP):**
- `school_curriculum/grammar_pep.json` - 量词, 的/地/得, punctuation rules

**Hindi (NCERT):**
- `school_curriculum/grammar_ncert.json` - Sandhi (संधि), samas (समास), spelling

**Persian (Iranian Curriculum):**
- `school_curriculum/grammar_iranian.json` - Ezafe (اضافه), nim-fasele (نیم‌فاصله), verbs

**Integration:**
- Used by `kttc check --self` (monolingual proofreading mode)
- Used by `kttc proofread` and `kttc lint` commands
- Loaded by GrammarAgent and SpellingAgent

### Glossary Expansion Project (2025-11-21)

Added 7 new comprehensive glossaries with 402+ terms:

**English (4 new files):**
- `mqm_error_taxonomy.json` - Complete W3C MQM error classification with 47 terms
- `translation_metrics.json` - 32 evaluation metrics (BLEU, COMET, TER, GEMBA, etc.)
- `transformer_nlp_terms.json` - 58 terms on attention, embeddings, tokenization
- `llm_terminology.json` - 45 modern LLM terms (hallucinations, RLHF, RAG, alignment)

**Chinese (1 new file):**
- `idioms_expressions_zh.json` - 120 terms covering 成语, 惯用语, 歇后语, 谚语

**Research:**
- Comprehensive research across 5 languages using native-language searches
- 40+ authoritative sources consulted
- Focus on 2024-2025 contemporary terminology

**Documentation:**
- `GLOSSARY_RESEARCH_REPORT_2025.md` - Complete research methodology and findings
- `GLOSSARY_EXPANSION_COMPLETE.md` - Implementation status and integration guide

---

**Version**: 2.1.0 (School curriculum support)
**Last Updated**: 2025-11-22
**Total Glossary Files**: 31 (was 24)
**Total Terms**: 1200+ (was ~1000)
**Languages**: 5 (en, ru, zh, fa, hi)
**Maintained by**: KTTC AI Team
