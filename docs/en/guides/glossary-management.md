# Glossary Management

**Leverage linguistic reference data to enhance translation quality checking.**

Glossaries provide language-specific linguistic knowledge that helps KTTC's agents perform more accurate quality assessments. This includes grammar rules, terminology definitions, and language-specific patterns.

## Overview

KTTC integrates comprehensive glossaries for:

- **MQM Error Definitions** - Standard error taxonomy from W3C Multidimensional Quality Metrics
- **Russian Grammar** - Case system, verb aspects, particles
- **Chinese Classifiers** - Measure words (量词) and usage patterns
- **Hindi Grammar** - Case system (कारक), postpositions, oblique forms
- **Persian Grammar** - Ezafe construction (اضافه), compound verbs

These glossaries are automatically loaded and used by the quality agents during evaluation.

## Supported Languages

### MQM Core Definitions

**Available in:** English, Russian, Chinese, Hindi, Persian

The MQM (Multidimensional Quality Metrics) glossary provides standardized error definitions used by all agents.

**Error categories:**
- Accuracy: mistranslation, omission, addition
- Fluency: grammar, spelling, punctuation
- Style: register, formality, consistency
- Terminology: domain-specific term errors

### Russian Language Glossary

**File:** `glossaries/ru/grammar_reference.json`

**Contains:**
- **6 grammatical cases** (nominative, genitive, dative, accusative, instrumental, prepositional)
- **Verb aspects** (perfective/imperfective usage rules)
- **Particles** (же, ли, бы, не, ни)
- **Register markers** (ты/вы distinction)

**Used by:** `RussianFluencyAgent`

### Chinese Language Glossary

**File:** `glossaries/zh/classifiers.json`

**Contains:**
- **Individual classifiers** (个, 只, 条, 张, 本 etc.)
- **Collective classifiers** (群, 堆, 批 etc.)
- **Container classifiers** (杯, 碗, 盒 etc.)
- **Measurement classifiers** (米, 公斤, 升 etc.)
- **Temporal classifiers** (年, 月, 天 etc.)
- **Verbal classifiers** (次, 遍, 趟 etc.)

**Used by:** `ChineseFluencyAgent`

### Hindi Language Glossary

**File:** `glossaries/hi/grammar_reference.json`

**Contains:**
- **8 grammatical cases** (कारक - kārak system)
- **Postpositions** (को, से, में, पर etc.)
- **Oblique forms** (formation rules)
- **Gender agreement** (masculine/feminine patterns)

**Used by:** `HindiFluencyAgent`

### Persian Language Glossary

**File:** `glossaries/fa/grammar_reference.json`

**Contains:**
- **Ezafe construction** (اضافه - e-zāfe rules)
- **Compound verbs** (light verbs: کردن, زدن, داشتن etc.)
- **Prepositions** (در, به, از etc.)
- **Formal/informal register** (شما/تو distinction)

**Used by:** `PersianFluencyAgent`

## How Glossaries Work

### Automatic Integration

Glossaries are automatically loaded when agents initialize:

```python
from kttc.agents import RussianFluencyAgent, ChineseFluencyAgent
from kttc.llm import OpenAIProvider

provider = OpenAIProvider(api_key="your-key")

# Russian agent automatically loads grammar_reference.json
russian_agent = RussianFluencyAgent(provider)

# Chinese agent automatically loads classifiers.json
chinese_agent = ChineseFluencyAgent(provider)
```

No configuration needed - glossaries are loaded based on the agent type.

### Error Enrichment

When agents detect errors, they automatically enrich error descriptions with glossary definitions:

```python
# Error detected by agent
error = {
    "category": "fluency",
    "subcategory": "grammar",
    "description": "Case mismatch"
}

# After glossary enrichment
enriched_error = {
    "category": "fluency",
    "subcategory": "grammar",
    "description": "Case mismatch. In Russian, adjectives must agree with nouns in case, number, and gender. The genitive case (родительный падеж) is used after negations, with numbers, and to show possession."
}
```

### MQM Score Weighting

The MQM glossary provides category weights for scoring:

```python
from kttc.core.mqm import MQMScorer

# Load weights from glossary
scorer = MQMScorer(use_glossary_weights=True)

# Category weights from glossary/en/mqm_core.json:
# accuracy: 1.5 (higher penalty for accuracy errors)
# terminology: 1.2
# fluency: 1.0
# style: 0.8 (lower penalty for style issues)
```

## CLI Commands

### View Glossary Contents

```bash
# List all available glossaries
kttc glossary list

# View specific glossary
kttc glossary view --language ru --name grammar_reference

# Search in glossary
kttc glossary search --language zh --query "measure word"

# View in JSON format
kttc glossary view --language en --name mqm_core --format json
```

### Terminology Commands

```bash
# Validate an error type
kttc terminology validate --error-type "mistranslation" --language en

# List MQM error categories
kttc terminology list-categories

# Get error definition
kttc terminology define --error-type "grammar" --language ru
```

## Glossary File Format

Glossaries are stored as JSON files following this structure:

### MQM Core Format

```json
{
  "metadata": {
    "name": "MQM Core Error Taxonomy",
    "version": "1.0",
    "language": "en",
    "description": "W3C Multidimensional Quality Metrics"
  },
  "categories": {
    "accuracy": {
      "weight": 1.5,
      "subcategories": {
        "mistranslation": {
          "definition": "The target text does not accurately represent the source text",
          "examples": ["cat → dog", "buy → sell"],
          "severity_guidelines": "Major or Critical"
        }
      }
    }
  }
}
```

### Language-Specific Format

```json
{
  "metadata": {
    "name": "Russian Grammar Reference",
    "version": "1.0",
    "language": "ru"
  },
  "cases": {
    "nominative": {
      "name": "Именительный",
      "usage": ["Subject of sentence", "Predicate nominative"],
      "question": "кто? что?",
      "examples": ["студент читает", "это книга"]
    }
  },
  "aspects": {
    "perfective": {
      "usage": "Completed actions, one-time events",
      "markers": ["prefixes: по-, на-, с-"],
      "examples": ["написать письмо"]
    }
  }
}
```

## Advanced Usage

### Custom Glossaries

You can add custom glossaries to the `glossaries/` directory:

```bash
# Create custom glossary
mkdir -p glossaries/en
cat > glossaries/en/medical_terms.json <<EOF
{
  "metadata": {
    "name": "Medical Terminology",
    "version": "1.0",
    "language": "en",
    "domain": "medical"
  },
  "terms": {
    "myocardial_infarction": {
      "definition": "Heart attack",
      "synonyms": ["MI", "heart attack"],
      "avoid": ["cardiac arrest"]
    }
  }
}
EOF
```

### Programmatic Access

```python
from kttc.terminology import GlossaryManager

# Initialize manager
manager = GlossaryManager()

# Load glossary
mqm_data = manager.load_glossary("en", "mqm_core")

# Get specific definition
definition = mqm_data["categories"]["accuracy"]["subcategories"]["mistranslation"]

# Validate error type
is_valid, info = manager.validate_mqm_error("mistranslation", "en")
```

### Disable Glossary Enrichment

```python
from kttc.agents.parser import ErrorParser

# Parse without enrichment
errors = ErrorParser.parse_errors(
    response=llm_response,
    enrich_with_glossary=False  # Disable enrichment
)

# Or use default weights for MQM scoring
from kttc.core.mqm import MQMScorer

scorer = MQMScorer(use_glossary_weights=False)  # Use default weights
```

## Benefits

### 1. Improved Accuracy

Glossaries provide precise linguistic definitions that help agents:
- Identify language-specific errors more accurately
- Understand grammatical patterns and rules
- Validate terminology usage

### 2. Better Error Descriptions

Enriched error descriptions include:
- Linguistic explanations
- Grammar rules
- Usage examples
- Corrections

### 3. Consistent Scoring

MQM glossary ensures:
- Standardized error taxonomy
- Consistent category weights
- Reproducible quality scores
- Alignment with industry standards (W3C MQM)

### 4. Multi-Language Support

Same glossary framework works across:
- 5 languages (en, ru, zh, hi, fa)
- Multiple error categories
- Language-specific features
- Cultural adaptations

## File Locations

```
glossaries/
├── en/
│   └── mqm_core.json          # English MQM definitions
├── ru/
│   ├── mqm_core.json          # Russian MQM definitions
│   └── grammar_reference.json # Russian grammar rules
├── zh/
│   ├── mqm_core.json          # Chinese MQM definitions
│   └── classifiers.json       # Chinese measure words
├── hi/
│   ├── mqm_core.json          # Hindi MQM definitions
│   └── grammar_reference.json # Hindi grammar rules
└── fa/
    ├── mqm_core.json          # Persian MQM definitions
    └── grammar_reference.json # Persian grammar rules
```

## Related Documentation

- **[CLI Commands](../reference/cli-commands.md)** - Glossary and terminology commands
- **[Architecture](../explanation/architecture.md)** - How glossaries integrate with agents
- **[Language Features](language-features.md)** - Language-specific capabilities

## Troubleshooting

### Glossary Not Loading

**Problem:** Agent doesn't use glossary data

**Solution:**
```python
# Check if glossary file exists
import os
glossary_path = "glossaries/ru/grammar_reference.json"
print(f"Exists: {os.path.exists(glossary_path)}")

# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Custom Glossary Not Found

**Problem:** Custom glossary not loaded

**Solution:**
- Ensure file is in correct directory: `glossaries/{language}/{name}.json`
- Check JSON syntax is valid
- Verify metadata section is present

### Error Enrichment Not Working

**Problem:** Errors not enriched with definitions

**Solution:**
```python
# Explicitly enable enrichment
from kttc.agents.parser import ErrorParser

errors = ErrorParser.parse_errors(
    response=response,
    enrich_with_glossary=True,  # Explicitly enable
    language="en"  # Specify language
)
```

## Performance

**Glossary Loading:**
- Load time: <100ms per glossary
- Memory usage: ~1-5MB per glossary
- Cached after first load

**Error Enrichment:**
- Lookup time: <1ms per error
- No API calls required
- Deterministic results

## Future Enhancements

Planned glossary features:
- User-editable glossaries via web UI
- Domain-specific glossaries (legal, medical, technical)
- Glossary version control
- Community-contributed glossaries
- Glossary merging and conflict resolution
