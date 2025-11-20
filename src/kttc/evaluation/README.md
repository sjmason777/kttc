# Lightweight Translation Quality Metrics

CPU-based evaluation metrics and rule-based error detection **without GPU or heavy neural models**.

## Overview

This module provides fast, reproducible translation quality assessment:

### 🎯 Lightweight Metrics (CPU-based)
- **chrF/chrF++**: Character-level F-score (best for morphologically rich languages)
- **BLEU**: Word-level n-gram overlap (widely used baseline)
- **TER**: Translation Edit Rate (minimum edits needed)

All metrics use [sacreBLEU](https://github.com/mjpost/sacrebleu) for reproducibility and WMT compatibility.

### 🔍 Rule-Based Error Detection (No AI)
Fast, deterministic checks for common translation errors:
- **Numbers consistency**: Missing/extra numbers
- **Length ratio**: Potential omissions/additions
- **Punctuation balance**: Unmatched quotes, brackets
- **Context preservation**: Negation, questions, exclamations
- **Named entities**: Capitalized words (potential names/places)

## Quick Start

### Basic Usage

```python
from kttc.evaluation import LightweightMetrics, ErrorDetector

# Initialize
metrics = LightweightMetrics()
detector = ErrorDetector()

# Evaluate translation
scores = metrics.evaluate(
    translation="Привет мир",
    reference="Привет, мир!",
    source="Hello world"
)

print(f"chrF: {scores.chrf:.2f}")
print(f"BLEU: {scores.bleu:.2f}")
print(f"Quality: {scores.quality_level}")

# Detect errors
errors = detector.detect_all_errors(
    source="Price: $100",
    translation="Price: $200"
)

if errors:
    for error in errors:
        print(f"{error.severity}: {error.description}")
```

### CLI Integration

```python
from kttc.cli.ui import print_lightweight_metrics, print_rule_based_errors

# Display metrics in CLI
print_lightweight_metrics(scores, verbose=True)

# Display errors in CLI
rule_score = detector.calculate_rule_based_score(errors)
print_rule_based_errors(errors, rule_score, verbose=True)
```

### Batch Evaluation

```python
# Corpus-level metrics
results = metrics.evaluate_batch(
    translations=["Привет", "Мир"],
    references=["Привет", "Мир"]
)

print(f"Corpus chrF: {results['chrf']:.2f}")
print(f"Sentences: {results['num_sentences']}")
```

## Quality Thresholds

### chrF (Primary Metric)
- **≥80**: Excellent (production-ready)
- **≥65**: Good (minor review)
- **≥50**: Acceptable (human review)
- **<50**: Poor (significant revision)

### BLEU (Secondary Metric)
- **≥50**: Excellent
- **≥40**: Good (deployment threshold)
- **≥25**: Acceptable
- **<25**: Poor

### Deployment Guidelines (WMT Best Practices)
- chrF improvement: **+4 points** for deployment
- BLEU improvement: **+5 points** for deployment

## Rule-Based Error Severity

### Critical (−20 points)
- Missing numbers
- Negation loss
- Major context changes

### Major (−10 points)
- Extra numbers
- Length anomalies
- Named entity loss

### Minor (−5 points)
- Punctuation issues
- Missing exclamation marks

## Examples

See `examples/test_lightweight_metrics.py` for comprehensive examples:

```bash
python3.11 examples/test_lightweight_metrics.py
```

## Performance

All metrics run on CPU:
- **No GPU required**
- **No neural models to download**
- **Fast evaluation** (milliseconds per sentence)
- **Zero API costs**

Perfect for:
- 🚀 CI/CD pipelines
- 💻 Development environments
- 📊 Quick quality checks
- 💰 Cost-sensitive applications

## Integration with Existing KTTC

These metrics complement existing MQM scoring:
- **MQM**: AI-based deep quality analysis
- **Lightweight**: Fast baseline metrics
- **Rule-based**: Deterministic error detection

Use together for comprehensive quality assessment!

## API Reference

### LightweightMetrics

```python
class LightweightMetrics:
    def evaluate(translation: str, reference: str, source: str | None = None) -> MetricScores
    def evaluate_batch(translations: list[str], references: list[str]) -> dict
    def get_interpretation(scores: MetricScores) -> str
    def passes_deployment_threshold(scores: MetricScores) -> bool
```

### ErrorDetector

```python
class ErrorDetector:
    def detect_all_errors(source: str, translation: str) -> list[RuleBasedError]
    def check_numbers_consistency(source: str, translation: str) -> list[RuleBasedError]
    def check_length_ratio(source: str, translation: str) -> RuleBasedError | None
    def check_punctuation_balance(source: str, translation: str) -> list[RuleBasedError]
    def check_context_preservation(source: str, translation: str) -> list[RuleBasedError]
    def check_named_entities(source: str, translation: str) -> RuleBasedError | None
    def calculate_rule_based_score(errors: list[RuleBasedError]) -> float
```

## Why These Metrics?

### Research-Backed
- Based on WMT (Workshop on Machine Translation) best practices
- sacreBLEU for reproducibility (used in academic research)
- chrF shown to correlate better than BLEU for diverse languages

### Practical
- No infrastructure requirements
- Works offline
- Consistent across environments
- Easy to interpret

### Language-Agnostic
- chrF works well for all languages (character-level)
- Rule-based checks use universal patterns
- No language-specific models needed

## References

- [sacreBLEU](https://github.com/mjpost/sacrebleu): Reference implementation
- [WMT Metrics](https://machinetranslate.org/metrics): Industry standards
- [chrF paper](https://aclanthology.org/W15-3049/): Character F-score
- [MQM framework](https://themqm.org/): Error taxonomy
