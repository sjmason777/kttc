# KTTC Translation Quality Benchmarks

Comprehensive benchmarking system for testing KTTC translation quality assessment capabilities.

## Overview

This benchmarking system evaluates KTTC's ability to:
- Assess high-quality professional translations (FLORES-200)
- Detect errors in poor-quality translations (synthetic bad translations)
- Provide consistent MQM scores across different language pairs
- Identify specific error types and severities

## Language Pairs

The benchmark covers 6 bidirectional language pairs:
- 🇬🇧 **English ↔ 🇷🇺 Russian** (en-ru, ru-en)
- 🇬🇧 **English ↔ 🇨🇳 Chinese** (en-zh, zh-en)
- 🇷🇺 **Russian ↔ 🇨🇳 Chinese** (ru-zh, zh-ru)

## Datasets

### 1. FLORES-200 (Good Translations)
- **Source:** Meta/Facebook AI
- **Description:** Professionally translated benchmark with 200 languages
- **Size:** 3,001 sentences per language pair
- **Splits:** dev, devtest, test
- **Expected MQM:** 95-100 (high quality)

### 2. WMT-MQM (Error Annotations)
- **Source:** Google/WMT
- **Description:** Human-annotated translation errors
- **Language Pairs:** en-de, zh-en
- **Annotations:** MQM framework (Accuracy, Fluency, Terminology, Style)
- **Use Case:** Validate error detection capabilities

### 3. Synthetic Bad Translations
- **Source:** Generated using LLMs
- **Description:** Intentionally flawed translations with known error types
- **Error Types:**
  - Mistranslation (major)
  - Omission (major)
  - Addition (minor)
  - Grammar errors (minor)
  - Terminology errors (major)
  - Style mismatches (minor)
  - Word order issues (minor)
- **Expected MQM:** 40-75 (poor quality)

## Quick Start

### Step 1: Download Benchmark Data

```bash
# Download FLORES-200 and cache locally
python3.11 scripts/download_benchmark_data.py
```

This will:
- Download FLORES-200 for all 6 language pairs
- Cache data in `tests/benchmarks/data/`
- Fallback to sample data if HuggingFace unavailable

### Step 2: Generate Bad Translations (Optional)

```bash
# Generate synthetic bad translations for testing
python3.11 scripts/generate_bad_translations.py
```

This creates translations with intentional errors for detection testing.

### Step 3: Run Comprehensive Benchmark

```bash
# Run full benchmark suite
python3.11 scripts/run_comprehensive_benchmark.py
```

## Usage Examples

### Run Benchmark Programmatically

```python
import asyncio
from kttc.agents.orchestrator import AgentOrchestrator
from kttc.llm import OpenAIProvider
from tests.benchmarks.enhanced_dataset_loader import EnhancedDatasetLoader
from scripts.run_comprehensive_benchmark import ComprehensiveBenchmark

async def run():
    # Setup
    llm = OpenAIProvider(api_key="your-key")
    orchestrator = AgentOrchestrator(llm, quality_threshold=95.0)

    # Run benchmark
    benchmark = ComprehensiveBenchmark()
    results = await benchmark.run_full_benchmark(
        orchestrator,
        language_pairs=[("en", "ru"), ("en", "zh")],
        sample_size=100,
        include_bad_translations=True
    )

    print(f"Average MQM: {results['summary']['flores200_avg_mqm']}")

asyncio.run(run())
```

### Load Specific Dataset

```python
from tests.benchmarks.enhanced_dataset_loader import EnhancedDatasetLoader

loader = EnhancedDatasetLoader()

# Load FLORES-200
samples = await loader.load_flores200("en", "ru", split="devtest", sample_size=100)

# Load WMT-MQM with errors
error_samples = await loader.load_wmt_mqm("en", "de", sample_size=50)
```

### Generate Bad Translations

```python
from scripts.generate_bad_translations import BadTranslationGenerator
from kttc.llm import OpenAIProvider

llm = OpenAIProvider(api_key="your-key")
generator = BadTranslationGenerator(llm)

# Generate single bad translation
bad = await generator.generate_bad_translation(
    "Hello, world!",
    source_lang="en",
    target_lang="ru",
    error_type="mistranslation"
)

# Generate multiple versions with different errors
versions = await generator.generate_multiple_bad_versions(
    "Artificial intelligence is transforming technology.",
    source_lang="en",
    target_lang="zh",
    num_versions=3
)
```

## Benchmark Metrics

### Primary Metrics
- **MQM Score** (0-100): Multidimensional Quality Metrics score
- **Pass Rate** (%): Percentage of translations meeting threshold (≥95)
- **Error Counts**: Critical, Major, Minor errors detected
- **Processing Time**: Average time per translation

### Good Translation Benchmarks
- **Target MQM:** 95-100
- **Expected Pass Rate:** >90%
- **Purpose:** Validate system doesn't over-flag good translations

### Bad Translation Benchmarks
- **Target MQM:** 40-75
- **Detection Rate:** >80% should score <95
- **Purpose:** Validate error detection capability

## Directory Structure

```
tests/benchmarks/
├── README.md                          # This file
├── __init__.py
├── wmt_benchmark.py                   # WMT benchmark runner
├── dataset_loader.py                  # Basic dataset loader
├── enhanced_dataset_loader.py         # Enhanced loader with FLORES-200/WMT-MQM
├── data/                              # Cached benchmark data
│   ├── flores200_en_ru_devtest.json
│   ├── flores200_en_zh_devtest.json
│   ├── flores200_ru_en_devtest.json
│   ├── flores200_zh_en_devtest.json
│   ├── flores200_ru_zh_devtest.json
│   ├── flores200_zh_ru_devtest.json
│   ├── wmt_mqm_en_de.json
│   ├── wmt_mqm_zh_en.json
│   └── synthetic_bad_*.json          # Generated bad translations
└── results/                           # Benchmark results (gitignored)

scripts/
├── download_benchmark_data.py         # Download and cache datasets
├── generate_bad_translations.py       # Generate synthetic errors
└── run_comprehensive_benchmark.py     # Run full benchmark suite

benchmark_results/                     # Results directory
├── comprehensive_benchmark_*.json     # Detailed JSON results
├── comprehensive_benchmark_*.md       # Human-readable reports
└── *.json                            # Individual benchmark runs
```

## Output Format

### JSON Report

```json
{
  "summary": {
    "total_language_pairs": 6,
    "total_execution_time_seconds": 450.5,
    "flores200_avg_mqm": 97.3,
    "bad_translations_avg_mqm": 62.1,
    "timestamp": "2025-11-19T16:00:00"
  },
  "results_by_pair": {
    "en-ru": {
      "flores200_good": {
        "sample_count": 100,
        "avg_mqm_score": 97.5,
        "pass_rate": 94.0,
        "error_counts": {"critical": 0, "major": 2, "minor": 8}
      },
      "synthetic_bad": {
        "sample_count": 30,
        "avg_mqm_score": 65.2,
        "detection_rate": 86.7
      }
    }
  }
}
```

### Markdown Report

See generated `comprehensive_benchmark_*.md` files for human-readable reports.

## CI/CD Integration

### GitHub Actions

```yaml
name: Translation Quality Benchmark

on:
  schedule:
    - cron: '0 0 * * 0'  # Weekly
  workflow_dispatch:

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install -e ".[dev]"
          pip install datasets

      - name: Download benchmark data
        run: python3.11 scripts/download_benchmark_data.py

      - name: Run benchmark
        env:
          KTTC_OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        run: python3.11 scripts/run_comprehensive_benchmark.py

      - name: Upload results
        uses: actions/upload-artifact@v3
        with:
          name: benchmark-results
          path: benchmark_results/
```

## Interpreting Results

### Good Translation Assessment
- **MQM 98-100:** Excellent - No errors detected
- **MQM 95-97:** Good - Minor issues only
- **MQM 90-94:** Acceptable - Some errors present
- **MQM <90:** Poor - Significant quality issues

### Bad Translation Detection
- **Detection Rate >90%:** Excellent error detection
- **Detection Rate 80-90%:** Good error detection
- **Detection Rate 70-80%:** Moderate error detection
- **Detection Rate <70%:** Needs improvement

### Error Distribution
- **Critical Errors:** Meaning-changing mistakes
- **Major Errors:** Significant quality issues
- **Minor Errors:** Small fluency or style issues

## Extending the Benchmark

### Add New Language Pair

```python
# In download_benchmark_data.py
language_pairs.append(("en", "fr"))  # English → French
```

### Add Custom Dataset

```python
from tests.benchmarks.enhanced_dataset_loader import EnhancedDatasetLoader

loader = EnhancedDatasetLoader()

# Add custom samples
custom_samples = [
    {
        "source": "Your source text",
        "translation": "Your translation",
        "source_lang": "en",
        "target_lang": "ru",
        "domain": "technical"
    }
]

# Save to cache
await loader.save_to_cache(custom_samples, "custom_dataset.json")
```

### Add New Error Type

```python
# In generate_bad_translations.py
ERROR_TYPES["cultural"] = {
    "description": "Culturally inappropriate translation",
    "severity": "major",
    "prompt_modifier": "Use culturally inappropriate terms or idioms"
}
```

## Performance Expectations

- **FLORES-200 (100 samples):** ~5-10 minutes
- **Synthetic Bad (30 samples):** ~2-5 minutes
- **Full 6-pair benchmark:** ~30-60 minutes
- **Memory Usage:** <2GB RAM
- **API Costs:** ~$1-5 per full benchmark (OpenAI gpt-4o)

## Troubleshooting

### HuggingFace datasets not available
- Install: `pip install datasets`
- Or use fallback sample data (automatic)

### API rate limits
- Reduce `sample_size` parameter
- Add delays between requests
- Use `gpt-4o-mini` for cheaper testing

### Cache not working
- Check `tests/benchmarks/data/` directory exists
- Verify write permissions
- Re-run download script

## References

- [FLORES-200 Paper](https://arxiv.org/abs/2207.04672)
- [WMT-MQM Dataset](https://github.com/google/wmt-mqm-human-evaluation)
- [MQM Framework](http://www.qt21.eu/mqm-definition/definition-2015-12-30.html)
- [KTTC Documentation](../../docs/README.md)

## Contributing

To improve the benchmark:
1. Add more language pairs
2. Integrate additional datasets (WMT, BLEU, etc.)
3. Enhance error type coverage
4. Improve reporting formats

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for guidelines.
