# Roadmap: Language Helpers and Fluency Agents Enhancement

**Project:** KTTC AI Translation Quality Assessment
**Date:** November 13, 2025
**Status:** Planning Phase
**Last Updated:** November 13, 2025

---

## Overview

This roadmap outlines the step-by-step implementation plan for enhancing EnglishLanguageHelper and ChineseLanguageHelper with deterministic grammar checking capabilities, and potentially creating specialized fluency agents (EnglishFluencyAgent, ChineseFluencyAgent).

**Key Principle:** Enhance helpers FIRST, validate results, THEN decide on specialized agents.

---

## Table of Contents

1. [Phase 1: English Helper Enhancement](#phase-1-english-helper-enhancement)
2. [Phase 2: English Helper Validation](#phase-2-english-helper-validation)
3. [Phase 3: Chinese Helper Enhancement](#phase-3-chinese-helper-enhancement)
4. [Phase 4: Chinese Helper Validation](#phase-4-chinese-helper-validation)
5. [Phase 5: Specialized Agents (Optional)](#phase-5-specialized-agents-optional)
6. [Dependencies & Requirements](#dependencies--requirements)
7. [Testing Strategy](#testing-strategy)
8. [Success Metrics](#success-metrics)

---

## Phase 1: English Helper Enhancement

**Priority:** 🔴 HIGH
**Estimated Time:** 2-3 hours
**Status:** 🟡 Ready to start
**Owner:** TBD

### Objective

Integrate LanguageTool into EnglishLanguageHelper to provide 5,000+ deterministic grammar rules.

### Tasks

#### 1.1 Install LanguageTool Dependencies

**Checklist:**
- [ ] Install `language-tool-python` package
- [ ] Verify Java 17.0+ is installed
- [ ] Test LanguageTool initialization
- [ ] Handle download of LanguageTool server (~200 MB)

**Commands:**
```bash
# Check Java version
java --version  # Should be 17.0 or higher

# Install LanguageTool
python3.11 -m pip install language-tool-python

# Test installation
python3.11 -c "import language_tool_python; tool = language_tool_python.LanguageTool('en-US'); print('LanguageTool OK')"
```

**Dependencies added to `pyproject.toml`:**
```toml
[project.optional-dependencies]
english = [
    "language-tool-python>=2.9.0",
]
```

---

#### 1.2 Modify EnglishLanguageHelper Class

**File:** `src/kttc/helpers/english.py`

**Checklist:**
- [ ] Add LanguageTool initialization in `__init__()`
- [ ] Handle optional dependency (graceful degradation)
- [ ] Add `_language_tool` instance variable
- [ ] Add `_lt_available` flag

**Code changes:**

```python
# Add import at top of file
try:
    import language_tool_python
    LANGUAGETOOL_AVAILABLE = True
    logger.info("LanguageTool available for English grammar checking")
except ImportError:
    LANGUAGETOOL_AVAILABLE = False
    logger.warning(
        "LanguageTool not installed. "
        "EnglishLanguageHelper will run in limited mode. "
        "Install with: pip install language-tool-python"
    )

class EnglishLanguageHelper(LanguageHelper):
    """Language helper for English with spaCy + LanguageTool."""

    def __init__(self) -> None:
        """Initialize English language helper."""
        self._nlp: Any = None
        self._initialized = False

        # Initialize spaCy (existing code)
        if SPACY_AVAILABLE:
            try:
                self._nlp = spacy.load("en_core_web_md")
                self._initialized = True
                logger.info("EnglishLanguageHelper initialized with spaCy en_core_web_md")
            except OSError:
                try:
                    self._nlp = spacy.load("en_core_web_sm")
                    self._initialized = True
                    logger.info("EnglishLanguageHelper initialized with spaCy en_core_web_sm")
                except OSError:
                    logger.error("spaCy English model not found.")
                    self._initialized = False
        else:
            logger.info("EnglishLanguageHelper running in limited mode (no spaCy)")

        # Initialize LanguageTool (NEW)
        self._language_tool: Any = None
        self._lt_available = False

        if LANGUAGETOOL_AVAILABLE:
            try:
                self._language_tool = language_tool_python.LanguageTool('en-US')
                self._lt_available = True
                logger.info("LanguageTool initialized successfully")
            except Exception as e:
                logger.warning(f"LanguageTool initialization failed: {e}")
                self._lt_available = False
```

---

#### 1.3 Implement check_grammar() Method

**Checklist:**
- [ ] Replace empty `check_grammar()` implementation
- [ ] Map LanguageTool matches to ErrorAnnotation
- [ ] Implement severity mapping
- [ ] Filter translation-irrelevant errors (style only)
- [ ] Handle exceptions gracefully

**Code changes:**

```python
def check_grammar(self, text: str) -> list[ErrorAnnotation]:
    """Check English grammar using LanguageTool.

    Args:
        text: English text to check

    Returns:
        List of detected grammar errors
    """
    if not self._lt_available:
        logger.debug("LanguageTool not available, skipping grammar checks")
        return []

    try:
        # Check with LanguageTool
        matches = self._language_tool.check(text)

        errors = []
        for match in matches:
            # Filter out style-only suggestions
            if not self._is_translation_relevant(match):
                continue

            # Map to our error format
            errors.append(ErrorAnnotation(
                category="fluency",
                subcategory=f"english_{match.ruleId}",
                severity=self._map_severity(match),
                location=(match.offset, match.offset + match.errorLength),
                description=match.message,
                suggestion=match.replacements[0] if match.replacements else None
            ))

        logger.debug(f"LanguageTool found {len(errors)} grammar errors")
        return errors

    except Exception as e:
        logger.error(f"LanguageTool check failed: {e}")
        return []

def _map_severity(self, match) -> ErrorSeverity:
    """Map LanguageTool match to ErrorSeverity.

    Args:
        match: LanguageTool Match object

    Returns:
        ErrorSeverity enum value
    """
    from kttc.core import ErrorSeverity

    rule_id = match.ruleId.lower()

    # Critical errors (spelling, clear grammar mistakes)
    if any(pattern in rule_id for pattern in ['spelling', 'typo', 'misspell']):
        return ErrorSeverity.CRITICAL

    # Major errors (agreement, verb form, tense)
    if any(pattern in rule_id for pattern in [
        'grammar', 'agreement', 'verb', 'subject_verb',
        'tense', 'article', 'preposition'
    ]):
        return ErrorSeverity.MAJOR

    # Minor errors (everything else)
    return ErrorSeverity.MINOR

def _is_translation_relevant(self, match) -> bool:
    """Filter out style-only suggestions not relevant for translation QA.

    Args:
        match: LanguageTool Match object

    Returns:
        True if error is relevant for translation, False otherwise
    """
    rule_id = match.ruleId.lower()

    # Exclude pure style suggestions
    exclude_patterns = [
        'style',
        'redundancy',
        'collocation',
        'cliche',
        'wordiness'
    ]

    if any(pattern in rule_id for pattern in exclude_patterns):
        return False

    return True
```

---

#### 1.4 Enhance get_enrichment_data() for LLM Prompts

**Checklist:**
- [ ] Add verb tense analysis
- [ ] Add article-noun pattern extraction
- [ ] Add subject-verb agreement hints
- [ ] Provide POS distribution

**Code changes:**

```python
def get_enrichment_data(self, text: str) -> dict[str, Any]:
    """Get comprehensive linguistic data for enriching LLM prompts.

    Args:
        text: Text to analyze

    Returns:
        Dictionary with morphological insights for LLM
    """
    if not self.is_available():
        return {"has_morphology": False}

    doc = self._nlp(text)

    # Verb tense analysis
    verb_tenses = {}
    for token in doc:
        if token.pos_ == "VERB":
            tense = token.morph.get("Tense")
            if tense:
                verb_tenses[token.text] = {
                    "tense": tense[0],
                    "aspect": token.morph.get("Aspect", [""])[0],
                    "person": token.morph.get("Person", [""])[0],
                    "number": token.morph.get("Number", [""])[0]
                }

    # Article-noun patterns
    article_noun_pairs = []
    for i, token in enumerate(doc):
        if token.pos_ == "DET" and token.text.lower() in ["a", "an", "the"]:
            # Find associated noun (look ahead)
            for j in range(i + 1, min(i + 5, len(doc))):
                if doc[j].pos_ in ["NOUN", "PROPN"]:
                    # Check if article matches (a vs an)
                    next_word = doc[i + 1].text if i + 1 < len(doc) else ""
                    correct_article = "an" if next_word and next_word[0].lower() in "aeiou" else "a"

                    article_noun_pairs.append({
                        "article": token.text.lower(),
                        "noun": doc[j].text,
                        "distance": j - i,
                        "correct": token.text.lower() == correct_article if token.text.lower() != "the" else True
                    })
                    break

    # Subject-verb pairs (for agreement checking)
    subject_verb_pairs = []
    for token in doc:
        if token.dep_ == "nsubj":  # Nominal subject
            verb = token.head
            if verb.pos_ == "VERB":
                subject_number = token.morph.get("Number")
                verb_number = verb.morph.get("Number")

                subject_verb_pairs.append({
                    "subject": token.text,
                    "verb": verb.text,
                    "subject_number": subject_number[0] if subject_number else None,
                    "verb_number": verb_number[0] if verb_number else None,
                    "agreement": subject_number == verb_number if subject_number and verb_number else None
                })

    # POS distribution
    pos_counts: dict[str, int] = {}
    for token in doc:
        if token.pos_:
            pos_counts[token.pos_] = pos_counts.get(token.pos_, 0) + 1

    # Named entities (existing code)
    entities = []
    for ent in doc.ents:
        entities.append({
            "text": ent.text,
            "label": ent.label_,
            "start": ent.start_char,
            "end": ent.end_char,
        })

    return {
        "has_morphology": True,
        "word_count": len([token for token in doc if not token.is_punct]),
        "verb_tenses": verb_tenses,
        "article_noun_pairs": article_noun_pairs,
        "subject_verb_pairs": subject_verb_pairs,
        "pos_distribution": pos_counts,
        "entities": entities,
        "sentence_count": len(list(doc.sents))
    }
```

---

#### 1.5 Update Unit Tests

**File:** `tests/unit/test_english_helper.py` (create if doesn't exist)

**Checklist:**
- [ ] Test LanguageTool initialization
- [ ] Test `check_grammar()` with known errors
- [ ] Test severity mapping
- [ ] Test error filtering
- [ ] Test enrichment data extraction
- [ ] Test graceful degradation (no LanguageTool)

**Test code:**

```python
import pytest
from kttc.helpers.english import EnglishLanguageHelper
from kttc.core import ErrorSeverity


class TestEnglishLanguageHelper:
    """Test EnglishLanguageHelper with LanguageTool integration."""

    @pytest.fixture
    def helper(self):
        """Create helper instance."""
        return EnglishLanguageHelper()

    def test_initialization(self, helper):
        """Test helper initializes correctly."""
        assert helper is not None
        assert helper.language_code == "en"

    def test_check_grammar_subject_verb_agreement(self, helper):
        """Test subject-verb agreement detection."""
        if not helper._lt_available:
            pytest.skip("LanguageTool not available")

        text = "He go to school every day"
        errors = helper.check_grammar(text)

        # Should find subject-verb agreement error
        assert len(errors) > 0
        assert any("agreement" in err.description.lower() for err in errors)

    def test_check_grammar_article_error(self, helper):
        """Test article usage error detection."""
        if not helper._lt_available:
            pytest.skip("LanguageTool not available")

        text = "I saw a elephant in the zoo"
        errors = helper.check_grammar(text)

        # Should find article error (a -> an)
        assert len(errors) > 0
        assert any("article" in err.subcategory.lower() for err in errors)

    def test_check_grammar_correct_text(self, helper):
        """Test that correct text has no errors."""
        if not helper._lt_available:
            pytest.skip("LanguageTool not available")

        text = "The quick brown fox jumps over the lazy dog."
        errors = helper.check_grammar(text)

        # Should have no errors
        assert len(errors) == 0

    def test_enrichment_data(self, helper):
        """Test enrichment data extraction."""
        if not helper.is_available():
            pytest.skip("spaCy not available")

        text = "He goes to school every day."
        enrichment = helper.get_enrichment_data(text)

        assert enrichment["has_morphology"] is True
        assert "verb_tenses" in enrichment
        assert "article_noun_pairs" in enrichment
        assert "subject_verb_pairs" in enrichment

    def test_severity_mapping(self, helper):
        """Test error severity mapping."""
        # Mock match object
        class MockMatch:
            def __init__(self, rule_id):
                self.ruleId = rule_id

        # Spelling error -> CRITICAL
        match = MockMatch("SPELLING_ERROR")
        assert helper._map_severity(match) == ErrorSeverity.CRITICAL

        # Grammar error -> MAJOR
        match = MockMatch("SUBJECT_VERB_AGREEMENT")
        assert helper._map_severity(match) == ErrorSeverity.MAJOR

        # Other -> MINOR
        match = MockMatch("SOME_OTHER_RULE")
        assert helper._map_severity(match) == ErrorSeverity.MINOR
```

---

#### 1.6 Documentation

**Checklist:**
- [ ] Update `README.md` with LanguageTool dependency
- [ ] Add installation instructions
- [ ] Document Java requirement
- [ ] Add usage examples

**Documentation snippet:**

```markdown
## English Language Helper

EnglishLanguageHelper provides deterministic grammar checking for English translations using LanguageTool.

### Installation

```bash
# Install optional dependency
pip install kttc[english]

# Or install directly
pip install language-tool-python
```

**Requirements:**
- Python 3.9+
- Java 17.0+ (for LanguageTool)

### Features

- 5,000+ grammatical rules
- Subject-verb agreement detection
- Article usage validation (a/an/the)
- Tense consistency checks
- Preposition error detection
- Anti-hallucination verification
- Morphological enrichment for LLM prompts

### Usage

```python
from kttc.helpers.english import EnglishLanguageHelper

helper = EnglishLanguageHelper()

# Check grammar
text = "He go to school every day"
errors = helper.check_grammar(text)
# Returns ErrorAnnotation for subject-verb agreement

# Get enrichment data
enrichment = helper.get_enrichment_data(text)
# Returns verb tenses, article-noun pairs, subject-verb pairs, etc.
```
```

---

### Phase 1 Deliverables

- [ ] EnglishLanguageHelper with LanguageTool integration
- [ ] Implemented `check_grammar()` method
- [ ] Enhanced `get_enrichment_data()` method
- [ ] Unit tests with >80% coverage
- [ ] Updated documentation
- [ ] Installation instructions

**Success Criteria:**
- ✅ LanguageTool initializes successfully
- ✅ `check_grammar()` finds known errors
- ✅ Tests pass with >80% coverage
- ✅ No regressions in existing functionality

---

## Phase 2: English Helper Validation

**Priority:** 🔴 HIGH
**Estimated Time:** 1-2 days
**Status:** 🔴 Blocked (depends on Phase 1)
**Owner:** TBD

### Objective

Validate that EnglishLanguageHelper with LanguageTool provides real value in translation QA.

### Tasks

#### 2.1 Prepare Test Dataset

**Checklist:**
- [ ] Collect 50+ English translation samples
- [ ] Include various translation types (technical, marketing, literary)
- [ ] Include known error cases (manually annotated)
- [ ] Include correct translations (no errors)

**Sources:**
- Existing test cases in `tests/integration/`
- WMT benchmark datasets
- Manual translation samples

**Format:**
```python
test_cases = [
    {
        "source": "Bonjour le monde",
        "translation": "Hello world",
        "expected_errors": 0,
        "category": "simple"
    },
    {
        "source": "Il va à l'école",
        "translation": "He go to school",
        "expected_errors": 1,  # Subject-verb agreement
        "category": "grammar"
    },
    # ... more cases
]
```

---

#### 2.2 Run Benchmark Tests

**Checklist:**
- [ ] Test EnglishLanguageHelper alone
- [ ] Test base FluencyAgent (LLM only)
- [ ] Test EnglishLanguageHelper + base FluencyAgent
- [ ] Measure precision, recall, F1 score
- [ ] Measure false positive rate
- [ ] Compare MQM scores

**Benchmark script:** `scripts/benchmark_english_helper.py`

```python
import asyncio
from kttc.helpers.english import EnglishLanguageHelper
from kttc.agents.fluency import FluencyAgent
from kttc.core import TranslationTask
from kttc.llm import OpenAIProvider  # Or GigaChatProvider

async def benchmark_helper():
    """Benchmark EnglishLanguageHelper."""
    helper = EnglishLanguageHelper()

    results = {
        "total": 0,
        "true_positives": 0,
        "false_positives": 0,
        "false_negatives": 0,
        "errors_found": []
    }

    for test_case in load_test_cases():
        errors = helper.check_grammar(test_case["translation"])

        # Calculate metrics
        if test_case["expected_errors"] > 0:
            if len(errors) > 0:
                results["true_positives"] += 1
            else:
                results["false_negatives"] += 1
        else:
            if len(errors) > 0:
                results["false_positives"] += 1

        results["total"] += 1
        results["errors_found"].append({
            "case": test_case,
            "errors": errors
        })

    # Calculate metrics
    precision = results["true_positives"] / (results["true_positives"] + results["false_positives"])
    recall = results["true_positives"] / (results["true_positives"] + results["false_negatives"])
    f1 = 2 * (precision * recall) / (precision + recall)

    print(f"Precision: {precision:.2%}")
    print(f"Recall: {recall:.2%}")
    print(f"F1 Score: {f1:.2%}")
    print(f"False Positive Rate: {results['false_positives'] / results['total']:.2%}")

    return results

if __name__ == "__main__":
    asyncio.run(benchmark_helper())
```

---

#### 2.3 Analyze Results

**Checklist:**
- [ ] Generate benchmark report
- [ ] Identify common false positives
- [ ] Identify missed errors (false negatives)
- [ ] Compare with LLM-only baseline
- [ ] Calculate cost savings (if any)

**Metrics to track:**
- Precision (what % of found errors are real?)
- Recall (what % of real errors are found?)
- F1 Score (harmonic mean)
- False positive rate
- MQM score improvement
- Execution time

**Decision matrix:**

| Precision | Recall | Decision |
|-----------|--------|----------|
| >80% | >70% | ✅ Proceed to create EnglishFluencyAgent |
| >70% | >60% | ⚠️ Improve filtering, then re-test |
| <70% | <60% | ❌ Keep helper-only, don't create agent |

---

#### 2.4 Generate Report

**Checklist:**
- [ ] Write benchmark report
- [ ] Include metrics and graphs
- [ ] Document findings and recommendations
- [ ] Decision: Create EnglishFluencyAgent or not?

**Report template:** `docs/benchmarks/english-helper-validation.md`

```markdown
# English Helper Validation Report

**Date:** [Date]
**Version:** EnglishLanguageHelper with LanguageTool v2.9.5

## Summary

[Executive summary of findings]

## Methodology

- Test cases: [N] translation samples
- Languages: English translations from [sources]
- Comparison: LanguageTool vs LLM-only baseline

## Results

### Metrics

| Metric | Value |
|--------|-------|
| Precision | X.X% |
| Recall | X.X% |
| F1 Score | X.X% |
| False Positive Rate | X.X% |
| Avg Execution Time | X.Xs |

### Examples

#### True Positives (Correctly Found Errors)

[Examples]

#### False Positives (Incorrectly Flagged)

[Examples]

#### False Negatives (Missed Errors)

[Examples]

## Comparison with LLM-only Baseline

[Comparison table and analysis]

## Recommendations

[Decision on whether to create EnglishFluencyAgent]
```

---

### Phase 2 Deliverables

- [ ] Benchmark test suite
- [ ] Test dataset (50+ samples)
- [ ] Benchmark results and metrics
- [ ] Validation report
- [ ] **Decision:** Create EnglishFluencyAgent or keep helper-only

**Success Criteria:**
- ✅ Precision >80%
- ✅ Recall >70%
- ✅ False positive rate <20%
- ✅ Clear value over LLM-only baseline

---

## Phase 3: Chinese Helper Enhancement

**Priority:** 🟡 MEDIUM
**Estimated Time:** 4-5 hours
**Status:** 🔴 Blocked (can run in parallel with Phase 2)
**Owner:** TBD

### Objective

Integrate HanLP into ChineseLanguageHelper for measure word validation and particle checking.

### Tasks

#### 3.1 Install HanLP Dependencies

**Checklist:**
- [ ] Install `hanlp` package
- [ ] Verify PyTorch or TensorFlow 2.x is installed
- [ ] Download Chinese models (~100 MB)
- [ ] Test HanLP initialization

**Commands:**
```bash
# Install HanLP
python3.11 -m pip install hanlp

# Test installation
python3.11 -c "import hanlp; print('HanLP OK')"
```

**Dependencies added to `pyproject.toml`:**
```toml
[project.optional-dependencies]
chinese = [
    "hanlp>=2.1.0",
]
```

---

#### 3.2 Modify ChineseLanguageHelper Class

**File:** `src/kttc/helpers/chinese.py`

**Checklist:**
- [ ] Add HanLP initialization in `__init__()`
- [ ] Handle optional dependency
- [ ] Add `_hanlp` instance variable
- [ ] Add `_hanlp_available` flag
- [ ] Create measure word dictionary

**Code changes:**

```python
# Add import at top of file
try:
    import hanlp
    HANLP_AVAILABLE = True
    logger.info("HanLP available for Chinese NLP")
except ImportError:
    HANLP_AVAILABLE = False
    logger.warning(
        "HanLP not installed. "
        "ChineseLanguageHelper will run in limited mode. "
        "Install with: pip install hanlp"
    )

class ChineseLanguageHelper(LanguageHelper):
    """Language helper for Chinese with jieba + spaCy + HanLP."""

    # Measure word dictionary
    MEASURE_WORDS = {
        "个": {"categories": ["person", "thing", "general"], "type": "general"},
        "本": {"categories": ["book", "notebook"], "type": "bound_items"},
        "只": {"categories": ["animal", "one_of_pair"], "type": "animals"},
        "条": {"categories": ["long_thin", "fish", "river"], "type": "long_thin"},
        "张": {"categories": ["flat", "paper", "table"], "type": "flat"},
        "辆": {"categories": ["vehicle"], "type": "vehicles"},
        "位": {"categories": ["person_polite"], "type": "people_formal"},
        "件": {"categories": ["piece", "clothing", "matter"], "type": "pieces"},
        "杯": {"categories": ["cup"], "type": "containers"},
        "瓶": {"categories": ["bottle"], "type": "containers"},
        "支": {"categories": ["stick", "pen", "gun"], "type": "stick_shaped"},
        "双": {"categories": ["pair", "shoes", "chopsticks"], "type": "pairs"},
        "把": {"categories": ["handle", "chair", "knife"], "type": "with_handle"},
        "颗": {"categories": ["small_round", "star", "tooth"], "type": "small_round"},
        "朵": {"categories": ["flower", "cloud"], "type": "flowers"},
        # Add more measure words as needed
    }

    def __init__(self) -> None:
        """Initialize Chinese language helper."""
        self._nlp: Any = None
        self._initialized = False

        # Initialize jieba (existing code)
        # ...

        # Initialize spaCy (existing code)
        # ...

        # Initialize HanLP (NEW)
        self._hanlp: Any = None
        self._hanlp_available = False

        if HANLP_AVAILABLE:
            try:
                # Load multi-task model
                self._hanlp = hanlp.load(
                    hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_BASE_ZH
                )
                self._hanlp_available = True
                logger.info("HanLP initialized successfully")
            except Exception as e:
                logger.warning(f"HanLP initialization failed: {e}")
                self._hanlp_available = False
```

---

#### 3.3 Implement Measure Word Validation

**Checklist:**
- [ ] Implement `_check_measure_words()` method
- [ ] Use HanLP POS tagging
- [ ] Detect missing measure words (NUM + NOUN pattern)
- [ ] Detect incorrect measure words (optional)
- [ ] Calculate accurate character positions

**Code changes:**

```python
def check_grammar(self, text: str) -> list[ErrorAnnotation]:
    """Check Chinese grammar including measure words and particles.

    Args:
        text: Chinese text to check

    Returns:
        List of detected grammar errors
    """
    if not self._hanlp_available:
        logger.debug("HanLP not available, skipping grammar checks")
        return []

    errors = []

    # Check measure words
    errors.extend(self._check_measure_words(text))

    # Check particles
    errors.extend(self._check_particles(text))

    return errors

def _check_measure_words(self, text: str) -> list[ErrorAnnotation]:
    """Check measure word usage using HanLP POS tagging.

    Args:
        text: Chinese text to check

    Returns:
        List of measure word errors
    """
    if not self._hanlp_available:
        return []

    try:
        doc = self._hanlp(text)
        tokens = doc['tok/fine']
        pos_tags = doc['pos/ctb']

        errors = []

        for i in range(len(tokens) - 1):
            # Pattern 1: CD (cardinal number) directly followed by NN (noun)
            # This indicates missing measure word
            if pos_tags[i] == 'CD':  # Cardinal number
                if i + 1 < len(pos_tags) and pos_tags[i + 1].startswith('NN'):
                    # Calculate character position
                    char_pos = sum(len(tokens[j]) for j in range(i))
                    error_length = len(tokens[i]) + len(tokens[i + 1])

                    errors.append(ErrorAnnotation(
                        category="fluency",
                        subcategory="chinese_missing_measure_word",
                        severity=ErrorSeverity.MAJOR,
                        location=(char_pos, char_pos + error_length),
                        description=(
                            f"Missing measure word between number '{tokens[i]}' "
                            f"and noun '{tokens[i + 1]}'"
                        ),
                        suggestion="Insert appropriate measure word (个/本/只/条/etc.)"
                    ))

            # Pattern 2: CD + M (measure word) + NN
            # Could validate if measure word is appropriate for noun
            elif pos_tags[i] == 'CD' and i + 2 < len(pos_tags):
                if pos_tags[i + 1] == 'M' and pos_tags[i + 2].startswith('NN'):
                    measure_word = tokens[i + 1]
                    noun = tokens[i + 2]

                    # Log for analysis (validation needs comprehensive dictionary)
                    if measure_word in self.MEASURE_WORDS:
                        logger.debug(
                            f"Measure word '{measure_word}' used with '{noun}' "
                            f"(type: {self.MEASURE_WORDS[measure_word]['type']})"
                        )

        logger.debug(f"Found {len(errors)} measure word errors")
        return errors

    except Exception as e:
        logger.error(f"Measure word check failed: {e}")
        return []
```

---

#### 3.4 Implement Particle Checking

**Checklist:**
- [ ] Implement `_check_particles()` method
- [ ] Check aspect particles (了/过)
- [ ] Check structural particles (的/地/得) - basic only
- [ ] Use conservative rules (avoid false positives)

**Code changes:**

```python
def _check_particles(self, text: str) -> list[ErrorAnnotation]:
    """Check Chinese particle usage (了/过/的/地/得/吗/呢).

    Args:
        text: Chinese text to check

    Returns:
        List of particle errors
    """
    errors = []

    # Check aspect particles
    errors.extend(self._check_aspect_particles(text))

    # Check structural particles (optional - may have high false positive rate)
    # errors.extend(self._check_structural_particles(text))

    return errors

def _check_aspect_particles(self, text: str) -> list[ErrorAnnotation]:
    """Check aspect particles 了 (le) and 过 (guo).

    Args:
        text: Chinese text to check

    Returns:
        List of aspect particle errors
    """
    if not self._hanlp_available:
        return []

    try:
        doc = self._hanlp(text)
        tokens = doc['tok/fine']
        pos_tags = doc['pos/ctb']

        errors = []

        # Past time indicators
        past_indicators = ["昨天", "以前", "已经", "刚才", "刚", "曾经"]

        # Check if text contains past time reference
        has_past_indicator = any(indicator in tokens for indicator in past_indicators)

        if has_past_indicator:
            # Check for aspect markers
            has_aspect = any(token in ['了', '过'] for token in tokens)

            if not has_aspect:
                # Find verb position
                for i, tag in enumerate(pos_tags):
                    if tag.startswith('V'):  # Verb
                        char_pos = sum(len(tokens[j]) for j in range(i))

                        errors.append(ErrorAnnotation(
                            category="fluency",
                            subcategory="chinese_missing_aspect_particle",
                            severity=ErrorSeverity.MINOR,  # MINOR because it's often optional
                            location=(char_pos, char_pos + len(tokens[i])),
                            description=(
                                "Past time reference without aspect marker. "
                                "Consider adding 了 (completed action) or 过 (experience)"
                            ),
                            suggestion="Add 了 or 过 after verb to indicate aspect"
                        ))
                        break  # Only flag once per sentence

        logger.debug(f"Found {len(errors)} aspect particle issues")
        return errors

    except Exception as e:
        logger.error(f"Aspect particle check failed: {e}")
        return []
```

---

#### 3.5 Enhance get_enrichment_data()

**Checklist:**
- [ ] Extract measure words with positions
- [ ] Extract particles with categories
- [ ] Add dependency parsing info
- [ ] Add POS distribution

**Code changes:**

```python
def get_enrichment_data(self, text: str) -> dict[str, Any]:
    """Get comprehensive linguistic data for enriching LLM prompts.

    Args:
        text: Text to analyze

    Returns:
        Dictionary with linguistic insights for LLM
    """
    if not self._hanlp_available:
        return {"has_morphology": False}

    try:
        doc = self._hanlp(text)
        tokens = doc['tok/fine']
        pos_tags = doc['pos/ctb']

        # Extract measure words
        measure_words = []
        for i, tag in enumerate(pos_tags):
            if tag == 'M':  # Measure word
                mw_info = self.MEASURE_WORDS.get(tokens[i], {"type": "unknown"})
                measure_words.append({
                    "word": tokens[i],
                    "position": i,
                    "type": mw_info.get("type", "unknown"),
                    "categories": mw_info.get("categories", [])
                })

        # Extract particles
        particles = []
        particle_map = {
            "了": "aspect",
            "过": "aspect",
            "的": "structural",
            "地": "structural",
            "得": "structural",
            "吗": "modal",
            "呢": "modal",
            "吧": "modal",
            "啊": "modal"
        }

        for i, token in enumerate(tokens):
            if token in particle_map:
                particles.append({
                    "particle": token,
                    "position": i,
                    "category": particle_map[token]
                })

        # Named entities
        entities = []
        for ent in doc.get('ner/msra', []):
            entities.append({
                "text": ent[0] if isinstance(ent, tuple) else ent,
                "label": ent[1] if isinstance(ent, tuple) and len(ent) > 1 else "UNKNOWN",
            })

        # POS distribution
        pos_counts: dict[str, int] = {}
        for tag in pos_tags:
            pos_counts[tag] = pos_counts.get(tag, 0) + 1

        return {
            "has_morphology": True,
            "word_count": len([t for t in tokens if t.strip()]),
            "measure_words": measure_words,
            "particles": particles,
            "pos_distribution": pos_counts,
            "entities": entities,
            "has_dependency": 'dep' in doc,
            "sentence_count": len(text.split('。')) if '。' in text else 1
        }

    except Exception as e:
        logger.error(f"Enrichment data extraction failed: {e}")
        return {"has_morphology": False}
```

---

#### 3.6 Update Unit Tests

**File:** `tests/unit/test_chinese_helper.py`

**Checklist:**
- [ ] Test HanLP initialization
- [ ] Test measure word detection
- [ ] Test particle checking
- [ ] Test enrichment data
- [ ] Test graceful degradation

**Test code:**

```python
import pytest
from kttc.helpers.chinese import ChineseLanguageHelper
from kttc.core import ErrorSeverity


class TestChineseLanguageHelper:
    """Test ChineseLanguageHelper with HanLP integration."""

    @pytest.fixture
    def helper(self):
        """Create helper instance."""
        return ChineseLanguageHelper()

    def test_initialization(self, helper):
        """Test helper initializes correctly."""
        assert helper is not None
        assert helper.language_code == "zh"

    def test_check_measure_word_missing(self, helper):
        """Test missing measure word detection."""
        if not helper._hanlp_available:
            pytest.skip("HanLP not available")

        text = "他买了三书"  # Missing measure word (should be 三本书)
        errors = helper.check_grammar(text)

        # Should find missing measure word
        assert len(errors) > 0
        assert any("measure_word" in err.subcategory for err in errors)

    def test_check_measure_word_correct(self, helper):
        """Test correct measure word usage."""
        if not helper._hanlp_available:
            pytest.skip("HanLP not available")

        text = "他买了三本书"  # Correct: 三本书
        errors = helper.check_grammar(text)

        # Should have no measure word errors
        measure_word_errors = [e for e in errors if "measure_word" in e.subcategory]
        assert len(measure_word_errors) == 0

    def test_check_aspect_particle(self, helper):
        """Test aspect particle checking."""
        if not helper._hanlp_available:
            pytest.skip("HanLP not available")

        text = "他昨天去学校"  # Past indicator without aspect marker
        errors = helper.check_grammar(text)

        # May suggest adding 了/过
        # (Note: This is MINOR severity, may not always flag)
        if len(errors) > 0:
            assert any("aspect" in err.subcategory.lower() for err in errors)

    def test_enrichment_data(self, helper):
        """Test enrichment data extraction."""
        if not helper._hanlp_available:
            pytest.skip("HanLP not available")

        text = "他买了三本书"
        enrichment = helper.get_enrichment_data(text)

        assert enrichment["has_morphology"] is True
        assert "measure_words" in enrichment
        assert "particles" in enrichment
        assert len(enrichment["measure_words"]) > 0  # Should find "本"
        assert len(enrichment["particles"]) > 0  # Should find "了"
```

---

### Phase 3 Deliverables

- [ ] ChineseLanguageHelper with HanLP integration
- [ ] Measure word validation
- [ ] Basic particle checking
- [ ] Enhanced enrichment data
- [ ] Unit tests with >80% coverage
- [ ] Updated documentation

**Success Criteria:**
- ✅ HanLP initializes successfully
- ✅ Measure word detection works
- ✅ Particle checking provides suggestions
- ✅ Tests pass with >80% coverage

---

## Phase 4: Chinese Helper Validation

**Priority:** 🟡 MEDIUM
**Estimated Time:** 1-2 days
**Status:** 🔴 Blocked (depends on Phase 3)
**Owner:** TBD

### Objective

Validate that ChineseLanguageHelper with HanLP provides real value.

### Tasks

#### 4.1 Prepare Test Dataset

**Checklist:**
- [ ] Collect 50+ Chinese translation samples
- [ ] Include measure word error cases
- [ ] Include particle usage cases
- [ ] Include correct translations
- [ ] Get native speaker validation (if possible)

---

#### 4.2 Run Benchmark Tests

**Similar to Phase 2, but for Chinese**

**Checklist:**
- [ ] Test measure word detection
- [ ] Test particle checking
- [ ] Measure precision/recall
- [ ] Measure false positive rate
- [ ] Compare with LLM-only baseline

---

#### 4.3 Decision Matrix

| Measure Words Precision | Particle Precision | Decision |
|------------------------|-------------------|----------|
| >70% | >60% | ✅ Create ChineseFluencyAgent |
| >60% | <60% | ⚠️ Use measure words only, disable particles |
| <60% | <60% | ❌ Keep helper-only, basic checks |

---

### Phase 4 Deliverables

- [ ] Benchmark results for Chinese
- [ ] Validation report
- [ ] **Decision:** Create ChineseFluencyAgent or keep helper-only

---

## Phase 5: Specialized Agents (Optional)

**Priority:** 🟢 LOW (only after Phases 1-4)
**Estimated Time:** 2-3 hours per agent
**Status:** 🔴 Blocked (depends on Phase 2 & 4 decisions)
**Owner:** TBD

### Objective

Create EnglishFluencyAgent and/or ChineseFluencyAgent if validation shows clear value.

### Tasks

#### 5.1 Create EnglishFluencyAgent (if approved)

**File:** `src/kttc/agents/fluency_english.py`

**Architecture:** Follow RussianFluencyAgent pattern

```python
from __future__ import annotations

import asyncio
import logging
from typing import cast

from kttc.core import ErrorAnnotation, TranslationTask
from kttc.helpers.english import EnglishLanguageHelper
from kttc.llm import BaseLLMProvider

from .fluency import FluencyAgent

logger = logging.getLogger(__name__)


class EnglishFluencyAgent(FluencyAgent):
    """Specialized fluency agent for English language.

    Extends base FluencyAgent with English-specific checks:
    - Subject-verb agreement
    - Article usage (a/an/the)
    - Tense consistency
    - Preposition correctness
    - 5,000+ LanguageTool grammar rules

    Uses hybrid approach:
    - LanguageTool for deterministic grammar checks
    - LLM for semantic and complex linguistic analysis
    - Parallel execution for optimal performance
    """

    def __init__(
        self,
        llm_provider: BaseLLMProvider,
        temperature: float = 0.1,
        max_tokens: int = 2000,
        helper: EnglishLanguageHelper | None = None,
    ):
        """Initialize English fluency agent."""
        super().__init__(llm_provider, temperature, max_tokens)

        # Initialize helper
        self.helper = helper if helper is not None else EnglishLanguageHelper()

        if self.helper.is_available():
            logger.info("EnglishFluencyAgent using spaCy + LanguageTool")
        else:
            logger.info("EnglishFluencyAgent running in LLM-only mode")

    async def evaluate(self, task: TranslationTask) -> list[ErrorAnnotation]:
        """Evaluate English fluency with hybrid NLP + LLM approach."""
        if task.target_lang != "en":
            return await super().evaluate(task)

        # Run base fluency checks
        base_errors = await super().evaluate(task)

        # Run NLP, LLM, and entity checks in parallel
        try:
            results = await asyncio.gather(
                self._nlp_check(task),
                self._llm_check(task),
                self._entity_check(task),
                return_exceptions=True,
            )

            # Handle results (similar to RussianFluencyAgent)
            nlp_errors = results[0] if not isinstance(results[0], Exception) else []
            llm_errors = results[1] if not isinstance(results[1], Exception) else []
            entity_errors = results[2] if not isinstance(results[2], Exception) else []

            # Verify LLM results with NLP
            verified_llm = self._verify_llm_errors(llm_errors, task.translation)

            # Remove duplicates
            unique_nlp = self._remove_duplicates(nlp_errors, verified_llm)

            all_errors = base_errors + unique_nlp + verified_llm + entity_errors

            logger.info(
                f"EnglishFluencyAgent: "
                f"base={len(base_errors)}, "
                f"nlp={len(unique_nlp)}, "
                f"llm={len(verified_llm)}, "
                f"entity={len(entity_errors)} "
                f"(total={len(all_errors)})"
            )

            return all_errors

        except Exception as e:
            logger.error(f"English fluency evaluation failed: {e}")
            return base_errors

    async def _nlp_check(self, task: TranslationTask) -> list[ErrorAnnotation]:
        """Perform LanguageTool grammar checks."""
        if not self.helper or not self.helper._lt_available:
            return []

        try:
            errors = self.helper.check_grammar(task.translation)
            logger.debug(f"LanguageTool found {len(errors)} grammar errors")
            return errors
        except Exception as e:
            logger.error(f"NLP check failed: {e}")
            return []

    async def _llm_check(self, task: TranslationTask) -> list[ErrorAnnotation]:
        """Perform LLM-based English-specific checks with enrichment."""
        try:
            errors = await self._check_english_specifics(task)
            logger.debug(f"LLM found {len(errors)} English-specific errors")
            return errors
        except Exception as e:
            logger.error(f"LLM check failed: {e}")
            return []

    async def _entity_check(self, task: TranslationTask) -> list[ErrorAnnotation]:
        """Perform NER-based entity preservation checks."""
        if not self.helper or not self.helper.is_available():
            return []

        try:
            errors = self.helper.check_entity_preservation(task.source_text, task.translation)
            logger.debug(f"Entity check found {len(errors)} preservation issues")
            return errors
        except Exception as e:
            logger.error(f"Entity check failed: {e}")
            return []

    async def _check_english_specifics(self, task: TranslationTask) -> list[ErrorAnnotation]:
        """Perform English-specific fluency checks with linguistic context."""
        # Get enrichment data
        enrichment_section = ""
        if self.helper and self.helper.is_available():
            enrichment = self.helper.get_enrichment_data(task.translation)

            if enrichment.get("has_morphology"):
                enrichment_section = "\n## LINGUISTIC ANALYSIS (for context):\n"

                # Add verb tense information
                if enrichment.get("verb_tenses"):
                    enrichment_section += "\n**Verbs in translation:**\n"
                    for verb, info in enrichment["verb_tenses"].items():
                        enrichment_section += f"- '{verb}': {info['tense']} tense"
                        if info.get('person'):
                            enrichment_section += f", {info['person']} person"
                        enrichment_section += "\n"

                # Add article-noun pairs
                if enrichment.get("article_noun_pairs"):
                    enrichment_section += "\n**Article-Noun pairs:**\n"
                    for pair in enrichment["article_noun_pairs"]:
                        status = "✓ correct" if pair["correct"] else "⚠ CHECK"
                        enrichment_section += (
                            f"- '{pair['article']}' + '{pair['noun']}' - {status}\n"
                        )

                # Add subject-verb pairs
                if enrichment.get("subject_verb_pairs"):
                    enrichment_section += "\n**Subject-Verb pairs:**\n"
                    for pair in enrichment["subject_verb_pairs"]:
                        agreement = "✓ agreement OK" if pair.get("agreement") else "⚠ CHECK agreement"
                        enrichment_section += (
                            f"- '{pair['subject']}' ({pair.get('subject_number', '?')}) + "
                            f"'{pair['verb']}' ({pair.get('verb_number', '?')}) - {agreement}\n"
                        )

        # Build prompt (similar to Russian agent)
        prompt = f"""You are a native English speaker and professional translator.

Your task: Identify ONLY clear English-specific linguistic errors in the translation.

## SOURCE TEXT:
{task.source_text}
{enrichment_section}
## TRANSLATION (English):
{task.translation}

## IMPORTANT GUIDELINES:

**What IS an error:**
- Clear grammatical mistakes (subject-verb agreement, tense errors)
- Incorrect article usage (a/an/the)
- Unnatural constructions that no native speaker would use
- Preposition errors that affect meaning

**What is NOT an error:**
- Stylistic preferences (multiple phrasings are often correct)
- Direct translations that are grammatically correct
- Natural English that differs from your personal preference
- Minor stylistic variations

## CHECKS TO PERFORM:

1. **Subject-Verb Agreement** - ONLY flag clear violations
2. **Tense Consistency** - Check if tenses match source context
3. **Article Usage** - a/an/the correctness
4. **Prepositions** - ONLY clear mistakes

Output JSON format:
{{
  "errors": [
    {{
      "subcategory": "subject_verb_agreement|tense|article|preposition",
      "severity": "critical|major|minor",
      "location": [start_char, end_char],
      "description": "Specific English linguistic issue",
      "suggestion": "Corrected version"
    }}
  ]
}}

Rules:
- CONSERVATIVE: Only report clear, unambiguous errors
- VERIFY: Ensure the word/phrase you mention actually exists in the text
- CONTEXT: Consider the source text when evaluating
- If the translation is natural and grammatically correct, return empty errors array

Output only valid JSON, no explanation."""

        # Send to LLM and parse (similar to Russian agent)
        try:
            response = await self.llm_provider.complete(
                prompt, temperature=self.temperature, max_tokens=self.max_tokens
            )

            response_data = self._parse_json_response(response)
            errors_data = response_data.get("errors", [])

            errors = []
            for error_dict in errors_data:
                location = error_dict.get("location", [0, 10])
                location_tuple = (location[0], location[1]) if isinstance(location, list) else (0, 10)

                from kttc.core import ErrorSeverity
                errors.append(
                    ErrorAnnotation(
                        category="fluency",
                        subcategory=f"english_{error_dict.get('subcategory', 'specific')}",
                        severity=ErrorSeverity(error_dict.get("severity", "minor")),
                        location=location_tuple,
                        description=error_dict.get("description", "English linguistic issue"),
                        suggestion=error_dict.get("suggestion"),
                    )
                )

            return errors

        except Exception as e:
            logger.error(f"English-specific check failed: {e}")
            return []

    # Include helper methods from RussianFluencyAgent:
    # - _verify_llm_errors()
    # - _remove_duplicates()
    # - _errors_overlap()
    # - _parse_json_response()
```

---

#### 5.2 Create ChineseFluencyAgent (if approved)

**Similar architecture to EnglishFluencyAgent, but with:**
- Measure word enrichment
- Particle enrichment
- Chinese-specific LLM prompt

---

#### 5.3 Update Dynamic Selector

**File:** `src/kttc/agents/dynamic_selector.py`

**Uncomment language-specific agent code:**

```python
def _add_language_specific_agents(self, agent_set: list[str], task: TranslationTask):
    """Add language-specific agents if available."""

    # Russian-specific fluency agent
    if task.target_lang == "ru":
        if "fluency" in agent_set and "fluency_russian" not in agent_set:
            agent_set.remove("fluency")
            agent_set.append("fluency_russian")

    # English-specific fluency agent (NEW)
    elif task.target_lang == "en":
        if "fluency" in agent_set and "fluency_english" not in agent_set:
            agent_set.remove("fluency")
            agent_set.append("fluency_english")

    # Chinese-specific fluency agent (NEW)
    elif task.target_lang == "zh":
        if "fluency" in agent_set and "fluency_chinese" not in agent_set:
            agent_set.remove("fluency")
            agent_set.append("fluency_chinese")

    return agent_set
```

---

### Phase 5 Deliverables

- [ ] EnglishFluencyAgent (if approved)
- [ ] ChineseFluencyAgent (if approved)
- [ ] Updated dynamic selector
- [ ] Unit tests for new agents
- [ ] Integration tests
- [ ] Updated documentation

---

## Dependencies & Requirements

### System Requirements

| Requirement | Version | Purpose |
|-------------|---------|---------|
| Python | 3.11+ | Base runtime |
| Java | 17.0+ | LanguageTool (English) |
| PyTorch / TensorFlow | 2.x | HanLP (Chinese) |

### Python Dependencies

```toml
[project.optional-dependencies]
english = [
    "language-tool-python>=2.9.0",
]

chinese = [
    "hanlp>=2.1.0",
]

all-languages = [
    "language-tool-python>=2.9.0",
    "hanlp>=2.1.0",
]
```

### Disk Space Requirements

| Component | Size |
|-----------|------|
| LanguageTool server | ~200 MB |
| HanLP Chinese models | ~100 MB |
| spaCy en_core_web_md | ~50 MB |
| spaCy zh_core_web_md | ~74 MB |
| **Total** | **~424 MB** |

---

## Testing Strategy

### Unit Tests

**Coverage target:** >80% for all new code

**Test files:**
- `tests/unit/test_english_helper.py`
- `tests/unit/test_chinese_helper.py`
- `tests/unit/test_fluency_english.py` (if agent created)
- `tests/unit/test_fluency_chinese.py` (if agent created)

**Run tests:**
```bash
python3.11 -m pytest tests/unit/ --cov=kttc --cov-report=html
```

### Integration Tests

**Test files:**
- `tests/integration/test_english_translation_qa.py`
- `tests/integration/test_chinese_translation_qa.py`

**Run tests:**
```bash
python3.11 -m pytest tests/integration/ -v
```

### Benchmark Tests

**Scripts:**
- `scripts/benchmark_english_helper.py`
- `scripts/benchmark_chinese_helper.py`

**Run benchmarks:**
```bash
python3.11 scripts/benchmark_english_helper.py
python3.11 scripts/benchmark_chinese_helper.py
```

---

## Success Metrics

### Phase 1 & 3 (Helper Enhancement)

- ✅ Installation successful
- ✅ `check_grammar()` implemented and working
- ✅ Unit test coverage >80%
- ✅ No regressions in existing functionality

### Phase 2 & 4 (Helper Validation)

| Metric | Target | Status |
|--------|--------|--------|
| Precision | >80% | 🔴 TBD |
| Recall | >70% | 🔴 TBD |
| F1 Score | >75% | 🔴 TBD |
| False Positive Rate | <20% | 🔴 TBD |
| Execution Time | <5s per translation | 🔴 TBD |

### Phase 5 (Agent Creation)

- ✅ Agent follows RussianFluencyAgent architecture
- ✅ Hybrid NLP+LLM approach working
- ✅ Anti-hallucination verification effective
- ✅ MQM score improvement >5% over base FluencyAgent

---

## Risk Mitigation

### High Risk Items

1. **LanguageTool large download size (~200 MB)**
   - Mitigation: Make optional dependency, document clearly
   - Fallback: Graceful degradation to spaCy-only

2. **HanLP requires PyTorch/TensorFlow**
   - Mitigation: Make optional dependency
   - Fallback: Keep jieba-only mode

3. **False positives in particle checking (Chinese)**
   - Mitigation: Use conservative rules, mark as suggestions
   - Fallback: Disable particle checks, keep measure words only

### Medium Risk Items

1. **Java dependency for LanguageTool**
   - Mitigation: Document Java 17+ requirement clearly
   - Provide installation instructions for common platforms

2. **Linguistic expertise needed for Chinese rules**
   - Mitigation: Start with conservative measure word rules
   - Consult native speakers for validation

---

## Timeline Estimate

| Phase | Duration | Dependencies | Status |
|-------|----------|-------------|--------|
| Phase 1: English Helper | 2-3 hours | None | 🟡 Ready |
| Phase 2: English Validation | 1-2 days | Phase 1 | 🔴 Blocked |
| Phase 3: Chinese Helper | 4-5 hours | None | 🟡 Ready |
| Phase 4: Chinese Validation | 1-2 days | Phase 3 | 🔴 Blocked |
| Phase 5: Agents (if approved) | 2-3 hours each | Phases 2 & 4 | 🔴 Blocked |

**Total estimated time:** 2-3 weeks (if all phases approved)

---

## Next Steps

### Immediate Actions (Week 1)

1. ✅ **Review this roadmap** - Team review and approval
2. 🟡 **Start Phase 1** - English Helper enhancement
3. 🟡 **Prepare test datasets** - Collect translation samples

### Short-term (Week 2-3)

4. 🔴 **Run Phase 2** - Validate English Helper
5. 🔴 **Start Phase 3** - Chinese Helper enhancement
6. 🔴 **Make decision** - Create EnglishFluencyAgent or not?

### Medium-term (Week 3-4)

7. 🔴 **Run Phase 4** - Validate Chinese Helper
8. 🔴 **Make decision** - Create ChineseFluencyAgent or not?
9. 🔴 **Phase 5 (optional)** - Create specialized agents

---

**Roadmap Version:** 1.0
**Last Updated:** November 13, 2025
**Status:** 🟡 Ready to start Phase 1
