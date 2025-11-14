# NLP Tools Research for English and Chinese Language Helpers Enhancement

**Date:** November 13, 2025
**Author:** Research conducted for KTTC AI Translation Quality Assessment
**Objective:** Research modern NLP tools and approaches to enhance EnglishLanguageHelper and ChineseLanguageHelper with deterministic grammar checking capabilities comparable to RussianLanguageHelper

---

## Executive Summary

After extensive research of 2024-2025 NLP tools, academic papers, and industry solutions, we found that:

1. **English:** Multiple mature tools exist for grammar checking (LanguageTool, GECToR, spaCy-based solutions)
2. **Chinese:** Fewer specialized grammar checkers, but strong foundational NLP libraries (HanLP, spaCy zh models, Stanford CoreNLP)
3. **Recent trend:** LLM-based evaluation is replacing rule-based systems, but hybrid approaches (NLP + LLM) remain most reliable
4. **Key insight:** RussianLanguageHelper's power comes from mawo-pymorphy3's morphological analysis — English and Chinese need similar specialized tools

---

## Part 1: Current State Analysis

### RussianLanguageHelper Analysis

**Location:** `src/kttc/agents/fluency_russian.py:49-511`

**Key Features:**
- **Hybrid approach:** NLP (mawo-pymorphy3 + mawo-razdel) + LLM in parallel
- **Deterministic checks:**
  - Case agreement (6 cases: Nominative, Genitive, Dative, Accusative, Instrumental, Prepositional)
  - Verb aspect (perfective/imperfective)
  - Particle usage (же, ли, бы)
  - Register consistency (ты/вы)
  - Word order validation
- **Anti-hallucination:** NLP verifies LLM results
- **Enrichment data:** Morphological context for LLM prompts (verb aspects, adjective-noun pairs)

**Architecture:**
```python
async def evaluate(self, task: TranslationTask):
    # Run in parallel
    results = await asyncio.gather(
        self._nlp_check(task),      # Fast, deterministic
        self._llm_check(task),       # Slow, semantic
        self._entity_check(task),    # NER-based
    )

    # Verify LLM results with NLP (anti-hallucination)
    verified_llm = self._verify_llm_errors(llm_errors, text)

    # Remove duplicates
    unique_nlp = self._remove_duplicates(nlp_errors, verified_llm)

    return base_errors + unique_nlp + verified_llm + entity_errors
```

### Current Agent Activation Logic

**Location:** `src/kttc/agents/dynamic_selector.py:311-324`

**Implementation:**
```python
def _add_language_specific_agents(self, agent_set: list[str], task: TranslationTask):
    """Add language-specific agents if available."""

    # Russian-specific fluency agent
    if task.target_lang == "ru":
        # Replace generic fluency with Russian-specific
        if "fluency" in agent_set and "fluency_russian" not in agent_set:
            agent_set.remove("fluency")
            agent_set.append("fluency_russian")

    # Future: Add more language-specific agents
    # elif task.target_lang == "zh":
    #     agent_set.append("fluency_chinese")
    # elif task.target_lang == "en":
    #     agent_set.append("fluency_english")

    return agent_set
```

**Conclusion:** ✅ RussianFluencyAgent is **already** activated only for Russian language (`target_lang == "ru"`), not always active.

### EnglishLanguageHelper - Current State

**Location:** `src/kttc/helpers/english.py`

**Current capabilities:**
- ✅ spaCy integration (en_core_web_md/sm)
- ✅ Tokenization with accurate positions
- ✅ POS tagging and morphological analysis
- ✅ Named entity recognition (NER)
- ✅ Entity preservation checks
- ✅ Anti-hallucination verification
- ❌ **`check_grammar()` returns empty list** (line 215-216)

**Missing features:**
- Subject-verb agreement detection
- Article usage validation (a/an/the)
- Tense consistency checks
- Preposition error detection

### ChineseLanguageHelper - Current State

**Location:** `src/kttc/helpers/chinese.py`

**Current capabilities:**
- ✅ jieba word segmentation (7 MB, fast)
- ✅ spaCy zh models (optional, 46-74 MB)
- ✅ Tokenization with positions
- ✅ POS tagging (if spaCy available)
- ✅ Named entity recognition
- ✅ Entity preservation checks
- ❌ **`check_grammar()` returns empty list** (line 232-237)

**Missing features:**
- Measure word (量词) validation
- Particle checking (了的吗呢)
- Grammatical structure validation

---

## Part 2: English Language Enhancement

### Recommended Tool: LanguageTool ⭐⭐⭐⭐⭐

**Library:** `language-tool-python` v2.9.5 (November 2025)
**GitHub:** https://github.com/myint/language-check
**PyPI:** https://pypi.org/project/language-tool-python/
**License:** LGPL (open-source)

#### Capabilities

- **5,000+ grammatical rules** for English
- **Error categories:**
  - Subject-verb agreement
  - Article usage (a/an/the)
  - Tense consistency
  - Preposition errors
  - Passive voice detection
  - Spelling mistakes
  - Style issues

#### Installation

```bash
pip install language-tool-python
```

**Requirements:**
- Python 3.9+
- Java 17.0+ (for versions ≥6.6)
- ~200 MB download for LanguageTool server

#### API Usage

```python
import language_tool_python

# Initialize
tool = language_tool_python.LanguageTool('en-US')

# Check text
text = "He go to school every day"
matches = tool.check(text)

# Each match contains:
# - ruleId: Error type identifier
# - message: Human-readable description
# - offset: Start position in text
# - errorLength: Length of error
# - replacements: List of suggestions

# Auto-correct
corrected = tool.correct(text)  # "He goes to school every day"
```

#### Integration Strategy

```python
class EnglishLanguageHelper(LanguageHelper):
    """Enhanced English helper with LanguageTool integration."""

    def __init__(self) -> None:
        self._spacy_nlp = spacy.load("en_core_web_md")

        # Initialize LanguageTool
        try:
            self._language_tool = language_tool_python.LanguageTool('en-US')
            self._lt_available = True
            logger.info("LanguageTool initialized successfully")
        except Exception as e:
            logger.warning(f"LanguageTool not available: {e}")
            self._lt_available = False

    def check_grammar(self, text: str) -> list[ErrorAnnotation]:
        """Check English grammar using LanguageTool.

        Returns:
            List of grammar errors with positions and suggestions
        """
        if not self._lt_available:
            return []

        try:
            matches = self._language_tool.check(text)
            errors = []

            for match in matches:
                # Map LanguageTool severity to our system
                severity = self._map_severity(match)

                # Skip style-only suggestions for translation QA
                if self._is_translation_relevant(match):
                    errors.append(ErrorAnnotation(
                        category="fluency",
                        subcategory=f"english_{match.ruleId}",
                        severity=severity,
                        location=(match.offset, match.offset + match.errorLength),
                        description=match.message,
                        suggestion=match.replacements[0] if match.replacements else None
                    ))

            return errors

        except Exception as e:
            logger.error(f"LanguageTool check failed: {e}")
            return []

    def _map_severity(self, match) -> ErrorSeverity:
        """Map LanguageTool match to ErrorSeverity."""
        # LanguageTool doesn't have severity, so we infer from rule category
        rule_id = match.ruleId.lower()

        if any(x in rule_id for x in ['grammar', 'agreement', 'verb']):
            return ErrorSeverity.MAJOR
        elif any(x in rule_id for x in ['spelling', 'typo']):
            return ErrorSeverity.CRITICAL
        else:
            return ErrorSeverity.MINOR

    def _is_translation_relevant(self, match) -> bool:
        """Filter out style-only suggestions not relevant for translation QA."""
        rule_id = match.ruleId.lower()

        # Exclude pure style suggestions
        exclude_patterns = ['style', 'redundancy', 'collocation']
        if any(pattern in rule_id for pattern in exclude_patterns):
            return False

        return True
```

### Advanced: spaCy Dependency Parsing for Custom Rules

For rules not covered by LanguageTool, we can use spaCy's dependency parsing:

#### Subject-Verb Agreement Detection

```python
def check_subject_verb_agreement(self, text: str) -> list[ErrorAnnotation]:
    """Check subject-verb agreement using spaCy dependency parsing."""
    if not self.is_available():
        return []

    doc = self._spacy_nlp(text)
    errors = []

    for token in doc:
        # Find subjects (nsubj = nominal subject)
        if token.dep_ == "nsubj":
            verb = token.head

            # Check if head is a verb
            if verb.pos_ == "VERB":
                # Get morphological number
                subject_number = token.morph.get("Number")
                verb_number = verb.morph.get("Number")

                if subject_number and verb_number:
                    # Check agreement
                    if subject_number[0] != verb_number[0]:
                        errors.append(ErrorAnnotation(
                            category="fluency",
                            subcategory="english_subject_verb_agreement",
                            severity=ErrorSeverity.MAJOR,
                            location=(verb.idx, verb.idx + len(verb.text)),
                            description=(
                                f"Subject-verb agreement error: "
                                f"'{token.text}' is {subject_number[0]} "
                                f"but verb '{verb.text}' is {verb_number[0]}"
                            ),
                            suggestion=None  # Could add lemmatization-based suggestion
                        ))

    return errors
```

### Enrichment Data for LLM Prompts

Similar to RussianFluencyAgent's verb aspect enrichment:

```python
def get_enrichment_data(self, text: str) -> dict[str, Any]:
    """Get comprehensive linguistic data for enriching LLM prompts."""
    if not self.is_available():
        return {"has_morphology": False}

    doc = self._spacy_nlp(text)

    # Tense analysis
    verb_tenses = {}
    for token in doc:
        if token.pos_ == "VERB":
            tense = token.morph.get("Tense")
            if tense:
                verb_tenses[token.text] = {
                    "tense": tense[0],
                    "aspect": token.morph.get("Aspect", [""])[0],
                    "person": token.morph.get("Person", [""])[0]
                }

    # Article-noun patterns
    article_noun_pairs = []
    for i, token in enumerate(doc):
        if token.pos_ == "DET" and token.text.lower() in ["a", "an", "the"]:
            # Find associated noun
            if i + 1 < len(doc):
                next_token = doc[i + 1]
                if next_token.pos_ in ["NOUN", "PROPN"]:
                    # Check if article matches (a vs an)
                    correct_article = "an" if next_token.text[0].lower() in "aeiou" else "a"
                    article_noun_pairs.append({
                        "article": token.text.lower(),
                        "noun": next_token.text,
                        "correct": token.text.lower() == correct_article if token.text.lower() != "the" else True
                    })

    # POS distribution
    pos_counts: dict[str, int] = {}
    for token in doc:
        if token.pos_:
            pos_counts[token.pos_] = pos_counts.get(token.pos_, 0) + 1

    # Named entities
    entities = []
    for ent in doc.ents:
        entities.append({
            "text": ent.text,
            "label": ent.label_,
            "start": ent.start_char,
            "end": ent.end_char
        })

    return {
        "has_morphology": True,
        "word_count": len([token for token in doc if not token.is_punct]),
        "verb_tenses": verb_tenses,
        "article_noun_pairs": article_noun_pairs,
        "pos_distribution": pos_counts,
        "entities": entities,
        "sentence_count": len(list(doc.sents))
    }
```

---

## Part 3: Chinese Language Enhancement

### Recommended Tool: HanLP ⭐⭐⭐⭐⭐

**Website:** https://hanlp.hankcs.com
**GitHub:** https://github.com/hankcs/HanLP
**PyPI:** https://pypi.org/project/hanlp/
**Maintainer:** Harbin Institute of Technology

#### Capabilities

- **Multi-task learning** framework (PyTorch/TensorFlow 2.x)
- **Chinese NLP tasks:**
  - Word segmentation (multiple standards: CTB, MSR, PKU)
  - Part-of-speech tagging (includes measure word classification)
  - Named entity recognition (PKU, MSRA, OntoNotes)
  - Dependency parsing
  - Semantic dependency parsing
  - Constituency parsing
  - Semantic role labeling
  - Text classification

#### Installation

```bash
pip install hanlp
```

**Requirements:**
- PyTorch or TensorFlow 2.x
- ~100 MB for Chinese models

#### API Usage

```python
import hanlp

# Load multi-task model
HanLP = hanlp.load(hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_BASE_ZH)

# Analyze text
text = "他买了三本书"
doc = HanLP(text)

# Returns:
# {
#   'tok/fine': ['他', '买', '了', '三', '本', '书'],
#   'pos/ctb': ['PN', 'VV', 'AS', 'CD', 'M', 'NN'],
#   'ner/msra': [],
#   'srl': [...],
#   'dep': [...],
#   'sdp': [...],
#   'con': [...]
# }
```

### Measure Word Validation

**Challenge:** Measure words (量词 liàngcí) are mandatory in Chinese: `NUM + MEASURE_WORD + NOUN`

**Common errors:**
- Missing measure word: ❌ "三书" (should be "三本书")
- Wrong measure word: ❌ "三只书" (should be "三本书")

**Implementation:**

```python
class ChineseLanguageHelper(LanguageHelper):
    """Enhanced Chinese helper with HanLP integration."""

    def __init__(self) -> None:
        self._jieba_available = JIEBA_AVAILABLE
        self._spacy_nlp = None

        # Initialize HanLP
        try:
            import hanlp
            self._hanlp = hanlp.load(
                hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_BASE_ZH
            )
            self._hanlp_available = True
            logger.info("HanLP initialized successfully")
        except Exception as e:
            logger.warning(f"HanLP not available: {e}")
            self._hanlp_available = False

        # Measure word dictionary (partial list)
        self._measure_words = {
            "个": ["person", "thing", "general"],  # General classifier
            "本": ["book", "notebook"],            # Books
            "只": ["animal", "one of pair"],       # Animals, one of pair
            "条": ["long thin object", "fish"],    # Rivers, fish
            "张": ["flat object", "paper"],        # Paper, table
            "辆": ["vehicle"],                     # Cars, bikes
            "位": ["person (polite)"],             # People (formal)
            "件": ["piece", "clothing"],           # Clothes, matters
            "杯": ["cup"],                         # Cups
            "瓶": ["bottle"],                      # Bottles
        }

    def check_grammar(self, text: str) -> list[ErrorAnnotation]:
        """Check Chinese grammar including measure words and particles."""
        if not self._hanlp_available:
            return []

        errors = []

        # Check measure words
        errors.extend(self._check_measure_words(text))

        # Check particles
        errors.extend(self._check_particles(text))

        return errors

    def _check_measure_words(self, text: str) -> list[ErrorAnnotation]:
        """Check measure word usage using HanLP POS tagging."""
        if not self._hanlp_available:
            return []

        try:
            doc = self._hanlp(text)
            tokens = doc['tok/fine']
            pos_tags = doc['pos/ctb']

            errors = []

            for i in range(len(tokens) - 1):
                # Pattern: CD (cardinal number) followed by NN (noun)
                if pos_tags[i] == 'CD':  # Cardinal number
                    # Check if next token is a noun without measure word
                    if pos_tags[i + 1].startswith('NN'):
                        # Calculate character position
                        char_pos = sum(len(tokens[j]) for j in range(i))

                        errors.append(ErrorAnnotation(
                            category="fluency",
                            subcategory="chinese_missing_measure_word",
                            severity=ErrorSeverity.MAJOR,
                            location=(char_pos, char_pos + len(tokens[i]) + len(tokens[i+1])),
                            description=(
                                f"Missing measure word between number '{tokens[i]}' "
                                f"and noun '{tokens[i+1]}'"
                            ),
                            suggestion=f"Insert appropriate measure word (个/本/只/条/etc.)"
                        ))

                    # Pattern: CD + M (measure word) + NN
                    elif i + 2 < len(pos_tags):
                        if pos_tags[i + 1] == 'M' and pos_tags[i + 2].startswith('NN'):
                            # Measure word exists - could validate correctness
                            measure_word = tokens[i + 1]
                            noun = tokens[i + 2]

                            # Basic validation (would need comprehensive dictionary)
                            if measure_word in self._measure_words:
                                logger.debug(
                                    f"Measure word '{measure_word}' used with '{noun}'"
                                )

            return errors

        except Exception as e:
            logger.error(f"Measure word check failed: {e}")
            return []

    def _check_particles(self, text: str) -> list[ErrorAnnotation]:
        """Check Chinese particle usage (了/过/的/地/得/吗/呢)."""
        if not self._hanlp_available:
            return []

        errors = []

        # Check aspect particles (了/过)
        errors.extend(self._check_aspect_particles(text))

        # Check structural particles (的/地/得)
        errors.extend(self._check_structural_particles(text))

        return errors

    def _check_aspect_particles(self, text: str) -> list[ErrorAnnotation]:
        """Check aspect particles 了 (le) and 过 (guo)."""
        try:
            doc = self._hanlp(text)
            tokens = doc['tok/fine']
            pos_tags = doc['pos/ctb']

            errors = []

            # Past time indicators
            past_indicators = ["昨天", "以前", "已经", "刚才", "刚"]

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
                                severity=ErrorSeverity.MINOR,
                                location=(char_pos, char_pos + len(tokens[i])),
                                description=(
                                    "Past time reference without aspect marker. "
                                    "Consider adding 了 (completed action) or 过 (experience)"
                                ),
                                suggestion="Add 了 or 过 after verb"
                            ))
                            break

            return errors

        except Exception as e:
            logger.error(f"Aspect particle check failed: {e}")
            return []

    def _check_structural_particles(self, text: str) -> list[ErrorAnnotation]:
        """Check structural particles 的/地/得 usage."""
        # Note: This is complex - 的/地/得 have different uses:
        # 的 (de) - modifies nouns (adjective + 的 + noun)
        # 地 (de) - modifies verbs (adverb + 地 + verb)
        # 得 (de) - complements (verb + 得 + complement)

        # For now, just detect presence - full validation needs deep analysis
        try:
            doc = self._hanlp(text)
            tokens = doc['tok/fine']
            pos_tags = doc['pos/ctb']

            errors = []

            for i, token in enumerate(tokens):
                if token in ['的', '地', '得']:
                    # Could add validation rules based on adjacent POS tags
                    logger.debug(f"Structural particle '{token}' at position {i}")

            return errors

        except Exception as e:
            logger.error(f"Structural particle check failed: {e}")
            return []
```

### Enrichment Data for LLM Prompts

```python
def get_enrichment_data(self, text: str) -> dict[str, Any]:
    """Get comprehensive linguistic data for enriching LLM prompts."""
    if not self._hanlp_available:
        return {"has_morphology": False}

    try:
        doc = self._hanlp(text)

        # Extract measure words
        measure_words = []
        tokens = doc['tok/fine']
        pos_tags = doc['pos/ctb']

        for i, tag in enumerate(pos_tags):
            if tag == 'M':  # Measure word
                measure_words.append({
                    "word": tokens[i],
                    "position": i,
                    "type": self._measure_words.get(tokens[i], ["unknown"])[0]
                })

        # Extract particles
        particles = []
        for i, token in enumerate(tokens):
            if token in ['了', '过', '的', '地', '得', '吗', '呢', '吧', '啊']:
                particles.append({
                    "particle": token,
                    "position": i,
                    "category": self._get_particle_category(token)
                })

        # Named entities
        entities = []
        for ent in doc.get('ner/msra', []):
            entities.append({
                "text": ent[0],
                "label": ent[1],
                "start": ent[2],
                "end": ent[3]
            })

        return {
            "has_morphology": True,
            "word_count": len([t for t in tokens if t.strip()]),
            "measure_words": measure_words,
            "particles": particles,
            "pos_distribution": dict(zip(*np.unique(pos_tags, return_counts=True))),
            "entities": entities,
            "dependency_info": doc.get('dep', None)
        }

    except Exception as e:
        logger.error(f"Enrichment data extraction failed: {e}")
        return {"has_morphology": False}

def _get_particle_category(self, particle: str) -> str:
    """Categorize Chinese particle."""
    categories = {
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
    return categories.get(particle, "unknown")
```

---

## Part 4: Recent Research (arXiv 2024-2025)

### Translation Quality Evaluation Trends

#### 1. Adequacy-Fluency Tradeoffs (arXiv:2509.20287, 2025)

**Key findings:**
- Current metrics (COMET, BLEURT) favor **adequacy** over **fluency**
- WMT meta-evaluation systematically advantages adequacy-focused metrics
- Proposed solution: Synthetic system composition for balanced evaluation

**Implications:**
- Separate fluency evaluation is necessary
- Our approach (dedicated FluencyAgent) aligns with best practices
- FluencyX metric recommended for fluency-specific evaluation

#### 2. Machine Translation Quality Estimation Survey (arXiv:2403.14118, 2024)

**Evolution of approaches:**
1. **Handcrafted features** (traditional)
2. **Deep learning** (classic neural + pre-trained LMs)
3. **Large Language Models** (current state-of-the-art)

**Key characteristic:**
- QE without reference translations (estimate quality in real-time)
- Word-level, sentence-level, document-level evaluation
- Explainable QE for interpretability

**Implications:**
- Our hybrid NLP+LLM approach is state-of-the-art
- Reference-free evaluation is the future

#### 3. Multi-Agent MT Systems (2025)

**Architecture:**
- Decompose translation into specialized roles (translation → adequacy → fluency)
- Dynamic interaction between AI agents
- Mirrors professional human workflows

**Fluency Agent configuration:**
- Temperature: 0.5 (deterministic validation + flexibility)
- Separate from adequacy evaluation

**Implications:**
- Our RussianFluencyAgent architecture is aligned with 2025 research
- Consider lowering temperature to 0.5 (currently 0.1)

#### 4. GECToR-2024 (Grammatical Error Correction)

**Performance:**
- F0.5 = 66.5 on CoNLL-2014 benchmark
- 10x faster inference than seq2seq models
- "Tag, not rewrite" approach

**Approach:**
- Transformer encoder + two linear layers (detection + correction)

**Status:**
- Available but not recommended for KTTC (overlaps with LLM capabilities)

### Chinese NLP Research

**Finding:** Limited research on Chinese grammar checking compared to English/Russian

**Available tools:**
- NTOU Chinese Grammar Checker (CGED Shared Task) - academic
- HanLP - production-ready
- No specialized particle/measure word checker found

**Implication:** Custom rule development needed for Chinese

---

## Part 5: Comparison Matrix

| Feature | Russian | English (Current) | English (Proposed) | Chinese (Current) | Chinese (Proposed) |
|---------|---------|-------------------|-------------------|-------------------|-------------------|
| **Morphology** | mawo-pymorphy3 ✅ | spaCy ✅ | spaCy ✅ | spaCy/jieba ⚠️ | HanLP ✅ |
| **Grammar checks** | Case, aspect, agreement ✅ | None ❌ | LanguageTool 5,000+ rules ✅ | None ❌ | Measure words, particles ⚠️ |
| **NER** | Yes ✅ | Yes ✅ | Yes ✅ | Yes ✅ | Yes ✅ |
| **Entity preservation** | Yes ✅ | Yes ✅ | Yes ✅ | Yes ✅ | Yes ✅ |
| **Anti-hallucination** | Yes ✅ | Yes ✅ | Yes ✅ | Yes ✅ | Yes ✅ |
| **Enrichment data** | Verb aspects, adj-noun ✅ | Basic ⚠️ | Tenses, articles ✅ | Basic ⚠️ | Measure words, particles ✅ |
| **Library size** | ~15 MB ✅ | ~50 MB ✅ | ~250 MB ⚠️ | ~7-46 MB ✅ | ~100 MB ⚠️ |
| **Offline mode** | Yes ✅ | Yes ✅ | Yes ✅ | Yes ✅ | Yes ✅ |
| **Maturity** | Production ✅ | Production ✅ | Production ✅ | Production ✅ | Production ✅ |

**Legend:**
- ✅ = Fully supported / Excellent
- ⚠️ = Partially supported / Needs work
- ❌ = Not available / Poor

---

## Part 6: Decision Framework

### Should We Create Specialized Fluency Agents?

#### Decision Criteria

**CREATE EnglishFluencyAgent IF:**
- ✅ LanguageTool integration proves valuable (provides deterministic checks)
- ✅ English-specific enrichment data is needed for LLM prompts
- ✅ Performance testing shows improvement over base FluencyAgent

**CREATE ChineseFluencyAgent IF:**
- ⚠️ HanLP measure word validation is effective
- ⚠️ Particle checking provides real value (not just warnings)
- ⚠️ Linguistic expertise available for rule development

**DON'T CREATE agents IF:**
- ❌ Helpers remain too simple (no deterministic checks added)
- ❌ LLM already catches most errors (no added value)
- ❌ False positive rate is too high

### Recommended Approach: Phased Implementation

**Phase 1: Enhance Helpers** (Priority: HIGH)
1. Integrate LanguageTool into EnglishLanguageHelper
2. Implement `check_grammar()` with error filtering
3. Add enrichment data for LLM prompts
4. Test with real translations

**Phase 2: Validate English Helper** (Priority: HIGH)
1. Run benchmark tests on English translation samples
2. Measure precision/recall of grammar checks
3. Compare with base FluencyAgent (LLM-only)
4. Decide on EnglishFluencyAgent creation

**Phase 3: Enhance Chinese Helper** (Priority: MEDIUM)
1. Integrate HanLP for POS tagging
2. Implement measure word validation
3. Implement basic particle checking
4. Test with Chinese translations

**Phase 4: Validate Chinese Helper** (Priority: MEDIUM)
1. Run benchmark tests on Chinese translation samples
2. Measure effectiveness of measure word checks
3. Assess particle checking (may need to be optional/warnings)
4. Decide on ChineseFluencyAgent creation

**Phase 5: Create Agents** (Priority: LOW - after validation)
1. Only if Phases 1-4 show clear value
2. Follow RussianFluencyAgent architecture
3. Implement hybrid NLP+LLM approach
4. Add language-specific enrichment

---

## Part 7: Risk Assessment

### Low Risk (Recommended for immediate implementation)

**LanguageTool for English:**
- ✅ Mature, stable, widely used
- ✅ Open-source with active maintenance
- ✅ Well-documented API
- ✅ Works offline
- ⚠️ Large download size (~200 MB)

**Mitigation:**
- Make it optional dependency
- Graceful fallback to spaCy-only mode

### Medium Risk (Needs careful implementation)

**HanLP for Chinese:**
- ⚠️ Model download size (~100 MB)
- ⚠️ Requires PyTorch/TensorFlow
- ⚠️ Measure word rules need linguistic expertise
- ⚠️ Particle checking may have high false-positive rate

**Mitigation:**
- Keep jieba as lightweight fallback
- Start with conservative rules
- Mark particle checks as "suggestions" not "errors"
- Use LLM for final verification (hybrid approach)

### High Risk (Defer to later phase)

**Creating specialized agents prematurely:**
- ❌ Wasted effort if helpers don't add value
- ❌ Maintenance burden (3 specialized agents vs 1 base)
- ❌ Code duplication

**Mitigation:**
- Wait for Phase 2/4 validation results
- Make data-driven decision
- Consider helper-only approach if sufficient

---

## Conclusion

### Key Takeaways

1. **RussianFluencyAgent is already optimal:**
   - Only activates for Russian (`target_lang == "ru"`)
   - Hybrid NLP+LLM approach is 2025 best practice
   - Anti-hallucination verification is critical

2. **English has mature solutions:**
   - LanguageTool provides 5,000+ rules immediately
   - spaCy enables custom advanced rules
   - High confidence in implementation

3. **Chinese requires more work:**
   - HanLP provides foundation (POS, parsing)
   - Measure word validation feasible
   - Particle checking needs linguistic expertise

4. **Don't create agents yet:**
   - Enhance helpers first
   - Validate with real data
   - Make evidence-based decision

### Recommended Next Actions

1. **Immediate (this week):**
   - Integrate LanguageTool into EnglishLanguageHelper
   - Implement basic `check_grammar()` method
   - Test with 10-20 English translations

2. **Short-term (next 2 weeks):**
   - Refine LanguageTool integration (filtering, severity mapping)
   - Add enrichment data for English
   - Benchmark against base FluencyAgent

3. **Medium-term (next month):**
   - Integrate HanLP into ChineseLanguageHelper
   - Implement measure word validation
   - Test with Chinese translations

4. **Long-term (decide after testing):**
   - Create EnglishFluencyAgent (if data supports it)
   - Create ChineseFluencyAgent (if data supports it)
   - Maintain helpers-only approach (if sufficient)

---

**Research compiled by:** AI Assistant for KTTC AI
**Date:** November 13, 2025
**Sources:** 40+ web pages, 5+ arXiv papers, PyPI, GitHub, academic publications
**Version:** 1.0
