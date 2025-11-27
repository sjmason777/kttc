# Copyright 2025 KTTC AI (https://github.com/kttc-ai)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Chinese language helper with HanLP + jieba + spaCy integration."""

from __future__ import annotations

import logging
from typing import Any

from kttc.core import ErrorAnnotation

from .base import LanguageHelper, MorphologyInfo

logger = logging.getLogger(__name__)

# Try to import HanLP (optional, for advanced Chinese grammar checking)
try:
    import hanlp

    HANLP_AVAILABLE = True
    logger.info("HanLP available for Chinese grammar checking")
except ImportError:
    hanlp = None  # Allow patching in tests
    HANLP_AVAILABLE = False
    logger.warning(
        "HanLP not installed. "
        "ChineseLanguageHelper will run without advanced grammar checking. "
        "Install with: pip install hanlp"
    )

# Try to import jieba
try:
    import jieba

    JIEBA_AVAILABLE = True
    logger.info("Using jieba for Chinese word segmentation")
except ImportError:
    jieba = None  # Allow patching in tests
    JIEBA_AVAILABLE = False
    logger.warning(
        "jieba not installed. "
        "ChineseLanguageHelper will run in limited mode. "
        "Install with: pip install jieba"
    )

# Try to import spaCy
try:
    import spacy

    SPACY_AVAILABLE = True
    logger.info("Using spaCy for Chinese NLP")
except ImportError:
    spacy = None  # type: ignore[assignment]  # Allow patching in tests
    SPACY_AVAILABLE = False
    logger.warning(
        "spaCy not installed. "
        "ChineseLanguageHelper will run in limited mode. "
        "Install with: pip install spacy && python -m spacy download zh_core_web_sm"
    )


class ChineseLanguageHelper(LanguageHelper):
    """Language helper for Chinese with HanLP + jieba + spaCy.

    Uses:
    - HanLP SMALL: Advanced grammar checking with CTB POS tags (300 MB, optional)
      - Measure word validation (CD + M + NN patterns)
      - Aspect particle checking (了/过)
      - High-accuracy POS tagging (~92-95%)
    - jieba: Fast and lightweight word segmentation (7 MB)
    - spaCy: POS tagging, NER, morphological analysis (46 MB)

    Example:
        >>> helper = ChineseLanguageHelper()
        >>> if helper.is_available():
        ...     tokens = helper.tokenize("我爱中文")
        ...     print([t[0] for t in tokens])
        ['我', '爱', '中文']
        ...     errors = helper.check_grammar("三个书")  # Wrong measure word
        ...     print(errors[0].description)
        'Incorrect measure word: "个" should be "本" for books'
    """

    # Common Chinese measure words (量词) by category
    MEASURE_WORDS = {
        "个": ["人", "学生", "老师", "朋友", "问题", "办法", "月", "星期"],  # General
        "本": ["书", "杂志", "词典"],  # Books
        "只": ["猫", "狗", "鸟", "手", "眼睛"],  # Animals, body parts
        "条": ["鱼", "河", "路", "裤子", "消息"],  # Long/thin objects
        "张": ["纸", "桌子", "床", "票", "照片"],  # Flat objects
        "辆": ["车", "汽车", "自行车"],  # Vehicles
        "位": ["老师", "先生", "女士", "客人"],  # People (polite)
        "件": ["衣服", "事情", "礼物"],  # Clothing, matters
        "杯": ["水", "茶", "咖啡", "酒"],  # Beverages
        "瓶": ["水", "酒", "啤酒"],  # Bottled items
        "支": ["笔", "烟"],  # Stick-like objects
        "双": ["鞋", "筷子", "手套"],  # Pairs
        "把": ["椅子", "刀", "伞", "钥匙"],  # Objects with handles
        "颗": ["星星", "牙齿", "心"],  # Small round objects
        "朵": ["花", "云"],  # Flowers, clouds
    }

    def __init__(self) -> None:
        """Initialize Chinese language helper with HanLP + jieba + spaCy."""
        self._nlp: Any = None
        self._hanlp: Any = None
        self._initialized = False
        self._hanlp_available = False

        # Initialize HanLP (optional, for advanced grammar checking)
        if HANLP_AVAILABLE:
            try:
                # Load HanLP SMALL model (~300 MB)
                self._hanlp = hanlp.load(
                    hanlp.pretrained.mtl.OPEN_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_SMALL_ZH
                )
                self._hanlp_available = True
                logger.info(
                    "ChineseLanguageHelper initialized with HanLP SMALL (300 MB, CTB POS tags)"
                )
            except Exception as e:
                logger.warning(f"HanLP initialization failed: {e}")
                self._hanlp_available = False

        # Initialize spaCy
        if SPACY_AVAILABLE:
            try:
                # Try medium model first (better accuracy with word vectors)
                self._nlp = spacy.load("zh_core_web_md")
                logger.info("ChineseLanguageHelper initialized with spaCy zh_core_web_md (74 MB)")
                self._initialized = True
            except OSError:
                try:
                    # Fallback to small model
                    self._nlp = spacy.load("zh_core_web_sm")
                    logger.info(
                        "ChineseLanguageHelper initialized with spaCy zh_core_web_sm (46 MB)"
                    )
                    self._initialized = True
                except OSError:
                    logger.warning(
                        "spaCy Chinese model not found. "
                        "Download with: python -m spacy download zh_core_web_md"
                    )
                    if JIEBA_AVAILABLE:
                        logger.info("Falling back to jieba-only mode")
                        self._initialized = True
        elif JIEBA_AVAILABLE:
            logger.info("ChineseLanguageHelper running in jieba-only mode")
            self._initialized = True
        else:
            logger.info("ChineseLanguageHelper running in limited mode (no NLP)")

    @property
    def language_code(self) -> str:
        """Get language code."""
        return "zh"

    def is_available(self) -> bool:
        """Check if NLP dependencies are available."""
        return self._initialized and (JIEBA_AVAILABLE or SPACY_AVAILABLE)

    def verify_word_exists(self, word: str, text: str) -> bool:
        """Verify word exists in text (anti-hallucination).

        Args:
            word: Word to search for
            text: Text to search in

        Returns:
            True if word found, False if not (LLM hallucination)
        """
        if not self.is_available():
            # Fallback: simple search
            return word in text

        # Use tokenization
        tokens = self.tokenize(text)
        return any(token[0] == word for token in tokens)

    def verify_error_position(self, error: ErrorAnnotation, text: str) -> bool:
        """Verify error position is valid.

        Args:
            error: Error with location field
            text: Full text

        Returns:
            True if position valid, False otherwise
        """
        start, end = error.location

        # Check bounds
        if start < 0 or end > len(text) or start >= end:
            return False

        # Extract text at position
        substring = text[start:end]

        # Check if it's not empty
        if not substring.strip():
            return False

        return True

    def tokenize(self, text: str) -> list[tuple[str, int, int]]:
        """Tokenize Chinese text with accurate positions.

        Uses jieba if available (fast), otherwise falls back to spaCy.

        Args:
            text: Text to tokenize

        Returns:
            List of (word, start, end) tuples
        """
        if not self.is_available():
            # Fallback: character-level tokenization
            return [(char, i, i + 1) for i, char in enumerate(text) if char.strip()]

        # Prefer spaCy if available (provides POS tagging)
        if SPACY_AVAILABLE and self._nlp:
            doc = self._nlp(text)
            return [(token.text, token.idx, token.idx + len(token.text)) for token in doc]

        # Use jieba for word segmentation
        if JIEBA_AVAILABLE:
            tokens = []
            start = 0
            for word in jieba.cut(text):
                idx = text.find(word, start)
                if idx != -1:
                    tokens.append((word, idx, idx + len(word)))
                    start = idx + len(word)
            return tokens

        return []

    def analyze_morphology(self, text: str) -> list[MorphologyInfo]:
        """Analyze morphology of all words in text.

        Args:
            text: Text to analyze

        Returns:
            List of MorphologyInfo objects
        """
        if not self.is_available() or not SPACY_AVAILABLE or not self._nlp:
            return []

        doc = self._nlp(text)
        results = []

        for token in doc:
            results.append(
                MorphologyInfo(
                    word=token.text,
                    pos=token.pos_,
                    gender=None,  # Chinese doesn't have grammatical gender
                    case=None,  # Chinese doesn't have case system
                    number=None,  # Chinese number is context-dependent
                    aspect=None,  # Chinese aspect is complex
                    start=token.idx,
                    stop=token.idx + len(token.text),
                )
            )

        return results

    def check_grammar(self, text: str) -> list[ErrorAnnotation]:
        """Check Chinese grammar using HanLP.

        Checks include:
        - Measure word validation (量词检查)
        - Aspect particle usage (了/过检查)

        Args:
            text: Chinese text to check

        Returns:
            List of detected grammar errors
        """
        if not self.is_available():
            return []

        errors: list[ErrorAnnotation] = []

        # Advanced grammar checks with HanLP
        if self._hanlp_available and self._hanlp:
            errors.extend(self._check_measure_words(text))
            errors.extend(self._check_particles(text))

        return errors

    @staticmethod
    def _find_measure_position(
        text: str, number: str, measure: str, noun: str
    ) -> tuple[int, int] | None:
        """Find the position of a measure word in text.

        Args:
            text: Original text
            number: Number token
            measure: Measure word token
            noun: Noun token

        Returns:
            Tuple of (start, end) positions or None if not found
        """
        start_pos = text.find(f"{number}{measure}{noun}")
        if start_pos == -1:
            start_pos = text.find(f"{measure}{noun}")
            if start_pos == -1:
                return None
            # When found without number, measure_start is at start_pos
            return (start_pos, start_pos + len(measure))

        measure_start = start_pos + len(number)
        return (measure_start, measure_start + len(measure))

    @staticmethod
    def _create_measure_word_error(
        measure: str, noun: str, location: tuple[int, int], suggested: list[str]
    ) -> ErrorAnnotation:
        """Create an error annotation for incorrect measure word.

        Args:
            measure: The incorrect measure word
            noun: The noun it modifies
            location: Position tuple (start, end)
            suggested: List of suggested measure words

        Returns:
            ErrorAnnotation for the measure word error
        """
        from kttc.core import ErrorSeverity

        return ErrorAnnotation(
            category="fluency",
            subcategory="measure_word",
            severity=ErrorSeverity.MINOR,
            location=location,
            description=(
                f'Incorrect measure word: "{measure}" may not be appropriate '
                f'for "{noun}". Consider using: {", ".join(suggested)}'
            ),
            suggestion=suggested[0],
        )

    def _check_measure_words(self, text: str) -> list[ErrorAnnotation]:
        """Check measure word usage (量词检查).

        Validates CD (number) + M (measure word) + NN (noun) patterns.

        Common mistakes:
        - 三个书 → 三本书 (books need "本")
        - 一本车 → 一辆车 (vehicles need "辆")
        - 两条狗 → 两只狗 (animals need "只")

        Args:
            text: Chinese text to check

        Returns:
            List of errors for incorrect measure words
        """
        if not self._hanlp_available or not self._hanlp:
            return []

        try:
            result = self._hanlp(text)
            tokens = result["tok"]
            pos_tags = result["pos"]
            errors = []

            for i in range(len(pos_tags) - 2):
                if not (pos_tags[i] == "CD" and pos_tags[i + 1] == "M" and pos_tags[i + 2] == "NN"):
                    continue

                number, measure, noun = tokens[i], tokens[i + 1], tokens[i + 2]
                suggested_measures = self._get_appropriate_measures(noun)

                if not suggested_measures or measure in suggested_measures:
                    continue

                location = self._find_measure_position(text, number, measure, noun)
                if location:
                    errors.append(
                        self._create_measure_word_error(measure, noun, location, suggested_measures)
                    )

            logger.debug(f"Found {len(errors)} measure word errors")
            return errors

        except Exception as e:
            logger.error(f"Measure word checking failed: {e}")
            return []

    def _get_appropriate_measures(self, noun: str) -> list[str]:
        """Get appropriate measure words for a given noun.

        Args:
            noun: Chinese noun

        Returns:
            List of appropriate measure words (empty if noun not in dictionary)
        """
        appropriate = []
        for measure, nouns in self.MEASURE_WORDS.items():
            if noun in nouns:
                appropriate.append(measure)

        # If noun not in dictionary, return empty (don't flag error)
        return appropriate

    def _check_particles(self, text: str) -> list[ErrorAnnotation]:
        """Check aspect particle usage (了/过检查).

        Validates:
        - 了 (le): Completed action / change of state
        - 过 (guo): Past experience

        Common mistakes:
        - Missing 了 after completed actions
        - Redundant 过 usage

        Args:
            text: Chinese text to check

        Returns:
            List of errors for particle usage
        """
        if not self._hanlp_available or not self._hanlp:
            return []

        try:
            # Get HanLP analysis
            result = self._hanlp(text)
            tokens = result["tok"]
            pos_tags = result["pos"]

            errors = []

            # Find AS (aspect marker) tags
            for i, (token, pos) in enumerate(zip(tokens, pos_tags)):
                if pos == "AS" and token in ["了", "过"]:
                    # Check if previous token is a verb
                    if i > 0 and pos_tags[i - 1] not in ["VV", "VA", "VC", "VE"]:
                        from kttc.core import ErrorSeverity

                        # Find position in text
                        start_pos = text.find(token)
                        if start_pos == -1:
                            continue

                        errors.append(
                            ErrorAnnotation(
                                category="fluency",
                                subcategory="aspect_particle",
                                severity=ErrorSeverity.MINOR,
                                location=(start_pos, start_pos + len(token)),
                                description=(
                                    f'Aspect particle "{token}" should follow a verb, '
                                    f'but follows "{tokens[i - 1]}" ({pos_tags[i - 1]})'
                                ),
                                suggestion=None,
                            )
                        )

            logger.debug(f"Found {len(errors)} particle errors")
            return errors

        except Exception as e:
            logger.error(f"Particle checking failed: {e}")
            return []

    def _find_measure_patterns(
        self, tokens: list[str], pos_tags: list[str]
    ) -> list[dict[str, str]]:
        """Find measure word patterns (CD + M + NN) in tokens."""
        patterns = []
        for i in range(len(pos_tags) - 2):
            if pos_tags[i] == "CD" and pos_tags[i + 1] == "M" and pos_tags[i + 2] == "NN":
                patterns.append(
                    {
                        "number": tokens[i],
                        "measure": tokens[i + 1],
                        "noun": tokens[i + 2],
                        "pattern": f"{tokens[i]}{tokens[i + 1]}{tokens[i + 2]}",
                    }
                )
        return patterns

    def _find_aspect_particles(
        self, tokens: list[str], pos_tags: list[str]
    ) -> list[dict[str, Any]]:
        """Find aspect particles (了, 过) in tokens."""
        particles = []
        for i, (token, pos) in enumerate(zip(tokens, pos_tags)):
            if pos == "AS" and token in ["了", "过"]:
                prev_verb = tokens[i - 1] if i > 0 else None
                particles.append({"particle": token, "verb": prev_verb, "position": i})
        return particles

    def _get_hanlp_enrichment(self, text: str) -> dict[str, Any] | None:
        """Get enrichment data using HanLP."""
        if not (self._hanlp_available and self._hanlp):
            return None

        try:
            result = self._hanlp(text)
            tokens, pos_tags = result["tok"], result["pos"]

            pos_counts: dict[str, int] = {}
            for pos in pos_tags:
                pos_counts[pos] = pos_counts.get(pos, 0) + 1

            entities = [
                {"text": e_text, "type": e_type, "start": start, "end": end}
                for e_text, e_type, start, end in result["ner"]
            ]

            return {
                "has_morphology": True,
                "word_count": len([t for t in tokens if t.strip()]),
                "pos_distribution": pos_counts,
                "measure_patterns": self._find_measure_patterns(tokens, pos_tags),
                "aspect_particles": self._find_aspect_particles(tokens, pos_tags),
                "entities": entities,
                "has_hanlp": True,
            }
        except Exception as e:
            logger.error(f"HanLP enrichment failed: {e}")
            return None

    def _get_spacy_enrichment(self, text: str) -> dict[str, Any] | None:
        """Get enrichment data using spaCy."""
        if not (SPACY_AVAILABLE and self._nlp):
            return None

        doc = self._nlp(text)

        pos_counts: dict[str, int] = {}
        for token in doc:
            if token.pos_:
                pos_counts[token.pos_] = pos_counts.get(token.pos_, 0) + 1

        entities = [
            {"text": ent.text, "label": ent.label_, "start": ent.start_char, "end": ent.end_char}
            for ent in doc.ents
        ]

        return {
            "has_morphology": True,
            "word_count": len([token for token in doc if not token.is_punct]),
            "pos_distribution": pos_counts,
            "entities": entities,
            "sentence_count": len(list(doc.sents)),
        }

    def _get_jieba_enrichment(self, text: str) -> dict[str, Any] | None:
        """Get enrichment data using jieba."""
        if not JIEBA_AVAILABLE:
            return None

        words = list(jieba.cut(text))
        return {
            "has_morphology": True,
            "word_count": len([w for w in words if w.strip()]),
            "segmentation_method": "jieba",
        }

    def get_enrichment_data(self, text: str) -> dict[str, Any]:
        """Get comprehensive linguistic data for enriching LLM prompts.

        Provides detailed Chinese linguistic context:
        - Measure word patterns (CD + M + NN)
        - Aspect particles (了/过 usage)
        - CTB POS tag distribution
        - Named entities
        - Sentence structure

        Args:
            text: Text to analyze

        Returns:
            Dictionary with linguistic insights for LLM
        """
        if not self.is_available():
            return {"has_morphology": False}

        # Try HanLP first (most accurate)
        hanlp_result = self._get_hanlp_enrichment(text)
        if hanlp_result:
            return hanlp_result

        # Try spaCy fallback
        spacy_result = self._get_spacy_enrichment(text)
        if spacy_result:
            return spacy_result

        # Try jieba fallback
        jieba_result = self._get_jieba_enrichment(text)
        if jieba_result:
            return jieba_result

        return {"has_morphology": False}

    def extract_entities(self, text: str) -> list[dict[str, Any]]:
        """Extract named entities from Chinese text using spaCy NER.

        Args:
            text: Text to extract entities from

        Returns:
            List of entities with type, text, and position
        """
        if not SPACY_AVAILABLE or not self._nlp:
            logger.debug("spaCy not available, returning empty list")
            return []

        try:
            doc = self._nlp(text)

            # Convert to our format
            entities = []
            for ent in doc.ents:
                entities.append(
                    {
                        "text": ent.text,
                        "type": ent.label_,
                        "start": ent.start_char,
                        "stop": ent.end_char,
                    }
                )

            logger.debug(f"Extracted {len(entities)} entities from text")
            return entities

        except Exception as e:
            logger.error(f"NER extraction failed: {e}")
            return []

    def check_entity_preservation(
        self, source_text: str, translation_text: str
    ) -> list[ErrorAnnotation]:
        """Check if named entities from source are preserved in translation.

        Args:
            source_text: Original text (may be in any language)
            translation_text: Chinese translation

        Returns:
            List of errors for missing/mismatched entities
        """
        if not SPACY_AVAILABLE or not self._nlp:
            logger.debug("spaCy not available, skipping entity preservation check")
            return []

        try:
            # Extract entities from translation (Chinese)
            translation_entities = self.extract_entities(translation_text)

            errors = []

            # Basic check: if source has obvious names (capitalized words),
            # check if translation has entities
            import re

            # Find capitalized sequences in source (potential entity names)
            source_caps = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", source_text)

            if len(source_caps) > 0 and len(translation_entities) == 0:
                # Source has potential entities but translation has none
                from kttc.core import ErrorSeverity

                errors.append(
                    ErrorAnnotation(
                        category="accuracy",
                        subcategory="entity_omission",
                        severity=ErrorSeverity.MAJOR,
                        location=(0, min(50, len(translation_text))),
                        description=(
                            f"Source text contains {len(source_caps)} potential entities "
                            f"but translation has no named entities detected"
                        ),
                        suggestion="Verify that proper nouns are correctly translated",
                    )
                )

            logger.debug(
                f"Entity preservation check: "
                f"source_caps={len(source_caps)}, "
                f"translation_entities={len(translation_entities)}"
            )

            return errors

        except Exception as e:
            logger.error(f"Entity preservation check failed: {e}")
            return []
