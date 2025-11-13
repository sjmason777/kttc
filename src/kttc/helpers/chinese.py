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

"""Chinese language helper with jieba + spaCy integration."""

from __future__ import annotations

import logging
from typing import Any

from kttc.core import ErrorAnnotation

from .base import LanguageHelper, MorphologyInfo

logger = logging.getLogger(__name__)

# Try to import jieba
try:
    import jieba  # type: ignore[import-untyped]

    JIEBA_AVAILABLE = True
    logger.info("Using jieba for Chinese word segmentation")
except ImportError:
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
    SPACY_AVAILABLE = False
    logger.warning(
        "spaCy not installed. "
        "ChineseLanguageHelper will run in limited mode. "
        "Install with: pip install spacy && python -m spacy download zh_core_web_sm"
    )


class ChineseLanguageHelper(LanguageHelper):
    """Language helper for Chinese with jieba + spaCy.

    Uses:
    - jieba: Fast and lightweight word segmentation (7 MB)
    - spaCy: POS tagging, NER, morphological analysis (46 MB)

    Example:
        >>> helper = ChineseLanguageHelper()
        >>> if helper.is_available():
        ...     tokens = helper.tokenize("我爱中文")
        ...     print([t[0] for t in tokens])
        ['我', '爱', '中文']
    """

    def __init__(self) -> None:
        """Initialize Chinese language helper."""
        self._nlp: Any = None
        self._initialized = False

        # Check if we have at least jieba or spaCy
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
        """Check Chinese grammar.

        Basic grammar checks (extensible).

        Args:
            text: Chinese text to check

        Returns:
            List of detected grammar errors
        """
        if not self.is_available():
            return []

        # For now, return empty list - can extend with custom rules
        return []

    def get_enrichment_data(self, text: str) -> dict[str, Any]:
        """Get comprehensive linguistic data for enriching LLM prompts.

        Args:
            text: Text to analyze

        Returns:
            Dictionary with linguistic insights for LLM
        """
        if not self.is_available():
            return {"has_morphology": False}

        # Use spaCy if available for richer analysis
        if SPACY_AVAILABLE and self._nlp:
            doc = self._nlp(text)

            # Count parts of speech
            pos_counts: dict[str, int] = {}
            for token in doc:
                if token.pos_:
                    pos_counts[token.pos_] = pos_counts.get(token.pos_, 0) + 1

            # Extract named entities
            entities = []
            for ent in doc.ents:
                entities.append(
                    {
                        "text": ent.text,
                        "label": ent.label_,
                        "start": ent.start_char,
                        "end": ent.end_char,
                    }
                )

            # Count sentences
            sent_count = len(list(doc.sents))

            return {
                "has_morphology": True,
                "word_count": len([token for token in doc if not token.is_punct]),
                "pos_distribution": pos_counts,
                "entities": entities,
                "sentence_count": sent_count,
            }

        # Fallback to jieba-only mode
        if JIEBA_AVAILABLE:
            words = list(jieba.cut(text))
            return {
                "has_morphology": True,
                "word_count": len([w for w in words if w.strip()]),
                "segmentation_method": "jieba",
            }

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
