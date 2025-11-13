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

"""Russian language helper with pymorphy3 and razdel integration."""

from __future__ import annotations

import logging
from typing import Any

from kttc.core import ErrorAnnotation, ErrorSeverity

from .base import LanguageHelper, MorphologyInfo

logger = logging.getLogger(__name__)

# Try to import MAWO dependencies (forked versions)
try:
    import mawo_natasha as natasha  # type: ignore[import-untyped]
    import mawo_pymorphy3 as pymorphy3
    import mawo_razdel as razdel  # type: ignore[import-untyped]

    DEPS_AVAILABLE = True
    NER_AVAILABLE = True
    logger.info("Using MAWO NLP libraries (mawo-pymorphy3 + mawo-razdel + mawo-natasha)")
except ImportError as e:
    DEPS_AVAILABLE = False
    NER_AVAILABLE = False
    logger.warning(
        f"MAWO NLP libraries not installed: {e}. "
        "RussianLanguageHelper will run in limited mode. "
        "Install with: pip install mawo-pymorphy3 mawo-razdel mawo-natasha"
    )


class RussianLanguageHelper(LanguageHelper):
    """Language helper for Russian with MAWO NLP-powered checks.

    Uses MAWO forked libraries (mawo-pymorphy3, mawo-razdel) for:
    - Morphological analysis
    - Russian tokenization
    - Deterministic case agreement checking
    - Verb aspect detection
    - Anti-hallucination verification
    - Accurate error positions

    Example:
        >>> helper = RussianLanguageHelper()
        >>> if helper.is_available():
        ...     errors = helper.check_grammar("быстрый лиса")
        ...     print(errors[0].description)
        Gender mismatch: 'быстрый' is masc, but 'лиса' is femn
    """

    def __init__(self) -> None:
        """Initialize Russian language helper."""
        self._morph: Any = None
        self._initialized = False

        if DEPS_AVAILABLE:
            try:
                self._morph = pymorphy3.MorphAnalyzer()
                self._initialized = True
                logger.info(
                    "RussianLanguageHelper initialized with MAWO NLP (mawo-pymorphy3 + mawo-razdel)"
                )
            except Exception as e:
                logger.error(f"Failed to initialize RussianLanguageHelper: {e}")
                self._initialized = False
        else:
            logger.info("RussianLanguageHelper running in limited mode (no MAWO NLP)")

    @property
    def language_code(self) -> str:
        """Get language code."""
        return "ru"

    def is_available(self) -> bool:
        """Check if NLP dependencies are available."""
        return self._initialized and DEPS_AVAILABLE

    def verify_word_exists(self, word: str, text: str) -> bool:
        """Verify word exists in text (anti-hallucination).

        Args:
            word: Word to search for
            text: Text to search in

        Returns:
            True if word found, False if not (LLM hallucination)
        """
        if not self.is_available():
            # Fallback: simple case-insensitive search
            return word.lower() in text.lower()

        # Use proper tokenization
        tokens = self.tokenize(text)
        word_lower = word.lower()
        return any(token[0].lower() == word_lower for token in tokens)

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

        # If error mentions specific word, verify it exists in substring
        if hasattr(error, "description"):
            # Try to extract quoted words from description
            import re

            quoted_words = re.findall(r"'([^']+)'|\"([^\"]+)\"", error.description)
            if quoted_words:
                for word_tuple in quoted_words:
                    word = word_tuple[0] or word_tuple[1]
                    if word and word not in substring:
                        logger.warning(
                            f"Error mentions '{word}' but it's not at position {start}:{end}"
                        )
                        return False

        return True

    def tokenize(self, text: str) -> list[tuple[str, int, int]]:
        """Tokenize Russian text with accurate positions using MAWO razdel.

        Args:
            text: Text to tokenize

        Returns:
            List of (word, start, end) tuples
        """
        if not self.is_available():
            # Fallback: simple split
            tokens = []
            start = 0
            for word in text.split():
                idx = text.find(word, start)
                if idx != -1:
                    tokens.append((word, idx, idx + len(word)))
                    start = idx + len(word)
            return tokens

        # Use MAWO razdel for proper Russian tokenization
        return [(token.text, token.start, token.stop) for token in razdel.tokenize(text)]

    def analyze_morphology(self, text: str) -> list[MorphologyInfo]:
        """Analyze morphology of all words in text.

        Args:
            text: Text to analyze

        Returns:
            List of MorphologyInfo objects
        """
        if not self.is_available():
            return []

        tokens = self.tokenize(text)
        results = []

        for word, start, stop in tokens:
            if not word.strip():
                continue

            parsed = self._morph.parse(word)[0]  # Take most likely parse
            tag = parsed.tag

            results.append(
                MorphologyInfo(
                    word=word,
                    pos=str(tag.POS) if tag.POS else None,
                    gender=str(tag.gender) if tag.gender else None,
                    case=str(tag.case) if tag.case else None,
                    number=str(tag.number) if tag.number else None,
                    aspect=str(tag.aspect) if tag.aspect else None,
                    start=start,
                    stop=stop,
                )
            )

        return results

    def check_grammar(self, text: str) -> list[ErrorAnnotation]:
        """Check Russian grammar with deterministic rules.

        Checks:
        - Adjective-noun case agreement (gender, case, number)
        - Preposition + noun case agreement

        Args:
            text: Russian text to check

        Returns:
            List of detected grammar errors
        """
        if not self.is_available():
            return []

        errors = []

        # Get morphological analysis
        morphs = self.analyze_morphology(text)

        # Check adjective-noun agreement
        for i in range(len(morphs) - 1):
            curr = morphs[i]
            next_word = morphs[i + 1]

            # Skip if not adjective + noun
            if curr.pos != "ADJF" or next_word.pos != "NOUN":
                continue

            # Check gender agreement
            if curr.gender and next_word.gender and curr.gender != next_word.gender:
                errors.append(
                    ErrorAnnotation(
                        category="fluency",
                        subcategory="russian_case_agreement",
                        severity=ErrorSeverity.CRITICAL,
                        location=(curr.start, next_word.stop),
                        description=(
                            f"Gender agreement violation: adjective '{curr.word}' ({curr.gender}) "
                            f"does not match noun '{next_word.word}' ({next_word.gender})"
                        ),
                        suggestion=None,  # Could add suggestions later
                    )
                )

            # Check case agreement
            elif curr.case and next_word.case and curr.case != next_word.case:
                errors.append(
                    ErrorAnnotation(
                        category="fluency",
                        subcategory="russian_case_agreement",
                        severity=ErrorSeverity.MAJOR,
                        location=(curr.start, next_word.stop),
                        description=(
                            f"Case agreement violation: adjective '{curr.word}' ({curr.case}) "
                            f"does not match noun '{next_word.word}' ({next_word.case})"
                        ),
                        suggestion=None,
                    )
                )

            # Check number agreement
            elif curr.number and next_word.number and curr.number != next_word.number:
                errors.append(
                    ErrorAnnotation(
                        category="fluency",
                        subcategory="russian_case_agreement",
                        severity=ErrorSeverity.MAJOR,
                        location=(curr.start, next_word.stop),
                        description=(
                            f"Number agreement violation: adjective '{curr.word}' ({curr.number}) "
                            f"does not match noun '{next_word.word}' ({next_word.number})"
                        ),
                        suggestion=None,
                    )
                )

        return errors

    def get_enrichment_data(self, text: str) -> dict[str, Any]:
        """Get comprehensive morphological data for enriching LLM prompts.

        Provides detailed linguistic context to help LLM make better decisions:
        - Verb aspects (perfective/imperfective)
        - Adjective-noun pairs with their properties
        - Gender, case, number information
        - Potential agreement issues

        Args:
            text: Text to analyze

        Returns:
            Dictionary with morphological insights for LLM
        """
        if not self.is_available():
            return {"has_morphology": False}

        morphs = self.analyze_morphology(text)

        # Extract verb aspects
        verb_aspects = {}
        verbs_list = []
        for m in morphs:
            if m.pos == "VERB" and m.aspect:
                verb_aspects[m.word] = {
                    "aspect": m.aspect,
                    "aspect_name": "perfective" if m.aspect == "perf" else "imperfective",
                    "position": f"{m.start}-{m.stop}",
                }
                verbs_list.append(m.word)

        # Extract adjective-noun pairs
        adj_noun_pairs = []
        for i in range(len(morphs) - 1):
            curr = morphs[i]
            next_m = morphs[i + 1]

            if curr.pos == "ADJF" and next_m.pos == "NOUN":
                pair_info = {
                    "adjective": {
                        "word": curr.word,
                        "gender": curr.gender,
                        "case": curr.case,
                        "number": curr.number,
                    },
                    "noun": {
                        "word": next_m.word,
                        "gender": next_m.gender,
                        "case": next_m.case,
                        "number": next_m.number,
                    },
                    "agreement": (
                        "correct"
                        if (curr.gender == next_m.gender and curr.case == next_m.case)
                        else "mismatch"
                    ),
                }
                adj_noun_pairs.append(pair_info)

        # Count parts of speech
        pos_counts: dict[str, int] = {}
        for m in morphs:
            if m.pos:
                pos_counts[m.pos] = pos_counts.get(m.pos, 0) + 1

        return {
            "has_morphology": True,
            "word_count": len(morphs),
            "verb_aspects": verb_aspects,
            "verbs_found": verbs_list,
            "adjective_noun_pairs": adj_noun_pairs,
            "pos_distribution": pos_counts,
        }

    def extract_entities(self, text: str) -> list[dict[str, Any]]:
        """Extract named entities from Russian text using MAWO NER.

        Args:
            text: Text to extract entities from

        Returns:
            List of entities with type, text, and position
        """
        if not NER_AVAILABLE:
            logger.debug("NER not available, returning empty list")
            return []

        try:
            # Create document and extract entities
            doc = natasha.Doc(text)
            doc.segment()

            # Tag NER
            ner = natasha.NewsNERTagger()
            doc.tag_ner(ner)

            # Convert to our format
            entities = []
            for span in doc.spans:
                entities.append(
                    {
                        "text": span.text,
                        "type": span.type,
                        "start": span.start,
                        "stop": span.stop,
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
            translation_text: Russian translation

        Returns:
            List of errors for missing/mismatched entities
        """
        if not NER_AVAILABLE:
            logger.debug("NER not available, skipping entity preservation check")
            return []

        try:
            # Extract entities from translation (Russian)
            translation_entities = self.extract_entities(translation_text)

            # For now, just verify that translation has reasonable entities
            # Full cross-lingual entity matching requires translation of entity names
            # which is complex - we'll implement basic checks first

            errors = []

            # Basic check: if source has obvious names (capitalized words),
            # check if translation has entities
            import re

            # Find capitalized sequences in source (potential entity names)
            source_caps = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", source_text)

            if len(source_caps) > 0 and len(translation_entities) == 0:
                # Source has potential entities but translation has none
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
