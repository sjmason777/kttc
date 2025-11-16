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

# Try to import MAWO dependencies (unified libraries)
try:
    from mawo import Russian  # mawo-core package
    from mawo_grammar import RussianGrammarChecker  # mawo-grammar package

    DEPS_AVAILABLE = True
    logger.info("Using MAWO unified libraries (mawo-core + mawo-grammar)")
except ImportError as e:
    DEPS_AVAILABLE = False
    logger.warning(
        f"MAWO libraries not installed: {e}. "
        "RussianLanguageHelper will run in limited mode. "
        "Install with: pip install 'mawo-core[all]>=0.1.1'"
    )


class RussianLanguageHelper(LanguageHelper):
    """Language helper for Russian with MAWO unified libraries.

    Uses MAWO unified libraries (mawo-core + mawo-grammar) for:
    - Morphological analysis with rich Document/Token objects
    - Russian tokenization and NER
    - Grammar checking with 690+ rules
    - Verb aspect detection
    - Adjective-noun agreement checking
    - Entity preservation validation
    - Anti-hallucination verification

    Example:
        >>> helper = RussianLanguageHelper()
        >>> if helper.is_available():
        ...     errors = helper.check_grammar("быстрый лиса")
        ...     print(errors[0].description)
        Gender mismatch: 'быстрый' is masc, but 'лиса' is femn
    """

    def __init__(self) -> None:
        """Initialize Russian language helper."""
        self._nlp: Any = None
        self._grammar_checker: Any = None
        self._initialized = False

        if DEPS_AVAILABLE:
            try:
                self._nlp = Russian()
                self._grammar_checker = RussianGrammarChecker()
                self._initialized = True
                logger.info(
                    "RussianLanguageHelper initialized with MAWO unified libraries "
                    "(mawo-core + mawo-grammar)"
                )
            except Exception as e:
                logger.error(f"Failed to initialize RussianLanguageHelper: {e}")
                self._initialized = False
        else:
            logger.info("RussianLanguageHelper running in limited mode (no MAWO libraries)")

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
        """Tokenize Russian text with accurate positions using MAWO core.

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

        # Use MAWO core for proper Russian tokenization
        doc = self._nlp(text)
        return [(token.text, token.start, token.end) for token in doc.tokens]

    def analyze_morphology(self, text: str) -> list[MorphologyInfo]:
        """Analyze morphology of all words using MAWO core.

        MAWO core provides automatic context-aware disambiguation with:
        - POS disambiguation for function words
        - Adjective-noun agreement matching
        - Preposition-driven case selection
        - Custom vocabulary support

        Args:
            text: Text to analyze

        Returns:
            List of MorphologyInfo objects with disambiguated tags
        """
        if not self.is_available():
            return []

        # Use MAWO core for morphological analysis
        doc = self._nlp(text)
        results: list[MorphologyInfo] = []

        for token in doc.tokens:
            results.append(
                MorphologyInfo(
                    word=token.text,
                    pos=token.pos,
                    gender=token.gender,
                    case=token.case,
                    number=token.number,
                    aspect=token.aspect,
                    start=token.start,
                    stop=token.end,
                )
            )

        return results

    def check_grammar(self, text: str) -> list[ErrorAnnotation]:
        """Check Russian grammar using MAWO grammar checker with 690+ rules.

        The grammar checker validates:
        - Case agreement (adjective-noun, numeral-noun)
        - Verb aspect usage
        - Particle correctness
        - Preposition + case government
        - Register consistency
        - And 685+ more rules

        Args:
            text: Russian text to check

        Returns:
            List of detected grammar errors
        """
        if not self.is_available():
            return []

        # Use MAWO grammar checker for comprehensive grammar checking
        grammar_errors = self._grammar_checker.check(text)

        # Convert to ErrorAnnotation format
        errors = []
        for error in grammar_errors:
            # Map severity from mawo-grammar to our ErrorSeverity
            if error.severity == "major":
                severity = ErrorSeverity.CRITICAL
            elif error.severity == "minor":
                severity = ErrorSeverity.MINOR
            else:
                severity = ErrorSeverity.MAJOR

            errors.append(
                ErrorAnnotation(
                    category="fluency",
                    subcategory=f"russian_{error.type}",
                    severity=severity,
                    location=error.location,
                    description=error.description,
                    suggestion=error.suggestion,
                )
            )

        return errors

    def get_enrichment_data(self, text: str) -> dict[str, Any]:
        """Get comprehensive morphological data using MAWO core.

        Provides detailed linguistic context to help LLM make better decisions:
        - Verb aspects (perfective/imperfective) with aspect pairs
        - Adjective-noun pairs with automatic agreement checking
        - Gender, case, number information
        - POS distribution

        Args:
            text: Text to analyze

        Returns:
            Dictionary with morphological insights for LLM
        """
        if not self.is_available():
            return {"has_morphology": False}

        # Use MAWO core document for rich linguistic features
        doc = self._nlp(text)

        # Extract verb aspects using doc.verbs
        verb_aspects = {}
        verbs_list = []
        for verb in doc.verbs:
            verb_aspects[verb.text] = {
                "aspect": verb.aspect,
                "aspect_name": "perfective" if verb.aspect == "perf" else "imperfective",
                "position": f"{verb.start}-{verb.end}",
            }
            verbs_list.append(verb.text)

        # Extract adjective-noun pairs using doc.adjective_noun_pairs
        adj_noun_pairs = []
        for pair in doc.adjective_noun_pairs:
            pair_info = {
                "adjective": {
                    "word": pair.adjective.text,
                    "gender": pair.adjective.gender,
                    "case": pair.adjective.case,
                    "number": pair.adjective.number,
                },
                "noun": {
                    "word": pair.noun.text,
                    "gender": pair.noun.gender,
                    "case": pair.noun.case,
                    "number": pair.noun.number,
                },
                "agreement": pair.agreement,  # Returns "correct" or "mismatch"
            }
            adj_noun_pairs.append(pair_info)

        # Count parts of speech
        pos_counts: dict[str, int] = {}
        for token in doc.tokens:
            if token.pos:
                pos_counts[token.pos] = pos_counts.get(token.pos, 0) + 1

        return {
            "has_morphology": True,
            "word_count": len(doc.tokens),
            "verb_aspects": verb_aspects,
            "verbs_found": verbs_list,
            "adjective_noun_pairs": adj_noun_pairs,
            "pos_distribution": pos_counts,
        }

    def extract_entities(self, text: str) -> list[dict[str, Any]]:
        """Extract named entities from Russian text using MAWO core.

        Args:
            text: Text to extract entities from

        Returns:
            List of entities with type, text, and position
        """
        if not self.is_available():
            logger.debug("MAWO core not available, returning empty list")
            return []

        try:
            # Use MAWO core for entity extraction
            doc = self._nlp(text)

            # Convert entities to our format
            entities = []
            for entity in doc.entities:
                entities.append(
                    {
                        "text": entity.text,
                        "type": entity.label,
                        "start": entity.start,
                        "stop": entity.end,
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
        """Check if named entities from source are preserved in translation using MAWO core.

        Args:
            source_text: Original text (may be in any language)
            translation_text: Russian translation

        Returns:
            List of errors for missing/mismatched entities
        """
        if not self.is_available():
            logger.debug("MAWO core not available, skipping entity preservation check")
            return []

        try:
            # Use MAWO core for cross-document entity matching
            source_doc = self._nlp(source_text)
            translation_doc = self._nlp(translation_text)

            # Check entity preservation using match_entities
            matches = self._nlp.match_entities(source_doc, translation_doc)

            errors = []

            # Check for missing entities
            source_entities = list(source_doc.entities)
            matched_source_entities = {match.source for match in matches}

            missing_entities = [
                entity for entity in source_entities if entity not in matched_source_entities
            ]

            if missing_entities:
                for entity in missing_entities[:3]:  # Limit to first 3 to avoid spam
                    errors.append(
                        ErrorAnnotation(
                            category="accuracy",
                            subcategory="entity_omission",
                            severity=ErrorSeverity.MAJOR,
                            location=(0, min(50, len(translation_text))),
                            description=(
                                f"Entity '{entity.text}' ({entity.label}) from source "
                                f"appears to be missing in translation"
                            ),
                            suggestion="Verify that all proper nouns are correctly translated",
                        )
                    )

            logger.debug(
                f"Entity preservation check: "
                f"source_entities={len(source_entities)}, "
                f"translation_entities={len(list(translation_doc.entities))}, "
                f"matches={len(matches)}, "
                f"missing={len(missing_entities)}"
            )

            return errors

        except Exception as e:
            logger.error(f"Entity preservation check failed: {e}")
            # Fallback to basic check
            return self._basic_entity_check(source_text, translation_text)

    def _basic_entity_check(self, source_text: str, translation_text: str) -> list[ErrorAnnotation]:
        """Basic entity preservation check using regex (fallback method).

        Args:
            source_text: Original text
            translation_text: Russian translation

        Returns:
            List of errors for potential entity issues
        """
        import re

        errors = []

        # Find capitalized sequences in source (potential entity names)
        source_caps = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", source_text)
        translation_entities = self.extract_entities(translation_text)

        if len(source_caps) > 0 and len(translation_entities) == 0:
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

        return errors
