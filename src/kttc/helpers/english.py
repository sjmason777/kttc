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

"""English language helper with spaCy integration."""

from __future__ import annotations

import logging
from typing import Any

from kttc.core import ErrorAnnotation

from .base import LanguageHelper, MorphologyInfo

logger = logging.getLogger(__name__)

# Try to import spaCy
try:
    import spacy

    SPACY_AVAILABLE = True
    logger.info("Using spaCy for English NLP")
except ImportError:
    SPACY_AVAILABLE = False
    logger.warning(
        "spaCy not installed. "
        "EnglishLanguageHelper will run in limited mode. "
        "Install with: pip install spacy && python -m spacy download en_core_web_sm"
    )

# Try to import LanguageTool
try:
    import language_tool_python

    LANGUAGETOOL_AVAILABLE = True
    logger.info("LanguageTool available for English grammar checking")
except ImportError:
    LANGUAGETOOL_AVAILABLE = False
    logger.warning(
        "LanguageTool not installed. "
        "EnglishLanguageHelper will run without grammar checking. "
        "Install with: pip install language-tool-python"
    )


class EnglishLanguageHelper(LanguageHelper):
    """Language helper for English with spaCy + LanguageTool integration.

    Uses spaCy for:
    - Tokenization
    - POS tagging
    - Lemmatization
    - Named entity recognition (NER)
    - Morphological analysis

    Uses LanguageTool for:
    - 5,000+ grammatical rules
    - Subject-verb agreement
    - Article usage (a/an/the)
    - Tense consistency
    - Preposition errors

    Example:
        >>> helper = EnglishLanguageHelper()
        >>> if helper.is_available():
        ...     errors = helper.check_grammar("He go to school")
        ...     print(errors[0].description)
        'Subject-verb agreement: Use "goes" instead of "go"'
    """

    def __init__(self) -> None:
        """Initialize English language helper with spaCy and LanguageTool."""
        self._nlp: Any = None
        self._initialized = False

        # Initialize spaCy
        if SPACY_AVAILABLE:
            try:
                # Try medium model first (better accuracy with word vectors)
                self._nlp = spacy.load("en_core_web_md")
                self._initialized = True
                logger.info("EnglishLanguageHelper initialized with spaCy en_core_web_md (50 MB)")
            except OSError:
                try:
                    # Fallback to small model
                    self._nlp = spacy.load("en_core_web_sm")
                    self._initialized = True
                    logger.info(
                        "EnglishLanguageHelper initialized with spaCy en_core_web_sm (15 MB)"
                    )
                except OSError:
                    logger.error(
                        "spaCy English model not found. "
                        "Download with: python -m spacy download en_core_web_md"
                    )
                    self._initialized = False
        else:
            logger.info("EnglishLanguageHelper running in limited mode (no spaCy)")

        # Initialize LanguageTool
        self._language_tool: Any = None
        self._lt_available = False

        if LANGUAGETOOL_AVAILABLE:
            try:
                self._language_tool = language_tool_python.LanguageTool("en-US")
                self._lt_available = True
                logger.info("LanguageTool initialized successfully (5,000+ grammar rules)")
            except Exception as e:
                logger.warning(f"LanguageTool initialization failed: {e}")
                self._lt_available = False

    @property
    def language_code(self) -> str:
        """Get language code."""
        return "en"

    def is_available(self) -> bool:
        """Check if NLP dependencies are available."""
        return self._initialized and SPACY_AVAILABLE

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

        return True

    def tokenize(self, text: str) -> list[tuple[str, int, int]]:
        """Tokenize English text with accurate positions using spaCy.

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

        # Use spaCy tokenization
        doc = self._nlp(text)
        return [(token.text, token.idx, token.idx + len(token.text)) for token in doc]

    def analyze_morphology(self, text: str) -> list[MorphologyInfo]:
        """Analyze morphology of all words in text.

        Args:
            text: Text to analyze

        Returns:
            List of MorphologyInfo objects
        """
        if not self.is_available():
            return []

        doc = self._nlp(text)
        results = []

        for token in doc:
            number_list = token.morph.get("Number")
            results.append(
                MorphologyInfo(
                    word=token.text,
                    pos=token.pos_,
                    gender=None,  # English doesn't have grammatical gender
                    case=None,  # English doesn't have case system
                    number=number_list[0] if number_list else None,
                    aspect=None,  # English aspect is handled differently
                    start=token.idx,
                    stop=token.idx + len(token.text),
                )
            )

        return results

    def check_grammar(self, text: str) -> list[ErrorAnnotation]:
        """Check English grammar using LanguageTool.

        Uses 5,000+ grammatical rules including:
        - Subject-verb agreement
        - Article usage (a/an/the)
        - Tense consistency
        - Preposition errors
        - Spelling mistakes

        Args:
            text: English text to check

        Returns:
            List of detected grammar errors with positions and suggestions
        """
        if not self._lt_available:
            logger.debug("LanguageTool not available, skipping grammar checks")
            return []

        try:
            # Check with LanguageTool
            matches = self._language_tool.check(text)

            errors = []
            for match in matches:
                # Filter out style-only suggestions not relevant for translation
                if not self._is_translation_relevant(match):
                    continue

                # Map to our error format
                errors.append(
                    ErrorAnnotation(
                        category="fluency",
                        subcategory=f"english_{match.rule_id}",
                        severity=self._map_severity(match),
                        location=(match.offset, match.offset + match.error_length),
                        description=match.message,
                        suggestion=match.replacements[0] if match.replacements else None,
                    )
                )

            logger.debug(f"LanguageTool found {len(errors)} grammar errors")
            return errors

        except Exception as e:
            logger.error(f"LanguageTool check failed: {e}")
            return []

    def _map_severity(self, match: Any) -> Any:
        """Map LanguageTool match to ErrorSeverity.

        Args:
            match: LanguageTool Match object

        Returns:
            ErrorSeverity enum value
        """
        from kttc.core import ErrorSeverity

        rule_id = match.rule_id.lower()

        # Critical errors (spelling, clear grammar mistakes)
        if any(pattern in rule_id for pattern in ["spelling", "typo", "misspell"]):
            return ErrorSeverity.CRITICAL

        # Major errors (agreement, verb form, tense)
        if any(
            pattern in rule_id
            for pattern in [
                "grammar",
                "agreement",
                "verb",
                "subject_verb",
                "tense",
                "article",
                "preposition",
            ]
        ):
            return ErrorSeverity.MAJOR

        # Minor errors (everything else)
        return ErrorSeverity.MINOR

    def _is_translation_relevant(self, match: Any) -> bool:
        """Filter out style-only suggestions not relevant for translation QA.

        Args:
            match: LanguageTool Match object

        Returns:
            True if error is relevant for translation, False otherwise
        """
        rule_id = match.rule_id.lower()

        # Exclude pure style suggestions
        exclude_patterns = ["style", "redundancy", "collocation", "cliche", "wordiness"]

        if any(pattern in rule_id for pattern in exclude_patterns):
            return False

        return True

    def _analyze_verb_tenses(self, doc: Any) -> dict[str, dict[str, str]]:
        """Extract verb tense information from document."""
        verb_tenses: dict[str, dict[str, str]] = {}
        for token in doc:
            if token.pos_ != "VERB":
                continue
            tense = token.morph.get("Tense")
            if tense:
                verb_tenses[token.text] = {
                    "tense": tense[0],
                    "aspect": token.morph.get("Aspect", [""])[0],
                    "person": token.morph.get("Person", [""])[0],
                    "number": token.morph.get("Number", [""])[0],
                }
        return verb_tenses

    def _analyze_article_noun_pairs(self, doc: Any) -> list[dict[str, Any]]:
        """Extract article-noun patterns from document."""
        pairs: list[dict[str, Any]] = []
        for i, token in enumerate(doc):
            if token.pos_ != "DET" or token.text.lower() not in ["a", "an", "the"]:
                continue
            for j in range(i + 1, min(i + 5, len(doc))):
                if doc[j].pos_ not in ["NOUN", "PROPN"]:
                    continue
                next_word = doc[i + 1].text if i + 1 < len(doc) else ""
                correct = "an" if next_word and next_word[0].lower() in "aeiou" else "a"
                is_correct = token.text.lower() == correct if token.text.lower() != "the" else True
                pairs.append(
                    {
                        "article": token.text.lower(),
                        "noun": doc[j].text,
                        "distance": j - i,
                        "correct": is_correct,
                    }
                )
                break
        return pairs

    def _analyze_subject_verb_pairs(self, doc: Any) -> list[dict[str, Any]]:
        """Extract subject-verb pairs for agreement checking."""
        pairs: list[dict[str, Any]] = []
        for token in doc:
            if token.dep_ != "nsubj":
                continue
            verb = token.head
            if verb.pos_ != "VERB":
                continue
            subject_number = token.morph.get("Number")
            verb_number = verb.morph.get("Number")
            pairs.append(
                {
                    "subject": token.text,
                    "verb": verb.text,
                    "subject_number": subject_number[0] if subject_number else None,
                    "verb_number": verb_number[0] if verb_number else None,
                    "agreement": (
                        subject_number == verb_number if subject_number and verb_number else None
                    ),
                }
            )
        return pairs

    def get_enrichment_data(self, text: str) -> dict[str, Any]:
        """Get comprehensive linguistic data for enriching LLM prompts.

        Provides detailed linguistic context to help LLM make better decisions:
        - Verb tenses and aspects
        - Article-noun patterns
        - Subject-verb pairs (for agreement checking)
        - POS distribution
        - Named entities
        - Sentence structure

        Args:
            text: Text to analyze

        Returns:
            Dictionary with morphological insights for LLM
        """
        if not self.is_available():
            return {"has_morphology": False}

        doc = self._nlp(text)

        # Count parts of speech
        pos_counts: dict[str, int] = {}
        for token in doc:
            if token.pos_:
                pos_counts[token.pos_] = pos_counts.get(token.pos_, 0) + 1

        # Extract named entities
        entities = [
            {"text": ent.text, "label": ent.label_, "start": ent.start_char, "end": ent.end_char}
            for ent in doc.ents
        ]

        return {
            "has_morphology": True,
            "word_count": len([token for token in doc if not token.is_punct]),
            "verb_tenses": self._analyze_verb_tenses(doc),
            "article_noun_pairs": self._analyze_article_noun_pairs(doc),
            "subject_verb_pairs": self._analyze_subject_verb_pairs(doc),
            "pos_distribution": pos_counts,
            "entities": entities,
            "sentence_count": len(list(doc.sents)),
        }

    def extract_entities(self, text: str) -> list[dict[str, Any]]:
        """Extract named entities from English text using spaCy NER.

        Args:
            text: Text to extract entities from

        Returns:
            List of entities with type, text, and position
        """
        if not self.is_available():
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
            translation_text: English translation

        Returns:
            List of errors for missing/mismatched entities
        """
        if not self.is_available():
            logger.debug("spaCy not available, skipping entity preservation check")
            return []

        try:
            # Extract entities from translation (English)
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
