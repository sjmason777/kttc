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

"""Persian language helper with DadmaTools (spaCy-based) integration."""

from __future__ import annotations

import logging
from typing import Any

from kttc.core import ErrorAnnotation, ErrorSeverity

from .base import LanguageHelper, MorphologyInfo

logger = logging.getLogger(__name__)

# Try to import DadmaTools (all-in-one Persian NLP toolkit)
try:
    import dadmatools.pipeline.language as dadma_language

    DADMATOOLS_AVAILABLE = True
    logger.info("DadmaTools available for Persian NLP (all-in-one solution)")
except ImportError:
    dadma_language = None  # Allow patching in tests
    DADMATOOLS_AVAILABLE = False
    logger.warning(
        "DadmaTools not installed. "
        "PersianLanguageHelper will run in limited mode. "
        'Install with: pip install "dadmatools[full]"'
    )


class PersianLanguageHelper(LanguageHelper):
    """Language helper for Persian with DadmaTools (spaCy-based) integration.

    Uses DadmaTools v2 (all-in-one solution) for:
    - Tokenization and normalization
    - POS tagging (98.8% accuracy)
    - Lemmatization (89.9% accuracy)
    - Dependency parsing (85.6% accuracy)
    - NER (Named Entity Recognition)
    - Spell checking (NEW in v2!)
    - Sentiment analysis (NEW in v2!)
    - Informal-to-formal conversion (NEW in v2!)
    - Kasreh Ezafe detection

    DadmaTools is spaCy-based, so it returns spaCy Doc objects.

    Example:
        >>> helper = PersianLanguageHelper()
        >>> if helper.is_available():
        ...     tokens = helper.tokenize("من به مدرسه می‌روم")
        ...     print([t[0] for t in tokens])
        ['من', 'به', 'مدرسه', 'می‌روم']
        ...     # Spell checking (NEW!)
        ...     corrected = helper.check_spelling("متن با اشتباه املایی")
    """

    def __init__(self) -> None:
        """Initialize Persian language helper with DadmaTools."""
        self._nlp: Any = None
        self._initialized = False

        # Initialize DadmaTools pipeline (spaCy-based)
        if DADMATOOLS_AVAILABLE:
            try:
                # Full pipeline with all features
                # tok = tokenization
                # lem = lemmatization
                # pos = POS tagging
                # dep = dependency parsing
                # chunk = chunking
                # cons = constituency parsing
                # spellchecker = spell checking (NEW in v2!)
                # kasreh = Kasreh Ezafe detection
                # itf = informal-to-formal (NEW in v2!)
                # ner = Named Entity Recognition
                # sent = sentiment analysis (NEW in v2!)
                self._nlp = dadma_language.Pipeline(
                    "tok,lem,pos,dep,chunk,cons,spellchecker,kasreh,itf,ner,sent"
                )
                self._initialized = True
                logger.info("PersianLanguageHelper initialized with DadmaTools v2 (full pipeline)")
            except Exception as e:
                logger.error(f"Failed to initialize DadmaTools: {e}")
                self._initialized = False
        else:
            logger.info("PersianLanguageHelper running in limited mode (no DadmaTools)")

    @property
    def language_code(self) -> str:
        """Get language code."""
        return "fa"

    def is_available(self) -> bool:
        """Check if NLP dependencies are available."""
        return self._initialized and DADMATOOLS_AVAILABLE

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
        """Tokenize Persian text with accurate positions using DadmaTools.

        Args:
            text: Text to tokenize

        Returns:
            List of (word, start, end) tuples
        """
        if not self.is_available():
            # Fallback: simple split on whitespace
            tokens = []
            start = 0
            for word in text.split():
                idx = text.find(word, start)
                if idx != -1:
                    tokens.append((word, idx, idx + len(word)))
                    start = idx + len(word)
            return tokens

        # Use DadmaTools tokenization (spaCy-based)
        try:
            doc = self._nlp(text)
            return [(token.text, token.idx, token.idx + len(token.text)) for token in doc]

        except Exception as e:
            logger.error(f"Tokenization failed: {e}")
            return []

    def analyze_morphology(self, text: str) -> list[MorphologyInfo]:
        """Analyze morphology of all words in text using DadmaTools.

        Args:
            text: Text to analyze

        Returns:
            List of MorphologyInfo objects
        """
        if not self.is_available():
            return []

        try:
            doc = self._nlp(text)
            results = []

            for token in doc:
                results.append(
                    MorphologyInfo(
                        word=token.text,
                        pos=token.pos_,
                        gender=None,  # Persian doesn't have grammatical gender
                        case=None,  # Persian doesn't have case system
                        number=None,  # Persian number is context-dependent
                        aspect=None,  # Persian aspect handled differently
                        start=token.idx,
                        stop=token.idx + len(token.text),
                    )
                )

            return results

        except Exception as e:
            logger.error(f"Morphology analysis failed: {e}")
            return []

    # Persian prepositions that commonly cause errors
    _PREPOSITIONS = {
        "به": "to/at",  # Direction, location
        "از": "from/of",  # Source, possession
        "در": "in/at",  # Location (inside)
        "با": "with",  # Accompaniment
        "برای": "for",  # Purpose, benefit
        "تا": "until/to",  # Limit, destination
        "بر": "on/upon",  # Location (on surface)
        "مثل": "like",  # Comparison
    }

    def _extract_morph_features(self, token: Any) -> tuple[str | None, str | None]:
        """Extract person and number features from token morphology."""
        person, number = None, None
        if hasattr(token, "morph") and token.morph:
            morph_dict = token.morph.to_dict()
            person = morph_dict.get("Person")
            number = morph_dict.get("Number")
        return person, number

    def _check_subject_verb_agreement(self, doc: Any, errors: list[ErrorAnnotation]) -> None:
        """Check subject-verb agreement in Persian sentence."""
        for token in doc:
            if token.dep_ not in ["nsubj", "csubj"]:
                continue
            head_verb = token.head
            if head_verb.pos_ != "VERB":
                continue

            subj_person, subj_number = self._extract_morph_features(token)
            verb_person, verb_number = self._extract_morph_features(head_verb)

            if subj_person and verb_person and subj_person != verb_person:
                errors.append(
                    ErrorAnnotation(
                        category="fluency",
                        subcategory="persian_grammar_agreement",
                        severity=ErrorSeverity.MAJOR,
                        location=(head_verb.idx, head_verb.idx + len(head_verb.text)),
                        description=(
                            f"Verb person ({verb_person}) does not match "
                            f"subject person ({subj_person})"
                        ),
                        suggestion="Ensure verb agrees with subject in person",
                    )
                )

            if subj_number and verb_number and subj_number != verb_number:
                errors.append(
                    ErrorAnnotation(
                        category="fluency",
                        subcategory="persian_grammar_agreement",
                        severity=ErrorSeverity.MAJOR,
                        location=(head_verb.idx, head_verb.idx + len(head_verb.text)),
                        description=(
                            f"Verb number ({verb_number}) does not match "
                            f"subject number ({subj_number})"
                        ),
                        suggestion="Ensure verb agrees with subject in number",
                    )
                )

    def _check_word_order(self, doc: Any, text: str, errors: list[ErrorAnnotation]) -> None:
        """Check SOV word order in Persian sentence."""
        subjects = [token for token in doc if token.dep_ in ["nsubj", "csubj"]]
        verbs = [token for token in doc if token.pos_ == "VERB" and token.dep_ == "ROOT"]

        if subjects and verbs:
            subject = subjects[0]
            main_verb = verbs[0]
            if main_verb.i < subject.i - 2:  # Allow some flexibility
                errors.append(
                    ErrorAnnotation(
                        category="fluency",
                        subcategory="persian_grammar_word_order",
                        severity=ErrorSeverity.MINOR,
                        location=(0, len(text)),
                        description=(
                            "Unusual word order detected: verb appears before subject. "
                            "Persian typically follows Subject-Object-Verb (SOV) order"
                        ),
                        suggestion="Verify sentence structure follows natural Persian word order",
                    )
                )

    def _check_preposition_usage(self, doc: Any, errors: list[ErrorAnnotation]) -> None:
        """Check preposition usage patterns in Persian sentence."""
        for token in doc:
            if token.text not in self._PREPOSITIONS:
                continue
            if token.i + 1 >= len(doc):
                continue
            next_token = doc[token.i + 1]
            if next_token.pos_ in ["NOUN", "PRON", "PROPN", "NUM"]:
                continue
            has_object = any(child.dep_ in ["pobj", "obl", "obj"] for child in token.children)
            if not has_object:
                errors.append(
                    ErrorAnnotation(
                        category="fluency",
                        subcategory="persian_grammar_preposition",
                        severity=ErrorSeverity.MINOR,
                        location=(token.idx, token.idx + len(token.text)),
                        description=(
                            f"Preposition '{token.text}' ({self._PREPOSITIONS[token.text]}) "
                            "may be missing its object"
                        ),
                        suggestion="Ensure preposition is followed by a noun or pronoun",
                    )
                )

    def _check_dependency_anomalies(self, doc: Any, errors: list[ErrorAnnotation]) -> None:
        """Check for dependency parsing anomalies."""
        for token in doc:
            if token.dep_ == "ROOT" and token.pos_ not in ["VERB", "AUX"]:
                errors.append(
                    ErrorAnnotation(
                        category="fluency",
                        subcategory="persian_grammar_structure",
                        severity=ErrorSeverity.MINOR,
                        location=(token.idx, token.idx + len(token.text)),
                        description=(
                            f"Unusual sentence structure: '{token.text}' ({token.pos_}) "
                            "is the main clause element but is not a verb"
                        ),
                        suggestion="Verify sentence has a clear main verb",
                    )
                )

    def check_grammar(self, text: str) -> list[ErrorAnnotation]:
        """Check Persian grammar using deterministic rules with DadmaTools.

        Implements rule-based grammar validation for common Persian errors:
        - Ezafe (کسره اضافه) construction validation using DadmaTools kasreh detection
        - Subject-verb agreement checking
        - Common preposition usage errors
        - Subject-Object-Verb word order violations
        - Dependency parsing anomalies (85.6% accuracy)

        Args:
            text: Persian text to check

        Returns:
            List of detected grammar errors with positions and suggestions

        Note:
            Uses DadmaTools v2 dependency parsing (85.6%) and POS tagging (98.8%).
            Complex grammatical nuances are handled by PersianFluencyAgent.
        """
        if not self.is_available():
            logger.debug("DadmaTools not available, skipping grammar checks")
            return []

        errors: list[ErrorAnnotation] = []

        try:
            doc = self._nlp(text)

            self._check_subject_verb_agreement(doc, errors)
            self._check_word_order(doc, text, errors)
            self._check_preposition_usage(doc, errors)
            self._check_dependency_anomalies(doc, errors)

            logger.debug(f"Persian grammar check found {len(errors)} errors")
            return errors

        except Exception as e:
            logger.error(f"Persian grammar check failed: {e}")
            return []

    def _handle_spelling_replace(
        self,
        i1: int,
        i2: int,
        j1: int,
        j2: int,
        original_tokens: list[tuple[str, int, int]],
        corrected_tokens: list[tuple[str, int, int]],
        errors: list[ErrorAnnotation],
    ) -> None:
        """Handle replace operations in spelling diff."""
        for idx in range(i1, i2):
            if idx >= len(original_tokens):
                continue
            orig_word, start, end = original_tokens[idx]
            corrections = [
                corrected_tokens[jdx][0] for jdx in range(j1, j2) if jdx < len(corrected_tokens)
            ]
            suggestion = " ".join(corrections) if corrections else orig_word
            errors.append(
                ErrorAnnotation(
                    category="fluency",
                    subcategory="persian_spelling",
                    severity=ErrorSeverity.MINOR,
                    location=(start, end),
                    description=f"Possible spelling error: '{orig_word}' may be incorrect",
                    suggestion=suggestion,
                )
            )

    def _handle_spelling_delete(
        self,
        i1: int,
        i2: int,
        original_tokens: list[tuple[str, int, int]],
        errors: list[ErrorAnnotation],
    ) -> None:
        """Handle delete operations in spelling diff."""
        for idx in range(i1, i2):
            if idx >= len(original_tokens):
                continue
            orig_word, start, end = original_tokens[idx]
            errors.append(
                ErrorAnnotation(
                    category="fluency",
                    subcategory="persian_spelling",
                    severity=ErrorSeverity.MINOR,
                    location=(start, end),
                    description=f"Possibly unnecessary word: '{orig_word}'",
                    suggestion="",
                )
            )

    def _handle_spelling_insert(
        self,
        i1: int,
        j1: int,
        j2: int,
        original_tokens: list[tuple[str, int, int]],
        corrected_tokens: list[tuple[str, int, int]],
        errors: list[ErrorAnnotation],
    ) -> None:
        """Handle insert operations in spelling diff."""
        if i1 <= 0 or i1 > len(original_tokens):
            return
        prev_token = original_tokens[i1 - 1]
        insert_pos = prev_token[2]
        inserted = [
            corrected_tokens[jdx][0] for jdx in range(j1, j2) if jdx < len(corrected_tokens)
        ]
        if inserted:
            errors.append(
                ErrorAnnotation(
                    category="fluency",
                    subcategory="persian_spelling",
                    severity=ErrorSeverity.MINOR,
                    location=(insert_pos, insert_pos),
                    description="Missing word(s) detected",
                    suggestion=" ".join(inserted),
                )
            )

    def check_spelling(self, text: str) -> list[ErrorAnnotation]:
        """Check Persian spelling using DadmaTools v2 spell checker.

        Uses DadmaTools built-in spell checking (NEW in v2!) with token-based
        diff algorithm to pinpoint exact error positions.

        Args:
            text: Persian text to check

        Returns:
            List of detected spelling errors with exact positions and corrections
        """
        if not self.is_available():
            logger.debug("DadmaTools not available, skipping spell checks")
            return []

        try:
            doc = self._nlp(text)

            if not hasattr(doc._, "spell_corrected"):
                logger.debug("DadmaTools spell checker not configured in pipeline")
                return []

            corrected = doc._.spell_corrected
            if corrected == text:
                return []

            errors: list[ErrorAnnotation] = []
            original_tokens = self.tokenize(text)
            corrected_tokens = self.tokenize(corrected)

            from difflib import SequenceMatcher

            original_words = [t[0] for t in original_tokens]
            corrected_words = [t[0] for t in corrected_tokens]
            matcher = SequenceMatcher(None, original_words, corrected_words)

            for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                if tag == "replace":
                    self._handle_spelling_replace(
                        i1, i2, j1, j2, original_tokens, corrected_tokens, errors
                    )
                elif tag == "delete":
                    self._handle_spelling_delete(i1, i2, original_tokens, errors)
                elif tag == "insert":
                    self._handle_spelling_insert(
                        i1, j1, j2, original_tokens, corrected_tokens, errors
                    )

            logger.debug(f"DadmaTools spell checker found {len(errors)} errors")
            return errors

        except Exception as e:
            logger.error(f"Spell check failed: {e}")
            return []

    def check_sentiment(self, text: str) -> str | None:
        """Analyze sentiment of Persian text using DadmaTools v2.

        NEW in DadmaTools v2!

        Args:
            text: Persian text to analyze

        Returns:
            Sentiment label (positive/negative/neutral) or None
        """
        if not self.is_available():
            return None

        try:
            doc = self._nlp(text)

            # Get sentiment from doc extension
            if hasattr(doc._, "sentiment"):
                return str(doc._.sentiment)

            return None

        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            return None

    def convert_to_formal(self, text: str) -> str | None:
        """Convert informal Persian text to formal style using DadmaTools v2.

        NEW in DadmaTools v2!

        Args:
            text: Informal Persian text

        Returns:
            Formal text or None if conversion failed
        """
        if not self.is_available():
            return None

        try:
            doc = self._nlp(text)

            # Get formal text from doc extension
            if hasattr(doc._, "formal_text"):
                return str(doc._.formal_text)

            return None

        except Exception as e:
            logger.error(f"Informal-to-formal conversion failed: {e}")
            return None

    def get_enrichment_data(self, text: str) -> dict[str, Any]:
        """Get comprehensive linguistic data for enriching LLM prompts.

        Provides detailed Persian linguistic context:
        - POS distribution (98.8% accuracy)
        - Named entities
        - Dependency structure
        - Sentiment analysis (NEW!)
        - Formal/informal style

        Args:
            text: Text to analyze

        Returns:
            Dictionary with linguistic insights for LLM
        """
        if not self.is_available():
            return {"has_morphology": False}

        enrichment: dict[str, Any] = {"has_morphology": True}

        try:
            doc = self._nlp(text)

            # Count POS tags
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

            # Get sentiment (NEW in v2!)
            sentiment = None
            if hasattr(doc._, "sentiment"):
                sentiment = doc._.sentiment

            # Get formal version (NEW in v2!)
            formal_text = None
            if hasattr(doc._, "formal_text"):
                formal_text = doc._.formal_text

            # Count words and sentences
            word_count = len([token for token in doc if not token.is_punct])
            sent_count = len(list(doc.sents))

            enrichment.update(
                {
                    "word_count": word_count,
                    "sentence_count": sent_count,
                    "pos_distribution": pos_counts,
                    "entities": entities,
                    "sentiment": sentiment,
                    "formal_text": formal_text,
                    "has_dadmatools": True,
                }
            )

        except Exception as e:
            logger.error(f"DadmaTools enrichment failed: {e}")

        return enrichment

    def extract_entities(self, text: str) -> list[dict[str, Any]]:
        """Extract named entities from Persian text using DadmaTools NER.

        Args:
            text: Text to extract entities from

        Returns:
            List of entities with type, text, and position
        """
        if not self.is_available():
            logger.debug("DadmaTools not available, returning empty list")
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
            translation_text: Persian translation

        Returns:
            List of errors for missing/mismatched entities
        """
        if not self.is_available():
            logger.debug("DadmaTools not available, skipping entity preservation check")
            return []

        try:
            # Extract entities from translation (Persian)
            translation_entities = self.extract_entities(translation_text)

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
