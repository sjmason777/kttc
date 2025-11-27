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

"""Hindi language helper with Indic NLP Library + Stanza + Spello integration."""

from __future__ import annotations

import logging
from typing import Any

from kttc.core import ErrorAnnotation, ErrorSeverity

from .base import LanguageHelper, MorphologyInfo

logger = logging.getLogger(__name__)

# Try to import Indic NLP Library (for tokenization and normalization)
try:
    from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
    from indicnlp.tokenize.indic_tokenize import trivial_tokenize

    INDIC_NLP_AVAILABLE = True
    logger.info("Using Indic NLP Library for Hindi tokenization and normalization")
except ImportError:
    INDIC_NLP_AVAILABLE = False
    logger.warning(
        "Indic NLP Library not installed. "
        "HindiLanguageHelper will run in limited mode. "
        "Install with: pip install indic-nlp-library"
    )

# Try to import Stanza (for POS tagging, NER, lemmatization)
try:
    import stanza

    STANZA_AVAILABLE = True
    logger.info("Stanza available for Hindi NLP (POS, NER, lemmatization)")
except ImportError:
    STANZA_AVAILABLE = False
    logger.warning(
        "Stanza not installed. "
        "HindiLanguageHelper will run without POS/NER/lemma. "
        "Install with: pip install stanza && python -c \"import stanza; stanza.download('hi')\""
    )

# Try to import Spello (for spell checking)
try:
    from spello.model import SpellCorrectionModel

    SPELLO_AVAILABLE = True
    logger.info("Spello available for Hindi spell checking")
except ImportError:
    SPELLO_AVAILABLE = False
    logger.warning(
        "Spello not installed. "
        "HindiLanguageHelper will run without spell checking. "
        "Install with: pip install spello"
    )


class HindiLanguageHelper(LanguageHelper):
    """Language helper for Hindi with Indic NLP + Stanza + Spello integration.

    Uses Indic NLP Library for:
    - Tokenization (word + sentence)
    - Text normalization for Devanagari script
    - Word segmentation

    Uses Stanza for:
    - POS tagging (Part-of-Speech)
    - NER (Named Entity Recognition) - NEW 2025!
    - Lemmatization
    - Dependency parsing

    Uses Spello for:
    - Spell checking with phonetic similarity
    - Context-aware corrections

    Example:
        >>> helper = HindiLanguageHelper()
        >>> if helper.is_available():
        ...     tokens = helper.tokenize("मैं स्कूल जाता हूं")
        ...     print([t[0] for t in tokens])
        ['मैं', 'स्कूल', 'जाता', 'हूं']
        ...     errors = helper.check_spelling("मैं सकूल जाता हूं")
        ...     print(errors[0].description if errors else "No errors")
    """

    def __init__(self) -> None:
        """Initialize Hindi language helper with Indic NLP + Stanza + Spello."""
        self._normalizer: Any = None
        self._stanza_nlp: Any = None
        self._spellchecker: Any = None
        self._initialized = False
        self._stanza_available = False
        self._spello_available = False

        # Initialize Indic NLP Library normalizer
        if INDIC_NLP_AVAILABLE:
            try:
                factory = IndicNormalizerFactory()
                self._normalizer = factory.get_normalizer("hi")
                self._initialized = True
                logger.info("HindiLanguageHelper initialized with Indic NLP Library")
            except Exception as e:
                logger.error(f"Failed to initialize Indic NLP normalizer: {e}")
                self._initialized = False
        else:
            logger.info("HindiLanguageHelper running in limited mode (no Indic NLP)")

        # Initialize Stanza for POS, NER, lemma
        if STANZA_AVAILABLE:
            try:
                # Download models if not present
                # stanza.download('hi', processors='tokenize,pos,ner,lemma')
                # Use REUSE_RESOURCES to avoid network calls (more resilient)
                self._stanza_nlp = stanza.Pipeline(
                    "hi",
                    processors="tokenize,pos,ner,lemma",
                    use_gpu=False,
                    download_method=stanza.DownloadMethod.REUSE_RESOURCES,
                )
                self._stanza_available = True
                logger.info("Stanza initialized successfully for Hindi (POS, NER, lemmatization)")
            except Exception as e:
                logger.warning(f"Stanza initialization failed: {e}")
                self._stanza_available = False

        # Initialize Spello for spell checking
        if SPELLO_AVAILABLE:
            try:
                self._spellchecker = SpellCorrectionModel(language="hi")
                self._spello_available = True
                logger.info("Spello initialized successfully for Hindi spell checking")
            except Exception as e:
                logger.warning(f"Spello initialization failed: {e}")
                self._spello_available = False

    @property
    def language_code(self) -> str:
        """Get language code."""
        return "hi"

    def is_available(self) -> bool:
        """Check if NLP dependencies are available."""
        return self._initialized and INDIC_NLP_AVAILABLE

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

    @staticmethod
    def _find_word_positions(text: str, words: list[str]) -> list[tuple[str, int, int]]:
        """Find positions of words in text.

        Args:
            text: Original text
            words: List of words to find

        Returns:
            List of (word, start, end) tuples
        """
        tokens = []
        start = 0
        for word in words:
            if not word or not word.strip():
                continue
            idx = text.find(word, start)
            if idx != -1:
                tokens.append((word, idx, idx + len(word)))
                start = idx + len(word)
        return tokens

    def tokenize(self, text: str) -> list[tuple[str, int, int]]:
        """Tokenize Hindi text with accurate positions using Indic NLP Library.

        Args:
            text: Text to tokenize

        Returns:
            List of (word, start, end) tuples
        """
        if not text or not text.strip():
            return []

        if not self.is_available():
            return self._find_word_positions(text, text.split())

        try:
            normalized_text = self._normalizer.normalize(text) if self._normalizer else text
            words = trivial_tokenize(normalized_text, lang="hi")
            return self._find_word_positions(normalized_text, words)

        except Exception as e:
            logger.error(f"Tokenization failed: {e}")
            return []

    @staticmethod
    def _parse_feats_string(feats: str | None) -> dict[str, str]:
        """Parse Stanza feats string to dictionary.

        Args:
            feats: String like "Gender=Masc|Case=Nom" or None

        Returns:
            Dictionary of feature key-value pairs
        """
        if not feats:
            return {}
        result = {}
        for feat in feats.split("|"):
            if "=" in feat:
                key, value = feat.split("=", 1)
                result[key] = value
        return result

    def _token_to_morphology_info(self, token: Any) -> MorphologyInfo:
        """Convert a Stanza token to MorphologyInfo.

        Args:
            token: Stanza token object

        Returns:
            MorphologyInfo instance
        """
        word = token.words[0]
        feats = self._parse_feats_string(word.feats)
        return MorphologyInfo(
            word=word.text,
            pos=word.upos,
            gender=feats.get("Gender"),
            case=feats.get("Case"),
            number=feats.get("Number"),
            aspect=feats.get("Aspect"),
            start=token.start_char,
            stop=token.end_char,
        )

    def analyze_morphology(self, text: str) -> list[MorphologyInfo]:
        """Analyze morphology of all words in text using Stanza.

        Args:
            text: Text to analyze

        Returns:
            List of MorphologyInfo objects
        """
        if not self._stanza_available or not self._stanza_nlp:
            return []

        try:
            doc = self._stanza_nlp(text)
            return [
                self._token_to_morphology_info(token)
                for sentence in doc.sentences
                for token in sentence.tokens
            ]

        except Exception as e:
            logger.error(f"Morphology analysis failed: {e}")
            return []

    # Hindi case markers (postpositions)
    _CASE_MARKERS = {
        "ने": "ERG",
        "को": "ACC/DAT",
        "से": "INS/ABL",
        "में": "LOC",
        "पर": "LOC",
        "का": "GEN",
        "की": "GEN",
        "के": "GEN",
    }

    def _check_ergative_case(self, sentence: Any, errors: list[ErrorAnnotation]) -> None:
        """Check ergative ने usage with past tense verbs."""
        ne_indices = [i for i, word in enumerate(sentence.words) if word.text == "ने"]
        if not ne_indices:
            return

        verbs = [w for w in sentence.words if w.upos == "VERB"]
        if not verbs:
            return

        main_verb = verbs[-1]
        if main_verb.feats and "Tense=Past" not in main_verb.feats:
            ne_word = sentence.words[ne_indices[0]]
            errors.append(
                ErrorAnnotation(
                    category="fluency",
                    subcategory="hindi_grammar_case",
                    severity=ErrorSeverity.MAJOR,
                    location=(ne_word.start_char, ne_word.end_char),
                    description="Ergative case marker 'ने' (ne) should be used with transitive past tense verbs",
                    suggestion="Check verb tense or remove 'ने'",
                )
            )

    def _extract_features(self, word: Any) -> tuple[str | None, str | None]:
        """Extract gender and number from word features."""
        gender, number = None, None
        if word.feats:
            for feat in word.feats.split("|"):
                if "Gender=" in feat:
                    gender = feat.split("=")[1]
                if "Number=" in feat:
                    number = feat.split("=")[1]
        return gender, number

    def _check_subject_verb_agreement(self, sentence: Any, errors: list[ErrorAnnotation]) -> None:
        """Check subject-verb agreement in gender and number."""
        subjects = [w for w in sentence.words if w.deprel and "subj" in w.deprel.lower()]
        verbs = [w for w in sentence.words if w.upos == "VERB"]

        if not subjects or not verbs:
            return

        subject = subjects[0]
        main_verb = verbs[-1]

        subj_gender, subj_number = self._extract_features(subject)
        verb_gender, verb_number = self._extract_features(main_verb)

        if subj_gender and verb_gender and subj_gender != verb_gender:
            errors.append(
                ErrorAnnotation(
                    category="fluency",
                    subcategory="hindi_grammar_agreement",
                    severity=ErrorSeverity.MAJOR,
                    location=(main_verb.start_char, main_verb.end_char),
                    description=f"Verb gender ({verb_gender}) does not match subject gender ({subj_gender})",
                    suggestion="Ensure verb agrees with subject in gender",
                )
            )

        if subj_number and verb_number and subj_number != verb_number:
            errors.append(
                ErrorAnnotation(
                    category="fluency",
                    subcategory="hindi_grammar_agreement",
                    severity=ErrorSeverity.MAJOR,
                    location=(main_verb.start_char, main_verb.end_char),
                    description=f"Verb number ({verb_number}) does not match subject number ({subj_number})",
                    suggestion="Ensure verb agrees with subject in number",
                )
            )

    def _check_word_order(self, sentence: Any, errors: list[ErrorAnnotation]) -> None:
        """Check Hindi SOV word order."""
        subjects = [w for w in sentence.words if w.deprel and "subj" in w.deprel.lower()]
        verbs = [w for w in sentence.words if w.upos == "VERB"]

        if not subjects or not verbs:
            return

        subject_idx = subjects[0].id
        main_verb_idx = verbs[-1].id

        if main_verb_idx < subject_idx - 2:
            errors.append(
                ErrorAnnotation(
                    category="fluency",
                    subcategory="hindi_grammar_word_order",
                    severity=ErrorSeverity.MINOR,
                    location=(sentence.words[0].start_char, sentence.words[-1].end_char),
                    description="Unusual word order detected: verb appears before subject. Hindi typically follows SOV order",
                    suggestion="Verify sentence structure follows natural Hindi word order",
                )
            )

    def _check_multiple_case_markers(self, sentence: Any, errors: list[ErrorAnnotation]) -> None:
        """Check for consecutive case markers."""
        for i in range(len(sentence.words) - 1):
            word1 = sentence.words[i]
            word2 = sentence.words[i + 1]

            if word1.text in self._CASE_MARKERS and word2.text in self._CASE_MARKERS:
                errors.append(
                    ErrorAnnotation(
                        category="fluency",
                        subcategory="hindi_grammar_case",
                        severity=ErrorSeverity.MAJOR,
                        location=(word1.start_char, word2.end_char),
                        description=f"Consecutive case markers '{word1.text}' and '{word2.text}' detected",
                        suggestion="Remove redundant case marker",
                    )
                )

    def check_grammar(self, text: str) -> list[ErrorAnnotation]:
        """Check Hindi grammar using deterministic rules with Stanza."""
        if not self._stanza_available or not self._stanza_nlp:
            logger.debug("Stanza not available, skipping grammar checks")
            return []

        errors: list[ErrorAnnotation] = []

        try:
            doc = self._stanza_nlp(text)

            for sentence in doc.sentences:
                self._check_ergative_case(sentence, errors)
                self._check_subject_verb_agreement(sentence, errors)
                self._check_word_order(sentence, errors)
                self._check_multiple_case_markers(sentence, errors)

            logger.debug(f"Hindi grammar check found {len(errors)} errors")
            return errors

        except Exception as e:
            logger.error(f"Hindi grammar check failed: {e}")
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
                    subcategory="hindi_spelling",
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
                    subcategory="hindi_spelling",
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
                    subcategory="hindi_spelling",
                    severity=ErrorSeverity.MINOR,
                    location=(insert_pos, insert_pos),
                    description="Missing word(s) detected",
                    suggestion=" ".join(inserted),
                )
            )

    def check_spelling(self, text: str) -> list[ErrorAnnotation]:
        """Check Hindi spelling using Spello.

        Uses phonetic similarity (Soundex) and edit-distance (Symspell)
        for context-aware spell checking. Implements token-based diff
        to pinpoint exact error positions.

        Args:
            text: Hindi text to check

        Returns:
            List of detected spelling errors with exact positions and suggestions
        """
        if not self._spello_available or not self._spellchecker:
            logger.debug("Spello not available, skipping spell checks")
            return []

        try:
            corrected = self._spellchecker.spell_correct(text)
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

            logger.debug(f"Spello found {len(errors)} spelling errors")
            return errors

        except Exception as e:
            logger.error(f"Spello check failed: {e}")
            return []

    @staticmethod
    def _count_pos_tags(doc: Any) -> dict[str, int]:
        """Count POS tags from Stanza document.

        Args:
            doc: Stanza document

        Returns:
            Dictionary of POS tag counts
        """
        pos_counts: dict[str, int] = {}
        for sentence in doc.sentences:
            for token in sentence.tokens:
                upos = token.words[0].upos
                if upos:
                    pos_counts[upos] = pos_counts.get(upos, 0) + 1
        return pos_counts

    @staticmethod
    def _extract_entities_from_doc(doc: Any) -> list[dict[str, Any]]:
        """Extract named entities from Stanza document.

        Args:
            doc: Stanza document

        Returns:
            List of entity dictionaries
        """
        return [
            {
                "text": ent.text,
                "type": ent.type,
                "start": ent.start_char,
                "end": ent.end_char,
            }
            for sentence in doc.sentences
            for ent in sentence.ents
        ]

    def _get_stanza_enrichment(self, text: str) -> dict[str, Any]:
        """Get Stanza-based linguistic enrichment.

        Args:
            text: Text to analyze

        Returns:
            Dictionary with Stanza analysis results
        """
        doc = self._stanza_nlp(text)
        return {
            "word_count": sum(len(sent.tokens) for sent in doc.sentences),
            "sentence_count": len(doc.sentences),
            "pos_distribution": self._count_pos_tags(doc),
            "entities": self._extract_entities_from_doc(doc),
            "has_stanza": True,
        }

    def get_enrichment_data(self, text: str) -> dict[str, Any]:
        """Get comprehensive linguistic data for enriching LLM prompts.

        Provides detailed Hindi linguistic context:
        - POS distribution (using Stanza)
        - Named entities (using Stanza NER)
        - Morphological features
        - Normalized text

        Args:
            text: Text to analyze

        Returns:
            Dictionary with linguistic insights for LLM
        """
        if not self.is_available():
            return {"has_morphology": False}

        enrichment: dict[str, Any] = {"has_morphology": True}

        if self._normalizer:
            try:
                enrichment["normalized_text"] = self._normalizer.normalize(text)
            except Exception as e:
                logger.error(f"Normalization failed: {e}")

        if self._stanza_available and self._stanza_nlp:
            try:
                enrichment.update(self._get_stanza_enrichment(text))
            except Exception as e:
                logger.error(f"Stanza enrichment failed: {e}")

        return enrichment

    def extract_entities(self, text: str) -> list[dict[str, Any]]:
        """Extract named entities from Hindi text using Stanza NER.

        Args:
            text: Text to extract entities from

        Returns:
            List of entities with type, text, and position
        """
        if not self._stanza_available or not self._stanza_nlp:
            logger.debug("Stanza not available, returning empty list")
            return []

        try:
            doc = self._stanza_nlp(text)

            # Convert to our format
            entities = []
            for sentence in doc.sentences:
                for ent in sentence.ents:
                    entities.append(
                        {
                            "text": ent.text,
                            "type": ent.type,
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
            translation_text: Hindi translation

        Returns:
            List of errors for missing/mismatched entities
        """
        if not self._stanza_available or not self._stanza_nlp:
            logger.debug("Stanza not available, skipping entity preservation check")
            return []

        try:
            # Extract entities from translation (Hindi)
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
