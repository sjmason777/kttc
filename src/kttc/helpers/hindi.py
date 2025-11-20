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

    def tokenize(self, text: str) -> list[tuple[str, int, int]]:
        """Tokenize Hindi text with accurate positions using Indic NLP Library.

        Args:
            text: Text to tokenize

        Returns:
            List of (word, start, end) tuples
        """
        # Handle empty string
        if not text or not text.strip():
            return []

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

        # Use Indic NLP Library tokenization
        try:
            # Normalize first
            if self._normalizer:
                normalized_text = self._normalizer.normalize(text)
            else:
                normalized_text = text

            # Tokenize
            words = trivial_tokenize(normalized_text, lang="hi")

            # Find positions
            tokens = []
            start = 0
            for word in words:
                # Skip empty tokens
                if not word or not word.strip():
                    continue
                idx = normalized_text.find(word, start)
                if idx != -1:
                    tokens.append((word, idx, idx + len(word)))
                    start = idx + len(word)

            return tokens

        except Exception as e:
            logger.error(f"Tokenization failed: {e}")
            return []

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
            results = []

            for sentence in doc.sentences:
                for token in sentence.tokens:
                    word = token.words[0]  # Get first word

                    # Parse feats string to dictionary
                    # feats is a string like "Gender=Masc|Case=Nom" or None
                    feats_dict = {}
                    if word.feats:
                        for feat in word.feats.split("|"):
                            if "=" in feat:
                                key, value = feat.split("=", 1)
                                feats_dict[key] = value

                    results.append(
                        MorphologyInfo(
                            word=word.text,
                            pos=word.upos,
                            gender=feats_dict.get("Gender"),
                            case=feats_dict.get("Case"),
                            number=feats_dict.get("Number"),
                            aspect=feats_dict.get("Aspect"),
                            start=token.start_char,
                            stop=token.end_char,
                        )
                    )

            return results

        except Exception as e:
            logger.error(f"Morphology analysis failed: {e}")
            return []

    def check_grammar(self, text: str) -> list[ErrorAnnotation]:
        """Check Hindi grammar using deterministic rules with Stanza.

        Implements rule-based grammar validation for common Hindi errors:
        - Case marker consistency (ने, को, से, में, पर, etc.)
        - Verb agreement with subject (gender/number/person)
        - Postposition usage patterns
        - Subject-Object-Verb word order violations
        - Dependency parsing anomalies

        Args:
            text: Hindi text to check

        Returns:
            List of detected grammar errors with positions and suggestions

        Note:
            Uses Stanza dependency parsing and POS tagging for validation.
            Complex grammatical nuances are still handled by HindiFluencyAgent.
        """
        if not self._stanza_available or not self._stanza_nlp:
            logger.debug("Stanza not available, skipping grammar checks")
            return []

        errors = []

        try:
            doc = self._stanza_nlp(text)

            # Define Hindi case markers (postpositions)
            case_markers = {
                "ने": "ERG",  # Ergative (transitive past subject)
                "को": "ACC/DAT",  # Accusative/Dative (object/indirect object)
                "से": "INS/ABL",  # Instrumental/Ablative (with/from)
                "में": "LOC",  # Locative (in/inside)
                "पर": "LOC",  # Locative (on/at)
                "का": "GEN",  # Genitive (possessive masculine)
                "की": "GEN",  # Genitive (possessive feminine)
                "के": "GEN",  # Genitive (possessive plural)
            }

            for sentence in doc.sentences:
                # Check 1: Ergative ने without transitive past verb
                ne_indices = [i for i, word in enumerate(sentence.words) if word.text == "ने"]
                if ne_indices:
                    # Look for main verb in sentence
                    verbs = [w for w in sentence.words if w.upos == "VERB"]
                    if verbs:
                        main_verb = verbs[-1]  # Last verb is usually main verb
                        # Check if verb is past tense transitive
                        if main_verb.feats:
                            # VerbForm=Fin and Tense=Past expected
                            is_past = "Tense=Past" in main_verb.feats
                            if not is_past:
                                # ने used but verb is not past tense
                                ne_word = sentence.words[ne_indices[0]]
                                errors.append(
                                    ErrorAnnotation(
                                        category="fluency",
                                        subcategory="hindi_grammar_case",
                                        severity=ErrorSeverity.MAJOR,
                                        location=(ne_word.start_char, ne_word.end_char),
                                        description=(
                                            "Ergative case marker 'ने' (ne) should be used "
                                            "with transitive past tense verbs"
                                        ),
                                        suggestion="Check verb tense or remove 'ने'",
                                    )
                                )

                # Check 2: Subject-verb agreement (gender/number)
                # Find subject (nsubj relation) and main verb
                subjects = [w for w in sentence.words if w.deprel and "subj" in w.deprel.lower()]
                verbs = [w for w in sentence.words if w.upos == "VERB"]

                if subjects and verbs:
                    subject = subjects[0]
                    main_verb = verbs[-1]  # Last verb is usually main verb

                    # Extract gender/number from subject and verb
                    subj_gender = None
                    subj_number = None
                    verb_gender = None
                    verb_number = None

                    if subject.feats:
                        for feat in subject.feats.split("|"):
                            if "Gender=" in feat:
                                subj_gender = feat.split("=")[1]
                            if "Number=" in feat:
                                subj_number = feat.split("=")[1]

                    if main_verb.feats:
                        for feat in main_verb.feats.split("|"):
                            if "Gender=" in feat:
                                verb_gender = feat.split("=")[1]
                            if "Number=" in feat:
                                verb_number = feat.split("=")[1]

                    # Check agreement
                    if subj_gender and verb_gender and subj_gender != verb_gender:
                        errors.append(
                            ErrorAnnotation(
                                category="fluency",
                                subcategory="hindi_grammar_agreement",
                                severity=ErrorSeverity.MAJOR,
                                location=(main_verb.start_char, main_verb.end_char),
                                description=(
                                    f"Verb gender ({verb_gender}) does not match "
                                    f"subject gender ({subj_gender})"
                                ),
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
                                description=(
                                    f"Verb number ({verb_number}) does not match "
                                    f"subject number ({subj_number})"
                                ),
                                suggestion="Ensure verb agrees with subject in number",
                            )
                        )

                # Check 3: Word order - Hindi is SOV (Subject-Object-Verb)
                # Flag if verb comes before subject (extreme violation)
                if subjects and verbs:
                    subject_idx = subjects[0].id
                    main_verb_idx = verbs[-1].id

                    # If verb comes significantly before subject (> 2 positions)
                    if main_verb_idx < subject_idx - 2:
                        errors.append(
                            ErrorAnnotation(
                                category="fluency",
                                subcategory="hindi_grammar_word_order",
                                severity=ErrorSeverity.MINOR,
                                location=(
                                    sentence.words[0].start_char,
                                    sentence.words[-1].end_char,
                                ),
                                description=(
                                    "Unusual word order detected: verb appears before subject. "
                                    "Hindi typically follows Subject-Object-Verb (SOV) order"
                                ),
                                suggestion="Verify sentence structure follows natural Hindi word order",
                            )
                        )

                # Check 4: Multiple case markers on same noun
                # Look for consecutive case markers
                for i in range(len(sentence.words) - 1):
                    word1 = sentence.words[i]
                    word2 = sentence.words[i + 1]

                    if word1.text in case_markers and word2.text in case_markers:
                        errors.append(
                            ErrorAnnotation(
                                category="fluency",
                                subcategory="hindi_grammar_case",
                                severity=ErrorSeverity.MAJOR,
                                location=(word1.start_char, word2.end_char),
                                description=(
                                    f"Consecutive case markers '{word1.text}' and '{word2.text}' detected. "
                                    "Each noun should have only one case marker"
                                ),
                                suggestion="Remove redundant case marker",
                            )
                        )

            logger.debug(f"Hindi grammar check found {len(errors)} errors")
            return errors

        except Exception as e:
            logger.error(f"Hindi grammar check failed: {e}")
            return []

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
            # Spello spell_correct returns corrected text
            corrected = self._spellchecker.spell_correct(text)

            errors = []

            # Token-based diff to find exact error positions
            if corrected != text:
                # Tokenize both original and corrected texts
                original_tokens = self.tokenize(text)
                corrected_tokens = self.tokenize(corrected)

                # Compare token-by-token
                # Use sequence alignment for more accurate matching
                from difflib import SequenceMatcher

                # Get matching blocks
                original_words = [t[0] for t in original_tokens]
                corrected_words = [t[0] for t in corrected_tokens]

                matcher = SequenceMatcher(None, original_words, corrected_words)

                # Find mismatched tokens
                for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                    if tag == "replace":
                        # Word(s) replaced - spelling error
                        for idx in range(i1, i2):
                            if idx < len(original_tokens):
                                orig_word, start, end = original_tokens[idx]

                                # Get corresponding corrected word(s)
                                corrections = []
                                for jdx in range(j1, j2):
                                    if jdx < len(corrected_tokens):
                                        corrections.append(corrected_tokens[jdx][0])

                                suggestion = " ".join(corrections) if corrections else orig_word

                                errors.append(
                                    ErrorAnnotation(
                                        category="fluency",
                                        subcategory="hindi_spelling",
                                        severity=ErrorSeverity.MINOR,
                                        location=(start, end),
                                        description=(
                                            f"Possible spelling error: '{orig_word}' "
                                            f"may be incorrect"
                                        ),
                                        suggestion=suggestion,
                                    )
                                )

                    elif tag == "delete":
                        # Extra word(s) - possible error or unnecessary repetition
                        for idx in range(i1, i2):
                            if idx < len(original_tokens):
                                orig_word, start, end = original_tokens[idx]
                                errors.append(
                                    ErrorAnnotation(
                                        category="fluency",
                                        subcategory="hindi_spelling",
                                        severity=ErrorSeverity.MINOR,
                                        location=(start, end),
                                        description=f"Possibly unnecessary word: '{orig_word}'",
                                        suggestion="",  # Remove word
                                    )
                                )

                    elif tag == "insert":
                        # Missing word(s) detected by spell checker
                        # Insert at position of previous token's end
                        if i1 > 0 and i1 <= len(original_tokens):
                            # Insert after previous token
                            prev_token = original_tokens[i1 - 1]
                            insert_pos = prev_token[2]  # End of previous token

                            # Get inserted words
                            inserted = []
                            for jdx in range(j1, j2):
                                if jdx < len(corrected_tokens):
                                    inserted.append(corrected_tokens[jdx][0])

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

            logger.debug(f"Spello found {len(errors)} spelling errors")
            return errors

        except Exception as e:
            logger.error(f"Spello check failed: {e}")
            return []

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

        # Normalize text
        if self._normalizer:
            try:
                normalized = self._normalizer.normalize(text)
                enrichment["normalized_text"] = normalized
            except Exception as e:
                logger.error(f"Normalization failed: {e}")

        # Use Stanza for detailed analysis
        if self._stanza_available and self._stanza_nlp:
            try:
                doc = self._stanza_nlp(text)

                # Count POS tags
                pos_counts: dict[str, int] = {}
                for sentence in doc.sentences:
                    for token in sentence.tokens:
                        word = token.words[0]
                        if word.upos:
                            pos_counts[word.upos] = pos_counts.get(word.upos, 0) + 1

                # Extract named entities
                entities = []
                for sentence in doc.sentences:
                    for ent in sentence.ents:
                        entities.append(
                            {
                                "text": ent.text,
                                "type": ent.type,
                                "start": ent.start_char,
                                "end": ent.end_char,
                            }
                        )

                # Count words and sentences
                word_count = sum(len(sent.tokens) for sent in doc.sentences)
                sent_count = len(doc.sentences)

                enrichment.update(
                    {
                        "word_count": word_count,
                        "sentence_count": sent_count,
                        "pos_distribution": pos_counts,
                        "entities": entities,
                        "has_stanza": True,
                    }
                )

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
