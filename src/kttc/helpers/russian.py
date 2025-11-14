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
        """Analyze morphology of all words with context-aware disambiguation.

        Implements research-based disambiguation (2024-2025 findings):
        - POS priority: Prefers PREP/CONJ over NOUN for function words
        - Adjective-noun agreement: Matches case/gender/number
        - Preposition-driven case selection: Applies Russian grammar rules
        - Compound adjective detection: Heuristics for -ные/-ные endings

        Args:
            text: Text to analyze

        Returns:
            List of MorphologyInfo objects with disambiguated tags
        """
        if not self.is_available():
            return []

        tokens = self.tokenize(text)
        results: list[MorphologyInfo] = []

        # Custom dictionary for technical compound adjectives
        # These words should be treated as ADJF even if pymorphy3 doesn't recognize them
        compound_adjectives = {
            "мультиагентные",
            "мультиагентных",
            "мультиагентными",
            "мультиагентной",
            "мультиагентная",
            "кроссплатформенные",
            "кроссплатформенных",
            "кроссплатформенными",
            "высокопроизводительные",
            "высокопроизводительных",
            "многопоточные",
            "многопоточных",
            "многопользовательские",
            "многопользовательских",
        }

        # Known function words that should always be PREP/CONJ
        prep_words = {
            "в",
            "на",
            "с",
            "со",
            "к",
            "по",
            "за",
            "из",
            "о",
            "об",
            "от",
            "до",
            "для",
            "при",
            "про",
            "без",
            "под",
            "над",
            "перед",
            "между",
            "у",
            "через",
            "около",
            "вокруг",
            "ради",
            "сквозь",
            "среди",
            "благодаря",
            "возле",
            "вдоль",
            "после",
            "навстречу",
            "вне",
            "кроме",
        }
        conj_words = {
            "и",
            "а",
            "но",
            "или",
            "да",
            "зато",
            "однако",
            "либо",
            "то",
            "ни",
            "также",
            "тоже",
            "причем",
            "притом",
            "же",
        }

        # Preposition-to-case mapping for grammar rules
        prep_case_map = {
            # Genitive (родительный)
            "без": "gent",
            "для": "gent",
            "до": "gent",
            "из": "gent",
            "от": "gent",
            "у": "gent",
            "около": "gent",
            "вокруг": "gent",
            "среди": "gent",
            "ради": "gent",
            "возле": "gent",
            "вдоль": "gent",
            "после": "gent",
            "кроме": "gent",
            "вне": "gent",
            # Dative (дательный)
            "к": "datv",
            "по": "datv",
            "благодаря": "datv",
            "навстречу": "datv",
            # Accusative (винительный)
            "через": "accs",
            "про": "accs",
            "сквозь": "accs",
            # Instrumental (творительный)
            "над": "ablt",
            "перед": "ablt",
            "между": "ablt",
            # Prepositional (предложный)
            "о": "loct",
            "об": "loct",
            "при": "loct",
        }

        for i, (word, start, stop) in enumerate(tokens):
            if not word.strip():
                continue

            word_lower = word.lower()
            parses = self._morph.parse(word)

            if not parses:
                continue

            # Default to first parse (highest frequency in dictionary)
            parsed = parses[0]

            # Track morphology override for compound_adjectives that need manual morphology
            morphology_override = None

            # LAYER 0: Custom Dictionary Override
            # Check if word is in our custom compound adjectives dictionary
            if word_lower in compound_adjectives:
                # First try to find existing ADJF parse
                found_adjf = False
                for p in parses:
                    if p.tag.POS == "ADJF":
                        parsed = p
                        found_adjf = True
                        break

                # If no ADJF parse exists, create synthetic one from first parse
                # by manually overriding POS (for words like "мультиагентные" which
                # pymorphy3 doesn't recognize as adjectives)
                if not found_adjf and parses:
                    # Keep the first parse but we'll mark it as ADJF later
                    # For now just note that this is a custom adjective
                    pass  # Will be handled by MorphologyInfo creation with custom POS

            # Apply multi-layer disambiguation if multiple parses exist
            if len(parses) > 1:
                # LAYER 1: POS Priority for Function Words
                # Single-letter prepositions/conjunctions are often tagged as NOUN Abbr
                # but PREP/CONJ parse exists deeper in the list
                if word_lower in prep_words:
                    for p in parses:
                        if p.tag.POS == "PREP":
                            parsed = p
                            break
                elif word_lower in conj_words:
                    for p in parses:
                        if p.tag.POS == "CONJ":
                            parsed = p
                            break

                # LAYER 2: Preposition-Driven Case Selection (MUST run before adjective matching!)
                # Check both direct preposition (i-1) and preposition before adjective (i-2)
                required_case_from_prep = None

                if i > 0 and results:
                    prev_morph = results[-1]
                    prev_word_lower = prev_morph.word.lower()

                    # Direct preposition (PREP + NOUN)
                    if prev_morph.pos == "PREP":
                        if prev_word_lower in ("в", "на"):
                            # Ambiguous: prefer loct or accs
                            for p in parses:
                                if p.tag.case in ("loct", "accs"):
                                    parsed = p
                                    required_case_from_prep = p.tag.case
                                    break
                        elif prev_word_lower in ("с", "со"):
                            # Ambiguous: prefer gent or ablt
                            for p in parses:
                                if p.tag.case in ("gent", "ablt"):
                                    parsed = p
                                    required_case_from_prep = p.tag.case
                                    break
                        elif prev_word_lower in ("под", "за"):
                            # Ambiguous: prefer accs or ablt
                            for p in parses:
                                if p.tag.case in ("accs", "ablt"):
                                    parsed = p
                                    required_case_from_prep = p.tag.case
                                    break
                        elif prev_word_lower in prep_case_map:
                            # Unambiguous prepositions
                            required_case = prep_case_map[prev_word_lower]
                            for p in parses:
                                if p.tag.case == required_case:
                                    parsed = p
                                    required_case_from_prep = required_case
                                    break

                # Check for PREP + ADJ + NOUN pattern (i-2)
                if i > 1 and results and required_case_from_prep is None:
                    prev_prev_morph = results[-2] if len(results) >= 2 else None
                    if prev_prev_morph and prev_prev_morph.pos == "PREP":
                        prev_prev_word_lower = prev_prev_morph.word.lower()
                        if prev_prev_word_lower in prep_case_map:
                            # Enforce case from preposition two positions back
                            required_case = prep_case_map[prev_prev_word_lower]
                            for p in parses:
                                if p.tag.case == required_case:
                                    parsed = p
                                    required_case_from_prep = required_case
                                    break

                # LAYER 3A: Adjective Look-Back (for ADJ + ADJ patterns)
                # If this is an adjective after another adjective, copy its gender/number
                # (for sequences like "специализированные мультиагентные системы")
                # Also handles compound_adjectives that pymorphy3 doesn't recognize as ADJF
                if (
                    i > 0
                    and results
                    and (parsed.tag.POS == "ADJF" or word_lower in compound_adjectives)
                    and results[-1].pos == "ADJF"
                ):
                    prev_adj = results[-1]
                    # Try to find adjective parse matching prev adjective's gender/number/case
                    matched = False
                    for p in parses:
                        if (
                            p.tag.POS == "ADJF"
                            and p.tag.case == prev_adj.case
                            and p.tag.gender == prev_adj.gender
                            and p.tag.number == prev_adj.number
                        ):
                            parsed = p
                            matched = True
                            break

                    # For compound_adjectives without ADJF parse, manually set morphology
                    if not matched and word_lower in compound_adjectives:
                        # Store the morphology from previous adjective to apply later
                        morphology_override = {
                            "pos": "ADJF",
                            "case": prev_adj.case,
                            "gender": prev_adj.gender,
                            "number": prev_adj.number,
                        }

                # LAYER 3B: Adjective Look-Ahead (for PREP + ADJ + NOUN patterns)
                # If this is an adjective after a preposition, look ahead to the noun
                # to select the correct case/gender/number before the noun is processed
                should_lookahead = False
                if i < len(tokens) - 1 and parsed.tag.POS == "ADJF" and i > 0 and results:
                    if results[-1].pos == "PREP":
                        should_lookahead = True
                    # Also do look-ahead for ADJ + ADJ + NOUN (even without PREP)
                    elif results[-1].pos == "ADJF":
                        should_lookahead = True

                # LAYER 3C: Direct ADJ + NOUN look-ahead (no PREP)
                # For patterns like "автоматического обнаружения" without preposition
                # Trigger if current parse is ADJF OR if any ADJF parse exists
                has_adjf = parsed.tag.POS == "ADJF" or any(p.tag.POS == "ADJF" for p in parses)
                if (
                    i < len(tokens) - 1
                    and has_adjf
                    and (i == 0 or (results and results[-1].pos not in ("PREP", "ADJF")))
                ):
                    should_lookahead = True

                if should_lookahead:
                    # Peek at next word
                    next_word = tokens[i + 1][0]
                    next_parses = self._morph.parse(next_word)
                    if next_parses and next_parses[0].tag.POS == "NOUN":
                        # Determine target case based on context
                        prev_word_lower = results[-1].word.lower() if results else ""

                        # Case 1: After PREP - use preposition rules
                        if (
                            results
                            and results[-1].pos == "PREP"
                            and prev_word_lower in prep_case_map
                        ):
                            required_case = prep_case_map[prev_word_lower]
                            # Find noun parse with required case
                            target_noun = None
                            for np in next_parses:
                                if np.tag.case == required_case:
                                    target_noun = np
                                    break
                            # Select adjective matching case/gender/number of noun
                            if target_noun:
                                best_adj = None
                                # First try exact match (case + gender + number)
                                for p in parses:
                                    if (
                                        p.tag.POS == "ADJF"
                                        and p.tag.case == target_noun.tag.case
                                        and p.tag.gender == target_noun.tag.gender
                                        and p.tag.number == target_noun.tag.number
                                    ):
                                        best_adj = p
                                        break
                                # If no exact match, try case only
                                if not best_adj:
                                    for p in parses:
                                        if (
                                            p.tag.POS == "ADJF"
                                            and p.tag.case == target_noun.tag.case
                                        ):
                                            best_adj = p
                                            break
                                if best_adj:
                                    parsed = best_adj
                        # Case 2: After PREP "с"/"со" - ambiguous case
                        elif (
                            results and results[-1].pos == "PREP" and prev_word_lower in ("с", "со")
                        ):
                            # Look for gent or ablt in noun
                            matched = False
                            target_noun = None
                            for np in next_parses:
                                if np.tag.case in ("gent", "ablt"):
                                    target_noun = np
                                    break
                            # Match adjective to noun's case/gender/number
                            if target_noun:
                                for p in parses:
                                    if (
                                        p.tag.POS == "ADJF"
                                        and p.tag.case == target_noun.tag.case
                                        and p.tag.gender == target_noun.tag.gender
                                        and p.tag.number == target_noun.tag.number
                                    ):
                                        parsed = p
                                        matched = True
                                        break
                                # Fallback to case-only match
                                if not matched:
                                    for p in parses:
                                        if (
                                            p.tag.POS == "ADJF"
                                            and p.tag.case == target_noun.tag.case
                                        ):
                                            parsed = p
                                            break
                        # Case 3: No PREP (direct ADJ + NOUN) - match noun's case/gender/number
                        # Try ALL noun parses to find best adjective-noun combination
                        # ONLY use inanimate nouns - don't match with surnames/names
                        # This prevents hiding gender errors like "новый машина"
                        else:
                            # Check if next word has genitive parse
                            # If so, current ADJF is likely substantivized (used as noun)
                            # Example: "переменную окружения" - don't apply look-ahead
                            has_genitive = any(
                                np.tag.POS == "NOUN" and np.tag.case == "gent" for np in next_parses
                            )
                            if has_genitive:
                                # Likely substantivized adjective + genitive complement
                                # Don't apply look-ahead - leave default first parse
                                best_adj = None
                            else:
                                best_adj = None

                                # Try all NOUN parses to find exact match (case + gender + number)
                                # ONLY consider inanimate nouns (common words, not surnames)
                                for np in next_parses:
                                    if np.tag.POS != "NOUN":
                                        continue
                                    # Skip animate nouns (surnames, names)
                                    if "anim" in np.tag:
                                        continue

                                    for p in parses:
                                        if (
                                            p.tag.POS == "ADJF"
                                            and p.tag.case == np.tag.case
                                            and p.tag.gender == np.tag.gender
                                            and p.tag.number == np.tag.number
                                        ):
                                            best_adj = p
                                            break
                                    if best_adj:
                                        break  # Found inanimate match, stop searching

                                # Fallback: Try case-only match across all noun parses
                                # Also skip animate nouns here
                                if not best_adj:
                                    for np in next_parses:
                                        if np.tag.POS != "NOUN":
                                            continue
                                        # Skip animate nouns (surnames, names)
                                        if "anim" in np.tag:
                                            continue
                                        for p in parses:
                                            if p.tag.POS == "ADJF" and p.tag.case == np.tag.case:
                                                best_adj = p
                                                break
                                        if best_adj:
                                            break

                                # Final fallback: If next word is NOUN but no match found,
                                # still prefer ADJF over NOUN (to allow error detection for gender mismatch)
                                if not best_adj:
                                    for p in parses:
                                        if p.tag.POS == "ADJF":
                                            best_adj = p
                                            break

                            if best_adj:
                                parsed = best_adj

                # LAYER 4: Adjective-Noun Agreement
                # If previous word is adjective, match case first (required)
                # then match gender/number if possible (flexible for genitive)
                # Prefer inanimate nouns over animate (surnames) for common words
                if i > 0 and results and parsed.tag.POS == "NOUN":
                    prev_morph = results[-1]
                    if prev_morph.pos == "ADJF":
                        # Check if current word has genitive parse
                        # If so, previous ADJF is likely substantivized (used as noun)
                        # Example: "переменную окружения" - select genitive for "окружения"
                        has_genitive = any(
                            p.tag.POS == "NOUN" and p.tag.case == "gent" for p in parses
                        )
                        if has_genitive:
                            # Select genitive parse (likely genitive complement)
                            best_match = None
                            for p in parses:
                                if p.tag.POS == "NOUN" and p.tag.case == "gent":
                                    best_match = p
                                    break
                            if best_match:
                                parsed = best_match
                        else:
                            # Normal adjective-noun agreement
                            best_match = None
                            best_match_inanimate = None

                            # Try exact match first, preferring inanimate over animate
                            for p in parses:
                                if p.tag.POS != "NOUN":
                                    continue
                                if (
                                    p.tag.case == prev_morph.case
                                    and p.tag.gender == prev_morph.gender
                                    and p.tag.number == prev_morph.number
                                ):
                                    # Prefer inanimate (common nouns) over animate (surnames)
                                    if "inan" in p.tag:
                                        best_match_inanimate = p
                                        break
                                    elif not best_match:
                                        best_match = p

                            # Use inanimate match if found, otherwise use any match
                            if best_match_inanimate:
                                best_match = best_match_inanimate

                            # If no exact match, try case-only match (for genitive/accusative)
                            # Genitive -ого adjectives can modify both masc and neut nouns
                            if not best_match and prev_morph.case in ("gent", "accs"):
                                for p in parses:
                                    if p.tag.POS == "NOUN" and p.tag.case == prev_morph.case:
                                        # Again, prefer inanimate
                                        if "inan" in p.tag:
                                            best_match = p
                                            break
                                        elif not best_match:
                                            best_match = p

                            if best_match:
                                parsed = best_match

                # LAYER 5: Compound Adjective Detection
                # Words ending in -ные/-ные often compound adjectives
                # but pymorphy3 may lack ADJF parse in dictionary
                # IMPORTANT: Only run if word NOT already identified as ADJF by earlier layers!
                if (
                    word_lower.endswith(("ные", "ная", "ной", "ное", "ным", "ными", "ных"))
                    and parsed.tag.POS != "ADJF"
                ):
                    # Check if any ADJF parse exists
                    has_adjf = any(p.tag.POS == "ADJF" for p in parses)
                    if has_adjf:
                        for p in parses:
                            if p.tag.POS == "ADJF":
                                parsed = p
                                break

            tag = parsed.tag

            # Override POS for custom compound adjectives if needed
            pos_override = str(tag.POS) if tag.POS else None
            if word_lower in compound_adjectives and tag.POS != "ADJF":
                # Force ADJF for words in custom dictionary that pymorphy3
                # doesn't recognize as adjectives
                pos_override = "ADJF"

            # Use morphology_override if set (for compound_adjectives with manual morphology)
            if morphology_override:
                results.append(
                    MorphologyInfo(
                        word=word,
                        pos=morphology_override.get("pos", pos_override),
                        gender=morphology_override.get("gender"),
                        case=morphology_override.get("case"),
                        number=morphology_override.get("number"),
                        aspect=morphology_override.get("aspect"),
                        start=start,
                        stop=stop,
                    )
                )
            else:
                results.append(
                    MorphologyInfo(
                        word=word,
                        pos=pos_override,
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

        # Russian prepositions and their required cases
        # Based on standard Russian grammar rules
        # For ambiguous prepositions, we list one case here and handle multiple
        # cases in the special handling section below
        prep_case_map = {
            # Genitive case (родительный падеж)
            "без": "gent",  # without
            "для": "gent",  # for
            "до": "gent",  # until
            "из": "gent",  # from
            "от": "gent",  # from
            "у": "gent",  # at/near
            "около": "gent",  # near
            "вокруг": "gent",  # around
            # Dative case (дательный падеж)
            "к": "datv",  # to/towards
            "по": "datv",  # along/according to
            # Accusative case (винительный падеж)
            "через": "accs",  # through
            "про": "accs",  # about
            # Instrumental case (творительный падеж)
            "над": "ablt",  # above
            "перед": "ablt",  # in front of
            "между": "ablt",  # between
            # Prepositional case (предложный падеж)
            "о": "loct",  # about
            "об": "loct",  # about
            "при": "loct",  # at/in the presence of
            # Ambiguous prepositions (handled specially below):
            # "в", "на" - accusative (direction) or prepositional (location)
            "в": "accs",  # in/into (direction) or in (location with loct)
            "на": "accs",  # on/onto (direction) or on (location with loct)
            # "с", "со" - genitive (from) or instrumental (with)
            "с": "gent",  # from (gent) or with (ablt)
            "со": "gent",  # from (gent) or with (ablt)
            # "под", "за" - accusative (direction) or instrumental (location)
            "под": "accs",  # under (direction) or under (location with ablt)
            "за": "accs",  # behind/for (direction) or behind (location with ablt)
        }

        # Check preposition + noun agreement
        for i in range(len(morphs)):
            curr = morphs[i]

            # Check if current word is preposition
            # Use heuristic: if word is in our preposition map, treat it as preposition
            # even if pymorphy3 tagged it differently (e.g., "О" might be tagged as NOUN)
            prep_lower = curr.word.lower()
            is_preposition = curr.pos == "PREP" or prep_lower in prep_case_map

            if not is_preposition:
                continue

            # Find the next noun (skip adjectives)
            # Be tolerant: pymorphy3 sometimes mislabels adjectives as NOUNs
            noun_word = None
            skipped_adjectives = 0
            for j in range(i + 1, len(morphs)):
                word = morphs[j]
                # Check if it's a noun
                if word.pos == "NOUN":
                    # If we've already skipped potential adjectives, this is likely the target noun
                    if skipped_adjectives > 0 or j == i + 1:
                        noun_word = word
                        break
                    # First noun after preposition - could be adjective misclassified
                    # Check if it has adjective-like properties (gender, case match with next word)
                    if j + 1 < len(morphs) and morphs[j + 1].pos == "NOUN":
                        # There's another noun after - this might be adjective
                        skipped_adjectives += 1
                        continue
                    else:
                        # No noun after - this must be the target noun
                        noun_word = word
                        break
                elif word.pos in ("ADJF", "ADJS"):
                    # True adjective - skip it
                    skipped_adjectives += 1
                    continue
                else:
                    # Stop if we encounter something else
                    break

            if noun_word is None:
                continue

            # Get required case for this preposition
            required_case = prep_case_map.get(prep_lower)

            if required_case and noun_word.case:
                # Special handling for ambiguous prepositions (в, на, с, под, за)
                # These can take different cases depending on meaning (direction vs location)
                # For now, accept both common cases for these prepositions
                if prep_lower in ("в", "на"):
                    # Accept both accusative (direction) and prepositional (location)
                    # Also accept nominative for masculine nouns (often homonymous with accusative)
                    if noun_word.case not in ("accs", "loct", "nomn"):
                        errors.append(
                            ErrorAnnotation(
                                category="fluency",
                                subcategory="russian_case_agreement",
                                severity=ErrorSeverity.CRITICAL,
                                location=(curr.start, noun_word.stop),
                                description=(
                                    f"Preposition case violation: '{curr.word}' requires "
                                    f"accusative or prepositional case, but '{noun_word.word}' "
                                    f"is in {noun_word.case} case"
                                ),
                                suggestion=None,
                            )
                        )
                elif prep_lower in ("с", "со"):
                    # Accept both genitive (from) and instrumental (with)
                    if noun_word.case not in ("gent", "ablt"):
                        errors.append(
                            ErrorAnnotation(
                                category="fluency",
                                subcategory="russian_case_agreement",
                                severity=ErrorSeverity.CRITICAL,
                                location=(curr.start, noun_word.stop),
                                description=(
                                    f"Preposition case violation: '{curr.word}' requires "
                                    f"genitive or instrumental case, but '{noun_word.word}' "
                                    f"is in {noun_word.case} case"
                                ),
                                suggestion=None,
                            )
                        )
                elif prep_lower in ("под", "за"):
                    # Accept both accusative (direction) and instrumental (location)
                    if noun_word.case not in ("accs", "ablt"):
                        errors.append(
                            ErrorAnnotation(
                                category="fluency",
                                subcategory="russian_case_agreement",
                                severity=ErrorSeverity.CRITICAL,
                                location=(curr.start, noun_word.stop),
                                description=(
                                    f"Preposition case violation: '{curr.word}' requires "
                                    f"accusative or instrumental case, but '{noun_word.word}' "
                                    f"is in {noun_word.case} case"
                                ),
                                suggestion=None,
                            )
                        )
                elif noun_word.case != required_case:
                    errors.append(
                        ErrorAnnotation(
                            category="fluency",
                            subcategory="russian_case_agreement",
                            severity=ErrorSeverity.CRITICAL,
                            location=(curr.start, noun_word.stop),
                            description=(
                                f"Preposition case violation: '{curr.word}' requires "
                                f"{required_case} case, but '{noun_word.word}' is in "
                                f"{noun_word.case} case"
                            ),
                            suggestion=None,
                        )
                    )

        # Check adjective-noun agreement
        for i in range(len(morphs) - 1):
            curr = morphs[i]
            next_word = morphs[i + 1]

            # Skip if not adjective + noun
            if curr.pos != "ADJF" or next_word.pos != "NOUN":
                continue

            # Skip substantivized adjectives (adjectives used as nouns)
            # Pattern: ADJF + NOUN in genitive case
            # Example: "переменную окружения" (environment variable)
            # Here "переменную" is technically ADJF but functions as NOUN
            # and "окружения" is genitive complement
            if next_word.case == "gent":
                # Likely substantivized adjective with genitive complement
                # Don't check gender/case agreement
                continue

            # Check if there's a numeral before adjective (skip agreement checks in this case)
            # Russian numerals: два, три, четыре, пять, etc.
            # After numerals 2-4, nouns are in genitive singular, adjectives in genitive plural
            # This is a special case in Russian grammar
            has_numeral_before = False
            if i > 0:
                prev_word = morphs[i - 1]
                # Check if previous word is a numeral (NUMR) or specific numeral words
                numeral_words = {
                    "два",
                    "две",
                    "три",
                    "четыре",
                    "пять",
                    "шесть",
                    "семь",
                    "восемь",
                    "девять",
                    "десять",
                }
                if prev_word.pos == "NUMR" or prev_word.word.lower() in numeral_words:
                    has_numeral_before = True

            if has_numeral_before:
                # Skip agreement checks after numerals
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
