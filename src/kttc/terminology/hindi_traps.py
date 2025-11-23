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

"""Hindi Language Traps Validator.

Validates Hindi-specific linguistic traps that are difficult for translation:
- Gender exceptions (words that don't follow standard -आ/-ई patterns)
- Idioms and proverbs (cannot be translated literally)
- Chandrabindu vs Anusvara (spelling traps)
- Homophones and paronyms (context-dependent meaning)
- Aspiration traps (meaning changes with aspiration)
- Ergativity rules (ने construction and verb agreement)

These checks are automatically enabled when target_lang='hi'.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class HindiTrapsValidator:
    """Validator for Hindi language traps and peculiarities.

    Loads glossaries for:
    - gender_traps_hi.json
    - idioms_hi.json
    - chandrabindu_anusvara_hi.json
    - homophones_paronyms_hi.json
    - aspiration_traps_hi.json
    - verb_ergativity_hi.json

    Example:
        >>> validator = HindiTrapsValidator()
        >>> gender_traps = validator.get_gender_exceptions_in_text("पानी ठंडी है")
        >>> # Returns: [{'word': 'पानी', 'correct_gender': 'masculine', ...}]
    """

    def __init__(self) -> None:
        """Initialize validator and load all glossaries."""
        self._glossaries: dict[str, dict[str, Any]] = {}
        self._load_glossaries()

    def _get_glossary_path(self) -> Path:
        """Get path to Hindi glossaries directory."""
        current_file = Path(__file__)
        # Go up to src/kttc, then to glossaries/hi
        project_root = current_file.parent.parent.parent.parent
        return project_root / "glossaries" / "hi"

    def _load_glossaries(self) -> None:
        """Load all Hindi trap glossaries."""
        glossary_path = self._get_glossary_path()

        glossary_files = [
            "gender_traps_hi.json",
            "idioms_hi.json",
            "chandrabindu_anusvara_hi.json",
            "homophones_paronyms_hi.json",
            "aspiration_traps_hi.json",
            "verb_ergativity_hi.json",
        ]

        for filename in glossary_files:
            filepath = glossary_path / filename
            if filepath.exists():
                try:
                    with open(filepath, encoding="utf-8") as f:
                        key = filename.replace("_hi.json", "").replace(".json", "")
                        self._glossaries[key] = json.load(f)
                        logger.debug(f"Loaded glossary: {filename}")
                except (json.JSONDecodeError, OSError) as e:
                    logger.warning(f"Failed to load {filename}: {e}")
            else:
                logger.debug(f"Glossary not found: {filepath}")

    def is_available(self) -> bool:
        """Check if glossaries are loaded."""
        return len(self._glossaries) > 0

    # ========== GENDER TRAPS ==========

    def get_gender_exceptions(self) -> dict[str, Any]:
        """Get all gender exception words from glossary."""
        glossary = self._glossaries.get("gender_traps", {})
        return {
            "aa_ending_feminine": glossary.get("exceptions_aa_ending_feminine", {}).get(
                "entries", {}
            ),
            "ii_ending_masculine": glossary.get("exceptions_ii_ending_masculine", {}).get(
                "entries", {}
            ),
            "consonant_ending_feminine": glossary.get("consonant_ending_feminine", {}).get(
                "entries", {}
            ),
            "loanwords": glossary.get("english_loanwords_gender", {}).get("entries", {}),
        }

    def get_gender_for_word(self, word: str) -> dict[str, Any] | None:
        """Get gender information for a specific word.

        Args:
            word: Hindi word to check

        Returns:
            Dict with gender info if word is an exception, None otherwise
        """
        exceptions = self.get_gender_exceptions()

        # Check each category
        for category, entries in exceptions.items():
            if word in entries:
                return {"word": word, "category": category, **entries[word]}

        return None

    def get_gender_exceptions_in_text(self, text: str) -> list[dict[str, Any]]:
        """Find gender exception words in text.

        Args:
            text: Hindi text to analyze

        Returns:
            List of gender exception words found with their info
        """
        found = []
        exceptions = self.get_gender_exceptions()

        for category, entries in exceptions.items():
            for word, info in entries.items():
                if word in text:
                    found.append({"word": word, "category": category, **info})

        return found

    # ========== IDIOMS ==========

    def get_idioms(self) -> dict[str, Any]:
        """Get all idioms from glossary."""
        glossary = self._glossaries.get("idioms", {})
        idioms = {}

        # Collect from all categories
        for category in [
            "body_parts_idioms",
            "animal_idioms",
            "number_idioms",
            "food_idioms",
        ]:
            if category in glossary:
                entries = glossary[category].get("entries", {})
                idioms.update(entries)

        return idioms

    def get_proverbs(self) -> dict[str, Any]:
        """Get all proverbs (lokokti) from glossary."""
        glossary = self._glossaries.get("idioms", {})
        result: dict[str, Any] = glossary.get("proverbs_lokokti", {}).get("entries", {})
        return result

    def get_idioms_in_text(self, text: str) -> list[dict[str, Any]]:
        """Find idioms in text.

        Args:
            text: Hindi text to analyze

        Returns:
            List of idioms found with their meanings
        """
        found = []
        idioms = self.get_idioms()

        for key, info in idioms.items():
            idiom = info.get("idiom", key.replace("_", " "))
            if idiom in text:
                found.append({"key": key, "idiom": idiom, **info})

        return found

    # ========== CHANDRABINDU / ANUSVARA ==========

    def get_chandrabindu_words(self) -> list[dict[str, Any]]:
        """Get words that should use chandrabindu."""
        glossary = self._glossaries.get("chandrabindu_anusvara", {})
        result: list[dict[str, Any]] = glossary.get("common_words_chandrabindu", {}).get(
            "entries", []
        )
        return result

    def get_meaning_change_pairs(self) -> dict[str, Any]:
        """Get pairs where chandrabindu/anusvara changes meaning."""
        glossary = self._glossaries.get("chandrabindu_anusvara", {})
        result: dict[str, Any] = glossary.get("meaning_change_pairs", {}).get("pairs", {})
        return result

    def check_chandrabindu_errors(self, text: str) -> list[dict[str, Any]]:
        """Check for chandrabindu/anusvara errors in text.

        Args:
            text: Hindi text to analyze

        Returns:
            List of potential errors found
        """
        errors = []
        chandrabindu_words = self.get_chandrabindu_words()

        for entry in chandrabindu_words:
            correct = entry.get("word", "")
            wrong = entry.get("wrong", "")

            if wrong and wrong in text:
                errors.append(
                    {
                        "found": wrong,
                        "correct": correct,
                        "meaning": entry.get("meaning", ""),
                        "type": "chandrabindu_missing",
                    }
                )

        return errors

    # ========== HOMOPHONES ==========

    def get_homophones(self) -> dict[str, Any]:
        """Get all homophone pairs from glossary."""
        glossary = self._glossaries.get("homophones_paronyms", {})
        result: dict[str, Any] = glossary.get("exact_homophones", {}).get("entries", {})
        return result

    def get_paronyms(self) -> dict[str, Any]:
        """Get all paronym pairs from glossary."""
        glossary = self._glossaries.get("homophones_paronyms", {})
        result: dict[str, Any] = glossary.get("paronyms", {}).get("entries", {})
        return result

    def get_homophones_in_text(self, text: str) -> list[dict[str, Any]]:
        """Find potential homophone confusion in text.

        Args:
            text: Hindi text to analyze

        Returns:
            List of homophones found that might need context verification
        """
        found = []
        homophones = self.get_homophones()

        for key, pair in homophones.items():
            word1 = pair.get("word_1", {}).get("word", "")
            word2 = pair.get("word_2", {}).get("word", "")

            if word1 and word1 in text:
                found.append(
                    {
                        "word": word1,
                        "pair_key": key,
                        "meaning": pair.get("word_1", {}).get("meaning_en", ""),
                        "confusable_with": word2,
                        "confusable_meaning": pair.get("word_2", {}).get("meaning_en", ""),
                    }
                )
            if word2 and word2 in text:
                found.append(
                    {
                        "word": word2,
                        "pair_key": key,
                        "meaning": pair.get("word_2", {}).get("meaning_en", ""),
                        "confusable_with": word1,
                        "confusable_meaning": pair.get("word_1", {}).get("meaning_en", ""),
                    }
                )

        return found

    # ========== ASPIRATION ==========

    def get_aspiration_pairs(self) -> dict[str, Any]:
        """Get minimal pairs where aspiration changes meaning."""
        glossary = self._glossaries.get("aspiration_traps", {})
        pairs = {}

        for row in ["ka_row", "cha_row", "ta_row_retroflex", "ta_row_dental", "pa_row"]:
            if row in glossary.get("minimal_pairs", {}):
                pairs.update(glossary["minimal_pairs"][row])

        return pairs

    def check_aspiration_words_in_text(self, text: str) -> list[dict[str, Any]]:
        """Find words that might have aspiration confusion.

        Args:
            text: Hindi text to analyze

        Returns:
            List of words that are part of aspiration minimal pairs
        """
        found = []
        pairs = self.get_aspiration_pairs()

        for key, pair in pairs.items():
            unasp = pair.get("unaspirated", "")
            asp = pair.get("aspirated", "")

            if unasp and unasp in text:
                found.append(
                    {
                        "word": unasp,
                        "type": "unaspirated",
                        "meaning": pair.get("unaspirated_meaning_en", ""),
                        "aspirated_form": asp,
                        "aspirated_meaning": pair.get("aspirated_meaning_en", ""),
                    }
                )
            if asp and asp in text:
                found.append(
                    {
                        "word": asp,
                        "type": "aspirated",
                        "meaning": pair.get("aspirated_meaning_en", ""),
                        "unaspirated_form": unasp,
                        "unaspirated_meaning": pair.get("unaspirated_meaning_en", ""),
                    }
                )

        return found

    # ========== ERGATIVITY ==========

    def get_ergativity_rules(self) -> dict[str, Any]:
        """Get ergativity rules from glossary."""
        glossary = self._glossaries.get("verb_ergativity", {})
        result: dict[str, Any] = glossary.get("ne_postposition_rules", {})
        return result

    def get_ergativity_examples(self) -> dict[str, Any]:
        """Get ergativity examples from glossary."""
        glossary = self._glossaries.get("verb_ergativity", {})
        result: dict[str, Any] = glossary.get("verb_agreement_rules", {})
        return result

    def get_common_ergativity_errors(self) -> dict[str, Any]:
        """Get common ergativity errors from glossary."""
        glossary = self._glossaries.get("verb_ergativity", {})
        result: dict[str, Any] = glossary.get("common_errors", {})
        return result

    def check_ne_usage(self, text: str) -> list[dict[str, Any]]:
        """Check for potential ने usage in text.

        Args:
            text: Hindi text to analyze

        Returns:
            List of ने occurrences that might need verification
        """
        found = []

        # Find all ने occurrences
        pattern = r"(\S+)\s+ने\s+"
        matches = re.finditer(pattern, text)

        for match in matches:
            subject = match.group(1)
            found.append(
                {
                    "subject": subject,
                    "position": match.start(),
                    "context": match.group(0),
                    "note": "Verify: transitive verb + perfective aspect?",
                }
            )

        return found

    # ========== COMBINED ANALYSIS ==========

    def analyze_text(self, text: str) -> dict[str, Any]:
        """Perform comprehensive Hindi traps analysis on text.

        Args:
            text: Hindi text to analyze

        Returns:
            Dict with all trap analysis results
        """
        return {
            "gender_exceptions": self.get_gender_exceptions_in_text(text),
            "idioms": self.get_idioms_in_text(text),
            "chandrabindu_errors": self.check_chandrabindu_errors(text),
            "homophones": self.get_homophones_in_text(text),
            "aspiration_words": self.check_aspiration_words_in_text(text),
            "ne_constructions": self.check_ne_usage(text),
        }

    def get_trap_summary(self, text: str) -> str:
        """Get a human-readable summary of traps found.

        Args:
            text: Hindi text to analyze

        Returns:
            Summary string
        """
        analysis = self.analyze_text(text)

        parts = []
        if analysis["gender_exceptions"]:
            parts.append(f"Gender exceptions: {len(analysis['gender_exceptions'])}")
        if analysis["idioms"]:
            parts.append(f"Idioms: {len(analysis['idioms'])}")
        if analysis["chandrabindu_errors"]:
            parts.append(f"Chandrabindu errors: {len(analysis['chandrabindu_errors'])}")
        if analysis["homophones"]:
            parts.append(f"Homophones: {len(analysis['homophones'])}")
        if analysis["aspiration_words"]:
            parts.append(f"Aspiration-sensitive words: {len(analysis['aspiration_words'])}")
        if analysis["ne_constructions"]:
            parts.append(f"ने constructions: {len(analysis['ne_constructions'])}")

        if not parts:
            return "No Hindi traps detected"

        return "Hindi traps found: " + ", ".join(parts)
