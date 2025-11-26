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

"""English Language Traps Validator.

Validates English-specific linguistic traps that are difficult for translation:
- Homophones (their/there/they're, your/you're, etc.)
- Phrasal verbs (take off, give up, etc. - cannot translate literally)
- Heteronyms (same spelling, different pronunciation: lead, read, etc.)
- Adjective order (opinion-size-age-shape-color-origin-material-purpose)
- Preposition traps (depend on, interested in, etc.)
- Idioms (piece of cake, break a leg, etc.)
- False friends (cross-language confusion)

These checks are automatically enabled when target_lang='en'.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class EnglishTrapsValidator:
    """Validator for English language traps and peculiarities.

    Loads glossaries for:
    - homophones_en.json
    - phrasal_verbs_en.json
    - heteronyms_en.json
    - adjective_order_en.json
    - preposition_traps_en.json
    - idioms_en.json
    - false_friends_en.json

    Example:
        >>> validator = EnglishTrapsValidator()
        >>> homophones = validator.find_homophones_in_text("Their going to school")
        >>> # Returns: [{'word': 'Their', 'likely_error': True, ...}]
    """

    # Common homophone error patterns
    HOMOPHONE_ERROR_PATTERNS = [
        # their/there/they're
        (r"\btheir\s+(is|are|was|were|going|coming|doing)\b", "their", "they're"),
        (r"\btheir\s+(a|an|the)\b", "their", "there"),
        # your/you're
        (r"\byour\s+(welcome|right|wrong|amazing|beautiful)\b", "your", "you're"),
        (r"\byour\s+going\b", "your", "you're"),
        # its/it's
        (r"\bit's\s+(tail|color|colour|name|size|shape)\b", "it's", "its"),
        (r"\bits\s+(going|raining|been|a|the)\b", "its", "it's"),
        # to/too
        (r"\bto\s+(much|many|late|early|big|small|hot|cold)\b", "to", "too"),
        # affect/effect
        (r"\bthe\s+affect\b", "affect", "effect"),
        (r"\ban\s+affect\b", "affect", "effect"),
        # loose/lose
        (r"\bloose\s+(the|a|my|your|weight|game|match)\b", "loose", "lose"),
    ]

    # Adjective categories for order checking
    ADJECTIVE_CATEGORIES = {
        "opinion": [
            "beautiful",
            "lovely",
            "ugly",
            "nice",
            "wonderful",
            "excellent",
            "horrible",
            "terrible",
            "amazing",
            "delicious",
            "awful",
            "gorgeous",
            "strange",
            "unusual",
            "interesting",
            "important",
            "perfect",
        ],
        "size": [
            "big",
            "small",
            "large",
            "tiny",
            "huge",
            "enormous",
            "massive",
            "little",
            "giant",
            "vast",
            "minute",
            "minuscule",
        ],
        "age": [
            "old",
            "young",
            "new",
            "ancient",
            "antique",
            "vintage",
            "modern",
            "contemporary",
            "recent",
            "elderly",
        ],
        "shape": [
            "round",
            "square",
            "rectangular",
            "triangular",
            "circular",
            "flat",
            "curved",
            "straight",
            "wide",
            "narrow",
            "oval",
        ],
        "color": [
            "red",
            "blue",
            "green",
            "yellow",
            "black",
            "white",
            "purple",
            "orange",
            "pink",
            "brown",
            "grey",
            "gray",
            "golden",
            "silver",
        ],
        "origin": [
            "american",
            "british",
            "chinese",
            "french",
            "german",
            "italian",
            "japanese",
            "russian",
            "indian",
            "spanish",
            "mexican",
            "brazilian",
        ],
        "material": [
            "wooden",
            "metal",
            "plastic",
            "glass",
            "cotton",
            "silk",
            "leather",
            "paper",
            "stone",
            "brick",
            "gold",
            "silver",
            "steel",
            "iron",
        ],
    }

    def __init__(self) -> None:
        """Initialize validator and load all glossaries."""
        self._glossaries: dict[str, dict[str, Any]] = {}
        self._load_glossaries()

    def _get_glossary_path(self) -> Path:
        """Get path to English glossaries directory."""
        current_file = Path(__file__)
        # Go up to src/kttc, then to glossaries/en
        project_root = current_file.parent.parent.parent.parent
        return project_root / "glossaries" / "en"

    def _load_glossaries(self) -> None:
        """Load all English trap glossaries."""
        glossary_path = self._get_glossary_path()

        glossary_files = [
            "homophones_en.json",
            "phrasal_verbs_en.json",
            "heteronyms_en.json",
            "adjective_order_en.json",
            "preposition_traps_en.json",
            "idioms_en.json",
            "false_friends_en.json",
        ]

        for filename in glossary_files:
            filepath = glossary_path / filename
            if filepath.exists():
                try:
                    with open(filepath, encoding="utf-8") as f:
                        key = filename.replace("_en.json", "").replace(".json", "")
                        self._glossaries[key] = json.load(f)
                        logger.debug(f"Loaded glossary: {filename}")
                except (json.JSONDecodeError, OSError) as e:
                    logger.warning(f"Failed to load {filename}: {e}")
            else:
                logger.debug(f"Glossary not found: {filepath}")

    def is_available(self) -> bool:
        """Check if glossaries are loaded."""
        return len(self._glossaries) > 0

    # ========== HOMOPHONES ==========

    def get_homophones(self) -> dict[str, Any]:
        """Get all homophones from glossary."""
        glossary = self._glossaries.get("homophones", {})
        result = {}

        # Flatten all categories
        for category in ["critical_homophones", "major_homophones", "common_homophones"]:
            pairs = glossary.get(category, {}).get("pairs", {})
            result.update(pairs)

        return result

    def find_homophones_in_text(self, text: str) -> list[dict[str, Any]]:
        """Find potential homophone errors in text.

        Args:
            text: English text to analyze

        Returns:
            List of potential homophone errors with suggestions
        """
        found = []

        # Check error patterns
        for pattern, wrong, correct in self.HOMOPHONE_ERROR_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                found.append(
                    {
                        "found_text": match.group(),
                        "position": (match.start(), match.end()),
                        "likely_error": True,
                        "wrong_word": wrong,
                        "correct_word": correct,
                        "severity": "major",
                        "suggestion": f"Did you mean '{correct}' instead of '{wrong}'?",
                    }
                )

        return found

    def check_homophone_context(self, text: str) -> list[dict[str, Any]]:
        """Check for homophones that need context verification.

        Args:
            text: English text to analyze

        Returns:
            List of homophones requiring manual review
        """
        warnings = []
        text_lower = text.lower()

        # Check critical homophones
        critical_words = [
            "their",
            "there",
            "they're",
            "your",
            "you're",
            "its",
            "it's",
            "to",
            "too",
            "two",
        ]

        for word in critical_words:
            if re.search(rf"\b{word}\b", text_lower):
                warnings.append(
                    {
                        "word": word,
                        "type": "critical_homophone",
                        "message": f"Verify correct usage of '{word}'",
                        "severity": "minor",
                    }
                )

        return warnings

    # ========== PHRASAL VERBS ==========

    def get_phrasal_verbs(self) -> dict[str, Any]:
        """Get all phrasal verbs from glossary."""
        glossary = self._glossaries.get("phrasal_verbs", {})
        result = {}

        # Collect from all verb categories
        for key, value in glossary.items():
            if key.startswith("verb_") and isinstance(value, dict):
                pv = value.get("phrasal_verbs", {})
                result.update(pv)

        return result

    def find_phrasal_verbs_in_text(self, text: str) -> list[dict[str, Any]]:
        """Find phrasal verbs in text that cannot be translated literally.

        Args:
            text: English text to analyze

        Returns:
            List of phrasal verbs with translation warnings
        """
        found = []
        phrasal_verbs = self.get_phrasal_verbs()
        text_lower = text.lower()

        for pv_key, data in phrasal_verbs.items():
            # Convert key to search pattern (take_off -> take off)
            search_pattern = pv_key.replace("_", " ")
            pattern = rf"\b{re.escape(search_pattern)}\b"

            if re.search(pattern, text_lower):
                found.append(
                    {
                        "phrasal_verb": search_pattern,
                        "meanings": data.get("meanings", []),
                        "separable": data.get("separable", False),
                        "severity": "critical",
                        "warning": "DO NOT translate word-by-word!",
                        "literal_error": data.get("literal_translation_error", ""),
                    }
                )

        return found

    # ========== HETERONYMS ==========

    def get_heteronyms(self) -> dict[str, Any]:
        """Get all heteronyms from glossary."""
        glossary = self._glossaries.get("heteronyms", {})
        result = {}

        # Collect from all categories
        for category in ["stress_shift_heteronyms", "vowel_change_heteronyms"]:
            entries = glossary.get(category, {}).get("entries", {})
            result.update(entries)

        return result

    def find_heteronyms_in_text(self, text: str) -> list[dict[str, Any]]:
        """Find heteronyms that may have different pronunciations/meanings.

        Args:
            text: English text to analyze

        Returns:
            List of heteronyms requiring context analysis
        """
        found = []
        heteronyms = self.get_heteronyms()
        text_lower = text.lower()

        for word, data in heteronyms.items():
            pattern = rf"\b{re.escape(word)}\b"
            if re.search(pattern, text_lower):
                # Collect all meanings
                meanings = []
                for key, val in data.items():
                    if isinstance(val, dict) and "meaning" in val:
                        meanings.append(
                            {
                                "form": key,
                                "meaning": val["meaning"],
                                "pronunciation": val.get("pronunciation", ""),
                            }
                        )

                found.append(
                    {
                        "word": word,
                        "meanings": meanings,
                        "severity": "major",
                        "warning": "Context determines pronunciation and meaning",
                    }
                )

        return found

    # ========== ADJECTIVE ORDER ==========

    def get_adjective_category(self, adjective: str) -> str | None:
        """Determine the category of an adjective.

        Args:
            adjective: The adjective to categorize

        Returns:
            Category name or None if not found
        """
        adj_lower = adjective.lower()
        for category, words in self.ADJECTIVE_CATEGORIES.items():
            if adj_lower in words:
                return category
        return None

    def check_adjective_order(self, text: str) -> list[dict[str, Any]]:
        """Check for adjective order violations.

        Args:
            text: English text to analyze

        Returns:
            List of adjective order violations
        """
        found = []
        order = ["opinion", "size", "age", "shape", "color", "origin", "material"]

        # Find noun phrases with multiple adjectives
        # Simple pattern: adjective adjective noun
        pattern = r"\b(\w+)\s+(\w+)\s+(\w+)\b"

        for match in re.finditer(pattern, text, re.IGNORECASE):
            word1, word2, word3 = match.groups()

            cat1 = self.get_adjective_category(word1)
            cat2 = self.get_adjective_category(word2)

            if cat1 and cat2:
                idx1 = order.index(cat1) if cat1 in order else -1
                idx2 = order.index(cat2) if cat2 in order else -1

                # Check if order is wrong
                if idx1 > idx2 >= 0:
                    found.append(
                        {
                            "found_text": match.group(),
                            "position": (match.start(), match.end()),
                            "violation": f"'{word1}' ({cat1}) should come after '{word2}' ({cat2})",
                            "correct_order": f"{word2} {word1} {word3}",
                            "severity": "major",
                            "rule": f"Order: {' → '.join(order)}",
                        }
                    )

        return found

    # ========== PREPOSITION TRAPS ==========

    def get_preposition_errors(self) -> list[dict[str, Any]]:
        """Get common preposition error patterns."""
        glossary = self._glossaries.get("preposition_traps", {})

        errors = []
        # Collect from verb_collocations
        collocations = glossary.get("verb_collocations", {}).get("common_errors", [])
        errors.extend(collocations)

        # Collect from adjective_collocations
        adj_errors = glossary.get("adjective_collocations", {}).get("common_errors", [])
        errors.extend(adj_errors)

        return errors

    def find_preposition_errors(self, text: str) -> list[dict[str, Any]]:
        """Find preposition errors in text.

        Args:
            text: English text to analyze

        Returns:
            List of preposition errors with corrections
        """
        found = []
        text_lower = text.lower()

        # Known error patterns
        error_patterns = [
            (r"\bdepend\s+(from|about|of)\b", "depend on"),
            (r"\binterested\s+(about|for|on)\b", "interested in"),
            (r"\bafraid\s+(from|about)\b", "afraid of"),
            (r"\bgood\s+(in|for)\s+\w+ing\b", "good at"),
            (r"\blisten\s+the\b", "listen to the"),
            (r"\bwait\s+you\b", "wait for you"),
            (r"\barrive\s+to\b", "arrive at/in"),
            (r"\bdiscuss\s+about\b", "discuss (no preposition)"),
            (r"\bmarried\s+with\b", "married to"),
            (r"\bsimilar\s+with\b", "similar to"),
            (r"\bborn\s+on\s+\d{4}\b", "born in [year]"),
        ]

        for pattern, correction in error_patterns:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                found.append(
                    {
                        "found_text": match.group(),
                        "position": (match.start(), match.end()),
                        "correction": correction,
                        "severity": "major",
                        "type": "preposition_error",
                    }
                )

        return found

    # ========== IDIOMS ==========

    def get_idioms(self) -> dict[str, Any]:
        """Get all idioms from glossary."""
        glossary = self._glossaries.get("idioms", {})
        result = {}

        # Flatten all categories
        for key, value in glossary.items():
            if key.endswith("_idioms") and isinstance(value, dict):
                idioms = value.get("idioms", {})
                result.update(idioms)

        # Also check common_expressions
        common = glossary.get("common_expressions", {}).get("idioms", {})
        result.update(common)

        return result

    def find_idioms_in_text(self, text: str) -> list[dict[str, Any]]:
        """Find idioms that cannot be translated literally.

        Args:
            text: English text to analyze

        Returns:
            List of idioms with translation warnings
        """
        found = []
        idioms = self.get_idioms()
        text_lower = text.lower()

        for idiom_key, data in idioms.items():
            # Convert key to search pattern
            search_pattern = idiom_key.replace("_", " ")
            if search_pattern in text_lower:
                found.append(
                    {
                        "idiom": data.get("idiom", search_pattern),
                        "meaning": data.get("meaning", ""),
                        "literal_error": data.get("literal_translation_error", ""),
                        "equivalents": data.get("equivalents", {}),
                        "severity": "critical",
                        "warning": "DO NOT translate literally!",
                    }
                )

        return found

    # ========== FALSE FRIENDS ==========

    def get_false_friends(self, source_lang: str | None = None) -> dict[str, Any]:
        """Get false friends from glossary.

        Args:
            source_lang: Filter by source language (ru, es, de, fr)

        Returns:
            Dictionary of false friends
        """
        glossary = self._glossaries.get("false_friends", {})

        if source_lang:
            lang_map = {
                "ru": "russian_english",
                "es": "spanish_english",
                "de": "german_english",
                "fr": "french_english",
            }
            key = lang_map.get(source_lang)
            if key:
                lang_data = glossary.get(key, {})
                if isinstance(lang_data, dict):
                    entries = lang_data.get("entries", {})
                    return dict(entries) if isinstance(entries, dict) else {}
                return {}

        # Return all
        result = {}
        for key, value in glossary.items():
            if "_english" in key and isinstance(value, dict):
                entries = value.get("entries", {})
                result.update(entries)

        return result

    def find_false_friends_in_context(
        self, _source_text: str, target_text: str, source_lang: str
    ) -> list[dict[str, Any]]:
        """Find potential false friend errors in translation.

        Args:
            source_text: Original text
            target_text: English translation
            source_lang: Source language code

        Returns:
            List of potential false friend issues
        """
        found = []
        false_friends = self.get_false_friends(source_lang)
        target_lower = target_text.lower()

        for _ff_key, data in false_friends.items():
            english_word = data.get("english_word", "").lower()
            if english_word and re.search(rf"\b{re.escape(english_word)}\b", target_lower):
                found.append(
                    {
                        "english_word": english_word,
                        "english_meaning": data.get("english_meaning", ""),
                        "false_friend": data.get(f"{source_lang}_false_friend", ""),
                        "false_meaning": data.get(f"{source_lang}_meaning", ""),
                        "severity": data.get("severity", "major"),
                        "warning": "Verify this is not a false friend translation",
                    }
                )

        return found

    # ========== COMPREHENSIVE CHECK ==========

    def analyze_text(self, text: str) -> dict[str, Any]:
        """Perform comprehensive analysis of English text for all traps.

        Args:
            text: English text to analyze

        Returns:
            Dictionary with all found traps categorized
        """
        return {
            "homophones": self.find_homophones_in_text(text),
            "homophone_warnings": self.check_homophone_context(text),
            "phrasal_verbs": self.find_phrasal_verbs_in_text(text),
            "heteronyms": self.find_heteronyms_in_text(text),
            "adjective_order": self.check_adjective_order(text),
            "preposition_errors": self.find_preposition_errors(text),
            "idioms": self.find_idioms_in_text(text),
        }

    def get_translation_warnings(self, text: str) -> list[str]:
        """Get all translation warnings for a text.

        Args:
            text: English text to analyze

        Returns:
            List of warning strings
        """
        warnings = []
        analysis = self.analyze_text(text)

        # Critical warnings - homophones
        for error in analysis["homophones"]:
            warnings.append(f"HOMOPHONE ERROR: '{error['found_text']}' - {error['suggestion']}")

        # Critical warnings - phrasal verbs
        for pv in analysis["phrasal_verbs"]:
            warnings.append(f"PHRASAL VERB: '{pv['phrasal_verb']}' - {pv['warning']}")

        # Critical warnings - idioms
        for idiom in analysis["idioms"]:
            warnings.append(
                f"IDIOM: '{idiom['idiom']}' means '{idiom['meaning']}' - {idiom['warning']}"
            )

        # Major warnings - heteronyms
        for het in analysis["heteronyms"]:
            warnings.append(f"HETERONYM: '{het['word']}' - context determines meaning")

        # Major warnings - adjective order
        for adj in analysis["adjective_order"]:
            warnings.append(f"ADJECTIVE ORDER: '{adj['found_text']}' - {adj['violation']}")

        # Major warnings - prepositions
        for prep in analysis["preposition_errors"]:
            warnings.append(
                f"PREPOSITION ERROR: '{prep['found_text']}' - use '{prep['correction']}'"
            )

        return warnings

    def _format_enrichment_section(
        self,
        items: list[Any],
        header: str,
        limit: int,
        formatter: Callable[[Any], str],
    ) -> list[str]:
        """Format a section of prompt enrichment."""
        if not items:
            return []
        lines = [header]
        for item in items[:limit]:
            lines.append(formatter(item))
        return lines

    def get_prompt_enrichment(self, text: str) -> str:
        """Generate prompt enrichment for LLM based on text analysis."""
        analysis = self.analyze_text(text)
        sections = []

        sections.extend(
            self._format_enrichment_section(
                analysis["homophones"],
                "## HOMOPHONE ERRORS DETECTED:",
                5,
                lambda h: f"- '{h['found_text']}': {h['suggestion']}",
            )
        )

        sections.extend(
            self._format_enrichment_section(
                analysis["phrasal_verbs"],
                "\n## PHRASAL VERBS (DO NOT translate literally!):",
                5,
                lambda pv: f"- '{pv['phrasal_verb']}': {'; '.join(m.get('meaning', '') for m in pv.get('meanings', [])[:2])}",
            )
        )

        def fmt_heteronym(h: dict[str, Any]) -> str:
            meanings = [f"{m['form']}: {m['meaning']}" for m in h.get("meanings", [])[:2]]
            return f"- '{h['word']}': {'; '.join(meanings)}"

        sections.extend(
            self._format_enrichment_section(
                analysis["heteronyms"],
                "\n## HETERONYMS (pronunciation depends on context):",
                3,
                fmt_heteronym,
            )
        )

        sections.extend(
            self._format_enrichment_section(
                analysis["idioms"],
                "\n## IDIOMS DETECTED (DO NOT translate literally!):",
                5,
                lambda i: f"- '{i['idiom']}': means '{i['meaning']}'",
            )
        )

        sections.extend(
            self._format_enrichment_section(
                analysis["adjective_order"],
                "\n## ADJECTIVE ORDER VIOLATIONS:",
                3,
                lambda a: f"- '{a['found_text']}' → '{a['correct_order']}'",
            )
        )

        sections.extend(
            self._format_enrichment_section(
                analysis["preposition_errors"],
                "\n## PREPOSITION ERRORS:",
                5,
                lambda p: f"- '{p['found_text']}' → '{p['correction']}'",
            )
        )

        return "\n".join(sections) if sections else ""
