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

"""Persian Language Traps Validator.

Validates Persian-specific linguistic traps that are difficult for translation:
- False friends (Persian-Arabic semantic differences)
- Ta'arof phrases (ritual politeness that cannot be translated literally)
- Colloquial vs formal register mixing
- Compound verb errors (wrong light verb selection)
- Idioms and expressions (cannot be translated literally)
- Untranslatable words (cultural concepts)

These checks are automatically enabled when target_lang='fa'.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class PersianTrapsValidator:
    """Validator for Persian language traps and peculiarities.

    Loads glossaries for:
    - false_friends_fa.json
    - taarof_politeness_fa.json
    - colloquial_formal_fa.json
    - compound_verbs_fa.json
    - idioms_fa.json
    - untranslatable_fa.json

    Example:
        >>> validator = PersianTrapsValidator()
        >>> false_friends = validator.find_false_friends_in_text("جهاز بزرگ")
        >>> # Returns: [{'word': 'جهاز', 'persian_meaning': 'ship', 'arabic_meaning': 'device'}]
    """

    def __init__(self) -> None:
        """Initialize validator and load all glossaries."""
        self._glossaries: dict[str, dict[str, Any]] = {}
        self._load_glossaries()

    def _get_glossary_path(self) -> Path:
        """Get path to Persian glossaries directory."""
        current_file = Path(__file__)
        # Go up to src/kttc, then to glossaries/fa
        project_root = current_file.parent.parent.parent.parent
        return project_root / "glossaries" / "fa"

    def _load_glossaries(self) -> None:
        """Load all Persian trap glossaries."""
        glossary_path = self._get_glossary_path()

        glossary_files = [
            "false_friends_fa.json",
            "taarof_politeness_fa.json",
            "colloquial_formal_fa.json",
            "compound_verbs_fa.json",
            "idioms_fa.json",
            "untranslatable_fa.json",
        ]

        for filename in glossary_files:
            filepath = glossary_path / filename
            if filepath.exists():
                try:
                    with open(filepath, encoding="utf-8") as f:
                        key = filename.replace("_fa.json", "").replace(".json", "")
                        self._glossaries[key] = json.load(f)
                        logger.debug(f"Loaded Persian glossary: {filename}")
                except (json.JSONDecodeError, OSError) as e:
                    logger.warning(f"Failed to load {filename}: {e}")
            else:
                logger.debug(f"Persian glossary not found: {filepath}")

    def is_available(self) -> bool:
        """Check if glossaries are loaded."""
        return len(self._glossaries) > 0

    # ========== FALSE FRIENDS (Persian-Arabic) ==========

    def get_false_friends(self) -> dict[str, Any]:
        """Get all Persian-Arabic false friends from glossary."""
        glossary = self._glossaries.get("false_friends", {})
        result: dict[str, Any] = glossary.get("persian_arabic_false_friends", {}).get("entries", {})
        return result

    def find_false_friends_in_text(self, text: str) -> list[dict[str, Any]]:
        """Find potential Persian-Arabic false friends in text.

        Args:
            text: Persian text to analyze

        Returns:
            List of false friends found with their different meanings
        """
        found = []
        false_friends = self.get_false_friends()

        for word, data in false_friends.items():
            # Check for word presence
            if word in text:
                found.append(
                    {
                        "word": word,
                        "persian_meaning": data.get("persian_meaning", ""),
                        "arabic_meaning": data.get("arabic_meaning", ""),
                        "severity": data.get("severity", "major"),
                        "translation_note": data.get("translation_note", ""),
                        "warning": "Persian-Arabic false friend - verify correct meaning for context",
                    }
                )

        return found

    # ========== TA'AROF (Ritual Politeness) ==========

    def get_taarof_phrases(self) -> dict[str, Any]:
        """Get all ta'arof phrases from glossary."""
        glossary = self._glossaries.get("taarof_politeness", {})
        result: dict[str, Any] = glossary.get("common_taarof_phrases", {})
        return result

    def get_taarof_scenarios(self) -> dict[str, Any]:
        """Get ta'arof scenarios from glossary."""
        glossary = self._glossaries.get("taarof_politeness", {})
        result: dict[str, Any] = glossary.get("taarof_scenarios", {})
        return result

    def find_taarof_in_text(self, text: str) -> list[dict[str, Any]]:
        """Find ta'arof phrases in text that require special translation handling.

        Args:
            text: Persian text to analyze

        Returns:
            List of ta'arof phrases found with translation warnings
        """
        found = []
        taarof_categories = self.get_taarof_phrases()

        # Flatten all ta'arof phrases from different categories
        all_phrases = {}
        for _category, phrases in taarof_categories.items():
            if isinstance(phrases, dict):
                for key, data in phrases.items():
                    if isinstance(data, dict) and "phrase_fa" in data:
                        all_phrases[key] = data

        # Search for ta'arof phrases
        for _phrase_key, data in all_phrases.items():
            phrase = data.get("phrase_fa", "")
            # Remove punctuation for matching
            clean_phrase = phrase.replace("؟", "").replace("!", "").strip()

            if clean_phrase and clean_phrase in text:
                found.append(
                    {
                        "phrase": clean_phrase,
                        "literal": data.get("literal", ""),
                        "actual_meaning": data.get("actual_meaning", ""),
                        "severity": data.get("severity", "major"),
                        "warning": "Ta'arof phrase - DO NOT translate literally!",
                        "translation_warning": data.get("translation_warning", ""),
                    }
                )

        return found

    # ========== COLLOQUIAL vs FORMAL ==========

    def get_colloquial_forms(self) -> dict[str, Any]:
        """Get colloquial-formal mappings."""
        glossary = self._glossaries.get("colloquial_formal", {})
        return {
            "verb_contractions": glossary.get("verb_contractions", {}).get("entries", {}),
            "particle_changes": glossary.get("particle_changes", {}).get("entries", {}),
            "vocabulary": glossary.get("vocabulary_differences", {}).get("entries", {}),
        }

    def check_register_consistency(self, text: str) -> list[dict[str, Any]]:
        """Check for mixing of formal and colloquial registers.

        Args:
            text: Persian text to analyze

        Returns:
            List of register inconsistencies found
        """
        found = []
        colloquial_data = self._glossaries.get("colloquial_formal", {})

        # Get register detection hints
        guidelines = colloquial_data.get("translation_guidelines", {})
        hints = guidelines.get("register_detection_hints", {})

        formal_indicators = hints.get("formal_indicators", [])
        colloquial_indicators = hints.get("colloquial_indicators", [])

        # Count formal vs colloquial indicators
        formal_count = sum(1 for ind in formal_indicators if ind in text)
        colloquial_count = sum(1 for ind in colloquial_indicators if ind in text)

        # If both types are present significantly, flag inconsistency
        if formal_count >= 2 and colloquial_count >= 2:
            found.append(
                {
                    "issue": "register_mixing",
                    "formal_found": formal_count,
                    "colloquial_found": colloquial_count,
                    "severity": "major",
                    "warning": "Text mixes formal and colloquial registers inconsistently",
                    "suggestion": "Maintain consistent register throughout the translation",
                }
            )

        return found

    def detect_register(self, text: str) -> str:
        """Detect the dominant register (formal or colloquial) of text.

        Args:
            text: Persian text to analyze

        Returns:
            'formal', 'colloquial', or 'mixed'
        """
        colloquial_data = self._glossaries.get("colloquial_formal", {})
        guidelines = colloquial_data.get("translation_guidelines", {})
        hints = guidelines.get("register_detection_hints", {})

        formal_indicators = hints.get("formal_indicators", [])
        colloquial_indicators = hints.get("colloquial_indicators", [])

        formal_count = sum(1 for ind in formal_indicators if ind in text)
        colloquial_count = sum(1 for ind in colloquial_indicators if ind in text)

        if formal_count > colloquial_count * 2:
            return "formal"
        if colloquial_count > formal_count * 2:
            return "colloquial"
        if formal_count > 0 and colloquial_count > 0:
            return "mixed"
        return "formal"  # Default to formal if unclear

    # ========== COMPOUND VERBS ==========

    def get_compound_verbs(self) -> dict[str, Any]:
        """Get compound verb patterns."""
        result: dict[str, Any] = self._glossaries.get("compound_verbs", {})
        return result

    def get_light_verbs(self) -> dict[str, Any]:
        """Get light verb information."""
        glossary = self._glossaries.get("compound_verbs", {})
        result: dict[str, Any] = glossary.get("light_verbs", {}).get("verbs", {})
        return result

    def get_common_compound_verb_errors(self) -> list[dict[str, Any]]:
        """Get common compound verb errors."""
        glossary = self._glossaries.get("compound_verbs", {})
        errors = glossary.get("common_errors", {}).get("entries", {})
        wrong_light_verb = errors.get("wrong_light_verb", {})
        result: list[dict[str, Any]] = wrong_light_verb.get("examples", [])
        return result

    def check_compound_verb_errors(self, text: str) -> list[dict[str, Any]]:
        """Check for common compound verb errors.

        Args:
            text: Persian text to analyze

        Returns:
            List of compound verb errors found
        """
        found = []
        errors = self.get_common_compound_verb_errors()

        for error in errors:
            wrong = error.get("wrong", "")
            if wrong and wrong in text:
                found.append(
                    {
                        "error": wrong,
                        "correct": error.get("correct", ""),
                        "meaning": error.get("meaning", ""),
                        "note": error.get("note", ""),
                        "severity": "critical",
                        "warning": f"Wrong light verb: '{wrong}' should be '{error.get('correct', '')}'",
                    }
                )

        return found

    # ========== IDIOMS ==========

    def get_idioms(self) -> dict[str, Any]:
        """Get all idioms from glossary."""
        glossary = self._glossaries.get("idioms", {})
        result: dict[str, Any] = glossary.get("idiom_categories", {})
        return result

    def find_idioms_in_text(self, text: str) -> list[dict[str, Any]]:
        """Find idioms in text that cannot be translated literally.

        Args:
            text: Persian text to analyze

        Returns:
            List of idioms found with translation warnings
        """
        found = []
        idiom_categories = self.get_idioms()

        # Flatten idioms from all categories
        all_idioms = {}
        for _category, category_data in idiom_categories.items():
            if isinstance(category_data, dict):
                entries = category_data.get("entries", {})
                all_idioms.update(entries)

        # Search for idioms
        for _idiom_key, data in all_idioms.items():
            idiom_fa = data.get("idiom_fa", "")
            # Try to find the idiom (may need partial matching)
            if idiom_fa:
                # Extract core words from the idiom
                core_words = idiom_fa.replace("کسی", "").replace("چیزی", "").strip()
                if core_words and len(core_words) > 3 and core_words in text:
                    found.append(
                        {
                            "idiom": idiom_fa,
                            "literal": data.get("literal", ""),
                            "meaning": data.get("meaning", ""),
                            "english_equivalent": data.get("english_equivalent", ""),
                            "severity": data.get("severity", "major"),
                            "warning": "Idiom - DO NOT translate literally!",
                        }
                    )

        return found

    # ========== UNTRANSLATABLE WORDS ==========

    def get_untranslatable_words(self) -> dict[str, Any]:
        """Get untranslatable words from glossary."""
        glossary = self._glossaries.get("untranslatable", {})
        result: dict[str, Any] = glossary.get("untranslatable_words", {})
        return result

    def find_untranslatable_in_text(self, text: str) -> list[dict[str, Any]]:
        """Find untranslatable Persian words in text.

        Args:
            text: Persian text to analyze

        Returns:
            List of untranslatable words with translation strategies
        """
        found = []
        untranslatable = self.get_untranslatable_words()

        # Flatten categories
        all_words = {}
        for _category, items in untranslatable.items():
            if isinstance(items, dict):
                entries = items.get("entries", items)
                if isinstance(entries, dict):
                    all_words.update(entries)

        for _word_key, data in all_words.items():
            word = data.get("word_fa", "")
            if word and word in text:
                found.append(
                    {
                        "word": word,
                        "transliteration": data.get("transliteration", ""),
                        "approximate_translations": data.get("approximate_translations", []),
                        "why_untranslatable": data.get("why_untranslatable", ""),
                        "translation_strategies": data.get("translation_strategies", []),
                        "severity": data.get("severity", "major"),
                    }
                )

        return found

    # ========== COMPREHENSIVE CHECK ==========

    def analyze_text(self, text: str) -> dict[str, Any]:
        """Perform comprehensive analysis of Persian text for all traps.

        Args:
            text: Persian text to analyze

        Returns:
            Dictionary with all found traps categorized
        """
        return {
            "false_friends": self.find_false_friends_in_text(text),
            "taarof": self.find_taarof_in_text(text),
            "register_issues": self.check_register_consistency(text),
            "detected_register": self.detect_register(text),
            "compound_verb_errors": self.check_compound_verb_errors(text),
            "idioms": self.find_idioms_in_text(text),
            "untranslatable": self.find_untranslatable_in_text(text),
        }

    def get_translation_warnings(self, text: str) -> list[str]:
        """Get all translation warnings for a text.

        Args:
            text: Persian text to analyze

        Returns:
            List of warning strings
        """
        warnings = []
        analysis = self.analyze_text(text)

        # Critical warnings
        for ff in analysis["false_friends"]:
            if ff["severity"] == "critical":
                warnings.append(
                    f"CRITICAL: False friend '{ff['word']}' - "
                    f"Persian: {ff['persian_meaning']}, Arabic: {ff['arabic_meaning']}"
                )

        for taarof in analysis["taarof"]:
            if taarof["severity"] == "critical":
                warnings.append(
                    f"CRITICAL: Ta'arof phrase '{taarof['phrase']}' - "
                    f"means '{taarof['actual_meaning']}', NOT '{taarof['literal']}'"
                )

        for idiom in analysis["idioms"]:
            if idiom["severity"] in ["critical", "critical_if_literal"]:
                warnings.append(
                    f"CRITICAL: Idiom '{idiom['idiom']}' - DO NOT translate literally! "
                    f"Means: {idiom['meaning']}"
                )

        # Major warnings
        for cv_error in analysis["compound_verb_errors"]:
            warnings.append(
                f"Compound verb error: '{cv_error['error']}' should be '{cv_error['correct']}'"
            )

        for register in analysis["register_issues"]:
            warnings.append(
                f"Register mixing detected: {register['formal_found']} formal, "
                f"{register['colloquial_found']} colloquial markers"
            )

        for untrans in analysis["untranslatable"]:
            approx = ", ".join(untrans["approximate_translations"][:3])
            warnings.append(f"Untranslatable word: '{untrans['word']}' - consider: {approx}")

        return warnings

    def _format_persian_section(
        self,
        items: list[Any],
        header: str,
        limit: int,
        formatter: Callable[[Any], str | list[str]],
    ) -> list[str]:
        """Format a section of Persian prompt enrichment."""
        if not items:
            return []
        lines = [header]
        for item in items[:limit]:
            result = formatter(item)
            if isinstance(result, list):
                lines.extend(result)
            else:
                lines.append(result)
        return lines

    def get_prompt_enrichment(self, text: str) -> str:
        """Generate prompt enrichment for LLM based on text analysis."""
        analysis = self.analyze_text(text)
        sections = []

        sections.extend(
            self._format_persian_section(
                analysis["false_friends"],
                "## PERSIAN-ARABIC FALSE FRIENDS DETECTED:",
                5,
                lambda ff: f"- '{ff['word']}': Persian={ff['persian_meaning']}, Arabic={ff['arabic_meaning']} - {ff['translation_note']}",
            )
        )

        sections.extend(
            self._format_persian_section(
                analysis["taarof"],
                "\n## TA'AROF PHRASES (DO NOT translate literally!):",
                5,
                lambda t: f"- '{t['phrase']}': means '{t['actual_meaning']}', NOT '{t['literal']}'",
            )
        )

        def fmt_idiom(i: dict[str, Any]) -> list[str]:
            lines = [f"- '{i['idiom']}': means '{i['meaning']}'"]
            if i.get("english_equivalent"):
                lines.append(f"  English equivalent: {i['english_equivalent']}")
            return lines

        sections.extend(
            self._format_persian_section(
                analysis["idioms"],
                "\n## IDIOMS DETECTED (DO NOT translate literally!):",
                5,
                fmt_idiom,
            )
        )

        sections.extend(
            self._format_persian_section(
                analysis["compound_verb_errors"],
                "\n## COMPOUND VERB ERRORS:",
                3,
                lambda cv: f"- '{cv['error']}' → '{cv['correct']}': {cv['note']}",
            )
        )

        sections.extend(
            self._format_persian_section(
                analysis["untranslatable"],
                "\n## UNTRANSLATABLE WORDS (require special handling):",
                3,
                lambda u: f"- '{u['word']}' ({u['transliteration']}): approximate: {', '.join(u['approximate_translations'][:3])}",
            )
        )

        if analysis["register_issues"]:
            sections.append(f"\n## REGISTER: Detected as '{analysis['detected_register']}'")
            sections.append("WARNING: Inconsistent register mixing detected")

        return "\n".join(sections) if sections else ""
