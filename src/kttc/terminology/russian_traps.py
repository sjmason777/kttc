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

"""Russian Language Traps Validator.

Validates Russian-specific linguistic traps that are difficult for translation:
- Homonyms and paronyms (context-dependent meaning)
- Position verbs semantics (stand/lie/sit/hang)
- Idioms and phraseological units (cannot be translated literally)
- Untranslatable words (cultural concepts)
- Stress patterns (meaning changes with stress)

These checks are automatically enabled when target_lang='ru'.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, cast

logger = logging.getLogger(__name__)


class RussianTrapsValidator:
    """Validator for Russian language traps and peculiarities.

    Loads glossaries for:
    - homonyms_paronyms_ru.json
    - position_verbs_ru.json
    - idioms_ru.json
    - untranslatable_ru.json
    - stress_patterns_ru.json

    Example:
        >>> validator = RussianTrapsValidator()
        >>> homonyms = validator.get_homonyms_in_text("Я нашёл ключ от замка")
        >>> # Returns: [{'word': 'ключ', 'meanings': [...], 'context_needed': True}]
    """

    def __init__(self) -> None:
        """Initialize validator and load all glossaries."""
        self._glossaries: dict[str, dict[str, Any]] = {}
        self._load_glossaries()

    def _get_glossary_path(self) -> Path:
        """Get path to Russian glossaries directory."""
        current_file = Path(__file__)
        # Go up to src/kttc, then to glossaries/ru
        project_root = current_file.parent.parent.parent.parent
        return project_root / "glossaries" / "ru"

    def _load_glossaries(self) -> None:
        """Load all Russian trap glossaries."""
        glossary_path = self._get_glossary_path()

        glossary_files = [
            "homonyms_paronyms_ru.json",
            "position_verbs_ru.json",
            "idioms_ru.json",
            "untranslatable_ru.json",
            "stress_patterns_ru.json",
        ]

        for filename in glossary_files:
            filepath = glossary_path / filename
            if filepath.exists():
                try:
                    with open(filepath, encoding="utf-8") as f:
                        key = filename.replace("_ru.json", "").replace(".json", "")
                        self._glossaries[key] = json.load(f)
                        logger.debug(f"Loaded glossary: {filename}")
                except (json.JSONDecodeError, OSError) as e:
                    logger.warning(f"Failed to load {filename}: {e}")
            else:
                logger.debug(f"Glossary not found: {filepath}")

    def is_available(self) -> bool:
        """Check if glossaries are loaded."""
        return len(self._glossaries) > 0

    # ========== HOMONYMS & PARONYMS ==========

    def get_homonyms(self) -> dict[str, Any]:
        """Get all homonyms from glossary."""
        glossary = self._glossaries.get("homonyms_paronyms", {})
        result = glossary.get("homonyms", {})
        if isinstance(result, dict):
            return cast(dict[str, Any], result.get("entries", {}))
        return {}

    def get_paronyms(self) -> dict[str, Any]:
        """Get all paronyms from glossary."""
        glossary = self._glossaries.get("homonyms_paronyms", {})
        result = glossary.get("paronyms", {})
        if isinstance(result, dict):
            return cast(dict[str, Any], result.get("entries", {}))
        return {}

    def find_homonyms_in_text(self, text: str) -> list[dict[str, Any]]:
        """Find potential homonyms in text that may need context disambiguation.

        Args:
            text: Russian text to analyze

        Returns:
            List of homonyms found with their meanings and severity
        """
        found = []
        homonyms = self.get_homonyms()
        text_lower = text.lower()

        for word, data in homonyms.items():
            # Check for word presence (as whole word)
            pattern = rf"\b{re.escape(word)}\b"
            if re.search(pattern, text_lower):
                found.append(
                    {
                        "word": word,
                        "meanings": data.get("meanings", []),
                        "severity": data.get("severity", "minor"),
                        "stress_difference": data.get("stress_difference", False),
                        "translation_warning": "Multiple meanings - context analysis required",
                    }
                )

        return found

    def find_paronyms_in_text(self, text: str) -> list[dict[str, Any]]:
        """Find potential paronym errors in text.

        Args:
            text: Russian text to analyze

        Returns:
            List of paronyms found that may be confused
        """
        found = []
        paronyms = self.get_paronyms()
        text_lower = text.lower()

        for _pair_name, data in paronyms.items():
            pair = data.get("pair", [])
            for word in pair:
                pattern = rf"\b{re.escape(word)}\b"
                if re.search(pattern, text_lower):
                    found.append(
                        {
                            "word": word,
                            "pair": pair,
                            "definitions": data.get("definitions", {}),
                            "common_error": data.get("common_error", False),
                            "severity": data.get("severity", "minor"),
                        }
                    )

        return found

    # ========== POSITION VERBS ==========

    def get_position_verbs(self) -> dict[str, Any]:
        """Get position verbs semantics."""
        glossary = self._glossaries.get("position_verbs", {})
        return cast(dict[str, Any], glossary.get("position_verbs", {}))

    def check_position_verb_usage(self, text: str) -> list[dict[str, Any]]:
        """Check for potential position verb errors (stand/lie/sit/hang).

        Args:
            text: Russian text to analyze

        Returns:
            List of position verb usages that may need review
        """
        found = []
        text_lower = text.lower()

        # Common error patterns
        error_patterns = [
            {
                "pattern": r"\bкнига\s+стоит\b",
                "correct": "книга лежит",
                "reason": "Flat objects lie, not stand",
            },
            {
                "pattern": r"\bвилка\s+стоит\b",
                "correct": "вилка лежит",
                "reason": "Cutlery lies, not stands",
            },
            {
                "pattern": r"\bкартина\s+стоит\s+на\s+стене\b",
                "correct": "картина висит на стене",
                "reason": "Objects on vertical surfaces hang",
            },
            {
                "pattern": r"\bптица\s+стоит\b",
                "correct": "птица сидит",
                "reason": "Birds sit, not stand",
            },
        ]

        for error in error_patterns:
            if re.search(error["pattern"], text_lower):
                found.append(
                    {
                        "found": error["pattern"],
                        "correct": error["correct"],
                        "reason": error["reason"],
                        "severity": "major",
                    }
                )

        return found

    def get_position_verb_paradoxes(self) -> dict[str, Any]:
        """Get famous Russian position verb paradoxes."""
        glossary = self._glossaries.get("position_verbs", {})
        return cast(dict[str, Any], glossary.get("paradoxes", {}))

    # ========== IDIOMS ==========

    def get_idioms(self) -> dict[str, Any]:
        """Get all idioms from glossary."""
        glossary = self._glossaries.get("idioms", {})
        return cast(dict[str, Any], glossary.get("idioms", {}))

    def find_idioms_in_text(self, text: str) -> list[dict[str, Any]]:
        """Find idioms in text that cannot be translated literally.

        Args:
            text: Russian text to analyze

        Returns:
            List of idioms found with translation warnings
        """
        found = []
        idioms_data = self.get_idioms()
        text_lower = text.lower()

        # Flatten idiom categories
        all_idioms = {}
        for _category, items in idioms_data.items():
            if isinstance(items, dict):
                all_idioms.update(items)

        # Search for idioms
        for idiom_key, data in all_idioms.items():
            # Convert key to search pattern (replace underscores with spaces)
            search_pattern = idiom_key.replace("_", " ")
            if search_pattern in text_lower:
                found.append(
                    {
                        "idiom": search_pattern,
                        "literal": data.get("literal", ""),
                        "meaning": data.get("meaning", ""),
                        "english_equivalent": data.get("english_equivalent", ""),
                        "severity": (
                            "critical" if data.get("severity") == "critical_if_literal" else "major"
                        ),
                        "warning": "DO NOT translate literally!",
                    }
                )

        return found

    # ========== UNTRANSLATABLE WORDS ==========

    def get_untranslatable_words(self) -> dict[str, Any]:
        """Get untranslatable words from glossary."""
        glossary = self._glossaries.get("untranslatable", {})
        return cast(dict[str, Any], glossary.get("untranslatable_words", {}))

    def find_untranslatable_in_text(self, text: str) -> list[dict[str, Any]]:
        """Find untranslatable Russian words in text.

        Args:
            text: Russian text to analyze

        Returns:
            List of untranslatable words with translation strategies
        """
        found = []
        untranslatable = self.get_untranslatable_words()
        text_lower = text.lower()

        # Flatten categories
        all_words = {}
        for _category, items in untranslatable.items():
            if isinstance(items, dict):
                all_words.update(items)

        for word, data in all_words.items():
            pattern = rf"\b{re.escape(word)}\b"
            if re.search(pattern, text_lower):
                found.append(
                    {
                        "word": word,
                        "approximate_translations": data.get("approximate_translations", []),
                        "why_untranslatable": data.get("why_untranslatable", ""),
                        "translation_strategies": data.get("translation_strategies", []),
                        "severity": data.get("severity", "major"),
                    }
                )

        return found

    # ========== STRESS PATTERNS ==========

    def get_stress_homographs(self) -> dict[str, Any]:
        """Get stress-dependent homographs."""
        glossary = self._glossaries.get("stress_patterns", {})
        result = glossary.get("stress_homographs", {})
        if isinstance(result, dict):
            return cast(dict[str, Any], result.get("critical_pairs", {}))
        return {}

    def get_common_stress_errors(self) -> dict[str, Any]:
        """Get common stress errors."""
        glossary = self._glossaries.get("stress_patterns", {})
        result = glossary.get("common_stress_errors", {})
        if isinstance(result, dict):
            return cast(dict[str, Any], result.get("frequently_mispronounced", {}))
        return {}

    def find_stress_homographs_in_text(self, text: str) -> list[dict[str, Any]]:
        """Find words where stress changes meaning.

        Args:
            text: Russian text to analyze

        Returns:
            List of stress homographs requiring context
        """
        found = []
        homographs = self.get_stress_homographs()
        text_lower = text.lower()

        for word, data in homographs.items():
            pattern = rf"\b{re.escape(word)}\b"
            if re.search(pattern, text_lower):
                found.append(
                    {
                        "word": word,
                        "variants": data.get("variants", []),
                        "severity": data.get("severity", "major"),
                        "warning": "Stress determines meaning - context analysis required",
                    }
                )

        return found

    # ========== COMPREHENSIVE CHECK ==========

    def analyze_text(self, text: str) -> dict[str, Any]:
        """Perform comprehensive analysis of Russian text for all traps.

        Args:
            text: Russian text to analyze

        Returns:
            Dictionary with all found traps categorized
        """
        return {
            "homonyms": self.find_homonyms_in_text(text),
            "paronyms": self.find_paronyms_in_text(text),
            "position_verbs": self.check_position_verb_usage(text),
            "idioms": self.find_idioms_in_text(text),
            "untranslatable": self.find_untranslatable_in_text(text),
            "stress_homographs": self.find_stress_homographs_in_text(text),
        }

    def get_translation_warnings(self, text: str) -> list[str]:
        """Get all translation warnings for a text.

        Args:
            text: Russian text to analyze

        Returns:
            List of warning strings
        """
        warnings = []
        analysis = self.analyze_text(text)

        # Critical warnings
        for idiom in analysis["idioms"]:
            if idiom["severity"] == "critical":
                warnings.append(f"CRITICAL: Idiom '{idiom['idiom']}' found - {idiom['warning']}")

        for homonym in analysis["homonyms"]:
            if homonym["severity"] == "critical":
                warnings.append(
                    f"CRITICAL: Homonym '{homonym['word']}' has multiple meanings - verify context"
                )

        # Major warnings
        for paronym in analysis["paronyms"]:
            if paronym.get("common_error"):
                warnings.append(
                    f"Check paronym: '{paronym['word']}' - common confusion with {paronym['pair']}"
                )

        for position in analysis["position_verbs"]:
            warnings.append(
                f"Position verb error: {position['reason']} - use '{position['correct']}'"
            )

        for untrans in analysis["untranslatable"]:
            warnings.append(
                f"Untranslatable word: '{untrans['word']}' - consider: {', '.join(untrans['approximate_translations'][:3])}"
            )

        return warnings

    def get_prompt_enrichment(self, text: str) -> str:
        """Generate prompt enrichment for LLM based on text analysis.

        Args:
            text: Russian text to analyze

        Returns:
            Prompt section with Russian-specific guidance
        """
        analysis = self.analyze_text(text)
        sections: list[str] = []

        sections.extend(self._format_homonyms_section(analysis["homonyms"]))
        sections.extend(self._format_idioms_section(analysis["idioms"]))
        sections.extend(self._format_untranslatable_section(analysis["untranslatable"]))
        sections.extend(self._format_stress_section(analysis["stress_homographs"]))

        return "\n".join(sections) if sections else ""

    def _format_homonyms_section(self, homonyms: list[dict[str, Any]]) -> list[str]:
        """Format homonyms section for prompt enrichment."""
        if not homonyms:
            return []
        lines = ["## HOMONYMS DETECTED (context-dependent meaning):"]
        for h in homonyms[:5]:
            meanings_str = ", ".join(
                [f"{m['meaning']} ({m['english']})" for m in h.get("meanings", [])[:3]]
            )
            lines.append(f"- '{h['word']}': {meanings_str}")
        return lines

    def _format_idioms_section(self, idioms: list[dict[str, Any]]) -> list[str]:
        """Format idioms section for prompt enrichment."""
        if not idioms:
            return []
        lines = ["\n## IDIOMS DETECTED (DO NOT translate literally!):"]
        for i in idioms[:5]:
            lines.append(f"- '{i['idiom']}': means '{i['meaning']}', NOT '{i['literal']}'")
            if i.get("english_equivalent"):
                lines.append(f"  English equivalent: {i['english_equivalent']}")
        return lines

    def _format_untranslatable_section(self, untranslatable: list[dict[str, Any]]) -> list[str]:
        """Format untranslatable words section for prompt enrichment."""
        if not untranslatable:
            return []
        lines = ["\n## UNTRANSLATABLE WORDS (require special handling):"]
        for u in untranslatable[:3]:
            lines.append(f"- '{u['word']}': {u['why_untranslatable'][:100]}...")
        return lines

    def _format_stress_section(self, stress_homographs: list[dict[str, Any]]) -> list[str]:
        """Format stress-dependent meanings section for prompt enrichment."""
        if not stress_homographs:
            return []
        lines = ["\n## STRESS-DEPENDENT MEANINGS (verify context):"]
        for s in stress_homographs[:3]:
            variants = s.get("variants", [])
            if variants:
                var_str = "; ".join([f"{v['stress']} = {v['meaning']}" for v in variants[:2]])
                lines.append(f"- '{s['word']}': {var_str}")
        return lines
