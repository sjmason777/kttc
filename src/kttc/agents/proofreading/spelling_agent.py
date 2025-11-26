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

"""Spelling Agent for orthography checking.

Specialized agent for checking spelling errors using:
- School curriculum orthography rules
- Language-specific patterns (e.g., Russian НЕ rules, English homophones)
- LLM fallback for context-dependent spelling

Used in self-check mode for proofreading without translation.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from kttc.core import ErrorAnnotation, ErrorSeverity, TranslationTask
from kttc.llm import BaseLLMProvider

logger = logging.getLogger(__name__)


# Language-specific spelling patterns
SPELLING_PATTERNS: dict[str, list[dict[str, Any]]] = {
    "ru": [
        {
            # Only match when НЕ is written together with the verb (no space)
            "pattern": r"\bне(хочу|могу|буду|знаю|понимаю|видел|слышал)\b",
            "correct": r"не \1",
            "description": "НЕ с глаголами пишется раздельно",
            "severity": "major",
        },
        {
            "pattern": r"\b(какой|кто|что|где|куда|когда|как)\s+(то|либо|нибудь)\b",
            "correct": r"\1-\2",
            "description": "Дефис в неопределённых местоимениях",
            "severity": "major",
        },
        {
            "pattern": r"\bчто\s+бы\s+(?!ни|такое|там)",
            "description": "Возможно нужно 'чтобы' (слитно)",
            "severity": "major",
        },
        {
            "pattern": r"\bпо\s+этому\s+(?!мост|повод|вопрос|делу|случаю|адрес|пути|принципу)",
            "description": "Возможно нужно 'поэтому' (слитно)",
            "severity": "major",
        },
    ],
    "en": [
        {
            "pattern": r"\b(should|could|would)\s+of\b",
            "correct": r"\1 have",
            "description": "Use 'have' not 'of' after modal verbs",
            "severity": "major",
        },
        {
            "pattern": r"\btheir\s+(is|are|was|were)\b",
            "description": "Check: did you mean 'there' (location)?",
            "severity": "major",
        },
        {
            "pattern": r"\byour\s+(welcome|right|wrong)\b",
            "description": "Check: did you mean 'you're' (you are)?",
            "severity": "major",
        },
        {
            "pattern": r"\bits\s+(a|an|the|going|been|not)\b",
            "description": "Check: did you mean 'it's' (it is)?",
            "severity": "major",
        },
    ],
    "zh": [
        {
            "pattern": r"[,.:;!?]",
            "description": "中文应使用全角标点符号",
            "severity": "minor",
        },
    ],
    "fa": [
        {
            "pattern": r"می\s+[آا-ی]+",
            "description": "پیشوند می- با نیم‌فاصله نوشته می‌شود",
            "severity": "major",
        },
    ],
}


class SpellingAgent:
    """Agent for checking spelling and orthography.

    This agent specializes in language-specific spelling rules,
    including patterns that are often confused by native speakers.

    Example:
        >>> agent = SpellingAgent(language="ru")
        >>> text = "Нехочу идти, какой то человек пришёл"
        >>> errors = await agent.check(text)
        >>> print(f"Found {len(errors)} spelling errors")
    """

    def __init__(
        self,
        llm_provider: BaseLLMProvider | None = None,
        language: str = "en",
        use_patterns: bool = True,
        use_school_rules: bool = True,
    ):
        """Initialize SpellingAgent.

        Args:
            llm_provider: LLM provider for context-aware checking (optional)
            language: Language code for the text being checked
            use_patterns: Whether to use regex patterns for common errors
            use_school_rules: Whether to use school curriculum glossaries
        """
        self.llm_provider = llm_provider
        self.language = language
        self.use_patterns = use_patterns
        self.use_school_rules = use_school_rules
        self._school_rules: dict[str, Any] | None = None

    @property
    def category(self) -> str:
        """Get error category this agent checks."""
        return "spelling"

    def _load_school_rules(self) -> dict[str, Any]:
        """Load school curriculum spelling rules for the language."""
        if self._school_rules is not None:
            return self._school_rules

        current_file = Path(__file__)
        project_root = current_file.parent.parent.parent.parent.parent
        glossaries_dir = project_root / "glossaries" / self.language / "school_curriculum"

        self._school_rules = {"spelling_rules": {}, "common_mistakes": {}}

        if not glossaries_dir.exists():
            return self._school_rules

        for json_file in glossaries_dir.glob("*.json"):
            try:
                with open(json_file, encoding="utf-8") as f:
                    data = json.load(f)

                if "spelling_rules" in data:
                    self._school_rules["spelling_rules"].update(data["spelling_rules"])
                if "common_mistakes" in data:
                    self._school_rules["common_mistakes"].update(data["common_mistakes"])

            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Failed to load {json_file}: {e}")

        return self._school_rules

    def _check_with_patterns(self, text: str) -> list[ErrorAnnotation]:
        """Check text using regex patterns for common spelling errors.

        Args:
            text: Text to check

        Returns:
            List of spelling errors found
        """
        errors: list[ErrorAnnotation] = []
        patterns = SPELLING_PATTERNS.get(self.language, [])

        for pattern_def in patterns:
            pattern = pattern_def["pattern"]
            try:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    severity_str = pattern_def.get("severity", "major")
                    severity = (
                        ErrorSeverity.CRITICAL
                        if severity_str == "critical"
                        else ErrorSeverity.MINOR
                        if severity_str == "minor"
                        else ErrorSeverity.MAJOR
                    )

                    suggestion = ""
                    if "correct" in pattern_def:
                        # Use match groups to construct the corrected string
                        try:
                            suggestion = re.sub(
                                pattern, pattern_def["correct"], match.group(), flags=re.IGNORECASE
                            )
                            # If substitution failed (same result), try using groups directly
                            if suggestion.lower() == match.group().lower():
                                groups = match.groups()
                                if groups:
                                    # For patterns like "не(хочу|...)" -> "не хочу"
                                    suggestion = f"не {groups[0]}"
                        except Exception:
                            suggestion = ""

                    error_text = match.group()
                    errors.append(
                        ErrorAnnotation(
                            category="fluency",
                            subcategory="spelling",
                            severity=severity,
                            location=(match.start(), match.end()),
                            description=f"{pattern_def['description']}: '{error_text}'",
                            suggestion=suggestion if suggestion else None,
                        )
                    )
            except re.error as e:
                logger.warning(f"Invalid regex pattern: {pattern} - {e}")

        return errors

    def _parse_severity(self, severity_str: str) -> ErrorSeverity:
        """Parse severity string to ErrorSeverity enum."""
        if severity_str == "critical":
            return ErrorSeverity.CRITICAL
        if severity_str == "minor":
            return ErrorSeverity.MINOR
        return ErrorSeverity.MAJOR

    def _check_school_spelling_rules(
        self, text: str, spelling_rules: dict[str, Any]
    ) -> list[ErrorAnnotation]:
        """Check text against school spelling rules."""
        errors: list[ErrorAnnotation] = []
        text_lower = text.lower()

        for rule_data in spelling_rules.values():
            if not isinstance(rule_data, dict):
                continue
            examples = rule_data.get("examples", {})
            incorrect = examples.get("incorrect", [])
            correct = examples.get("correct", [])

            for i, wrong in enumerate(incorrect):
                if not isinstance(wrong, str) or wrong.lower() not in text_lower:
                    continue
                pos = text_lower.find(wrong.lower())
                actual_text = text[pos : pos + len(wrong)]
                suggestion = correct[i] if i < len(correct) else ""
                severity = self._parse_severity(rule_data.get("severity", "major"))
                errors.append(
                    ErrorAnnotation(
                        category="fluency",
                        subcategory="spelling",
                        severity=severity,
                        location=(pos, pos + len(wrong)),
                        description=f"{rule_data.get('name', 'Spelling error')}: '{actual_text}' → '{suggestion}'",
                        suggestion=suggestion if isinstance(suggestion, str) else None,
                    )
                )
        return errors

    def _check_school_common_mistakes(
        self, text: str, common_mistakes: dict[str, Any]
    ) -> list[ErrorAnnotation]:
        """Check text for common spelling mistakes."""
        errors: list[ErrorAnnotation] = []
        text_lower = text.lower()

        for mistake_data in common_mistakes.values():
            if not isinstance(mistake_data, dict):
                continue
            for wrong in mistake_data.get("examples", []):
                if not isinstance(wrong, str) or wrong.lower() not in text_lower:
                    continue
                pos = text_lower.find(wrong.lower())
                correct_forms = mistake_data.get("correct_forms", [])
                suggestion = correct_forms[0] if correct_forms else ""
                actual_text = text[pos : pos + len(wrong)]
                errors.append(
                    ErrorAnnotation(
                        category="fluency",
                        subcategory="spelling",
                        severity=ErrorSeverity.MAJOR,
                        location=(pos, pos + len(wrong)),
                        description=f"{mistake_data.get('description', 'Common spelling mistake')}: '{actual_text}'",
                        suggestion=suggestion or None,
                    )
                )
        return errors

    def _check_with_school_rules(self, text: str) -> list[ErrorAnnotation]:
        """Check text against school curriculum spelling rules."""
        rules = self._load_school_rules()
        errors = self._check_school_spelling_rules(text, rules.get("spelling_rules", {}))
        errors.extend(self._check_school_common_mistakes(text, rules.get("common_mistakes", {})))
        return errors

    def check(self, text: str) -> list[ErrorAnnotation]:
        """Check text for spelling errors.

        Args:
            text: Text to check

        Returns:
            List of spelling error annotations
        """
        all_errors: list[ErrorAnnotation] = []
        seen_positions: set[tuple[int, int]] = set()

        # 1. Check with regex patterns (fast)
        if self.use_patterns:
            pattern_errors = self._check_with_patterns(text)
            for err in pattern_errors:
                if err.location not in seen_positions:
                    all_errors.append(err)
                    seen_positions.add(err.location)

        # 2. Check with school rules
        if self.use_school_rules:
            school_errors = self._check_with_school_rules(text)
            for err in school_errors:
                if err.location not in seen_positions:
                    all_errors.append(err)
                    seen_positions.add(err.location)

        return all_errors

    async def evaluate(self, task: TranslationTask) -> list[ErrorAnnotation]:
        """Evaluate task in proofreading mode (compatibility method).

        Args:
            task: Translation task (uses translation field as text to check)

        Returns:
            List of error annotations
        """
        text_to_check = task.translation or task.source_text
        self.language = task.target_lang or task.source_lang

        return self.check(text_to_check)
