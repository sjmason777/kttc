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

"""Rule-based error detection for translation quality (no AI/LLM required).

This module provides fast, deterministic checks for common translation errors:
- Number consistency (missing/extra numbers)
- Length anomalies (potential omissions/additions)
- Punctuation balance (quotes, brackets)
- Context preservation (negation, questions, exclamations)
- Named entity presence (capitalized words)

All checks are language-agnostic regular expression patterns.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field


class RuleBasedError(BaseModel):
    """Single error detected by rule-based checks.

    Similar to ErrorAnnotation but specifically for rule-based detection
    (no AI/LLM involved).
    """

    check_type: str = Field(..., description="Type of check that found this error")
    severity: str = Field(
        ...,
        description="Error severity: critical, major, or minor",
        pattern=r"^(critical|major|minor)$",
    )
    description: str = Field(..., description="Human-readable error description")
    details: dict[str, Any] = Field(default_factory=dict, description="Additional error details")


class ErrorDetector:
    """Rule-based error detector for translations (no AI/LLM required).

    Provides fast, deterministic checks for common translation errors.
    All checks are language-agnostic and run on CPU.

    Example:
        >>> detector = ErrorDetector()
        >>> errors = detector.detect_all_errors(
        ...     source="Price: $100",
        ...     translation="Price: $200"
        ... )
        >>> len(errors)
        1
        >>> errors[0].check_type
        'numbers_consistency'
    """

    # Length ratio thresholds for detecting anomalies
    LENGTH_MIN_RATIO = 0.5  # Too short (possible omission)
    LENGTH_MAX_RATIO = 2.0  # Too long (possible addition)

    def detect_all_errors(
        self,
        source: str,
        translation: str,
        reference: str | None = None,
    ) -> list[RuleBasedError]:
        """Run all error detection checks.

        Args:
            source: Source text
            translation: Translation to check
            reference: Optional reference translation

        Returns:
            List of detected errors

        Example:
            >>> detector = ErrorDetector()
            >>> errors = detector.detect_all_errors(
            ...     source="Hello world!",
            ...     translation="Hello"
            ... )
            >>> any(e.check_type == 'length_ratio' for e in errors)
            True
        """
        errors: list[RuleBasedError] = []

        # Check 1: Numbers consistency
        numbers_errors = self.check_numbers_consistency(source, translation)
        errors.extend(numbers_errors)

        # Check 2: Length ratio
        length_errors = self.check_length_ratio(source, translation)
        if length_errors:
            errors.append(length_errors)

        # Check 3: Punctuation balance
        punct_errors = self.check_punctuation_balance(source, translation)
        errors.extend(punct_errors)

        # Check 4: Context preservation
        context_errors = self.check_context_preservation(source, translation)
        errors.extend(context_errors)

        # Check 5: Named entities (capitalized words)
        entity_errors = self.check_named_entities(source, translation)
        if entity_errors:
            errors.append(entity_errors)

        return errors

    def check_numbers_consistency(self, source: str, translation: str) -> list[RuleBasedError]:
        """Check if all numbers from source appear in translation.

        Args:
            source: Source text
            translation: Translation text

        Returns:
            List of errors (empty if no issues)
        """
        errors: list[RuleBasedError] = []

        # Extract numbers from both texts
        source_numbers = set(re.findall(r"\d+(?:[.,]\d+)?", source))
        trans_numbers = set(re.findall(r"\d+(?:[.,]\d+)?", translation))

        # Check for missing numbers
        missing = source_numbers - trans_numbers
        if missing:
            errors.append(
                RuleBasedError(
                    check_type="numbers_consistency",
                    severity="critical",
                    description=f"Missing numbers in translation: {', '.join(sorted(missing))}",
                    details={"missing_numbers": sorted(list(missing))},
                )
            )

        # Check for extra numbers
        extra = trans_numbers - source_numbers
        if extra:
            errors.append(
                RuleBasedError(
                    check_type="numbers_consistency",
                    severity="major",
                    description=f"Extra numbers in translation: {', '.join(sorted(extra))}",
                    details={"extra_numbers": sorted(list(extra))},
                )
            )

        return errors

    def check_length_ratio(self, source: str, translation: str) -> RuleBasedError | None:
        """Check if translation length is within acceptable range.

        Args:
            source: Source text
            translation: Translation text

        Returns:
            Error if length ratio is suspicious, None otherwise
        """
        if not source or not translation:
            return None

        ratio = len(translation) / len(source)

        if ratio < self.LENGTH_MIN_RATIO:
            return RuleBasedError(
                check_type="length_ratio",
                severity="major",
                description=f"Translation too short (ratio: {ratio:.2f}) - possible omission",
                details={"ratio": ratio, "threshold": self.LENGTH_MIN_RATIO},
            )
        if ratio > self.LENGTH_MAX_RATIO:
            return RuleBasedError(
                check_type="length_ratio",
                severity="major",
                description=f"Translation too long (ratio: {ratio:.2f}) - possible addition",
                details={"ratio": ratio, "threshold": self.LENGTH_MAX_RATIO},
            )

        return None

    def check_punctuation_balance(self, source: str, translation: str) -> list[RuleBasedError]:
        """Check if paired punctuation marks are balanced.

        Args:
            source: Source text
            translation: Translation text

        Returns:
            List of errors for unbalanced punctuation
        """
        errors: list[RuleBasedError] = []

        # Paired punctuation marks to check
        pairs = {
            "(": ")",
            "[": "]",
            "{": "}",
            '"': '"',
            "'": "'",
            "«": "»",
            "\u201c": "\u201d",  # " and "
            "\u2018": "\u2019",  # ' and '
        }

        for open_mark, close_mark in pairs.items():
            source_open = source.count(open_mark)
            source_close = source.count(close_mark)
            trans_open = translation.count(open_mark)
            trans_close = translation.count(close_mark)

            # Check if source is balanced and translation differs
            if source_open == source_close and (
                trans_open != trans_close or trans_open != source_open
            ):
                errors.append(
                    RuleBasedError(
                        check_type="punctuation_balance",
                        severity="minor",
                        description=f"Unbalanced punctuation: {open_mark}{close_mark}",
                        details={
                            "mark": f"{open_mark}{close_mark}",
                            "source_count": source_open,
                            "translation_count": trans_open,
                        },
                    )
                )

        return errors

    def check_context_preservation(self, source: str, translation: str) -> list[RuleBasedError]:
        """Check if important context markers are preserved.

        Checks:
        - Question marks (?)
        - Exclamation marks (!)
        - Negation words (not, no, never, etc.)

        Args:
            source: Source text
            translation: Translation text

        Returns:
            List of context preservation errors
        """
        errors: list[RuleBasedError] = []

        # Check question marks
        if source.strip().endswith("?") and not translation.strip().endswith("?"):
            errors.append(
                RuleBasedError(
                    check_type="context_preservation",
                    severity="major",
                    description="Question mark missing in translation",
                    details={"marker": "?"},
                )
            )

        # Check exclamation marks
        if source.strip().endswith("!") and not translation.strip().endswith("!"):
            errors.append(
                RuleBasedError(
                    check_type="context_preservation",
                    severity="minor",
                    description="Exclamation mark missing in translation",
                    details={"marker": "!"},
                )
            )

        # Check negation (English)
        negation_words_en = r"\b(not|no|never|neither|none|nobody|nothing|nowhere|n't)\b"
        source_has_negation = bool(re.search(negation_words_en, source, re.IGNORECASE))

        # Common negation patterns in other languages
        negation_words_ru = r"\b(не|нет|ни|никто|ничто|никогда|нигде)\b"
        negation_words_es = r"\b(no|nunca|nada|nadie|ningún|jamás)\b"
        negation_words_fr = r"\b(ne|pas|non|jamais|rien|personne)\b"
        negation_words_de = r"\b(nicht|kein|keine|niemals|niemand|nichts)\b"

        negation_patterns = [
            negation_words_ru,
            negation_words_es,
            negation_words_fr,
            negation_words_de,
        ]

        trans_has_negation = any(
            re.search(pattern, translation, re.IGNORECASE) for pattern in negation_patterns
        )

        # Only flag if source has negation and translation clearly doesn't
        if source_has_negation and not trans_has_negation and len(translation) > 10:
            errors.append(
                RuleBasedError(
                    check_type="context_preservation",
                    severity="critical",
                    description="Negation present in source but appears missing in translation",
                    details={"negation_type": "detected in source"},
                )
            )

        return errors

    def check_named_entities(self, source: str, translation: str) -> RuleBasedError | None:
        """Check if capitalized words (potential names/places) are preserved.

        This is a simple heuristic check for proper nouns.

        Args:
            source: Source text
            translation: Translation text

        Returns:
            Error if significant named entities appear missing, None otherwise
        """
        # Find capitalized words (not at sentence start)
        # Pattern: word that starts with capital letter, not after sentence-ending punctuation
        source_caps = set(re.findall(r"(?<!^)(?<![.!?]\s)\b([A-Z][a-z]{2,})\b", source))

        # Remove common words that might be capitalized
        common_caps = {"The", "A", "An", "In", "On", "At", "To", "For", "Of", "With", "By"}
        source_caps = source_caps - common_caps

        if not source_caps:
            return None

        # Check if these appear in translation (allow case variations)
        trans_lower = translation.lower()
        missing_entities = [cap for cap in source_caps if cap.lower() not in trans_lower]

        if missing_entities and len(missing_entities) >= 2:
            return RuleBasedError(
                check_type="named_entities",
                severity="major",
                description=f"Potential named entities missing: {', '.join(sorted(missing_entities)[:3])}",
                details={
                    "missing_count": len(missing_entities),
                    "examples": sorted(missing_entities)[:5],
                },
            )

        return None

    def get_severity_counts(self, errors: list[RuleBasedError]) -> dict[str, int]:
        """Count errors by severity level.

        Args:
            errors: List of detected errors

        Returns:
            Dictionary with counts per severity level
        """
        counts = {"critical": 0, "major": 0, "minor": 0}

        for error in errors:
            counts[error.severity] += 1

        return counts

    def calculate_rule_based_score(self, errors: list[RuleBasedError]) -> float:
        """Calculate quality score based on rule-based errors.

        Scoring:
        - Start at 100
        - Critical error: -20 points
        - Major error: -10 points
        - Minor error: -5 points

        Args:
            errors: List of detected errors

        Returns:
            Score 0-100 (higher is better)
        """
        score = 100.0

        for error in errors:
            if error.severity == "critical":
                score -= 20.0
            elif error.severity == "major":
                score -= 10.0
            elif error.severity == "minor":
                score -= 5.0

        return max(0.0, score)
