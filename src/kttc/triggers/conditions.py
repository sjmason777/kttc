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

"""Condition evaluators for QA triggers.

Provides extensible condition system for trigger evaluation.
"""

from __future__ import annotations

import fnmatch
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any


class BaseCondition(ABC):
    """Abstract base class for trigger conditions.

    Subclasses must implement the evaluate method.
    """

    @abstractmethod
    def evaluate(self, context: dict[str, Any]) -> bool:
        """Evaluate condition against context.

        Args:
            context: Dictionary with event context data

        Returns:
            True if condition is met
        """
        ...

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> BaseCondition:
        """Create condition from configuration dictionary.

        Args:
            config: Condition configuration

        Returns:
            Appropriate condition instance

        Raises:
            ValueError: If condition type is unknown
        """
        condition_type = config.get("type", "")

        if condition_type == "file_pattern":
            return FilePatternCondition(patterns=config.get("patterns", []))
        if condition_type == "score_threshold":
            return ScoreThresholdCondition(
                metric=config.get("metric", "mqm"),
                min_score=config.get("min_score"),
                max_score=config.get("max_score"),
            )
        if condition_type == "time_based":
            return TimeBasedCondition(schedule=config.get("schedule", ""))
        if condition_type == "language_pair":
            return LanguagePairCondition(
                source_langs=config.get("source_langs", []),
                target_langs=config.get("target_langs", []),
            )
        if condition_type == "error_count":
            return ErrorCountCondition(
                min_errors=config.get("min_errors"),
                max_errors=config.get("max_errors"),
            )
        raise ValueError(f"Unknown condition type: {condition_type}")


class FilePatternCondition(BaseCondition):
    """Match files against glob patterns.

    Example:
        >>> condition = FilePatternCondition(patterns=["*.xliff", "*.po"])
        >>> condition.evaluate({"file_path": "translations.xliff"})
        True
    """

    def __init__(self, patterns: list[str]):
        """Initialize with file patterns.

        Args:
            patterns: List of glob patterns to match
        """
        self.patterns = patterns

    def evaluate(self, context: dict[str, Any]) -> bool:
        """Check if file matches any pattern.

        Args:
            context: Must contain 'file_path' key

        Returns:
            True if file matches any pattern
        """
        file_path = context.get("file_path", "")
        if not file_path:
            return False

        # Extract filename from path
        filename = file_path.split("/")[-1].split("\\")[-1]

        for pattern in self.patterns:
            if fnmatch.fnmatch(filename, pattern):
                return True
            # Also check full path
            if fnmatch.fnmatch(file_path, pattern):
                return True

        return False


class ScoreThresholdCondition(BaseCondition):
    """Check if score is within threshold.

    Example:
        >>> condition = ScoreThresholdCondition(metric="mqm", min_score=85.0)
        >>> condition.evaluate({"mqm_score": 82.0})
        False  # Below threshold
    """

    def __init__(
        self,
        metric: str = "mqm",
        min_score: float | None = None,
        max_score: float | None = None,
    ):
        """Initialize with threshold values.

        Args:
            metric: Score metric name (mqm, bleu, ter, chrf)
            min_score: Minimum acceptable score
            max_score: Maximum acceptable score
        """
        self.metric = metric
        self.min_score = min_score
        self.max_score = max_score

    def evaluate(self, context: dict[str, Any]) -> bool:
        """Check if score violates threshold.

        Args:
            context: Must contain score for the metric

        Returns:
            True if score violates threshold (triggers alert)
        """
        # Look for score in context
        score_key = f"{self.metric}_score"
        score = context.get(score_key) or context.get(self.metric)

        if score is None:
            return False

        try:
            score = float(score)
        except (TypeError, ValueError):
            return False

        # Check violations
        if self.min_score is not None and score < self.min_score:
            return True

        if self.max_score is not None and score > self.max_score:
            return True

        return False


class TimeBasedCondition(BaseCondition):
    """Check if current time matches cron-like schedule.

    Supports simplified cron format: minute hour day_of_month month day_of_week

    Example:
        >>> condition = TimeBasedCondition(schedule="0 9 * * *")  # 9 AM daily
        >>> condition.evaluate({"current_time": datetime(2025, 1, 15, 9, 0)})
        True
    """

    def __init__(self, schedule: str):
        """Initialize with cron schedule.

        Args:
            schedule: Cron-like schedule string
        """
        self.schedule = schedule
        self._parse_schedule()

    def _parse_schedule(self) -> None:
        """Parse cron schedule into components."""
        parts = self.schedule.split()
        if len(parts) != 5:
            # Default to always match
            self.minute = "*"
            self.hour = "*"
            self.day = "*"
            self.month = "*"
            self.weekday = "*"
            return

        self.minute, self.hour, self.day, self.month, self.weekday = parts

    def _matches_field(self, field: str, value: int) -> bool:
        """Check if value matches cron field.

        Args:
            field: Cron field pattern
            value: Current value to check

        Returns:
            True if matches
        """
        if field == "*":
            return True

        # Handle ranges (e.g., "1-5")
        if "-" in field:
            start, end = map(int, field.split("-"))
            return start <= value <= end

        # Handle lists (e.g., "1,3,5")
        if "," in field:
            values = [int(v) for v in field.split(",")]
            return value in values

        # Handle step (e.g., "*/5")
        if "/" in field:
            base, step_str = field.split("/")
            step_int = int(step_str)
            if base == "*":
                return value % step_int == 0
            return (value - int(base)) % step_int == 0

        # Direct match
        return int(field) == value

    def evaluate(self, context: dict[str, Any]) -> bool:
        """Check if current time matches schedule.

        Args:
            context: May contain 'current_time' datetime

        Returns:
            True if current time matches schedule
        """
        now = context.get("current_time", datetime.now())

        if not isinstance(now, datetime):
            return False

        return (
            self._matches_field(self.minute, now.minute)
            and self._matches_field(self.hour, now.hour)
            and self._matches_field(self.day, now.day)
            and self._matches_field(self.month, now.month)
            and self._matches_field(self.weekday, now.weekday())
        )


class LanguagePairCondition(BaseCondition):
    """Match specific language pairs.

    Example:
        >>> condition = LanguagePairCondition(source_langs=["en"], target_langs=["ru", "de"])
        >>> condition.evaluate({"source_lang": "en", "target_lang": "ru"})
        True
    """

    def __init__(
        self,
        source_langs: list[str] | None = None,
        target_langs: list[str] | None = None,
    ):
        """Initialize with language filters.

        Args:
            source_langs: Allowed source languages (empty = any)
            target_langs: Allowed target languages (empty = any)
        """
        self.source_langs = [lang.lower() for lang in (source_langs or [])]
        self.target_langs = [lang.lower() for lang in (target_langs or [])]

    def evaluate(self, context: dict[str, Any]) -> bool:
        """Check if language pair matches.

        Args:
            context: Must contain 'source_lang' and/or 'target_lang'

        Returns:
            True if languages match filters
        """
        source = context.get("source_lang", "").lower()
        target = context.get("target_lang", "").lower()

        # Check source language
        if self.source_langs and source not in self.source_langs:
            return False

        # Check target language
        if self.target_langs and target not in self.target_langs:
            return False

        return True


class ErrorCountCondition(BaseCondition):
    """Check error count thresholds.

    Example:
        >>> condition = ErrorCountCondition(min_errors=5)
        >>> condition.evaluate({"error_count": 10})
        True
    """

    def __init__(
        self,
        min_errors: int | None = None,
        max_errors: int | None = None,
    ):
        """Initialize with error count thresholds.

        Args:
            min_errors: Minimum errors to trigger
            max_errors: Maximum errors to trigger
        """
        self.min_errors = min_errors
        self.max_errors = max_errors

    def evaluate(self, context: dict[str, Any]) -> bool:
        """Check if error count meets threshold.

        Args:
            context: Must contain 'error_count'

        Returns:
            True if error count triggers condition
        """
        error_count = context.get("error_count", 0)

        try:
            error_count = int(error_count)
        except (TypeError, ValueError):
            return False

        if self.min_errors is not None and error_count >= self.min_errors:
            return True

        if self.max_errors is not None and error_count <= self.max_errors:
            return True

        return False


class CompositeCondition(BaseCondition):
    """Combine multiple conditions with AND/OR logic.

    Example:
        >>> cond1 = FilePatternCondition(patterns=["*.xliff"])
        >>> cond2 = ScoreThresholdCondition(metric="mqm", min_score=85)
        >>> composite = CompositeCondition([cond1, cond2], operator="and")
    """

    def __init__(
        self,
        conditions: list[BaseCondition],
        operator: str = "and",
    ):
        """Initialize with child conditions.

        Args:
            conditions: List of conditions to combine
            operator: "and" or "or"
        """
        self.conditions = conditions
        self.operator = operator.lower()

    def evaluate(self, context: dict[str, Any]) -> bool:
        """Evaluate all conditions with specified operator.

        Args:
            context: Context for evaluation

        Returns:
            Combined result
        """
        if not self.conditions:
            return True

        results = [c.evaluate(context) for c in self.conditions]

        if self.operator == "or":
            return any(results)
        return all(results)
