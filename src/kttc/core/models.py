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

"""Core data models for KTTC translation quality assurance.

This module defines the fundamental data structures used throughout the platform:
- Error annotations and severity levels
- Translation tasks and metadata
- Quality assessment reports
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

# Default severity penalty multipliers (industry standard based on Lokalise/Phrase)
# Can be overridden via configuration or profiles
SEVERITY_PENALTIES: dict[str, float] = {
    "neutral": 0.0,  # 0x - no error, informational
    "minor": 1.0,  # 1x - noticeable but doesn't affect understanding
    "major": 5.0,  # 5x - affects understanding or quality
    "critical": 25.0,  # 25x - severe meaning change or unusable
}

SEVERITY_MULTIPLIER_LABELS: dict[str, str] = {
    "neutral": "0x",
    "minor": "1x",
    "major": "5x",
    "critical": "25x",
}


class ErrorSeverity(str, Enum):
    """MQM error severity levels.

    Based on Multidimensional Quality Metrics (MQM) framework.
    Each level has a different penalty weight in quality scoring.

    Penalty multipliers (industry standard based on Lokalise/Phrase):
    - Neutral: 0x (no error, informational)
    - Minor: 1x (noticeable but doesn't affect understanding)
    - Major: 5x (affects understanding or quality)
    - Critical: 25x (severe meaning change or unusable)
    """

    NEUTRAL = "neutral"  # No penalty (0x multiplier)
    MINOR = "minor"  # Minor penalty (1x multiplier)
    MAJOR = "major"  # Major penalty (5x multiplier)
    CRITICAL = "critical"  # Critical penalty (25x multiplier)

    @property
    def penalty_value(self) -> float:
        """Get numeric penalty value for scoring.

        Uses default multipliers from SEVERITY_PENALTIES.
        For custom multipliers, override SEVERITY_PENALTIES or use profiles.
        """
        return SEVERITY_PENALTIES.get(self.value, 1.0)

    @property
    def multiplier_label(self) -> str:
        """Get human-readable multiplier label (e.g., '25x' for critical)."""
        return SEVERITY_MULTIPLIER_LABELS.get(self.value, "1x")


class ErrorAnnotation(BaseModel):
    """Represents a single quality error found by an agent.

    Follows MQM error typology with category, subcategory, and severity.
    """

    category: str = Field(
        ..., description="MQM error category (e.g., 'accuracy', 'fluency', 'terminology')"
    )
    subcategory: str = Field(
        ..., description="Specific error type (e.g., 'mistranslation', 'grammar', 'inconsistency')"
    )
    severity: ErrorSeverity = Field(..., description="Error severity level")
    location: tuple[int, int] = Field(
        ..., description="Character span (start, end) in translation where error occurs"
    )
    description: str = Field(..., description="Human-readable explanation of the error")
    suggestion: str | None = Field(default=None, description="Suggested fix or improvement")
    confidence: float | None = Field(
        default=None, description="Confidence in suggestion (0.0-1.0)", ge=0.0, le=1.0
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "category": "accuracy",
                "subcategory": "mistranslation",
                "severity": "major",
                "location": [0, 5],
                "description": "Incorrect translation of 'hello'",
                "suggestion": "Use 'hola' instead",
            }
        }
    )


class TranslationTask(BaseModel):
    """Represents a translation to be evaluated.

    Contains source text, translation, language codes, and optional metadata.
    """

    source_text: str = Field(..., description="Original text in source language", min_length=1)
    translation: str = Field(..., description="Translated text in target language", min_length=1)
    source_lang: str = Field(
        ..., description="Source language code (ISO 639-1, e.g., 'en', 'ru')", pattern=r"^[a-z]{2}$"
    )
    target_lang: str = Field(
        ..., description="Target language code (ISO 639-1, e.g., 'es', 'fr')", pattern=r"^[a-z]{2}$"
    )
    context: dict[str, Any] | None = Field(
        default=None, description="Additional context (domain, style guide, glossary, etc.)"
    )

    @property
    def word_count(self) -> int:
        """Calculate word count of source text for scoring."""
        return len(self.source_text.split())

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "source_text": "Hello, world!",
                "translation": "Hola, mundo!",
                "source_lang": "en",
                "target_lang": "es",
                "context": {"domain": "general"},
            }
        }
    )


class QAReport(BaseModel):
    """Quality assessment report generated by the system.

    Contains MQM score, detected errors, and pass/fail status.

    New in weighted consensus mode:
    - confidence: Confidence level based on agent agreement (0.0-1.0)
    - agent_agreement: How much agents agree with each other
    - agent_scores: Individual MQM scores per agent
    - consensus_metadata: Additional consensus calculation details
    """

    task: TranslationTask = Field(..., description="The translation task that was evaluated")
    mqm_score: float = Field(
        ..., description="MQM quality score (0-100, higher is better)", ge=0.0, le=100.0
    )
    comet_score: float | None = Field(
        default=None, description="COMET neural metric score (0-1)", ge=0.0, le=1.0
    )
    kiwi_score: float | None = Field(
        default=None,
        description="CometKiwi reference-free quality score (0-1)",
        ge=0.0,
        le=1.0,
    )
    neural_quality_estimate: str | None = Field(
        default=None,
        description="Neural metrics quality classification: high, medium, or low",
        pattern=r"^(high|medium|low)$",
    )
    composite_score: float | None = Field(
        default=None,
        description="Composite score combining MQM and neural metrics (0-100)",
        ge=0.0,
        le=100.0,
    )
    errors: list[ErrorAnnotation] = Field(
        default_factory=list, description="List of errors found by QA agents"
    )
    status: str = Field(
        ..., description="Overall status: 'pass' or 'fail'", pattern=r"^(pass|fail)$"
    )
    agent_details: dict[str, Any] | None = Field(
        default=None, description="Detailed results from individual agents"
    )
    score_breakdown: dict[str, Any] | None = Field(
        default=None,
        description="Detailed MQM score breakdown (from MQMScorer.get_score_breakdown)",
    )

    # Weighted consensus fields
    confidence: float | None = Field(
        default=None,
        description="Confidence level (0.0-1.0) based on agent agreement. "
        "High confidence means agents agree, low means disagreement.",
        ge=0.0,
        le=1.0,
    )
    agent_agreement: float | None = Field(
        default=None,
        description="Agent agreement metric (0.0-1.0). "
        "1.0 = perfect agreement, 0.0 = high disagreement",
        ge=0.0,
        le=1.0,
    )
    agent_scores: dict[str, float] | None = Field(
        default=None,
        description="Individual MQM scores per agent (e.g., {'accuracy': 95.0, 'fluency': 93.0})",
    )
    consensus_metadata: dict[str, Any] | None = Field(
        default=None,
        description="Additional weighted consensus metadata "
        "(variance, std_dev, agent_weights, etc.)",
    )

    # Cost tracking fields
    usage_stats: dict[str, Any] | None = Field(
        default=None,
        description="Token usage and cost statistics",
    )

    @property
    def error_count(self) -> int:
        """Total number of errors found."""
        return len(self.errors)

    @property
    def critical_error_count(self) -> int:
        """Number of critical errors."""
        return sum(1 for e in self.errors if e.severity == ErrorSeverity.CRITICAL)

    @property
    def major_error_count(self) -> int:
        """Number of major errors."""
        return sum(1 for e in self.errors if e.severity == ErrorSeverity.MAJOR)

    @property
    def minor_error_count(self) -> int:
        """Number of minor errors."""
        return sum(1 for e in self.errors if e.severity == ErrorSeverity.MINOR)

    def is_high_confidence(self, threshold: float = 0.8) -> bool:
        """Check if evaluation has high confidence.

        Args:
            threshold: Minimum confidence level (default: 0.8)

        Returns:
            True if confidence >= threshold, False otherwise or if confidence is None

        Example:
            >>> report = QAReport(...)
            >>> if not report.is_high_confidence():
            ...     print("Warning: Low confidence - recommend human review")
        """
        if self.confidence is None:
            return False
        return self.confidence >= threshold

    def needs_human_review(self, confidence_threshold: float = 0.7) -> bool:
        """Determine if translation needs human review based on confidence.

        Args:
            confidence_threshold: Minimum acceptable confidence (default: 0.7)

        Returns:
            True if human review recommended (low confidence or fail status)

        Example:
            >>> report = QAReport(...)
            >>> if report.needs_human_review():
            ...     print("Flagged for human review")
        """
        # Always review if failed
        if self.status == "fail":
            return True

        # Review if low confidence (when available)
        if self.confidence is not None and self.confidence < confidence_threshold:
            return True

        return False

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "task": {
                    "source_text": "Hello",
                    "translation": "Hola",
                    "source_lang": "en",
                    "target_lang": "es",
                },
                "mqm_score": 96.5,
                "comet_score": 0.92,
                "errors": [],
                "status": "pass",
            }
        }
    )
