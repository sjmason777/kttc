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

"""MQM (Multidimensional Quality Metrics) scoring engine.

This module implements the MQM framework for translation quality assessment.
MQM is the industry-standard approach used in WMT benchmarks and professional
translation quality evaluation.

References:
    - MQM Framework: https://themqm.org/
    - WMT Metrics Task: https://machinetranslate.org/wmt
"""

from __future__ import annotations

import logging
from typing import Any

from kttc.core.models import ErrorAnnotation
from kttc.terminology import GlossaryManager

logger = logging.getLogger(__name__)


class MQMScorer:
    """MQM scoring engine for translation quality assessment.

    Implements the MQM scoring formula:
        MQM Score = 100 - (total_penalty / word_count * 1000)

    Category weights reflect the relative importance of different error types:
    - Accuracy: 1.0 (highest - semantic correctness is critical)
    - Terminology: 0.9 (very important - domain-specific terms)
    - Fluency: 0.8 (important - natural language)
    - Style: 0.6 (moderate - tone and register)
    - Locale: 0.7 (moderate - regional conventions)
    - Context: 0.7 (moderate - situational appropriateness)

    Severity penalties (MQM standard):
    - Neutral: 0 points (no error)
    - Minor: 1 point (noticeable but doesn't affect understanding)
    - Major: 5 points (affects understanding or quality)
    - Critical: 10 points (severe meaning change or unusable)

    Score interpretation:
    - 95-100: Excellent quality (production-ready)
    - 90-94: Good quality (minor fixes needed)
    - 80-89: Acceptable quality (revision needed)
    - < 80: Poor quality (significant rework required)
    """

    # Default category weight multipliers (based on MQM framework and WMT practices)
    # These can be overridden by glossary-based weights
    DEFAULT_CATEGORY_WEIGHTS = {
        "accuracy": 1.0,  # Semantic correctness (highest priority)
        "terminology": 0.9,  # Domain-specific terms
        "fluency": 0.8,  # Grammar and naturalness
        "style": 0.6,  # Tone and register
        "locale": 0.7,  # Regional conventions
        "context": 0.7,  # Situational appropriateness
    }

    # Score thresholds for quality levels
    THRESHOLD_EXCELLENT = 95.0
    THRESHOLD_GOOD = 90.0
    THRESHOLD_ACCEPTABLE = 80.0

    def __init__(self, use_glossary_weights: bool = True):
        """Initialize MQM Scorer.

        Args:
            use_glossary_weights: If True, load category weights from MQM glossary
        """
        self.glossary_manager = GlossaryManager() if use_glossary_weights else None
        self.category_weights = (
            self._load_category_weights() if use_glossary_weights else self.DEFAULT_CATEGORY_WEIGHTS
        )

        if use_glossary_weights:
            logger.info(
                f"MQMScorer initialized with glossary-based weights: {self.category_weights}"
            )
        else:
            logger.info("MQMScorer initialized with default weights")

    def _load_category_weights(self) -> dict[str, float]:
        """Load MQM category weights from glossary.

        Returns:
            Dictionary mapping category names to weight multipliers
        """
        if self.glossary_manager is None:
            return self.DEFAULT_CATEGORY_WEIGHTS.copy()

        try:
            # Load English MQM glossary
            mqm_glossary = self.glossary_manager.load_glossary("en", "mqm_core")

            weights = {}
            error_dimensions = mqm_glossary.get("error_dimensions", [])

            # Handle both list and dict formats for backwards compatibility
            if isinstance(error_dimensions, list):
                # New format: list of dimension objects with 'id' and 'severity_weight'
                for dimension in error_dimensions:
                    if isinstance(dimension, dict):
                        category_id = dimension.get("id", "").lower()
                        weight = dimension.get("severity_weight", 1.0)
                        if category_id:
                            weights[category_id] = weight
            elif isinstance(error_dimensions, dict):
                # Legacy format: dict mapping category to data
                for category, category_data in error_dimensions.items():
                    if isinstance(category_data, dict):
                        weight = category_data.get(
                            "weight", self.DEFAULT_CATEGORY_WEIGHTS.get(category, 1.0)
                        )
                        weights[category] = weight
                    else:
                        weights[category] = self.DEFAULT_CATEGORY_WEIGHTS.get(category, 1.0)

            # Fill in any missing categories with defaults
            for category, default_weight in self.DEFAULT_CATEGORY_WEIGHTS.items():
                if category not in weights:
                    weights[category] = default_weight

            logger.info(f"Loaded MQM category weights from glossary: {len(weights)} categories")
            return weights

        except Exception as e:
            logger.warning(f"Failed to load glossary weights, using defaults: {e}")
            return self.DEFAULT_CATEGORY_WEIGHTS.copy()

    def calculate_score(
        self,
        errors: list[ErrorAnnotation],
        word_count: int,
        custom_weights: dict[str, float] | None = None,
    ) -> float:
        """Calculate MQM score for a translation.

        Args:
            errors: List of error annotations found by QA agents
            word_count: Number of words in source text
            custom_weights: Optional custom category weights (overrides defaults)

        Returns:
            MQM score from 0-100 (higher is better)

        Raises:
            ValueError: If word_count is <= 0

        Example:
            >>> scorer = MQMScorer()
            >>> errors = [
            ...     ErrorAnnotation(
            ...         category="accuracy",
            ...         subcategory="mistranslation",
            ...         severity=ErrorSeverity.MAJOR,
            ...         location=(0, 5),
            ...         description="Wrong meaning"
            ...     )
            ... ]
            >>> score = scorer.calculate_score(errors, word_count=100)
            >>> score
            95.0
        """
        if word_count <= 0:
            raise ValueError("word_count must be greater than 0")

        # Use custom weights if provided, otherwise use instance weights
        weights = custom_weights or self.category_weights

        # Calculate total penalty
        total_penalty = 0.0
        for error in errors:
            # Get severity penalty value
            severity_penalty = error.severity.penalty_value

            # Get category weight (default to 1.0 for unknown categories)
            category_weight = weights.get(error.category.lower(), 1.0)

            # Calculate weighted penalty
            penalty = severity_penalty * category_weight
            total_penalty += penalty

        # Calculate penalty per 1000 words (MQM standard normalization)
        penalty_per_1k = (total_penalty / word_count) * 1000

        # Calculate final score (100 - penalty, minimum 0)
        score = max(0.0, 100.0 - penalty_per_1k)

        return round(score, 2)

    def get_score_breakdown(
        self,
        errors: list[ErrorAnnotation],
        word_count: int,
    ) -> dict[str, Any]:
        """Get detailed breakdown of MQM score calculation.

        Args:
            errors: List of error annotations
            word_count: Number of words in source text

        Returns:
            Dictionary containing:
            - total_penalty: Total penalty points
            - penalty_per_1k: Penalty per 1000 words
            - score: Final MQM score
            - category_breakdown: Penalty by category
            - severity_breakdown: Penalty by severity

        Example:
            >>> scorer = MQMScorer()
            >>> breakdown = scorer.get_score_breakdown(errors, 100)
            >>> breakdown['category_breakdown']
            {'accuracy': 5.0, 'fluency': 0.8}
        """
        if word_count <= 0:
            raise ValueError("word_count must be greater than 0")

        # Initialize tracking dictionaries
        category_penalties: dict[str, float] = {}
        severity_penalties: dict[str, float] = {}
        total_penalty = 0.0

        # Calculate penalties with tracking
        for error in errors:
            severity_penalty = error.severity.penalty_value
            category_weight = self.category_weights.get(error.category.lower(), 1.0)
            penalty = severity_penalty * category_weight

            # Track by category
            category = error.category.lower()
            category_penalties[category] = category_penalties.get(category, 0.0) + penalty

            # Track by severity
            severity = error.severity.value
            severity_penalties[severity] = severity_penalties.get(severity, 0.0) + penalty

            total_penalty += penalty

        # Calculate derived values
        penalty_per_1k = (total_penalty / word_count) * 1000
        score = max(0.0, 100.0 - penalty_per_1k)

        return {
            "total_penalty": round(total_penalty, 2),
            "penalty_per_1k": round(penalty_per_1k, 2),
            "score": round(score, 2),
            "category_breakdown": {k: round(v, 2) for k, v in category_penalties.items()},
            "severity_breakdown": {k: round(v, 2) for k, v in severity_penalties.items()},
            "word_count": word_count,
            "error_count": len(errors),
        }

    def get_quality_level(self, score: float) -> str:
        """Get quality level description for a given MQM score.

        Args:
            score: MQM score (0-100)

        Returns:
            Quality level: "excellent", "good", "acceptable", or "poor"

        Example:
            >>> scorer = MQMScorer()
            >>> scorer.get_quality_level(96.5)
            'excellent'
        """
        if score >= self.THRESHOLD_EXCELLENT:
            return "excellent"
        elif score >= self.THRESHOLD_GOOD:
            return "good"
        elif score >= self.THRESHOLD_ACCEPTABLE:
            return "acceptable"
        else:
            return "poor"

    def passes_threshold(self, score: float, threshold: float = 95.0) -> bool:
        """Check if MQM score meets quality threshold.

        Args:
            score: MQM score to check
            threshold: Minimum acceptable score (default: 95.0)

        Returns:
            True if score >= threshold, False otherwise

        Example:
            >>> scorer = MQMScorer()
            >>> scorer.passes_threshold(96.5, threshold=95.0)
            True
        """
        return score >= threshold
