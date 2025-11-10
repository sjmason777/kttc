"""Unit tests for MQM scoring engine.

Tests the MQM (Multidimensional Quality Metrics) scorer implementation
including score calculations, category weights, and edge cases.
"""

import pytest

from kttc.core import ErrorAnnotation, ErrorSeverity, MQMScorer


@pytest.mark.unit
class TestMQMScorer:
    """Tests for MQM scoring engine."""

    def test_perfect_score_no_errors(self) -> None:
        """Test that no errors results in perfect score of 100."""
        scorer = MQMScorer()
        score = scorer.calculate_score(errors=[], word_count=100)
        assert score == 100.0

    def test_single_minor_error(self) -> None:
        """Test score with one minor error."""
        scorer = MQMScorer()
        errors = [
            ErrorAnnotation(
                category="fluency",
                subcategory="grammar",
                severity=ErrorSeverity.MINOR,
                location=(0, 5),
                description="Minor grammar issue",
            )
        ]
        # Minor (1 point) * fluency weight (0.8) = 0.8 penalty
        # 0.8 / 100 * 1000 = 8 penalty per 1k
        # Score = 100 - 8 = 92
        score = scorer.calculate_score(errors, word_count=100)
        assert score == 92.0

    def test_single_major_error(self) -> None:
        """Test score with one major error."""
        scorer = MQMScorer()
        errors = [
            ErrorAnnotation(
                category="accuracy",
                subcategory="mistranslation",
                severity=ErrorSeverity.MAJOR,
                location=(0, 10),
                description="Significant mistranslation",
            )
        ]
        # Major (5 points) * accuracy weight (1.0) = 5 penalty
        # 5 / 100 * 1000 = 50 penalty per 1k
        # Score = 100 - 50 = 50
        score = scorer.calculate_score(errors, word_count=100)
        assert score == 50.0

    def test_single_critical_error(self) -> None:
        """Test score with one critical error."""
        scorer = MQMScorer()
        errors = [
            ErrorAnnotation(
                category="accuracy",
                subcategory="mistranslation",
                severity=ErrorSeverity.CRITICAL,
                location=(0, 10),
                description="Critical error",
            )
        ]
        # Critical (10 points) * accuracy weight (1.0) = 10 penalty
        # 10 / 100 * 1000 = 100 penalty per 1k
        # Score = 100 - 100 = 0
        score = scorer.calculate_score(errors, word_count=100)
        assert score == 0.0

    def test_multiple_errors_different_severities(self) -> None:
        """Test score with multiple errors of different severities."""
        scorer = MQMScorer()
        errors = [
            ErrorAnnotation(
                category="accuracy",
                subcategory="mistranslation",
                severity=ErrorSeverity.MAJOR,  # 5 * 1.0 = 5
                location=(0, 5),
                description="Major error",
            ),
            ErrorAnnotation(
                category="fluency",
                subcategory="grammar",
                severity=ErrorSeverity.MINOR,  # 1 * 0.8 = 0.8
                location=(10, 15),
                description="Minor error",
            ),
            ErrorAnnotation(
                category="terminology",
                subcategory="inconsistency",
                severity=ErrorSeverity.MINOR,  # 1 * 0.9 = 0.9
                location=(20, 25),
                description="Term inconsistency",
            ),
        ]
        # Total penalty = 5 + 0.8 + 0.9 = 6.7
        # 6.7 / 100 * 1000 = 67 penalty per 1k
        # Score = 100 - 67 = 33
        score = scorer.calculate_score(errors, word_count=100)
        assert score == 33.0

    def test_category_weights_applied(self) -> None:
        """Test that category weights are properly applied."""
        scorer = MQMScorer()

        # Same severity, different categories
        accuracy_error = [
            ErrorAnnotation(
                category="accuracy",
                subcategory="mistranslation",
                severity=ErrorSeverity.MINOR,  # 1 * 1.0 = 1.0
                location=(0, 5),
                description="Accuracy error",
            )
        ]

        style_error = [
            ErrorAnnotation(
                category="style",
                subcategory="register",
                severity=ErrorSeverity.MINOR,  # 1 * 0.6 = 0.6
                location=(0, 5),
                description="Style error",
            )
        ]

        accuracy_score = scorer.calculate_score(accuracy_error, word_count=100)
        style_score = scorer.calculate_score(style_error, word_count=100)

        # Accuracy (weight 1.0) should penalize more than style (weight 0.6)
        assert accuracy_score < style_score
        assert accuracy_score == 90.0  # 100 - (1.0 / 100 * 1000) = 90
        assert style_score == 94.0  # 100 - (0.6 / 100 * 1000) = 94

    def test_word_count_scaling(self) -> None:
        """Test that scores scale correctly with word count."""
        scorer = MQMScorer()
        error = [
            ErrorAnnotation(
                category="accuracy",
                subcategory="mistranslation",
                severity=ErrorSeverity.MAJOR,  # 5 points
                location=(0, 5),
                description="Error",
            )
        ]

        # Same error, different word counts
        score_100 = scorer.calculate_score(error, word_count=100)
        score_1000 = scorer.calculate_score(error, word_count=1000)

        # Longer text should have higher score (less penalty per word)
        assert score_1000 > score_100
        assert score_100 == 50.0  # 100 - (5 / 100 * 1000) = 50
        assert score_1000 == 95.0  # 100 - (5 / 1000 * 1000) = 95

    def test_custom_weights(self) -> None:
        """Test using custom category weights."""
        scorer = MQMScorer()
        custom_weights = {
            "accuracy": 2.0,  # Double weight
            "fluency": 0.5,  # Half weight
        }

        error = [
            ErrorAnnotation(
                category="accuracy",
                subcategory="mistranslation",
                severity=ErrorSeverity.MINOR,  # 1 * 2.0 = 2.0
                location=(0, 5),
                description="Error",
            )
        ]

        default_score = scorer.calculate_score(error, word_count=100)
        custom_score = scorer.calculate_score(error, word_count=100, custom_weights=custom_weights)

        # Custom weight (2.0) should penalize more than default (1.0)
        assert custom_score < default_score
        assert default_score == 90.0  # 100 - (1.0 / 100 * 1000) = 90
        assert custom_score == 80.0  # 100 - (2.0 / 100 * 1000) = 80

    def test_unknown_category_defaults_to_one(self) -> None:
        """Test that unknown categories default to weight 1.0."""
        scorer = MQMScorer()
        error = [
            ErrorAnnotation(
                category="unknown_category",
                subcategory="some_issue",
                severity=ErrorSeverity.MINOR,  # 1 * 1.0 = 1.0
                location=(0, 5),
                description="Unknown category error",
            )
        ]

        score = scorer.calculate_score(error, word_count=100)
        assert score == 90.0  # 100 - (1.0 / 100 * 1000) = 90

    def test_zero_word_count_raises_error(self) -> None:
        """Test that zero word count raises ValueError."""
        scorer = MQMScorer()
        with pytest.raises(ValueError, match="word_count must be greater than 0"):
            scorer.calculate_score([], word_count=0)

    def test_negative_word_count_raises_error(self) -> None:
        """Test that negative word count raises ValueError."""
        scorer = MQMScorer()
        with pytest.raises(ValueError, match="word_count must be greater than 0"):
            scorer.calculate_score([], word_count=-100)

    def test_score_never_negative(self) -> None:
        """Test that score never goes below 0."""
        scorer = MQMScorer()
        # Many critical errors
        errors = [
            ErrorAnnotation(
                category="accuracy",
                subcategory="mistranslation",
                severity=ErrorSeverity.CRITICAL,  # 10 points each
                location=(i, i + 5),
                description=f"Critical error {i}",
            )
            for i in range(20)  # 20 critical errors = 200 penalty points
        ]

        score = scorer.calculate_score(errors, word_count=10)
        assert score >= 0.0
        assert score == 0.0  # Should be clamped to 0

    def test_get_score_breakdown(self) -> None:
        """Test detailed score breakdown calculation."""
        scorer = MQMScorer()
        errors = [
            ErrorAnnotation(
                category="accuracy",
                subcategory="mistranslation",
                severity=ErrorSeverity.MAJOR,  # 5 * 1.0 = 5.0
                location=(0, 5),
                description="Accuracy error",
            ),
            ErrorAnnotation(
                category="fluency",
                subcategory="grammar",
                severity=ErrorSeverity.MINOR,  # 1 * 0.8 = 0.8
                location=(10, 15),
                description="Fluency error",
            ),
        ]

        breakdown = scorer.get_score_breakdown(errors, word_count=100)

        assert breakdown["total_penalty"] == 5.8
        assert breakdown["penalty_per_1k"] == 58.0
        assert breakdown["score"] == 42.0
        assert breakdown["word_count"] == 100
        assert breakdown["error_count"] == 2
        assert breakdown["category_breakdown"]["accuracy"] == 5.0
        assert breakdown["category_breakdown"]["fluency"] == 0.8
        assert "major" in breakdown["severity_breakdown"]
        assert "minor" in breakdown["severity_breakdown"]

    def test_get_score_breakdown_zero_word_count(self) -> None:
        """Test get_score_breakdown raises error for zero word count."""
        scorer = MQMScorer()
        with pytest.raises(ValueError, match="word_count must be greater than 0"):
            scorer.get_score_breakdown([], word_count=0)

    def test_get_score_breakdown_negative_word_count(self) -> None:
        """Test get_score_breakdown raises error for negative word count."""
        scorer = MQMScorer()
        with pytest.raises(ValueError, match="word_count must be greater than 0"):
            scorer.get_score_breakdown([], word_count=-50)

    def test_get_quality_level(self) -> None:
        """Test quality level classification."""
        scorer = MQMScorer()

        assert scorer.get_quality_level(98.0) == "excellent"
        assert scorer.get_quality_level(95.0) == "excellent"
        assert scorer.get_quality_level(92.0) == "good"
        assert scorer.get_quality_level(90.0) == "good"
        assert scorer.get_quality_level(85.0) == "acceptable"
        assert scorer.get_quality_level(80.0) == "acceptable"
        assert scorer.get_quality_level(75.0) == "poor"
        assert scorer.get_quality_level(50.0) == "poor"

    def test_passes_threshold(self) -> None:
        """Test threshold checking."""
        scorer = MQMScorer()

        assert scorer.passes_threshold(96.0, threshold=95.0) is True
        assert scorer.passes_threshold(95.0, threshold=95.0) is True
        assert scorer.passes_threshold(94.9, threshold=95.0) is False
        assert scorer.passes_threshold(90.0, threshold=90.0) is True
        assert scorer.passes_threshold(89.9, threshold=90.0) is False

    def test_realistic_translation_scenario(self) -> None:
        """Test a realistic translation QA scenario."""
        scorer = MQMScorer()

        # Simulated translation: 150 words with mixed errors
        errors = [
            # One critical accuracy error
            ErrorAnnotation(
                category="accuracy",
                subcategory="mistranslation",
                severity=ErrorSeverity.CRITICAL,
                location=(0, 20),
                description="Critical meaning error",
            ),
            # Two major fluency errors
            ErrorAnnotation(
                category="fluency",
                subcategory="grammar",
                severity=ErrorSeverity.MAJOR,
                location=(50, 60),
                description="Grammar error",
            ),
            ErrorAnnotation(
                category="fluency",
                subcategory="word_order",
                severity=ErrorSeverity.MAJOR,
                location=(100, 110),
                description="Word order issue",
            ),
            # Three minor errors
            ErrorAnnotation(
                category="terminology",
                subcategory="inconsistency",
                severity=ErrorSeverity.MINOR,
                location=(120, 125),
                description="Term variant",
            ),
            ErrorAnnotation(
                category="style",
                subcategory="register",
                severity=ErrorSeverity.MINOR,
                location=(130, 135),
                description="Register mismatch",
            ),
            ErrorAnnotation(
                category="locale",
                subcategory="convention",
                severity=ErrorSeverity.MINOR,
                location=(140, 145),
                description="Locale convention",
            ),
        ]

        score = scorer.calculate_score(errors, word_count=150)
        breakdown = scorer.get_score_breakdown(errors, word_count=150)

        # Expected calculation:
        # Critical accuracy: 10 * 1.0 = 10.0
        # Major fluency: 5 * 0.8 = 4.0 (x2 = 8.0)
        # Minor terminology: 1 * 0.9 = 0.9
        # Minor style: 1 * 0.6 = 0.6
        # Minor locale: 1 * 0.7 = 0.7
        # Total: 10 + 8 + 0.9 + 0.6 + 0.7 = 20.2
        # Per 1k: 20.2 / 150 * 1000 = 134.67
        # Score: 100 - 134.67 = -34.67 -> 0.0 (clamped)

        assert score == 0.0  # Too many serious errors
        assert breakdown["total_penalty"] == 20.2
        assert breakdown["error_count"] == 6
        assert scorer.get_quality_level(score) == "poor"
        assert scorer.passes_threshold(score, threshold=95.0) is False
