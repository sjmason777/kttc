"""Unit tests for MQM scoring engine.

Tests MQM score calculation and quality assessment.
"""

import pytest

from kttc.core.models import ErrorAnnotation, ErrorSeverity
from kttc.core.mqm import MQMScorer


@pytest.mark.unit
class TestMQMScorerInitialization:
    """Test MQMScorer initialization."""

    def test_init_with_default_weights(self) -> None:
        """Test initialization without glossary weights."""
        scorer = MQMScorer(use_glossary_weights=False)

        assert scorer.category_weights is not None
        assert "accuracy" in scorer.category_weights
        assert "fluency" in scorer.category_weights
        assert "terminology" in scorer.category_weights

    def test_init_with_glossary_weights(self) -> None:
        """Test initialization with glossary weights."""
        scorer = MQMScorer(use_glossary_weights=True)

        assert scorer.category_weights is not None
        # Should have default categories even if glossary fails
        assert len(scorer.category_weights) >= 6

    def test_default_weights_values(self) -> None:
        """Test default weight values."""
        assert MQMScorer.DEFAULT_CATEGORY_WEIGHTS["accuracy"] == 1.0
        assert MQMScorer.DEFAULT_CATEGORY_WEIGHTS["terminology"] == 0.9
        assert MQMScorer.DEFAULT_CATEGORY_WEIGHTS["fluency"] == 0.8
        assert MQMScorer.DEFAULT_CATEGORY_WEIGHTS["style"] == 0.6


@pytest.mark.unit
class TestMQMScoreCalculation:
    """Test MQM score calculation."""

    @pytest.fixture
    def scorer(self) -> MQMScorer:
        """Provide MQMScorer with default weights."""
        return MQMScorer(use_glossary_weights=False)

    def test_perfect_score_no_errors(self, scorer: MQMScorer) -> None:
        """Test perfect score when no errors."""
        score = scorer.calculate_score(errors=[], word_count=100)
        assert score == 100.0

    def test_score_with_minor_error(self, scorer: MQMScorer) -> None:
        """Test score with minor error."""
        errors = [
            ErrorAnnotation(
                category="fluency",
                subcategory="grammar",
                severity=ErrorSeverity.MINOR,
                location=(0, 5),
                description="Minor grammar issue",
            )
        ]
        score = scorer.calculate_score(errors=errors, word_count=100)
        # Score should be reduced but still good
        assert 90.0 <= score < 100.0

    def test_score_with_major_error(self, scorer: MQMScorer) -> None:
        """Test score with major error."""
        errors = [
            ErrorAnnotation(
                category="accuracy",
                subcategory="mistranslation",
                severity=ErrorSeverity.MAJOR,
                location=(0, 10),
                description="Major accuracy issue",
            )
        ]
        score = scorer.calculate_score(errors=errors, word_count=100)
        # Score should be significantly reduced
        assert score < 100.0

    def test_score_with_critical_error(self, scorer: MQMScorer) -> None:
        """Test score with critical error."""
        errors = [
            ErrorAnnotation(
                category="accuracy",
                subcategory="addition",
                severity=ErrorSeverity.CRITICAL,
                location=(0, 10),
                description="Critical accuracy issue",
            )
        ]
        score = scorer.calculate_score(errors=errors, word_count=100)
        # Score should be very low with critical error
        assert score < 90.0

    def test_score_with_multiple_errors(self, scorer: MQMScorer) -> None:
        """Test score with multiple errors."""
        errors = [
            ErrorAnnotation(
                category="fluency",
                subcategory="grammar",
                severity=ErrorSeverity.MINOR,
                location=(0, 5),
                description="Grammar issue",
            ),
            ErrorAnnotation(
                category="terminology",
                subcategory="inconsistency",
                severity=ErrorSeverity.MINOR,
                location=(10, 20),
                description="Terminology issue",
            ),
        ]
        score = scorer.calculate_score(errors=errors, word_count=100)
        # Score should be lower than single error
        assert score < 100.0

    def test_score_short_text(self, scorer: MQMScorer) -> None:
        """Test score with very short text."""
        errors = [
            ErrorAnnotation(
                category="fluency",
                subcategory="grammar",
                severity=ErrorSeverity.MINOR,
                location=(0, 3),
                description="Grammar issue",
            )
        ]
        # Short text should have more penalty impact
        score = scorer.calculate_score(errors=errors, word_count=5)
        assert score < 100.0

    def test_score_never_negative(self, scorer: MQMScorer) -> None:
        """Test that score never goes negative."""
        # Create many critical errors
        errors = [
            ErrorAnnotation(
                category="accuracy",
                subcategory="mistranslation",
                severity=ErrorSeverity.CRITICAL,
                location=(i * 10, i * 10 + 5),
                description=f"Critical error {i}",
            )
            for i in range(20)
        ]
        score = scorer.calculate_score(errors=errors, word_count=10)
        # Score should be capped at 0, not negative
        assert score >= 0.0


@pytest.mark.unit
class TestMQMThresholds:
    """Test MQM score thresholds."""

    def test_threshold_excellent(self) -> None:
        """Test excellent threshold value."""
        assert MQMScorer.THRESHOLD_EXCELLENT == 95.0

    def test_threshold_good(self) -> None:
        """Test good threshold value."""
        assert MQMScorer.THRESHOLD_GOOD == 90.0

    def test_threshold_acceptable(self) -> None:
        """Test acceptable threshold value."""
        assert MQMScorer.THRESHOLD_ACCEPTABLE == 80.0


@pytest.mark.unit
class TestMQMCategoryWeights:
    """Test category weight application."""

    @pytest.fixture
    def scorer(self) -> MQMScorer:
        """Provide MQMScorer with default weights."""
        return MQMScorer(use_glossary_weights=False)

    def test_accuracy_has_highest_weight(self, scorer: MQMScorer) -> None:
        """Test accuracy has highest weight."""
        assert scorer.category_weights["accuracy"] >= scorer.category_weights["fluency"]
        assert scorer.category_weights["accuracy"] >= scorer.category_weights["style"]

    def test_terminology_weight(self, scorer: MQMScorer) -> None:
        """Test terminology weight is high."""
        assert scorer.category_weights["terminology"] >= 0.8

    def test_style_has_lower_weight(self, scorer: MQMScorer) -> None:
        """Test style has lower weight than accuracy."""
        assert scorer.category_weights["style"] < scorer.category_weights["accuracy"]


@pytest.mark.unit
class TestMQMQualityLevel:
    """Test quality level determination."""

    @pytest.fixture
    def scorer(self) -> MQMScorer:
        """Provide MQMScorer with default weights."""
        return MQMScorer(use_glossary_weights=False)

    def test_get_quality_level_excellent(self, scorer: MQMScorer) -> None:
        """Test excellent quality level."""
        assert scorer.get_quality_level(96.0) == "excellent"
        assert scorer.get_quality_level(95.0) == "excellent"
        assert scorer.get_quality_level(100.0) == "excellent"

    def test_get_quality_level_good(self, scorer: MQMScorer) -> None:
        """Test good quality level."""
        assert scorer.get_quality_level(94.5) == "good"
        assert scorer.get_quality_level(90.0) == "good"

    def test_get_quality_level_acceptable(self, scorer: MQMScorer) -> None:
        """Test acceptable quality level."""
        assert scorer.get_quality_level(89.9) == "acceptable"
        assert scorer.get_quality_level(80.0) == "acceptable"

    def test_get_quality_level_poor(self, scorer: MQMScorer) -> None:
        """Test poor quality level."""
        assert scorer.get_quality_level(79.9) == "poor"
        assert scorer.get_quality_level(50.0) == "poor"
        assert scorer.get_quality_level(0.0) == "poor"


@pytest.mark.unit
class TestMQMPassesThreshold:
    """Test passes_threshold method."""

    @pytest.fixture
    def scorer(self) -> MQMScorer:
        """Provide MQMScorer with default weights."""
        return MQMScorer(use_glossary_weights=False)

    def test_passes_default_threshold(self, scorer: MQMScorer) -> None:
        """Test passing default threshold."""
        assert scorer.passes_threshold(95.0) is True
        assert scorer.passes_threshold(100.0) is True

    def test_fails_default_threshold(self, scorer: MQMScorer) -> None:
        """Test failing default threshold."""
        assert scorer.passes_threshold(94.9) is False
        assert scorer.passes_threshold(50.0) is False

    def test_custom_threshold(self, scorer: MQMScorer) -> None:
        """Test with custom threshold."""
        assert scorer.passes_threshold(85.0, threshold=80.0) is True
        assert scorer.passes_threshold(75.0, threshold=80.0) is False


@pytest.mark.unit
class TestMQMScoreBreakdown:
    """Test get_score_breakdown method."""

    @pytest.fixture
    def scorer(self) -> MQMScorer:
        """Provide MQMScorer with default weights."""
        return MQMScorer(use_glossary_weights=False)

    def test_breakdown_no_errors(self, scorer: MQMScorer) -> None:
        """Test breakdown with no errors."""
        breakdown = scorer.get_score_breakdown(errors=[], word_count=100)

        assert breakdown["total_penalty"] == 0.0
        assert breakdown["penalty_per_1k"] == 0.0
        assert breakdown["score"] == 100.0
        assert breakdown["category_breakdown"] == {}
        assert breakdown["severity_breakdown"] == {}
        assert breakdown["word_count"] == 100
        assert breakdown["error_count"] == 0

    def test_breakdown_single_error(self, scorer: MQMScorer) -> None:
        """Test breakdown with single error."""
        errors = [
            ErrorAnnotation(
                category="accuracy",
                subcategory="mistranslation",
                severity=ErrorSeverity.MAJOR,
                location=(0, 10),
                description="Test error",
            )
        ]
        breakdown = scorer.get_score_breakdown(errors=errors, word_count=100)

        assert breakdown["total_penalty"] > 0
        assert breakdown["error_count"] == 1
        assert "accuracy" in breakdown["category_breakdown"]
        assert "major" in breakdown["severity_breakdown"]

    def test_breakdown_multiple_categories(self, scorer: MQMScorer) -> None:
        """Test breakdown with multiple categories."""
        errors = [
            ErrorAnnotation(
                category="accuracy",
                subcategory="mistranslation",
                severity=ErrorSeverity.MAJOR,
                location=(0, 5),
                description="Accuracy error",
            ),
            ErrorAnnotation(
                category="fluency",
                subcategory="grammar",
                severity=ErrorSeverity.MINOR,
                location=(10, 15),
                description="Fluency error",
            ),
        ]
        breakdown = scorer.get_score_breakdown(errors=errors, word_count=100)

        assert "accuracy" in breakdown["category_breakdown"]
        assert "fluency" in breakdown["category_breakdown"]
        assert breakdown["error_count"] == 2

    def test_breakdown_invalid_word_count(self, scorer: MQMScorer) -> None:
        """Test breakdown with invalid word count raises error."""
        with pytest.raises(ValueError, match="word_count must be greater than 0"):
            scorer.get_score_breakdown(errors=[], word_count=0)

        with pytest.raises(ValueError, match="word_count must be greater than 0"):
            scorer.get_score_breakdown(errors=[], word_count=-5)


@pytest.mark.unit
class TestMQMScoreValidation:
    """Test score calculation edge cases."""

    @pytest.fixture
    def scorer(self) -> MQMScorer:
        """Provide MQMScorer with default weights."""
        return MQMScorer(use_glossary_weights=False)

    def test_calculate_score_invalid_word_count_zero(self, scorer: MQMScorer) -> None:
        """Test calculate_score with zero word count raises error."""
        with pytest.raises(ValueError, match="word_count must be greater than 0"):
            scorer.calculate_score(errors=[], word_count=0)

    def test_calculate_score_invalid_word_count_negative(self, scorer: MQMScorer) -> None:
        """Test calculate_score with negative word count raises error."""
        with pytest.raises(ValueError, match="word_count must be greater than 0"):
            scorer.calculate_score(errors=[], word_count=-10)

    def test_calculate_score_with_custom_weights(self, scorer: MQMScorer) -> None:
        """Test calculate_score with custom weights."""
        errors = [
            ErrorAnnotation(
                category="accuracy",
                subcategory="mistranslation",
                severity=ErrorSeverity.MAJOR,
                location=(0, 10),
                description="Test error",
            )
        ]
        custom_weights = {"accuracy": 2.0}  # Double the accuracy weight

        score_default = scorer.calculate_score(errors=errors, word_count=100)
        score_custom = scorer.calculate_score(
            errors=errors, word_count=100, custom_weights=custom_weights
        )

        # Custom weights should result in different score
        assert score_custom != score_default
        # Higher weight means more penalty, so lower score
        assert score_custom < score_default

    def test_calculate_score_unknown_category(self, scorer: MQMScorer) -> None:
        """Test calculate_score with unknown category defaults to weight 1.0."""
        errors = [
            ErrorAnnotation(
                category="unknown_category",
                subcategory="unknown_sub",
                severity=ErrorSeverity.MINOR,
                location=(0, 5),
                description="Unknown error",
            )
        ]

        # Should not raise error, should use default weight 1.0
        score = scorer.calculate_score(errors=errors, word_count=100)
        assert score < 100.0
