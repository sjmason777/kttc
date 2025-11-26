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
