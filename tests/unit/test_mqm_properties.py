"""Property-based tests for MQM Scorer.

Uses Hypothesis to test invariants and properties that should always hold,
regardless of the specific input values. This approach helps find edge cases
that unit tests might miss.

Philosophy: "Property-based tests verify mathematical invariants,
not just specific examples."
"""

from typing import Any

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from kttc.core.models import ErrorAnnotation, ErrorSeverity
from kttc.core.mqm import MQMScorer

# ============================================================================
# Custom Strategies
# ============================================================================


@st.composite
def error_annotation_strategy(draw: Any) -> ErrorAnnotation:
    """Generate random ErrorAnnotation objects."""
    categories = ["accuracy", "terminology", "fluency", "style", "locale", "context"]
    subcategories = ["general", "mistranslation", "grammar", "spelling", "tone"]
    severities = [
        ErrorSeverity.NEUTRAL,
        ErrorSeverity.MINOR,
        ErrorSeverity.MAJOR,
        ErrorSeverity.CRITICAL,
    ]

    category = draw(st.sampled_from(categories))
    subcategory = draw(st.sampled_from(subcategories))
    severity = draw(st.sampled_from(severities))

    # Generate valid location (start < end, both >= 0)
    start = draw(st.integers(min_value=0, max_value=100))
    end = start + draw(st.integers(min_value=1, max_value=50))

    # Use simple ASCII text for speed
    description = draw(st.text(alphabet="abcdefghijklmnopqrstuvwxyz ", min_size=5, max_size=30))

    return ErrorAnnotation(
        category=category,
        subcategory=subcategory,
        severity=severity,
        location=(start, end),
        description=description,
    )


@st.composite
def error_list_strategy(draw: Any, max_size: int = 20) -> list[ErrorAnnotation]:
    """Generate a list of error annotations."""
    return draw(st.lists(error_annotation_strategy(), max_size=max_size))


# ============================================================================
# Property Tests for MQMScorer
# ============================================================================


@pytest.mark.unit
class TestMQMScorerProperties:
    """Property-based tests for MQMScorer."""

    @given(
        word_count=st.integers(min_value=1, max_value=100000),
    )
    @settings(max_examples=100)
    def test_no_errors_means_perfect_score(self, word_count: int) -> None:
        """Property: No errors should always result in score 100.0."""
        scorer = MQMScorer(use_glossary_weights=False)
        score = scorer.calculate_score(errors=[], word_count=word_count)
        assert score == 100.0

    @given(
        errors=error_list_strategy(max_size=10),
        word_count=st.integers(min_value=1, max_value=100000),
    )
    @settings(max_examples=100)
    def test_score_is_bounded(self, errors: list[ErrorAnnotation], word_count: int) -> None:
        """Property: Score should always be between 0 and 100."""
        scorer = MQMScorer(use_glossary_weights=False)
        score = scorer.calculate_score(errors=errors, word_count=word_count)
        assert 0.0 <= score <= 100.0

    @given(
        errors=error_list_strategy(max_size=10),
        word_count=st.integers(min_value=1, max_value=100000),
    )
    @settings(max_examples=100)
    def test_more_errors_lower_or_equal_score(
        self, errors: list[ErrorAnnotation], word_count: int
    ) -> None:
        """Property: Adding an error should not increase the score."""
        assume(len(errors) > 0)  # Need at least one error

        scorer = MQMScorer(use_glossary_weights=False)

        # Score with fewer errors
        fewer_errors = errors[:-1]
        score_fewer = scorer.calculate_score(errors=fewer_errors, word_count=word_count)

        # Score with all errors
        score_all = scorer.calculate_score(errors=errors, word_count=word_count)

        # Adding errors should never increase score
        assert score_all <= score_fewer

    @given(
        errors=error_list_strategy(max_size=5),
        word_count_small=st.integers(min_value=1, max_value=100),
        word_count_large=st.integers(min_value=101, max_value=10000),
    )
    @settings(max_examples=100)
    def test_more_words_higher_score_with_same_errors(
        self,
        errors: list[ErrorAnnotation],
        word_count_small: int,
        word_count_large: int,
    ) -> None:
        """Property: More words with same errors should give higher or equal score."""
        assume(len(errors) > 0)  # Need errors for meaningful comparison

        scorer = MQMScorer(use_glossary_weights=False)

        score_small = scorer.calculate_score(errors=errors, word_count=word_count_small)
        score_large = scorer.calculate_score(errors=errors, word_count=word_count_large)

        # More words = lower penalty density = higher score
        assert score_large >= score_small

    @given(
        word_count=st.integers(min_value=1, max_value=100000),
    )
    @settings(max_examples=50)
    def test_severity_ordering(self, word_count: int) -> None:
        """Property: Higher severity errors should result in lower scores."""
        scorer = MQMScorer(use_glossary_weights=False)

        # Create errors with different severities
        severities = [
            ErrorSeverity.NEUTRAL,
            ErrorSeverity.MINOR,
            ErrorSeverity.MAJOR,
            ErrorSeverity.CRITICAL,
        ]
        scores = []

        for severity in severities:
            error = ErrorAnnotation(
                category="accuracy",
                subcategory="mistranslation",
                severity=severity,
                location=(0, 5),
                description="Test error",
            )
            score = scorer.calculate_score(errors=[error], word_count=word_count)
            scores.append(score)

        # Scores should be in descending order (neutral -> critical)
        # (higher severity = lower score)
        for i in range(len(scores) - 1):
            assert (
                scores[i] >= scores[i + 1]
            ), f"Severity ordering violated: {severities[i]} vs {severities[i+1]}"

    @given(
        errors=error_list_strategy(max_size=10),
        word_count=st.integers(min_value=1, max_value=100000),
    )
    @settings(max_examples=100)
    def test_breakdown_consistency(self, errors: list[ErrorAnnotation], word_count: int) -> None:
        """Property: Score from breakdown should match direct calculation."""
        scorer = MQMScorer(use_glossary_weights=False)

        direct_score = scorer.calculate_score(errors=errors, word_count=word_count)
        breakdown = scorer.get_score_breakdown(errors=errors, word_count=word_count)

        assert breakdown["score"] == direct_score
        assert breakdown["word_count"] == word_count
        assert breakdown["error_count"] == len(errors)

    @given(
        errors=error_list_strategy(max_size=10),
        word_count=st.integers(min_value=1, max_value=100000),
    )
    @settings(max_examples=100)
    def test_category_penalties_sum_to_total(
        self, errors: list[ErrorAnnotation], word_count: int
    ) -> None:
        """Property: Sum of category penalties should equal total penalty."""
        scorer = MQMScorer(use_glossary_weights=False)
        breakdown = scorer.get_score_breakdown(errors=errors, word_count=word_count)

        category_sum = sum(breakdown["category_breakdown"].values())
        total_penalty = breakdown["total_penalty"]

        # Allow small floating point tolerance
        assert abs(category_sum - total_penalty) < 0.1

    @given(score=st.floats(min_value=0.0, max_value=100.0, allow_nan=False))
    @settings(max_examples=100)
    def test_quality_level_consistency(self, score: float) -> None:
        """Property: Quality level should be consistent with thresholds."""
        scorer = MQMScorer(use_glossary_weights=False)
        level = scorer.get_quality_level(score)

        if score >= 95.0:
            assert level == "excellent"
        elif score >= 90.0:
            assert level == "good"
        elif score >= 80.0:
            assert level == "acceptable"
        else:
            assert level == "poor"

    @given(
        score=st.floats(min_value=0.0, max_value=100.0, allow_nan=False),
        threshold=st.floats(min_value=0.0, max_value=100.0, allow_nan=False),
    )
    @settings(max_examples=100)
    def test_passes_threshold_consistency(self, score: float, threshold: float) -> None:
        """Property: passes_threshold should be consistent with comparison."""
        scorer = MQMScorer(use_glossary_weights=False)
        result = scorer.passes_threshold(score, threshold)

        if score >= threshold:
            assert result is True
        else:
            assert result is False

    @given(word_count=st.integers(max_value=0))
    @settings(max_examples=50)
    def test_invalid_word_count_raises_error(self, word_count: int) -> None:
        """Property: Invalid word count should raise ValueError."""
        scorer = MQMScorer(use_glossary_weights=False)

        with pytest.raises(ValueError, match="word_count must be greater than 0"):
            scorer.calculate_score(errors=[], word_count=word_count)

        with pytest.raises(ValueError, match="word_count must be greater than 0"):
            scorer.get_score_breakdown(errors=[], word_count=word_count)


@pytest.mark.unit
class TestMQMScorerCategoryWeights:
    """Property tests for category weight behavior."""

    @given(
        word_count=st.integers(min_value=1, max_value=10000),
    )
    @settings(max_examples=50)
    def test_higher_weight_category_has_more_impact(self, word_count: int) -> None:
        """Property: Higher weighted categories should have more score impact."""
        scorer = MQMScorer(use_glossary_weights=False)

        # Create same error in different categories
        high_weight_error = ErrorAnnotation(
            category="accuracy",  # weight 1.0
            subcategory="mistranslation",
            severity=ErrorSeverity.MAJOR,
            location=(0, 5),
            description="High weight error",
        )

        low_weight_error = ErrorAnnotation(
            category="style",  # weight 0.6
            subcategory="tone",
            severity=ErrorSeverity.MAJOR,
            location=(0, 5),
            description="Low weight error",
        )

        score_high_weight = scorer.calculate_score(
            errors=[high_weight_error], word_count=word_count
        )
        score_low_weight = scorer.calculate_score(errors=[low_weight_error], word_count=word_count)

        # Higher weight category error should result in lower score
        assert score_high_weight <= score_low_weight

    @given(
        weight_multiplier=st.floats(min_value=0.1, max_value=5.0, allow_nan=False),
        word_count=st.integers(min_value=100, max_value=10000),
    )
    @settings(max_examples=50)
    def test_custom_weights_affect_score(self, weight_multiplier: float, word_count: int) -> None:
        """Property: Custom weights should proportionally affect scores."""
        scorer = MQMScorer(use_glossary_weights=False)

        error = ErrorAnnotation(
            category="accuracy",
            subcategory="mistranslation",
            severity=ErrorSeverity.MAJOR,
            location=(0, 5),
            description="Test error",
        )

        # Default weight (1.0)
        score_default = scorer.calculate_score(
            errors=[error],
            word_count=word_count,
            custom_weights={"accuracy": 1.0},
        )

        # Custom weight
        score_custom = scorer.calculate_score(
            errors=[error],
            word_count=word_count,
            custom_weights={"accuracy": weight_multiplier},
        )

        # Higher weight should give lower score (more penalty)
        if weight_multiplier > 1.0:
            assert score_custom <= score_default
        elif weight_multiplier < 1.0:
            assert score_custom >= score_default
        # Equal weight should give equal score
        else:
            assert abs(score_custom - score_default) < 0.01


@pytest.mark.unit
class TestMQMScorerEdgeCases:
    """Property tests for edge cases."""

    @given(
        word_count=st.integers(min_value=1, max_value=100),
    )
    @settings(max_examples=50)
    def test_many_critical_errors_floor_at_zero(self, word_count: int) -> None:
        """Property: Many critical errors should floor the score at 0, not go negative."""
        scorer = MQMScorer(use_glossary_weights=False)

        # Create many critical errors
        errors = [
            ErrorAnnotation(
                category="accuracy",
                subcategory="mistranslation",
                severity=ErrorSeverity.CRITICAL,
                location=(i, i + 1),
                description=f"Critical error {i}",
            )
            for i in range(50)
        ]

        score = scorer.calculate_score(errors=errors, word_count=word_count)
        assert score == 0.0  # Should floor at 0, never negative

    @given(
        word_count=st.integers(min_value=1, max_value=100000),
    )
    @settings(max_examples=50)
    def test_neutral_errors_no_impact(self, word_count: int) -> None:
        """Property: Neutral severity errors should have zero penalty impact."""
        scorer = MQMScorer(use_glossary_weights=False)

        # Create neutral errors
        neutral_errors = [
            ErrorAnnotation(
                category="accuracy",
                subcategory="info",
                severity=ErrorSeverity.NEUTRAL,
                location=(0, 5),
                description="Informational note",
            )
            for _ in range(10)
        ]

        score = scorer.calculate_score(errors=neutral_errors, word_count=word_count)
        # Neutral errors have 0 penalty, so score should be 100
        assert score == 100.0

    @given(word_count=st.integers(min_value=1, max_value=100000))
    @settings(max_examples=50)
    def test_unknown_category_uses_default_weight(self, word_count: int) -> None:
        """Property: Unknown category should use default weight (1.0)."""
        scorer = MQMScorer(use_glossary_weights=False)

        known_error = ErrorAnnotation(
            category="accuracy",  # Known category, weight 1.0
            subcategory="test",
            severity=ErrorSeverity.MAJOR,
            location=(0, 5),
            description="Known category error",
        )

        unknown_error = ErrorAnnotation(
            category="unknown_category",  # Unknown, should default to 1.0
            subcategory="test",
            severity=ErrorSeverity.MAJOR,
            location=(0, 5),
            description="Unknown category error",
        )

        score_known = scorer.calculate_score(errors=[known_error], word_count=word_count)
        score_unknown = scorer.calculate_score(errors=[unknown_error], word_count=word_count)

        # Both should have same score since both use weight 1.0
        assert score_known == score_unknown
