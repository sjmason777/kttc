"""Unit tests for lightweight translation quality metrics.

Tests the LightweightMetrics class and MetricScores model.
"""

import pytest

from kttc.evaluation.metrics import LightweightMetrics, MetricScores


@pytest.mark.unit
class TestMetricScores:
    """Test MetricScores Pydantic model."""

    def test_create_valid_scores(self) -> None:
        """Test creating valid metric scores."""
        scores = MetricScores(
            chrf=85.0,
            bleu=60.0,
            ter=70.0,
            length_ratio=1.05,
            quality_level="excellent",
        )

        assert scores.chrf == 85.0
        assert scores.bleu == 60.0
        assert scores.ter == 70.0
        assert scores.length_ratio == 1.05
        assert scores.quality_level == "excellent"

    def test_composite_score_calculation(self) -> None:
        """Test composite score calculation."""
        scores = MetricScores(
            chrf=80.0,  # 50% weight = 40
            bleu=60.0,  # 30% weight = 18
            ter=50.0,  # 20% weight = 10
            length_ratio=1.0,
            quality_level="good",
        )

        # 40 + 18 + 10 = 68
        assert scores.composite_score == 68.0

    def test_composite_score_perfect(self) -> None:
        """Test composite score for perfect translation."""
        scores = MetricScores(
            chrf=100.0,
            bleu=100.0,
            ter=100.0,
            length_ratio=1.0,
            quality_level="excellent",
        )

        assert scores.composite_score == 100.0

    def test_composite_score_zero(self) -> None:
        """Test composite score for worst case."""
        scores = MetricScores(
            chrf=0.0,
            bleu=0.0,
            ter=0.0,
            length_ratio=1.0,
            quality_level="poor",
        )

        assert scores.composite_score == 0.0

    def test_score_bounds_validation(self) -> None:
        """Test that scores outside bounds are rejected."""
        with pytest.raises(ValueError):
            MetricScores(
                chrf=110.0,  # Over 100
                bleu=50.0,
                ter=50.0,
                length_ratio=1.0,
                quality_level="good",
            )

    def test_negative_score_rejected(self) -> None:
        """Test that negative scores are rejected."""
        with pytest.raises(ValueError):
            MetricScores(
                chrf=-5.0,  # Negative
                bleu=50.0,
                ter=50.0,
                length_ratio=1.0,
                quality_level="good",
            )

    def test_length_ratio_bounds(self) -> None:
        """Test length ratio must be non-negative."""
        with pytest.raises(ValueError):
            MetricScores(
                chrf=50.0,
                bleu=50.0,
                ter=50.0,
                length_ratio=-0.5,  # Negative
                quality_level="good",
            )


@pytest.mark.unit
class TestLightweightMetricsInitialization:
    """Test LightweightMetrics initialization."""

    def test_default_initialization(self) -> None:
        """Test default initialization."""
        metrics = LightweightMetrics()

        assert metrics.THRESHOLD_EXCELLENT == 80.0
        assert metrics.THRESHOLD_GOOD == 65.0
        assert metrics.THRESHOLD_ACCEPTABLE == 50.0

    def test_metrics_initialized(self) -> None:
        """Test that metric calculators are initialized."""
        metrics = LightweightMetrics()

        assert metrics.bleu is not None
        assert metrics.chrf is not None
        assert metrics.ter is not None


@pytest.mark.unit
class TestEvaluateMethod:
    """Test evaluate method."""

    def test_perfect_match(self) -> None:
        """Test evaluation of perfect match."""
        metrics = LightweightMetrics()

        scores = metrics.evaluate(
            translation="Hello world",
            reference="Hello world",
        )

        assert scores.chrf >= 99.0
        assert scores.bleu >= 99.0
        assert scores.quality_level == "excellent"

    def test_partial_match(self) -> None:
        """Test evaluation of partial match."""
        metrics = LightweightMetrics()

        scores = metrics.evaluate(
            translation="Hello world",
            reference="Hello there",
        )

        assert 0 < scores.chrf < 100
        assert isinstance(scores.bleu, float)
        assert isinstance(scores.ter, float)

    def test_no_match(self) -> None:
        """Test evaluation of completely different texts."""
        metrics = LightweightMetrics()

        scores = metrics.evaluate(
            translation="abcdef",
            reference="xyz123",
        )

        assert scores.chrf < 30.0
        assert scores.quality_level == "poor"

    def test_returns_metric_scores(self) -> None:
        """Test that method returns MetricScores instance."""
        metrics = LightweightMetrics()

        scores = metrics.evaluate(
            translation="Test",
            reference="Test",
        )

        assert isinstance(scores, MetricScores)

    def test_length_ratio_calculated(self) -> None:
        """Test length ratio calculation."""
        metrics = LightweightMetrics()

        scores = metrics.evaluate(
            translation="abc",
            reference="abcdef",
        )

        assert scores.length_ratio == 0.5

    def test_length_ratio_perfect(self) -> None:
        """Test length ratio for same length."""
        metrics = LightweightMetrics()

        scores = metrics.evaluate(
            translation="hello",
            reference="world",
        )

        assert scores.length_ratio == 1.0

    def test_source_parameter_optional(self) -> None:
        """Test that source parameter is optional."""
        metrics = LightweightMetrics()

        # Should work without source
        scores = metrics.evaluate(
            translation="Hello",
            reference="Hello",
        )

        assert scores.chrf > 0

        # Should also work with source
        scores_with_source = metrics.evaluate(
            translation="Hello",
            reference="Hello",
            source="Привет",
        )

        assert scores_with_source.chrf > 0


@pytest.mark.unit
class TestEvaluateBatch:
    """Test evaluate_batch method."""

    def test_batch_perfect_match(self) -> None:
        """Test batch evaluation with perfect matches."""
        metrics = LightweightMetrics()

        results = metrics.evaluate_batch(
            translations=["Hello", "World"],
            references=["Hello", "World"],
        )

        assert results["chrf"] >= 99.0
        assert results["bleu"] >= 99.0
        assert results["num_sentences"] == 2

    def test_batch_statistics(self) -> None:
        """Test batch statistics calculation."""
        metrics = LightweightMetrics()

        results = metrics.evaluate_batch(
            translations=["Hello world", "Good morning"],
            references=["Hello world", "Good morning"],
        )

        assert "chrf" in results
        assert "bleu" in results
        assert "ter" in results
        assert "avg_length_ratio" in results
        assert "num_sentences" in results
        assert "total_characters" in results
        assert "quality_level" in results

    def test_batch_mismatched_length_raises(self) -> None:
        """Test that mismatched lists raise ValueError."""
        metrics = LightweightMetrics()

        with pytest.raises(ValueError):
            metrics.evaluate_batch(
                translations=["Hello"],
                references=["Hello", "World"],  # Different length
            )

    def test_batch_empty_lists(self) -> None:
        """Test handling of empty lists raises error.

        Note: sacreBLEU doesn't support empty lists, so this raises IndexError.
        """
        metrics = LightweightMetrics()

        with pytest.raises(IndexError):
            metrics.evaluate_batch(
                translations=[],
                references=[],
            )

    def test_batch_single_item(self) -> None:
        """Test batch with single item."""
        metrics = LightweightMetrics()

        results = metrics.evaluate_batch(
            translations=["Test"],
            references=["Test"],
        )

        assert results["num_sentences"] == 1


@pytest.mark.unit
class TestGetQualityLevel:
    """Test _get_quality_level method."""

    def test_excellent_threshold(self) -> None:
        """Test excellent quality threshold."""
        metrics = LightweightMetrics()

        assert metrics._get_quality_level(85.0) == "excellent"
        assert metrics._get_quality_level(80.0) == "excellent"

    def test_good_threshold(self) -> None:
        """Test good quality threshold."""
        metrics = LightweightMetrics()

        assert metrics._get_quality_level(75.0) == "good"
        assert metrics._get_quality_level(65.0) == "good"

    def test_acceptable_threshold(self) -> None:
        """Test acceptable quality threshold."""
        metrics = LightweightMetrics()

        assert metrics._get_quality_level(55.0) == "acceptable"
        assert metrics._get_quality_level(50.0) == "acceptable"

    def test_poor_threshold(self) -> None:
        """Test poor quality threshold."""
        metrics = LightweightMetrics()

        assert metrics._get_quality_level(40.0) == "poor"
        assert metrics._get_quality_level(0.0) == "poor"

    def test_boundary_values(self) -> None:
        """Test boundary values."""
        metrics = LightweightMetrics()

        # Just below excellent
        assert metrics._get_quality_level(79.9) == "good"

        # Just below good
        assert metrics._get_quality_level(64.9) == "acceptable"

        # Just below acceptable
        assert metrics._get_quality_level(49.9) == "poor"


@pytest.mark.unit
class TestGetInterpretation:
    """Test get_interpretation method."""

    def test_excellent_interpretation(self) -> None:
        """Test interpretation for excellent quality."""
        metrics = LightweightMetrics()
        scores = MetricScores(
            chrf=90.0,
            bleu=80.0,
            ter=85.0,
            length_ratio=1.0,
            quality_level="excellent",
        )

        interpretation = metrics.get_interpretation(scores)

        assert "Excellent" in interpretation
        assert "deployment" in interpretation

    def test_good_interpretation(self) -> None:
        """Test interpretation for good quality."""
        metrics = LightweightMetrics()
        scores = MetricScores(
            chrf=70.0,
            bleu=60.0,
            ter=65.0,
            length_ratio=1.0,
            quality_level="good",
        )

        interpretation = metrics.get_interpretation(scores)

        assert "Good" in interpretation
        assert "review" in interpretation.lower()

    def test_acceptable_interpretation(self) -> None:
        """Test interpretation for acceptable quality."""
        metrics = LightweightMetrics()
        scores = MetricScores(
            chrf=55.0,
            bleu=45.0,
            ter=50.0,
            length_ratio=1.0,
            quality_level="acceptable",
        )

        interpretation = metrics.get_interpretation(scores)

        assert "Acceptable" in interpretation
        assert "review" in interpretation.lower()

    def test_poor_interpretation(self) -> None:
        """Test interpretation for poor quality."""
        metrics = LightweightMetrics()
        scores = MetricScores(
            chrf=30.0,
            bleu=20.0,
            ter=25.0,
            length_ratio=1.0,
            quality_level="poor",
        )

        interpretation = metrics.get_interpretation(scores)

        assert "Poor" in interpretation
        assert "revision" in interpretation.lower()


@pytest.mark.unit
class TestPassesDeploymentThreshold:
    """Test passes_deployment_threshold method."""

    def test_passes_both_thresholds(self) -> None:
        """Test translation that passes both thresholds."""
        metrics = LightweightMetrics()
        scores = MetricScores(
            chrf=75.0,
            bleu=55.0,
            ter=70.0,
            length_ratio=1.0,
            quality_level="good",
        )

        assert metrics.passes_deployment_threshold(scores) is True

    def test_fails_chrf_threshold(self) -> None:
        """Test translation that fails chrF threshold."""
        metrics = LightweightMetrics()
        scores = MetricScores(
            chrf=50.0,  # Below default 60
            bleu=55.0,
            ter=70.0,
            length_ratio=1.0,
            quality_level="acceptable",
        )

        assert metrics.passes_deployment_threshold(scores) is False

    def test_fails_bleu_threshold(self) -> None:
        """Test translation that fails BLEU threshold."""
        metrics = LightweightMetrics()
        scores = MetricScores(
            chrf=75.0,
            bleu=30.0,  # Below default 40
            ter=70.0,
            length_ratio=1.0,
            quality_level="good",
        )

        assert metrics.passes_deployment_threshold(scores) is False

    def test_custom_thresholds(self) -> None:
        """Test with custom thresholds."""
        metrics = LightweightMetrics()
        scores = MetricScores(
            chrf=55.0,
            bleu=35.0,
            ter=60.0,
            length_ratio=1.0,
            quality_level="acceptable",
        )

        # Should fail with defaults
        assert metrics.passes_deployment_threshold(scores) is False

        # Should pass with lower thresholds
        assert (
            metrics.passes_deployment_threshold(scores, chrf_threshold=50.0, bleu_threshold=30.0)
            is True
        )

    def test_exact_threshold_values(self) -> None:
        """Test exact threshold values (should pass)."""
        metrics = LightweightMetrics()
        scores = MetricScores(
            chrf=60.0,
            bleu=40.0,
            ter=60.0,
            length_ratio=1.0,
            quality_level="acceptable",
        )

        assert metrics.passes_deployment_threshold(scores) is True


@pytest.mark.unit
class TestMultilingualEvaluation:
    """Test evaluation across different languages."""

    def test_russian_evaluation(self) -> None:
        """Test evaluation of Russian text."""
        metrics = LightweightMetrics()

        scores = metrics.evaluate(
            translation="Привет мир",
            reference="Привет мир",
        )

        assert scores.chrf >= 99.0

    def test_chinese_evaluation(self) -> None:
        """Test evaluation of Chinese text."""
        metrics = LightweightMetrics()

        scores = metrics.evaluate(
            translation="你好世界",
            reference="你好世界",
        )

        assert scores.chrf >= 99.0

    def test_mixed_scripts(self) -> None:
        """Test evaluation with mixed scripts."""
        metrics = LightweightMetrics()

        scores = metrics.evaluate(
            translation="Hello Мир 世界",
            reference="Hello Мир 世界",
        )

        assert scores.chrf >= 99.0


@pytest.mark.unit
class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_strings(self) -> None:
        """Test handling of empty strings."""
        metrics = LightweightMetrics()

        scores = metrics.evaluate(
            translation="",
            reference="",
        )

        # Should handle without error
        assert isinstance(scores, MetricScores)

    def test_single_character(self) -> None:
        """Test with single character strings."""
        metrics = LightweightMetrics()

        scores = metrics.evaluate(
            translation="a",
            reference="a",
        )

        assert scores.chrf >= 99.0

    def test_unicode_characters(self) -> None:
        """Test with various Unicode characters."""
        metrics = LightweightMetrics()

        scores = metrics.evaluate(
            translation="emoji 😀 test",
            reference="emoji 😀 test",
        )

        assert scores.chrf >= 99.0

    def test_whitespace_only(self) -> None:
        """Test with whitespace-only strings."""
        metrics = LightweightMetrics()

        scores = metrics.evaluate(
            translation="   ",
            reference="   ",
        )

        assert isinstance(scores, MetricScores)

    def test_very_long_text(self) -> None:
        """Test with very long text."""
        metrics = LightweightMetrics()

        long_text = "word " * 1000

        scores = metrics.evaluate(
            translation=long_text,
            reference=long_text,
        )

        assert scores.chrf >= 99.0
