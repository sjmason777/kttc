"""Tests for neural metrics module."""

import pytest

from kttc.metrics.neural import NeuralMetrics, NeuralMetricsResult

pytestmark = pytest.mark.asyncio


class TestNeuralMetricsResult:
    """Tests for NeuralMetricsResult model."""

    def test_create_result_with_both_scores(self):
        """Test creating result with both COMET and CometKiwi scores."""
        result = NeuralMetricsResult(
            comet_score=0.85,
            kiwi_score=0.80,
            quality_estimate="high",
        )

        assert result.comet_score == 0.85
        assert result.kiwi_score == 0.80
        assert result.quality_estimate == "high"

    def test_composite_score_with_both_metrics(self):
        """Test composite score calculation with both metrics."""
        result = NeuralMetricsResult(
            comet_score=0.90,
            kiwi_score=0.80,
            quality_estimate="high",
        )

        composite = result.get_composite_score(comet_weight=0.6, kiwi_weight=0.4)

        # 0.6 * 0.90 + 0.4 * 0.80 = 0.54 + 0.32 = 0.86
        assert composite == pytest.approx(0.86)

    def test_composite_score_with_only_comet(self):
        """Test composite score with only COMET score."""
        result = NeuralMetricsResult(
            comet_score=0.85,
            kiwi_score=None,
            quality_estimate="high",
        )

        composite = result.get_composite_score()
        assert composite == pytest.approx(0.85)

    def test_composite_score_with_only_kiwi(self):
        """Test composite score with only CometKiwi score."""
        result = NeuralMetricsResult(
            comet_score=None,
            kiwi_score=0.80,
            quality_estimate="medium",
        )

        composite = result.get_composite_score()
        assert composite == pytest.approx(0.80)

    def test_composite_score_with_no_scores_raises_error(self):
        """Test that composite score raises error when no scores available."""
        result = NeuralMetricsResult(
            comet_score=None,
            kiwi_score=None,
            quality_estimate="low",
        )

        with pytest.raises(ValueError, match="No neural metric scores available"):
            result.get_composite_score()


class TestNeuralMetrics:
    """Tests for NeuralMetrics class."""

    @pytest.fixture
    def metrics(self):
        """Create NeuralMetrics instance."""
        return NeuralMetrics(use_gpu=False)

    def test_initialization(self, metrics):
        """Test neural metrics initialization."""
        assert metrics.use_gpu is False
        assert metrics.comet_model is None
        assert metrics.kiwi_model is None
        assert metrics._initialized is False

    @pytest.mark.slow
    async def test_initialize_downloads_models(self, metrics):
        """Test that initialize downloads models (slow test)."""
        await metrics.initialize()

        assert metrics._initialized is True
        assert metrics.comet_model is not None
        assert metrics.kiwi_model is not None

        await metrics.cleanup()

    async def test_evaluate_without_initialization_raises_error(self, metrics):
        """Test that evaluation without initialization raises error."""
        with pytest.raises(RuntimeError, match="not initialized"):
            await metrics.evaluate(source="Hello", translation="Hola", reference="¡Hola!")

    def test_classify_quality_high(self, metrics):
        """Test quality classification for high scores."""
        assert metrics._classify_quality(0.85) == "high"
        assert metrics._classify_quality(0.90) == "high"
        assert metrics._classify_quality(1.0) == "high"

    def test_classify_quality_medium(self, metrics):
        """Test quality classification for medium scores."""
        assert metrics._classify_quality(0.75) == "medium"
        assert metrics._classify_quality(0.65) == "medium"
        assert metrics._classify_quality(0.60) == "medium"

    def test_classify_quality_low(self, metrics):
        """Test quality classification for low scores."""
        assert metrics._classify_quality(0.55) == "low"
        assert metrics._classify_quality(0.30) == "low"
        assert metrics._classify_quality(0.0) == "low"

    async def test_cleanup(self, metrics):
        """Test cleanup releases resources."""
        metrics._initialized = True
        metrics.comet_model = "mock_model"
        metrics.kiwi_model = "mock_model"

        await metrics.cleanup()

        assert metrics._initialized is False
        assert metrics.comet_model is None
        assert metrics.kiwi_model is None
