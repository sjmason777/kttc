"""Tests for Neural Metrics module."""

from unittest.mock import Mock, patch

import pytest

from kttc.metrics.neural import NeuralMetrics, NeuralMetricsResult

pytestmark = pytest.mark.asyncio


class TestNeuralMetricsResult:
    """Tests for NeuralMetricsResult model."""

    def test_create_result_with_both_scores(self):
        """Test creating result with both scores."""
        result = NeuralMetricsResult(
            comet_score=0.85,
            kiwi_score=0.80,
            quality_estimate="high",
        )

        assert result.comet_score == 0.85
        assert result.kiwi_score == 0.80
        assert result.quality_estimate == "high"

    def test_create_result_minimal(self):
        """Test creating result with minimal fields."""
        result = NeuralMetricsResult()

        assert result.comet_score is None
        assert result.kiwi_score is None
        assert result.quality_estimate is None

    def test_get_composite_score_both_metrics(self):
        """Test composite score with both metrics."""
        result = NeuralMetricsResult(
            comet_score=0.80,
            kiwi_score=0.70,
        )

        # Default weights: 60% COMET + 40% Kiwi
        composite = result.get_composite_score()
        expected = 0.6 * 0.80 + 0.4 * 0.70  # 0.48 + 0.28 = 0.76
        assert composite == pytest.approx(expected)

    def test_get_composite_score_custom_weights(self):
        """Test composite score with custom weights."""
        result = NeuralMetricsResult(
            comet_score=0.80,
            kiwi_score=0.70,
        )

        composite = result.get_composite_score(comet_weight=0.7, kiwi_weight=0.3)
        expected = 0.7 * 0.80 + 0.3 * 0.70  # 0.56 + 0.21 = 0.77
        assert composite == pytest.approx(expected)

    def test_get_composite_score_only_comet(self):
        """Test composite score with only COMET."""
        result = NeuralMetricsResult(
            comet_score=0.85,
            kiwi_score=None,
        )

        composite = result.get_composite_score()
        assert composite == pytest.approx(0.85)  # Should return just COMET score

    def test_get_composite_score_only_kiwi(self):
        """Test composite score with only Kiwi."""
        result = NeuralMetricsResult(
            comet_score=None,
            kiwi_score=0.75,
        )

        composite = result.get_composite_score()
        assert composite == pytest.approx(0.75)  # Should return just Kiwi score

    def test_get_composite_score_no_scores_error(self):
        """Test composite score with no scores raises error."""
        result = NeuralMetricsResult()

        with pytest.raises(ValueError, match="No neural metric scores available"):
            result.get_composite_score()


class TestNeuralMetrics:
    """Tests for NeuralMetrics class."""

    @pytest.fixture
    def metrics(self):
        """Create NeuralMetrics instance."""
        return NeuralMetrics(use_gpu=False)

    @pytest.fixture
    def mock_comet_model(self):
        """Create mock COMET model."""
        model = Mock()
        model.predict = Mock()
        results = Mock()
        results.scores = [0.85]
        model.predict.return_value = results
        return model

    @pytest.fixture
    def mock_kiwi_model(self):
        """Create mock CometKiwi model."""
        model = Mock()
        model.predict = Mock()
        results = Mock()
        results.scores = [0.80]
        model.predict.return_value = results
        return model

    def test_initialization(self, metrics):
        """Test metrics initialization."""
        assert metrics.use_gpu is False
        assert metrics.comet_model is None
        assert metrics.kiwi_model is None
        assert metrics._initialized is False

    def test_initialization_with_gpu(self):
        """Test initialization with GPU enabled."""
        metrics = NeuralMetrics(use_gpu=True)
        assert metrics.use_gpu is True

    async def test_initialize_success(self, metrics):
        """Test successful initialization."""
        # Mock the imports from comet
        with patch.dict("sys.modules", {"comet": Mock()}):
            from unittest.mock import MagicMock

            mock_comet_module = MagicMock()
            mock_download = MagicMock(side_effect=["/path/comet", "/path/kiwi"])
            mock_comet_model = Mock()
            mock_kiwi_model = Mock()
            mock_load = MagicMock(side_effect=[mock_comet_model, mock_kiwi_model])

            mock_comet_module.download_model = mock_download
            mock_comet_module.load_from_checkpoint = mock_load

            with patch.dict("sys.modules", {"comet": mock_comet_module}):
                await metrics.initialize()

                assert metrics._initialized is True
                assert metrics.comet_model is mock_comet_model
                assert metrics.kiwi_model is mock_kiwi_model
                assert mock_download.call_count == 2

    async def test_initialize_already_initialized(self, metrics):
        """Test initialize when already initialized."""
        metrics._initialized = True
        initial_comet = Mock()
        initial_kiwi = Mock()
        metrics.comet_model = initial_comet
        metrics.kiwi_model = initial_kiwi

        await metrics.initialize()

        # Should not change models
        assert metrics.comet_model is initial_comet
        assert metrics.kiwi_model is initial_kiwi

    async def test_initialize_import_error(self, metrics):
        """Test initialization with import error."""
        # Remove comet from sys.modules if it exists
        import sys

        old_modules = sys.modules.copy()
        if "comet" in sys.modules:
            del sys.modules["comet"]

        # Mock builtins.__import__ to raise ImportError for comet
        import builtins

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if "comet" in name:
                raise ImportError("No module named 'comet'")
            return original_import(name, *args, **kwargs)

        try:
            builtins.__import__ = mock_import

            with pytest.raises(RuntimeError, match="Failed to import COMET library"):
                await metrics.initialize()

        finally:
            builtins.__import__ = original_import
            sys.modules.update(old_modules)

    async def test_initialize_general_error(self, metrics):
        """Test initialization with general error."""
        # Mock comet module but make download_model raise exception
        with patch.dict("sys.modules", {"comet": Mock()}):
            from unittest.mock import MagicMock

            mock_comet_module = MagicMock()
            mock_comet_module.download_model = MagicMock(side_effect=Exception("Download failed"))

            with patch.dict("sys.modules", {"comet": mock_comet_module}):
                with pytest.raises(RuntimeError, match="Failed to initialize neural metrics"):
                    await metrics.initialize()

    async def test_evaluate_with_reference_success(self, metrics, mock_comet_model):
        """Test evaluation with reference."""
        metrics._initialized = True
        metrics.comet_model = mock_comet_model

        result = await metrics.evaluate_with_reference(
            source="Hello world",
            translation="Hola mundo",
            reference="Hola mundo",
        )

        assert result.comet_score == 0.85
        assert result.kiwi_score is None
        assert result.quality_estimate == "high"

        # Verify predict was called correctly
        mock_comet_model.predict.assert_called_once()
        call_args = mock_comet_model.predict.call_args
        assert call_args[0][0][0]["src"] == "Hello world"
        assert call_args[0][0][0]["mt"] == "Hola mundo"
        assert call_args[0][0][0]["ref"] == "Hola mundo"
        assert call_args[1]["gpus"] == 0  # CPU

    async def test_evaluate_with_reference_gpu(self, mock_comet_model):
        """Test evaluation with GPU enabled."""
        metrics = NeuralMetrics(use_gpu=True)
        metrics._initialized = True
        metrics.comet_model = mock_comet_model

        await metrics.evaluate_with_reference("src", "mt", "ref")

        # Verify GPU was used
        call_args = mock_comet_model.predict.call_args
        assert call_args[1]["gpus"] == 1

    async def test_evaluate_with_reference_not_initialized(self, metrics):
        """Test evaluation when not initialized."""
        with pytest.raises(RuntimeError, match="not initialized"):
            await metrics.evaluate_with_reference("src", "mt", "ref")

    async def test_evaluate_with_reference_error_handling(self, metrics, mock_comet_model):
        """Test error handling in reference-based evaluation."""
        metrics._initialized = True
        metrics.comet_model = mock_comet_model
        mock_comet_model.predict.side_effect = Exception("Model failed")

        result = await metrics.evaluate_with_reference("src", "mt", "ref")

        # Should return low quality on error
        assert result.comet_score is None
        assert result.quality_estimate == "low"

    async def test_evaluate_reference_free_success(self, metrics, mock_kiwi_model):
        """Test reference-free evaluation."""
        metrics._initialized = True
        metrics.kiwi_model = mock_kiwi_model

        result = await metrics.evaluate_reference_free(
            source="Hello world",
            translation="Hola mundo",
        )

        assert result.comet_score is None
        assert result.kiwi_score == 0.80
        assert result.quality_estimate == "high"

        # Verify predict was called correctly
        mock_kiwi_model.predict.assert_called_once()
        call_args = mock_kiwi_model.predict.call_args
        assert call_args[0][0][0]["src"] == "Hello world"
        assert call_args[0][0][0]["mt"] == "Hola mundo"
        assert "ref" not in call_args[0][0][0]  # No reference for Kiwi

    async def test_evaluate_reference_free_not_initialized(self, metrics):
        """Test reference-free evaluation when not initialized."""
        with pytest.raises(RuntimeError, match="not initialized"):
            await metrics.evaluate_reference_free("src", "mt")

    async def test_evaluate_reference_free_error_handling(self, metrics, mock_kiwi_model):
        """Test error handling in reference-free evaluation."""
        metrics._initialized = True
        metrics.kiwi_model = mock_kiwi_model
        mock_kiwi_model.predict.side_effect = Exception("Model failed")

        result = await metrics.evaluate_reference_free("src", "mt")

        # Should return low quality on error
        assert result.kiwi_score is None
        assert result.quality_estimate == "low"

    async def test_evaluate_with_reference(self, metrics, mock_comet_model, mock_kiwi_model):
        """Test combined evaluation with reference."""
        metrics._initialized = True
        metrics.comet_model = mock_comet_model
        metrics.kiwi_model = mock_kiwi_model

        result = await metrics.evaluate(
            source="Hello",
            translation="Hola",
            reference="Hola",
        )

        # Should have both scores
        assert result.comet_score == 0.85
        assert result.kiwi_score == 0.80
        assert result.quality_estimate == "high"  # From COMET

        # Both models should be called
        assert mock_comet_model.predict.called
        assert mock_kiwi_model.predict.called

    async def test_evaluate_without_reference(self, metrics, mock_kiwi_model):
        """Test evaluation without reference."""
        metrics._initialized = True
        metrics.kiwi_model = mock_kiwi_model

        result = await metrics.evaluate(
            source="Hello",
            translation="Hola",
            reference=None,
        )

        # Should only have Kiwi score
        assert result.comet_score is None
        assert result.kiwi_score == 0.80
        assert result.quality_estimate == "high"

        # Only Kiwi should be called
        assert mock_kiwi_model.predict.called

    async def test_evaluate_not_initialized(self, metrics):
        """Test evaluate when not initialized."""
        with pytest.raises(RuntimeError, match="not initialized"):
            await metrics.evaluate("src", "mt")

    def test_classify_quality_high(self, metrics):
        """Test quality classification for high score."""
        assert metrics._classify_quality(0.95) == "high"
        assert metrics._classify_quality(0.80) == "high"

    def test_classify_quality_medium(self, metrics):
        """Test quality classification for medium score."""
        assert metrics._classify_quality(0.75) == "medium"
        assert metrics._classify_quality(0.60) == "medium"

    def test_classify_quality_low(self, metrics):
        """Test quality classification for low score."""
        assert metrics._classify_quality(0.50) == "low"
        assert metrics._classify_quality(0.30) == "low"
        assert metrics._classify_quality(0.0) == "low"

    def test_classify_quality_boundaries(self, metrics):
        """Test quality classification at boundaries."""
        # Test exact boundaries
        assert metrics._classify_quality(0.80) == "high"  # >= 0.80
        assert metrics._classify_quality(0.79) == "medium"  # < 0.80
        assert metrics._classify_quality(0.60) == "medium"  # >= 0.60
        assert metrics._classify_quality(0.59) == "low"  # < 0.60

    async def test_cleanup(self, metrics):
        """Test cleanup releases models."""
        metrics._initialized = True
        metrics.comet_model = Mock()
        metrics.kiwi_model = Mock()

        await metrics.cleanup()

        assert metrics.comet_model is None
        assert metrics.kiwi_model is None
        assert metrics._initialized is False
