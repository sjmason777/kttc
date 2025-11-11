"""Comprehensive tests for neural metrics with XCOMET integration."""

from unittest.mock import MagicMock, patch

import pytest

from kttc.metrics.neural import ErrorSpan, NeuralMetrics, NeuralMetricsResult


class TestNeuralMetricsResultCompositeScore:
    """Test composite score calculation with various combinations."""

    def test_composite_score_all_metrics(self):
        """Test composite score with all three metrics present."""
        result = NeuralMetricsResult(
            comet_score=0.80,
            kiwi_score=0.75,
            xcomet_score=0.85,
            quality_estimate="high",
        )

        # XCOMET (50%), COMET (30%), Kiwi (20%)
        # 0.85*0.5 + 0.80*0.3 + 0.75*0.2 = 0.425 + 0.24 + 0.15 = 0.815
        composite = result.get_composite_score()
        assert pytest.approx(composite, abs=0.01) == 0.815

    def test_composite_score_comet_kiwi_only(self):
        """Test composite score with COMET and Kiwi (no XCOMET)."""
        result = NeuralMetricsResult(
            comet_score=0.80,
            kiwi_score=0.75,
            quality_estimate="high",
        )

        # Renormalize: COMET (0.3), Kiwi (0.2) -> total weight 0.5
        # 0.80*0.3 + 0.75*0.2 = 0.24 + 0.15 = 0.39 / 0.5 = 0.78
        composite = result.get_composite_score()
        assert pytest.approx(composite, abs=0.01) == 0.78

    def test_composite_score_comet_xcomet_only(self):
        """Test composite score with COMET and XCOMET (no Kiwi)."""
        result = NeuralMetricsResult(
            comet_score=0.80,
            xcomet_score=0.85,
            quality_estimate="high",
        )

        # Renormalize: XCOMET (0.5), COMET (0.3) -> total weight 0.8
        # 0.85*0.5 + 0.80*0.3 = 0.425 + 0.24 = 0.665 / 0.8 = 0.83125
        composite = result.get_composite_score()
        assert pytest.approx(composite, abs=0.01) == 0.83125

    def test_composite_score_custom_weights(self):
        """Test composite score with custom weights."""
        result = NeuralMetricsResult(
            comet_score=0.80,
            kiwi_score=0.75,
            xcomet_score=0.85,
            quality_estimate="high",
        )

        # Custom weights: equal
        composite = result.get_composite_score(
            comet_weight=0.33, kiwi_weight=0.33, xcomet_weight=0.34
        )
        expected = (0.80 * 0.33 + 0.75 * 0.33 + 0.85 * 0.34) / (0.33 + 0.33 + 0.34)
        assert pytest.approx(composite, abs=0.01) == expected

    def test_composite_score_no_metrics_raises(self):
        """Test composite score raises when no metrics available."""
        result = NeuralMetricsResult(quality_estimate="low")

        with pytest.raises(ValueError, match="No neural metric scores available"):
            result.get_composite_score()

    def test_composite_score_boundary_values(self):
        """Test composite score with boundary values (0.0 and 1.0)."""
        result = NeuralMetricsResult(
            comet_score=1.0,
            kiwi_score=0.0,
            xcomet_score=0.5,
            quality_estimate="medium",
        )

        # 1.0*0.3 + 0.0*0.2 + 0.5*0.5 = 0.3 + 0.0 + 0.25 = 0.55
        composite = result.get_composite_score()
        assert pytest.approx(composite, abs=0.01) == 0.55


class TestNeuralMetricsInitialization:
    """Test NeuralMetrics initialization and error handling."""

    def test_init_default_parameters(self):
        """Test initialization with default parameters."""
        metrics = NeuralMetrics()

        assert metrics.use_gpu is False
        assert metrics.use_xcomet is True
        assert metrics.comet_model is None
        assert metrics.kiwi_model is None
        assert metrics.xcomet_model is None
        assert metrics._initialized is False

    def test_init_custom_parameters(self):
        """Test initialization with custom parameters."""
        metrics = NeuralMetrics(use_gpu=True, use_xcomet=False)

        assert metrics.use_gpu is True
        assert metrics.use_xcomet is False
        assert metrics.xcomet_model is None

    @pytest.mark.asyncio
    async def test_initialize_success_without_xcomet(self):
        """Test successful initialization without XCOMET."""
        metrics = NeuralMetrics(use_xcomet=False)

        with (
            patch("comet.download_model") as mock_download,
            patch("comet.load_from_checkpoint") as mock_load,
        ):
            mock_download.side_effect = ["/path/to/comet", "/path/to/kiwi"]
            mock_load.side_effect = [MagicMock(), MagicMock()]

            await metrics.initialize()

            assert metrics._initialized is True
            assert metrics.comet_model is not None
            assert metrics.kiwi_model is not None
            assert metrics.xcomet_model is None
            assert mock_download.call_count == 2
            assert mock_load.call_count == 2

    @pytest.mark.asyncio
    async def test_initialize_success_with_xcomet(self):
        """Test successful initialization with XCOMET."""
        metrics = NeuralMetrics(use_xcomet=True)

        with (
            patch("comet.download_model") as mock_download,
            patch("comet.load_from_checkpoint") as mock_load,
        ):
            mock_download.side_effect = ["/path/to/comet", "/path/to/kiwi", "/path/to/xcomet"]
            mock_load.side_effect = [MagicMock(), MagicMock(), MagicMock()]

            await metrics.initialize()

            assert metrics._initialized is True
            assert metrics.comet_model is not None
            assert metrics.kiwi_model is not None
            assert metrics.xcomet_model is not None
            assert mock_download.call_count == 3
            assert mock_load.call_count == 3

    @pytest.mark.asyncio
    async def test_initialize_already_initialized(self):
        """Test that initialize returns early if already initialized."""
        metrics = NeuralMetrics()
        metrics._initialized = True

        with patch("comet.download_model") as mock_download:
            await metrics.initialize()
            mock_download.assert_not_called()

    @pytest.mark.asyncio
    async def test_initialize_import_error(self):
        """Test initialization raises RuntimeError on ImportError."""
        metrics = NeuralMetrics()

        with patch.dict("sys.modules", {"comet": None}):
            with pytest.raises(RuntimeError, match="Failed to import COMET library"):
                await metrics.initialize()

    @pytest.mark.asyncio
    async def test_initialize_generic_error(self):
        """Test initialization raises RuntimeError on generic error."""
        metrics = NeuralMetrics()

        with patch("comet.download_model", side_effect=Exception("Network error")):
            with pytest.raises(RuntimeError, match="Failed to initialize neural metrics"):
                await metrics.initialize()


class TestNeuralMetricsClassifyQuality:
    """Test quality classification based on scores."""

    def test_classify_high_quality(self):
        """Test classification of high quality scores."""
        metrics = NeuralMetrics()

        assert metrics._classify_quality(0.80) == "high"
        assert metrics._classify_quality(0.90) == "high"
        assert metrics._classify_quality(1.00) == "high"

    def test_classify_medium_quality(self):
        """Test classification of medium quality scores."""
        metrics = NeuralMetrics()

        assert metrics._classify_quality(0.60) == "medium"
        assert metrics._classify_quality(0.70) == "medium"
        assert metrics._classify_quality(0.79) == "medium"

    def test_classify_low_quality(self):
        """Test classification of low quality scores."""
        metrics = NeuralMetrics()

        assert metrics._classify_quality(0.00) == "low"
        assert metrics._classify_quality(0.30) == "low"
        assert metrics._classify_quality(0.59) == "low"

    def test_classify_boundary_values(self):
        """Test classification at exact boundaries."""
        metrics = NeuralMetrics()

        # Boundary at 0.80
        assert metrics._classify_quality(0.799) == "medium"
        assert metrics._classify_quality(0.800) == "high"

        # Boundary at 0.60
        assert metrics._classify_quality(0.599) == "low"
        assert metrics._classify_quality(0.600) == "medium"


class TestNeuralMetricsEvaluateWithXCOMET:
    """Test XCOMET evaluation method."""

    @pytest.mark.asyncio
    async def test_evaluate_with_xcomet_not_initialized(self):
        """Test evaluate_with_xcomet raises when not initialized."""
        metrics = NeuralMetrics(use_xcomet=True)

        with pytest.raises(RuntimeError, match="XCOMET not initialized"):
            await metrics.evaluate_with_xcomet("source", "translation", "reference")

    @pytest.mark.asyncio
    async def test_evaluate_with_xcomet_disabled(self):
        """Test evaluate_with_xcomet raises when XCOMET disabled."""
        metrics = NeuralMetrics(use_xcomet=False)
        metrics._initialized = True

        with pytest.raises(RuntimeError, match="XCOMET not initialized"):
            await metrics.evaluate_with_xcomet("source", "translation", "reference")

    @pytest.mark.asyncio
    async def test_evaluate_with_xcomet_success_with_spans(self):
        """Test successful XCOMET evaluation with error spans."""
        metrics = NeuralMetrics(use_xcomet=True)
        metrics._initialized = True

        # Mock XCOMET model
        mock_model = MagicMock()
        mock_results = MagicMock()
        mock_results.scores = [0.85]
        mock_results.metadata = MagicMock()
        mock_results.metadata.error_spans = [
            [
                {
                    "text": "error1",
                    "start": 0,
                    "end": 6,
                    "severity": "critical",
                    "confidence": 0.95,
                },
                {
                    "text": "error2",
                    "start": 10,
                    "end": 16,
                    "severity": "minor",
                    "confidence": 0.65,
                },
            ]
        ]
        mock_model.predict.return_value = mock_results
        metrics.xcomet_model = mock_model

        result = await metrics.evaluate_with_xcomet("Hello", "Hola error1 error2", "Hola")

        assert result.xcomet_score == 0.85
        assert result.quality_estimate == "high"
        assert len(result.error_spans) == 2
        assert result.error_spans[0].text == "error1"
        assert result.error_spans[0].severity == "critical"
        assert result.error_spans[1].text == "error2"
        assert result.error_spans[1].severity == "minor"
        assert result.comet_score is None
        assert result.kiwi_score is None

    @pytest.mark.asyncio
    async def test_evaluate_with_xcomet_success_no_spans(self):
        """Test successful XCOMET evaluation without error spans."""
        metrics = NeuralMetrics(use_xcomet=True)
        metrics._initialized = True

        # Mock XCOMET model without error spans
        mock_model = MagicMock()
        mock_results = MagicMock()
        mock_results.scores = [0.92]
        mock_model.predict.return_value = mock_results
        metrics.xcomet_model = mock_model

        result = await metrics.evaluate_with_xcomet("Hello", "Hola", "Hola")

        assert result.xcomet_score == 0.92
        assert result.quality_estimate == "high"
        assert len(result.error_spans) == 0

    @pytest.mark.asyncio
    async def test_evaluate_with_xcomet_gpu_enabled(self):
        """Test XCOMET evaluation uses GPU when enabled."""
        metrics = NeuralMetrics(use_gpu=True, use_xcomet=True)
        metrics._initialized = True

        mock_model = MagicMock()
        mock_results = MagicMock()
        mock_results.scores = [0.85]
        mock_model.predict.return_value = mock_results
        metrics.xcomet_model = mock_model

        await metrics.evaluate_with_xcomet("Hello", "Hola", "Hola")

        # Verify GPU was requested
        call_args = mock_model.predict.call_args
        assert call_args[1]["gpus"] == 1

    @pytest.mark.asyncio
    async def test_evaluate_with_xcomet_exception_handling(self):
        """Test XCOMET evaluation handles exceptions gracefully."""
        metrics = NeuralMetrics(use_xcomet=True)
        metrics._initialized = True

        # Mock XCOMET model that raises exception
        mock_model = MagicMock()
        mock_model.predict.side_effect = Exception("CUDA out of memory")
        metrics.xcomet_model = mock_model

        result = await metrics.evaluate_with_xcomet("Hello", "Hola", "Hola")

        # Should return fallback result
        assert result.xcomet_score is None
        assert result.quality_estimate == "low"
        assert len(result.error_spans) == 0


class TestNeuralMetricsEvaluateWithReference:
    """Test COMET evaluation method."""

    @pytest.mark.asyncio
    async def test_evaluate_with_reference_not_initialized(self):
        """Test evaluate_with_reference raises when not initialized."""
        metrics = NeuralMetrics()

        with pytest.raises(RuntimeError, match="Neural metrics not initialized"):
            await metrics.evaluate_with_reference("source", "translation", "reference")

    @pytest.mark.asyncio
    async def test_evaluate_with_reference_success(self):
        """Test successful COMET evaluation."""
        metrics = NeuralMetrics()
        metrics._initialized = True

        mock_model = MagicMock()
        mock_results = MagicMock()
        mock_results.scores = [0.87]
        mock_model.predict.return_value = mock_results
        metrics.comet_model = mock_model

        result = await metrics.evaluate_with_reference("Hello", "Hola", "Hola")

        assert result.comet_score == 0.87
        assert result.quality_estimate == "high"
        assert result.kiwi_score is None
        assert result.xcomet_score is None
        assert len(result.error_spans) == 0

    @pytest.mark.asyncio
    async def test_evaluate_with_reference_exception(self):
        """Test COMET evaluation handles exceptions."""
        metrics = NeuralMetrics()
        metrics._initialized = True

        mock_model = MagicMock()
        mock_model.predict.side_effect = Exception("Model error")
        metrics.comet_model = mock_model

        result = await metrics.evaluate_with_reference("Hello", "Hola", "Hola")

        assert result.comet_score is None
        assert result.quality_estimate == "low"


class TestNeuralMetricsEvaluateReferenceFree:
    """Test CometKiwi evaluation method."""

    @pytest.mark.asyncio
    async def test_evaluate_reference_free_not_initialized(self):
        """Test evaluate_reference_free raises when not initialized."""
        metrics = NeuralMetrics()

        with pytest.raises(RuntimeError, match="Neural metrics not initialized"):
            await metrics.evaluate_reference_free("source", "translation")

    @pytest.mark.asyncio
    async def test_evaluate_reference_free_success(self):
        """Test successful CometKiwi evaluation."""
        metrics = NeuralMetrics()
        metrics._initialized = True

        mock_model = MagicMock()
        mock_results = MagicMock()
        mock_results.scores = [0.73]
        mock_model.predict.return_value = mock_results
        metrics.kiwi_model = mock_model

        result = await metrics.evaluate_reference_free("Hello", "Hola")

        assert result.kiwi_score == 0.73
        assert result.quality_estimate == "medium"
        assert result.comet_score is None
        assert result.xcomet_score is None

    @pytest.mark.asyncio
    async def test_evaluate_reference_free_exception(self):
        """Test CometKiwi evaluation handles exceptions."""
        metrics = NeuralMetrics()
        metrics._initialized = True

        mock_model = MagicMock()
        mock_model.predict.side_effect = Exception("Model error")
        metrics.kiwi_model = mock_model

        result = await metrics.evaluate_reference_free("Hello", "Hola")

        assert result.kiwi_score is None
        assert result.quality_estimate == "low"


class TestNeuralMetricsEvaluateOrchestration:
    """Test main evaluate method that orchestrates all metrics."""

    @pytest.mark.asyncio
    async def test_evaluate_not_initialized(self):
        """Test evaluate raises when not initialized."""
        metrics = NeuralMetrics()

        with pytest.raises(RuntimeError, match="Neural metrics not initialized"):
            await metrics.evaluate("source", "translation")

    @pytest.mark.asyncio
    async def test_evaluate_without_reference(self):
        """Test evaluate with only source and translation (no reference)."""
        metrics = NeuralMetrics()
        metrics._initialized = True

        # Mock CometKiwi
        mock_kiwi = MagicMock()
        mock_results = MagicMock()
        mock_results.scores = [0.75]
        mock_kiwi.predict.return_value = mock_results
        metrics.kiwi_model = mock_kiwi

        result = await metrics.evaluate("Hello", "Hola")

        # Should only run CometKiwi
        assert result.kiwi_score == 0.75
        assert result.comet_score is None
        assert result.xcomet_score is None

    @pytest.mark.asyncio
    async def test_evaluate_with_reference_no_xcomet(self):
        """Test evaluate with reference but XCOMET disabled."""
        metrics = NeuralMetrics(use_xcomet=False)
        metrics._initialized = True

        # Mock COMET and CometKiwi
        mock_comet = MagicMock()
        mock_comet_results = MagicMock()
        mock_comet_results.scores = [0.87]
        mock_comet.predict.return_value = mock_comet_results
        metrics.comet_model = mock_comet

        mock_kiwi = MagicMock()
        mock_kiwi_results = MagicMock()
        mock_kiwi_results.scores = [0.75]
        mock_kiwi.predict.return_value = mock_kiwi_results
        metrics.kiwi_model = mock_kiwi

        result = await metrics.evaluate("Hello", "Hola", "Hola")

        assert result.comet_score == 0.87
        assert result.kiwi_score == 0.75
        assert result.xcomet_score is None
        assert result.quality_estimate == "high"  # From COMET

    @pytest.mark.asyncio
    async def test_evaluate_with_reference_and_xcomet(self):
        """Test evaluate with reference and XCOMET enabled."""
        metrics = NeuralMetrics(use_xcomet=True)
        metrics._initialized = True

        # Mock all three models
        mock_comet = MagicMock()
        mock_comet_results = MagicMock()
        mock_comet_results.scores = [0.87]
        mock_comet.predict.return_value = mock_comet_results
        metrics.comet_model = mock_comet

        mock_kiwi = MagicMock()
        mock_kiwi_results = MagicMock()
        mock_kiwi_results.scores = [0.75]
        mock_kiwi.predict.return_value = mock_kiwi_results
        metrics.kiwi_model = mock_kiwi

        mock_xcomet = MagicMock()
        mock_xcomet_results = MagicMock()
        mock_xcomet_results.scores = [0.90]
        mock_xcomet_results.metadata = MagicMock()
        mock_xcomet_results.metadata.error_spans = [[]]
        mock_xcomet.predict.return_value = mock_xcomet_results
        metrics.xcomet_model = mock_xcomet

        result = await metrics.evaluate("Hello", "Hola", "Hola")

        # All three metrics should be present
        assert result.comet_score == 0.87
        assert result.kiwi_score == 0.75
        assert result.xcomet_score == 0.90
        assert result.quality_estimate == "high"  # From XCOMET (preferred)

    @pytest.mark.asyncio
    async def test_evaluate_prefers_xcomet_quality_estimate(self):
        """Test that XCOMET quality estimate is preferred over COMET."""
        metrics = NeuralMetrics(use_xcomet=True)
        metrics._initialized = True

        # COMET says "medium", XCOMET says "high"
        mock_comet = MagicMock()
        mock_comet_results = MagicMock()
        mock_comet_results.scores = [0.65]  # medium
        mock_comet.predict.return_value = mock_comet_results
        metrics.comet_model = mock_comet

        mock_kiwi = MagicMock()
        mock_kiwi_results = MagicMock()
        mock_kiwi_results.scores = [0.70]
        mock_kiwi.predict.return_value = mock_kiwi_results
        metrics.kiwi_model = mock_kiwi

        mock_xcomet = MagicMock()
        mock_xcomet_results = MagicMock()
        mock_xcomet_results.scores = [0.85]  # high
        mock_xcomet_results.metadata = MagicMock()
        mock_xcomet_results.metadata.error_spans = [[]]
        mock_xcomet.predict.return_value = mock_xcomet_results
        metrics.xcomet_model = mock_xcomet

        result = await metrics.evaluate("Hello", "Hola", "Hola")

        # Should prefer XCOMET's "high" over COMET's "medium"
        assert result.quality_estimate == "high"


class TestNeuralMetricsCleanup:
    """Test cleanup method."""

    @pytest.mark.asyncio
    async def test_cleanup_resets_models(self):
        """Test cleanup resets all models and initialization state."""
        metrics = NeuralMetrics()
        metrics._initialized = True
        metrics.comet_model = MagicMock()
        metrics.kiwi_model = MagicMock()
        metrics.xcomet_model = MagicMock()

        await metrics.cleanup()

        assert metrics.comet_model is None
        assert metrics.kiwi_model is None
        assert metrics.xcomet_model is None
        assert metrics._initialized is False

    @pytest.mark.asyncio
    async def test_cleanup_can_reinitialize(self):
        """Test that metrics can be reinitialized after cleanup."""
        metrics = NeuralMetrics()
        metrics._initialized = True
        metrics.comet_model = MagicMock()

        await metrics.cleanup()
        assert metrics._initialized is False

        # Should be able to initialize again
        with (
            patch("comet.download_model") as mock_download,
            patch("comet.load_from_checkpoint") as mock_load,
        ):
            mock_download.side_effect = ["/path/to/comet", "/path/to/kiwi", "/path/to/xcomet"]
            mock_load.side_effect = [MagicMock(), MagicMock(), MagicMock()]

            await metrics.initialize()
            assert metrics._initialized is True


class TestErrorSpanEdgeCases:
    """Test ErrorSpan model with edge cases."""

    def test_error_span_zero_length(self):
        """Test error span with zero length (start == end)."""
        span = ErrorSpan(text="", start=5, end=5, severity="minor", confidence=0.5)

        assert span.start == 5
        assert span.end == 5
        assert span.text == ""

    def test_error_span_boundary_confidence(self):
        """Test error span with boundary confidence values."""
        # Minimum confidence
        span1 = ErrorSpan(text="test", start=0, end=4, severity="minor", confidence=0.0)
        assert span1.confidence == 0.0

        # Maximum confidence
        span2 = ErrorSpan(text="test", start=0, end=4, severity="critical", confidence=1.0)
        assert span2.confidence == 1.0

    def test_error_span_invalid_confidence_above_max(self):
        """Test error span validation fails for confidence > 1.0."""
        with pytest.raises(ValueError):
            ErrorSpan(text="test", start=0, end=4, severity="minor", confidence=1.5)

    def test_error_span_invalid_confidence_below_min(self):
        """Test error span validation fails for confidence < 0.0."""
        with pytest.raises(ValueError):
            ErrorSpan(text="test", start=0, end=4, severity="minor", confidence=-0.1)

    def test_error_span_invalid_start_position(self):
        """Test error span validation fails for negative start."""
        with pytest.raises(ValueError):
            ErrorSpan(text="test", start=-1, end=4, severity="minor", confidence=0.5)

    def test_error_span_all_severity_levels(self):
        """Test error span accepts all valid severity levels."""
        for severity in ["critical", "major", "minor"]:
            span = ErrorSpan(text="test", start=0, end=4, severity=severity, confidence=0.8)
            assert span.severity == severity


class TestNeuralMetricsResultValidation:
    """Test NeuralMetricsResult validation."""

    def test_result_invalid_comet_score(self):
        """Test validation fails for invalid COMET score."""
        with pytest.raises(ValueError):
            NeuralMetricsResult(comet_score=1.5, quality_estimate="high")

    def test_result_invalid_kiwi_score(self):
        """Test validation fails for invalid Kiwi score."""
        with pytest.raises(ValueError):
            NeuralMetricsResult(kiwi_score=-0.1, quality_estimate="low")

    def test_result_invalid_xcomet_score(self):
        """Test validation fails for invalid XCOMET score."""
        with pytest.raises(ValueError):
            NeuralMetricsResult(xcomet_score=2.0, quality_estimate="high")

    def test_result_invalid_quality_estimate(self):
        """Test validation fails for invalid quality estimate."""
        with pytest.raises(ValueError):
            NeuralMetricsResult(comet_score=0.8, quality_estimate="excellent")

    def test_result_valid_quality_estimates(self):
        """Test all valid quality estimates are accepted."""
        for quality in ["high", "medium", "low"]:
            result = NeuralMetricsResult(comet_score=0.8, quality_estimate=quality)
            assert result.quality_estimate == quality
