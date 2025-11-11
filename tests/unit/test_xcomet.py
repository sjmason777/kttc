"""Tests for XCOMET integration."""

import pytest

from kttc.metrics.neural import ErrorSpan, NeuralMetrics, NeuralMetricsResult
from kttc.metrics.visualization import ErrorSpanVisualizer


class TestErrorSpan:
    """Test ErrorSpan model."""

    def test_error_span_creation(self):
        """Test creating an error span."""
        span = ErrorSpan(
            text="wrong word",
            start=10,
            end=20,
            severity="major",
            confidence=0.85,
        )

        assert span.text == "wrong word"
        assert span.start == 10
        assert span.end == 20
        assert span.severity == "major"
        assert span.confidence == 0.85

    def test_error_span_validation(self):
        """Test error span validation."""
        # Valid span
        ErrorSpan(text="test", start=0, end=4, severity="minor", confidence=0.5)

        # Invalid confidence (out of range)
        with pytest.raises(ValueError):
            ErrorSpan(text="test", start=0, end=4, severity="minor", confidence=1.5)

        # Invalid start (negative)
        with pytest.raises(ValueError):
            ErrorSpan(text="test", start=-1, end=4, severity="minor", confidence=0.5)


class TestNeuralMetricsResultWithXCOMET:
    """Test NeuralMetricsResult with XCOMET fields."""

    def test_result_with_xcomet_score(self):
        """Test result with XCOMET score."""
        result = NeuralMetricsResult(
            comet_score=0.85,
            kiwi_score=0.80,
            xcomet_score=0.88,
            quality_estimate="high",
            error_spans=[],
        )

        assert result.xcomet_score == 0.88
        assert result.error_spans == []

    def test_result_with_error_spans(self):
        """Test result with error spans."""
        spans = [
            ErrorSpan(text="error1", start=0, end=6, severity="critical", confidence=0.95),
            ErrorSpan(text="error2", start=10, end=16, severity="minor", confidence=0.70),
        ]

        result = NeuralMetricsResult(
            xcomet_score=0.75,
            quality_estimate="medium",
            error_spans=spans,
        )

        assert len(result.error_spans) == 2
        assert result.error_spans[0].severity == "critical"
        assert result.error_spans[1].severity == "minor"

    def test_composite_score_with_xcomet(self):
        """Test composite score calculation with XCOMET."""
        result = NeuralMetricsResult(
            comet_score=0.80,
            kiwi_score=0.75,
            xcomet_score=0.85,
            quality_estimate="high",
        )

        # XCOMET gets highest weight (0.5), COMET (0.3), Kiwi (0.2)
        # Expected: 0.85*0.5 + 0.80*0.3 + 0.75*0.2 = 0.425 + 0.24 + 0.15 = 0.815
        composite = result.get_composite_score()
        assert pytest.approx(composite, abs=0.01) == 0.815

    def test_composite_score_xcomet_only(self):
        """Test composite score with only XCOMET."""
        result = NeuralMetricsResult(xcomet_score=0.90, quality_estimate="high")

        composite = result.get_composite_score()
        assert composite == 0.90


class TestNeuralMetricsXCOMETIntegration:
    """Test NeuralMetrics class with XCOMET."""

    def test_initialization_with_xcomet_enabled(self):
        """Test initializing with XCOMET enabled."""
        metrics = NeuralMetrics(use_xcomet=True)

        assert metrics.use_xcomet is True
        assert metrics.xcomet_model is None  # Not initialized yet

    def test_initialization_with_xcomet_disabled(self):
        """Test initializing with XCOMET disabled."""
        metrics = NeuralMetrics(use_xcomet=False)

        assert metrics.use_xcomet is False
        assert metrics.xcomet_model is None


class TestErrorSpanVisualizer:
    """Test error span visualization."""

    def test_visualizer_creation(self):
        """Test creating visualizer."""
        visualizer = ErrorSpanVisualizer()
        assert visualizer is not None

    def test_format_terminal_no_spans(self):
        """Test terminal formatting with no error spans."""
        visualizer = ErrorSpanVisualizer()
        text = "This is correct text"

        result = visualizer.format_terminal(text, [])
        assert result == text

    def test_format_terminal_with_spans(self):
        """Test terminal formatting with error spans."""
        visualizer = ErrorSpanVisualizer()
        text = "This is wrong text"
        spans = [
            ErrorSpan(text="wrong", start=8, end=13, severity="major", confidence=0.85),
        ]

        result = visualizer.format_terminal(text, spans)
        # Should contain ANSI color codes
        assert "\033[" in result
        assert "wrong" in result

    def test_format_html_no_spans(self):
        """Test HTML formatting with no error spans."""
        visualizer = ErrorSpanVisualizer()
        text = "This is correct text"

        result = visualizer.format_html(text, [])
        assert result == f"<p>{text}</p>"

    def test_format_html_with_spans(self):
        """Test HTML formatting with error spans."""
        visualizer = ErrorSpanVisualizer()
        text = "This is wrong text"
        spans = [
            ErrorSpan(text="wrong", start=8, end=13, severity="critical", confidence=0.95),
        ]

        result = visualizer.format_html(text, spans)
        assert "<span" in result
        assert "#FF4444" in result  # Critical color
        assert "wrong" in result

    def test_format_markdown_no_spans(self):
        """Test Markdown formatting with no error spans."""
        visualizer = ErrorSpanVisualizer()
        text = "This is correct text"

        result = visualizer.format_markdown(text, [])
        assert text in result

    def test_format_markdown_with_spans(self):
        """Test Markdown formatting with error spans."""
        visualizer = ErrorSpanVisualizer()
        text = "This has multiple errors here"
        spans = [
            ErrorSpan(text="multiple", start=9, end=17, severity="major", confidence=0.80),
            ErrorSpan(text="errors", start=18, end=24, severity="critical", confidence=0.90),
        ]

        result = visualizer.format_markdown(text, spans)
        assert "**Translation**" in result
        assert "**Detected Errors**" in result
        assert "🔴" in result  # Critical emoji
        assert "🟡" in result  # Major emoji
        assert "multiple" in result
        assert "errors" in result

    def test_get_summary_empty(self):
        """Test summary with no error spans."""
        visualizer = ErrorSpanVisualizer()
        summary = visualizer.get_summary([])

        assert summary["total"] == 0
        assert summary["critical"] == 0
        assert summary["major"] == 0
        assert summary["minor"] == 0

    def test_get_summary_with_spans(self):
        """Test summary with multiple error spans."""
        visualizer = ErrorSpanVisualizer()
        spans = [
            ErrorSpan(text="e1", start=0, end=2, severity="critical", confidence=0.9),
            ErrorSpan(text="e2", start=5, end=7, severity="critical", confidence=0.85),
            ErrorSpan(text="e3", start=10, end=12, severity="major", confidence=0.75),
            ErrorSpan(text="e4", start=15, end=17, severity="minor", confidence=0.60),
        ]

        summary = visualizer.get_summary(spans)

        assert summary["total"] == 4
        assert summary["critical"] == 2
        assert summary["major"] == 1
        assert summary["minor"] == 1

    def test_format_terminal_multiple_spans(self):
        """Test terminal formatting with multiple overlapping concepts."""
        visualizer = ErrorSpanVisualizer()
        text = "The quick brown fox jumps"
        spans = [
            ErrorSpan(text="quick", start=4, end=9, severity="minor", confidence=0.6),
            ErrorSpan(text="brown", start=10, end=15, severity="major", confidence=0.8),
            ErrorSpan(text="jumps", start=20, end=25, severity="critical", confidence=0.95),
        ]

        result = visualizer.format_terminal(text, spans)

        # Check all words are present
        assert "quick" in result
        assert "brown" in result
        assert "jumps" in result
        # Check ANSI codes are present
        assert "\033[" in result
        assert "\033[0m" in result  # Reset code


class TestXCOMETDocumentation:
    """Test that XCOMET usage is properly documented."""

    def test_neural_metrics_docstring(self):
        """Test NeuralMetrics has XCOMET in docstring."""
        assert "XCOMET" in NeuralMetrics.__doc__

    def test_error_span_docstring(self):
        """Test ErrorSpan has proper documentation."""
        assert ErrorSpan.__doc__ is not None
        assert "error" in ErrorSpan.__doc__.lower()

    def test_visualizer_docstring(self):
        """Test ErrorSpanVisualizer has examples."""
        assert ErrorSpanVisualizer.__doc__ is not None
        assert "Example" in ErrorSpanVisualizer.__doc__
