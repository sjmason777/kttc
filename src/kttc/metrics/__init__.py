"""Neural metrics module for translation quality evaluation.

This module provides integration with state-of-the-art neural metrics:
- COMET (reference-based quality estimation)
- CometKiwi (reference-free quality estimation)
"""

from .neural import NeuralMetrics, NeuralMetricsResult

__all__ = ["NeuralMetrics", "NeuralMetricsResult"]
