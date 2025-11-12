"""Validation suite for KTTC quality assurance.

Provides validation tools for:
- Error detection accuracy (precision, recall, F1)
- Agent performance evaluation
- Regression testing
- Quality metrics validation
"""

from __future__ import annotations

from validation.error_detection_accuracy import (
    ErrorDetectionAccuracyTest,
    ValidationMetrics,
)

__all__ = ["ErrorDetectionAccuracyTest", "ValidationMetrics"]
