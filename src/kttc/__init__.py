"""
KTTC - Knowledge Translation Transmutation Core

Transforming translations into gold-standard quality.

An autonomous multi-agent platform that transmutes raw translations into
certified quality through AI-powered quality assurance.
"""

__version__ = "0.1.0"
__author__ = "KTTC Development"
__email__ = "dev@kt.tc"

from kttc.core.models import (
    ErrorAnnotation,
    ErrorSeverity,
    QAReport,
    TranslationTask,
)
from kttc.core.mqm import MQMScorer

__all__ = [
    "ErrorAnnotation",
    "ErrorSeverity",
    "MQMScorer",
    "QAReport",
    "TranslationTask",
    "__version__",
]
