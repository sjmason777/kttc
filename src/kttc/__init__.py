"""
KTTC - Translation Quality Assurance Platform

Autonomous multi-agent platform for translation quality assurance.
"""

__version__ = "0.1.0"
__author__ = "KTTC Development"
__email__ = "dev@kttc.ai"

from kttc.core.models import (
    TranslationTask,
    QAReport,
    ErrorAnnotation,
    ErrorSeverity,
)

__all__ = [
    "TranslationTask",
    "QAReport",
    "ErrorAnnotation",
    "ErrorSeverity",
    "__version__",
]
