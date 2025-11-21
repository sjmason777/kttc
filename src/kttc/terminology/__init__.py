"""
Terminology management module for KTTC.

This module provides multi-lingual glossaries and terminology validation
for translation quality assessment.
"""

from kttc.terminology.glossary_manager import GlossaryManager
from kttc.terminology.language_validators import (
    ChineseMeasureWordValidator,
    HindiPostpositionValidator,
    LanguageValidatorFactory,
    PersianEzafeValidator,
    RussianCaseAspectValidator,
)
from kttc.terminology.term_validator import TermValidator

__all__ = [
    "GlossaryManager",
    "TermValidator",
    "RussianCaseAspectValidator",
    "ChineseMeasureWordValidator",
    "HindiPostpositionValidator",
    "PersianEzafeValidator",
    "LanguageValidatorFactory",
]
