"""
Terminology management module for KTTC.

This module provides multi-lingual glossaries and terminology validation
for translation quality assessment.

Includes language-specific trap validators:
- RussianTrapsValidator: homonyms, paronyms, position verbs, idioms
- EnglishTrapsValidator: homophones, phrasal verbs, heteronyms, adjective order
- HindiTrapsValidator: gender traps, idioms, chandrabindu/anusvara, ergativity
- PersianTrapsValidator: false friends, ta'arof, compound verbs, idioms, diglossia
"""

from kttc.terminology.english_traps import EnglishTrapsValidator
from kttc.terminology.glossary_manager import GlossaryManager
from kttc.terminology.hindi_traps import HindiTrapsValidator
from kttc.terminology.language_validators import (
    ChineseMeasureWordValidator,
    HindiPostpositionValidator,
    LanguageValidatorFactory,
    PersianEzafeValidator,
    RussianCaseAspectValidator,
)
from kttc.terminology.persian_traps import PersianTrapsValidator
from kttc.terminology.russian_traps import RussianTrapsValidator
from kttc.terminology.term_validator import TermValidator

__all__ = [
    "GlossaryManager",
    "TermValidator",
    "RussianCaseAspectValidator",
    "RussianTrapsValidator",
    "EnglishTrapsValidator",
    "HindiTrapsValidator",
    "PersianTrapsValidator",
    "ChineseMeasureWordValidator",
    "HindiPostpositionValidator",
    "PersianEzafeValidator",
    "LanguageValidatorFactory",
]
