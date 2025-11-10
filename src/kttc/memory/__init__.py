"""Translation Memory and Terminology Management.

Provides database-backed storage for:
- Translation Memory (TM): Reusable translation segments
- Terminology Base: Domain-specific term management
"""

from .termbase import TermEntry, TerminologyBase
from .tm import TMSegment, TranslationMemory

__all__ = ["TranslationMemory", "TMSegment", "TerminologyBase", "TermEntry"]
