"""Benchmarking suite for KTTC.

Provides benchmarking and evaluation tools:
- WMT benchmark integration
- Translation Arena (ELO-based)
- Regression testing
"""

from __future__ import annotations

from benchmarks.translation_arena import TranslationArena
from benchmarks.wmt_benchmark import WMTBenchmark, quick_benchmark

__all__ = ["TranslationArena", "WMTBenchmark", "quick_benchmark"]
