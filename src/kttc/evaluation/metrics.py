# Copyright 2025 KTTC AI (https://github.com/kttc-ai)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Lightweight translation quality metrics (CPU-based, no GPU required).

This module provides fast, reproducible metrics using sacreBLEU:
- chrF/chrF++: Character-level F-score (best for morphologically rich languages)
- BLEU: Word-level n-gram overlap (fast, widely used)
- TER: Translation Edit Rate (minimum edits needed)

These metrics run on CPU and don't require neural models or embeddings.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field
from sacrebleu import BLEU, CHRF, TER


class MetricScores(BaseModel):
    """Container for lightweight metric scores.

    All scores are normalized to 0-100 scale where higher is better
    (except TER which is inverted for consistency).
    """

    chrf: float = Field(..., description="chrF++ score (0-100, higher is better)", ge=0.0, le=100.0)
    bleu: float = Field(..., description="BLEU score (0-100, higher is better)", ge=0.0, le=100.0)
    ter: float = Field(
        ..., description="Translation Edit Rate (0-100, higher is better)", ge=0.0, le=100.0
    )
    length_ratio: float = Field(..., description="Translation/Reference length ratio", ge=0.0)
    quality_level: str = Field(
        ..., description="Overall quality level: excellent, good, acceptable, or poor"
    )

    @property
    def composite_score(self) -> float:
        """Weighted composite score combining all metrics.

        Weights:
        - chrF: 50% (most important for morphologically rich languages)
        - BLEU: 30% (widely used reference metric)
        - TER: 20% (edit distance perspective)

        Returns:
            Composite score 0-100
        """
        return round(self.chrf * 0.5 + self.bleu * 0.3 + self.ter * 0.2, 2)


class LightweightMetrics:
    """CPU-based translation quality metrics using sacreBLEU.

    Provides fast, reproducible evaluation without requiring GPU or
    heavy neural models. Perfect for:
    - Quick quality checks during development
    - CI/CD pipelines
    - Resource-constrained environments
    - Baseline comparisons

    All metrics use sacreBLEU's reference implementation for reproducibility.

    Example:
        >>> metrics = LightweightMetrics()
        >>> scores = metrics.evaluate(
        ...     translation="Hello world",
        ...     reference="Hello world"
        ... )
        >>> print(f"chrF: {scores.chrf:.2f}")
        chrF: 100.00
    """

    # Quality thresholds based on research and industry practice
    THRESHOLD_EXCELLENT = 80.0  # chrF-based
    THRESHOLD_GOOD = 65.0
    THRESHOLD_ACCEPTABLE = 50.0

    def __init__(self) -> None:
        """Initialize metric calculators with default settings.

        Uses sacreBLEU defaults:
        - BLEU: 4-gram with smoothing and effective_order for sentence-level
        - chrF: chrF++ (word order=2) for better correlation
        - TER: Normalized edit distance
        """
        self.bleu = BLEU(effective_order=True)  # Recommended for sentence-level BLEU
        self.chrf = CHRF(word_order=2)  # chrF++ with word order
        self.ter = TER()

    def evaluate(
        self,
        translation: str,
        reference: str,
        source: str | None = None,
    ) -> MetricScores:
        """Evaluate single translation against reference.

        Args:
            translation: Translated text to evaluate
            reference: Human reference translation
            source: Optional source text (for future enhancements)

        Returns:
            MetricScores with all metrics and quality level

        Example:
            >>> metrics = LightweightMetrics()
            >>> scores = metrics.evaluate(
            ...     translation="Привет мир",
            ...     reference="Привет, мир!"
            ... )
            >>> scores.quality_level
            'excellent'
        """
        # Calculate all metrics using sacreBLEU
        chrf_result = self.chrf.sentence_score(translation, [reference])
        bleu_result = self.bleu.sentence_score(translation, [reference])
        ter_result = self.ter.sentence_score(translation, [reference])

        # Length ratio (for detecting omissions/additions)
        length_ratio = len(translation) / max(len(reference), 1)

        # TER is "error rate" - invert it to match "higher is better"
        ter_inverted = max(0.0, 100.0 - ter_result.score)

        # Determine quality level based on chrF (most reliable for diverse languages)
        quality_level = self._get_quality_level(chrf_result.score)

        return MetricScores(
            chrf=round(chrf_result.score, 2),
            bleu=round(bleu_result.score, 2),
            ter=round(ter_inverted, 2),
            length_ratio=round(length_ratio, 2),
            quality_level=quality_level,
        )

    def evaluate_batch(
        self,
        translations: list[str],
        references: list[str],
        _sources: list[str] | None = None,  # Unused but kept for API compatibility
    ) -> dict[str, Any]:
        """Evaluate batch of translations (corpus-level metrics).

        Corpus-level metrics are more stable than sentence-level for
        evaluating system performance.

        Args:
            translations: List of translations
            references: List of reference translations
            sources: Optional list of source texts

        Returns:
            Dictionary with corpus-level scores and statistics

        Example:
            >>> metrics = LightweightMetrics()
            >>> results = metrics.evaluate_batch(
            ...     translations=["Hello", "World"],
            ...     references=["Hello", "World"]
            ... )
            >>> results['chrf']
            100.0
        """
        if len(translations) != len(references):
            raise ValueError("Number of translations must match number of references")

        # Corpus-level metrics
        chrf_corpus = self.chrf.corpus_score(translations, [references])
        bleu_corpus = self.bleu.corpus_score(translations, [references])
        ter_corpus = self.ter.corpus_score(translations, [references])

        # Invert TER
        ter_inverted = max(0.0, 100.0 - ter_corpus.score)

        # Statistics
        total_trans_len = sum(len(t) for t in translations)
        total_ref_len = sum(len(r) for r in references)
        avg_length_ratio = total_trans_len / max(total_ref_len, 1)

        return {
            "chrf": round(chrf_corpus.score, 2),
            "bleu": round(bleu_corpus.score, 2),
            "ter": round(ter_inverted, 2),
            "avg_length_ratio": round(avg_length_ratio, 2),
            "num_sentences": len(translations),
            "total_characters": total_trans_len,
            "quality_level": self._get_quality_level(chrf_corpus.score),
        }

    def _get_quality_level(self, chrf_score: float) -> str:
        """Determine quality level based on chrF score.

        Args:
            chrf_score: chrF score (0-100)

        Returns:
            Quality level string
        """
        if chrf_score >= self.THRESHOLD_EXCELLENT:
            return "excellent"
        if chrf_score >= self.THRESHOLD_GOOD:
            return "good"
        if chrf_score >= self.THRESHOLD_ACCEPTABLE:
            return "acceptable"
        return "poor"

    def get_interpretation(self, scores: MetricScores) -> str:
        """Get human-readable interpretation of scores.

        Args:
            scores: MetricScores to interpret

        Returns:
            Interpretation string with recommendations

        Example:
            >>> metrics = LightweightMetrics()
            >>> scores = metrics.evaluate("test", "test")
            >>> print(metrics.get_interpretation(scores))
            ✓ Excellent quality - ready for deployment
        """
        level = scores.quality_level

        if level == "excellent":
            return "✓ Excellent quality - ready for deployment"
        if level == "good":
            return "✓ Good quality - minor review recommended"
        if level == "acceptable":
            return "⚠ Acceptable quality - human review required"
        return "✗ Poor quality - significant revision needed"

    def passes_deployment_threshold(
        self,
        scores: MetricScores,
        chrf_threshold: float = 60.0,
        bleu_threshold: float = 40.0,
    ) -> bool:
        """Check if translation meets deployment thresholds.

        Based on WMT best practices:
        - chrF threshold: +4 points for deployment
        - BLEU threshold: +5 points for deployment

        Args:
            scores: MetricScores to check
            chrf_threshold: Minimum chrF score
            bleu_threshold: Minimum BLEU score

        Returns:
            True if passes both thresholds

        Example:
            >>> metrics = LightweightMetrics()
            >>> scores = metrics.evaluate("excellent", "excellent")
            >>> metrics.passes_deployment_threshold(scores)
            True
        """
        return scores.chrf >= chrf_threshold and scores.bleu >= bleu_threshold
