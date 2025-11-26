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

"""Comparative style analyzer for translation evaluation.

Compares style profiles of source and target texts to assess
how well stylistic features are preserved in translation.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .analyzer import StyleFingerprint
from .models import (
    StyleComparisonResult,
    StyleDeviation,
    StyleDeviationType,
    StyleProfile,
)

logger = logging.getLogger(__name__)


class ComparativeStyleAnalyzer:
    """Compares stylistic features between source and target texts.

    The key principle: if there's a "deviation" in the source,
    it should be preserved in the translation (it's style, not error).

    Example:
        >>> analyzer = ComparativeStyleAnalyzer()
        >>> result = analyzer.compare(
        ...     source_text="Душа его желала жить жизнью...",
        ...     translation="His soul desired to live a life...",
        ...     source_lang="ru",
        ...     target_lang="en"
        ... )
        >>> print(f"Style preservation: {result.style_preservation_score:.2f}")
    """

    def __init__(self, fingerprint: StyleFingerprint | None = None) -> None:
        """Initialize comparative analyzer.

        Args:
            fingerprint: StyleFingerprint instance (creates new if not provided)
        """
        self.fingerprint = fingerprint or StyleFingerprint()

    def compare(
        self,
        source_text: str,
        translation: str,
        source_lang: str,
        target_lang: str,
        source_profile: StyleProfile | None = None,
        target_profile: StyleProfile | None = None,
    ) -> StyleComparisonResult:
        """Compare style profiles of source and target texts.

        Args:
            source_text: Original source text
            translation: Translated text
            source_lang: Source language code
            target_lang: Target language code
            source_profile: Pre-computed source profile (optional)
            target_profile: Pre-computed target profile (optional)

        Returns:
            StyleComparisonResult with preservation metrics
        """
        # Analyze texts if profiles not provided
        if source_profile is None:
            source_profile = self.fingerprint.analyze(source_text, source_lang)

        if target_profile is None:
            target_profile = self.fingerprint.analyze(translation, target_lang)

        # If source has no significant deviations, simple comparison
        if not source_profile.has_significant_deviations:
            return StyleComparisonResult(
                style_preservation_score=1.0,
                deviation_transfer_rate=1.0,
                recommendations=["Source text has standard style - no special preservation needed"],
            )

        # Compare deviations
        preserved, lost = self._compare_deviations(
            source_profile.detected_deviations,
            target_profile.detected_deviations,
        )

        # Calculate transfer rate
        total_source = len(source_profile.detected_deviations)
        transfer_rate = len(preserved) / total_source if total_source > 0 else 1.0

        # Calculate overall preservation score
        preservation_score = self._calculate_preservation_score(
            source_profile=source_profile,
            target_profile=target_profile,
            transfer_rate=transfer_rate,
        )

        # Find new issues (not from source)
        new_errors = self._find_new_errors(
            source_profile=source_profile,
            target_profile=target_profile,
        )

        # Generate recommendations
        recommendations = self._generate_recommendations(
            preserved=preserved,
            lost=lost,
            new_errors=new_errors,
            source_profile=source_profile,
        )

        result = StyleComparisonResult(
            style_preservation_score=preservation_score,
            deviation_transfer_rate=transfer_rate,
            preserved_deviations=preserved,
            lost_deviations=lost,
            new_errors=new_errors,
            recommendations=recommendations,
        )

        logger.info(
            f"Style comparison: preservation={preservation_score:.2f}, "
            f"transfer_rate={transfer_rate:.2f}, "
            f"lost={len(lost)}, new_errors={len(new_errors)}"
        )

        return result

    def _compare_deviations(
        self,
        source_deviations: list[StyleDeviation],
        target_deviations: list[StyleDeviation],
    ) -> tuple[list[StyleDeviation], list[StyleDeviation]]:
        """Compare source and target deviations.

        Returns:
            Tuple of (preserved_deviations, lost_deviations)
        """
        preserved: list[StyleDeviation] = []
        lost: list[StyleDeviation] = []

        target_types = {d.type for d in target_deviations}

        for source_dev in source_deviations:
            # Check if similar deviation type exists in target
            if self._is_deviation_preserved(source_dev, target_types):
                preserved.append(source_dev)
            else:
                lost.append(source_dev)

        return preserved, lost

    def _is_deviation_preserved(
        self,
        source_deviation: StyleDeviation,
        target_types: set[StyleDeviationType],
    ) -> bool:
        """Check if a source deviation is preserved in target."""
        # Direct type match
        if source_deviation.type in target_types:
            return True

        # Related types that might indicate preservation
        related_types = {
            StyleDeviationType.PLEONASM: {StyleDeviationType.PLEONASM},
            StyleDeviationType.SKAZ: {
                StyleDeviationType.COLLOQUIALISM,
                StyleDeviationType.DIALECTISM,
                StyleDeviationType.FOLK_ETYMOLOGY,
            },
            StyleDeviationType.INVERSION: {StyleDeviationType.INVERSION},
            StyleDeviationType.STREAM_OF_CONSCIOUSNESS: {
                StyleDeviationType.STREAM_OF_CONSCIOUSNESS,
                StyleDeviationType.RUN_ON,
                StyleDeviationType.FRAGMENTATION,
            },
            StyleDeviationType.REGISTER_MIXING: {StyleDeviationType.REGISTER_MIXING},
        }

        related = related_types.get(source_deviation.type, set())
        return bool(related & target_types)

    def _calculate_preservation_score(
        self,
        source_profile: StyleProfile,
        target_profile: StyleProfile,
        transfer_rate: float,
    ) -> float:
        """Calculate overall style preservation score."""
        # Base score from deviation transfer rate (weight: 0.5)
        score = transfer_rate * 0.5

        # Pattern matching (weight: 0.2)
        if source_profile.detected_pattern == target_profile.detected_pattern:
            score += 0.2
        elif target_profile.detected_pattern.value != "standard":
            # At least some pattern detected
            score += 0.1

        # Deviation score similarity (weight: 0.15)
        dev_diff = abs(source_profile.deviation_score - target_profile.deviation_score)
        if dev_diff < 0.1:
            score += 0.15
        elif dev_diff < 0.2:
            score += 0.1
        elif dev_diff < 0.3:
            score += 0.05

        # Stylometric similarity (weight: 0.15)
        stylometric_sim = self._calculate_stylometric_similarity(source_profile, target_profile)
        score += stylometric_sim * 0.15

        return min(score, 1.0)

    def _calculate_stylometric_similarity(
        self,
        source_profile: StyleProfile,
        target_profile: StyleProfile,
    ) -> float:
        """Calculate similarity of stylometric features."""
        similarities: list[float] = []

        # Sentence length comparison
        if source_profile.avg_sentence_length > 0:
            len_ratio = target_profile.avg_sentence_length / source_profile.avg_sentence_length
            similarities.append(1.0 - min(abs(1.0 - len_ratio), 1.0))

        # Punctuation density comparison
        if source_profile.punctuation_density > 0:
            punct_ratio = target_profile.punctuation_density / source_profile.punctuation_density
            similarities.append(1.0 - min(abs(1.0 - punct_ratio) * 0.5, 1.0))

        return sum(similarities) / len(similarities) if similarities else 0.5

    def _find_new_errors(
        self,
        source_profile: StyleProfile,
        target_profile: StyleProfile,
    ) -> list[str]:
        """Find issues in target that don't originate from source style."""
        new_errors: list[str] = []

        # If target has much lower deviation score but has fluency issues
        # that weren't in source, those might be actual errors
        if target_profile.deviation_score < source_profile.deviation_score - 0.2:
            new_errors.append("Translation appears to have normalized some stylistic features")

        # Check for deviations in target not present in source
        source_types = {d.type for d in source_profile.detected_deviations}
        for target_dev in target_profile.detected_deviations:
            if target_dev.type not in source_types:
                # This might be a new error or new style
                if not target_dev.is_intentional:
                    new_errors.append(
                        f"New deviation in target: {target_dev.type.value} - "
                        f"may be translation error"
                    )

        return new_errors

    def _generate_recommendations(
        self,
        preserved: list[StyleDeviation],
        lost: list[StyleDeviation],
        new_errors: list[str],
        source_profile: StyleProfile,
    ) -> list[str]:
        """Generate recommendations for improving style preservation."""
        recommendations: list[str] = []

        if not lost and not new_errors:
            recommendations.append(
                "Excellent style preservation - all source stylistic features maintained"
            )
            return recommendations

        # Recommendations for lost features
        for dev in lost:
            if dev.type == StyleDeviationType.PLEONASM:
                recommendations.append(
                    "Consider preserving deliberate redundancies - they may be "
                    "intentional stylistic choices (Platanov-style)"
                )
            elif dev.type == StyleDeviationType.SKAZ:
                recommendations.append(
                    "Folk speech patterns were not preserved - consider using "
                    "colloquial expressions in target language to maintain skaz style"
                )
            elif dev.type == StyleDeviationType.INVERSION:
                recommendations.append(
                    "Non-standard word order was normalized - if source uses "
                    "inversions for effect, try to recreate this in target"
                )
            elif dev.type == StyleDeviationType.STREAM_OF_CONSCIOUSNESS:
                recommendations.append(
                    "Stream of consciousness markers were lost - preserve "
                    "long sentences and unusual punctuation patterns"
                )

        # General recommendations
        if len(lost) > len(preserved):
            recommendations.append(
                f"IMPORTANT: {len(lost)} of {len(lost) + len(preserved)} stylistic "
                f"features were lost. The translator may have 'over-corrected' "
                f"intentionally awkward or unusual style elements."
            )

        if source_profile.detected_pattern.value != "standard":
            recommendations.append(
                f"Source uses '{source_profile.detected_pattern.value}' style pattern - "
                f"ensure translation maintains this distinctive voice"
            )

        return recommendations
