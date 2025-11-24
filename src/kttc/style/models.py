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

"""Data models for style analysis.

Defines the core data structures for representing stylistic features,
deviations, and profiles used in literary translation evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class StyleDeviationType(str, Enum):
    """Types of stylistic deviations that can be detected.

    These represent intentional artistic choices, not errors.
    """

    # Lexical deviations
    PLEONASM = "pleonasm"  # Deliberate redundancy (Platanov)
    NEOLOGISM = "neologism"  # Author-invented words (Leskov)
    ARCHAISM = "archaism"  # Intentional archaic language
    FOLK_ETYMOLOGY = "folk_etymology"  # Folk word transformations (Leskov)
    DIALECTISM = "dialectism"  # Regional/dialect words
    COLLOQUIALISM = "colloquialism"  # Informal/spoken language

    # Syntactic deviations
    INVERSION = "syntactic_inversion"  # Non-standard word order
    FRAGMENTATION = "fragmentation"  # Incomplete sentences
    RUN_ON = "run_on"  # Very long sentences without breaks
    ELLIPSIS = "ellipsis"  # Intentional omissions

    # Stylistic deviations
    REGISTER_MIXING = "register_mixing"  # Formal/informal mix
    STREAM_OF_CONSCIOUSNESS = "stream_of_consciousness"  # Free association
    SKAZ = "skaz"  # Oral storytelling voice (Russian narrative technique)
    FREE_INDIRECT_DISCOURSE = "free_indirect_discourse"  # Narrator/character voice blend

    # Phonetic/rhythmic
    ALLITERATION = "alliteration"  # Sound repetition
    RHYTHM_BREAK = "rhythm_break"  # Intentional rhythm disruption

    # Other
    WORDPLAY = "wordplay"  # Puns, double meanings
    IRONY_MARKER = "irony_marker"  # Stylistic irony indicators


class StylePattern(str, Enum):
    """High-level style patterns that group multiple deviation types.

    These patterns help identify the overall stylistic approach.
    """

    STANDARD = "standard"  # Standard literary language
    TECHNICAL = "technical"  # Technical documentation (CLI, API docs, README)
    SKAZ_NARRATIVE = "skaz_narrative"  # Leskov-style oral storytelling
    MODERNIST = "modernist"  # Platanov-style intentional awkwardness
    STREAM = "stream_of_consciousness"  # Joyce/Erofeev stream
    POETIC = "poetic"  # Poetry-like prose
    COLLOQUIAL = "colloquial"  # Heavy colloquial/spoken style
    ARCHAIC = "archaic"  # Deliberately archaic style
    MIXED = "mixed"  # Multiple patterns combined


@dataclass
class StyleDeviation:
    """Represents a single detected stylistic deviation.

    Attributes:
        type: Type of deviation detected
        examples: List of example occurrences in text
        locations: Character spans where deviations occur
        confidence: Detection confidence (0.0-1.0)
        interpretation: Human-readable explanation
        is_intentional: Whether this appears intentional (vs error)
    """

    type: StyleDeviationType
    examples: list[str] = field(default_factory=list)
    locations: list[tuple[int, int]] = field(default_factory=list)
    confidence: float = 0.8
    interpretation: str = ""
    is_intentional: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "type": self.type.value,
            "examples": self.examples,
            "locations": self.locations,
            "confidence": self.confidence,
            "interpretation": self.interpretation,
            "is_intentional": self.is_intentional,
        }


@dataclass
class StyleProfile:
    """Complete stylistic profile of a text.

    This is the main output of StyleFingerprint analysis.

    Attributes:
        deviation_score: Overall deviation from standard language (0.0-1.0)
        detected_pattern: Primary style pattern identified
        detected_deviations: List of specific deviations found
        lexical_diversity: Vocabulary richness metric
        avg_sentence_length: Average sentence length in words
        sentence_length_variance: Variance in sentence lengths
        punctuation_density: Punctuation marks per 100 words
        is_literary: Whether text appears to be literary
        recommended_fluency_tolerance: Suggested fluency tolerance (0.0-1.0)
        metadata: Additional analysis data
    """

    deviation_score: float = 0.0
    detected_pattern: StylePattern = StylePattern.STANDARD
    detected_deviations: list[StyleDeviation] = field(default_factory=list)

    # Stylometric features
    lexical_diversity: float = 0.0
    avg_sentence_length: float = 0.0
    sentence_length_variance: float = 0.0
    punctuation_density: float = 0.0

    # Derived properties
    is_literary: bool = False
    is_technical: bool = False  # Technical documentation (CLI, API docs, Markdown)
    recommended_fluency_tolerance: float = 0.0

    # Raw data
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def deviation_types(self) -> set[StyleDeviationType]:
        """Get set of all detected deviation types."""
        return {d.type for d in self.detected_deviations}

    @property
    def has_significant_deviations(self) -> bool:
        """Check if text has significant stylistic deviations."""
        return self.deviation_score > 0.3 or len(self.detected_deviations) >= 3

    def get_agent_weight_adjustments(self) -> dict[str, float]:
        """Get recommended agent weight adjustments based on style.

        Returns:
            Dictionary of agent category to weight multiplier.
            Values > 1.0 increase importance, < 1.0 decrease.
        """
        adjustments: dict[str, float] = {
            "accuracy": 1.0,
            "fluency": 1.0,
            "terminology": 1.0,
            "style_preservation": 1.0,
        }

        # Technical documentation: skip literary style analysis
        if self.is_technical or self.detected_pattern == StylePattern.TECHNICAL:
            adjustments["style_preservation"] = 0.5  # Reduce style weight for technical docs
            adjustments["terminology"] = 1.5  # Increase terminology importance
            return adjustments

        if not self.has_significant_deviations:
            return adjustments

        # Reduce fluency weight for texts with intentional deviations
        fluency_reduction = min(self.deviation_score * 0.5, 0.4)
        adjustments["fluency"] = 1.0 - fluency_reduction

        # Increase style preservation weight
        adjustments["style_preservation"] = 1.0 + (self.deviation_score * 0.5)

        # Pattern-specific adjustments
        if self.detected_pattern == StylePattern.SKAZ_NARRATIVE:
            adjustments["fluency"] = max(adjustments["fluency"] - 0.1, 0.3)
            adjustments["style_preservation"] = min(adjustments["style_preservation"] + 0.2, 2.0)

        elif self.detected_pattern == StylePattern.MODERNIST:
            adjustments["fluency"] = max(adjustments["fluency"] - 0.15, 0.2)
            adjustments["style_preservation"] = min(adjustments["style_preservation"] + 0.3, 2.0)

        elif self.detected_pattern == StylePattern.STREAM:
            adjustments["fluency"] = 0.2  # Almost ignore fluency
            adjustments["style_preservation"] = 1.8

        return adjustments

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "deviation_score": self.deviation_score,
            "detected_pattern": self.detected_pattern.value,
            "detected_deviations": [d.to_dict() for d in self.detected_deviations],
            "lexical_diversity": self.lexical_diversity,
            "avg_sentence_length": self.avg_sentence_length,
            "sentence_length_variance": self.sentence_length_variance,
            "punctuation_density": self.punctuation_density,
            "is_literary": self.is_literary,
            "is_technical": self.is_technical,
            "recommended_fluency_tolerance": self.recommended_fluency_tolerance,
            "has_significant_deviations": self.has_significant_deviations,
            "agent_weight_adjustments": self.get_agent_weight_adjustments(),
            "metadata": self.metadata,
        }


@dataclass
class StyleComparisonResult:
    """Result of comparing source and target style profiles.

    Attributes:
        style_preservation_score: How well style is preserved (0.0-1.0)
        deviation_transfer_rate: Percentage of source deviations preserved
        preserved_deviations: Deviations successfully transferred
        lost_deviations: Deviations lost in translation
        new_errors: New issues in target not from source
        recommendations: Suggestions for improvement
    """

    style_preservation_score: float = 0.0
    deviation_transfer_rate: float = 0.0
    preserved_deviations: list[StyleDeviation] = field(default_factory=list)
    lost_deviations: list[StyleDeviation] = field(default_factory=list)
    new_errors: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "style_preservation_score": self.style_preservation_score,
            "deviation_transfer_rate": self.deviation_transfer_rate,
            "preserved_deviations": [d.to_dict() for d in self.preserved_deviations],
            "lost_deviations": [d.to_dict() for d in self.lost_deviations],
            "new_errors": self.new_errors,
            "recommendations": self.recommendations,
        }
