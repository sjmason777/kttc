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

"""Domain-specific cognitive fingerprints for adaptive agent selection.

Implements task-adaptive agent selection based on translation domain,
inspired by cognitive fingerprints from multi-agent thinking systems.
Different domains (technical, medical, literary, etc.) require different
agent priorities and quality thresholds.

Example:
    >>> from kttc.agents.domain_profiles import DomainDetector, get_domain_profile
    >>> detector = DomainDetector()
    >>> domain = detector.detect_domain(
    ...     source_text="The API endpoint returns JSON",
    ...     target_lang="ru"
    ... )
    >>> profile = get_domain_profile(domain)
    >>> print(profile.agent_weights)
    {'accuracy': 1.0, 'terminology': 0.95, 'fluency': 0.8}
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class DomainProfile:
    """Translation domain profile with agent priorities and thresholds.

    Attributes:
        domain_type: Domain identifier (e.g., 'technical', 'medical')
        description: Human-readable description
        complexity: Estimated complexity level (0.0-1.0)
        priority_agents: List of agent IDs in priority order
        agent_weights: Custom agent trust weights for this domain
        quality_threshold: Minimum MQM score required
        confidence_threshold: Minimum confidence for auto-approval
        require_human_review: Force human review regardless of score
        keywords: Domain-specific keywords for detection
    """

    domain_type: str
    description: str
    complexity: float
    priority_agents: list[str]
    agent_weights: dict[str, float]
    quality_threshold: float = 95.0
    confidence_threshold: float = 0.7
    require_human_review: bool = False
    keywords: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


# Domain profile definitions
DOMAIN_PROFILES: dict[str, DomainProfile] = {
    "technical": DomainProfile(
        domain_type="technical",
        description="Technical documentation, software, API docs",
        complexity=0.8,
        priority_agents=["accuracy", "terminology", "fluency"],
        agent_weights={
            "accuracy": 1.0,  # Critical: exact technical meaning
            "terminology": 0.95,  # Very important: technical terms must be correct
            "fluency": 0.8,  # Less critical: clarity over style
            "hallucination": 0.9,  # Important: no invented technical details
            "context": 0.7,
        },
        quality_threshold=97.0,  # Higher threshold for technical accuracy
        confidence_threshold=0.8,
        keywords=[
            "API",
            "endpoint",
            "function",
            "method",
            "class",
            "database",
            "server",
            "client",
            "algorithm",
            "parameter",
            "variable",
            "interface",
            "protocol",
            "authentication",
            "configuration",
            "deployment",
        ],
        metadata={"requires_consistency": True, "strict_terminology": True},
    ),
    "medical": DomainProfile(
        domain_type="medical",
        description="Medical texts, pharmaceutical, clinical",
        complexity=0.9,
        priority_agents=["accuracy", "terminology", "hallucination"],
        agent_weights={
            "accuracy": 1.0,  # Critical: patient safety
            "terminology": 1.0,  # Critical: medical terms must be exact
            "hallucination": 1.0,  # Critical: no invented medical information
            "fluency": 0.85,
            "context": 0.8,
        },
        quality_threshold=98.0,  # Highest threshold: medical accuracy critical
        confidence_threshold=0.85,
        require_human_review=True,  # Always require review for medical
        keywords=[
            "patient",
            "diagnosis",
            "treatment",
            "medication",
            "symptom",
            "disease",
            "clinical",
            "therapy",
            "dosage",
            "prescription",
            "surgery",
            "medical",
            "healthcare",
            "pharmaceutical",
            "adverse",
        ],
        metadata={"safety_critical": True, "requires_expert_review": True},
    ),
    "legal": DomainProfile(
        domain_type="legal",
        description="Legal documents, contracts, regulations",
        complexity=0.9,
        priority_agents=["accuracy", "terminology", "context"],
        agent_weights={
            "accuracy": 1.0,  # Critical: legal meaning must be preserved
            "terminology": 0.95,  # Very important: legal terms are precise
            "context": 0.9,  # Important: legal context matters
            "fluency": 0.8,
            "hallucination": 0.95,
        },
        quality_threshold=97.0,
        confidence_threshold=0.85,
        require_human_review=True,  # Legal texts need expert review
        keywords=[
            "contract",
            "agreement",
            "clause",
            "party",
            "liability",
            "jurisdiction",
            "pursuant",
            "herein",
            "thereby",
            "whereas",
            "provision",
            "statute",
            "regulation",
            "compliance",
            "litigation",
        ],
        metadata={"legal_binding": True, "requires_lawyer_review": True},
    ),
    "marketing": DomainProfile(
        domain_type="marketing",
        description="Marketing materials, advertising, promotional",
        complexity=0.7,
        priority_agents=["fluency", "accuracy", "context"],
        agent_weights={
            "fluency": 1.0,  # Critical: must sound natural and appealing
            "accuracy": 0.9,  # Important: message must be correct
            "context": 0.85,  # Important: cultural adaptation
            "terminology": 0.7,
            "hallucination": 0.8,
        },
        quality_threshold=93.0,  # Lower threshold: creativity valued
        confidence_threshold=0.7,
        keywords=[
            "brand",
            "product",
            "customer",
            "buy",
            "discount",
            "offer",
            "campaign",
            "promotion",
            "advertisement",
            "marketing",
            "sale",
            "exclusive",
            "limited",
            "premium",
        ],
        metadata={"creative_freedom": True, "cultural_adaptation": True},
    ),
    "literary": DomainProfile(
        domain_type="literary",
        description="Literary texts, creative writing, poetry",
        complexity=0.8,
        priority_agents=["style_preservation", "accuracy", "fluency", "context"],
        agent_weights={
            "style_preservation": 1.0,  # Critical: authorial voice must be preserved
            "accuracy": 0.9,  # Important: meaning must be correct
            "fluency": 0.7,  # Lower: intentional "errors" may be style
            "context": 0.85,  # Important: narrative coherence
            "terminology": 0.5,  # Less critical: creative freedom
            "hallucination": 0.6,  # Some creative interpretation acceptable
        },
        quality_threshold=88.0,  # Lower threshold: art is subjective
        confidence_threshold=0.6,
        keywords=[
            "chapter",
            "character",
            "story",
            "novel",
            "poem",
            "metaphor",
            "narrative",
            "protagonist",
            "dialogue",
            "scene",
            "plot",
            "fiction",
            "soul",
            "heart",
            "dream",
            "fate",
            "destiny",
        ],
        metadata={
            "creative_work": True,
            "subjective_quality": True,
            "style_aware": True,
            "allow_intentional_deviations": True,
        },
    ),
    # Specialized literary sub-profiles (auto-detected by StyleFingerprint)
    "literary_skaz": DomainProfile(
        domain_type="literary_skaz",
        description="Skaz narrative style (Leskov, oral storytelling voice)",
        complexity=0.9,
        priority_agents=["style_preservation", "accuracy", "context"],
        agent_weights={
            "style_preservation": 1.0,
            "accuracy": 0.85,
            "fluency": 0.4,  # Very low: folk speech "errors" are intentional
            "context": 0.8,
            "terminology": 0.4,
            "hallucination": 0.5,
        },
        quality_threshold=85.0,
        confidence_threshold=0.55,
        keywords=[],  # Detected by StyleFingerprint, not keywords
        metadata={
            "style_pattern": "skaz_narrative",
            "allow_folk_speech": True,
            "allow_dialectisms": True,
            "fluency_tolerance": 0.7,
        },
    ),
    "literary_modernist": DomainProfile(
        domain_type="literary_modernist",
        description="Modernist style (Platanov, intentional awkwardness)",
        complexity=0.95,
        priority_agents=["style_preservation", "accuracy"],
        agent_weights={
            "style_preservation": 1.0,
            "accuracy": 0.9,
            "fluency": 0.3,  # Minimal: pleonasms/inversions are intentional
            "context": 0.75,
            "terminology": 0.4,
            "hallucination": 0.5,
        },
        quality_threshold=82.0,
        confidence_threshold=0.5,
        keywords=[],
        metadata={
            "style_pattern": "modernist",
            "allow_pleonasms": True,
            "allow_inversions": True,
            "fluency_tolerance": 0.8,
        },
    ),
    "literary_stream": DomainProfile(
        domain_type="literary_stream",
        description="Stream of consciousness (Joyce, Erofeev)",
        complexity=0.95,
        priority_agents=["style_preservation", "context", "accuracy"],
        agent_weights={
            "style_preservation": 1.0,
            "accuracy": 0.8,
            "fluency": 0.2,  # Almost ignore: fragmentation is the style
            "context": 0.9,
            "terminology": 0.3,
            "hallucination": 0.4,
        },
        quality_threshold=80.0,
        confidence_threshold=0.5,
        keywords=[],
        metadata={
            "style_pattern": "stream_of_consciousness",
            "allow_fragmentation": True,
            "allow_run_on_sentences": True,
            "fluency_tolerance": 0.9,
        },
    ),
    "general": DomainProfile(
        domain_type="general",
        description="General purpose text, news, articles",
        complexity=0.6,
        priority_agents=["accuracy", "fluency", "terminology"],
        agent_weights={
            "accuracy": 1.0,
            "fluency": 0.9,
            "terminology": 0.85,
            "hallucination": 0.9,
            "context": 0.8,
        },
        quality_threshold=95.0,
        confidence_threshold=0.7,
        keywords=[],  # No specific keywords - default domain
        metadata={"balanced_approach": True},
    ),
}


class DomainDetector:
    """Automatic domain detection based on text content.

    Analyzes source text to determine the most likely translation domain,
    enabling adaptive agent selection and quality thresholds.

    Example:
        >>> detector = DomainDetector()
        >>> domain = detector.detect_domain(
        ...     source_text="The API endpoint accepts POST requests",
        ...     target_lang="ru"
        ... )
        >>> print(domain)
        'technical'
    """

    def __init__(self) -> None:
        """Initialize domain detector."""
        self.profiles = DOMAIN_PROFILES

    def detect_domain(
        self,
        source_text: str,
        target_lang: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> str:
        """Detect translation domain from source text.

        Uses keyword matching and heuristics to identify the most likely
        domain for the given text.

        Args:
            source_text: Source text to analyze
            target_lang: Target language code (optional, for future use)
            context: Additional context (optional, may contain 'domain' hint)

        Returns:
            Domain type identifier (e.g., 'technical', 'medical')

        Example:
            >>> detector = DomainDetector()
            >>> detector.detect_domain("Configure the database connection")
            'technical'
            >>> detector.detect_domain("Patient exhibits symptoms of fever")
            'medical'
        """
        # Check if domain explicitly provided in context
        if context and "domain" in context:
            explicit_domain = str(context["domain"])
            if explicit_domain in self.profiles:
                return explicit_domain

        # Normalize text for analysis
        text_lower = source_text.lower()

        # Count keyword matches for each domain
        domain_scores: dict[str, int] = {}
        for domain_type, profile in self.profiles.items():
            if domain_type == "general":
                continue  # Skip general - it's the default

            score = 0
            for keyword in profile.keywords:
                # Count occurrences of keyword
                keyword_lower = keyword.lower()
                score += len(re.findall(r"\b" + re.escape(keyword_lower) + r"\b", text_lower))

            domain_scores[domain_type] = score

        # Return domain with highest score, or 'general' if no matches
        if domain_scores:
            best_domain = max(domain_scores.items(), key=lambda x: x[1])
            if best_domain[1] > 0:  # At least one keyword match
                return best_domain[0]

        return "general"

    def get_domain_confidence(self, source_text: str, detected_domain: str) -> float:
        """Calculate confidence in domain detection.

        Args:
            source_text: Source text that was analyzed
            detected_domain: The domain that was detected

        Returns:
            Confidence score (0.0-1.0)
        """
        if detected_domain == "general":
            return 0.5  # Low confidence for general domain

        profile = self.profiles[detected_domain]
        text_lower = source_text.lower()

        # Count keyword matches
        matches = sum(
            len(re.findall(r"\b" + re.escape(kw.lower()) + r"\b", text_lower))
            for kw in profile.keywords
        )

        # Confidence based on number of matches
        # 1-2 matches: 0.6, 3-4: 0.75, 5+: 0.9
        if matches == 0:
            return 0.5
        if matches <= 2:
            return 0.6
        if matches <= 4:
            return 0.75
        return 0.9


def get_domain_profile(domain_type: str) -> DomainProfile:
    """Get domain profile by type.

    Args:
        domain_type: Domain identifier

    Returns:
        DomainProfile for the specified domain

    Raises:
        ValueError: If domain type is unknown

    Example:
        >>> profile = get_domain_profile('technical')
        >>> print(profile.quality_threshold)
        97.0
    """
    if domain_type not in DOMAIN_PROFILES:
        raise ValueError(f"Unknown domain type: {domain_type}")
    return DOMAIN_PROFILES[domain_type]


def list_available_domains() -> list[str]:
    """List all available domain types.

    Returns:
        List of domain type identifiers

    Example:
        >>> domains = list_available_domains()
        >>> 'technical' in domains
        True
    """
    return list(DOMAIN_PROFILES.keys())


# Mapping from StylePattern to domain profile
STYLE_PATTERN_TO_DOMAIN: dict[str, str] = {
    "standard": "general",
    "skaz_narrative": "literary_skaz",
    "modernist": "literary_modernist",
    "stream_of_consciousness": "literary_stream",
    "poetic": "literary",
    "colloquial": "literary",
    "archaic": "literary",
    "mixed": "literary",
}


def get_domain_for_style_pattern(style_pattern: str) -> str:
    """Get appropriate domain type for a style pattern.

    Args:
        style_pattern: Style pattern from StyleFingerprint analysis

    Returns:
        Domain type identifier

    Example:
        >>> domain = get_domain_for_style_pattern("skaz_narrative")
        >>> print(domain)
        'literary_skaz'
    """
    return STYLE_PATTERN_TO_DOMAIN.get(style_pattern, "literary")


def get_literary_profile_for_style(
    style_pattern: str,
    deviation_score: float,
) -> DomainProfile:
    """Get the most appropriate literary profile based on style analysis.

    Uses StyleFingerprint analysis results to select the best domain profile.

    Args:
        style_pattern: Detected style pattern
        deviation_score: Overall deviation score (0.0-1.0)

    Returns:
        Appropriate DomainProfile for the detected style

    Example:
        >>> profile = get_literary_profile_for_style("modernist", 0.7)
        >>> print(profile.domain_type)
        'literary_modernist'
    """
    # If low deviation, use general even if pattern detected
    if deviation_score < 0.2:
        return DOMAIN_PROFILES["general"]

    # Map style pattern to domain
    domain_type = get_domain_for_style_pattern(style_pattern)

    # Return the profile
    return DOMAIN_PROFILES.get(domain_type, DOMAIN_PROFILES["literary"])
