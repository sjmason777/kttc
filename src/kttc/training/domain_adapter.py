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

"""Domain Adaptation for KTTC Agents.

Provides domain-specific customization of QA agents through:
- Pattern extraction from domain corpora
- Prompt enhancement with domain knowledge
- Domain-specific error detection rules
- Terminology and style adaptation

Supports domains: legal, medical, technical, financial, general
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from kttc.agents.base import BaseAgent

logger = logging.getLogger(__name__)


class DomainPatterns(BaseModel):
    """Extracted patterns from domain-specific training data.

    Contains domain knowledge for customizing agent behavior.
    """

    domain: str = Field(description="Domain name (legal, medical, technical, etc.)")
    common_terms: list[str] = Field(
        default_factory=list, description="Frequently used domain terms"
    )
    error_patterns: dict[str, list[str]] = Field(
        default_factory=dict, description="Common error patterns by category"
    )
    style_guidelines: str = Field(default="", description="Domain-specific style requirements")
    terminology_pairs: dict[str, str] = Field(
        default_factory=dict, description="Source→Target term mappings"
    )
    severity_weights: dict[str, float] = Field(
        default_factory=dict, description="Domain-specific severity adjustments"
    )
    examples: list[dict[str, str]] = Field(
        default_factory=list, description="Example translations and errors"
    )


class DomainAdapter:
    """Domain-specific agent adaptation system.

    Customizes QA agents for specific domains (legal, medical, technical)
    by extracting patterns from training data and enhancing prompts.

    Example:
        >>> adapter = DomainAdapter(domain="legal")
        >>> adapter.add_training_sample(
        ...     source="The contract is void",
        ...     translation="El contrato es nulo",
        ...     errors=[...],
        ...     terminology={"contract": "contrato", "void": "nulo"}
        ... )
        >>> patterns = adapter.extract_patterns()
        >>> adapted_agent = adapter.adapt_agent(base_agent, patterns)
    """

    # Domain-specific prompt templates
    DOMAIN_PROMPTS = {
        "legal": """
Legal Translation Guidelines:
- Maintain precise legal terminology
- Preserve formal register and tone
- Ensure accurate translation of legal concepts
- Pay attention to jurisdiction-specific terms
- Contract clauses must be unambiguous
Common legal terms: {common_terms}
""",
        "medical": """
Medical Translation Guidelines:
- Use standardized medical terminology
- Preserve drug names (generic and brand)
- Maintain anatomical precision
- Follow medical style guides
- Critical: patient safety implications
Common medical terms: {common_terms}
""",
        "technical": """
Technical Translation Guidelines:
- Preserve technical accuracy
- Maintain consistent terminology
- Follow industry standards
- Keep technical specifications precise
- Use standard units and measurements
Common technical terms: {common_terms}
""",
        "financial": """
Financial Translation Guidelines:
- Maintain precise financial terminology
- Preserve numerical accuracy
- Use standard accounting terms
- Follow financial reporting standards
- Pay attention to regulatory terms
Common financial terms: {common_terms}
""",
        "general": """
General Translation Guidelines:
- Maintain natural flow
- Preserve intended meaning
- Use appropriate register
- Follow standard grammar
Common terms: {common_terms}
""",
    }

    def __init__(self, domain: str = "general"):
        """Initialize domain adapter.

        Args:
            domain: Domain name (legal, medical, technical, financial, general)
        """
        self.domain = domain.lower()
        self.training_samples: list[dict[str, Any]] = []

        if self.domain not in self.DOMAIN_PROMPTS:
            logger.warning(
                f"Domain '{domain}' not recognized. Using 'general'. "
                f"Available: {list(self.DOMAIN_PROMPTS.keys())}"
            )
            self.domain = "general"

        logger.info(f"DomainAdapter initialized for domain: {self.domain}")

    def add_training_sample(
        self,
        source: str,
        translation: str,
        source_lang: str,
        target_lang: str,
        errors: list[dict[str, Any]] | None = None,
        terminology: dict[str, str] | None = None,
        reference: str | None = None,
    ) -> None:
        """Add training sample to domain corpus.

        Args:
            source: Source text
            translation: Translation
            source_lang: Source language code
            target_lang: Target language code
            errors: Optional list of error annotations (gold standard)
            terminology: Optional domain-specific term pairs
            reference: Optional reference translation
        """
        self.training_samples.append(
            {
                "source": source,
                "translation": translation,
                "source_lang": source_lang,
                "target_lang": target_lang,
                "errors": errors or [],
                "terminology": terminology or {},
                "reference": reference,
            }
        )

        logger.debug(f"Added training sample ({len(self.training_samples)} total)")

    def load_training_data(self, data_path: str | Path) -> None:
        """Load training data from JSON file.

        Args:
            data_path: Path to JSON file with training samples

        Expected format:
        {
            "domain": "legal",
            "samples": [
                {
                    "source": "...",
                    "translation": "...",
                    "source_lang": "en",
                    "target_lang": "es",
                    "errors": [...],
                    "terminology": {...}
                }
            ]
        }
        """
        data_path = Path(data_path)

        if not data_path.exists():
            raise FileNotFoundError(f"Training data not found: {data_path}")

        with open(data_path) as f:
            data = json.load(f)

        # Update domain if specified
        if "domain" in data:
            self.domain = data["domain"].lower()

        # Load samples
        for sample in data.get("samples", []):
            self.add_training_sample(
                source=sample["source"],
                translation=sample["translation"],
                source_lang=sample["source_lang"],
                target_lang=sample["target_lang"],
                errors=sample.get("errors"),
                terminology=sample.get("terminology"),
                reference=sample.get("reference"),
            )

        logger.info(f"Loaded {len(self.training_samples)} training samples from {data_path}")

    def _extract_common_terms(self) -> list[str]:
        """Extract common domain terms from training samples."""
        all_source_words: list[str] = []
        for sample in self.training_samples:
            all_source_words.extend(sample["source"].split())

        source_counter = Counter(all_source_words)
        return [
            word for word, _ in source_counter.most_common(50) if len(word) > 3 and word.isalpha()
        ][:20]

    def _extract_error_patterns(self) -> dict[str, list[str]]:
        """Extract error patterns categorized by type."""
        error_patterns: dict[str, list[str]] = {}
        for sample in self.training_samples:
            for error in sample.get("errors", []):
                category = error.get("category", "other")
                if category not in error_patterns:
                    error_patterns[category] = []
                description = error.get("description", "")
                if description:
                    error_patterns[category].append(description)
        return error_patterns

    def _extract_terminology_pairs(self) -> dict[str, str]:
        """Extract terminology pairs from training samples."""
        terminology_pairs: dict[str, str] = {}
        for sample in self.training_samples:
            if sample.get("terminology"):
                terminology_pairs.update(sample["terminology"])
        return terminology_pairs

    def _calculate_severity_weights(self) -> dict[str, float]:
        """Calculate severity weights based on error distribution."""
        severity_counts: Counter[str] = Counter()
        for sample in self.training_samples:
            for error in sample.get("errors", []):
                severity_counts[error.get("severity", "minor")] += 1

        total_errors = sum(severity_counts.values())
        if total_errors == 0:
            return {}
        return {severity: count / total_errors for severity, count in severity_counts.items()}

    def _collect_examples(self) -> list[dict[str, str]]:
        """Collect example translations from training samples."""
        return [
            {
                "source": sample["source"][:100],
                "translation": sample["translation"][:100],
                "quality": "good" if len(sample.get("errors", [])) == 0 else "needs_review",
            }
            for sample in self.training_samples[:10]
        ]

    def extract_patterns(self) -> DomainPatterns:
        """Extract domain patterns from training data.

        Analyzes training samples to identify:
        - Common terminology
        - Frequent error patterns
        - Style characteristics
        - Severity distributions

        Returns:
            DomainPatterns object with extracted knowledge
        """
        if not self.training_samples:
            logger.warning("No training samples available for pattern extraction")
            return DomainPatterns(domain=self.domain)

        logger.info(f"Extracting patterns from {len(self.training_samples)} samples...")

        common_terms = self._extract_common_terms()
        error_patterns = self._extract_error_patterns()
        terminology_pairs = self._extract_terminology_pairs()
        severity_weights = self._calculate_severity_weights()
        examples = self._collect_examples()

        patterns = DomainPatterns(
            domain=self.domain,
            common_terms=common_terms,
            error_patterns=error_patterns,
            style_guidelines=self.DOMAIN_PROMPTS.get(self.domain, "").strip(),
            terminology_pairs=terminology_pairs,
            severity_weights=severity_weights,
            examples=examples,
        )

        logger.info(
            f"Extracted patterns: {len(common_terms)} terms, "
            f"{len(terminology_pairs)} term pairs, "
            f"{sum(len(v) for v in error_patterns.values())} error patterns"
        )

        return patterns

    def adapt_agent(
        self, base_agent: BaseAgent, patterns: DomainPatterns | None = None
    ) -> BaseAgent:
        """Adapt agent for domain-specific QA.

        Creates a domain-specialized version of the agent with:
        - Enhanced prompts with domain knowledge
        - Domain-specific error detection
        - Adjusted severity weights

        Args:
            base_agent: Base QA agent to adapt
            patterns: Domain patterns (extracted if not provided)

        Returns:
            Domain-adapted agent
        """
        if patterns is None:
            patterns = self.extract_patterns()

        logger.info(f"Adapting {base_agent.__class__.__name__} for domain: {self.domain}")

        # Create domain-enhanced prompt using agent's base prompt
        enhanced_prompt = self._enhance_prompt(
            base_agent.get_base_prompt(),
            patterns,
        )

        # Create adapted agent with modified prompt
        # Note: This is a simplified approach. In production, you might want to
        # create a subclass or use a more sophisticated adaptation mechanism

        # Store domain knowledge in agent (using setattr for dynamic attributes)
        setattr(base_agent, "_domain", self.domain)
        setattr(base_agent, "_domain_patterns", patterns)
        setattr(
            base_agent,
            "_original_prompt",
            base_agent.get_base_prompt(),
        )
        setattr(base_agent, "_enhanced_prompt", enhanced_prompt)

        # Override prompt method
        def get_domain_prompt() -> str:
            return enhanced_prompt

        setattr(base_agent, "get_base_prompt", get_domain_prompt)

        logger.info(f"Agent adapted with {len(patterns.common_terms)} domain terms")

        return base_agent

    def _enhance_prompt(self, base_prompt: str, patterns: DomainPatterns) -> str:
        """Enhance agent prompt with domain knowledge.

        Args:
            base_prompt: Original agent prompt
            patterns: Domain patterns

        Returns:
            Enhanced prompt with domain-specific instructions
        """
        # Add domain-specific guidelines
        domain_section = self.DOMAIN_PROMPTS.get(self.domain, "").format(
            common_terms=", ".join(patterns.common_terms[:10])
        )

        enhanced = f"""{base_prompt}

DOMAIN-SPECIFIC GUIDELINES ({patterns.domain.upper()}):
{domain_section}
"""

        # Add terminology section if available
        if patterns.terminology_pairs:
            enhanced += "\nAPPROVED TERMINOLOGY:\n"
            for source_term, target_term in list(patterns.terminology_pairs.items())[:10]:
                enhanced += f"  - {source_term} → {target_term}\n"

        # Add error pattern awareness
        if patterns.error_patterns:
            enhanced += "\nCOMMON ERROR PATTERNS IN THIS DOMAIN:\n"
            for category, pattern_list in list(patterns.error_patterns.items())[:3]:
                enhanced += (
                    f"  - {category}: Watch for {pattern_list[0] if pattern_list else 'errors'}\n"
                )

        return enhanced.strip()

    def save_patterns(self, patterns: DomainPatterns, output_path: str | Path) -> None:
        """Save extracted domain patterns to file.

        Args:
            patterns: Domain patterns to save
            output_path: Output JSON file path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(patterns.model_dump(), f, indent=2)

        logger.info(f"Domain patterns saved to {output_path}")

    def load_patterns(self, patterns_path: str | Path) -> DomainPatterns:
        """Load domain patterns from file.

        Args:
            patterns_path: Path to patterns JSON file

        Returns:
            DomainPatterns object
        """
        patterns_path = Path(patterns_path)

        if not patterns_path.exists():
            raise FileNotFoundError(f"Patterns file not found: {patterns_path}")

        with open(patterns_path) as f:
            data = json.load(f)

        patterns = DomainPatterns(**data)
        logger.info(f"Loaded domain patterns from {patterns_path}")

        return patterns


# Helper function for quick domain adaptation
async def quick_adapt(agent: BaseAgent, domain: str, training_data_path: str | Path) -> BaseAgent:
    """Quickly adapt an agent to a domain using training data.

    Args:
        agent: Agent to adapt
        domain: Domain name
        training_data_path: Path to training data JSON

    Returns:
        Domain-adapted agent
    """
    adapter = DomainAdapter(domain=domain)
    adapter.load_training_data(training_data_path)
    patterns = adapter.extract_patterns()
    return adapter.adapt_agent(agent, patterns)
