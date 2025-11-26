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

"""Custom MQM profiles for domain-specific quality assessment.

Provides YAML-based profile system for customizing:
- Agent selection and weights
- Severity multipliers
- Quality thresholds
- Glossary references

Example profile (legal.yaml):
    name: legal
    description: Legal document translation profile
    agents:
      - accuracy
      - terminology
      - context
    agent_weights:
      accuracy: 1.0
      terminology: 0.95
      context: 0.9
    severity_multipliers:
      critical: 25
      major: 5
      minor: 1
    quality_threshold: 97.0
    glossaries:
      - legal_en
      - legal_ru
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Check if PyYAML is available
try:
    import yaml

    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    yaml = None


@dataclass
class MQMProfile:
    """Custom MQM profile configuration.

    Attributes:
        name: Profile identifier
        description: Human-readable description
        agents: List of agent IDs to use
        agent_weights: Custom agent trust weights (0.0-1.0)
        severity_multipliers: Severity penalty multipliers
        quality_threshold: Minimum MQM score to pass (0-100)
        confidence_threshold: Minimum confidence for auto-approval
        glossaries: List of glossary names to use
        metadata: Additional profile metadata
    """

    name: str
    description: str = ""
    agents: list[str] = field(default_factory=lambda: ["accuracy", "fluency", "terminology"])
    agent_weights: dict[str, float] = field(default_factory=dict)
    severity_multipliers: dict[str, int] = field(
        default_factory=lambda: {"critical": 25, "major": 5, "minor": 1, "neutral": 0}
    )
    quality_threshold: float = 95.0
    confidence_threshold: float = 0.7
    glossaries: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> list[str]:
        """Validate profile configuration.

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        # Validate name
        if not self.name or not self.name.strip():
            errors.append("Profile name is required")

        # Validate agents
        valid_agents = {
            "accuracy",
            "fluency",
            "terminology",
            "hallucination",
            "context",
            "style_preservation",
            "fluency_russian",
            "fluency_chinese",
            "fluency_hindi",
            "fluency_persian",
            "fluency_english",
        }
        for agent in self.agents:
            if agent not in valid_agents:
                errors.append(
                    f"Unknown agent: {agent}. Valid agents: {', '.join(sorted(valid_agents))}"
                )

        # Validate agent weights
        for agent, weight in self.agent_weights.items():
            if not 0.0 <= weight <= 1.0:
                errors.append(f"Agent weight for '{agent}' must be 0.0-1.0, got {weight}")

        # Validate severity multipliers
        valid_severities = {"critical", "major", "minor", "neutral"}
        for severity, multiplier in self.severity_multipliers.items():
            if severity not in valid_severities:
                errors.append(f"Unknown severity: {severity}")
            if not isinstance(multiplier, (int, float)) or multiplier < 0:
                errors.append(f"Severity multiplier for '{severity}' must be non-negative number")

        # Validate quality threshold
        if not 0.0 <= self.quality_threshold <= 100.0:
            errors.append(f"Quality threshold must be 0-100, got {self.quality_threshold}")

        # Validate confidence threshold
        if not 0.0 <= self.confidence_threshold <= 1.0:
            errors.append(f"Confidence threshold must be 0.0-1.0, got {self.confidence_threshold}")

        return errors


# Built-in profiles
BUILTIN_PROFILES: dict[str, MQMProfile] = {
    "default": MQMProfile(
        name="default",
        description="Standard KTTC profile - balanced quality assessment",
        agents=["accuracy", "fluency", "terminology"],
        agent_weights={"accuracy": 1.0, "fluency": 0.9, "terminology": 0.85},
        severity_multipliers={"critical": 25, "major": 5, "minor": 1, "neutral": 0},
        quality_threshold=95.0,
    ),
    "strict": MQMProfile(
        name="strict",
        description="Strict profile - higher thresholds, all agents including hallucination",
        agents=["accuracy", "fluency", "terminology", "hallucination", "context"],
        agent_weights={
            "accuracy": 1.0,
            "fluency": 0.95,
            "terminology": 0.95,
            "hallucination": 1.0,
            "context": 0.9,
        },
        severity_multipliers={"critical": 30, "major": 8, "minor": 2, "neutral": 0},
        quality_threshold=98.0,
        confidence_threshold=0.85,
    ),
    "minimal": MQMProfile(
        name="minimal",
        description="Quick check profile - 3 core agents, lower threshold",
        agents=["accuracy", "fluency", "terminology"],
        agent_weights={"accuracy": 1.0, "fluency": 0.8, "terminology": 0.7},
        severity_multipliers={"critical": 25, "major": 5, "minor": 1, "neutral": 0},
        quality_threshold=90.0,
    ),
    "legal": MQMProfile(
        name="legal",
        description="Legal documents - terminology and accuracy focused",
        agents=["accuracy", "terminology", "context", "hallucination"],
        agent_weights={
            "accuracy": 1.0,
            "terminology": 1.0,
            "context": 0.95,
            "hallucination": 0.95,
            "fluency": 0.8,
        },
        severity_multipliers={"critical": 30, "major": 10, "minor": 2, "neutral": 0},
        quality_threshold=97.0,
        confidence_threshold=0.85,
        metadata={"requires_human_review": True, "domain": "legal"},
    ),
    "medical": MQMProfile(
        name="medical",
        description="Medical/pharmaceutical - safety critical, highest accuracy",
        agents=["accuracy", "terminology", "hallucination", "context"],
        agent_weights={
            "accuracy": 1.0,
            "terminology": 1.0,
            "hallucination": 1.0,
            "context": 0.9,
            "fluency": 0.85,
        },
        severity_multipliers={"critical": 50, "major": 15, "minor": 3, "neutral": 0},
        quality_threshold=98.0,
        confidence_threshold=0.9,
        metadata={"safety_critical": True, "requires_expert_review": True, "domain": "medical"},
    ),
    "marketing": MQMProfile(
        name="marketing",
        description="Marketing/creative - fluency focused, creative freedom",
        agents=["fluency", "accuracy", "context"],
        agent_weights={
            "fluency": 1.0,
            "accuracy": 0.9,
            "context": 0.85,
            "terminology": 0.7,
        },
        severity_multipliers={"critical": 20, "major": 4, "minor": 1, "neutral": 0},
        quality_threshold=93.0,
        metadata={"creative_freedom": True, "domain": "marketing"},
    ),
    "literary": MQMProfile(
        name="literary",
        description="Literary/creative writing - style preservation priority",
        agents=["style_preservation", "accuracy", "context"],
        agent_weights={
            "style_preservation": 1.0,
            "accuracy": 0.9,
            "context": 0.85,
            "fluency": 0.6,  # Lower - intentional style may break fluency rules
            "terminology": 0.5,
        },
        severity_multipliers={"critical": 15, "major": 3, "minor": 1, "neutral": 0},
        quality_threshold=88.0,
        metadata={"style_aware": True, "allow_intentional_deviations": True, "domain": "literary"},
    ),
    "technical": MQMProfile(
        name="technical",
        description="Technical documentation - accuracy and terminology focus",
        agents=["accuracy", "terminology", "fluency", "hallucination"],
        agent_weights={
            "accuracy": 1.0,
            "terminology": 0.95,
            "fluency": 0.8,
            "hallucination": 0.9,
        },
        severity_multipliers={"critical": 25, "major": 7, "minor": 2, "neutral": 0},
        quality_threshold=97.0,
        metadata={"strict_terminology": True, "domain": "technical"},
    ),
}


def load_profile(profile_name_or_path: str) -> MQMProfile:
    """Load MQM profile by name or path.

    Args:
        profile_name_or_path: Built-in profile name or path to YAML file

    Returns:
        Loaded MQMProfile

    Raises:
        ValueError: If profile not found or invalid
        ImportError: If YAML file specified but PyYAML not installed
    """
    # Check if it's a built-in profile
    if profile_name_or_path in BUILTIN_PROFILES:
        logger.info(f"Using built-in profile: {profile_name_or_path}")
        return BUILTIN_PROFILES[profile_name_or_path]

    # Check if it's a file path
    profile_path = Path(profile_name_or_path)
    if profile_path.exists() and profile_path.is_file():
        return load_profile_from_file(profile_path)

    # Check in profiles directory
    profiles_dir = Path(__file__).parent.parent.parent.parent / "profiles"
    potential_paths = [
        profiles_dir / f"{profile_name_or_path}.yaml",
        profiles_dir / f"{profile_name_or_path}.yml",
    ]

    for path in potential_paths:
        if path.exists():
            return load_profile_from_file(path)

    # Profile not found
    available = list(BUILTIN_PROFILES.keys())
    raise ValueError(
        f"Profile '{profile_name_or_path}' not found. "
        f"Available built-in profiles: {', '.join(available)}"
    )


def load_profile_from_file(path: Path) -> MQMProfile:
    """Load MQM profile from YAML file.

    Args:
        path: Path to YAML profile file

    Returns:
        Loaded MQMProfile

    Raises:
        ImportError: If PyYAML not installed
        ValueError: If profile is invalid
    """
    if not YAML_AVAILABLE:
        raise ImportError(
            "PyYAML is required for custom profiles. Install with: pip install pyyaml"
        )

    logger.info(f"Loading profile from: {path}")

    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Profile file must contain a YAML dictionary, got {type(data)}")

    # Create profile from data
    profile = MQMProfile(
        name=data.get("name", path.stem),
        description=data.get("description", ""),
        agents=data.get("agents", ["accuracy", "fluency", "terminology"]),
        agent_weights=data.get("agent_weights", {}),
        severity_multipliers=data.get(
            "severity_multipliers", {"critical": 25, "major": 5, "minor": 1, "neutral": 0}
        ),
        quality_threshold=float(data.get("quality_threshold", 95.0)),
        confidence_threshold=float(data.get("confidence_threshold", 0.7)),
        glossaries=data.get("glossaries", []),
        metadata=data.get("metadata", {}),
    )

    # Validate profile
    errors = profile.validate()
    if errors:
        raise ValueError(
            f"Invalid profile '{profile.name}':\n" + "\n".join(f"  - {e}" for e in errors)
        )

    return profile


def list_available_profiles() -> list[str]:
    """List all available profile names.

    Returns:
        List of available profile names (built-in + custom from profiles dir)
    """
    available = list(BUILTIN_PROFILES.keys())

    # Check profiles directory
    profiles_dir = Path(__file__).parent.parent.parent.parent / "profiles"
    if profiles_dir.exists():
        for path in profiles_dir.glob("*.yaml"):
            name = path.stem
            if name not in available:
                available.append(name)
        for path in profiles_dir.glob("*.yml"):
            name = path.stem
            if name not in available:
                available.append(name)

    return sorted(available)


def get_profile_info(profile_name: str) -> dict[str, Any]:
    """Get detailed information about a profile.

    Args:
        profile_name: Profile name

    Returns:
        Dictionary with profile details
    """
    try:
        profile = load_profile(profile_name)
        return {
            "name": profile.name,
            "description": profile.description,
            "agents": profile.agents,
            "agent_weights": profile.agent_weights,
            "severity_multipliers": profile.severity_multipliers,
            "quality_threshold": profile.quality_threshold,
            "confidence_threshold": profile.confidence_threshold,
            "glossaries": profile.glossaries,
            "metadata": profile.metadata,
            "is_builtin": profile_name in BUILTIN_PROFILES,
        }
    except ValueError as e:
        return {"error": str(e)}
