"""Unit tests for MQM profiles module.

Tests custom MQM profile system for domain-specific quality assessment.
"""

from pathlib import Path
from unittest.mock import patch

import pytest

from kttc.core.profiles import (
    BUILTIN_PROFILES,
    MQMProfile,
    get_profile_info,
    list_available_profiles,
    load_profile,
    load_profile_from_file,
)


class TestMQMProfile:
    """Tests for MQMProfile dataclass."""

    def test_default_values(self) -> None:
        """Test default values are set correctly."""
        profile = MQMProfile(name="test")

        assert profile.name == "test"
        assert profile.description == ""
        assert profile.agents == ["accuracy", "fluency", "terminology"]
        assert profile.agent_weights == {}
        assert profile.severity_multipliers == {
            "critical": 25,
            "major": 5,
            "minor": 1,
            "neutral": 0,
        }
        assert profile.quality_threshold == 95.0
        assert profile.confidence_threshold == 0.7
        assert profile.glossaries == []
        assert profile.metadata == {}

    def test_custom_values(self) -> None:
        """Test profile with custom values."""
        profile = MQMProfile(
            name="custom",
            description="Custom profile",
            agents=["accuracy", "hallucination"],
            agent_weights={"accuracy": 1.0, "hallucination": 0.9},
            severity_multipliers={"critical": 50, "major": 10, "minor": 2, "neutral": 0},
            quality_threshold=98.0,
            confidence_threshold=0.85,
            glossaries=["legal_en"],
            metadata={"domain": "legal"},
        )

        assert profile.name == "custom"
        assert profile.description == "Custom profile"
        assert profile.agents == ["accuracy", "hallucination"]
        assert profile.agent_weights == {"accuracy": 1.0, "hallucination": 0.9}
        assert profile.quality_threshold == 98.0
        assert profile.metadata == {"domain": "legal"}


class TestMQMProfileValidation:
    """Tests for MQMProfile.validate() method."""

    def test_valid_profile(self) -> None:
        """Test validation passes for valid profile."""
        profile = MQMProfile(
            name="valid",
            agents=["accuracy", "fluency"],
            agent_weights={"accuracy": 1.0, "fluency": 0.9},
            quality_threshold=95.0,
            confidence_threshold=0.7,
        )

        errors = profile.validate()
        assert errors == []

    def test_empty_name_invalid(self) -> None:
        """Test validation fails for empty name."""
        profile = MQMProfile(name="")
        errors = profile.validate()
        assert "Profile name is required" in errors

    def test_whitespace_name_invalid(self) -> None:
        """Test validation fails for whitespace-only name."""
        profile = MQMProfile(name="   ")
        errors = profile.validate()
        assert "Profile name is required" in errors

    def test_unknown_agent_invalid(self) -> None:
        """Test validation fails for unknown agent."""
        profile = MQMProfile(name="test", agents=["accuracy", "unknown_agent"])
        errors = profile.validate()
        assert any("Unknown agent: unknown_agent" in e for e in errors)

    def test_all_valid_agents(self) -> None:
        """Test all valid agents are accepted."""
        valid_agents = [
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
        ]
        profile = MQMProfile(name="test", agents=valid_agents)
        errors = profile.validate()
        # No agent-related errors
        assert not any("Unknown agent" in e for e in errors)

    def test_agent_weight_out_of_range_low(self) -> None:
        """Test validation fails for agent weight < 0."""
        profile = MQMProfile(name="test", agent_weights={"accuracy": -0.1})
        errors = profile.validate()
        assert any("must be 0.0-1.0" in e for e in errors)

    def test_agent_weight_out_of_range_high(self) -> None:
        """Test validation fails for agent weight > 1."""
        profile = MQMProfile(name="test", agent_weights={"accuracy": 1.5})
        errors = profile.validate()
        assert any("must be 0.0-1.0" in e for e in errors)

    def test_agent_weight_boundary_values(self) -> None:
        """Test validation accepts boundary values 0.0 and 1.0."""
        profile = MQMProfile(name="test", agent_weights={"accuracy": 0.0, "fluency": 1.0})
        errors = profile.validate()
        assert not any("must be 0.0-1.0" in e for e in errors)

    def test_unknown_severity_invalid(self) -> None:
        """Test validation fails for unknown severity level."""
        profile = MQMProfile(name="test", severity_multipliers={"unknown_severity": 10})
        errors = profile.validate()
        assert any("Unknown severity" in e for e in errors)

    def test_negative_severity_multiplier_invalid(self) -> None:
        """Test validation fails for negative severity multiplier."""
        profile = MQMProfile(name="test", severity_multipliers={"critical": -5})
        errors = profile.validate()
        assert any("must be non-negative" in e for e in errors)

    def test_quality_threshold_out_of_range_low(self) -> None:
        """Test validation fails for quality threshold < 0."""
        profile = MQMProfile(name="test", quality_threshold=-10.0)
        errors = profile.validate()
        assert any("must be 0-100" in e for e in errors)

    def test_quality_threshold_out_of_range_high(self) -> None:
        """Test validation fails for quality threshold > 100."""
        profile = MQMProfile(name="test", quality_threshold=150.0)
        errors = profile.validate()
        assert any("must be 0-100" in e for e in errors)

    def test_quality_threshold_boundary_values(self) -> None:
        """Test validation accepts boundary values 0.0 and 100.0."""
        profile = MQMProfile(name="test", quality_threshold=0.0)
        errors = profile.validate()
        assert not any("must be 0-100" in e for e in errors)

        profile = MQMProfile(name="test", quality_threshold=100.0)
        errors = profile.validate()
        assert not any("must be 0-100" in e for e in errors)

    def test_confidence_threshold_out_of_range_low(self) -> None:
        """Test validation fails for confidence threshold < 0."""
        profile = MQMProfile(name="test", confidence_threshold=-0.1)
        errors = profile.validate()
        assert any("must be 0.0-1.0" in e for e in errors)

    def test_confidence_threshold_out_of_range_high(self) -> None:
        """Test validation fails for confidence threshold > 1."""
        profile = MQMProfile(name="test", confidence_threshold=1.5)
        errors = profile.validate()
        assert any("must be 0.0-1.0" in e for e in errors)

    def test_multiple_validation_errors(self) -> None:
        """Test validation returns all errors."""
        profile = MQMProfile(
            name="",
            agents=["unknown"],
            agent_weights={"test": 2.0},
            quality_threshold=200.0,
            confidence_threshold=5.0,
        )
        errors = profile.validate()
        assert len(errors) >= 4  # At least 4 different errors


class TestBuiltinProfiles:
    """Tests for built-in profiles."""

    def test_builtin_profiles_exist(self) -> None:
        """Test expected built-in profiles exist."""
        expected = [
            "default",
            "strict",
            "minimal",
            "legal",
            "medical",
            "marketing",
            "literary",
            "technical",
        ]
        for name in expected:
            assert name in BUILTIN_PROFILES

    def test_builtin_profiles_are_valid(self) -> None:
        """Test all built-in profiles pass validation."""
        for name, profile in BUILTIN_PROFILES.items():
            errors = profile.validate()
            assert errors == [], f"Profile '{name}' has validation errors: {errors}"

    def test_default_profile_has_core_agents(self) -> None:
        """Test default profile uses core agents."""
        profile = BUILTIN_PROFILES["default"]
        assert "accuracy" in profile.agents
        assert "fluency" in profile.agents
        assert "terminology" in profile.agents

    def test_medical_profile_is_strict(self) -> None:
        """Test medical profile has high thresholds for safety."""
        profile = BUILTIN_PROFILES["medical"]
        assert profile.quality_threshold >= 98.0
        assert profile.severity_multipliers["critical"] >= 50
        assert profile.metadata.get("safety_critical") is True

    def test_literary_profile_has_style_agent(self) -> None:
        """Test literary profile uses style_preservation agent."""
        profile = BUILTIN_PROFILES["literary"]
        assert "style_preservation" in profile.agents


class TestLoadProfile:
    """Tests for load_profile function."""

    def test_load_builtin_profile(self) -> None:
        """Test loading built-in profile by name."""
        profile = load_profile("default")
        assert profile.name == "default"
        assert profile == BUILTIN_PROFILES["default"]

    def test_load_all_builtin_profiles(self) -> None:
        """Test loading all built-in profiles."""
        for name in BUILTIN_PROFILES:
            profile = load_profile(name)
            assert profile.name == name

    def test_load_nonexistent_profile_raises(self) -> None:
        """Test loading nonexistent profile raises ValueError."""
        with pytest.raises(ValueError, match="not found"):
            load_profile("nonexistent_profile_xyz")

    def test_load_profile_error_shows_available(self) -> None:
        """Test error message includes available profiles."""
        with pytest.raises(ValueError) as exc_info:
            load_profile("nonexistent")
        assert "Available built-in profiles" in str(exc_info.value)


class TestLoadProfileFromFile:
    """Tests for load_profile_from_file function."""

    def test_load_valid_yaml_profile(self, tmp_path: Path) -> None:
        """Test loading valid YAML profile."""
        profile_content = """
name: custom_test
description: Test profile from file
agents:
  - accuracy
  - fluency
agent_weights:
  accuracy: 1.0
  fluency: 0.9
quality_threshold: 96.0
"""
        profile_file = tmp_path / "test_profile.yaml"
        profile_file.write_text(profile_content, encoding="utf-8")

        profile = load_profile_from_file(profile_file)

        assert profile.name == "custom_test"
        assert profile.description == "Test profile from file"
        assert profile.agents == ["accuracy", "fluency"]
        assert profile.agent_weights == {"accuracy": 1.0, "fluency": 0.9}
        assert profile.quality_threshold == 96.0

    def test_load_minimal_yaml_profile(self, tmp_path: Path) -> None:
        """Test loading minimal YAML profile with defaults."""
        profile_content = "name: minimal_test"
        profile_file = tmp_path / "minimal.yaml"
        profile_file.write_text(profile_content, encoding="utf-8")

        profile = load_profile_from_file(profile_file)

        assert profile.name == "minimal_test"
        assert profile.agents == ["accuracy", "fluency", "terminology"]  # default
        assert profile.quality_threshold == 95.0  # default

    def test_load_profile_without_name_uses_filename(self, tmp_path: Path) -> None:
        """Test profile without name field uses filename as name."""
        profile_content = """
description: Profile without name
agents:
  - accuracy
"""
        profile_file = tmp_path / "my_profile.yaml"
        profile_file.write_text(profile_content, encoding="utf-8")

        profile = load_profile_from_file(profile_file)
        assert profile.name == "my_profile"

    def test_load_invalid_profile_raises(self, tmp_path: Path) -> None:
        """Test loading invalid profile raises ValueError."""
        profile_content = """
name: invalid
agents:
  - unknown_agent_xyz
quality_threshold: 200.0
"""
        profile_file = tmp_path / "invalid.yaml"
        profile_file.write_text(profile_content, encoding="utf-8")

        with pytest.raises(ValueError, match="Invalid profile"):
            load_profile_from_file(profile_file)

    def test_load_non_dict_yaml_raises(self, tmp_path: Path) -> None:
        """Test loading non-dict YAML raises ValueError."""
        profile_content = "- item1\n- item2"
        profile_file = tmp_path / "list.yaml"
        profile_file.write_text(profile_content, encoding="utf-8")

        with pytest.raises(ValueError, match="must contain a YAML dictionary"):
            load_profile_from_file(profile_file)

    def test_load_profile_without_yaml_raises(self, tmp_path: Path) -> None:
        """Test loading profile without PyYAML raises ImportError."""
        profile_file = tmp_path / "test.yaml"
        profile_file.write_text("name: test", encoding="utf-8")

        with patch("kttc.core.profiles.YAML_AVAILABLE", False):
            with pytest.raises(ImportError, match="PyYAML"):
                load_profile_from_file(profile_file)


class TestLoadProfileFromPath:
    """Tests for loading profiles from file paths."""

    def test_load_profile_from_file_path(self, tmp_path: Path) -> None:
        """Test loading profile by file path."""
        profile_content = """
name: file_path_test
description: Test loading by path
"""
        profile_file = tmp_path / "custom.yaml"
        profile_file.write_text(profile_content, encoding="utf-8")

        profile = load_profile(str(profile_file))
        assert profile.name == "file_path_test"


class TestListAvailableProfiles:
    """Tests for list_available_profiles function."""

    def test_lists_builtin_profiles(self) -> None:
        """Test function lists all built-in profiles."""
        available = list_available_profiles()
        for name in BUILTIN_PROFILES:
            assert name in available

    def test_returns_sorted_list(self) -> None:
        """Test returned list is sorted."""
        available = list_available_profiles()
        assert available == sorted(available)


class TestGetProfileInfo:
    """Tests for get_profile_info function."""

    def test_get_builtin_profile_info(self) -> None:
        """Test getting info for built-in profile."""
        info = get_profile_info("default")

        assert info["name"] == "default"
        assert "description" in info
        assert "agents" in info
        assert "agent_weights" in info
        assert "severity_multipliers" in info
        assert "quality_threshold" in info
        assert info["is_builtin"] is True

    def test_get_nonexistent_profile_returns_error(self) -> None:
        """Test getting info for nonexistent profile returns error."""
        info = get_profile_info("nonexistent_xyz")
        assert "error" in info

    def test_get_all_builtin_profiles_info(self) -> None:
        """Test getting info for all built-in profiles."""
        for name in BUILTIN_PROFILES:
            info = get_profile_info(name)
            assert info["name"] == name
            assert info["is_builtin"] is True


class TestProfileIntegration:
    """Integration tests for profile system."""

    def test_create_and_validate_custom_profile(self, tmp_path: Path) -> None:
        """Test full workflow: create, save, load, validate profile."""
        profile_content = """
name: integration_test
description: Integration test profile
agents:
  - accuracy
  - terminology
  - hallucination
agent_weights:
  accuracy: 1.0
  terminology: 0.95
  hallucination: 0.9
severity_multipliers:
  critical: 30
  major: 8
  minor: 2
  neutral: 0
quality_threshold: 97.0
confidence_threshold: 0.8
glossaries:
  - legal_glossary
metadata:
  domain: legal
  requires_review: true
"""
        profile_file = tmp_path / "integration.yaml"
        profile_file.write_text(profile_content, encoding="utf-8")

        # Load profile
        profile = load_profile(str(profile_file))

        # Validate loaded data
        assert profile.name == "integration_test"
        assert profile.agents == ["accuracy", "terminology", "hallucination"]
        assert profile.agent_weights["accuracy"] == 1.0
        assert profile.severity_multipliers["critical"] == 30
        assert profile.quality_threshold == 97.0
        assert profile.glossaries == ["legal_glossary"]
        assert profile.metadata["domain"] == "legal"

        # Validate profile passes validation
        errors = profile.validate()
        assert errors == []

    def test_profile_with_all_severity_levels(self) -> None:
        """Test profile with custom severity multipliers."""
        profile = MQMProfile(
            name="custom_severity",
            severity_multipliers={
                "critical": 100,
                "major": 20,
                "minor": 5,
                "neutral": 0,
            },
        )
        errors = profile.validate()
        assert errors == []
        assert profile.severity_multipliers["critical"] == 100
