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

"""Tests for domain profiles and adaptive agent selection."""

import pytest

from kttc.agents.domain_profiles import (
    DOMAIN_PROFILES,
    DomainDetector,
    DomainProfile,
    get_domain_profile,
    list_available_domains,
)


class TestDomainProfile:
    """Tests for DomainProfile dataclass."""

    def test_domain_profile_creation(self) -> None:
        """Test creating a DomainProfile instance."""
        profile = DomainProfile(
            domain_type="test",
            description="Test domain",
            complexity=0.5,
            priority_agents=["accuracy", "fluency"],
            agent_weights={"accuracy": 1.0, "fluency": 0.9},
            quality_threshold=95.0,
            confidence_threshold=0.7,
            require_human_review=False,
            keywords=["test", "example"],
        )

        assert profile.domain_type == "test"
        assert profile.description == "Test domain"
        assert profile.complexity == 0.5
        assert profile.priority_agents == ["accuracy", "fluency"]
        assert profile.agent_weights == {"accuracy": 1.0, "fluency": 0.9}
        assert profile.quality_threshold == 95.0
        assert profile.confidence_threshold == 0.7
        assert profile.require_human_review is False
        assert profile.keywords == ["test", "example"]

    def test_domain_profile_defaults(self) -> None:
        """Test DomainProfile default values."""
        profile = DomainProfile(
            domain_type="test",
            description="Test",
            complexity=0.5,
            priority_agents=["accuracy"],
            agent_weights={"accuracy": 1.0},
        )

        assert profile.quality_threshold == 95.0  # Default
        assert profile.confidence_threshold == 0.7  # Default
        assert profile.require_human_review is False  # Default
        assert profile.keywords == []  # Default empty list
        assert profile.metadata == {}  # Default empty dict


class TestDomainProfiles:
    """Tests for predefined DOMAIN_PROFILES."""

    def test_all_domains_exist(self) -> None:
        """Test that all expected domains are defined."""
        expected_domains = [
            "technical",
            "medical",
            "legal",
            "marketing",
            "literary",
            "general",
        ]

        for domain in expected_domains:
            assert domain in DOMAIN_PROFILES, f"Missing domain: {domain}"

    def test_technical_profile(self) -> None:
        """Test technical domain profile configuration."""
        profile = DOMAIN_PROFILES["technical"]

        assert profile.domain_type == "technical"
        assert profile.complexity == 0.8
        assert "accuracy" in profile.priority_agents
        assert "terminology" in profile.priority_agents
        assert profile.agent_weights["accuracy"] == 1.0
        assert profile.agent_weights["terminology"] == 0.95
        assert profile.quality_threshold == 97.0  # Higher for technical
        assert "API" in profile.keywords
        assert "endpoint" in profile.keywords

    def test_medical_profile(self) -> None:
        """Test medical domain profile configuration."""
        profile = DOMAIN_PROFILES["medical"]

        assert profile.domain_type == "medical"
        assert profile.complexity == 0.9  # High complexity
        assert profile.agent_weights["accuracy"] == 1.0
        assert profile.agent_weights["terminology"] == 1.0
        assert profile.agent_weights["hallucination"] == 1.0
        assert profile.quality_threshold == 98.0  # Highest threshold
        assert profile.require_human_review is True  # Always review
        assert "patient" in profile.keywords
        assert "diagnosis" in profile.keywords

    def test_legal_profile(self) -> None:
        """Test legal domain profile configuration."""
        profile = DOMAIN_PROFILES["legal"]

        assert profile.domain_type == "legal"
        assert profile.complexity == 0.9
        assert profile.agent_weights["accuracy"] == 1.0
        assert profile.quality_threshold == 97.0
        assert profile.require_human_review is True
        assert "contract" in profile.keywords
        assert "agreement" in profile.keywords

    def test_marketing_profile(self) -> None:
        """Test marketing domain profile configuration."""
        profile = DOMAIN_PROFILES["marketing"]

        assert profile.domain_type == "marketing"
        assert profile.complexity == 0.7  # Lower complexity
        assert profile.agent_weights["fluency"] == 1.0  # Fluency prioritized
        assert profile.quality_threshold == 93.0  # Lower threshold
        assert profile.require_human_review is False
        assert "brand" in profile.keywords
        assert "customer" in profile.keywords

    def test_literary_profile(self) -> None:
        """Test literary domain profile configuration."""
        profile = DOMAIN_PROFILES["literary"]

        assert profile.domain_type == "literary"
        assert profile.complexity == 0.8
        assert profile.agent_weights["fluency"] == 1.0  # Style important
        assert profile.quality_threshold == 90.0  # Lowest threshold
        assert "chapter" in profile.keywords
        assert "story" in profile.keywords

    def test_general_profile(self) -> None:
        """Test general domain profile configuration."""
        profile = DOMAIN_PROFILES["general"]

        assert profile.domain_type == "general"
        assert profile.complexity == 0.6
        assert profile.quality_threshold == 95.0  # Balanced threshold
        assert profile.keywords == []  # No specific keywords

    def test_all_profiles_have_required_agents(self) -> None:
        """Test that all profiles define weights for core agents."""
        for domain, profile in DOMAIN_PROFILES.items():
            assert "accuracy" in profile.agent_weights, f"{domain} missing accuracy weight"
            assert "fluency" in profile.agent_weights, f"{domain} missing fluency weight"

    def test_all_profiles_have_valid_thresholds(self) -> None:
        """Test that all profiles have valid threshold values."""
        for domain, profile in DOMAIN_PROFILES.items():
            assert 0 <= profile.quality_threshold <= 100, f"{domain} has invalid quality_threshold"
            assert (
                0 <= profile.confidence_threshold <= 1.0
            ), f"{domain} has invalid confidence_threshold"
            assert 0 <= profile.complexity <= 1.0, f"{domain} has invalid complexity"


class TestDomainDetector:
    """Tests for DomainDetector class."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.detector = DomainDetector()

    def test_detector_initialization(self) -> None:
        """Test DomainDetector initialization."""
        assert self.detector.profiles == DOMAIN_PROFILES

    def test_detect_technical_domain(self) -> None:
        """Test detecting technical domain."""
        texts = [
            "The API endpoint accepts POST requests with JSON payload",
            "Configure the database connection parameters",
            "The function returns a list of objects",
            "Implement authentication using OAuth2 protocol",
        ]

        for text in texts:
            domain = self.detector.detect_domain(text)
            assert domain == "technical", f"Failed to detect technical for: {text}"

    def test_detect_medical_domain(self) -> None:
        """Test detecting medical domain."""
        texts = [
            "The patient exhibits symptoms of fever and cough",
            "Prescribe medication for the treatment of hypertension",
            "Clinical diagnosis indicates possible infection",
            "Monitor dosage and adverse reactions carefully",
        ]

        for text in texts:
            domain = self.detector.detect_domain(text)
            assert domain == "medical", f"Failed to detect medical for: {text}"

    def test_detect_legal_domain(self) -> None:
        """Test detecting legal domain."""
        texts = [
            "The contract specifies liability clauses for both parties",
            "Pursuant to the agreement, the jurisdiction is established",
            "The provision in the statute requires compliance",
            "Whereas the parties agree to the terms herein",
        ]

        for text in texts:
            domain = self.detector.detect_domain(text)
            assert domain == "legal", f"Failed to detect legal for: {text}"

    def test_detect_marketing_domain(self) -> None:
        """Test detecting marketing domain."""
        texts = [
            "Buy now and get an exclusive discount on our premium product",
            "Limited offer for customers who join our brand campaign",
            "Promotional sale with special prices for new products",
            "Advertisement for our latest marketing campaign",
        ]

        for text in texts:
            domain = self.detector.detect_domain(text)
            assert domain == "marketing", f"Failed to detect marketing for: {text}"

    def test_detect_literary_domain(self) -> None:
        """Test detecting literary domain."""
        texts = [
            "In this chapter, the protagonist discovers a hidden truth",
            "The narrative follows the character through various scenes",
            "The story unfolds with a dramatic plot twist",
            "The poem uses metaphor to convey deep meaning",
        ]

        for text in texts:
            domain = self.detector.detect_domain(text)
            assert domain == "literary", f"Failed to detect literary for: {text}"

    def test_detect_general_domain_default(self) -> None:
        """Test that general domain is default for ambiguous text."""
        texts = [
            "The weather is nice today",
            "Hello, how are you?",
            "This is a simple sentence",
            "Random text without specific keywords",
        ]

        for text in texts:
            domain = self.detector.detect_domain(text)
            assert domain == "general", f"Should default to general for: {text}"

    def test_explicit_domain_in_context(self) -> None:
        """Test explicit domain specification in context."""
        text = "Some random text"
        context = {"domain": "medical"}

        domain = self.detector.detect_domain(text, context=context)
        assert domain == "medical"

    def test_invalid_explicit_domain_ignored(self) -> None:
        """Test that invalid explicit domain is ignored."""
        text = "The API endpoint returns JSON"
        context = {"domain": "invalid_domain_name"}

        domain = self.detector.detect_domain(text, context=context)
        assert domain == "technical"  # Falls back to keyword detection

    def test_domain_confidence_high(self) -> None:
        """Test high confidence for texts with many keywords."""
        text = "The API endpoint uses REST protocol for authentication and configuration"
        domain = self.detector.detect_domain(text)
        confidence = self.detector.get_domain_confidence(text, domain)

        assert confidence >= 0.75  # Many technical keywords

    def test_domain_confidence_medium(self) -> None:
        """Test medium confidence for texts with few keywords."""
        text = "The API is good"
        domain = self.detector.detect_domain(text)
        confidence = self.detector.get_domain_confidence(text, domain)

        assert 0.5 <= confidence <= 0.75  # Few keywords

    def test_domain_confidence_low_for_general(self) -> None:
        """Test low confidence for general domain."""
        text = "Some random text"
        domain = self.detector.detect_domain(text)
        confidence = self.detector.get_domain_confidence(text, domain)

        assert domain == "general"
        assert confidence == 0.5  # Default for general

    def test_case_insensitive_detection(self) -> None:
        """Test that keyword detection is case-insensitive."""
        texts = [
            "The API endpoint is available",
            "The api endpoint is available",
            "The API ENDPOINT is available",
        ]

        for text in texts:
            domain = self.detector.detect_domain(text)
            assert domain == "technical"

    def test_keyword_boundary_matching(self) -> None:
        """Test that keywords match on word boundaries."""
        # "API" should match, but not "rapid" or "erapist"
        text1 = "The API is good"
        domain1 = self.detector.detect_domain(text1)
        assert domain1 == "technical"

        # "rapid" contains "api" but shouldn't match
        text2 = "The rapid process"
        domain2 = self.detector.detect_domain(text2)
        assert domain2 == "general"


class TestDomainProfileFunctions:
    """Tests for domain profile utility functions."""

    def test_get_domain_profile_valid(self) -> None:
        """Test getting a valid domain profile."""
        profile = get_domain_profile("technical")
        assert isinstance(profile, DomainProfile)
        assert profile.domain_type == "technical"

    def test_get_domain_profile_invalid(self) -> None:
        """Test getting an invalid domain profile raises error."""
        with pytest.raises(ValueError, match="Unknown domain type"):
            get_domain_profile("invalid_domain")

    def test_list_available_domains(self) -> None:
        """Test listing all available domains."""
        domains = list_available_domains()

        assert isinstance(domains, list)
        assert len(domains) == 6
        assert "technical" in domains
        assert "medical" in domains
        assert "legal" in domains
        assert "marketing" in domains
        assert "literary" in domains
        assert "general" in domains


class TestDomainDetectorEdgeCases:
    """Tests for edge cases in domain detection."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.detector = DomainDetector()

    def test_empty_text(self) -> None:
        """Test detection with empty text."""
        domain = self.detector.detect_domain("")
        assert domain == "general"

    def test_very_short_text(self) -> None:
        """Test detection with very short text."""
        domain = self.detector.detect_domain("API")
        assert domain == "technical"

    def test_multiple_domain_keywords(self) -> None:
        """Test text with keywords from multiple domains."""
        # Technical + medical keywords
        text = "The medical API endpoint for patient data"
        domain = self.detector.detect_domain(text)

        # Should pick domain with more keywords, or technical if tied
        # (depends on keyword count and order)
        assert domain in ["technical", "medical"]

    def test_text_with_no_matching_keywords(self) -> None:
        """Test text with no domain-specific keywords."""
        text = "The quick brown fox jumps over the lazy dog"
        domain = self.detector.detect_domain(text)
        assert domain == "general"

    def test_confidence_calculation_consistency(self) -> None:
        """Test that confidence calculation is consistent."""
        text = "The API endpoint uses REST protocol"
        domain = self.detector.detect_domain(text)

        # Call confidence multiple times - should be consistent
        conf1 = self.detector.get_domain_confidence(text, domain)
        conf2 = self.detector.get_domain_confidence(text, domain)
        conf3 = self.detector.get_domain_confidence(text, domain)

        assert conf1 == conf2 == conf3

    def test_confidence_range_validation(self) -> None:
        """Test that confidence is always in valid range."""
        texts = [
            "API endpoint function method protocol",  # Many keywords
            "API endpoint",  # Few keywords
            "API",  # Single keyword
            "Random text",  # No keywords
        ]

        for text in texts:
            domain = self.detector.detect_domain(text)
            confidence = self.detector.get_domain_confidence(text, domain)

            assert 0.0 <= confidence <= 1.0, f"Invalid confidence for: {text}"
