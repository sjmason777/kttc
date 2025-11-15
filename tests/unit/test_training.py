"""Unit tests for training modules (domain adaptation).

Tests domain adaptation functionality with mocked dependencies.
"""

import pytest

from kttc.training.domain_adapter import DomainAdapter, DomainPatterns


@pytest.mark.unit
class TestDomainPatterns:
    """Test DomainPatterns model."""

    def test_domain_patterns_creation(self) -> None:
        """Test creating domain patterns."""
        # Arrange & Act
        patterns = DomainPatterns(
            domain="medical",
            common_terms=["patient", "diagnosis", "treatment"],
            error_patterns={"accuracy": ["mistranslation"]},
        )

        # Assert
        assert patterns.domain == "medical"
        assert len(patterns.common_terms) == 3
        assert "accuracy" in patterns.error_patterns


@pytest.mark.unit
class TestDomainAdapter:
    """Test DomainAdapter with mocked components."""

    def test_initialization(self) -> None:
        """Test domain adapter initialization."""
        # Arrange & Act
        adapter = DomainAdapter(domain="medical")

        # Assert
        assert adapter.domain == "medical"
        assert isinstance(adapter.training_samples, list)

    def test_add_training_sample(self) -> None:
        """Test adding training samples."""
        # Arrange
        adapter = DomainAdapter(domain="technical")

        # Act
        adapter.add_training_sample(
            source="API endpoint",
            translation="punto final de API",
            source_lang="en",
            target_lang="es",
            errors=[],
            terminology={"API": "API"},
        )

        # Assert
        assert len(adapter.training_samples) == 1
