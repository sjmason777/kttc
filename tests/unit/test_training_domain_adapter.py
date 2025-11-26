"""Unit tests for domain adaptation training module.

Tests domain-specific training functionality.
"""

import pytest

from kttc.training.domain_adapter import DomainAdapter, DomainPatterns


@pytest.mark.unit
class TestDomainAdapter:
    """Test DomainAdapter functionality."""

    def test_adapter_initialization(self) -> None:
        """Test adapter initialization."""
        adapter = DomainAdapter(domain="technical")

        assert adapter.domain == "technical"

    def test_adapter_default_domain(self) -> None:
        """Test adapter with default domain."""
        adapter = DomainAdapter()

        # Should have default domain
        assert hasattr(adapter, "domain")

    def test_adapter_with_patterns(self) -> None:
        """Test adapter with domain patterns."""
        adapter = DomainAdapter(domain="medical")

        # Should have patterns loaded
        assert adapter.domain == "medical"

    def test_adapter_legal_domain(self) -> None:
        """Test adapter with legal domain."""
        adapter = DomainAdapter(domain="legal")
        assert adapter.domain == "legal"

    def test_adapter_financial_domain(self) -> None:
        """Test adapter with financial domain."""
        adapter = DomainAdapter(domain="financial")
        assert adapter.domain == "financial"

    def test_adapter_general_domain(self) -> None:
        """Test adapter with general domain."""
        adapter = DomainAdapter(domain="general")
        assert adapter.domain == "general"

    def test_adapter_has_domain_prompts(self) -> None:
        """Test adapter has domain prompts defined."""
        assert hasattr(DomainAdapter, "DOMAIN_PROMPTS")
        prompts = DomainAdapter.DOMAIN_PROMPTS
        assert isinstance(prompts, dict)

    def test_domain_prompts_exist_for_key_domains(self) -> None:
        """Test domain prompts exist for key domains."""
        prompts = DomainAdapter.DOMAIN_PROMPTS
        # At least some domains should have prompts
        assert len(prompts) > 0


@pytest.mark.unit
class TestDomainPatterns:
    """Test DomainPatterns model."""

    def test_domain_patterns_creation(self) -> None:
        """Test creating DomainPatterns instance."""
        patterns = DomainPatterns(domain="legal")
        assert patterns.domain == "legal"
        assert isinstance(patterns.common_terms, list)
        assert isinstance(patterns.error_patterns, dict)
        assert isinstance(patterns.terminology_pairs, dict)

    def test_domain_patterns_with_terms(self) -> None:
        """Test DomainPatterns with common terms."""
        patterns = DomainPatterns(
            domain="medical",
            common_terms=["diagnosis", "treatment", "prognosis"],
        )
        assert len(patterns.common_terms) == 3
        assert "diagnosis" in patterns.common_terms

    def test_domain_patterns_with_terminology(self) -> None:
        """Test DomainPatterns with terminology pairs."""
        patterns = DomainPatterns(
            domain="legal",
            terminology_pairs={"contract": "contrato", "agreement": "acuerdo"},
        )
        assert patterns.terminology_pairs["contract"] == "contrato"

    def test_domain_patterns_with_error_patterns(self) -> None:
        """Test DomainPatterns with error patterns."""
        patterns = DomainPatterns(
            domain="technical",
            error_patterns={
                "terminology": ["API", "SDK"],
                "consistency": ["inconsistent capitalization"],
            },
        )
        assert "terminology" in patterns.error_patterns
        assert "API" in patterns.error_patterns["terminology"]

    def test_domain_patterns_with_style_guidelines(self) -> None:
        """Test DomainPatterns with style guidelines."""
        patterns = DomainPatterns(
            domain="legal",
            style_guidelines="Use formal register throughout",
        )
        assert "formal" in patterns.style_guidelines

    def test_domain_patterns_with_severity_weights(self) -> None:
        """Test DomainPatterns with severity weights."""
        patterns = DomainPatterns(
            domain="medical",
            severity_weights={"critical": 10.0, "major": 5.0, "minor": 1.0},
        )
        assert patterns.severity_weights["critical"] == 10.0

    def test_domain_patterns_with_examples(self) -> None:
        """Test DomainPatterns with examples."""
        patterns = DomainPatterns(
            domain="technical",
            examples=[
                {"source": "API call", "translation": "llamada API"},
                {"source": "endpoint", "translation": "punto final"},
            ],
        )
        assert len(patterns.examples) == 2


@pytest.mark.unit
class TestDomainAdapterMethods:
    """Test DomainAdapter methods."""

    @pytest.fixture
    def adapter(self) -> DomainAdapter:
        """Create adapter instance."""
        return DomainAdapter(domain="technical")

    def test_adapter_has_training_samples_list(self, adapter: DomainAdapter) -> None:
        """Test adapter has training samples storage."""
        # Check if adapter can store training samples
        assert hasattr(adapter, "_training_samples") or hasattr(adapter, "training_samples")

    def test_extract_patterns_method_exists(self, adapter: DomainAdapter) -> None:
        """Test extract_patterns method exists."""
        # Method should exist
        assert hasattr(adapter, "extract_patterns") or hasattr(adapter, "get_patterns")


@pytest.mark.unit
class TestQuickAdapt:
    """Test quick_adapt function."""

    def test_quick_adapt_exists(self) -> None:
        """Test quick_adapt function is available."""
        from kttc.training.domain_adapter import quick_adapt

        assert callable(quick_adapt)
