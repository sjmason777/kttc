"""Tests for domain adaptation functionality.

These tests verify that domain adaptation works correctly,
including pattern extraction, prompt enhancement, and agent adaptation.
"""

import json
import tempfile
from pathlib import Path

import pytest

from kttc.agents.base import BaseAgent
from kttc.core.models import ErrorAnnotation, TranslationTask
from kttc.training.domain_adapter import DomainAdapter, DomainPatterns, quick_adapt


class MockAgent(BaseAgent):
    """Mock agent for testing."""

    def __init__(self, llm_provider):
        super().__init__(llm_provider)

    def get_base_prompt(self) -> str:
        return "You are a translation QA agent."

    async def evaluate(self, task: TranslationTask) -> list[ErrorAnnotation]:
        return []

    @property
    def category(self) -> str:
        return "test"


class TestDomainAdapter:
    """Test DomainAdapter functionality."""

    def test_initialization_valid_domain(self):
        """Test initialization with valid domain."""
        adapter = DomainAdapter(domain="legal")
        assert adapter.domain == "legal"
        assert len(adapter.training_samples) == 0

    def test_initialization_invalid_domain_fallback(self):
        """Test that invalid domain falls back to general."""
        adapter = DomainAdapter(domain="invalid_domain")
        assert adapter.domain == "general"

    def test_add_training_sample(self):
        """Test adding training samples."""
        adapter = DomainAdapter()
        adapter.add_training_sample(
            source="The contract is void.",
            translation="El contrato es nulo.",
            source_lang="en",
            target_lang="es",
            errors=[{"category": "accuracy", "severity": "minor"}],
            terminology={"contract": "contrato", "void": "nulo"},
        )

        assert len(adapter.training_samples) == 1
        assert adapter.training_samples[0]["source"] == "The contract is void."
        assert adapter.training_samples[0]["terminology"]["contract"] == "contrato"

    def test_add_training_sample_minimal(self):
        """Test adding training sample with minimal data."""
        adapter = DomainAdapter()
        adapter.add_training_sample(
            source="Hello", translation="Hola", source_lang="en", target_lang="es"
        )

        assert len(adapter.training_samples) == 1
        assert adapter.training_samples[0]["errors"] == []
        assert adapter.training_samples[0]["terminology"] == {}

    def test_extract_patterns_no_samples(self):
        """Test pattern extraction with no training data."""
        adapter = DomainAdapter("legal")
        patterns = adapter.extract_patterns()

        assert patterns.domain == "legal"
        assert len(patterns.common_terms) == 0
        assert len(patterns.error_patterns) == 0
        assert len(patterns.terminology_pairs) == 0

    def test_extract_patterns_with_samples(self):
        """Test pattern extraction with training data."""
        adapter = DomainAdapter("legal")

        # Add multiple samples
        for _ in range(5):
            adapter.add_training_sample(
                source="The contract agreement party hereby acknowledge",
                translation="El contrato acuerdo parte por la presente reconoce",
                source_lang="en",
                target_lang="es",
                errors=[
                    {
                        "category": "accuracy",
                        "subcategory": "mistranslation",
                        "severity": "major",
                        "description": "Wrong term used",
                    }
                ],
                terminology={"contract": "contrato", "party": "parte"},
            )

        patterns = adapter.extract_patterns()

        # Check common terms extracted
        assert len(patterns.common_terms) > 0
        assert "contract" in patterns.common_terms or "agreement" in patterns.common_terms

        # Check error patterns
        assert "accuracy" in patterns.error_patterns
        assert len(patterns.error_patterns["accuracy"]) > 0

        # Check terminology
        assert patterns.terminology_pairs["contract"] == "contrato"
        assert patterns.terminology_pairs["party"] == "parte"

        # Check severity weights
        assert "major" in patterns.severity_weights
        assert patterns.severity_weights["major"] > 0

    def test_extract_patterns_filters_short_words(self):
        """Test that short words are filtered from common terms."""
        adapter = DomainAdapter()
        adapter.add_training_sample(
            source="a is to the of",  # All short words
            translation="es al del",
            source_lang="en",
            target_lang="es",
        )

        patterns = adapter.extract_patterns()
        # Should have no common terms (all filtered out as too short)
        assert len(patterns.common_terms) == 0

    def test_extract_patterns_multiple_error_categories(self):
        """Test pattern extraction with multiple error types."""
        adapter = DomainAdapter()
        adapter.add_training_sample(
            source="Test",
            translation="Test",
            source_lang="en",
            target_lang="es",
            errors=[
                {"category": "accuracy", "severity": "critical", "description": "Wrong meaning"},
                {"category": "fluency", "severity": "major", "description": "Grammar error"},
                {"category": "terminology", "severity": "minor", "description": "Wrong term"},
            ],
        )

        patterns = adapter.extract_patterns()

        assert len(patterns.error_patterns) == 3
        assert "accuracy" in patterns.error_patterns
        assert "fluency" in patterns.error_patterns
        assert "terminology" in patterns.error_patterns

    def test_save_and_load_patterns(self):
        """Test saving and loading domain patterns."""
        adapter = DomainAdapter("legal")
        adapter.add_training_sample(
            source="Legal document",
            translation="Documento legal",
            source_lang="en",
            target_lang="es",
        )

        patterns = adapter.extract_patterns()

        with tempfile.TemporaryDirectory() as tmpdir:
            patterns_file = Path(tmpdir) / "patterns.json"

            # Save
            adapter.save_patterns(patterns, patterns_file)
            assert patterns_file.exists()

            # Load
            loaded_patterns = adapter.load_patterns(patterns_file)
            assert loaded_patterns.domain == patterns.domain
            assert loaded_patterns.common_terms == patterns.common_terms

    def test_load_patterns_nonexistent_file(self):
        """Test loading patterns from nonexistent file raises error."""
        adapter = DomainAdapter()

        with pytest.raises(FileNotFoundError):
            adapter.load_patterns("nonexistent_patterns.json")

    def test_load_training_data_from_file(self):
        """Test loading training data from JSON file."""
        adapter = DomainAdapter()

        training_data = {
            "domain": "medical",
            "samples": [
                {
                    "source": "The patient",
                    "translation": "El paciente",
                    "source_lang": "en",
                    "target_lang": "es",
                    "errors": [],
                    "terminology": {"patient": "paciente"},
                }
            ],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            data_file = Path(tmpdir) / "training.json"
            with open(data_file, "w") as f:
                json.dump(training_data, f)

            adapter.load_training_data(data_file)

            assert adapter.domain == "medical"
            assert len(adapter.training_samples) == 1

    def test_load_training_data_nonexistent_file(self):
        """Test loading training data from nonexistent file raises error."""
        adapter = DomainAdapter()

        with pytest.raises(FileNotFoundError):
            adapter.load_training_data("nonexistent.json")

    def test_adapt_agent(self):
        """Test adapting an agent with domain knowledge."""
        from unittest.mock import Mock

        mock_llm = Mock()
        agent = MockAgent(mock_llm)

        adapter = DomainAdapter("legal")
        adapter.add_training_sample(
            source="contract", translation="contrato", source_lang="en", target_lang="es"
        )

        patterns = adapter.extract_patterns()
        adapted_agent = adapter.adapt_agent(agent, patterns)

        # Check that agent was modified
        assert hasattr(adapted_agent, "get_base_prompt")
        new_prompt = adapted_agent.get_base_prompt()

        # Should contain domain information
        assert "LEGAL" in new_prompt or "legal" in new_prompt.lower()

    def test_adapt_agent_extracts_patterns_if_none(self):
        """Test that adapt_agent extracts patterns if not provided."""
        from unittest.mock import Mock

        mock_llm = Mock()
        agent = MockAgent(mock_llm)

        adapter = DomainAdapter("technical")
        adapter.add_training_sample(
            source="system", translation="sistema", source_lang="en", target_lang="es"
        )

        # Don't pass patterns - should extract them
        adapted_agent = adapter.adapt_agent(agent)

        new_prompt = adapted_agent.get_base_prompt()
        assert "TECHNICAL" in new_prompt or "technical" in new_prompt.lower()

    def test_enhance_prompt_with_terminology(self):
        """Test that enhanced prompt includes terminology."""
        adapter = DomainAdapter("legal")

        patterns = DomainPatterns(
            domain="legal",
            common_terms=["contract", "agreement"],
            terminology_pairs={"contract": "contrato", "clause": "cláusula"},
        )

        base_prompt = "Evaluate translation quality."
        enhanced = adapter._enhance_prompt(base_prompt, patterns)

        assert "contrato" in enhanced
        assert "cláusula" in enhanced
        assert "APPROVED TERMINOLOGY" in enhanced

    def test_enhance_prompt_with_error_patterns(self):
        """Test that enhanced prompt includes error patterns."""
        adapter = DomainAdapter("legal")

        patterns = DomainPatterns(
            domain="legal",
            error_patterns={"accuracy": ["Wrong term used", "Mistranslation"]},
        )

        base_prompt = "Evaluate translation quality."
        enhanced = adapter._enhance_prompt(base_prompt, patterns)

        assert "COMMON ERROR PATTERNS" in enhanced
        assert "accuracy" in enhanced.lower()

    def test_domain_specific_prompts_exist(self):
        """Test that all domains have prompt templates."""
        required_domains = ["legal", "medical", "technical", "financial", "general"]

        for domain in required_domains:
            assert domain in DomainAdapter.DOMAIN_PROMPTS
            assert len(DomainAdapter.DOMAIN_PROMPTS[domain]) > 0

    def test_severity_weights_calculation(self):
        """Test severity weights are calculated correctly."""
        adapter = DomainAdapter()

        # Add samples with known severity distribution
        for _ in range(7):
            adapter.add_training_sample(
                source="test",
                translation="test",
                source_lang="en",
                target_lang="es",
                errors=[{"severity": "critical"}],
            )
        for _ in range(3):
            adapter.add_training_sample(
                source="test",
                translation="test",
                source_lang="en",
                target_lang="es",
                errors=[{"severity": "minor"}],
            )

        patterns = adapter.extract_patterns()

        # 7 critical, 3 minor = 70% critical, 30% minor
        assert patterns.severity_weights["critical"] == pytest.approx(0.7, rel=0.01)
        assert patterns.severity_weights["minor"] == pytest.approx(0.3, rel=0.01)

    def test_examples_limited_to_10(self):
        """Test that examples are limited to 10."""
        adapter = DomainAdapter()

        # Add 20 samples
        for i in range(20):
            adapter.add_training_sample(
                source=f"test {i}",
                translation=f"prueba {i}",
                source_lang="en",
                target_lang="es",
            )

        patterns = adapter.extract_patterns()

        # Should only have 10 examples
        assert len(patterns.examples) == 10

    def test_examples_truncated_to_100_chars(self):
        """Test that example texts are truncated."""
        adapter = DomainAdapter()

        long_text = "word " * 50  # 250 chars
        adapter.add_training_sample(
            source=long_text, translation=long_text, source_lang="en", target_lang="es"
        )

        patterns = adapter.extract_patterns()

        assert len(patterns.examples[0]["source"]) == 100
        assert len(patterns.examples[0]["translation"]) == 100

    @pytest.mark.asyncio
    async def test_quick_adapt_function(self):
        """Test quick_adapt convenience function."""
        from unittest.mock import Mock

        mock_llm = Mock()
        agent = MockAgent(mock_llm)

        training_data = {
            "domain": "technical",
            "samples": [
                {
                    "source": "system",
                    "translation": "sistema",
                    "source_lang": "en",
                    "target_lang": "es",
                }
            ],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            data_file = Path(tmpdir) / "training.json"
            with open(data_file, "w") as f:
                json.dump(training_data, f)

            adapted = await quick_adapt(agent, "technical", data_file)

            # Check adaptation worked
            prompt = adapted.get_base_prompt()
            assert "TECHNICAL" in prompt or "technical" in prompt.lower()


class TestDomainPatterns:
    """Test DomainPatterns model."""

    def test_domain_patterns_defaults(self):
        """Test DomainPatterns with default values."""
        patterns = DomainPatterns(domain="legal")

        assert patterns.domain == "legal"
        assert patterns.common_terms == []
        assert patterns.error_patterns == {}
        assert patterns.terminology_pairs == {}
        assert patterns.style_guidelines == ""
        assert patterns.severity_weights == {}
        assert patterns.examples == []

    def test_domain_patterns_full(self):
        """Test DomainPatterns with all fields."""
        patterns = DomainPatterns(
            domain="medical",
            common_terms=["patient", "diagnosis"],
            error_patterns={"accuracy": ["Wrong term"]},
            style_guidelines="Use medical terminology",
            terminology_pairs={"patient": "paciente"},
            severity_weights={"critical": 0.5},
            examples=[{"source": "test", "translation": "prueba", "quality": "good"}],
        )

        assert patterns.domain == "medical"
        assert len(patterns.common_terms) == 2
        assert "accuracy" in patterns.error_patterns
        assert patterns.style_guidelines != ""
        assert "patient" in patterns.terminology_pairs
        assert "critical" in patterns.severity_weights
        assert len(patterns.examples) == 1

    def test_domain_patterns_serialization(self):
        """Test that DomainPatterns can be serialized/deserialized."""
        patterns = DomainPatterns(
            domain="legal",
            common_terms=["contract", "agreement"],
            terminology_pairs={"void": "nulo"},
        )

        # Serialize
        data = patterns.model_dump()
        assert isinstance(data, dict)
        assert data["domain"] == "legal"

        # Deserialize
        restored = DomainPatterns(**data)
        assert restored.domain == patterns.domain
        assert restored.common_terms == patterns.common_terms
