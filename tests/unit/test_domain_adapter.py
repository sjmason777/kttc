"""Unit tests for domain adapter module.

Tests domain-specific adaptation for translation quality.
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from kttc.training.domain_adapter import (
    DomainAdapter,
    DomainPatterns,
    quick_adapt,
)


@pytest.mark.unit
class TestDomainPatternsModel:
    """Test DomainPatterns Pydantic model."""

    def test_default_values(self) -> None:
        """Test default values for DomainPatterns."""
        patterns = DomainPatterns(domain="test")

        assert patterns.domain == "test"
        assert patterns.common_terms == []
        assert patterns.error_patterns == {}
        assert patterns.style_guidelines == ""
        assert patterns.terminology_pairs == {}
        assert patterns.severity_weights == {}
        assert patterns.examples == []

    def test_with_all_fields(self) -> None:
        """Test DomainPatterns with all fields populated."""
        patterns = DomainPatterns(
            domain="legal",
            common_terms=["contract", "agreement"],
            error_patterns={"terminology": ["wrong term"]},
            style_guidelines="Use formal register",
            terminology_pairs={"contract": "contrato"},
            severity_weights={"critical": 0.5, "minor": 0.5},
            examples=[{"source": "test", "translation": "test"}],
        )

        assert patterns.domain == "legal"
        assert len(patterns.common_terms) == 2
        assert "terminology" in patterns.error_patterns
        assert patterns.terminology_pairs["contract"] == "contrato"


@pytest.mark.unit
class TestDomainAdapterInitialization:
    """Test domain adapter initialization."""

    def test_initialization_default(self) -> None:
        """Test default initialization."""
        adapter = DomainAdapter()

        assert adapter.domain == "general"
        assert adapter.training_samples == []

    def test_initialization_with_domain(self) -> None:
        """Test initialization with specific domain."""
        adapter = DomainAdapter(domain="legal")

        assert adapter.domain == "legal"

    def test_initialization_with_unknown_domain(self) -> None:
        """Test initialization with unknown domain falls back to general."""
        adapter = DomainAdapter(domain="unknown_xyz")

        assert adapter.domain == "general"

    def test_all_supported_domains(self) -> None:
        """Test that all expected domains are supported."""
        for domain in ["legal", "medical", "technical", "financial", "general"]:
            adapter = DomainAdapter(domain=domain)
            assert adapter.domain == domain

    def test_domain_case_insensitive(self) -> None:
        """Test domain name is case insensitive."""
        adapter = DomainAdapter(domain="LEGAL")
        assert adapter.domain == "legal"


@pytest.mark.unit
class TestAddTrainingSample:
    """Test adding training samples."""

    def test_add_training_sample_minimal(self) -> None:
        """Test adding minimal training sample."""
        adapter = DomainAdapter()

        adapter.add_training_sample(
            source="Hello",
            translation="Привет",
            source_lang="en",
            target_lang="ru",
        )

        assert len(adapter.training_samples) == 1
        assert adapter.training_samples[0]["source"] == "Hello"
        assert adapter.training_samples[0]["translation"] == "Привет"

    def test_add_training_sample_full(self) -> None:
        """Test adding full training sample with all fields."""
        adapter = DomainAdapter()

        adapter.add_training_sample(
            source="The contract is void",
            translation="El contrato es nulo",
            source_lang="en",
            target_lang="es",
            errors=[{"category": "terminology", "description": "test"}],
            terminology={"contract": "contrato", "void": "nulo"},
            reference="El contrato está nulo",
        )

        sample = adapter.training_samples[0]
        assert sample["source"] == "The contract is void"
        assert sample["errors"] == [{"category": "terminology", "description": "test"}]
        assert sample["terminology"]["contract"] == "contrato"
        assert sample["reference"] == "El contrato está nulo"

    def test_add_multiple_samples(self) -> None:
        """Test adding multiple training samples."""
        adapter = DomainAdapter()

        for i in range(5):
            adapter.add_training_sample(
                source=f"Source {i}",
                translation=f"Translation {i}",
                source_lang="en",
                target_lang="ru",
            )

        assert len(adapter.training_samples) == 5


@pytest.mark.unit
class TestLoadTrainingData:
    """Test loading training data from files."""

    def test_load_training_data(self, tmp_path: Path) -> None:
        """Test loading training data from JSON file."""
        import json

        data = {
            "domain": "legal",
            "samples": [
                {
                    "source": "The contract",
                    "translation": "El contrato",
                    "source_lang": "en",
                    "target_lang": "es",
                }
            ],
        }

        data_file = tmp_path / "training.json"
        with open(data_file, "w") as f:
            json.dump(data, f)

        adapter = DomainAdapter()
        adapter.load_training_data(data_file)

        assert adapter.domain == "legal"
        assert len(adapter.training_samples) == 1

    def test_load_training_data_file_not_found(self) -> None:
        """Test loading from non-existent file raises error."""
        adapter = DomainAdapter()

        with pytest.raises(FileNotFoundError):
            adapter.load_training_data("/nonexistent/path.json")


@pytest.mark.unit
class TestExtractPatterns:
    """Test pattern extraction from training data."""

    def test_extract_patterns_empty(self) -> None:
        """Test extracting patterns with no training data."""
        adapter = DomainAdapter(domain="technical")

        patterns = adapter.extract_patterns()

        assert patterns.domain == "technical"
        assert patterns.common_terms == []

    def test_extract_patterns_with_samples(self) -> None:
        """Test extracting patterns from training samples."""
        adapter = DomainAdapter(domain="legal")

        # Add samples with terminology
        adapter.add_training_sample(
            source="The contract agreement was signed by parties",
            translation="El contrato fue firmado",
            source_lang="en",
            target_lang="es",
            terminology={"contract": "contrato"},
        )
        adapter.add_training_sample(
            source="The contract terms are binding",
            translation="Los términos del contrato son vinculantes",
            source_lang="en",
            target_lang="es",
            terminology={"contract": "contrato", "terms": "términos"},
        )

        patterns = adapter.extract_patterns()

        assert patterns.domain == "legal"
        assert "contract" in patterns.terminology_pairs
        assert patterns.terminology_pairs["contract"] == "contrato"

    def test_extract_patterns_with_errors(self) -> None:
        """Test extracting error patterns from samples."""
        adapter = DomainAdapter(domain="medical")

        adapter.add_training_sample(
            source="Patient has hypertension",
            translation="El paciente tiene hipertensión",
            source_lang="en",
            target_lang="es",
            errors=[
                {"category": "terminology", "description": "Wrong term used"},
                {"category": "terminology", "description": "Inconsistent terminology"},
            ],
        )

        patterns = adapter.extract_patterns()

        assert "terminology" in patterns.error_patterns
        assert len(patterns.error_patterns["terminology"]) == 2

    def test_extract_patterns_severity_weights(self) -> None:
        """Test extracting severity weights from samples."""
        adapter = DomainAdapter(domain="technical")

        adapter.add_training_sample(
            source="API endpoint",
            translation="Endpoint API",
            source_lang="en",
            target_lang="es",
            errors=[
                {"severity": "critical"},
                {"severity": "minor"},
                {"severity": "minor"},
            ],
        )

        patterns = adapter.extract_patterns()

        assert "critical" in patterns.severity_weights
        assert "minor" in patterns.severity_weights
        # 1 critical out of 3 = ~0.33, 2 minor out of 3 = ~0.67
        assert patterns.severity_weights["minor"] > patterns.severity_weights["critical"]


@pytest.mark.unit
class TestExtractCommonTerms:
    """Test common terms extraction."""

    def test_extract_common_terms(self) -> None:
        """Test extracting common terms from samples."""
        adapter = DomainAdapter()

        # Add samples with repeated terms
        for _ in range(3):
            adapter.add_training_sample(
                source="The contract agreement legal document",
                translation="Translation",
                source_lang="en",
                target_lang="ru",
            )

        terms = adapter._extract_common_terms()

        assert isinstance(terms, list)
        # Should include "contract", "agreement", "legal", "document"
        # (words with length > 3)

    def test_extract_common_terms_filters_short(self) -> None:
        """Test that short words are filtered out."""
        adapter = DomainAdapter()

        adapter.add_training_sample(
            source="a an the is to of",
            translation="Translation",
            source_lang="en",
            target_lang="ru",
        )

        terms = adapter._extract_common_terms()

        # All words are 3 chars or less, should be filtered
        for term in terms:
            assert len(term) > 3


@pytest.mark.unit
class TestAdaptAgent:
    """Test agent adaptation."""

    def test_adapt_agent_basic(self) -> None:
        """Test basic agent adaptation."""
        adapter = DomainAdapter(domain="legal")

        mock_agent = MagicMock()
        mock_agent.get_base_prompt.return_value = "Base prompt"

        adapted = adapter.adapt_agent(mock_agent)

        # Agent should have domain attributes set
        assert hasattr(adapted, "_domain")
        assert adapted._domain == "legal"
        assert hasattr(adapted, "_domain_patterns")

    def test_adapt_agent_with_patterns(self) -> None:
        """Test agent adaptation with provided patterns."""
        adapter = DomainAdapter(domain="medical")

        patterns = DomainPatterns(
            domain="medical",
            common_terms=["patient", "diagnosis"],
            terminology_pairs={"patient": "paciente"},
        )

        mock_agent = MagicMock()
        mock_agent.get_base_prompt.return_value = "Base prompt"

        adapted = adapter.adapt_agent(mock_agent, patterns)

        assert adapted._domain_patterns == patterns


@pytest.mark.unit
class TestEnhancePrompt:
    """Test prompt enhancement."""

    def test_enhance_prompt_adds_domain_section(self) -> None:
        """Test that enhance prompt adds domain guidelines."""
        adapter = DomainAdapter(domain="legal")

        patterns = DomainPatterns(
            domain="legal",
            common_terms=["contract"],
        )

        enhanced = adapter._enhance_prompt("Base prompt", patterns)

        assert "DOMAIN-SPECIFIC GUIDELINES" in enhanced
        assert "LEGAL" in enhanced
        assert "Base prompt" in enhanced

    def test_enhance_prompt_adds_terminology(self) -> None:
        """Test that enhance prompt adds terminology section."""
        adapter = DomainAdapter(domain="technical")

        patterns = DomainPatterns(
            domain="technical",
            terminology_pairs={"API": "API", "endpoint": "точка доступа"},
        )

        enhanced = adapter._enhance_prompt("Base", patterns)

        assert "APPROVED TERMINOLOGY" in enhanced

    def test_enhance_prompt_adds_error_patterns(self) -> None:
        """Test that enhance prompt adds error patterns section."""
        adapter = DomainAdapter(domain="medical")

        patterns = DomainPatterns(
            domain="medical",
            error_patterns={"terminology": ["Wrong term used"]},
        )

        enhanced = adapter._enhance_prompt("Base", patterns)

        assert "COMMON ERROR PATTERNS" in enhanced


@pytest.mark.unit
class TestSaveLoadPatterns:
    """Test pattern persistence."""

    def test_save_patterns(self, tmp_path: Path) -> None:
        """Test saving patterns to file."""
        adapter = DomainAdapter()

        patterns = DomainPatterns(
            domain="legal",
            common_terms=["contract", "agreement"],
        )

        output_file = tmp_path / "patterns.json"
        adapter.save_patterns(patterns, output_file)

        assert output_file.exists()

    def test_load_patterns(self, tmp_path: Path) -> None:
        """Test loading patterns from file."""
        import json

        adapter = DomainAdapter()

        pattern_data = {
            "domain": "technical",
            "common_terms": ["API", "endpoint"],
            "error_patterns": {},
            "style_guidelines": "",
            "terminology_pairs": {},
            "severity_weights": {},
            "examples": [],
        }

        pattern_file = tmp_path / "patterns.json"
        with open(pattern_file, "w") as f:
            json.dump(pattern_data, f)

        patterns = adapter.load_patterns(pattern_file)

        assert patterns.domain == "technical"
        assert "API" in patterns.common_terms

    def test_load_patterns_file_not_found(self) -> None:
        """Test loading patterns from non-existent file."""
        adapter = DomainAdapter()

        with pytest.raises(FileNotFoundError):
            adapter.load_patterns("/nonexistent/patterns.json")

    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        """Test saving and loading patterns preserves data."""
        adapter = DomainAdapter()

        original = DomainPatterns(
            domain="financial",
            common_terms=["asset", "liability"],
            terminology_pairs={"asset": "актив"},
            error_patterns={"accuracy": ["calculation error"]},
        )

        output_file = tmp_path / "roundtrip.json"
        adapter.save_patterns(original, output_file)
        loaded = adapter.load_patterns(output_file)

        assert loaded.domain == original.domain
        assert loaded.common_terms == original.common_terms
        assert loaded.terminology_pairs == original.terminology_pairs


@pytest.mark.unit
class TestQuickAdapt:
    """Test quick_adapt helper function."""

    def test_quick_adapt(self, tmp_path: Path) -> None:
        """Test quick adaptation helper."""
        import json

        # Create training data file
        data = {
            "domain": "legal",
            "samples": [
                {
                    "source": "Contract terms",
                    "translation": "Términos del contrato",
                    "source_lang": "en",
                    "target_lang": "es",
                }
            ],
        }

        data_file = tmp_path / "training.json"
        with open(data_file, "w") as f:
            json.dump(data, f)

        mock_agent = MagicMock()
        mock_agent.get_base_prompt.return_value = "Base prompt"

        adapted = quick_adapt(mock_agent, "legal", data_file)

        assert hasattr(adapted, "_domain")
        assert adapted._domain == "legal"


@pytest.mark.unit
class TestDomainPrompts:
    """Test domain-specific prompts."""

    def test_all_domains_have_prompts(self) -> None:
        """Test that all expected domains have prompt templates."""
        expected_domains = ["legal", "medical", "technical", "financial", "general"]

        for domain in expected_domains:
            assert domain in DomainAdapter.DOMAIN_PROMPTS
            assert len(DomainAdapter.DOMAIN_PROMPTS[domain]) > 0

    def test_legal_prompt_content(self) -> None:
        """Test legal domain prompt contains relevant terms."""
        prompt = DomainAdapter.DOMAIN_PROMPTS["legal"]

        assert "legal" in prompt.lower()
        assert "terminology" in prompt.lower()

    def test_medical_prompt_content(self) -> None:
        """Test medical domain prompt contains relevant terms."""
        prompt = DomainAdapter.DOMAIN_PROMPTS["medical"]

        assert "medical" in prompt.lower()
        assert "patient" in prompt.lower() or "drug" in prompt.lower()

    def test_technical_prompt_content(self) -> None:
        """Test technical domain prompt contains relevant terms."""
        prompt = DomainAdapter.DOMAIN_PROMPTS["technical"]

        assert "technical" in prompt.lower()


@pytest.mark.unit
class TestPrivateMethods:
    """Test private helper methods."""

    def test_extract_terminology_pairs(self) -> None:
        """Test terminology pairs extraction."""
        adapter = DomainAdapter()

        adapter.add_training_sample(
            source="Source",
            translation="Target",
            source_lang="en",
            target_lang="ru",
            terminology={"term1": "терм1", "term2": "терм2"},
        )
        adapter.add_training_sample(
            source="Source2",
            translation="Target2",
            source_lang="en",
            target_lang="ru",
            terminology={"term3": "терм3"},
        )

        pairs = adapter._extract_terminology_pairs()

        assert len(pairs) == 3
        assert pairs["term1"] == "терм1"
        assert pairs["term3"] == "терм3"

    def test_collect_examples(self) -> None:
        """Test example collection."""
        adapter = DomainAdapter()

        adapter.add_training_sample(
            source="Good translation",
            translation="Хороший перевод",
            source_lang="en",
            target_lang="ru",
        )
        adapter.add_training_sample(
            source="Bad translation",
            translation="Плохой перевод",
            source_lang="en",
            target_lang="ru",
            errors=[{"category": "accuracy"}],
        )

        examples = adapter._collect_examples()

        assert len(examples) == 2
        assert examples[0]["quality"] == "good"
        assert examples[1]["quality"] == "needs_review"

    def test_calculate_severity_weights_empty(self) -> None:
        """Test severity weights with no errors."""
        adapter = DomainAdapter()

        adapter.add_training_sample(
            source="Source",
            translation="Target",
            source_lang="en",
            target_lang="ru",
        )

        weights = adapter._calculate_severity_weights()

        assert weights == {}
