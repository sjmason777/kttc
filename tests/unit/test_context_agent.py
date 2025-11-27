"""Unit tests for context agent module.

Tests document-level context awareness and consistency checking.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from kttc.agents.context import ContextAgent
from kttc.core import ErrorSeverity, TranslationTask


@pytest.mark.unit
class TestContextAgentInitialization:
    """Test context agent initialization."""

    def test_default_initialization(self) -> None:
        """Test default initialization parameters."""
        mock_provider = MagicMock()
        agent = ContextAgent(mock_provider)

        assert agent.llm_provider == mock_provider
        assert agent.temperature == 0.1
        assert agent.max_tokens == 2000
        assert agent.document_context is None
        assert agent.document_segments == []

    def test_custom_initialization(self) -> None:
        """Test custom initialization parameters."""
        mock_provider = MagicMock()
        agent = ContextAgent(mock_provider, temperature=0.5, max_tokens=3000)

        assert agent.temperature == 0.5
        assert agent.max_tokens == 3000

    def test_category_property(self) -> None:
        """Test category property returns correct value."""
        mock_provider = MagicMock()
        agent = ContextAgent(mock_provider)

        assert agent.category == "context"


@pytest.mark.unit
class TestDocumentContext:
    """Test document context management."""

    def test_set_document_context(self) -> None:
        """Test setting document context."""
        mock_provider = MagicMock()
        agent = ContextAgent(mock_provider)

        full_doc = "This is a full document with multiple sections."
        agent.set_document_context(full_doc)

        assert agent.document_context == full_doc

    def test_add_segment(self) -> None:
        """Test adding translation segments."""
        mock_provider = MagicMock()
        agent = ContextAgent(mock_provider)

        agent.add_segment("Hello", "Hola", "seg1")
        agent.add_segment("World", "Mundo", "seg2")

        assert len(agent.document_segments) == 2
        assert agent.document_segments[0]["id"] == "seg1"
        assert agent.document_segments[0]["source"] == "Hello"
        assert agent.document_segments[0]["translation"] == "Hola"

    def test_add_segment_auto_id(self) -> None:
        """Test adding segment with auto-generated ID."""
        mock_provider = MagicMock()
        agent = ContextAgent(mock_provider)

        agent.add_segment("Test", "Prueba")

        assert agent.document_segments[0]["id"] == "seg_0"

    def test_clear_context(self) -> None:
        """Test clearing document context."""
        mock_provider = MagicMock()
        agent = ContextAgent(mock_provider)

        agent.set_document_context("Some document")
        agent.add_segment("Test", "Prueba")

        agent.clear_context()

        assert agent.document_context is None
        assert agent.document_segments == []


@pytest.mark.unit
class TestCrossReferenceChecking:
    """Test cross-reference preservation checking."""

    def test_extract_references(self) -> None:
        """Test reference extraction from text."""
        mock_provider = MagicMock()
        agent = ContextAgent(mock_provider)

        text = "See Section 3.2 and Figure 1. Also check Table 5 on page 42."
        refs = agent._extract_references(text)

        assert "Section 3.2" in refs
        assert "Figure 1" in refs
        assert "Table 5" in refs
        assert "page 42" in refs

    def test_extract_references_empty(self) -> None:
        """Test reference extraction with no references."""
        mock_provider = MagicMock()
        agent = ContextAgent(mock_provider)

        text = "This is plain text without any references."
        refs = agent._extract_references(text)

        assert refs == []

    def test_check_cross_references_preserved(self) -> None:
        """Test cross-reference check when references are preserved."""
        mock_provider = MagicMock()
        agent = ContextAgent(mock_provider)

        task = TranslationTask(
            source_text="See Section 3.2 for details.",
            translation="Ver la Sección 3.2 para más detalles.",
            source_lang="en",
            target_lang="es",
        )

        errors = agent._check_cross_references(task)
        assert len(errors) == 0

    def test_check_cross_references_missing(self) -> None:
        """Test cross-reference check when references are missing."""
        mock_provider = MagicMock()
        agent = ContextAgent(mock_provider)

        task = TranslationTask(
            source_text="See Section 3.2 for details.",
            translation="Ver la sección para más detalles.",  # Missing "3.2"
            source_lang="en",
            target_lang="es",
        )

        errors = agent._check_cross_references(task)
        assert len(errors) == 1
        assert errors[0].subcategory == "cross_reference_missing"
        assert errors[0].severity == ErrorSeverity.MAJOR


@pytest.mark.unit
class TestTermConsistency:
    """Test term consistency checking."""

    def test_extract_technical_terms(self) -> None:
        """Test technical term extraction."""
        mock_provider = MagicMock()
        agent = ContextAgent(mock_provider)

        text = "The API uses REST and JSON. Check the FAQ for more info."
        terms = agent._extract_technical_terms(text)

        assert "API" in terms
        assert "REST" in terms
        assert "JSON" in terms
        assert "FAQ" in terms

    def test_extract_technical_terms_with_compound(self) -> None:
        """Test extraction of compound technical terms."""
        mock_provider = MagicMock()
        agent = ContextAgent(mock_provider)

        text = "Use machine-learning for data-driven analysis."
        terms = agent._extract_technical_terms(text)

        assert "machine-learning" in terms
        assert "data-driven" in terms

    def test_find_term_location(self) -> None:
        """Test finding term location in text."""
        mock_provider = MagicMock()
        agent = ContextAgent(mock_provider)

        text = "The API provides functionality."
        location = agent._find_term_location(text, "API")

        assert location == (4, 7)

    def test_find_term_location_not_found(self) -> None:
        """Test finding term location when term not found."""
        mock_provider = MagicMock()
        agent = ContextAgent(mock_provider)

        text = "Simple text"
        location = agent._find_term_location(text, "missing")

        # Should return fallback location
        assert location == (0, 10)

    def test_check_term_consistency_no_segments(self) -> None:
        """Test term consistency check with no previous segments."""
        mock_provider = MagicMock()
        agent = ContextAgent(mock_provider)

        task = TranslationTask(
            source_text="Test",
            translation="Prueba",
            source_lang="en",
            target_lang="es",
        )

        errors = agent._check_term_consistency(task)
        assert errors == []


@pytest.mark.unit
class TestContextBuilding:
    """Test context text building."""

    def test_build_context_text_empty(self) -> None:
        """Test building context with no segments."""
        mock_provider = MagicMock()
        agent = ContextAgent(mock_provider)

        context = agent._build_context_text()
        assert context == "(No previous context)"

    def test_build_context_text_with_segments(self) -> None:
        """Test building context with segments."""
        mock_provider = MagicMock()
        agent = ContextAgent(mock_provider)

        agent.add_segment("Hello", "Hola", "seg1")
        agent.add_segment("World", "Mundo", "seg2")

        context = agent._build_context_text()

        assert "[seg1]" in context
        assert "Hola" in context
        assert "[seg2]" in context
        assert "Mundo" in context

    def test_build_context_text_max_segments(self) -> None:
        """Test building context respects max_segments limit."""
        mock_provider = MagicMock()
        agent = ContextAgent(mock_provider)

        for i in range(10):
            agent.add_segment(f"Source {i}", f"Translation {i}", f"seg{i}")

        context = agent._build_context_text(max_segments=3)

        # Should only include last 3 segments
        assert "[seg7]" in context
        assert "[seg8]" in context
        assert "[seg9]" in context
        assert "[seg0]" not in context


@pytest.mark.unit
class TestJSONParsing:
    """Test JSON response parsing."""

    def test_parse_valid_json(self) -> None:
        """Test parsing valid JSON response."""
        mock_provider = MagicMock()
        agent = ContextAgent(mock_provider)

        response = '{"errors": []}'
        result = agent._parse_json_response(response)

        assert result == {"errors": []}

    def test_parse_json_from_markdown(self) -> None:
        """Test parsing JSON from markdown code block."""
        mock_provider = MagicMock()
        agent = ContextAgent(mock_provider)

        response = """Here is the analysis:
```json
{"errors": [{"subcategory": "test"}]}
```"""
        result = agent._parse_json_response(response)

        assert result == {"errors": [{"subcategory": "test"}]}

    def test_parse_json_embedded(self) -> None:
        """Test parsing JSON embedded in text."""
        mock_provider = MagicMock()
        agent = ContextAgent(mock_provider)

        response = 'Here is the result: {"errors": []} End of response.'
        result = agent._parse_json_response(response)

        assert result == {"errors": []}

    def test_parse_invalid_json_raises(self) -> None:
        """Test parsing invalid JSON raises error."""
        from kttc.agents.base import AgentParsingError

        mock_provider = MagicMock()
        agent = ContextAgent(mock_provider)

        response = "This is not valid JSON at all."

        with pytest.raises(AgentParsingError):
            agent._parse_json_response(response)


@pytest.mark.unit
class TestEvaluate:
    """Test the main evaluate method."""

    @pytest.mark.asyncio
    async def test_evaluate_no_context(self) -> None:
        """Test evaluation with no document context."""
        mock_provider = AsyncMock()
        agent = ContextAgent(mock_provider)

        task = TranslationTask(
            source_text="Hello world.",
            translation="Hola mundo.",
            source_lang="en",
            target_lang="es",
        )

        errors = await agent.evaluate(task)

        # Should complete without errors (no cross-refs, no context)
        assert isinstance(errors, list)

    @pytest.mark.asyncio
    async def test_evaluate_with_cross_references(self) -> None:
        """Test evaluation detects missing cross-references."""
        mock_provider = AsyncMock()
        agent = ContextAgent(mock_provider)

        task = TranslationTask(
            source_text="See Figure 1 and Table 2.",
            translation="Ver la figura y la tabla.",  # Missing numbers
            source_lang="en",
            target_lang="es",
        )

        errors = await agent.evaluate(task)

        # Should find missing cross-references
        assert len(errors) >= 1
        assert any(e.subcategory == "cross_reference_missing" for e in errors)

    @pytest.mark.asyncio
    async def test_evaluate_with_coherence_check(self) -> None:
        """Test evaluation with coherence checking."""
        mock_provider = AsyncMock()
        mock_provider.complete = AsyncMock(return_value='{"errors": []}')

        agent = ContextAgent(mock_provider)
        agent.add_segment("First segment", "Primer segmento", "seg1")

        task = TranslationTask(
            source_text="Second segment.",
            translation="Segundo segmento.",
            source_lang="en",
            target_lang="es",
        )

        _errors = await agent.evaluate(task)

        # LLM should be called for coherence check
        mock_provider.complete.assert_called_once()

    @pytest.mark.asyncio
    async def test_evaluate_handles_llm_error(self) -> None:
        """Test evaluation handles LLM errors gracefully."""
        mock_provider = AsyncMock()
        mock_provider.complete = AsyncMock(side_effect=Exception("LLM error"))

        agent = ContextAgent(mock_provider)
        agent.add_segment("Previous", "Anterior", "seg1")

        task = TranslationTask(
            source_text="Current.",
            translation="Actual.",
            source_lang="en",
            target_lang="es",
        )

        # Should not raise, just skip coherence check
        errors = await agent.evaluate(task)
        assert isinstance(errors, list)


@pytest.mark.unit
class TestBasePrompt:
    """Test base prompt generation."""

    def test_get_base_prompt(self) -> None:
        """Test base prompt contains required sections."""
        mock_provider = MagicMock()
        agent = ContextAgent(mock_provider)

        prompt = agent.get_base_prompt()

        assert "Cross-Reference Preservation" in prompt
        assert "Term Consistency" in prompt
        assert "Coherence Across Segments" in prompt
        assert "Document Structure" in prompt
        assert "JSON" in prompt
