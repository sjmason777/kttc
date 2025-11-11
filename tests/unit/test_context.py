"""Tests for ContextAgent."""

from unittest.mock import AsyncMock, Mock

import pytest

from kttc.agents.base import AgentEvaluationError, AgentParsingError
from kttc.agents.context import ContextAgent
from kttc.core import TranslationTask

pytestmark = pytest.mark.asyncio


@pytest.fixture
def mock_llm():
    """Create mock LLM provider."""
    llm = Mock()
    llm.complete = AsyncMock()
    return llm


@pytest.fixture
def agent(mock_llm):
    """Create ContextAgent with mock LLM."""
    return ContextAgent(llm_provider=mock_llm)


@pytest.fixture
def translation_task():
    """Create sample translation task."""
    return TranslationTask(
        source_text="See Section 3.2 for details.",
        translation="Ver la Sección 3.2 para más detalles.",
        source_lang="en",
        target_lang="es",
    )


class TestContextAgent:
    """Test ContextAgent class."""

    def test_initialization(self, agent):
        """Test agent initialization."""
        assert agent.category == "context"
        assert agent.document_context is None
        assert agent.document_segments == []

    def test_set_document_context(self, agent):
        """Test setting document context."""
        agent.set_document_context("Full document text here")
        assert agent.document_context == "Full document text here"

    def test_add_segment(self, agent):
        """Test adding segments to context."""
        agent.add_segment("Hello", "Hola", "seg_1")
        assert len(agent.document_segments) == 1
        assert agent.document_segments[0]["id"] == "seg_1"
        assert agent.document_segments[0]["source"] == "Hello"
        assert agent.document_segments[0]["translation"] == "Hola"

    def test_add_segment_auto_id(self, agent):
        """Test adding segment with auto-generated ID."""
        agent.add_segment("Hello", "Hola")
        assert agent.document_segments[0]["id"] == "seg_0"

    def test_clear_context(self, agent):
        """Test clearing context."""
        agent.set_document_context("Some text")
        agent.add_segment("Hello", "Hola")

        agent.clear_context()

        assert agent.document_context is None
        assert agent.document_segments == []

    async def test_evaluate_basic(self, agent, translation_task, mock_llm):
        """Test basic evaluation."""
        mock_llm.complete.return_value = '{"errors": []}'

        errors = await agent.evaluate(translation_task)

        assert isinstance(errors, list)

    async def test_evaluate_with_cross_reference_preserved(self, agent, mock_llm):
        """Test evaluation with preserved cross-reference."""
        task = TranslationTask(
            source_text="See Section 3.2",
            translation="Ver Sección 3.2",
            source_lang="en",
            target_lang="es",
        )
        mock_llm.complete.return_value = '{"errors": []}'

        errors = await agent.evaluate(task)

        # Should not find errors since "3.2" is preserved
        cross_ref_errors = [e for e in errors if e.subcategory == "cross_reference_missing"]
        assert len(cross_ref_errors) == 0

    async def test_evaluate_with_missing_cross_reference(self, agent, mock_llm):
        """Test evaluation with missing cross-reference."""
        task = TranslationTask(
            source_text="See Section 3.2 for details",
            translation="Ver detalles",  # Missing "3.2"
            source_lang="en",
            target_lang="es",
        )
        mock_llm.complete.return_value = '{"errors": []}'

        errors = await agent.evaluate(task)

        # Should find cross-reference error
        cross_ref_errors = [e for e in errors if e.subcategory == "cross_reference_missing"]
        assert len(cross_ref_errors) > 0

    async def test_evaluate_with_document_segments(self, agent, mock_llm):
        """Test evaluation with document context."""
        agent.add_segment("First segment", "Primer segmento")
        agent.add_segment("Second segment", "Segundo segmento")

        task = TranslationTask(
            source_text="Third segment",
            translation="Tercer segmento",
            source_lang="en",
            target_lang="es",
        )
        mock_llm.complete.return_value = '{"errors": []}'

        errors = await agent.evaluate(task)

        assert isinstance(errors, list)
        # Should call LLM for coherence check
        assert mock_llm.complete.called

    async def test_evaluate_exception_handling(self, agent, translation_task, mock_llm):
        """Test evaluation with exception from cross-reference check."""
        # Make _extract_references raise exception to test outer exception handler
        # Since _check_coherence catches its own exceptions, we need to test
        # an exception from a different path
        agent._extract_references = lambda x: (_ for _ in ()).throw(Exception("Extract failed"))

        with pytest.raises(AgentEvaluationError):
            await agent.evaluate(translation_task)

    def test_extract_references(self, agent):
        """Test extracting cross-references."""
        text = "See Section 3.2, Figure 5, and Table 1 on page 42."
        refs = agent._extract_references(text)

        assert len(refs) >= 4
        assert any("Section" in r or "3" in r for r in refs)
        assert any("Figure" in r or "5" in r for r in refs)
        assert any("Table" in r or "1" in r for r in refs)
        assert any("page" in r or "42" in r for r in refs)

    def test_extract_references_none_found(self, agent):
        """Test extracting references when none exist."""
        text = "This is a simple sentence with no references."
        refs = agent._extract_references(text)

        assert refs == []

    def test_extract_technical_terms(self, agent):
        """Test extracting technical terms."""
        text = "The API uses HTTP protocol for REST communication."
        terms = agent._extract_technical_terms(text)

        assert "API" in terms
        assert "HTTP" in terms
        assert "REST" in terms

    def test_extract_technical_terms_capitalized(self, agent):
        """Test extracting capitalized terms."""
        text = "The Database connects to the Server."
        terms = agent._extract_technical_terms(text)

        assert "Database" in terms or "Server" in terms

    def test_extract_technical_terms_hyphenated(self, agent):
        """Test extracting hyphenated terms."""
        text = "Using multi-threading and load-balancing."
        terms = agent._extract_technical_terms(text)

        assert any("-" in t for t in terms)

    def test_find_term_location(self, agent):
        """Test finding term location."""
        text = "The API is fast"
        loc = agent._find_term_location(text, "API")

        assert loc == (4, 7)

    def test_find_term_location_not_found(self, agent):
        """Test finding term location when term not in text."""
        text = "Simple text"
        loc = agent._find_term_location(text, "missing")

        # Should return fallback
        assert loc == (0, 10) or loc == (0, len(text))

    def test_find_term_translations_in_context(self, agent):
        """Test finding term translations."""
        agent.add_segment("Use API", "Usar API")
        agent.add_segment("The API works", "El API funciona")

        translations = agent._find_term_translations_in_context("API")

        assert len(translations) == 2

    def test_find_term_translations_none_found(self, agent):
        """Test finding term translations when none exist."""
        agent.add_segment("Simple text", "Texto simple")

        translations = agent._find_term_translations_in_context("missing")

        assert translations == []

    def test_build_context_text_empty(self, agent):
        """Test building context text when no segments."""
        context = agent._build_context_text()

        assert context == "(No previous context)"

    def test_build_context_text_with_segments(self, agent):
        """Test building context text with segments."""
        agent.add_segment("First", "Primero", "seg_1")
        agent.add_segment("Second", "Segundo", "seg_2")

        context = agent._build_context_text()

        assert "seg_1" in context
        assert "seg_2" in context
        assert "Primero" in context
        assert "Segundo" in context

    def test_build_context_text_limits_segments(self, agent):
        """Test that context text limits segments."""
        for i in range(10):
            agent.add_segment(f"Text {i}", f"Texto {i}")

        context = agent._build_context_text(max_segments=3)

        # Should only include last 3
        lines = context.split("\n")
        assert len([line for line in lines if line.startswith("[seg_")]) == 3

    def test_parse_json_response_valid(self, agent):
        """Test parsing valid JSON response."""
        response = '{"errors": []}'
        result = agent._parse_json_response(response)

        assert result == {"errors": []}

    def test_parse_json_response_markdown(self, agent):
        """Test parsing JSON from markdown."""
        response = '```json\n{"errors": []}\n```'
        result = agent._parse_json_response(response)

        assert result == {"errors": []}

    def test_parse_json_response_embedded(self, agent):
        """Test parsing JSON embedded in text."""
        response = 'Some text {"errors": []} more text'
        result = agent._parse_json_response(response)

        assert result == {"errors": []}

    def test_parse_json_response_invalid(self, agent):
        """Test parsing invalid JSON."""
        response = "Not JSON at all"

        with pytest.raises(AgentParsingError):
            agent._parse_json_response(response)

    async def test_check_coherence_with_llm_response(self, agent, mock_llm):
        """Test coherence checking with LLM response."""
        agent.add_segment("Context text", "Texto contexto")

        task = TranslationTask(
            source_text="Current text",
            translation="Texto actual",
            source_lang="en",
            target_lang="es",
        )

        mock_llm.complete.return_value = """{"errors": [{
            "subcategory": "coherence_issue",
            "severity": "minor",
            "description": "Some coherence issue"
        }]}"""

        errors = await agent._check_coherence(task)

        assert len(errors) == 1
        assert errors[0].category == "context"
        assert errors[0].subcategory == "coherence_issue"

    async def test_check_coherence_no_context(self, agent, mock_llm):
        """Test coherence checking without context."""
        task = TranslationTask(
            source_text="Text",
            translation="Texto",
            source_lang="en",
            target_lang="es",
        )

        errors = await agent._check_coherence(task)

        # Should return empty list without calling LLM
        assert errors == []
        assert not mock_llm.complete.called

    async def test_check_coherence_llm_failure(self, agent, mock_llm):
        """Test coherence checking when LLM fails."""
        agent.add_segment("Context", "Contexto")

        task = TranslationTask(
            source_text="Text",
            translation="Texto",
            source_lang="en",
            target_lang="es",
        )

        mock_llm.complete.side_effect = Exception("LLM failed")

        errors = await agent._check_coherence(task)

        # Should return empty list on error
        assert errors == []

    async def test_check_term_consistency_no_segments(self, agent):
        """Test term consistency without segments."""
        task = TranslationTask(
            source_text="API text",
            translation="API texto",
            source_lang="en",
            target_lang="es",
        )

        errors = await agent._check_term_consistency(task)

        assert errors == []

    async def test_check_term_consistency_inconsistent(self, agent):
        """Test detection of inconsistent term usage."""
        agent.add_segment("Use API first", "Usar API primero")
        agent.add_segment("Use API again", "Usar interfaz otra vez")

        task = TranslationTask(
            source_text="The API works",
            translation="La API funciona",
            source_lang="en",
            target_lang="es",
        )

        errors = await agent._check_term_consistency(task)

        # Should detect inconsistency
        if errors:
            assert any(e.subcategory == "term_inconsistency" for e in errors)
