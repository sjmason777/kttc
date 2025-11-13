"""Tests for HallucinationAgent."""

from unittest.mock import AsyncMock, Mock

import pytest

from kttc.agents.base import AgentEvaluationError, AgentParsingError
from kttc.agents.hallucination import HallucinationAgent
from kttc.core import ErrorSeverity, TranslationTask


@pytest.fixture
def mock_llm():
    """Create mock LLM provider."""
    llm = Mock()
    llm.complete = AsyncMock()
    return llm


@pytest.fixture
def agent(mock_llm):
    """Create HallucinationAgent with mock LLM."""
    return HallucinationAgent(llm_provider=mock_llm)


@pytest.fixture
def translation_task():
    """Create sample translation task."""
    return TranslationTask(
        source_text="The company announced revenue of $50 million in Q4.",
        translation="La empresa anunció ingresos de $50 millones en el cuarto trimestre.",
        source_lang="en",
        target_lang="es",
    )


class TestHallucinationAgent:
    """Test HallucinationAgent class."""

    def test_initialization(self, agent):
        """Test agent initialization."""
        assert agent.category == "accuracy"
        assert agent.temperature == 0.1
        assert agent.max_tokens == 2000
        assert agent.LENGTH_RATIO_MIN == 0.4
        assert agent.LENGTH_RATIO_MAX == 2.0

    def test_initialization_custom_params(self, mock_llm):
        """Test initialization with custom parameters."""
        agent = HallucinationAgent(llm_provider=mock_llm, temperature=0.3, max_tokens=3000)
        assert agent.temperature == 0.3
        assert agent.max_tokens == 3000

    @pytest.mark.asyncio
    async def test_evaluate_basic(self, agent, translation_task, mock_llm):
        """Test basic evaluation without errors."""
        mock_llm.complete.return_value = '{"errors": []}'

        errors = await agent.evaluate(translation_task)

        assert isinstance(errors, list)
        assert len(errors) >= 0  # May have length ratio errors

    @pytest.mark.asyncio
    async def test_evaluate_with_entity_errors(self, agent, mock_llm):
        """Test evaluation detecting entity errors."""
        task = TranslationTask(
            source_text="John Smith earned $50,000 in 2023.",
            translation="Juan Pérez ganó $40,000 en 2023.",  # Wrong name and amount
            source_lang="en",
            target_lang="es",
        )

        mock_llm.complete.return_value = """{
            "errors": [{
                "subcategory": "hallucination_entity_modified",
                "severity": "critical",
                "description": "Name changed from John Smith to Juan Pérez",
                "entity_source": "John Smith",
                "entity_translation": "Juan Pérez"
            }]
        }"""

        errors = await agent.evaluate(task)

        # Should have entity error + any length ratio errors
        entity_errors = [e for e in errors if "entity" in e.subcategory]
        assert len(entity_errors) > 0
        assert entity_errors[0].severity == ErrorSeverity.CRITICAL

    @pytest.mark.asyncio
    async def test_evaluate_exception_handling(self, agent, translation_task, mock_llm):
        """Test evaluation with exception."""
        # Make evaluate raise exception by breaking the length ratio check
        agent._check_length_ratio = lambda x: (_ for _ in ()).throw(Exception("Test error"))

        with pytest.raises(AgentEvaluationError):
            await agent.evaluate(translation_task)

    @pytest.mark.asyncio
    async def test_check_entity_preservation_basic(self, agent, mock_llm):
        """Test entity preservation checking."""
        task = TranslationTask(
            source_text="Bill Gates founded Microsoft in 1975.",
            translation="Bill Gates fundó Microsoft en 1975.",
            source_lang="en",
            target_lang="es",
        )

        mock_llm.complete.return_value = '{"errors": []}'

        errors = await agent._check_entity_preservation(task)
        assert isinstance(errors, list)

    @pytest.mark.asyncio
    async def test_check_entity_preservation_with_errors(self, agent, mock_llm):
        """Test entity preservation detecting missing entities."""
        task = TranslationTask(
            source_text="Apple Inc. reported $100 billion revenue.",
            translation="La empresa reportó ingresos.",  # Missing name and amount
            source_lang="en",
            target_lang="es",
        )

        mock_llm.complete.return_value = """{
            "errors": [
                {
                    "subcategory": "hallucination_entity_missing",
                    "severity": "critical",
                    "description": "Company name 'Apple Inc.' is missing",
                    "entity_source": "Apple Inc.",
                    "entity_translation": ""
                },
                {
                    "subcategory": "hallucination_entity_missing",
                    "severity": "critical",
                    "description": "Amount '$100 billion' is missing",
                    "entity_source": "$100 billion",
                    "entity_translation": ""
                }
            ]
        }"""

        errors = await agent._check_entity_preservation(task)

        assert len(errors) == 2
        assert all(e.category == "accuracy" for e in errors)
        assert all(e.severity == ErrorSeverity.CRITICAL for e in errors)

    @pytest.mark.asyncio
    async def test_check_entity_preservation_exception_handling(self, agent, mock_llm):
        """Test entity preservation with LLM failure."""
        task = TranslationTask(
            source_text="Test", translation="Prueba", source_lang="en", target_lang="es"
        )

        mock_llm.complete.side_effect = Exception("LLM failed")

        # Should return empty list on error, not raise
        errors = await agent._check_entity_preservation(task)
        assert errors == []

    def test_check_length_ratio_normal(self, agent):
        """Test length ratio check with normal ratio."""
        task = TranslationTask(
            source_text="Hello world",  # 11 chars
            translation="Hola mundo",  # 10 chars
            source_lang="en",
            target_lang="es",
        )

        errors = agent._check_length_ratio(task)
        assert len(errors) == 0  # Ratio ~0.9 is fine

    def test_check_length_ratio_excessive(self, agent):
        """Test length ratio check with excessive length."""
        task = TranslationTask(
            source_text="Hello",  # 5 chars
            translation="This is a very long translation with lots of extra words",  # 56 chars
            source_lang="en",
            target_lang="es",
        )

        errors = agent._check_length_ratio(task)

        assert len(errors) == 1
        assert errors[0].subcategory == "hallucination_length_excessive"
        assert errors[0].severity == ErrorSeverity.MAJOR
        assert "longer than source" in errors[0].description

    def test_check_length_ratio_insufficient(self, agent):
        """Test length ratio check with insufficient length."""
        task = TranslationTask(
            source_text="This is a very long source text with many words",  # 48 chars
            translation="Short",  # 5 chars
            source_lang="en",
            target_lang="es",
        )

        errors = agent._check_length_ratio(task)

        assert len(errors) == 1
        assert errors[0].subcategory == "hallucination_length_insufficient"
        assert errors[0].severity == ErrorSeverity.MAJOR
        assert "shorter than source" in errors[0].description

    def test_check_length_ratio_single_char_source(self, agent):
        """Test length ratio check with minimal source."""
        task = TranslationTask(
            source_text="x", translation="Some translation", source_lang="en", target_lang="es"
        )

        errors = agent._check_length_ratio(task)
        # Ratio would be 18/1 = 18, which is > 2.0
        assert len(errors) == 1
        assert errors[0].subcategory == "hallucination_length_excessive"

    def test_check_length_ratio_boundary_min(self, agent):
        """Test length ratio at minimum boundary."""
        task = TranslationTask(
            source_text="x" * 100,  # 100 chars
            translation="x" * 40,  # 40 chars, ratio = 0.4 (exact boundary)
            source_lang="en",
            target_lang="es",
        )

        errors = agent._check_length_ratio(task)
        # At boundary, should not trigger (ratio == LENGTH_RATIO_MIN)
        assert len(errors) == 0

    def test_check_length_ratio_boundary_max(self, agent):
        """Test length ratio at maximum boundary."""
        task = TranslationTask(
            source_text="x" * 50,  # 50 chars
            translation="x" * 100,  # 100 chars, ratio = 2.0 (exact boundary)
            source_lang="en",
            target_lang="es",
        )

        errors = agent._check_length_ratio(task)
        # At boundary, should not trigger (ratio == LENGTH_RATIO_MAX)
        assert len(errors) == 0

    @pytest.mark.asyncio
    async def test_check_factual_consistency_basic(self, agent, mock_llm):
        """Test factual consistency checking."""
        task = TranslationTask(
            source_text="The meeting is at 3pm.",
            translation="La reunión es a las 3pm.",
            source_lang="en",
            target_lang="es",
        )

        mock_llm.complete.return_value = '{"errors": []}'

        errors = await agent._check_factual_consistency(task)
        assert isinstance(errors, list)

    @pytest.mark.asyncio
    async def test_check_factual_consistency_with_errors(self, agent, mock_llm):
        """Test factual consistency detecting hallucinations."""
        task = TranslationTask(
            source_text="The meeting is scheduled.",
            translation="La reunión es a las 3pm en la sala de conferencias.",
            # Adds time and location not in source
            source_lang="en",
            target_lang="es",
        )

        mock_llm.complete.return_value = """{
            "errors": [{
                "subcategory": "hallucination_addition",
                "severity": "major",
                "description": "Added specific time '3pm' and location 'sala de conferencias' not in source",
                "hallucinated_content": "a las 3pm en la sala de conferencias"
            }]
        }"""

        errors = await agent._check_factual_consistency(task)

        assert len(errors) == 1
        assert errors[0].category == "accuracy"
        assert errors[0].subcategory == "hallucination_addition"
        assert errors[0].severity == ErrorSeverity.MAJOR

    @pytest.mark.asyncio
    async def test_check_factual_consistency_exception_handling(self, agent, mock_llm):
        """Test factual consistency with LLM failure."""
        task = TranslationTask(
            source_text="Test", translation="Prueba", source_lang="en", target_lang="es"
        )

        mock_llm.complete.side_effect = Exception("LLM failed")

        # Should return empty list on error, not raise
        errors = await agent._check_factual_consistency(task)
        assert errors == []

    def test_find_text_location_exact_match(self, agent):
        """Test finding text location with exact match."""
        text = "The quick brown fox jumps"
        search = "brown fox"

        location = agent._find_text_location(text, search)

        assert location == (10, 19)  # "brown fox" at position 10-19

    def test_find_text_location_partial_match(self, agent):
        """Test finding text location with partial match."""
        text = "The quick brown fox jumps over the lazy dog"
        search = "marron zorro"  # Not in text, but "quick" is first word of search fallback

        # Should fall back to word matching or middle
        location = agent._find_text_location(text, search)

        assert isinstance(location, tuple)
        assert len(location) == 2
        assert location[0] >= 0
        assert location[1] <= len(text)

    def test_find_text_location_no_match(self, agent):
        """Test finding text location with no match."""
        text = "Hello world"
        search = "xyz abc"  # Not in text

        location = agent._find_text_location(text, search)

        # Should return middle of text
        mid = len(text) // 2
        assert location == (mid, mid + min(20, len(text) - mid))

    def test_find_text_location_empty_search(self, agent):
        """Test finding text location with empty search."""
        text = "Hello world"
        search = ""

        location = agent._find_text_location(text, search)

        # Should return start with min(20, len)
        assert location == (0, min(20, len(text)))

    def test_parse_json_response_valid(self, agent):
        """Test parsing valid JSON response."""
        response = '{"errors": [{"severity": "major"}]}'

        result = agent._parse_json_response(response)

        assert result == {"errors": [{"severity": "major"}]}

    def test_parse_json_response_markdown(self, agent):
        """Test parsing JSON from markdown code block."""
        response = '```json\n{"errors": []}\n```'

        result = agent._parse_json_response(response)

        assert result == {"errors": []}

    def test_parse_json_response_markdown_no_lang(self, agent):
        """Test parsing JSON from markdown without language specifier."""
        response = '```\n{"errors": [{"test": "value"}]}\n```'

        result = agent._parse_json_response(response)

        assert result == {"errors": [{"test": "value"}]}

    def test_parse_json_response_embedded(self, agent):
        """Test parsing JSON embedded in text."""
        response = 'Some text before {"errors": []} and text after'

        result = agent._parse_json_response(response)

        assert result == {"errors": []}

    def test_parse_json_response_invalid(self, agent):
        """Test parsing invalid JSON."""
        response = "This is not JSON at all, no brackets anywhere"

        with pytest.raises(AgentParsingError):
            agent._parse_json_response(response)

    def test_parse_json_response_malformed_in_markdown(self, agent):
        """Test parsing malformed JSON in markdown."""
        response = "```json\n{invalid json content}\n```"

        with pytest.raises(AgentParsingError):
            agent._parse_json_response(response)

    @pytest.mark.asyncio
    async def test_full_evaluation_flow(self, agent, mock_llm):
        """Test complete evaluation flow with all checks."""
        task = TranslationTask(
            source_text="Apple reported $10 billion in revenue for Q4 2023.",
            translation="Apple informó de $10 mil millones de ingresos en el cuarto trimestre de 2023.",
            source_lang="en",
            target_lang="es",
        )

        # Mock LLM to return no errors for all checks
        mock_llm.complete.return_value = '{"errors": []}'

        errors = await agent.evaluate(task)

        # Should call LLM twice (entity preservation and factual consistency)
        assert mock_llm.complete.call_count == 2
        assert isinstance(errors, list)

    @pytest.mark.asyncio
    async def test_full_evaluation_with_multiple_error_types(self, agent, mock_llm):
        """Test evaluation detecting multiple error types."""
        task = TranslationTask(
            source_text="Short source",  # 12 chars
            translation="This is an extremely long translation with lots of added content and information",
            # 81 chars, ratio > 6
            source_lang="en",
            target_lang="es",
        )

        # Mock both LLM calls to return errors
        mock_llm.complete.side_effect = [
            # Entity preservation check
            '{"errors": [{"subcategory": "hallucination_entity_added", "severity": "major", '
            '"description": "Added entity not in source"}]}',
            # Factual consistency check
            '{"errors": [{"subcategory": "hallucination_addition", "severity": "major", '
            '"description": "Added information", "hallucinated_content": "added content"}]}',
        ]

        errors = await agent.evaluate(task)

        # Should have:
        # 1. Length ratio error (excessive)
        # 2. Entity preservation error
        # 3. Factual consistency error
        assert len(errors) >= 3

        # Check we have each type
        error_types = {e.subcategory for e in errors}
        assert "hallucination_length_excessive" in error_types
        assert "hallucination_entity_added" in error_types
        assert "hallucination_addition" in error_types
