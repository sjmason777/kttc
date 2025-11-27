"""Unit tests for hallucination detection agent.

Tests factual consistency and hallucination detection.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from kttc.agents.hallucination import HallucinationAgent
from kttc.core import ErrorSeverity, TranslationTask


@pytest.mark.unit
class TestHallucinationAgentInitialization:
    """Test hallucination agent initialization."""

    def test_default_initialization(self) -> None:
        """Test default initialization parameters."""
        mock_provider = MagicMock()
        agent = HallucinationAgent(mock_provider)

        assert agent.llm_provider == mock_provider
        assert agent.temperature == 0.1
        assert agent.max_tokens == 2000

    def test_custom_initialization(self) -> None:
        """Test custom initialization parameters."""
        mock_provider = MagicMock()
        agent = HallucinationAgent(mock_provider, temperature=0.3, max_tokens=1500)

        assert agent.temperature == 0.3
        assert agent.max_tokens == 1500

    def test_category_property(self) -> None:
        """Test category property returns correct value."""
        mock_provider = MagicMock()
        agent = HallucinationAgent(mock_provider)

        assert agent.category == "accuracy"

    def test_thresholds_defined(self) -> None:
        """Test that thresholds are properly defined."""
        assert HallucinationAgent.LENGTH_RATIO_MIN == 0.4
        assert HallucinationAgent.LENGTH_RATIO_MAX == 2.0
        assert HallucinationAgent.FACTUAL_CONSISTENCY_THRESHOLD == 0.80


@pytest.mark.unit
class TestLengthRatioChecking:
    """Test length ratio analysis."""

    def test_check_length_ratio_normal(self) -> None:
        """Test length ratio check with normal translation."""
        mock_provider = MagicMock()
        agent = HallucinationAgent(mock_provider)

        task = TranslationTask(
            source_text="Hello world, this is a test.",
            translation="Hola mundo, esto es una prueba.",
            source_lang="en",
            target_lang="es",
        )

        errors = agent._check_length_ratio(task)
        assert len(errors) == 0

    def test_check_length_ratio_too_long(self) -> None:
        """Test length ratio check with excessively long translation."""
        mock_provider = MagicMock()
        agent = HallucinationAgent(mock_provider)

        task = TranslationTask(
            source_text="Hello.",
            translation="Hola mundo, esto es una prueba muy larga con mucho texto adicional que no corresponde al original.",
            source_lang="en",
            target_lang="es",
        )

        errors = agent._check_length_ratio(task)

        assert len(errors) == 1
        assert errors[0].subcategory == "hallucination_length_excessive"
        assert errors[0].severity == ErrorSeverity.MAJOR
        assert "longer than source" in errors[0].description

    def test_check_length_ratio_too_short(self) -> None:
        """Test length ratio check with very short translation."""
        mock_provider = MagicMock()
        agent = HallucinationAgent(mock_provider)

        task = TranslationTask(
            source_text="Hello world, this is a very long sentence with many words that should be translated completely.",
            translation="Hola.",
            source_lang="en",
            target_lang="es",
        )

        errors = agent._check_length_ratio(task)

        assert len(errors) == 1
        assert errors[0].subcategory == "hallucination_length_insufficient"
        assert errors[0].severity == ErrorSeverity.MAJOR
        assert "shorter than source" in errors[0].description

    def test_check_length_ratio_minimal_source(self) -> None:
        """Test length ratio check with minimal source."""
        mock_provider = MagicMock()
        agent = HallucinationAgent(mock_provider)

        # Use a minimal but valid source (TranslationTask requires non-empty source)
        task = TranslationTask(
            source_text="X",
            translation="Y",  # Same length - should be OK
            source_lang="en",
            target_lang="es",
        )

        errors = agent._check_length_ratio(task)
        assert len(errors) == 0  # Should handle gracefully


@pytest.mark.unit
class TestFindTextLocation:
    """Test text location finding."""

    def test_find_exact_match(self) -> None:
        """Test finding exact text match."""
        mock_provider = MagicMock()
        agent = HallucinationAgent(mock_provider)

        text = "Hello world, this is a test."
        location = agent._find_text_location(text, "world")

        assert location == (6, 11)

    def test_find_partial_match(self) -> None:
        """Test finding partial text match."""
        mock_provider = MagicMock()
        agent = HallucinationAgent(mock_provider)

        text = "The quick brown fox jumps."
        # Search for a phrase where only a word matches
        location = agent._find_text_location(text, "quick fox")

        assert location[0] == 4  # Should find "quick"

    def test_find_no_match(self) -> None:
        """Test location when no match found."""
        mock_provider = MagicMock()
        agent = HallucinationAgent(mock_provider)

        text = "Hello world"
        location = agent._find_text_location(text, "xyz")

        # Should return middle of text as fallback
        assert location[0] == len(text) // 2

    def test_find_empty_search(self) -> None:
        """Test location with empty search text."""
        mock_provider = MagicMock()
        agent = HallucinationAgent(mock_provider)

        text = "Hello world"
        location = agent._find_text_location(text, "")

        assert location == (0, min(20, len(text)))


@pytest.mark.unit
class TestJSONParsing:
    """Test JSON response parsing."""

    def test_parse_valid_json(self) -> None:
        """Test parsing valid JSON response."""
        mock_provider = MagicMock()
        agent = HallucinationAgent(mock_provider)

        response = '{"errors": []}'
        result = agent._parse_json_response(response)

        assert result == {"errors": []}

    def test_parse_json_with_errors(self) -> None:
        """Test parsing JSON with error entries."""
        mock_provider = MagicMock()
        agent = HallucinationAgent(mock_provider)

        response = """{"errors": [{"subcategory": "hallucination_factual", "severity": "critical", "description": "Test"}]}"""
        result = agent._parse_json_response(response)

        assert len(result["errors"]) == 1
        assert result["errors"][0]["subcategory"] == "hallucination_factual"

    def test_parse_json_from_markdown_block(self) -> None:
        """Test parsing JSON from markdown code block."""
        mock_provider = MagicMock()
        agent = HallucinationAgent(mock_provider)

        response = """Analysis complete:
```json
{"errors": [{"subcategory": "test"}]}
```
End of response."""
        result = agent._parse_json_response(response)

        assert result == {"errors": [{"subcategory": "test"}]}

    def test_parse_json_embedded_in_text(self) -> None:
        """Test parsing JSON embedded in text."""
        mock_provider = MagicMock()
        agent = HallucinationAgent(mock_provider)

        response = 'Here is the analysis: {"errors": []} Done.'
        result = agent._parse_json_response(response)

        assert result == {"errors": []}

    def test_parse_invalid_json_raises(self) -> None:
        """Test parsing invalid JSON raises error."""
        from kttc.agents.base import AgentParsingError

        mock_provider = MagicMock()
        agent = HallucinationAgent(mock_provider)

        response = "This contains no valid JSON whatsoever."

        with pytest.raises(AgentParsingError):
            agent._parse_json_response(response)


@pytest.mark.unit
class TestEntityPreservationCheck:
    """Test entity preservation checking."""

    @pytest.mark.asyncio
    async def test_check_entity_preservation_no_errors(self) -> None:
        """Test entity preservation check with no errors."""
        mock_provider = AsyncMock()
        mock_provider.complete = AsyncMock(return_value='{"errors": []}')

        agent = HallucinationAgent(mock_provider)

        task = TranslationTask(
            source_text="John Smith bought 3 apples on January 1st.",
            translation="John Smith compró 3 manzanas el 1 de enero.",
            source_lang="en",
            target_lang="es",
        )

        errors = await agent._check_entity_preservation(task)

        assert len(errors) == 0

    @pytest.mark.asyncio
    async def test_check_entity_preservation_with_errors(self) -> None:
        """Test entity preservation check detects missing entities."""
        mock_provider = AsyncMock()
        mock_provider.complete = AsyncMock(
            return_value="""{
            "errors": [
                {
                    "subcategory": "hallucination_entity_missing",
                    "severity": "critical",
                    "description": "Name 'John Smith' was not preserved",
                    "entity_source": "John Smith"
                }
            ]
        }"""
        )

        agent = HallucinationAgent(mock_provider)

        task = TranslationTask(
            source_text="John Smith bought 3 apples.",
            translation="Alguien compró 3 manzanas.",  # Name missing
            source_lang="en",
            target_lang="es",
        )

        errors = await agent._check_entity_preservation(task)

        assert len(errors) == 1
        assert errors[0].subcategory == "hallucination_entity_missing"
        assert errors[0].severity == ErrorSeverity.CRITICAL

    @pytest.mark.asyncio
    async def test_check_entity_preservation_handles_llm_error(self) -> None:
        """Test entity preservation check handles LLM errors gracefully."""
        mock_provider = AsyncMock()
        mock_provider.complete = AsyncMock(side_effect=Exception("LLM error"))

        agent = HallucinationAgent(mock_provider)

        task = TranslationTask(
            source_text="Test",
            translation="Prueba",
            source_lang="en",
            target_lang="es",
        )

        errors = await agent._check_entity_preservation(task)

        # Should return empty list, not raise
        assert errors == []


@pytest.mark.unit
class TestFactualConsistencyCheck:
    """Test factual consistency checking."""

    @pytest.mark.asyncio
    async def test_check_factual_consistency_no_issues(self) -> None:
        """Test factual consistency check with no issues."""
        mock_provider = AsyncMock()
        mock_provider.complete = AsyncMock(return_value='{"errors": []}')

        agent = HallucinationAgent(mock_provider)

        task = TranslationTask(
            source_text="The meeting is at 3 PM.",
            translation="La reunión es a las 3 PM.",
            source_lang="en",
            target_lang="es",
        )

        errors = await agent._check_factual_consistency(task)

        assert len(errors) == 0

    @pytest.mark.asyncio
    async def test_check_factual_consistency_detects_hallucination(self) -> None:
        """Test factual consistency detects hallucinated content."""
        mock_provider = AsyncMock()
        mock_provider.complete = AsyncMock(
            return_value="""{
            "errors": [
                {
                    "subcategory": "hallucination_addition",
                    "severity": "major",
                    "description": "Translation adds information not in source",
                    "hallucinated_content": "importante reunión de negocios"
                }
            ]
        }"""
        )

        agent = HallucinationAgent(mock_provider)

        task = TranslationTask(
            source_text="The meeting is at 3 PM.",
            translation="La importante reunión de negocios es a las 3 PM.",  # Added content
            source_lang="en",
            target_lang="es",
        )

        errors = await agent._check_factual_consistency(task)

        assert len(errors) == 1
        assert errors[0].subcategory == "hallucination_addition"
        assert errors[0].severity == ErrorSeverity.MAJOR

    @pytest.mark.asyncio
    async def test_check_factual_consistency_handles_error(self) -> None:
        """Test factual consistency check handles errors gracefully."""
        mock_provider = AsyncMock()
        mock_provider.complete = AsyncMock(side_effect=Exception("API error"))

        agent = HallucinationAgent(mock_provider)

        task = TranslationTask(
            source_text="Test",
            translation="Prueba",
            source_lang="en",
            target_lang="es",
        )

        errors = await agent._check_factual_consistency(task)

        # Should return empty list, not raise
        assert errors == []


@pytest.mark.unit
class TestEvaluate:
    """Test the main evaluate method."""

    @pytest.mark.asyncio
    async def test_evaluate_combines_all_checks(self) -> None:
        """Test that evaluate runs all checks."""
        mock_provider = AsyncMock()
        mock_provider.complete = AsyncMock(return_value='{"errors": []}')

        agent = HallucinationAgent(mock_provider)

        task = TranslationTask(
            source_text="Hello world.",
            translation="Hola mundo.",
            source_lang="en",
            target_lang="es",
        )

        errors = await agent.evaluate(task)

        # Should complete without errors for normal translation
        assert isinstance(errors, list)
        # LLM should be called for entity and factual checks
        assert mock_provider.complete.call_count == 2

    @pytest.mark.asyncio
    async def test_evaluate_detects_length_issue(self) -> None:
        """Test that evaluate detects length ratio issues."""
        mock_provider = AsyncMock()
        mock_provider.complete = AsyncMock(return_value='{"errors": []}')

        agent = HallucinationAgent(mock_provider)

        task = TranslationTask(
            source_text="Short.",
            translation="Esta es una traducción muy larga que claramente no corresponde al texto original corto.",
            source_lang="en",
            target_lang="es",
        )

        errors = await agent.evaluate(task)

        # Should find length ratio issue
        length_errors = [e for e in errors if "length" in e.subcategory]
        assert len(length_errors) == 1

    @pytest.mark.asyncio
    async def test_evaluate_raises_on_critical_failure(self) -> None:
        """Test evaluate raises AgentEvaluationError on critical failure."""
        from kttc.agents.base import AgentEvaluationError

        mock_provider = AsyncMock()
        # Make the check raise an exception that propagates
        mock_provider.complete = AsyncMock(side_effect=Exception("Critical failure"))

        agent = HallucinationAgent(mock_provider)

        # Patch to make error propagate
        with pytest.raises(AgentEvaluationError):
            # Create a scenario where error propagates
            agent._check_length_ratio = MagicMock(side_effect=Exception("Test error"))
            task = TranslationTask(
                source_text="Test",
                translation="Prueba",
                source_lang="en",
                target_lang="es",
            )
            await agent.evaluate(task)


@pytest.mark.unit
class TestBasePrompt:
    """Test base prompt generation."""

    def test_get_base_prompt(self) -> None:
        """Test base prompt contains required sections."""
        mock_provider = MagicMock()
        agent = HallucinationAgent(mock_provider)

        prompt = agent.get_base_prompt()

        assert "Entity Preservation" in prompt
        assert "Factual Consistency" in prompt
        assert "Length Ratio" in prompt
        assert "hallucinations" in prompt.lower()
        assert "JSON" in prompt
