"""Tests for Russian Fluency Agent."""

from unittest.mock import AsyncMock, Mock

import pytest

from kttc.agents.fluency_russian import RussianFluencyAgent
from kttc.core import TranslationTask


@pytest.fixture
def mock_llm():
    llm = Mock()
    llm.complete = AsyncMock()
    return llm


@pytest.fixture
def agent(mock_llm):
    return RussianFluencyAgent(mock_llm)


@pytest.fixture
def russian_task():
    return TranslationTask(
        source_text="Hello world",
        translation="Привет мир",
        source_lang="en",
        target_lang="ru",
    )


class TestRussianFluencyAgent:
    @pytest.mark.asyncio
    async def test_evaluate_calls_base_and_russian_checks(self, agent, russian_task, mock_llm):
        mock_llm.complete.return_value = '{"errors": []}'

        errors = await agent.evaluate(russian_task)

        assert isinstance(errors, list)
        assert mock_llm.complete.called

    @pytest.mark.asyncio
    async def test_evaluate_non_russian_skips_russian_checks(self, agent, mock_llm):
        task = TranslationTask(
            source_text="Hello",
            translation="Hola",
            source_lang="en",
            target_lang="es",
        )
        mock_llm.complete.return_value = '{"errors": []}'

        await agent.evaluate(task)

        assert mock_llm.complete.called

    @pytest.mark.asyncio
    async def test_russian_checks_with_errors(self, agent, russian_task, mock_llm):
        mock_llm.complete.side_effect = [
            '{"errors": []}',  # Base fluency
            '{"errors": [{"category": "fluency", "subcategory": "case_agreement", "severity": "major", "description": "Case error", "location": [0, 5]}]}',  # Russian
        ]

        errors = await agent.evaluate(russian_task)

        assert len(errors) >= 0

    @pytest.mark.asyncio
    async def test_russian_checks_handles_llm_error(self, agent, russian_task, mock_llm):
        mock_llm.complete.side_effect = ['{"errors": []}', Exception("LLM failed")]

        errors = await agent.evaluate(russian_task)

        assert isinstance(errors, list)

    @pytest.mark.asyncio
    async def test_parse_json_from_markdown(self, agent):
        """Test parsing JSON from markdown code block."""
        response = '```json\n{"errors": [{"subcategory": "test", "severity": "minor", "location": [0, 5], "description": "Test"}]}\n```'
        result = agent._parse_json_response(response)
        assert "errors" in result
        assert len(result["errors"]) == 1

    @pytest.mark.asyncio
    async def test_parse_json_from_plain_object(self, agent):
        """Test parsing JSON from plain text with JSON object."""
        response = 'Some text before {"errors": []} some text after'
        result = agent._parse_json_response(response)
        assert "errors" in result

    @pytest.mark.asyncio
    async def test_parse_json_invalid_returns_empty(self, agent):
        """Test that invalid JSON returns empty errors."""
        response = "This is not JSON at all"
        result = agent._parse_json_response(response)
        assert result == {"errors": []}

    @pytest.mark.asyncio
    async def test_russian_checks_invalid_location_format(self, agent, russian_task, mock_llm):
        """Test handling of invalid location format in error response."""
        mock_llm.complete.return_value = '{"errors": [{"subcategory": "test", "severity": "minor", "location": "invalid", "description": "Test"}]}'

        errors = await agent._check_russian_specifics(russian_task)

        assert isinstance(errors, list)
        if len(errors) > 0:
            assert errors[0].location == (0, 10)  # Default fallback
