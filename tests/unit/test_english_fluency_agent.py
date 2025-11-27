"""Unit tests for English fluency agent module.

Tests English-specific fluency checking functionality.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from kttc.agents.fluency_english import EnglishFluencyAgent
from kttc.core import TranslationTask


@pytest.mark.unit
class TestEnglishFluencyAgentInitialization:
    """Test English fluency agent initialization."""

    def test_default_initialization(self) -> None:
        """Test default initialization parameters."""
        mock_provider = MagicMock()
        agent = EnglishFluencyAgent(mock_provider)

        assert agent.llm_provider == mock_provider
        assert agent.temperature == 0.1
        assert agent.max_tokens == 2000
        assert agent.helper is not None
        assert agent.traps_validator is not None

    def test_custom_initialization(self) -> None:
        """Test custom initialization parameters."""
        mock_provider = MagicMock()
        mock_helper = MagicMock()
        mock_traps = MagicMock()

        agent = EnglishFluencyAgent(
            mock_provider,
            temperature=0.5,
            max_tokens=3000,
            helper=mock_helper,
            traps_validator=mock_traps,
        )

        assert agent.temperature == 0.5
        assert agent.max_tokens == 3000
        assert agent.helper == mock_helper
        assert agent.traps_validator == mock_traps

    def test_english_checks_defined(self) -> None:
        """Test that English checks are defined."""
        assert "grammar" in EnglishFluencyAgent.ENGLISH_CHECKS
        assert "spelling" in EnglishFluencyAgent.ENGLISH_CHECKS
        assert "tense" in EnglishFluencyAgent.ENGLISH_CHECKS
        assert "register" in EnglishFluencyAgent.ENGLISH_CHECKS
        assert "homophones" in EnglishFluencyAgent.ENGLISH_CHECKS
        assert "phrasal_verbs" in EnglishFluencyAgent.ENGLISH_CHECKS
        assert "idioms" in EnglishFluencyAgent.ENGLISH_CHECKS


@pytest.mark.unit
class TestGetBasePrompt:
    """Test base prompt generation."""

    def test_get_base_prompt_includes_english_specific(self) -> None:
        """Test base prompt includes English-specific checks."""
        mock_provider = MagicMock()
        agent = EnglishFluencyAgent(mock_provider)

        prompt = agent.get_base_prompt()

        assert "ENGLISH-SPECIFIC CHECKS" in prompt
        assert "fluency" in prompt.lower()


@pytest.mark.unit
class TestEnglishFluencyAgentEvaluate:
    """Test the evaluate method."""

    @pytest.mark.asyncio
    async def test_evaluate_non_english_fallback(self) -> None:
        """Test evaluation falls back to base for non-English."""
        mock_provider = AsyncMock()
        mock_provider.complete = AsyncMock(return_value='{"errors": []}')

        agent = EnglishFluencyAgent(mock_provider)

        task = TranslationTask(
            source_text="Hello",
            translation="Bonjour",
            source_lang="en",
            target_lang="fr",  # Not English
        )

        errors = await agent.evaluate(task)

        # Should complete without crashing
        assert isinstance(errors, list)

    @pytest.mark.asyncio
    async def test_evaluate_english_target(self) -> None:
        """Test evaluation for English target language."""
        mock_provider = AsyncMock()
        mock_provider.complete = AsyncMock(return_value='{"errors": []}')

        mock_helper = MagicMock()
        mock_helper.is_available.return_value = False

        mock_traps = MagicMock()
        mock_traps.is_available.return_value = False

        agent = EnglishFluencyAgent(mock_provider, helper=mock_helper, traps_validator=mock_traps)

        task = TranslationTask(
            source_text="Привет мир",
            translation="Hello world",
            source_lang="ru",
            target_lang="en",
        )

        errors = await agent.evaluate(task)

        assert isinstance(errors, list)

    @pytest.mark.asyncio
    async def test_evaluate_with_helper_available(self) -> None:
        """Test evaluation when helper is available."""
        mock_provider = AsyncMock()
        mock_provider.complete = AsyncMock(return_value='{"errors": []}')

        mock_helper = MagicMock()
        mock_helper.is_available.return_value = True
        mock_helper.check_grammar.return_value = []
        mock_helper.check_spelling.return_value = []

        mock_traps = MagicMock()
        mock_traps.is_available.return_value = False

        agent = EnglishFluencyAgent(mock_provider, helper=mock_helper, traps_validator=mock_traps)

        task = TranslationTask(
            source_text="Test",
            translation="Test",
            source_lang="ru",
            target_lang="en",
        )

        errors = await agent.evaluate(task)

        assert isinstance(errors, list)

    @pytest.mark.asyncio
    async def test_evaluate_with_traps_validator(self) -> None:
        """Test evaluation when traps validator is available."""
        mock_provider = AsyncMock()
        mock_provider.complete = AsyncMock(return_value='{"errors": []}')

        mock_helper = MagicMock()
        mock_helper.is_available.return_value = False

        mock_traps = MagicMock()
        mock_traps.is_available.return_value = True
        mock_traps.detect_homophones.return_value = []
        mock_traps.detect_phrasal_verbs.return_value = []
        mock_traps.detect_idioms.return_value = []

        agent = EnglishFluencyAgent(mock_provider, helper=mock_helper, traps_validator=mock_traps)

        task = TranslationTask(
            source_text="Test",
            translation="Test",
            source_lang="ru",
            target_lang="en",
        )

        errors = await agent.evaluate(task)

        assert isinstance(errors, list)


@pytest.mark.unit
class TestHelperIntegration:
    """Test integration with EnglishLanguageHelper."""

    def test_helper_auto_created_when_none(self) -> None:
        """Test that helper is auto-created when None provided."""
        mock_provider = MagicMock()
        agent = EnglishFluencyAgent(mock_provider, helper=None)

        assert agent.helper is not None

    def test_traps_validator_auto_created_when_none(self) -> None:
        """Test that traps validator is auto-created when None provided."""
        mock_provider = MagicMock()
        agent = EnglishFluencyAgent(mock_provider, traps_validator=None)

        assert agent.traps_validator is not None

    def test_uses_provided_helper(self) -> None:
        """Test that provided helper is used."""
        mock_provider = MagicMock()
        mock_helper = MagicMock()
        mock_helper.is_available.return_value = True

        agent = EnglishFluencyAgent(mock_provider, helper=mock_helper)

        assert agent.helper is mock_helper


@pytest.mark.unit
class TestEnglishChecksConstants:
    """Test English checks constants."""

    def test_all_checks_have_descriptions(self) -> None:
        """Test that all checks have non-empty descriptions."""
        for check_name, description in EnglishFluencyAgent.ENGLISH_CHECKS.items():
            assert check_name, "Check name should not be empty"
            assert description, f"Description for {check_name} should not be empty"

    def test_expected_checks_present(self) -> None:
        """Test that expected checks are defined."""
        expected_checks = [
            "grammar",
            "spelling",
            "tense",
            "register",
            "homophones",
            "phrasal_verbs",
            "heteronyms",
            "adjective_order",
            "prepositions",
            "idioms",
        ]

        for check in expected_checks:
            assert check in EnglishFluencyAgent.ENGLISH_CHECKS, f"Missing check: {check}"
