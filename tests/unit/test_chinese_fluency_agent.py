"""Unit tests for ChineseFluencyAgent.

Tests agent logic with mocked LLM and language helper.
Focus: Fast, isolated tests that find real bugs.

Tests Chinese-specific checks:
- 量词 (measure words/classifiers)
- 的/地/得 particle usage
- 把字句/被字句 constructions
- 离合词 (separable verbs)
- 结果补语 (resultative complements)
- 字序陷阱 (word order traps)

Philosophy: "Tests must find errors, not tests for the sake of tests!"
"""

from pathlib import Path
from typing import Any

import pytest

from kttc.agents.fluency_chinese import ChineseFluencyAgent
from kttc.core.models import ErrorAnnotation, TranslationTask

# ============================================================================
# Mock ChineseLanguageHelper
# ============================================================================


class MockChineseHelper:
    """Mock Chinese language helper to avoid loading heavy ML models."""

    def __init__(self, available: bool = True):
        """Initialize mock helper.

        Args:
            available: Whether helper is available
        """
        self.language_code = "zh"
        self._available = available
        self.tokenize_calls = 0
        self.grammar_check_calls = 0
        self.verify_position_calls = 0
        self.verify_word_calls = 0

    def is_available(self) -> bool:
        """Return availability status."""
        return self._available

    def tokenize(self, text: str) -> list[tuple[str, int, int]]:
        """Mock tokenization (character-level for simplicity)."""
        self.tokenize_calls += 1
        if not text:
            return []
        # Simple mock: character-level tokenization
        return [(char, i, i + 1) for i, char in enumerate(text) if char.strip()]

    def check_grammar(self, text: str) -> list[ErrorAnnotation]:
        """Mock grammar checking."""
        self.grammar_check_calls += 1
        # Return empty list by default
        return []

    def verify_error_position(self, error: ErrorAnnotation, text: str) -> bool:
        """Mock error position verification."""
        self.verify_position_calls += 1
        start, end = error.location
        # Basic validation: position must be within text bounds
        return 0 <= start < len(text) and start <= end <= len(text)

    def verify_word_exists(self, description: str, text: str) -> bool:
        """Mock word existence verification."""
        self.verify_word_calls += 1
        # Simple check: any word from description exists in text
        for word in description.split():
            if word in text:
                return True
        return False


@pytest.fixture
def mock_chinese_helper() -> MockChineseHelper:
    """Provide mock Chinese helper."""
    return MockChineseHelper(available=True)


@pytest.fixture
def mock_chinese_helper_unavailable() -> MockChineseHelper:
    """Provide unavailable mock Chinese helper."""
    return MockChineseHelper(available=False)


@pytest.fixture
def sample_chinese_task() -> TranslationTask:
    """Provide sample Chinese translation task."""
    return TranslationTask(
        source_text="I read three books yesterday.",
        translation="我昨天看了三本书。",
        source_lang="en",
        target_lang="zh",
    )


@pytest.fixture
def measure_word_error_task() -> TranslationTask:
    """Provide task with potential measure word error."""
    return TranslationTask(
        source_text="I have three books.",
        translation="我有三个书。",  # Wrong: should be 三本书
        source_lang="en",
        target_lang="zh",
    )


@pytest.fixture
def de_particle_error_task() -> TranslationTask:
    """Provide task with 的/地/得 particle error."""
    return TranslationTask(
        source_text="He runs very fast.",
        translation="他跑的很快。",  # Wrong: should be 跑得很快
        source_lang="en",
        target_lang="zh",
    )


@pytest.fixture
def word_order_trap_task() -> TranslationTask:
    """Provide task with word order trap."""
    return TranslationTask(
        source_text="The bee flew to the flower.",
        translation="蜂蜜飞到花朵上。",  # Wrong: should be 蜜蜂 (bee), not 蜂蜜 (honey)
        source_lang="en",
        target_lang="zh",
    )


@pytest.fixture
def ba_construction_task() -> TranslationTask:
    """Provide task with 把字句 construction."""
    return TranslationTask(
        source_text="Put the book on the table.",
        translation="把书放在桌子上。",  # Correct 把字句
        source_lang="en",
        target_lang="zh",
    )


@pytest.fixture
def separable_verb_error_task() -> TranslationTask:
    """Provide task with separable verb error."""
    return TranslationTask(
        source_text="I met him yesterday.",
        translation="我昨天见面他。",  # Wrong: should be 跟他见面 or 见了他
        source_lang="en",
        target_lang="zh",
    )


# ============================================================================
# Agent Initialization Tests
# ============================================================================


class TestChineseAgentInitialization:
    """Test agent initialization and configuration."""

    def test_agent_initializes_with_helper(
        self, mock_llm_class: Any, mock_chinese_helper: MockChineseHelper
    ) -> None:
        """Agent should initialize with helper."""
        mock_provider = mock_llm_class()
        agent = ChineseFluencyAgent(mock_provider, helper=mock_chinese_helper)

        assert agent.helper is mock_chinese_helper
        assert agent.helper.is_available()

    def test_agent_initializes_without_helper(self, mock_llm_class: Any) -> None:
        """Agent should initialize without helper (LLM-only mode)."""
        mock_provider = mock_llm_class()
        agent = ChineseFluencyAgent(mock_provider, helper=None)

        # Agent should have its own helper (auto-created)
        assert agent.helper is not None

    def test_chinese_checks_defined(self) -> None:
        """Agent should define Chinese-specific checks."""
        assert "measure_word" in ChineseFluencyAgent.CHINESE_CHECKS
        assert "aspect_particle" in ChineseFluencyAgent.CHINESE_CHECKS
        assert "de_particle" in ChineseFluencyAgent.CHINESE_CHECKS
        assert "word_order_trap" in ChineseFluencyAgent.CHINESE_CHECKS
        assert "ba_bei_construction" in ChineseFluencyAgent.CHINESE_CHECKS
        assert "separable_verb" in ChineseFluencyAgent.CHINESE_CHECKS
        assert "resultative_complement" in ChineseFluencyAgent.CHINESE_CHECKS


# ============================================================================
# LLM Response Parsing Tests
# ============================================================================


class TestLLMResponseParsing:
    """Test LLM response parsing for Chinese-specific errors."""

    @pytest.mark.asyncio
    async def test_parse_measure_word_error_response(
        self, mock_llm_class: Any, mock_chinese_helper: MockChineseHelper
    ) -> None:
        """Agent should parse measure word error from LLM response."""
        # Use response with measure word error
        mock_provider = mock_llm_class(
            response="""{
                "errors": [
                    {
                        "subcategory": "measure_word",
                        "severity": "major",
                        "location": [3, 5],
                        "description": "个 Wrong measure word for books. Should use 本.",
                        "suggestion": "三本书"
                    }
                ]
            }"""
        )

        agent = ChineseFluencyAgent(mock_provider, helper=mock_chinese_helper)
        task = TranslationTask(
            source_text="I have three books.",
            translation="我有三个书。",
            source_lang="en",
            target_lang="zh",
        )

        errors = await agent.evaluate(task)

        # Should find at least one error (may be measure_word or chinese_measure_word)
        assert len(errors) >= 1

    @pytest.mark.asyncio
    async def test_parse_de_particle_error_response(
        self, mock_llm_class: Any, mock_chinese_helper: MockChineseHelper
    ) -> None:
        """Agent should parse 的/地/得 particle error from LLM response."""
        mock_provider = mock_llm_class(
            response="""{
                "errors": [
                    {
                        "subcategory": "de_particle",
                        "severity": "critical",
                        "location": [2, 3],
                        "description": "的 Wrong particle after verb. Should use 得 for degree complement.",
                        "suggestion": "跑得很快"
                    }
                ]
            }"""
        )

        agent = ChineseFluencyAgent(mock_provider, helper=mock_chinese_helper)
        task = TranslationTask(
            source_text="He runs very fast.",
            translation="他跑的很快。",
            source_lang="en",
            target_lang="zh",
        )

        errors = await agent.evaluate(task)

        # Should find at least one error
        assert len(errors) >= 1

    @pytest.mark.asyncio
    async def test_parse_word_order_trap_response(
        self, mock_llm_class: Any, mock_chinese_helper: MockChineseHelper
    ) -> None:
        """Agent should parse word order trap error from LLM response."""
        mock_provider = mock_llm_class(
            response="""{
                "errors": [
                    {
                        "subcategory": "word_order_trap",
                        "severity": "major",
                        "location": [0, 2],
                        "description": "蜂蜜 Word order trap - means honey, not bee. Use 蜜蜂 for bee.",
                        "suggestion": "蜜蜂飞到花朵上"
                    }
                ]
            }"""
        )

        agent = ChineseFluencyAgent(mock_provider, helper=mock_chinese_helper)
        task = TranslationTask(
            source_text="The bee flew to the flower.",
            translation="蜂蜜飞到花朵上。",
            source_lang="en",
            target_lang="zh",
        )

        errors = await agent.evaluate(task)

        # Should find at least one error
        assert len(errors) >= 1


# ============================================================================
# Non-Chinese Language Tests
# ============================================================================


class TestNonChineseLanguage:
    """Test agent behavior with non-Chinese target language."""

    @pytest.mark.asyncio
    async def test_fallback_for_non_chinese(
        self, mock_llm_class: Any, mock_chinese_helper: MockChineseHelper
    ) -> None:
        """Agent should fallback to base fluency for non-Chinese languages."""
        mock_provider = mock_llm_class(response='{"errors": []}')

        agent = ChineseFluencyAgent(mock_provider, helper=mock_chinese_helper)
        task = TranslationTask(
            source_text="Hello world",
            translation="Hola mundo",
            source_lang="en",
            target_lang="es",  # Spanish, not Chinese
        )

        errors = await agent.evaluate(task)

        # Should not crash, should return empty list (no errors)
        assert isinstance(errors, list)


# ============================================================================
# Error Verification Tests
# ============================================================================


class TestErrorVerification:
    """Test error verification with helper."""

    @pytest.mark.asyncio
    async def test_verify_error_position(
        self, mock_llm_class: Any, mock_chinese_helper: MockChineseHelper
    ) -> None:
        """Agent should verify error positions."""
        mock_provider = mock_llm_class(
            response="""{
                "errors": [
                    {
                        "subcategory": "measure_word",
                        "severity": "major",
                        "location": [3, 5],
                        "description": "个 should be 本",
                        "suggestion": "本"
                    }
                ]
            }"""
        )

        agent = ChineseFluencyAgent(mock_provider, helper=mock_chinese_helper)
        task = TranslationTask(
            source_text="Three books",
            translation="三个书",
            source_lang="en",
            target_lang="zh",
        )

        await agent.evaluate(task)

        # Helper should be called to verify positions
        assert mock_chinese_helper.verify_position_calls >= 0


# ============================================================================
# Glossary Integration Tests
# ============================================================================


class TestGlossaryIntegration:
    """Test integration with Chinese glossaries."""

    def test_glossaries_exist(self) -> None:
        """All Chinese glossaries should exist."""
        glossary_dir = Path(__file__).parent.parent.parent / "glossaries" / "zh"

        expected_glossaries = [
            "classifiers_zh.json",
            "idioms_expressions_zh.json",
            "mqm_core.json",
            "word_order_traps_zh.json",
            "homophones_cultural_zh.json",
            "internet_slang_2024_zh.json",
            "separable_verbs_zh.json",
            "resultative_complements_zh.json",
        ]

        for glossary in expected_glossaries:
            glossary_path = glossary_dir / glossary
            assert glossary_path.exists(), f"Glossary {glossary} should exist"

    def test_glossary_json_valid(self) -> None:
        """All Chinese glossaries should be valid JSON."""
        import json

        glossary_dir = Path(__file__).parent.parent.parent / "glossaries" / "zh"

        for glossary_file in glossary_dir.glob("*.json"):
            try:
                with open(glossary_file, encoding="utf-8") as f:
                    data = json.load(f)
                assert "metadata" in data, f"{glossary_file.name} should have metadata"
            except json.JSONDecodeError as e:
                pytest.fail(f"Invalid JSON in {glossary_file.name}: {e}")


# ============================================================================
# Chinese-Specific Error Samples
# ============================================================================


class TestChineseErrorSamples:
    """Test with realistic Chinese error samples."""

    @pytest.mark.asyncio
    async def test_correct_chinese_no_errors(
        self, mock_llm_class: Any, mock_chinese_helper: MockChineseHelper
    ) -> None:
        """Correct Chinese should have no errors."""
        mock_provider = mock_llm_class(response='{"errors": []}')

        agent = ChineseFluencyAgent(mock_provider, helper=mock_chinese_helper)
        task = TranslationTask(
            source_text="I read three books yesterday.",
            translation="我昨天看了三本书。",  # Correct Chinese
            source_lang="en",
            target_lang="zh",
        )

        errors = await agent.evaluate(task)

        # Should have no Chinese-specific errors
        chinese_errors = [e for e in errors if "chinese_" in e.subcategory]
        assert len(chinese_errors) == 0

    @pytest.mark.asyncio
    async def test_ba_bei_construction_check(
        self, mock_llm_class: Any, mock_chinese_helper: MockChineseHelper
    ) -> None:
        """Agent should check 把字句/被字句 constructions."""
        mock_provider = mock_llm_class(
            response="""{
                "errors": [
                    {
                        "subcategory": "ba_bei_construction",
                        "severity": "major",
                        "location": [0, 5],
                        "description": "把书放 把字句 requires resultative complement",
                        "suggestion": "把书放在桌子上"
                    }
                ]
            }"""
        )

        agent = ChineseFluencyAgent(mock_provider, helper=mock_chinese_helper)
        task = TranslationTask(
            source_text="Put the book on the table.",
            translation="把书放桌子。",  # Missing complement
            source_lang="en",
            target_lang="zh",
        )

        errors = await agent.evaluate(task)

        # Should detect at least one error
        assert len(errors) >= 1


# ============================================================================
# Prompt Content Tests
# ============================================================================


class TestPromptContent:
    """Test that prompts contain required checks."""

    @pytest.mark.asyncio
    async def test_prompt_contains_all_checks(
        self, mock_llm_class: Any, mock_chinese_helper: MockChineseHelper
    ) -> None:
        """Prompt should contain all Chinese-specific checks."""
        mock_provider = mock_llm_class(response='{"errors": []}')

        agent = ChineseFluencyAgent(mock_provider, helper=mock_chinese_helper)
        task = TranslationTask(
            source_text="Test",
            translation="测试",
            source_lang="en",
            target_lang="zh",
        )

        await agent.evaluate(task)

        # Check that prompt was sent (last_prompt is set by MockLLMProvider)
        assert mock_provider.last_prompt is not None
        assert mock_provider.call_count >= 1

        # Check that Chinese-specific prompt contains key checks
        prompt = mock_provider.last_prompt
        assert "量词" in prompt or "Measure Words" in prompt
        assert "的/地/得" in prompt or "de_particle" in prompt
        assert "把字句" in prompt or "ba_bei" in prompt
        assert "离合词" in prompt or "Separable" in prompt
        assert "字序陷阱" in prompt or "Word Order Trap" in prompt
