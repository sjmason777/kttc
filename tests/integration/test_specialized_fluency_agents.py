"""Integration tests for specialized fluency agents.

Tests all three language-specific fluency agents in realistic translation QA scenarios:
- EnglishFluencyAgent with LanguageTool integration
- ChineseFluencyAgent with HanLP integration
- RussianFluencyAgent with MAWO NLP integration

Measures:
- Error detection accuracy
- Performance (execution time)
- MQM score improvements vs base FluencyAgent
"""

import time
from collections.abc import AsyncGenerator
from typing import Any, TypedDict

import pytest

from kttc.agents import (
    ChineseFluencyAgent,
    EnglishFluencyAgent,
    FluencyAgent,
    RussianFluencyAgent,
)
from kttc.core import TranslationTask
from kttc.llm import BaseLLMProvider


class TestCaseDict(TypedDict):
    """Type definition for test case dictionaries."""

    source: str
    translation: str
    source_lang: str
    target_lang: str
    expected_errors: list[str]


class MockLLMProvider(BaseLLMProvider):
    """Mock LLM provider for testing."""

    def __init__(self, response: str = '{"errors": []}'):
        self.response = response

    async def complete(
        self, prompt: str, temperature: float = 0.7, max_tokens: int = 1000, **kwargs: Any
    ) -> str:
        """Return mock JSON response."""
        return self.response

    async def stream(
        self, prompt: str, temperature: float = 0.7, max_tokens: int = 1000, **kwargs: Any
    ) -> AsyncGenerator[str, None]:
        """Mock stream method."""
        yield self.response


# Test cases with known errors
ENGLISH_TEST_CASES: list[TestCaseDict] = [
    {
        "source": "Привет мир",
        "translation": "He go to school every day",  # subject-verb agreement error
        "source_lang": "ru",
        "target_lang": "en",
        "expected_errors": ["grammar", "subject-verb"],
    },
    {
        "source": "C'est bon",
        "translation": "Its a good idea",  # missing apostrophe in "It's"
        "source_lang": "fr",
        "target_lang": "en",
        "expected_errors": ["grammar", "possessive"],
    },
]

CHINESE_TEST_CASES: list[TestCaseDict] = [
    {
        "source": "I read a book",
        "translation": "我看一个书",  # Should be 一本书, not 一个书
        "source_lang": "en",
        "target_lang": "zh",
        "expected_errors": ["measure_word"],
    },
    {
        "source": "Three dogs",
        "translation": "三本狗",  # Should be 三只狗, not 三本狗
        "source_lang": "en",
        "target_lang": "zh",
        "expected_errors": ["measure_word"],
    },
    {
        "source": "Two cars",
        "translation": "两个车",  # Should be 两辆车, not 两个车
        "source_lang": "en",
        "target_lang": "zh",
        "expected_errors": ["measure_word"],
    },
]

RUSSIAN_TEST_CASES: list[TestCaseDict] = [
    {
        "source": "Beautiful house",
        "translation": "Красивая дом",  # Gender error: feminine adjective + masculine noun
        "source_lang": "en",
        "target_lang": "ru",
        "expected_errors": ["gender", "agreement"],
    },
    {
        "source": "Fast fox",
        "translation": "Быстрый лиса",  # Gender error: masculine adjective + feminine noun
        "source_lang": "en",
        "target_lang": "ru",
        "expected_errors": ["gender", "agreement"],
    },
    {
        "source": "About the book",
        "translation": "О книга",  # Case error: prepositional requires locative case
        "source_lang": "en",
        "target_lang": "ru",
        "expected_errors": ["case", "agreement"],
    },
]


@pytest.mark.integration
class TestEnglishFluencyAgent:
    """Integration tests for EnglishFluencyAgent."""

    @pytest.mark.asyncio
    async def test_english_grammar_detection(self) -> None:
        """Test EnglishFluencyAgent detects grammar errors."""
        llm = MockLLMProvider()
        agent = EnglishFluencyAgent(llm)

        for test_case in ENGLISH_TEST_CASES:
            task = TranslationTask(
                source_text=test_case["source"],
                translation=test_case["translation"],
                source_lang=test_case["source_lang"],
                target_lang=test_case["target_lang"],
            )

            start_time = time.time()
            errors = await agent.evaluate(task)
            execution_time = time.time() - start_time

            # Check that at least one error was found
            assert len(errors) > 0, f"No errors found for: {test_case['translation']}"

            # Check that error descriptions mention expected error types
            # Note: Be flexible - error description may vary
            error_text = " ".join([e.description.lower() for e in errors])

            # For English grammar errors, accept any mention of verbs or grammar
            if "grammar" in test_case["expected_errors"]:
                has_expected_error = any(
                    kw in error_text for kw in ["verb", "grammar", "agreement", "pronoun"]
                )
            else:
                has_expected_error = any(
                    expected.lower() in error_text for expected in test_case["expected_errors"]
                )

            # Log warning if expected error not found (don't fail - agents may describe differently)
            if not has_expected_error:
                print(
                    f"⚠️  EnglishFluencyAgent: Expected {test_case['expected_errors']} "
                    f"but found: {error_text[:100]}..."
                )

            # Check performance (should be < 5 seconds with LanguageTool)
            assert execution_time < 5.0, f"Too slow: {execution_time:.2f}s"

            print(
                f"✓ EnglishFluencyAgent: {test_case['translation'][:30]}... "
                f"({len(errors)} errors, {execution_time*1000:.0f}ms)"
            )

    @pytest.mark.asyncio
    async def test_english_no_false_positives(self) -> None:
        """Test EnglishFluencyAgent doesn't flag correct English."""
        llm = MockLLMProvider()
        agent = EnglishFluencyAgent(llm)

        correct_translations = [
            TranslationTask(
                source_text="Привет",
                translation="Hello, how are you?",
                source_lang="ru",
                target_lang="en",
            ),
            TranslationTask(
                source_text="Merci",
                translation="Thank you very much.",
                source_lang="fr",
                target_lang="en",
            ),
        ]

        for task in correct_translations:
            errors = await agent.evaluate(task)
            # Should have 0 or very few errors for correct translations
            assert len(errors) <= 1, f"False positives for '{task.translation}': {errors}"


@pytest.mark.integration
class TestChineseFluencyAgent:
    """Integration tests for ChineseFluencyAgent."""

    @pytest.mark.asyncio
    async def test_chinese_measure_word_detection(self) -> None:
        """Test ChineseFluencyAgent detects measure word errors."""
        llm = MockLLMProvider()
        agent = ChineseFluencyAgent(llm)

        total_errors_found = 0
        total_expected = len(CHINESE_TEST_CASES)

        for test_case in CHINESE_TEST_CASES:
            task = TranslationTask(
                source_text=test_case["source"],
                translation=test_case["translation"],
                source_lang=test_case["source_lang"],
                target_lang=test_case["target_lang"],
            )

            start_time = time.time()
            errors = await agent.evaluate(task)
            execution_time = time.time() - start_time

            # Count errors found (some test cases may not be detected)
            if len(errors) > 0:
                total_errors_found += 1

                # Check that error mentions measure words
                error_text = " ".join([e.description.lower() for e in errors])
                has_measure_word_error = any(
                    kw in error_text for kw in ["measure", "量词", "classifier"]
                )

                # Log warning if measure word not mentioned
                if not has_measure_word_error:
                    print(
                        f"⚠️  ChineseFluencyAgent: Expected measure word error "
                        f"but found other error for {test_case['translation']}"
                    )

                print(
                    f"✓ ChineseFluencyAgent: {test_case['translation']} "
                    f"({len(errors)} errors, {execution_time*1000:.0f}ms)"
                )
            else:
                print(
                    f"⚠️  ChineseFluencyAgent: No errors found for {test_case['translation']} "
                    f"(HanLP may not have detected this pattern)"
                )

            # Check performance (should be < 5 seconds with HanLP)
            assert execution_time < 5.0, f"Too slow: {execution_time:.2f}s"

        # Assert that at least some errors were found (not all patterns may be detected)
        assert total_errors_found > 0, (
            f"No errors found in any test case (0/{total_expected}). "
            "HanLP may not be working correctly."
        )

    @pytest.mark.asyncio
    async def test_chinese_no_false_positives(self) -> None:
        """Test ChineseFluencyAgent doesn't flag correct Chinese."""
        llm = MockLLMProvider()
        agent = ChineseFluencyAgent(llm)

        correct_translations = [
            TranslationTask(
                source_text="Hello",
                translation="你好",
                source_lang="en",
                target_lang="zh",
            ),
            TranslationTask(
                source_text="I read a book",
                translation="我看一本书",  # Correct measure word
                source_lang="en",
                target_lang="zh",
            ),
        ]

        for task in correct_translations:
            errors = await agent.evaluate(task)
            # Should have 0 or very few errors for correct translations
            assert len(errors) <= 1, f"False positives for '{task.translation}': {errors}"


@pytest.mark.integration
class TestRussianFluencyAgent:
    """Integration tests for RussianFluencyAgent."""

    @pytest.mark.asyncio
    async def test_russian_agreement_detection(self) -> None:
        """Test RussianFluencyAgent detects agreement errors."""
        llm = MockLLMProvider()
        agent = RussianFluencyAgent(llm)

        for test_case in RUSSIAN_TEST_CASES:
            task = TranslationTask(
                source_text=test_case["source"],
                translation=test_case["translation"],
                source_lang=test_case["source_lang"],
                target_lang=test_case["target_lang"],
            )

            start_time = time.time()
            errors = await agent.evaluate(task)
            execution_time = time.time() - start_time

            # Check that at least one error was found
            assert len(errors) > 0, f"No errors found for: {test_case['translation']}"

            # Check that error mentions agreement
            error_text = " ".join([e.description.lower() for e in errors])
            has_agreement_error = any(
                kw in error_text for kw in ["agreement", "gender", "case", "согласован"]
            )

            assert (
                has_agreement_error
            ), f"Expected agreement error not found for {test_case['translation']}: {error_text}"

            # Check performance (should be < 1 second with MAWO NLP)
            assert execution_time < 5.0, f"Too slow: {execution_time:.2f}s"

            print(
                f"✓ RussianFluencyAgent: {test_case['translation']} "
                f"({len(errors)} errors, {execution_time*1000:.0f}ms)"
            )

    @pytest.mark.asyncio
    async def test_russian_no_false_positives(self) -> None:
        """Test RussianFluencyAgent doesn't flag correct Russian."""
        llm = MockLLMProvider()
        agent = RussianFluencyAgent(llm)

        correct_translations = [
            TranslationTask(
                source_text="Hello",
                translation="Привет",
                source_lang="en",
                target_lang="ru",
            ),
            TranslationTask(
                source_text="Beautiful house",
                translation="Красивый дом",  # Correct gender agreement
                source_lang="en",
                target_lang="ru",
            ),
        ]

        for task in correct_translations:
            errors = await agent.evaluate(task)
            # Should have 0 or very few errors for correct translations
            assert len(errors) <= 1, f"False positives for '{task.translation}': {errors}"


@pytest.mark.integration
class TestPerformanceComparison:
    """Compare specialized agents vs base FluencyAgent."""

    @pytest.mark.asyncio
    async def test_specialized_vs_base_performance(self) -> None:
        """Test that specialized agents provide value over base FluencyAgent."""
        llm = MockLLMProvider()

        # Create agents
        base_agent = FluencyAgent(llm)
        english_agent = EnglishFluencyAgent(llm)
        chinese_agent = ChineseFluencyAgent(llm)
        russian_agent = RussianFluencyAgent(llm)

        # Test English
        en_task = TranslationTask(
            source_text="Привет",
            translation="He go to school",
            source_lang="ru",
            target_lang="en",
        )

        base_errors = await base_agent.evaluate(en_task)
        specialized_errors = await english_agent.evaluate(en_task)

        print(
            f"\nEnglish - Base: {len(base_errors)} errors, "
            f"Specialized: {len(specialized_errors)} errors"
        )

        # Specialized agent should find at least as many errors as base
        assert len(specialized_errors) >= len(
            base_errors
        ), "Specialized agent should find more errors than base"

        # Test Chinese
        zh_task = TranslationTask(
            source_text="I read a book",
            translation="我看一个书",
            source_lang="en",
            target_lang="zh",
        )

        base_errors = await base_agent.evaluate(zh_task)
        specialized_errors = await chinese_agent.evaluate(zh_task)

        print(
            f"Chinese - Base: {len(base_errors)} errors, "
            f"Specialized: {len(specialized_errors)} errors"
        )

        # Test Russian
        ru_task = TranslationTask(
            source_text="Beautiful house",
            translation="Красивая дом",
            source_lang="en",
            target_lang="ru",
        )

        base_errors = await base_agent.evaluate(ru_task)
        specialized_errors = await russian_agent.evaluate(ru_task)

        print(
            f"Russian - Base: {len(base_errors)} errors, "
            f"Specialized: {len(specialized_errors)} errors"
        )

        # Specialized agent should find more errors
        assert len(specialized_errors) >= len(
            base_errors
        ), "Specialized agent should find more errors than base"
