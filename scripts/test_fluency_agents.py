#!/usr/bin/env python3.11
"""Test all three specialized fluency agents.

This script tests:
- EnglishFluencyAgent with LanguageTool integration
- ChineseFluencyAgent with HanLP integration
- RussianFluencyAgent with MAWO NLP integration
"""

import asyncio
import logging
import sys
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from kttc.agents import ChineseFluencyAgent, EnglishFluencyAgent, RussianFluencyAgent
from kttc.core import TranslationTask
from kttc.llm import BaseLLMProvider

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MockLLMProvider(BaseLLMProvider):
    """Mock LLM provider for testing."""

    async def complete(
        self, prompt: str, temperature: float = 0.7, max_tokens: int = 1000, **kwargs: Any
    ) -> str:
        """Return mock JSON response."""
        return '{"errors": []}'

    async def stream(
        self, prompt: str, temperature: float = 0.7, max_tokens: int = 1000, **kwargs: Any
    ) -> AsyncGenerator[str, None]:
        """Mock stream method."""
        yield '{"errors": []}'


async def test_english_fluency_agent() -> None:
    """Test EnglishFluencyAgent."""
    print("\n" + "=" * 60)
    print("Testing EnglishFluencyAgent")
    print("=" * 60)

    # Use mock LLM provider for testing
    llm = MockLLMProvider()

    # Create agent
    agent = EnglishFluencyAgent(llm)

    # Test case with grammar error
    task = TranslationTask(
        source_text="Привет мир",
        translation="He go to school",  # subject-verb agreement error
        source_lang="ru",
        target_lang="en",
    )

    try:
        errors = await agent.evaluate(task)
        print("✓ EnglishFluencyAgent evaluated successfully")
        print(f"  Found {len(errors)} error(s)")
        for error in errors:
            print(f"    - {error.description[:80]}...")
    except Exception as e:
        print(f"✗ EnglishFluencyAgent failed: {e}")
        raise


async def test_chinese_fluency_agent() -> None:
    """Test ChineseFluencyAgent."""
    print("\n" + "=" * 60)
    print("Testing ChineseFluencyAgent")
    print("=" * 60)

    # Use mock LLM provider for testing
    llm = MockLLMProvider()

    # Create agent
    agent = ChineseFluencyAgent(llm)

    # Test case with measure word error
    task = TranslationTask(
        source_text="I read a book",
        translation="我看一个书",  # Should be 一本书, not 一个书
        source_lang="en",
        target_lang="zh",
    )

    try:
        errors = await agent.evaluate(task)
        print("✓ ChineseFluencyAgent evaluated successfully")
        print(f"  Found {len(errors)} error(s)")
        for error in errors:
            print(f"    - {error.description[:80]}...")
    except Exception as e:
        print(f"✗ ChineseFluencyAgent failed: {e}")
        raise


async def test_russian_fluency_agent() -> None:
    """Test RussianFluencyAgent."""
    print("\n" + "=" * 60)
    print("Testing RussianFluencyAgent")
    print("=" * 60)

    # Use mock LLM provider for testing
    llm = MockLLMProvider()

    # Create agent
    agent = RussianFluencyAgent(llm)

    # Test case with gender agreement error
    task = TranslationTask(
        source_text="Beautiful house",
        translation="Красивая дом",  # Gender error: feminine adjective + masculine noun
        source_lang="en",
        target_lang="ru",
    )

    try:
        errors = await agent.evaluate(task)
        print("✓ RussianFluencyAgent evaluated successfully")
        print(f"  Found {len(errors)} error(s)")
        for error in errors:
            print(f"    - {error.description[:80]}...")
    except Exception as e:
        print(f"✗ RussianFluencyAgent failed: {e}")
        raise


async def main() -> None:
    """Run all tests."""
    print("\n" + "=" * 60)
    print("Testing Specialized Fluency Agents")
    print("=" * 60)

    try:
        # Test all three agents
        await test_english_fluency_agent()
        await test_chinese_fluency_agent()
        await test_russian_fluency_agent()

        print("\n" + "=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)

    except Exception as e:
        print("\n" + "=" * 60)
        print(f"✗ Tests failed: {e}")
        print("=" * 60)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
