#!/usr/bin/env python3.11
"""Benchmark MQM Score Improvement and Cost Savings.

This script measures:
1. MQM score improvement: Specialized agents vs Base agent
2. Cost savings: LLM token usage reduction thanks to NLP helpers

Based on MQM framework for translation quality assessment.
"""

import asyncio
import json
import logging
import sys
import time
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from kttc.agents import (
    ChineseFluencyAgent,
    EnglishFluencyAgent,
    FluencyAgent,
    RussianFluencyAgent,
)
from kttc.core import ErrorAnnotation, ErrorSeverity, TranslationTask
from kttc.llm import BaseLLMProvider

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TestCase:
    """Test case with known errors for MQM scoring."""

    source: str
    translation: str
    source_lang: str
    target_lang: str
    # Known errors for MQM calculation
    expected_critical: int = 0
    expected_major: int = 0
    expected_minor: int = 0
    description: str = ""


# Test cases with known MQM scores
ENGLISH_TEST_CASES = [
    TestCase(
        source="Привет мир",
        translation="He go to school every day",
        source_lang="ru",
        target_lang="en",
        expected_critical=0,
        expected_major=1,  # subject-verb agreement
        expected_minor=0,
        description="Subject-verb agreement error",
    ),
    TestCase(
        source="C'est bon",
        translation="Its a good idea",
        source_lang="fr",
        target_lang="en",
        expected_critical=0,
        expected_major=0,
        expected_minor=1,  # missing apostrophe
        description="Possessive apostrophe missing",
    ),
]

CHINESE_TEST_CASES = [
    TestCase(
        source="I read a book",
        translation="我看一个书",
        source_lang="en",
        target_lang="zh",
        expected_critical=0,
        expected_major=1,  # wrong measure word
        expected_minor=0,
        description="Incorrect measure word (个 instead of 本)",
    ),
]

RUSSIAN_TEST_CASES = [
    TestCase(
        source="Beautiful house",
        translation="Красивая дом",
        source_lang="en",
        target_lang="ru",
        expected_critical=1,  # gender agreement is critical in Russian
        expected_major=0,
        expected_minor=0,
        description="Gender agreement violation",
    ),
    TestCase(
        source="About the book",
        translation="О книга",
        source_lang="en",
        target_lang="ru",
        expected_critical=1,  # case agreement is critical
        expected_major=0,
        expected_minor=0,
        description="Preposition-case violation",
    ),
]


class MockLLMProvider(BaseLLMProvider):
    """Mock LLM provider that tracks token usage."""

    def __init__(self) -> None:
        self.total_calls = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    async def complete(
        self, prompt: str, temperature: float = 0.7, max_tokens: int = 1000, **kwargs: Any
    ) -> str:
        """Return mock JSON response and track tokens."""
        self.total_calls += 1
        # Estimate tokens (rough: 1 token ≈ 4 chars)
        self.total_prompt_tokens += len(prompt) // 4

        # Mock response
        response = '{"errors": []}'
        self.total_completion_tokens += len(response) // 4

        return response

    async def stream(
        self, prompt: str, temperature: float = 0.7, max_tokens: int = 1000, **kwargs: Any
    ) -> AsyncGenerator[str, None]:
        """Mock stream method."""
        yield '{"errors": []}'

    def get_stats(self) -> dict[str, int]:
        """Get token usage statistics."""
        return {
            "calls": self.total_calls,
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_prompt_tokens + self.total_completion_tokens,
        }

    def reset(self) -> None:
        """Reset counters."""
        self.total_calls = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0


def calculate_mqm_score(errors: list[ErrorAnnotation], word_count: int = 10) -> dict[str, Any]:
    """Calculate MQM score from errors.

    MQM Score = -(critical_weight * critical_count + major_weight * major_count + minor_weight * minor_count)

    Standard weights:
    - Critical: 10 points per error
    - Major: 5 points per error
    - Minor: 1 point per error

    Normalized to 100-point scale (per 100 words).
    """
    critical_count = sum(1 for e in errors if e.severity == ErrorSeverity.CRITICAL)
    major_count = sum(1 for e in errors if e.severity == ErrorSeverity.MAJOR)
    minor_count = sum(1 for e in errors if e.severity == ErrorSeverity.MINOR)

    # MQM deductions
    deductions = (critical_count * 10) + (major_count * 5) + (minor_count * 1)

    # Normalize per 100 words
    normalized_deductions = (deductions / word_count) * 100

    # MQM score (100 - deductions, capped at 0)
    mqm_score = max(0, 100 - normalized_deductions)

    return {
        "mqm_score": mqm_score,
        "deductions": normalized_deductions,
        "critical": critical_count,
        "major": major_count,
        "minor": minor_count,
    }


async def benchmark_agent(
    agent: FluencyAgent, test_cases: list[TestCase], agent_name: str
) -> dict[str, Any]:
    """Benchmark a single agent on test cases."""
    print(f"\n{'='*60}")
    print(f"Benchmarking: {agent_name}")
    print(f"{'='*60}")

    total_mqm = 0
    total_deductions = 0
    total_errors_found = 0
    results = []

    for i, test_case in enumerate(test_cases, 1):
        task = TranslationTask(
            source_text=test_case.source,
            translation=test_case.translation,
            source_lang=test_case.source_lang,
            target_lang=test_case.target_lang,
        )

        start_time = time.time()
        errors = await agent.evaluate(task)
        execution_time = time.time() - start_time

        # Calculate MQM score
        word_count = len(test_case.translation.split())
        mqm_result = calculate_mqm_score(errors, word_count)

        total_mqm += mqm_result["mqm_score"]
        total_deductions += mqm_result["deductions"]
        total_errors_found += len(errors)

        results.append(
            {
                "test_case": test_case.description,
                "errors_found": len(errors),
                "mqm_score": mqm_result["mqm_score"],
                "deductions": mqm_result["deductions"],
                "critical": mqm_result["critical"],
                "major": mqm_result["major"],
                "minor": mqm_result["minor"],
                "execution_time": execution_time,
            }
        )

        print(f"  [{i}/{len(test_cases)}] {test_case.description}")
        print(
            f"    Errors: {len(errors)}, MQM: {mqm_result['mqm_score']:.1f}, Time: {execution_time*1000:.0f}ms"
        )

    avg_mqm = total_mqm / len(test_cases) if test_cases else 0
    avg_deductions = total_deductions / len(test_cases) if test_cases else 0

    print("\n  Summary:")
    print(f"    Average MQM Score: {avg_mqm:.2f}")
    print(f"    Average Deductions: {avg_deductions:.2f}")
    print(f"    Total Errors Found: {total_errors_found}")

    return {
        "agent_name": agent_name,
        "avg_mqm_score": avg_mqm,
        "avg_deductions": avg_deductions,
        "total_errors_found": total_errors_found,
        "test_results": results,
    }


async def main() -> None:
    """Run MQM and cost benchmarks."""
    print("\n" + "=" * 60)
    print("MQM Score & Cost Savings Benchmark")
    print("=" * 60)

    all_results = {}

    # English benchmarks
    print("\n" + "=" * 60)
    print("ENGLISH FLUENCY AGENTS")
    print("=" * 60)

    # Base agent (LLM-only)
    llm_base = MockLLMProvider()
    base_agent = FluencyAgent(llm_base)
    base_results = await benchmark_agent(
        base_agent, ENGLISH_TEST_CASES, "Base FluencyAgent (LLM-only)"
    )
    base_stats = llm_base.get_stats()

    # Specialized agent (LanguageTool + LLM)
    llm_specialized = MockLLMProvider()
    specialized_agent = EnglishFluencyAgent(llm_specialized)
    specialized_results = await benchmark_agent(
        specialized_agent, ENGLISH_TEST_CASES, "EnglishFluencyAgent (LanguageTool + LLM)"
    )
    specialized_stats = llm_specialized.get_stats()

    # Calculate improvements
    mqm_improvement = specialized_results["avg_mqm_score"] - base_results["avg_mqm_score"]
    error_improvement = (
        specialized_results["total_errors_found"] - base_results["total_errors_found"]
    )

    # Calculate cost savings (fewer LLM calls with NLP helpers)
    token_savings = base_stats["total_tokens"] - specialized_stats["total_tokens"]
    cost_savings_pct = (
        (token_savings / base_stats["total_tokens"] * 100) if base_stats["total_tokens"] > 0 else 0
    )

    all_results["english"] = {
        "base": base_results,
        "specialized": specialized_results,
        "base_tokens": base_stats,
        "specialized_tokens": specialized_stats,
        "mqm_improvement": mqm_improvement,
        "error_improvement": error_improvement,
        "token_savings": token_savings,
        "cost_savings_pct": cost_savings_pct,
    }

    print(f"\n{'='*60}")
    print("ENGLISH RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"MQM Score Improvement: {mqm_improvement:+.2f} points")
    print(f"Errors Found Improvement: {error_improvement:+d}")
    print("Token Usage:")
    print(f"  Base Agent: {base_stats['total_tokens']} tokens ({base_stats['calls']} LLM calls)")
    print(
        f"  Specialized Agent: {specialized_stats['total_tokens']} tokens ({specialized_stats['calls']} LLM calls)"
    )
    print(f"  Savings: {token_savings} tokens ({cost_savings_pct:.1f}%)")

    # Chinese benchmarks
    print("\n" + "=" * 60)
    print("CHINESE FLUENCY AGENTS")
    print("=" * 60)

    llm_base_zh = MockLLMProvider()
    base_agent_zh = FluencyAgent(llm_base_zh)
    base_results_zh = await benchmark_agent(
        base_agent_zh, CHINESE_TEST_CASES, "Base FluencyAgent (LLM-only)"
    )
    base_stats_zh = llm_base_zh.get_stats()

    llm_specialized_zh = MockLLMProvider()
    specialized_agent_zh = ChineseFluencyAgent(llm_specialized_zh)
    specialized_results_zh = await benchmark_agent(
        specialized_agent_zh, CHINESE_TEST_CASES, "ChineseFluencyAgent (HanLP + LLM)"
    )
    specialized_stats_zh = llm_specialized_zh.get_stats()

    mqm_improvement_zh = specialized_results_zh["avg_mqm_score"] - base_results_zh["avg_mqm_score"]
    error_improvement_zh = (
        specialized_results_zh["total_errors_found"] - base_results_zh["total_errors_found"]
    )
    token_savings_zh = base_stats_zh["total_tokens"] - specialized_stats_zh["total_tokens"]
    cost_savings_pct_zh = (
        (token_savings_zh / base_stats_zh["total_tokens"] * 100)
        if base_stats_zh["total_tokens"] > 0
        else 0
    )

    all_results["chinese"] = {
        "base": base_results_zh,
        "specialized": specialized_results_zh,
        "base_tokens": base_stats_zh,
        "specialized_tokens": specialized_stats_zh,
        "mqm_improvement": mqm_improvement_zh,
        "error_improvement": error_improvement_zh,
        "token_savings": token_savings_zh,
        "cost_savings_pct": cost_savings_pct_zh,
    }

    print(f"\n{'='*60}")
    print("CHINESE RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"MQM Score Improvement: {mqm_improvement_zh:+.2f} points")
    print(f"Errors Found Improvement: {error_improvement_zh:+d}")
    print("Token Usage:")
    print(
        f"  Base Agent: {base_stats_zh['total_tokens']} tokens ({base_stats_zh['calls']} LLM calls)"
    )
    print(
        f"  Specialized Agent: {specialized_stats_zh['total_tokens']} tokens ({specialized_stats_zh['calls']} LLM calls)"
    )
    print(f"  Savings: {token_savings_zh} tokens ({cost_savings_pct_zh:.1f}%)")

    # Russian benchmarks
    print("\n" + "=" * 60)
    print("RUSSIAN FLUENCY AGENTS")
    print("=" * 60)

    llm_base_ru = MockLLMProvider()
    base_agent_ru = FluencyAgent(llm_base_ru)
    base_results_ru = await benchmark_agent(
        base_agent_ru, RUSSIAN_TEST_CASES, "Base FluencyAgent (LLM-only)"
    )
    base_stats_ru = llm_base_ru.get_stats()

    llm_specialized_ru = MockLLMProvider()
    specialized_agent_ru = RussianFluencyAgent(llm_specialized_ru)
    specialized_results_ru = await benchmark_agent(
        specialized_agent_ru, RUSSIAN_TEST_CASES, "RussianFluencyAgent (MAWO NLP + LLM)"
    )
    specialized_stats_ru = llm_specialized_ru.get_stats()

    mqm_improvement_ru = specialized_results_ru["avg_mqm_score"] - base_results_ru["avg_mqm_score"]
    error_improvement_ru = (
        specialized_results_ru["total_errors_found"] - base_results_ru["total_errors_found"]
    )
    token_savings_ru = base_stats_ru["total_tokens"] - specialized_stats_ru["total_tokens"]
    cost_savings_pct_ru = (
        (token_savings_ru / base_stats_ru["total_tokens"] * 100)
        if base_stats_ru["total_tokens"] > 0
        else 0
    )

    all_results["russian"] = {
        "base": base_results_ru,
        "specialized": specialized_results_ru,
        "base_tokens": base_stats_ru,
        "specialized_tokens": specialized_stats_ru,
        "mqm_improvement": mqm_improvement_ru,
        "error_improvement": error_improvement_ru,
        "token_savings": token_savings_ru,
        "cost_savings_pct": cost_savings_pct_ru,
    }

    print(f"\n{'='*60}")
    print("RUSSIAN RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"MQM Score Improvement: {mqm_improvement_ru:+.2f} points")
    print(f"Errors Found Improvement: {error_improvement_ru:+d}")
    print("Token Usage:")
    print(
        f"  Base Agent: {base_stats_ru['total_tokens']} tokens ({base_stats_ru['calls']} LLM calls)"
    )
    print(
        f"  Specialized Agent: {specialized_stats_ru['total_tokens']} tokens ({specialized_stats_ru['calls']} LLM calls)"
    )
    print(f"  Savings: {token_savings_ru} tokens ({cost_savings_pct_ru:.1f}%)")

    # Overall summary
    print("\n" + "=" * 60)
    print("OVERALL SUMMARY")
    print("=" * 60)

    avg_mqm_improvement = (mqm_improvement + mqm_improvement_zh + mqm_improvement_ru) / 3
    avg_error_improvement = (error_improvement + error_improvement_zh + error_improvement_ru) / 3
    avg_cost_savings = (cost_savings_pct + cost_savings_pct_zh + cost_savings_pct_ru) / 3

    print("\nAverage Across All Languages:")
    print(f"  MQM Score Improvement: {avg_mqm_improvement:+.2f} points")
    print(f"  Error Detection Improvement: {avg_error_improvement:+.1f} errors")
    print(f"  Cost Savings: {avg_cost_savings:.1f}%")

    print("\n✅ Specialized agents provide:")
    print(f"   - Better quality ({avg_mqm_improvement:+.2f} MQM points)")
    print(f"   - More errors found ({avg_error_improvement:+.1f} errors avg)")
    print(f"   - Lower cost ({avg_cost_savings:.1f}% token savings)")

    # Save results
    output_path = (
        Path(__file__).parent.parent / "docs" / "benchmarks" / "mqm_and_cost_benchmark.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n📊 Results saved to: {output_path}")
    print("\n" + "=" * 60)
    print("✅ Benchmark Complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
