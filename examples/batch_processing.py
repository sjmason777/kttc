"""Batch processing example for KTTC.

This example shows how to:
1. Process multiple translation pairs
2. Run evaluations in parallel
3. Generate aggregate statistics
4. Export results to JSON
"""

import asyncio
import json
import os
from pathlib import Path
from typing import Any

from kttc.agents.orchestrator import AgentOrchestrator
from kttc.core.models import TranslationTask
from kttc.llm.openai_provider import OpenAIProvider


async def evaluate_batch(
    tasks: list[TranslationTask], llm_provider: OpenAIProvider, max_parallel: int = 3
) -> list[dict[str, Any]]:
    """Evaluate multiple translation tasks in parallel.

    Args:
        tasks: List of translation tasks to evaluate
        llm_provider: LLM provider instance
        max_parallel: Maximum number of parallel evaluations

    Returns:
        List of evaluation results as dictionaries
    """
    orchestrator = AgentOrchestrator(llm_provider)

    # Use semaphore to limit parallelism
    semaphore = asyncio.Semaphore(max_parallel)

    async def evaluate_with_semaphore(task: TranslationTask) -> dict[str, Any]:
        async with semaphore:
            report = await orchestrator.evaluate(task)
            return {
                "source": (
                    task.source_text[:50] + "..."
                    if len(task.source_text) > 50
                    else task.source_text
                ),
                "translation": (
                    task.translation[:50] + "..."
                    if len(task.translation) > 50
                    else task.translation
                ),
                "mqm_score": report.mqm_score,
                "status": report.status,
                "error_count": len(report.errors),
                "errors": [
                    {
                        "category": err.category,
                        "severity": err.severity.name,
                        "description": err.description,
                    }
                    for err in report.errors
                ],
            }

    results = await asyncio.gather(*[evaluate_with_semaphore(task) for task in tasks])
    return results


def print_summary(results: list[dict[str, Any]]) -> None:
    """Print summary statistics."""
    passed = sum(1 for r in results if r["status"] == "pass")
    failed = sum(1 for r in results if r["status"] == "fail")
    avg_score = sum(r["mqm_score"] for r in results) / len(results)
    total_errors = sum(r["error_count"] for r in results)

    print("\n" + "=" * 60)
    print("BATCH PROCESSING SUMMARY")
    print("=" * 60)
    print(f"Total Translations: {len(results)}")
    print(f"Passed: {passed} ({passed/len(results)*100:.1f}%)")
    print(f"Failed: {failed} ({failed/len(results)*100:.1f}%)")
    print(f"Average MQM Score: {avg_score:.2f}")
    print(f"Total Errors Found: {total_errors}")
    print("=" * 60 + "\n")


async def main() -> None:
    """Batch processing example."""

    # 1. Setup LLM provider
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        return

    llm = OpenAIProvider(api_key=api_key)

    # 2. Create multiple translation tasks
    tasks = [
        TranslationTask(
            source_text="Hello, world!",
            translation="¡Hola, mundo!",
            source_lang="en",
            target_lang="es",
        ),
        TranslationTask(
            source_text="Good morning!",
            translation="¡Buenos días!",
            source_lang="en",
            target_lang="es",
        ),
        TranslationTask(
            source_text="How are you?",
            translation="¿Cómo estás?",
            source_lang="en",
            target_lang="es",
        ),
        TranslationTask(
            source_text="Thank you very much.",
            translation="Muchas gracias.",
            source_lang="en",
            target_lang="es",
        ),
        TranslationTask(
            source_text="I love programming.",
            translation="Me encanta programar.",
            source_lang="en",
            target_lang="es",
        ),
    ]

    # 3. Evaluate all tasks in parallel
    print(f"Evaluating {len(tasks)} translations (max 3 parallel)...")
    results = await evaluate_batch(tasks, llm, max_parallel=3)

    # 4. Print summary
    print_summary(results)

    # 5. Print individual results
    for i, result in enumerate(results, 1):
        status_icon = "✅" if result["status"] == "pass" else "❌"
        print(
            f"{i}. {status_icon} Score: {result['mqm_score']:.1f} | Errors: {result['error_count']}"
        )
        print(f"   Source: {result['source']}")
        print(f"   Translation: {result['translation']}")
        if result["errors"]:
            for err in result["errors"]:
                print(f"   - [{err['severity']}] {err['category']}: {err['description']}")
        print()

    # 6. Export to JSON
    output_file = Path("batch_results.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results exported to: {output_file.absolute()}")


if __name__ == "__main__":
    asyncio.run(main())
