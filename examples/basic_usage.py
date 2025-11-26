"""Basic usage example for KTTC.

This example shows how to:
1. Setup an LLM provider
2. Create a translation task
3. Evaluate translation quality
4. Print the results
"""

import asyncio
import os

from kttc.agents.orchestrator import AgentOrchestrator
from kttc.core.models import TranslationTask
from kttc.llm.openai_provider import OpenAIProvider


async def main() -> None:
    """Basic translation quality check example."""
    # 1. Setup LLM provider
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        return

    llm = OpenAIProvider(api_key=api_key)

    # 2. Create orchestrator
    orchestrator = AgentOrchestrator(llm)

    # 3. Create translation task
    task = TranslationTask(
        source_text="The quick brown fox jumps over the lazy dog.",
        translation="El rápido zorro marrón salta sobre el perro perezoso.",
        source_lang="en",
        target_lang="es",
    )

    # 4. Evaluate translation quality
    print("Evaluating translation quality...")
    report = await orchestrator.evaluate(task)

    # 5. Print results
    print("\n" + "=" * 60)
    print("TRANSLATION QUALITY REPORT")
    print("=" * 60)
    print(f"\nMQM Score: {report.mqm_score:.2f}")
    print(f"Status: {'✅ PASS' if report.status == 'pass' else '❌ FAIL'}")
    print(f"Errors Found: {len(report.errors)}")

    if report.errors:
        print("\nIssues:")
        for i, error in enumerate(report.errors, 1):
            print(f"\n{i}. [{error.severity.name}] {error.category}")
            print(f"   Location: {error.location[0]}-{error.location[1]}")
            print(f"   Description: {error.description}")
            if error.suggestion:
                print(f"   Suggestion: {error.suggestion}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
