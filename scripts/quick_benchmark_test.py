#!/usr/bin/env python3
# Copyright 2025 KTTC AI (https://github.com/kttc-ai)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Quick benchmark test - runs a small benchmark to verify system works."""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from kttc.agents.orchestrator import AgentOrchestrator
from kttc.llm import OpenAIProvider
from kttc.utils.config import get_settings
from tests.benchmarks.enhanced_dataset_loader import EnhancedDatasetLoader


async def quick_test() -> None:
    """Run a quick benchmark test."""
    print("\n" + "=" * 80)
    print("KTTC QUICK BENCHMARK TEST")
    print("=" * 80)
    print("\nThis is a quick test to verify the benchmark system works.\n")

    # Setup
    print("📋 Setting up...")
    settings = get_settings()
    llm = None

    # Try OpenAI first
    try:
        api_key = settings.get_llm_provider_key("openai")
        llm = OpenAIProvider(api_key=api_key, model="gpt-4o-mini")
        print("✅ LLM provider ready (OpenAI gpt-4o-mini)")
    except Exception:
        # Silently ignore Anthropic setup errors and continue with error message
        pass

    # Fall back to Anthropic
    if llm is None:
        try:
            from kttc.llm import AnthropicProvider

            api_key = settings.get_llm_provider_key("anthropic")
            llm = AnthropicProvider(api_key=api_key, model="claude-3-5-haiku-20241022")
            print("✅ LLM provider ready (Anthropic claude-3-5-haiku)")
        except Exception:
            # Silently ignore OpenAI setup errors and try Anthropic instead
            pass

    if llm is None:
        print("❌ Error: Could not initialize LLM provider")
        print("   Make sure KTTC_OPENAI_API_KEY or KTTC_ANTHROPIC_API_KEY is set in .env")
        return

    orchestrator = AgentOrchestrator(
        llm,
        quality_threshold=95.0,
        agent_temperature=settings.default_temperature,
        agent_max_tokens=settings.default_max_tokens,
    )

    # Load sample data
    print("\n📥 Loading test data...")
    loader = EnhancedDatasetLoader()
    samples = await loader.load_flores200("en", "ru", split="devtest", sample_size=5)
    print(f"✅ Loaded {len(samples)} samples")

    # Run evaluation
    print("\n🔍 Running evaluation...")

    from kttc.core.models import TranslationTask

    results = []
    for idx, sample in enumerate(samples, 1):
        print(f"  [{idx}/{len(samples)}] Evaluating...", end="\r")

        task = TranslationTask(
            source_text=sample["source"],
            translation=sample["translation"],
            source_lang=sample["source_lang"],
            target_lang=sample["target_lang"],
        )

        # Retry logic for API errors
        max_retries = 3
        for attempt in range(max_retries):
            try:
                report = await orchestrator.evaluate(task)
                results.append(
                    {
                        "id": sample["id"],
                        "mqm_score": report.mqm_score,
                        "status": report.status,
                        "errors": len(report.errors),
                    }
                )
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"\n  ⚠️  Retry {attempt + 1}/{max_retries} due to: {str(e)[:50]}...")
                    await asyncio.sleep(2**attempt)  # Exponential backoff
                else:
                    print(f"\n  ❌ Failed after {max_retries} retries: {str(e)[:100]}")
                    raise

    print()  # New line after progress

    # Display results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80 + "\n")

    avg_mqm = sum(r["mqm_score"] for r in results) / len(results)
    pass_count = sum(1 for r in results if r["status"] == "pass")

    print("📊 Summary:")
    print(f"   Samples: {len(results)}")
    print(f"   Average MQM: {avg_mqm:.2f}")
    print(f"   Pass Rate: {pass_count}/{len(results)} ({pass_count/len(results)*100:.1f}%)")
    print()

    print("📋 Individual Results:")
    for r in results:
        status_icon = "✅" if r["status"] == "pass" else "❌"
        print(f"   {status_icon} {r['id']}: MQM={r['mqm_score']:.2f}, Errors={r['errors']}")

    print("\n" + "=" * 80)
    print("✅ QUICK TEST COMPLETE!")
    print("=" * 80)
    print("\nNext steps:")
    print("  1. Run full benchmark: python3.11 scripts/run_comprehensive_benchmark.py")
    print("  2. Generate bad translations: python3.11 scripts/generate_bad_translations.py")
    print("  3. Install datasets library for real FLORES-200: pip install datasets")


if __name__ == "__main__":
    asyncio.run(quick_test())
