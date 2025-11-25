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

"""Generate bad translations for all language pairs."""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from kttc.utils.config import get_settings
from scripts.generate_bad_translations import BadTranslationGenerator
from tests.benchmarks.enhanced_dataset_loader import EnhancedDatasetLoader


async def generate_all_bad_translations() -> None:
    """Generate bad translations for all language pairs."""
    print("\n" + "=" * 80)
    print("GENERATING BAD TRANSLATIONS FOR ALL LANGUAGE PAIRS")
    print("=" * 80)
    print()

    # Setup LLM provider
    settings = get_settings()
    llm = None

    # Try OpenAI first
    try:
        api_key = settings.get_llm_provider_key("openai")
        from kttc.llm import OpenAIProvider

        llm = OpenAIProvider(api_key=api_key, model="gpt-4o-mini")
        print("✅ Using OpenAI (gpt-4o-mini) for generation\n")
    except Exception:
        # Silently ignore Anthropic setup errors and continue with error message
        pass

    # Fall back to Anthropic
    if llm is None:
        try:
            api_key = settings.get_llm_provider_key("anthropic")
            from kttc.llm import AnthropicProvider

            llm = AnthropicProvider(api_key=api_key, model="claude-3-5-haiku-20241022")
            print("✅ Using Anthropic (claude-3-5-haiku) for generation\n")
        except Exception:
            # Silently ignore OpenAI setup errors and try Anthropic instead
            pass

    if llm is None:
        print("❌ Error: Could not initialize LLM provider")
        print("   Make sure KTTC_OPENAI_API_KEY or KTTC_ANTHROPIC_API_KEY is set in .env")
        return

    generator = BadTranslationGenerator(llm)
    loader = EnhancedDatasetLoader()

    # Language pairs to process
    language_pairs = [
        ("en", "ru"),
        ("en", "zh"),
        ("ru", "en"),
        ("zh", "en"),
        ("ru", "zh"),
        ("zh", "ru"),
    ]

    for idx, (src_lang, tgt_lang) in enumerate(language_pairs, 1):
        print(f"[{idx}/{len(language_pairs)}] Generating bad translations: {src_lang} → {tgt_lang}")
        print("-" * 80)

        try:
            # Load good samples
            good_samples = await loader.load_flores200(
                src_lang, tgt_lang, split="devtest", sample_size=10
            )

            if not good_samples:
                print(f"  ⚠️  No samples found for {src_lang}-{tgt_lang}, skipping\n")
                continue

            print(f"  ✓ Loaded {len(good_samples)} good samples")

            # Generate bad translations
            output_file = loader.data_dir / f"synthetic_bad_{src_lang}_{tgt_lang}.json"
            print("  🔧 Generating bad translations (30% bad ratio)...")

            await generator.process_dataset(
                good_samples,
                output_file,
                bad_ratio=0.3,  # 30% will be bad
                max_samples=10,  # Process all samples
            )

            print(f"  ✅ Saved to: {output_file.name}\n")

        except Exception as e:
            print(f"  ❌ Error: {e}\n")
            import traceback

            traceback.print_exc()

    print("=" * 80)
    print("🎉 ALL BAD TRANSLATIONS GENERATED!")
    print("=" * 80)
    print(f"\nGenerated bad translations for {len(language_pairs)} language pairs")
    print("Data saved to: tests/benchmarks/data/synthetic_bad_*.json")
    print("\nNext step:")
    print("  python3.11 scripts/run_comprehensive_benchmark.py")


if __name__ == "__main__":
    asyncio.run(generate_all_bad_translations())
