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

"""Generate bad translations with CRITICAL, OBVIOUS errors for testing."""

import asyncio
import json
import random
import sys
from pathlib import Path
from typing import Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from kttc.llm import BaseLLMProvider
from kttc.utils.config import get_settings
from tests.benchmarks.enhanced_dataset_loader import EnhancedDatasetLoader


class CriticalErrorGenerator:
    """Generate translations with CRITICAL, OBVIOUS errors."""

    # More aggressive error types
    CRITICAL_ERROR_TYPES = {
        "complete_mistranslation": {
            "description": "Translate with completely wrong meaning",
            "severity": "critical",
            "mqm_range": [20, 50],
            "prompt_modifier": "Translate with COMPLETELY WRONG meaning. Change the subject, action, or object to something entirely different. Make it clearly incorrect.",
        },
        "omit_key_content": {
            "description": "Skip critical information (50% of content)",
            "severity": "critical",
            "mqm_range": [30, 60],
            "prompt_modifier": "Translate but OMIT at least half of the important information. Skip key phrases, numbers, or entire clauses.",
        },
        "untranslated_parts": {
            "description": "Leave parts in source language",
            "severity": "major",
            "mqm_range": [40, 70],
            "prompt_modifier": "Translate but leave 2-3 key words completely UNTRANSLATED in the source language.",
        },
        "wrong_numbers": {
            "description": "Change numbers drastically",
            "severity": "critical",
            "mqm_range": [25, 55],
            "prompt_modifier": "If there are any numbers, dates, or quantities, change them to COMPLETELY DIFFERENT values (e.g., 100 → 10, 2020 → 2015).",
        },
        "opposite_meaning": {
            "description": "Use opposite/negated meaning",
            "severity": "critical",
            "mqm_range": [15, 45],
            "prompt_modifier": "Translate but REVERSE the meaning. Turn positive statements negative, affirmations into denials, etc.",
        },
        "wrong_terminology": {
            "description": "Use completely incorrect technical terms",
            "severity": "major",
            "mqm_range": [35, 65],
            "prompt_modifier": "If there are technical or domain-specific terms, replace them with WRONG terms from different domains.",
        },
        "mixed_language": {
            "description": "Mix source and target languages",
            "severity": "major",
            "mqm_range": [40, 70],
            "prompt_modifier": "Translate but randomly mix in 3-5 words from the SOURCE language throughout the translation.",
        },
    }

    def __init__(self, llm_provider: BaseLLMProvider):
        """Initialize the generator.

        Args:
            llm_provider: LLM provider for generating translations
        """
        self.llm = llm_provider

    async def generate_critical_bad_translation(
        self,
        source_text: str,
        source_lang: str,
        target_lang: str,
        error_type: str | None = None,
    ) -> dict[str, Any]:
        """Generate a bad translation with CRITICAL errors.

        Args:
            source_text: Source text to translate
            source_lang: Source language code
            target_lang: Target language code
            error_type: Type of error to inject (random if None)

        Returns:
            Dictionary with bad translation and metadata
        """
        # Select error type
        if error_type is None or error_type not in self.CRITICAL_ERROR_TYPES:
            error_type = random.choice(list(self.CRITICAL_ERROR_TYPES.keys()))

        error_info = self.CRITICAL_ERROR_TYPES[error_type]

        # Create prompt for generating CRITICAL bad translation
        prompt = f"""You are generating a DELIBERATELY BAD translation with CRITICAL, OBVIOUS errors for testing translation quality systems.

Source text (in {source_lang}):
{source_text}

Task: Translate this to {target_lang}, but {error_info['prompt_modifier']}

IMPORTANT:
- The error should be OBVIOUS and CRITICAL
- Make it clearly wrong, not subtle
- The translation should be noticeably bad
- Return ONLY the bad translation, no explanations
- Make sure the error is severe enough to be caught

Bad translation:"""

        # Generate bad translation
        bad_translation = await self.llm.complete(
            prompt,
            temperature=0.9,  # Higher temperature for more variation
            max_tokens=500,
        )

        return {
            "source": source_text,
            "bad_translation": bad_translation.strip(),
            "source_lang": source_lang,
            "target_lang": target_lang,
            "injected_error_type": error_type,
            "injected_error_severity": error_info["severity"],
            "injected_error_description": error_info["description"],
            "expected_mqm_range": error_info["mqm_range"],
        }

    async def process_dataset_critical(
        self,
        good_samples: list[dict[str, Any]],
        output_file: str | Path,
        bad_ratio: float = 0.5,  # Higher ratio for testing
        max_samples: int | None = None,
    ) -> None:
        """Process a dataset and generate CRITICAL bad translations.

        Args:
            good_samples: List of good translation samples
            output_file: Output file path
            bad_ratio: Ratio of bad translations to generate (0.0-1.0)
            max_samples: Maximum number of samples to process
        """
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Limit samples if specified
        if max_samples:
            good_samples = good_samples[:max_samples]

        # Calculate how many bad translations to generate
        num_bad = int(len(good_samples) * bad_ratio)

        print(f"Processing {len(good_samples)} samples...")
        print(f"Generating {num_bad} CRITICAL bad translations...")

        # Select random samples for bad translation
        samples_to_corrupt = random.sample(good_samples, min(num_bad, len(good_samples)))

        results = []
        for idx, sample in enumerate(good_samples):
            print(f"\rProgress: {idx + 1}/{len(good_samples)}", end="", flush=True)

            if sample in samples_to_corrupt:
                # Generate CRITICAL bad translation
                bad_version = await self.generate_critical_bad_translation(
                    sample["source"],
                    sample["source_lang"],
                    sample["target_lang"],
                )

                results.append(
                    {
                        **sample,
                        "translation": bad_version["bad_translation"],
                        "quality": "bad",
                        "error_type": bad_version["injected_error_type"],
                        "error_severity": bad_version["injected_error_severity"],
                        "expected_mqm_range": bad_version["expected_mqm_range"],
                        "error_description": bad_version["injected_error_description"],
                    }
                )
            else:
                # Keep good translation
                results.append({**sample, "quality": "good", "expected_mqm_range": [95, 100]})

        print()  # New line after progress

        # Save results
        output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n✅ Saved {len(results)} samples to {output_path}")
        print(f"   Good: {len(results) - num_bad}, Critical Bad: {num_bad}")


async def main() -> None:
    """Main function for generating CRITICAL bad translations."""
    print("\n🔧 KTTC CRITICAL Bad Translation Generator\n")
    print("=" * 80)
    print("Generating OBVIOUS, CRITICAL errors for better detection testing")
    print("=" * 80 + "\n")

    # Setup LLM provider
    settings = get_settings()
    llm = None

    # Try OpenAI first
    try:
        api_key = settings.get_llm_provider_key("openai")
        from kttc.llm import OpenAIProvider

        llm = OpenAIProvider(api_key=api_key, model="gpt-4o-mini")
        print("✅ Using OpenAI (gpt-4o-mini) for generation")
    except Exception:
        # Silently ignore OpenAI setup errors and try Anthropic instead
        pass

    # Fall back to Anthropic
    if llm is None:
        try:
            api_key = settings.get_llm_provider_key("anthropic")
            from kttc.llm import AnthropicProvider

            llm = AnthropicProvider(api_key=api_key, model="claude-3-5-haiku-20241022")
            print("✅ Using Anthropic (claude-3-5-haiku) for generation")
        except Exception:
            # Silently ignore Anthropic setup errors and continue with error message
            pass

    if llm is None:
        print("❌ Error: Could not initialize LLM provider")
        print("   Make sure KTTC_OPENAI_API_KEY or KTTC_ANTHROPIC_API_KEY is set in .env")
        return

    generator = CriticalErrorGenerator(llm)
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

    print("\n" + "=" * 80)
    print("GENERATING CRITICAL BAD TRANSLATIONS FOR ALL LANGUAGE PAIRS")
    print("=" * 80 + "\n")

    for idx, (src_lang, tgt_lang) in enumerate(language_pairs, 1):
        print(f"[{idx}/{len(language_pairs)}] Generating CRITICAL bad: {src_lang} → {tgt_lang}")
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

            # Generate CRITICAL bad translations
            output_file = loader.data_dir / f"critical_bad_{src_lang}_{tgt_lang}.json"
            print("  🔧 Generating CRITICAL bad translations (50% bad ratio)...")

            await generator.process_dataset_critical(
                good_samples,
                output_file,
                bad_ratio=0.5,  # 50% will be CRITICAL bad
                max_samples=10,
            )

            print(f"  ✅ Saved to: {output_file.name}\n")

        except Exception as e:
            print(f"  ❌ Error: {e}\n")
            import traceback

            traceback.print_exc()

    print("=" * 80)
    print("🎉 ALL CRITICAL BAD TRANSLATIONS GENERATED!")
    print("=" * 80)
    print(f"\nGenerated critical bad translations for {len(language_pairs)} language pairs")
    print("Data saved to: tests/benchmarks/data/critical_bad_*.json")
    print("\nError types included:")
    for error_type, info in CriticalErrorGenerator.CRITICAL_ERROR_TYPES.items():
        print(
            f"  - {error_type}: {info['description']} (MQM {info['mqm_range'][0]}-{info['mqm_range'][1]})"
        )
    print("\nNext step:")
    print("  python3.11 scripts/run_comprehensive_benchmark.py")


if __name__ == "__main__":
    asyncio.run(main())
