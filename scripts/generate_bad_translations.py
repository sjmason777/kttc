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

"""Generate intentionally bad translations for benchmark testing."""

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


class BadTranslationGenerator:
    """Generate translations with intentional errors for testing."""

    ERROR_TYPES = {
        "mistranslation": {
            "description": "Translate key terms incorrectly",
            "severity": "major",
            "prompt_modifier": "Intentionally mistranslate 1-2 important words",
        },
        "omission": {
            "description": "Skip translating some parts",
            "severity": "major",
            "prompt_modifier": "Skip translating one sentence or phrase",
        },
        "addition": {
            "description": "Add extra content not in source",
            "severity": "minor",
            "prompt_modifier": "Add 1-2 extra words that weren't in the original",
        },
        "grammar": {
            "description": "Use incorrect grammar",
            "severity": "minor",
            "prompt_modifier": "Use incorrect verb tenses or grammar",
        },
        "terminology": {
            "description": "Use wrong technical terms",
            "severity": "major",
            "prompt_modifier": "Use incorrect technical terminology",
        },
        "style": {
            "description": "Use inappropriate register/tone",
            "severity": "minor",
            "prompt_modifier": "Use overly formal or informal language inappropriately",
        },
        "word_order": {
            "description": "Awkward word order",
            "severity": "minor",
            "prompt_modifier": "Use awkward or unnatural word order",
        },
    }

    def __init__(self, llm_provider: BaseLLMProvider):
        """Initialize the generator.

        Args:
            llm_provider: LLM provider for generating translations
        """
        self.llm = llm_provider

    async def generate_bad_translation(
        self,
        source_text: str,
        source_lang: str,
        target_lang: str,
        error_type: str | None = None,
    ) -> dict[str, Any]:
        """Generate a bad translation with specific error types.

        Args:
            source_text: Source text to translate
            source_lang: Source language code
            target_lang: Target language code
            error_type: Type of error to inject (random if None)

        Returns:
            Dictionary with bad translation and metadata
        """
        # Select error type
        if error_type is None:
            error_type = random.choice(list(self.ERROR_TYPES.keys()))

        error_info = self.ERROR_TYPES[error_type]

        # Create prompt for generating bad translation
        prompt = f"""You are generating a DELIBERATELY BAD translation for testing translation quality assessment systems.

Source text (in {source_lang}):
{source_text}

Task: Translate this to {target_lang}, but {error_info['prompt_modifier']}.

Important:
- The translation should look plausible but contain the specified error
- Don't be too obvious about the error
- Return ONLY the bad translation, no explanations

Bad translation:"""

        # Generate bad translation
        bad_translation = await self.llm.complete(
            prompt,
            temperature=0.8,  # Higher temperature for more variation
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
        }

    async def generate_multiple_bad_versions(
        self,
        source_text: str,
        source_lang: str,
        target_lang: str,
        num_versions: int = 3,
    ) -> list[dict[str, Any]]:
        """Generate multiple bad translation versions with different errors.

        Args:
            source_text: Source text
            source_lang: Source language code
            target_lang: Target language code
            num_versions: Number of bad versions to generate

        Returns:
            List of bad translations with different error types
        """
        # Select different error types
        error_types = random.sample(
            list(self.ERROR_TYPES.keys()), min(num_versions, len(self.ERROR_TYPES))
        )

        bad_versions = []
        for error_type in error_types:
            bad_version = await self.generate_bad_translation(
                source_text, source_lang, target_lang, error_type
            )
            bad_versions.append(bad_version)

        return bad_versions

    async def process_dataset(
        self,
        good_samples: list[dict[str, Any]],
        output_file: str | Path,
        bad_ratio: float = 0.3,
        max_samples: int | None = None,
    ) -> None:
        """Process a dataset and generate bad translations.

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
        print(f"Generating {num_bad} bad translations...")

        # Select random samples for bad translation
        samples_to_corrupt = random.sample(good_samples, num_bad)

        results = []
        for idx, sample in enumerate(good_samples):
            print(f"\rProgress: {idx + 1}/{len(good_samples)}", end="", flush=True)

            if sample in samples_to_corrupt:
                # Generate bad translation
                bad_version = await self.generate_bad_translation(
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
                        "expected_mqm_range": [40, 75],  # Expected low MQM score
                    }
                )
            else:
                # Keep good translation
                results.append({**sample, "quality": "good", "expected_mqm_range": [95, 100]})

        print()  # New line after progress

        # Save results
        output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n✅ Saved {len(results)} samples to {output_path}")
        print(f"   Good: {len(results) - num_bad}, Bad: {num_bad}")


async def main() -> None:
    """Main function for generating bad translations."""
    print("\n🔧 KTTC Bad Translation Generator\n")
    print("=" * 80)

    # Setup LLM provider
    settings = get_settings()
    llm = None

    # Try OpenAI first
    try:
        api_key = settings.get_llm_provider_key("openai")
        from kttc.llm import OpenAIProvider

        llm = OpenAIProvider(api_key=api_key, model="gpt-4o-mini")  # Use cheaper model
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

    generator = BadTranslationGenerator(llm)

    # Example: Generate a few bad translations
    print("\n" + "=" * 80)
    print("EXAMPLE: Generating bad translations")
    print("=" * 80 + "\n")

    test_sentences = [
        {
            "source": "Artificial intelligence is transforming the world of technology.",
            "source_lang": "en",
            "target_lang": "ru",
        },
        {
            "source": "Machine translation quality assessment is crucial for production systems.",
            "source_lang": "en",
            "target_lang": "zh",
        },
    ]

    for test in test_sentences:
        print(f"\nSource ({test['source_lang']}): {test['source']}")
        print(f"Target language: {test['target_lang']}\n")

        bad_versions = await generator.generate_multiple_bad_versions(
            test["source"], test["source_lang"], test["target_lang"], num_versions=2
        )

        for idx, bad in enumerate(bad_versions, 1):
            print(f"  Bad version {idx} ({bad['injected_error_type']}):")
            print(f"    {bad['bad_translation']}")

    print("\n" + "=" * 80)
    print("✅ Generation complete!")
    print("=" * 80)
    print("\nTo process a full dataset, use:")
    print("  from this script import BadTranslationGenerator")
    print("  generator.process_dataset(samples, 'output.json', bad_ratio=0.3)")


if __name__ == "__main__":
    asyncio.run(main())
