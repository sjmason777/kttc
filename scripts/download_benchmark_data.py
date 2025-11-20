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

"""Download and cache benchmark datasets (FLORES-200, WMT-MQM)."""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.benchmarks.enhanced_dataset_loader import EnhancedDatasetLoader


async def download_flores200() -> None:
    """Download FLORES-200 dataset for all language pairs."""
    print("=" * 80)
    print("DOWNLOADING FLORES-200 BENCHMARK DATA")
    print("=" * 80)
    print()

    loader = EnhancedDatasetLoader()

    # Language pairs to download
    language_pairs = [
        ("en", "ru"),  # English → Russian
        ("en", "zh"),  # English → Chinese
        ("ru", "en"),  # Russian → English
        ("zh", "en"),  # Chinese → English
        ("ru", "zh"),  # Russian → Chinese
        ("zh", "ru"),  # Chinese → Russian
    ]

    splits = ["dev", "devtest"]

    for src_lang, tgt_lang in language_pairs:
        for split in splits:
            print(f"\n📥 Downloading FLORES-200: {src_lang}-{tgt_lang} ({split})...")
            try:
                samples = await loader.load_flores200(src_lang, tgt_lang, split=split)

                # Save to cache
                cache_filename = f"flores200_{src_lang}_{tgt_lang}_{split}.json"
                await loader.save_to_cache(samples, cache_filename)

                print(f"✅ Downloaded {len(samples)} samples for {src_lang}-{tgt_lang} ({split})")

            except Exception as e:
                print(f"❌ Error downloading {src_lang}-{tgt_lang} ({split}): {e}")
                print("   Will use fallback data for this language pair")

    print("\n" + "=" * 80)
    print("✅ FLORES-200 download complete!")
    print("=" * 80)


async def download_wmt_mqm() -> None:
    """Download WMT-MQM dataset with error annotations."""
    print("\n" + "=" * 80)
    print("DOWNLOADING WMT-MQM ERROR ANNOTATION DATA")
    print("=" * 80)
    print()

    loader = EnhancedDatasetLoader()

    # WMT-MQM available language pairs
    language_pairs = [
        ("en", "de"),  # English → German
        ("zh", "en"),  # Chinese → English
    ]

    for src_lang, tgt_lang in language_pairs:
        print(f"\n📥 Downloading WMT-MQM: {src_lang}-{tgt_lang}...")
        try:
            samples = await loader.load_wmt_mqm(src_lang, tgt_lang, sample_size=1000)

            if samples:
                # Save to cache
                cache_filename = f"wmt_mqm_{src_lang}_{tgt_lang}.json"
                await loader.save_to_cache(samples, cache_filename)

                print(f"✅ Downloaded {len(samples)} samples with error annotations")
            else:
                print(f"⚠️  No WMT-MQM data available for {src_lang}-{tgt_lang}")

        except Exception as e:
            print(f"❌ Error downloading WMT-MQM {src_lang}-{tgt_lang}: {e}")

    print("\n" + "=" * 80)
    print("✅ WMT-MQM download complete!")
    print("=" * 80)


async def main() -> None:
    """Main download function."""
    print("\n🚀 KTTC Benchmark Data Downloader\n")

    # Create data directory
    data_dir = Path("tests/benchmarks/data")
    data_dir.mkdir(parents=True, exist_ok=True)

    print(f"📁 Data directory: {data_dir.absolute()}\n")

    # Check if datasets library is installed
    try:
        import datasets  # noqa: F401

        print("✅ HuggingFace datasets library found")
    except ImportError:
        print("⚠️  HuggingFace datasets library not installed")
        print("   Install with: pip install datasets")
        print("   Falling back to sample data\n")

    # Download datasets
    await download_flores200()
    await download_wmt_mqm()

    print("\n" + "=" * 80)
    print("🎉 ALL DOWNLOADS COMPLETE!")
    print("=" * 80)
    print(f"\nData saved to: {data_dir.absolute()}")
    print("\nYou can now run benchmarks with:")
    print("  python3.11 scripts/run_comprehensive_benchmark.py")


if __name__ == "__main__":
    asyncio.run(main())
