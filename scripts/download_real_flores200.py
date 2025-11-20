#!/usr/bin/env python3
"""Download REAL FLORES-200 dataset using alternative method."""

import asyncio
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


async def download_flores200_alternative():
    """Download FLORES-200 using datasets library with correct format."""
    print("\n🌐 Downloading REAL FLORES-200 Dataset\n")
    print("=" * 80)

    try:
        from datasets import load_dataset

        # Language code mappings
        lang_map = {
            "en": "eng_Latn",
            "ru": "rus_Cyrl",
            "zh": "zho_Hans",
        }

        # Load full FLORES-200 dataset (alternative source without loading scripts)
        print("Loading FLORES-200 dataset from HuggingFace (Muennighoff/flores200)...")
        dataset = load_dataset("Muennighoff/flores200", split="devtest")

        print(f"✅ Loaded {len(dataset)} sentences")
        print(f"Available columns: {dataset.column_names[:10]}...")

        # Process language pairs
        pairs = [
            ("en", "ru"),
            ("en", "zh"),
            ("ru", "en"),
            ("zh", "en"),
            ("ru", "zh"),
            ("zh", "ru"),
        ]

        data_dir = Path("tests/benchmarks/data")
        data_dir.mkdir(parents=True, exist_ok=True)

        for src, tgt in pairs:
            src_col = f"sentence_{lang_map[src]}"
            tgt_col = f"sentence_{lang_map[tgt]}"

            print(f"\n📝 Processing {src} → {tgt}...")

            if src_col not in dataset.column_names or tgt_col not in dataset.column_names:
                print(f"   ⚠️  Columns not found: {src_col}, {tgt_col}")
                continue

            samples = []
            for idx, item in enumerate(dataset):
                samples.append(
                    {
                        "id": f"flores200_{src}-{tgt}_{idx}",
                        "source": item[src_col],
                        "translation": item[tgt_col],
                        "source_lang": src,
                        "target_lang": tgt,
                        "domain": "general",
                        "dataset": "flores200_real",
                    }
                )

            # Save
            output_file = data_dir / f"flores200_real_{src}_{tgt}_devtest.json"
            output_file.write_text(
                json.dumps(samples, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            print(f"   ✅ Saved {len(samples)} samples to {output_file.name}")

        print("\n" + "=" * 80)
        print("✅ REAL FLORES-200 download complete!")
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(download_flores200_alternative())
