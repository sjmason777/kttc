#!/usr/bin/env python3
"""Download FLORES-200 directly from GitHub repository."""

import asyncio
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


async def download_from_github():
    """Download FLORES-200 from official GitHub repo."""
    import urllib.request

    print("\n🌐 Downloading FLORES-200 from GitHub\n")
    print("=" * 80)

    # Official FLORES-200 repository
    base_url = "https://raw.githubusercontent.com/facebookresearch/flores/main/flores200/devtest"

    # Language codes
    langs = {
        "en": "eng_Latn",
        "ru": "rus_Cyrl",
        "zh": "zho_Hans",
    }

    data_dir = Path("tests/benchmarks/data")
    data_dir.mkdir(parents=True, exist_ok=True)

    # Download each language file
    downloaded_data = {}

    for lang_code, flores_code in langs.items():
        url = f"{base_url}/{flores_code}.devtest"
        print(f"\n📥 Downloading {lang_code} ({flores_code})...")
        print(f"   URL: {url}")

        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                content = response.read().decode("utf-8")
                lines = content.strip().split("\n")
                downloaded_data[lang_code] = lines
                print(f"   ✅ Downloaded {len(lines)} sentences")
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return False

    # Create language pairs
    pairs = [
        ("en", "ru"),
        ("en", "zh"),
        ("ru", "en"),
        ("zh", "en"),
        ("ru", "zh"),
        ("zh", "ru"),
    ]

    for src, tgt in pairs:
        print(f"\n📝 Creating {src} → {tgt} dataset...")

        src_lines = downloaded_data.get(src, [])
        tgt_lines = downloaded_data.get(tgt, [])

        if not src_lines or not tgt_lines:
            print(f"   ⚠️  Missing data for {src} or {tgt}")
            continue

        # Create samples
        samples = []
        for idx, (src_text, tgt_text) in enumerate(zip(src_lines, tgt_lines)):
            samples.append(
                {
                    "id": f"flores200_github_{src}-{tgt}_{idx}",
                    "source": src_text.strip(),
                    "translation": tgt_text.strip(),
                    "source_lang": src,
                    "target_lang": tgt,
                    "domain": "general",
                    "dataset": "flores200_github",
                }
            )

        # Save
        output_file = data_dir / f"flores200_github_{src}_{tgt}.json"
        output_file.write_text(json.dumps(samples, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"   ✅ Saved {len(samples)} samples to {output_file.name}")

    print("\n" + "=" * 80)
    print("✅ FLORES-200 download from GitHub complete!")
    print("=" * 80)
    return True


if __name__ == "__main__":
    success = asyncio.run(download_from_github())
    sys.exit(0 if success else 1)
