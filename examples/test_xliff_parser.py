#!/usr/bin/env python3.11
"""Quick test to verify XLIFF parser implementation."""

from pathlib import Path

from kttc.core import BatchFileParser

# Test XLIFF parsing
xliff_file = Path(__file__).parent / "test_translations.xliff"
print(f"Testing XLIFF parser with: {xliff_file}\n")

try:
    translations = BatchFileParser.parse_xliff(xliff_file)
    print(f"✓ Successfully parsed {len(translations)} translations\n")

    for i, trans in enumerate(translations, 1):
        print(f"Translation {i}:")
        print(f"  Source ({trans.source_lang}): {trans.source_text}")
        print(f"  Target ({trans.target_lang}): {trans.translation}")
        print(f"  Metadata: {trans.metadata}")
        print()

    # Test auto-detect parsing
    print("\nTesting auto-detect parsing...")
    translations2 = BatchFileParser.parse(xliff_file)
    print(f"✓ Auto-detect successfully parsed {len(translations2)} translations")

    print("\n✅ All XLIFF parsing tests passed!")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback

    traceback.print_exc()
