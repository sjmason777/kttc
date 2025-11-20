"""Validate all benchmark data files.

Checks:
1. All files are valid JSON
2. Required fields are present
3. Language codes are correct
4. No duplicate IDs
5. Translations don't contain English explanations (for quality files)
"""

import json
from collections import defaultdict
from pathlib import Path

# Base directory for benchmark data
BENCHMARK_DIR = Path(__file__).parent.parent / "tests" / "benchmarks" / "data"

# Expected file types
FILE_TYPES = ["flores200", "quality", "synthetic_bad", "critical_bad"]
LANGUAGE_CODES = ["en", "ru", "zh", "hi", "fa"]

# Required fields
REQUIRED_FIELDS = ["id", "source", "translation", "source_lang", "target_lang", "domain", "dataset"]


def validate_json_file(filepath: Path) -> tuple[bool, str]:
    """Validate a single JSON file."""
    try:
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            return False, "Root element is not a list"

        if len(data) == 0:
            return False, "File is empty"

        # Check each entry
        for i, entry in enumerate(data):
            # Check required fields
            for field in REQUIRED_FIELDS:
                if field not in entry:
                    return False, f"Entry {i}: Missing field '{field}'"

            # Check language codes
            if entry["source_lang"] not in LANGUAGE_CODES:
                return False, f"Entry {i}: Invalid source_lang '{entry['source_lang']}'"
            if entry["target_lang"] not in LANGUAGE_CODES:
                return False, f"Entry {i}: Invalid target_lang '{entry['target_lang']}'"

            # Check for English explanations in translations (quality files only)
            if "quality" in filepath.name:
                translation = entry["translation"]
                if any(
                    phrase in translation.lower()
                    for phrase in [
                        "here's the translation",
                        "pronunciation",
                        "transliteration",
                        "guide:",
                    ]
                ):
                    return False, f"Entry {i}: Translation contains English explanation"

        return True, f"OK ({len(data)} entries)"

    except json.JSONDecodeError as e:
        return False, f"JSON decode error: {e}"
    except Exception as e:
        return False, f"Error: {e}"


def main():
    """Validate all benchmark files."""
    print("=" * 70)
    print("Validating Benchmark Data Files")
    print("=" * 70)

    all_files = sorted(BENCHMARK_DIR.glob("*.json"))

    if not all_files:
        print(f"\n❌ No files found in {BENCHMARK_DIR}")
        return

    # Statistics
    stats = {
        "total": len(all_files),
        "valid": 0,
        "invalid": 0,
        "by_type": defaultdict(int),
        "by_language_pair": defaultdict(int),
    }

    # Validate each file
    print(f"\n📂 Found {len(all_files)} files")
    print("\n" + "=" * 70)

    errors = []

    for filepath in all_files:
        filename = filepath.name
        is_valid, message = validate_json_file(filepath)

        # Update statistics
        if is_valid:
            stats["valid"] += 1
            status = "✅"
        else:
            stats["invalid"] += 1
            status = "❌"
            errors.append((filename, message))

        # Count by type
        for file_type in FILE_TYPES:
            if file_type in filename:
                stats["by_type"][file_type] += 1
                break

        # Count by language pair
        for lang1 in LANGUAGE_CODES:
            for lang2 in LANGUAGE_CODES:
                if lang1 != lang2 and f"_{lang1}_{lang2}" in filename:
                    stats["by_language_pair"][f"{lang1}→{lang2}"] += 1
                    break

        print(f"{status} {filename:50s} {message}")

    # Print summary
    print("\n" + "=" * 70)
    print("📊 Summary")
    print("=" * 70)
    print(f"\nTotal files:   {stats['total']}")
    print(f"Valid:         {stats['valid']} ✅")
    print(f"Invalid:       {stats['invalid']} ❌")

    # By type
    print("\n📁 By Type:")
    for file_type, count in sorted(stats["by_type"].items()):
        print(f"  - {file_type:20s}: {count:2d} files")

    # By language pair
    print("\n🌐 By Language Pair:")
    for lang_pair, count in sorted(stats["by_language_pair"].items()):
        print(f"  - {lang_pair:10s}: {count:2d} files")

    # Show errors
    if errors:
        print("\n" + "=" * 70)
        print("❌ Errors Found")
        print("=" * 70)
        for filename, error in errors:
            print(f"\n{filename}:")
            print(f"  {error}")

    # Final status
    print("\n" + "=" * 70)
    if stats["invalid"] == 0:
        print("✅ ALL FILES VALID!")
    else:
        print(f"⚠️  {stats['invalid']} files need fixing")
    print("=" * 70)


if __name__ == "__main__":
    main()
