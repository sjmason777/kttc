"""Clean quality_*.json files by removing English explanations from translations.

Problem: Old quality files contain LLM-generated translations with English explanations like:
  "Here's the translation to Russian:\n\nТекст перевода...\n\nPronunciation: ..."

Solution: Extract only the actual translation part (Cyrillic/Chinese text).
"""

import json
from pathlib import Path

# Base directory for benchmark data
BENCHMARK_DIR = Path(__file__).parent.parent / "tests" / "benchmarks" / "data"


def extract_clean_translation(text: str, target_lang: str) -> str:
    """Extract clean translation from LLM output with English explanations.

    Args:
        text: LLM output that may contain English explanations
        target_lang: Target language code (ru, zh)

    Returns:
        Clean translation text
    """
    # If already clean (no English phrases), return as is
    if not any(
        phrase in text.lower()
        for phrase in [
            "here's the translation",
            "pronunciation",
            "transliteration",
            "guide:",
        ]
    ):
        return text

    # Try to extract the actual translation
    lines = text.split("\n")
    clean_lines = []

    for line in lines:
        line = line.strip()

        # Skip empty lines
        if not line:
            continue

        # Skip English explanation lines
        if any(
            phrase in line.lower()
            for phrase in [
                "here's the translation",
                "pronunciation",
                "transliteration",
                "guide:",
                "= quantum",
                "= computer",
                "= promise",
                "simplified chinese:",
                "traditional chinese:",
                "pinyin:",
            ]
        ):
            continue

        # Skip lines that are mostly Latin characters (English explanations)
        if target_lang in ["ru", "zh"]:
            # Count non-Latin characters
            if target_lang == "ru":
                non_latin = sum(
                    1 for c in line if ord(c) >= 0x0400 and ord(c) <= 0x04FF
                )  # Cyrillic
            elif target_lang == "zh":
                non_latin = sum(1 for c in line if ord(c) >= 0x4E00 and ord(c) <= 0x9FFF)  # Chinese

            # If less than 50% non-Latin, skip (likely English)
            if len(line) > 0 and non_latin / len(line) < 0.5:
                continue

        clean_lines.append(line)

    # Join clean lines
    if clean_lines:
        return " ".join(clean_lines)

    # Fallback: return original if we couldn't extract anything
    return text


def clean_quality_file(filepath: Path) -> tuple[bool, str]:
    """Clean a single quality file.

    Returns:
        (success, message)
    """
    try:
        # Read file
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)

        # Determine target language from filename
        # Format: quality_XX_YY.json where YY is target language
        filename = filepath.stem  # Remove .json
        parts = filename.split("_")
        if len(parts) != 3:
            return False, f"Invalid filename format: {filepath.name}"

        target_lang = parts[2]

        # Clean each entry
        cleaned_count = 0
        for entry in data:
            original_translation = entry["translation"]
            clean_translation = extract_clean_translation(original_translation, target_lang)

            if original_translation != clean_translation:
                entry["translation"] = clean_translation
                cleaned_count += 1

        # Write back
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        if cleaned_count > 0:
            return True, f"Cleaned {cleaned_count}/{len(data)} entries"
        else:
            return True, "Already clean"

    except Exception as e:
        return False, f"Error: {e}"


def main():
    """Clean all quality files."""
    print("=" * 70)
    print("Cleaning Quality Files")
    print("=" * 70)

    # Find all quality files
    quality_files = sorted(BENCHMARK_DIR.glob("quality_*.json"))

    if not quality_files:
        print(f"\n❌ No quality files found in {BENCHMARK_DIR}")
        return

    print(f"\n📂 Found {len(quality_files)} quality files\n")
    print("=" * 70)

    # Clean each file
    for filepath in quality_files:
        success, message = clean_quality_file(filepath)

        if success:
            print(f"✅ {filepath.name:30s} {message}")
        else:
            print(f"❌ {filepath.name:30s} {message}")

    print("\n" + "=" * 70)
    print("✅ All quality files processed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
