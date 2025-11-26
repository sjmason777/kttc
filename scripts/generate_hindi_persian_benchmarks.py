"""Generate benchmark data for Hindi and Persian languages.

This script creates complete benchmark datasets for Hindi and Persian:
- flores200 dev/devtest (reference translations)
- quality (clean LLM translations)
- synthetic_bad (with intentional errors)
- critical_bad (critical translation errors)
"""

import json
from pathlib import Path

# Base directory for benchmark data
BENCHMARK_DIR = Path(__file__).parent.parent / "tests" / "benchmarks" / "data"

# Sample sentences for benchmarking
ENGLISH_SENTENCES = [
    "Artificial intelligence is transforming the world.",
    "Machine translation quality has improved significantly.",
    "Natural language processing enables human-computer interaction.",
    "Deep learning models process vast amounts of data.",
    "Neural networks mimic human brain structure.",
    "Cloud computing provides scalable infrastructure.",
    "Cybersecurity protects sensitive information.",
    "Blockchain technology ensures data integrity.",
    "Quantum computing promises exponential speedup.",
    "Internet of Things connects everyday devices.",
]

# Hindi translations (हिन्दी)
HINDI_TRANSLATIONS = [
    "कृत्रिम बुद्धिमत्ता दुनिया को बदल रही है।",
    "मशीनी अनुवाद की गुणवत्ता में काफी सुधार हुआ है।",
    "प्राकृतिक भाषा प्रसंस्करण मानव-कंप्यूटर संवाद को सक्षम बनाता है।",
    "डीप लर्निंग मॉडल विशाल मात्रा में डेटा को संसाधित करते हैं।",
    "तंत्रिका नेटवर्क मानव मस्तिष्क की संरचना की नकल करते हैं।",
    "क्लाउड कंप्यूटिंग स्केलेबल बुनियादी ढांचा प्रदान करता है।",
    "साइबर सुरक्षा संवेदनशील जानकारी की रक्षा करती है।",
    "ब्लॉकचेन तकनीक डेटा की अखंडता सुनिश्चित करती है।",
    "क्वांटम कंप्यूटिंग घातांकीय गति का वादा करती है।",
    "इंटरनेट ऑफ थिंग्स रोजमर्रा के उपकरणों को जोड़ता है।",
]

# Persian translations (فارسی)
PERSIAN_TRANSLATIONS = [
    "هوش مصنوعی در حال تغییر جهان است.",
    "کیفیت ترجمه ماشینی به طور قابل توجهی بهبود یافته است.",
    "پردازش زبان طبیعی تعامل انسان و کامپیوتر را امکان‌پذیر می‌سازد.",
    "مدل‌های یادگیری عمیق مقادیر عظیمی از داده‌ها را پردازش می‌کنند.",
    "شبکه‌های عصبی ساختار مغز انسان را تقلید می‌کنند.",
    "رایانش ابری زیرساخت مقیاس‌پذیر را فراهم می‌کند.",
    "امنیت سایبری از اطلاعات حساس محافظت می‌کند.",
    "فناوری بلاکچین یکپارچگی داده‌ها را تضمین می‌کند.",
    "محاسبات کوانتومی سرعت نمایی را وعده می‌دهد.",
    "اینترنت اشیا دستگاه‌های روزمره را به هم متصل می‌کند.",
]


def generate_flores200_data(
    source_lang: str, target_lang: str, translations: list[str]
) -> list[dict]:
    """Generate FLORES200-style reference data."""
    data = []
    for i, (source, translation) in enumerate(zip(ENGLISH_SENTENCES, translations)):
        data.append(
            {
                "id": f"fallback_{source_lang}-{target_lang}_{i}",
                "source": source if source_lang == "en" else translation,
                "translation": translation if source_lang == "en" else source,
                "source_lang": source_lang,
                "target_lang": target_lang,
                "domain": "general",
                "dataset": "fallback",
            }
        )
    return data


def generate_quality_data(
    source_lang: str, target_lang: str, translations: list[str]
) -> list[dict]:
    """Generate quality LLM translation data."""
    data = []
    for i, (source, translation) in enumerate(zip(ENGLISH_SENTENCES, translations)):
        data.append(
            {
                "id": f"quality_{source_lang}-{target_lang}_{i}",
                "source": source if source_lang == "en" else translation,
                "translation": translation if source_lang == "en" else source,
                "source_lang": source_lang,
                "target_lang": target_lang,
                "domain": "diverse",
                "dataset": "quality_llm",
            }
        )
    return data


def introduce_error(text: str, error_type: str, lang: str) -> str:
    """Introduce synthetic errors in translation."""
    if error_type == "addition":
        # Add extra word
        if lang == "hi":
            return text.replace("।", " आज।")  # Add "today"
        if lang == "fa":
            return text.replace(".", " امروز.")  # Add "today"
        return text.replace(".", " today.")
    if error_type == "omission":
        # Remove a word
        words = text.split()
        if len(words) > 3:
            return " ".join(words[:-2] + [words[-1]])
    elif error_type == "mistranslation":
        # Wrong word
        if lang == "hi":
            return text.replace("बुद्धिमत्ता", "खुफिया")  # AI -> intelligence (wrong context)
        if lang == "fa":
            return text.replace("هوش", "اطلاعات")  # AI -> information (wrong)
        return text.replace("artificial", "natural")
    return text


def generate_synthetic_bad_data(
    source_lang: str, target_lang: str, translations: list[str]
) -> list[dict]:
    """Generate synthetic bad translation data with intentional errors."""
    data = []
    error_types = ["addition", "omission", "mistranslation"]

    for i, (source, translation) in enumerate(zip(ENGLISH_SENTENCES, translations)):
        # Good translation
        data.append(
            {
                "id": f"fallback_{source_lang}-{target_lang}_{i*2}",
                "source": source if source_lang == "en" else translation,
                "translation": translation if source_lang == "en" else source,
                "source_lang": source_lang,
                "target_lang": target_lang,
                "domain": "general",
                "dataset": "fallback",
                "quality": "good",
                "expected_mqm_range": [95, 100],
            }
        )

        # Bad translation (if we have errors defined)
        if i < len(error_types):
            error_type = error_types[i % len(error_types)]
            bad_translation = introduce_error(
                translation if source_lang == "en" else source,
                error_type,
                target_lang if source_lang == "en" else source_lang,
            )
            data.append(
                {
                    "id": f"fallback_{source_lang}-{target_lang}_{i*2+1}",
                    "source": source if source_lang == "en" else translation,
                    "translation": bad_translation,
                    "source_lang": source_lang,
                    "target_lang": target_lang,
                    "domain": "general",
                    "dataset": "fallback",
                    "quality": "bad",
                    "error_type": error_type,
                    "error_severity": "minor",
                    "expected_mqm_range": [40, 75],
                }
            )

    return data


def generate_critical_bad_data(
    source_lang: str, target_lang: str, translations: list[str]
) -> list[dict]:
    """Generate critical bad translation data."""
    data = []

    for i, (source, translation) in enumerate(
        zip(ENGLISH_SENTENCES[:5], translations[:5])
    ):  # Just 5 samples
        # Introduce critical errors
        if target_lang == "hi":
            # Complete mistranslation
            bad_translation = "यह बिल्कुल गलत अनुवाद है जो मूल अर्थ को बदल देता है।"
        elif target_lang == "fa":
            bad_translation = "این ترجمه کاملاً نادرست است که معنی اصلی را تغییر می‌دهد."
        else:
            bad_translation = (
                "This is a completely wrong translation that changes the original meaning."
            )

        data.append(
            {
                "id": f"critical_{source_lang}-{target_lang}_{i}",
                "source": source if source_lang == "en" else translation,
                "translation": bad_translation,
                "source_lang": source_lang,
                "target_lang": target_lang,
                "domain": "general",
                "dataset": "critical_errors",
                "quality": "critical",
                "error_type": "complete_mistranslation",
                "error_severity": "critical",
                "expected_mqm_range": [0, 30],
            }
        )

    return data


def save_json(data: list[dict], filename: str) -> None:
    """Save data to JSON file."""
    filepath = BENCHMARK_DIR / filename
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"✅ Created: {filename} ({len(data)} samples)")


def main():
    """Generate all benchmark data for Hindi and Persian."""
    print("=" * 60)
    print("Generating Hindi and Persian Benchmark Data")
    print("=" * 60)

    # Ensure directory exists
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)

    # Language pairs to generate
    language_pairs = [
        ("en", "hi", HINDI_TRANSLATIONS, "Hindi"),
        ("hi", "en", HINDI_TRANSLATIONS, "Hindi"),
        ("en", "fa", PERSIAN_TRANSLATIONS, "Persian"),
        ("fa", "en", PERSIAN_TRANSLATIONS, "Persian"),
    ]

    for source_lang, target_lang, translations, lang_name in language_pairs:
        print(f"\n--- Generating {source_lang} → {target_lang} ({lang_name}) ---")

        # FLORES200 dev and devtest
        flores_data = generate_flores200_data(source_lang, target_lang, translations)
        save_json(flores_data, f"flores200_{source_lang}_{target_lang}_dev.json")
        save_json(flores_data, f"flores200_{source_lang}_{target_lang}_devtest.json")

        # Quality data
        quality_data = generate_quality_data(source_lang, target_lang, translations)
        save_json(quality_data, f"quality_{source_lang}_{target_lang}.json")

        # Synthetic bad data
        synthetic_data = generate_synthetic_bad_data(source_lang, target_lang, translations)
        save_json(synthetic_data, f"synthetic_bad_{source_lang}_{target_lang}.json")

        # Critical bad data
        critical_data = generate_critical_bad_data(source_lang, target_lang, translations)
        save_json(critical_data, f"critical_bad_{source_lang}_{target_lang}.json")

    print("\n" + "=" * 60)
    print("✅ All Hindi and Persian benchmark data generated!")
    print(f"📁 Location: {BENCHMARK_DIR}")
    print("=" * 60)

    # Summary
    print("\n📊 Summary:")
    print("  - Hindi: 8 files (4 pairs × 2 directions)")
    print("  - Persian: 8 files (4 pairs × 2 directions)")
    print("  - Total new files: 16")


if __name__ == "__main__":
    main()
