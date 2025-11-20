#!/usr/bin/env python3
"""Test Russian error detection quality with mawo-grammar rules."""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from kttc.agents.fluency_russian import RussianFluencyAgent
from kttc.core import TranslationTask
from kttc.utils.config import get_settings

# Test cases covering different error types
TEST_CASES = [
    # 1. Case agreement errors (падежное согласование)
    {
        "id": "case_agreement_1",
        "source": "the fast fox",
        "translation": "быстрый лиса",  # ERROR: masc adj + fem noun
        "correct": "быстрая лиса",
        "error_type": "case_agreement",
        "description": "Gender mismatch: 'быстрый' (masc) with 'лиса' (fem)",
    },
    {
        "id": "case_agreement_2",
        "source": "beautiful house",
        "translation": "красивая дом",  # ERROR: fem adj + masc noun
        "correct": "красивый дом",
        "error_type": "case_agreement",
        "description": "Gender mismatch: 'красивая' (fem) with 'дом' (masc)",
    },
    # 2. НЕ/НИ particles (частицы) - most common error in EGE 2025
    {
        "id": "particle_ne_ni_1",
        "source": "He never saw it",
        "translation": "Он не разу не видел",  # ERROR: should be "ни разу"
        "correct": "Он ни разу не видел",
        "error_type": "particles",
        "description": "НЕ/НИ confusion: should use 'ни разу' not 'не разу'",
    },
    {
        "id": "particle_ne_ni_2",
        "source": "Not a single person",
        "translation": "Не один человек не пришел",  # ERROR: should be "ни один"
        "correct": "Ни один человек не пришел",
        "error_type": "particles",
        "description": "НЕ/НИ confusion: should use 'ни один' not 'не один' in negative context",
    },
    # 3. Verb aspect errors (совершенный/несовершенный вид)
    {
        "id": "aspect_1",
        "source": "I was reading when he called",
        "translation": "Я прочитал, когда он позвонил",  # ERROR: perfective in progressive context
        "correct": "Я читал, когда он позвонил",
        "error_type": "aspect",
        "description": "Wrong aspect: perfective 'прочитал' in continuous context",
    },
    # 4. Preposition + case government (управление)
    {
        "id": "preposition_case_1",
        "source": "I'm going to the store",
        "translation": "Я иду к магазину",  # ERROR: should be "в магазин" (acc)
        "correct": "Я иду в магазин",
        "error_type": "prepositions",
        "description": "Wrong preposition + case: 'к + dat' instead of 'в + acc'",
    },
    {
        "id": "preposition_case_2",
        "source": "I came from Moscow",
        "translation": "Я приехал с Москвы",  # ERROR: should be "из Москвы" (gen)
        "correct": "Я приехал из Москвы",
        "error_type": "prepositions",
        "description": "Wrong preposition: 'с' instead of 'из' with city name",
    },
    # 5. Word order errors (порядок слов)
    {
        "id": "word_order_1",
        "source": "I don't know",
        "translation": "Не я знаю",  # ERROR: unnatural word order
        "correct": "Я не знаю",
        "error_type": "word_order",
        "description": "Unnatural word order: НЕ before pronoun",
    },
    # 6. Register consistency (ты/вы)
    {
        "id": "register_1",
        "source": "Hello, how are you?",
        "translation": "Здравствуйте, как ты?",  # ERROR: formal + informal
        "correct": "Здравствуйте, как вы?",
        "error_type": "register",
        "description": "Register mismatch: formal 'здравствуйте' with informal 'ты'",
    },
    # 7. Numeral-noun agreement (числительные)
    {
        "id": "numeral_agreement_1",
        "source": "two cars",
        "translation": "два машины",  # ERROR: should be genitive singular "машины" or "две машины"
        "correct": "две машины",
        "error_type": "agreement",
        "description": "Numeral gender: 'два' (masc) with 'машина' (fem), should be 'две'",
    },
    # 8. Comma with gerund (деепричастный оборот)
    {
        "id": "gerund_comma_1",
        "source": "Walking down the street, I saw him",
        "translation": "Идя по улице я увидел его",  # ERROR: missing comma after gerund
        "correct": "Идя по улице, я увидел его",
        "error_type": "punctuation",
        "description": "Missing comma after gerund phrase",
    },
    # 9. Good translations (should have NO errors)
    {
        "id": "good_1",
        "source": "The cat is sleeping",
        "translation": "Кошка спит",
        "correct": "Кошка спит",
        "error_type": None,
        "description": "Correct translation, no errors expected",
    },
    {
        "id": "good_2",
        "source": "I love learning Russian",
        "translation": "Я люблю изучать русский язык",
        "correct": "Я люблю изучать русский язык",
        "error_type": None,
        "description": "Correct translation, no errors expected",
    },
]


async def test_russian_error_detection():
    """Test Russian error detection across different error types."""
    print("\n" + "=" * 80)
    print("RUSSIAN ERROR DETECTION QUALITY TEST")
    print("=" * 80)
    print(f"\nTest cases: {len(TEST_CASES)}")
    print("Error types: case agreement, НЕ/НИ particles, aspect, prepositions,")
    print("             word order, register, numerals, punctuation\n")

    # Initialize
    settings = get_settings()

    try:
        api_key = settings.get_llm_provider_key("anthropic")
        from kttc.llm import AnthropicProvider

        llm = AnthropicProvider(api_key=api_key, model="claude-3-5-haiku-20241022")
        print("✅ Using Anthropic Claude 3.5 Haiku\n")
    except Exception:
        try:
            api_key = settings.get_llm_provider_key("openai")
            from kttc.llm import OpenAIProvider

            llm = OpenAIProvider(api_key=api_key, model="gpt-4o-mini")
            print("✅ Using OpenAI GPT-4o-mini\n")
        except Exception:
            print("❌ No LLM provider available")
            return

    # Create Russian fluency agent
    agent = RussianFluencyAgent(llm)

    # Check if mawo-grammar is available
    if agent.helper.is_available():
        print("✅ mawo-grammar libraries available")
    else:
        print("⚠️  mawo-grammar not available - running in limited mode")

    print()

    # Track results
    results = {
        "detected": 0,  # Errors correctly found
        "missed": 0,  # Errors not found
        "false_positives": 0,  # Errors in good translations
        "total_errors": 0,  # Total error test cases
        "total_good": 0,  # Total good test cases
    }

    error_type_stats = {}

    # Test each case
    for i, test_case in enumerate(TEST_CASES, 1):
        print(f"[{i}/{len(TEST_CASES)}] Testing: {test_case['id']}")
        print(f"  Translation: {test_case['translation']}")

        task = TranslationTask(
            source_text=test_case["source"],
            translation=test_case["translation"],
            source_lang="en",
            target_lang="ru",
        )

        errors = await agent.evaluate(task)

        error_type = test_case["error_type"]

        if error_type is None:
            # Good translation - should have no errors
            results["total_good"] += 1
            if errors:
                results["false_positives"] += 1
                print(f"  ❌ FALSE POSITIVE: Found {len(errors)} error(s) in good translation")
                for err in errors:
                    print(f"     - {err.category}: {err.description}")
            else:
                print("  ✅ CORRECT: No errors found (as expected)")
        else:
            # Bad translation - should have errors
            results["total_errors"] += 1

            # Track by error type
            if error_type not in error_type_stats:
                error_type_stats[error_type] = {"detected": 0, "missed": 0}

            if errors:
                results["detected"] += 1
                error_type_stats[error_type]["detected"] += 1
                print(f"  ✅ DETECTED: Found {len(errors)} error(s)")
                print(f"     Expected: {test_case['description']}")
                for err in errors:
                    print(f"     Found: {err.category} - {err.description}")
            else:
                results["missed"] += 1
                error_type_stats[error_type]["missed"] += 1
                print("  ❌ MISSED: No errors found")
                print(f"     Expected: {test_case['description']}")
                print(f"     Correct: {test_case['correct']}")

        print()

    # Print summary
    print("=" * 80)
    print("TEST RESULTS SUMMARY")
    print("=" * 80)
    print()

    print("ERROR DETECTION:")
    print(f"  Total error cases: {results['total_errors']}")
    print(
        f"  Detected: {results['detected']} ({results['detected']/results['total_errors']*100:.1f}%)"
    )
    print(f"  Missed: {results['missed']} ({results['missed']/results['total_errors']*100:.1f}%)")
    print()

    print("GOOD TRANSLATION VALIDATION:")
    print(f"  Total good cases: {results['total_good']}")
    print(
        f"  False positives: {results['false_positives']} ({results['false_positives']/results['total_good']*100:.1f}% if results['total_good'] > 0 else 0)%)"
    )
    print()

    print("BY ERROR TYPE:")
    for error_type, stats in sorted(error_type_stats.items()):
        total = stats["detected"] + stats["missed"]
        detection_rate = stats["detected"] / total * 100 if total > 0 else 0
        print(f"  {error_type:20s}: {stats['detected']}/{total} detected ({detection_rate:.1f}%)")
    print()

    # Overall score
    overall_detection = (
        results["detected"] / results["total_errors"] * 100 if results["total_errors"] > 0 else 0
    )
    false_positive_rate = (
        results["false_positives"] / results["total_good"] * 100 if results["total_good"] > 0 else 0
    )

    print("OVERALL QUALITY:")
    print(f"  Error detection rate: {overall_detection:.1f}%")
    print(f"  False positive rate: {false_positive_rate:.1f}%")

    # Quality assessment
    if overall_detection >= 90 and false_positive_rate <= 10:
        print("  Assessment: ✅ EXCELLENT - High quality Russian error detection")
    elif overall_detection >= 70 and false_positive_rate <= 20:
        print("  Assessment: ✓ GOOD - Reliable Russian error detection")
    elif overall_detection >= 50:
        print("  Assessment: ⚠️  MODERATE - Needs improvement")
    else:
        print("  Assessment: ❌ POOR - Significant gaps in error detection")

    print()
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(test_russian_error_detection())
