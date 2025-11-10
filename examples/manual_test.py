#!/usr/bin/env python3
"""Quick test script for KTTC with free API options.

This script helps you test KTTC with minimal setup using free API tiers.
"""

import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, cast

from kttc.agents.orchestrator import AgentOrchestrator
from kttc.core.models import TranslationTask
from kttc.llm.base import BaseLLMProvider

# Load .env file if it exists
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    # python-dotenv not installed, try manual loading
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ.setdefault(key.strip(), value.strip())


async def test_translation_quality(
    api_key: str | dict[str, str],
    provider: str = "openai",
    model: str | None = None,
) -> None:
    """Test translation quality with specified provider.

    Args:
        api_key: API key for the provider (string for OpenAI/Anthropic,
                dict with credentials for GigaChat)
        provider: Provider name ('openai', 'anthropic', or 'gigachat')
        model: Model name (optional, uses cheap defaults)
    """
    # Setup provider
    llm: BaseLLMProvider
    if provider == "openai":
        from kttc.llm.openai_provider import OpenAIProvider

        # Use cheaper model by default
        model = model or "gpt-4o-mini"
        if not isinstance(api_key, str):
            print("❌ OpenAI requires string API key")
            sys.exit(1)
        llm = OpenAIProvider(api_key=api_key, model=model)
        print(f"✓ Using OpenAI with model: {model}")
    elif provider == "anthropic":
        from kttc.llm.anthropic_provider import AnthropicProvider

        model = model or "claude-3-5-sonnet-20241022"
        if not isinstance(api_key, str):
            print("❌ Anthropic requires string API key")
            sys.exit(1)
        llm = AnthropicProvider(api_key=api_key, model=model)
        print(f"✓ Using Anthropic with model: {model}")
    elif provider == "gigachat":
        from kttc.llm.gigachat_provider import GigaChatProvider

        # api_key here is actually a dict with credentials
        if isinstance(api_key, dict):
            model = model or "GigaChat"
            llm = GigaChatProvider(
                client_id=api_key["client_id"],
                client_secret=api_key["client_secret"],
                scope=api_key.get("scope", "GIGACHAT_API_PERS"),
                model=model,
            )
            print(f"✓ Using GigaChat with model: {model}")
        else:
            print("❌ GigaChat requires client_id and client_secret")
            sys.exit(1)
    else:
        print(f"❌ Unknown provider: {provider}")
        sys.exit(1)

    # Create orchestrator
    orchestrator = AgentOrchestrator(llm)

    # Test cases with different quality levels
    # Format: (name, task, expected_quality)
    test_cases = [
        # GOOD translation - should PASS
        (
            "GOOD: Technical text EN→ES",
            TranslationTask(
                source_text="Machine learning algorithms analyze large datasets to identify patterns.",
                translation="Los algoritmos de aprendizaje automático analizan grandes conjuntos de datos para identificar patrones.",
                source_lang="en",
                target_lang="es",
            ),
            "PASS",
        ),
        # GOOD translation - should PASS
        (
            "GOOD: Business text EN→ES",
            TranslationTask(
                source_text="Our company provides excellent customer service.",
                translation="Nuestra empresa proporciona un excelente servicio al cliente.",
                source_lang="en",
                target_lang="es",
            ),
            "PASS",
        ),
        # MINOR ERROR: Unnatural word order in Spanish
        (
            "MINOR ERROR: Word order EN→ES",
            TranslationTask(
                source_text="The user can easily change the settings.",
                translation="El usuario puede cambiar fácilmente la configuración.",  # OK, natural
                source_lang="en",
                target_lang="es",
            ),
            "PASS/MINOR",
        ),
        # MAJOR ERROR: Wrong terminology
        (
            "MAJOR ERROR: Wrong terminology EN→ES",
            TranslationTask(
                source_text="Click the submit button to save your changes.",
                translation="Haga clic en el botón enviar para guardar sus cambios.",  # Missing comma, "enviar" could be better as "envío"
                source_lang="en",
                target_lang="es",
            ),
            "FAIL",
        ),
        # MAJOR ERROR: Meaning loss
        (
            "MAJOR ERROR: Meaning changed EN→ES",
            TranslationTask(
                source_text="The service is available 24/7 for all users.",
                translation="El servicio está disponible para algunos usuarios.",  # Lost "24/7" and changed "all" to "some"
                source_lang="en",
                target_lang="es",
            ),
            "FAIL",
        ),
        # CRITICAL ERROR: Completely wrong translation
        (
            "CRITICAL ERROR: Wrong meaning EN→ES",
            TranslationTask(
                source_text="Delete all user data immediately.",
                translation="Guardar todos los datos del usuario inmediatamente.",  # Says "Save" instead of "Delete"!
                source_lang="en",
                target_lang="es",
            ),
            "FAIL",
        ),
        # MINOR ERROR: Grammar issue
        (
            "MINOR ERROR: Grammar EN→ES",
            TranslationTask(
                source_text="She has been working here for three years.",
                translation="Ella ha trabajado aquí por tres años.",  # Minor: "ha trabajado" vs "ha estado trabajando"
                source_lang="en",
                target_lang="es",
            ),
            "PASS/MINOR",
        ),
        # MAJOR ERROR: Literal translation (doesn't sound natural)
        (
            "MAJOR ERROR: Too literal EN→ES",
            TranslationTask(
                source_text="Break a leg!",
                translation="¡Rompe una pierna!",  # Literal translation of idiom - wrong!
                source_lang="en",
                target_lang="es",
            ),
            "FAIL",
        ),
    ]

    print(f"\n{'=' * 80}")
    print("KTTC Translation Quality Test Suite")
    print(f"Testing {len(test_cases)} translations with varying quality levels")
    print(f"{'=' * 80}\n")

    # Statistics
    stats = {
        "total": len(test_cases),
        "evaluated": 0,
        "passed": 0,
        "failed": 0,
        "errors": 0,
        "correct_predictions": 0,
    }

    # Detailed results for saving
    detailed_results: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "provider": provider,
        "model": model or "default",
        "test_cases": [],
    }

    # Evaluate each translation
    for i, (test_name, task, expected) in enumerate(test_cases, 1):
        print(f"\n{'─' * 80}")
        print(f"Test {i}/{len(test_cases)}: {test_name}")
        print(f"{'─' * 80}")
        print(f"Direction: {task.source_lang.upper()} → {task.target_lang.upper()}")
        print(f"Source: {task.source_text}")
        print(f"Translation: {task.translation}")
        print(f"Expected: {expected}")

        try:
            report = await orchestrator.evaluate(task)
            stats["evaluated"] += 1

            # Analyze results
            actual = "PASS" if report.status == "pass" else "FAIL"
            if report.status == "pass":
                stats["passed"] += 1
            else:
                stats["failed"] += 1

            # Check if prediction matches expectation
            prediction_correct = False
            if expected == "PASS" and actual == "PASS":
                prediction_correct = True
            elif expected == "FAIL" and actual == "FAIL":
                prediction_correct = True
            elif expected == "PASS/MINOR" and (actual == "PASS" or len(report.errors) <= 2):
                prediction_correct = True

            if prediction_correct:
                stats["correct_predictions"] += 1
                result_emoji = "✅ CORRECT"
            else:
                result_emoji = "❌ INCORRECT"

            # Save detailed results
            test_result = {
                "test_name": test_name,
                "source_text": task.source_text,
                "translation": task.translation,
                "source_lang": task.source_lang,
                "target_lang": task.target_lang,
                "expected": expected,
                "actual": actual,
                "mqm_score": report.mqm_score,
                "prediction_correct": prediction_correct,
                "errors": [
                    {
                        "severity": error.severity.name,
                        "category": error.category,
                        "description": error.description,
                        "suggestion": error.suggestion,
                        "location": error.location,
                    }
                    for error in report.errors
                ],
            }
            cast(list[Any], detailed_results["test_cases"]).append(test_result)

            # Print results
            print(f"\nResult: {result_emoji}")
            print(f"  MQM Score: {report.mqm_score:.2f}")
            print(f"  Status: {actual}")
            print(f"  Errors found: {len(report.errors)}")

            if report.errors:
                print("\n  Issues detected:")
                for j, error in enumerate(report.errors, 1):
                    severity_emoji = {
                        "neutral": "ℹ️",
                        "minor": "⚠️",
                        "major": "⚠️⚠️",
                        "critical": "🚨",
                    }.get(error.severity.name, "⚠️")
                    print(
                        f"    {j}. {severity_emoji} [{error.severity.name.upper()}] {error.category}"
                    )
                    print(f"       Description: {error.description}")
                    if error.suggestion:
                        print(f"       Suggestion: {error.suggestion}")
            else:
                print("  ✓ No issues detected")

        except Exception as e:
            stats["errors"] += 1
            print(f"\n❌ ERROR during evaluation: {e}")
            import traceback

            traceback.print_exc()

            # Save error in results
            cast(list[Any], detailed_results["test_cases"]).append(
                {
                    "test_name": test_name,
                    "source_text": task.source_text,
                    "translation": task.translation,
                    "expected": expected,
                    "error": str(e),
                }
            )

    # Final statistics
    print(f"\n{'=' * 80}")
    print("TEST RESULTS SUMMARY")
    print(f"{'=' * 80}")
    print(f"Total tests: {stats['total']}")
    print(f"Successfully evaluated: {stats['evaluated']}")
    print(f"Errors during evaluation: {stats['errors']}")
    print("\nTranslation Results:")
    print(f"  Passed: {stats['passed']}")
    print(f"  Failed: {stats['failed']}")
    print("\nAccuracy:")
    accuracy = (
        (stats["correct_predictions"] / stats["evaluated"] * 100) if stats["evaluated"] > 0 else 0
    )
    print(f"  Correct predictions: {stats['correct_predictions']}/{stats['evaluated']}")
    print(f"  Accuracy: {accuracy:.1f}%")

    if accuracy >= 70:
        print("\n✅ Test suite PASSED - KTTC is working correctly!")
    else:
        print("\n⚠️  Test suite NEEDS ATTENTION - accuracy below 70%")

    print(f"{'=' * 80}\n")

    # Save detailed results to file
    detailed_results["stats"] = stats
    detailed_results["accuracy"] = accuracy

    results_dir = Path("test_results")
    results_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f"kttc_test_results_{timestamp}.json"

    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(detailed_results, f, indent=2, ensure_ascii=False)

    print(f"📊 Detailed results saved to: {results_file}\n")


def main() -> None:
    """Main entry point."""
    print("KTTC Test Script - Free API Testing\n")

    # Check for API keys
    openai_key = os.getenv("OPENAI_API_KEY") or os.getenv("KTTC_OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("KTTC_ANTHROPIC_API_KEY")
    gigachat_client_id = os.getenv("KTTC_GIGACHAT_CLIENT_ID")
    gigachat_client_secret = os.getenv("KTTC_GIGACHAT_CLIENT_SECRET")
    gigachat_scope = os.getenv("KTTC_GIGACHAT_SCOPE", "GIGACHAT_API_PERS")

    # Check if we have GigaChat credentials
    has_gigachat = gigachat_client_id and gigachat_client_secret

    if not openai_key and not anthropic_key and not has_gigachat:
        print("❌ No API key found!")
        print("\nPlease set one of the following environment variables:")
        print("  • OPENAI_API_KEY (or KTTC_OPENAI_API_KEY)")
        print("  • ANTHROPIC_API_KEY (or KTTC_ANTHROPIC_API_KEY)")
        print("  • KTTC_GIGACHAT_CLIENT_ID and KTTC_GIGACHAT_CLIENT_SECRET")
        print("\nGet free credits:")
        print("  • OpenAI: https://platform.openai.com/signup ($5 free)")
        print("  • Anthropic: https://console.anthropic.com/ ($5 free)")
        print("  • GigaChat: https://developers.sber.ru/studio (FREE for individuals)")
        sys.exit(1)

    # Choose provider (priority: GigaChat > OpenAI > Anthropic)
    api_key: str | dict[str, str]
    if has_gigachat:
        provider = "gigachat"
        # gigachat_client_id and gigachat_client_secret are guaranteed to be str here
        # because has_gigachat checks they are not None
        assert gigachat_client_id is not None
        assert gigachat_client_secret is not None
        api_key = {
            "client_id": gigachat_client_id,
            "client_secret": gigachat_client_secret,
            "scope": gigachat_scope,
        }
        print("✓ Found GigaChat credentials")
    elif openai_key:
        provider = "openai"
        api_key = openai_key
        print("✓ Found OpenAI API key")
    else:
        provider = "anthropic"
        # anthropic_key is guaranteed to be str here because we checked all keys above
        assert anthropic_key is not None
        api_key = anthropic_key
        print("✓ Found Anthropic API key")

    # Run tests
    try:
        asyncio.run(test_translation_quality(api_key, provider))
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
