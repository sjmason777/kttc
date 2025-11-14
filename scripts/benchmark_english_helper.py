#!/usr/bin/env python3.11
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

"""Benchmark EnglishLanguageHelper for translation QA validation.

This script validates that EnglishLanguageHelper with LanguageTool provides
real value in translation quality assurance.

Usage:
    python3.11 scripts/benchmark_english_helper.py

Output:
    - Console metrics (precision, recall, F1, false positive rate)
    - Detailed report in docs/benchmarks/english-helper-validation.md
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from kttc.helpers.english import EnglishLanguageHelper


def load_test_cases() -> list[dict[str, Any]]:
    """Load test cases for benchmark.

    Returns:
        List of test cases with source, translation, and expected errors
    """
    return [
        # Correct translations (no errors)
        {
            "source": "Bonjour le monde",
            "translation": "Hello world",
            "expected_errors": 0,
            "category": "simple",
        },
        {
            "source": "Comment allez-vous?",
            "translation": "How are you?",
            "expected_errors": 0,
            "category": "simple",
        },
        {
            "source": "Le chat dort sur le tapis",
            "translation": "The cat sleeps on the carpet",
            "expected_errors": 0,
            "category": "simple",
        },
        {
            "source": "Je vais à l'école",
            "translation": "I go to school",
            "expected_errors": 0,
            "category": "simple",
        },
        {
            "source": "Il fait beau aujourd'hui",
            "translation": "The weather is nice today",
            "expected_errors": 0,
            "category": "simple",
        },
        # Subject-verb agreement errors
        {
            "source": "Il va à l'école",
            "translation": "He go to school",
            "expected_errors": 1,
            "category": "grammar_subject_verb",
            "error_type": "SUBJECT_VERB",
        },
        {
            "source": "Elle aime lire",
            "translation": "She like to read",
            "expected_errors": 1,
            "category": "grammar_subject_verb",
            "error_type": "SUBJECT_VERB",
        },
        {
            "source": "Ils jouent au football",
            "translation": "They plays football",
            "expected_errors": 1,
            "category": "grammar_subject_verb",
            "error_type": "SUBJECT_VERB",
        },
        # Article errors (a vs an)
        {
            "source": "Je vois un éléphant",
            "translation": "I see a elephant",
            "expected_errors": 1,
            "category": "grammar_article",
            "error_type": "ARTICLE",
        },
        {
            "source": "C'est une pomme",
            "translation": "It is an apple",
            "expected_errors": 0,  # Correct
            "category": "grammar_article",
        },
        {
            "source": "Il y a un homme",
            "translation": "There is an man",
            "expected_errors": 1,
            "category": "grammar_article",
            "error_type": "ARTICLE",
        },
        # Spelling errors
        {
            "source": "Le chien est grand",
            "translation": "The dog is beutiful",  # beautiful
            "expected_errors": 1,
            "category": "spelling",
            "error_type": "SPELLING",
        },
        {
            "source": "Je mange une pomme",
            "translation": "I eat an aple",  # apple
            "expected_errors": 1,
            "category": "spelling",
            "error_type": "SPELLING",
        },
        # Tense errors
        {
            "source": "Je suis allé à Paris hier",
            "translation": "I go to Paris yesterday",
            "expected_errors": 1,
            "category": "grammar_tense",
            "error_type": "TENSE",
        },
        {
            "source": "Il mangera demain",
            "translation": "He eats tomorrow",
            "expected_errors": 1,
            "category": "grammar_tense",
            "error_type": "TENSE",
        },
        # Multiple errors
        {
            "source": "Il va à l'école avec un ami",
            "translation": "He go to school with a friend",
            "expected_errors": 1,  # He go -> goes
            "category": "grammar_multiple",
            "error_type": "SUBJECT_VERB",
        },
        # Technical translations
        {
            "source": "L'algorithme optimise les performances",
            "translation": "The algorithm optimizes performance",
            "expected_errors": 0,
            "category": "technical",
        },
        {
            "source": "Le serveur traite les requêtes",
            "translation": "The server process requests",
            "expected_errors": 1,  # process -> processes
            "category": "technical",
            "error_type": "SUBJECT_VERB",
        },
        # Marketing translations
        {
            "source": "Achetez maintenant et économisez",
            "translation": "Buy now and save money",
            "expected_errors": 0,
            "category": "marketing",
        },
        {
            "source": "Offre limitée dans le temps",
            "translation": "Limited time offer",
            "expected_errors": 0,
            "category": "marketing",
        },
        # Complex sentences
        {
            "source": "Quand il pleut, je reste à la maison",
            "translation": "When it rains, I stay at home",
            "expected_errors": 0,
            "category": "complex",
        },
        {
            "source": "Si j'avais su, je serais venu",
            "translation": "If I had known, I would have come",
            "expected_errors": 0,
            "category": "complex",
        },
        # Edge cases
        {
            "source": "OK",
            "translation": "OK",
            "expected_errors": 0,
            "category": "edge_case",
        },
        {
            "source": "2025",
            "translation": "2025",
            "expected_errors": 0,
            "category": "edge_case",
        },
    ]


def benchmark_helper() -> dict[str, Any]:
    """Benchmark EnglishLanguageHelper.

    Returns:
        Dictionary with benchmark results and metrics
    """
    print("=" * 80)
    print("ENGLISH LANGUAGE HELPER BENCHMARK")
    print("=" * 80)
    print()

    # Initialize helper
    print("Initializing EnglishLanguageHelper...")
    helper = EnglishLanguageHelper()

    if not helper.is_available():
        print("ERROR: EnglishLanguageHelper not available (spaCy required)")
        sys.exit(1)

    if not helper._lt_available:
        print("WARNING: LanguageTool not available")
        print("Install with: pip install language-tool-python")
        sys.exit(1)

    print("✓ Initialized successfully")
    print()

    # Load test cases
    test_cases = load_test_cases()
    print(f"Loaded {len(test_cases)} test cases")
    print()

    # Run benchmark
    results: dict[str, Any] = {
        "total": 0,
        "true_positives": 0,  # Found error when expected
        "false_positives": 0,  # Found error when not expected
        "true_negatives": 0,  # No error when not expected
        "false_negatives": 0,  # No error when expected
        "errors_found": [],
        "execution_times": [],
    }

    print("Running benchmark...")
    print("-" * 80)

    for i, test_case in enumerate(test_cases, 1):
        translation = test_case["translation"]
        expected_errors = test_case["expected_errors"]
        category = test_case["category"]

        # Measure execution time
        start_time = time.time()
        errors = helper.check_grammar(translation)
        execution_time = time.time() - start_time

        results["execution_times"].append(execution_time)

        # Calculate metrics
        has_errors = len(errors) > 0

        if expected_errors > 0:
            if has_errors:
                results["true_positives"] += 1
                status = "✓ TP"
            else:
                results["false_negatives"] += 1
                status = "✗ FN"
        else:
            if has_errors:
                results["false_positives"] += 1
                status = "⚠ FP"
            else:
                results["true_negatives"] += 1
                status = "✓ TN"

        results["total"] += 1
        results["errors_found"].append(
            {
                "case": test_case,
                "errors": [
                    {
                        "category": e.category,
                        "subcategory": e.subcategory,
                        "severity": e.severity.name,
                        "description": e.description,
                        "location": e.location,
                    }
                    for e in errors
                ],
                "execution_time": execution_time,
            }
        )

        # Print progress
        print(
            f"[{i:2d}/{len(test_cases)}] {status} | "
            f"{category:20s} | "
            f"Expected: {expected_errors} | "
            f"Found: {len(errors)} | "
            f"{execution_time*1000:.1f}ms"
        )

    print("-" * 80)
    print()

    # Calculate final metrics
    tp: int = results["true_positives"]
    fp: int = results["false_positives"]
    tn: int = results["true_negatives"]
    fn: int = results["false_negatives"]
    total: int = results["total"]

    # Avoid division by zero
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / total if total > 0 else 0
    false_positive_rate = fp / total if total > 0 else 0

    execution_times: list[float] = results["execution_times"]
    avg_time = sum(execution_times) / len(execution_times) if execution_times else 0.0

    # Print metrics
    print("RESULTS")
    print("=" * 80)
    print()
    print(f"Total test cases:       {total}")
    print(f"True Positives (TP):    {tp}")
    print(f"False Positives (FP):   {fp}")
    print(f"True Negatives (TN):    {tn}")
    print(f"False Negatives (FN):   {fn}")
    print()
    print(f"Precision:              {precision:.2%}")
    print(f"Recall:                 {recall:.2%}")
    print(f"F1 Score:               {f1:.2%}")
    print(f"Accuracy:               {accuracy:.2%}")
    print(f"False Positive Rate:    {false_positive_rate:.2%}")
    print()
    print(f"Avg Execution Time:     {avg_time*1000:.2f}ms")
    print()

    # Decision matrix
    print("DECISION MATRIX")
    print("=" * 80)
    print()

    if precision >= 0.80 and recall >= 0.70:
        decision = "✅ PROCEED - Create EnglishFluencyAgent"
        recommendation = "Excellent performance! Proceed with agent creation."
    elif precision >= 0.70 and recall >= 0.60:
        decision = "⚠️ IMPROVE - Enhance filtering, then re-test"
        recommendation = "Good performance, but needs improvement. Tune filters and re-test."
    else:
        decision = "❌ KEEP HELPER ONLY - Don't create specialized agent"
        recommendation = "Performance below threshold. Use helper-only mode."

    print(f"Decision:      {decision}")
    print(f"Recommendation: {recommendation}")
    print()

    # Add to results
    results["metrics"] = {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "accuracy": accuracy,
        "false_positive_rate": false_positive_rate,
        "avg_execution_time": avg_time,
    }
    results["decision"] = decision
    results["recommendation"] = recommendation

    return results


def main() -> None:
    """Run benchmark and save results."""
    # Run benchmark
    results = benchmark_helper()

    # Save detailed results
    output_dir = Path(__file__).parent.parent / "docs" / "benchmarks"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "english-helper-benchmark.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Detailed results saved to: {output_file}")
    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
