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

"""Benchmark RussianLanguageHelper for translation QA validation.

This script validates that RussianLanguageHelper with MAWO NLP libraries
provides real value in translation quality assurance for Russian translations.

Usage:
    python3.11 scripts/benchmark_russian_helper.py

Output:
    - Console metrics (precision, recall, F1, false positive rate)
    - Detailed report in docs/benchmarks/russian-helper-validation.md
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from kttc.helpers.russian import RussianLanguageHelper


def load_test_cases() -> list[dict[str, Any]]:
    """Load test cases for benchmark.

    Returns:
        List of test cases with source, translation, and expected errors
    """
    return [
        # Correct translations (no errors)
        {
            "source": "Hello world",
            "translation": "Привет мир",
            "expected_errors": 0,
            "category": "simple",
        },
        {
            "source": "How are you?",
            "translation": "Как дела?",
            "expected_errors": 0,
            "category": "simple",
        },
        {
            "source": "The cat sleeps on the carpet",
            "translation": "Кот спит на ковре",
            "expected_errors": 0,
            "category": "simple",
        },
        # Correct adjective-noun agreements
        {
            "source": "Big house",
            "translation": "Большой дом",
            "expected_errors": 0,
            "category": "adjective_correct",
        },
        {
            "source": "Beautiful girl",
            "translation": "Красивая девушка",
            "expected_errors": 0,
            "category": "adjective_correct",
        },
        {
            "source": "Fast car",
            "translation": "Быстрая машина",
            "expected_errors": 0,
            "category": "adjective_correct",
        },
        {
            "source": "Red apple",
            "translation": "Красное яблоко",
            "expected_errors": 0,
            "category": "adjective_correct",
        },
        # Gender agreement errors
        {
            "source": "Fast fox",
            "translation": "Быстрый лиса",  # Should be быстрая лиса
            "expected_errors": 1,
            "category": "gender_error",
            "error_type": "GENDER",
        },
        {
            "source": "Beautiful house",
            "translation": "Красивая дом",  # Should be красивый дом
            "expected_errors": 1,
            "category": "gender_error",
            "error_type": "GENDER",
        },
        {
            "source": "Big girl",
            "translation": "Большой девушка",  # Should be большая девушка
            "expected_errors": 1,
            "category": "gender_error",
            "error_type": "GENDER",
        },
        # Correct preposition cases
        {
            "source": "In the house",
            "translation": "В доме",  # Prepositional case
            "expected_errors": 0,
            "category": "preposition_correct",
        },
        {
            "source": "To the city",
            "translation": "В город",  # Accusative case
            "expected_errors": 0,
            "category": "preposition_correct",
        },
        {
            "source": "From the table",
            "translation": "Со стола",  # Genitive case
            "expected_errors": 0,
            "category": "preposition_correct",
        },
        # Case agreement with prepositions
        {
            "source": "About the book",
            "translation": "О книга",  # Should be "о книге" (prepositional)
            "expected_errors": 1,
            "category": "case_error",
            "error_type": "CASE",
        },
        {
            "source": "Without friends",
            "translation": "Без друг",  # Should be "без друзей" (genitive)
            "expected_errors": 1,
            "category": "case_error",
            "error_type": "CASE",
        },
        # Number agreement
        {
            "source": "Two big houses",
            "translation": "Два больших дома",
            "expected_errors": 0,
            "category": "number_correct",
        },
        {
            "source": "One beautiful car",
            "translation": "Одна красивая машина",
            "expected_errors": 0,
            "category": "number_correct",
        },
        # Technical translations
        {
            "source": "The algorithm processes data",
            "translation": "Алгоритм обрабатывает данные",
            "expected_errors": 0,
            "category": "technical",
        },
        {
            "source": "Fast server",
            "translation": "Быстрый сервер",
            "expected_errors": 0,
            "category": "technical",
        },
        # Marketing translations
        {
            "source": "Buy now",
            "translation": "Купите сейчас",
            "expected_errors": 0,
            "category": "marketing",
        },
        {
            "source": "Best offer",
            "translation": "Лучшее предложение",
            "expected_errors": 0,
            "category": "marketing",
        },
        # Complex sentences
        {
            "source": "The beautiful girl reads a big book",
            "translation": "Красивая девушка читает большую книгу",
            "expected_errors": 0,
            "category": "complex",
        },
        {
            "source": "Fast cars drive on new roads",
            "translation": "Быстрые машины едут по новым дорогам",
            "expected_errors": 0,
            "category": "complex",
        },
        # Edge cases
        {
            "source": "OK",
            "translation": "Хорошо",
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
    """Benchmark RussianLanguageHelper.

    Returns:
        Dictionary with benchmark results and metrics
    """
    print("=" * 80)
    print("RUSSIAN LANGUAGE HELPER BENCHMARK")
    print("=" * 80)
    print()

    # Initialize helper
    print("Initializing RussianLanguageHelper...")
    helper = RussianLanguageHelper()

    if not helper.is_available():
        print("ERROR: RussianLanguageHelper not available (MAWO NLP required)")
        print("Install with: pip install mawo-pymorphy3 mawo-razdel mawo-natasha")
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
            f"{category:22s} | "
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
        decision = "✅ PROCEED - Create RussianFluencyAgent"
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

    output_file = output_dir / "russian-helper-benchmark.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Detailed results saved to: {output_file}")
    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
