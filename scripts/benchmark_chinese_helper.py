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

"""Benchmark ChineseLanguageHelper for translation QA validation.

This script validates that ChineseLanguageHelper with HanLP provides
real value in translation quality assurance for Chinese translations.

Usage:
    python3.11 scripts/benchmark_chinese_helper.py

Output:
    - Console metrics (precision, recall, F1, false positive rate)
    - Detailed report in docs/benchmarks/chinese-helper-validation.md
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from kttc.helpers.chinese import ChineseLanguageHelper


def load_test_cases() -> list[dict[str, Any]]:
    """Load test cases for benchmark.

    Returns:
        List of test cases with source, translation, and expected errors
    """
    return [
        # Correct translations (no errors)
        {
            "source": "Hello world",
            "translation": "你好世界",
            "expected_errors": 0,
            "category": "simple",
        },
        {
            "source": "How are you?",
            "translation": "你好吗",
            "expected_errors": 0,
            "category": "simple",
        },
        {
            "source": "The cat sleeps on the carpet",
            "translation": "猫在地毯上睡觉",
            "expected_errors": 0,
            "category": "simple",
        },
        # Correct measure words
        {
            "source": "Three books",
            "translation": "三本书",
            "expected_errors": 0,
            "category": "measure_word_correct",
        },
        {
            "source": "Two cars",
            "translation": "两辆车",
            "expected_errors": 0,
            "category": "measure_word_correct",
        },
        {
            "source": "One cat",
            "translation": "一只猫",
            "expected_errors": 0,
            "category": "measure_word_correct",
        },
        {
            "source": "Five apples",
            "translation": "五个苹果",
            "expected_errors": 0,
            "category": "measure_word_correct",
        },
        {
            "source": "A cup of water",
            "translation": "一杯水",
            "expected_errors": 0,
            "category": "measure_word_correct",
        },
        # Incorrect measure words
        {
            "source": "Three books",
            "translation": "三个书",  # Should be 三本书
            "expected_errors": 1,
            "category": "measure_word_error",
            "error_type": "MEASURE_WORD",
        },
        {
            "source": "One car",
            "translation": "一本车",  # Should be 一辆车
            "expected_errors": 1,
            "category": "measure_word_error",
            "error_type": "MEASURE_WORD",
        },
        {
            "source": "Two cats",
            "translation": "两条猫",  # Should be 两只猫
            "expected_errors": 1,
            "category": "measure_word_error",
            "error_type": "MEASURE_WORD",
        },
        {
            "source": "A piece of paper",
            "translation": "一个纸",  # Should be 一张纸
            "expected_errors": 1,
            "category": "measure_word_error",
            "error_type": "MEASURE_WORD",
        },
        # Correct aspect particles
        {
            "source": "I bought books",
            "translation": "我买了书",
            "expected_errors": 0,
            "category": "particle_correct",
        },
        {
            "source": "I have been to Beijing",
            "translation": "我去过北京",
            "expected_errors": 0,
            "category": "particle_correct",
        },
        {
            "source": "He ate dinner",
            "translation": "他吃了晚饭",
            "expected_errors": 0,
            "category": "particle_correct",
        },
        # Complex sentences with multiple features
        {
            "source": "I bought three books yesterday",
            "translation": "我昨天买了三本书",
            "expected_errors": 0,
            "category": "complex_correct",
        },
        {
            "source": "She has two cats at home",
            "translation": "她家里有两只猫",
            "expected_errors": 0,
            "category": "complex_correct",
        },
        # Technical translations
        {
            "source": "The algorithm processes data",
            "translation": "算法处理数据",
            "expected_errors": 0,
            "category": "technical",
        },
        {
            "source": "Three servers are running",
            "translation": "三台服务器正在运行",
            "expected_errors": 0,
            "category": "technical",
        },
        # Marketing translations
        {
            "source": "Buy now and save",
            "translation": "现在购买可省钱",
            "expected_errors": 0,
            "category": "marketing",
        },
        {
            "source": "Limited time offer",
            "translation": "限时优惠",
            "expected_errors": 0,
            "category": "marketing",
        },
        # Mixed content (Chinese + English)
        {
            "source": "I use Python programming",
            "translation": "我使用Python编程",
            "expected_errors": 0,
            "category": "mixed",
        },
        # Edge cases
        {
            "source": "2025",
            "translation": "2025",
            "expected_errors": 0,
            "category": "edge_case",
        },
        {
            "source": "OK",
            "translation": "好的",
            "expected_errors": 0,
            "category": "edge_case",
        },
    ]


def benchmark_helper() -> dict[str, Any]:
    """Benchmark ChineseLanguageHelper.

    Returns:
        Dictionary with benchmark results and metrics
    """
    print("=" * 80)
    print("CHINESE LANGUAGE HELPER BENCHMARK")
    print("=" * 80)
    print()

    # Initialize helper
    print("Initializing ChineseLanguageHelper...")
    helper = ChineseLanguageHelper()

    if not helper.is_available():
        print("ERROR: ChineseLanguageHelper not available (jieba or spaCy required)")
        sys.exit(1)

    if not helper._hanlp_available:
        print("WARNING: HanLP not available")
        print("Install with: pip install hanlp")
        print("Grammar checking will be limited")
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
        decision = "✅ PROCEED - Create ChineseFluencyAgent"
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

    output_file = output_dir / "chinese-helper-benchmark.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Detailed results saved to: {output_file}")
    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
