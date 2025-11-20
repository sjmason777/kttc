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

"""Example: Testing lightweight metrics and rule-based error detection.

This script demonstrates the new CPU-based evaluation features:
- chrF, BLEU, TER metrics (no GPU needed)
- Rule-based error detection (numbers, punctuation, context)
- Beautiful CLI output

Usage:
    python3.11 examples/test_lightweight_metrics.py
"""

from kttc.cli.ui import (
    console,
    print_header,
    print_lightweight_metrics,
    print_rule_based_errors,
)
from kttc.evaluation import ErrorDetector, LightweightMetrics


def test_good_translation():
    """Test case 1: Good quality translation."""
    print_header("Test 1: Good Quality Translation", "Perfect translation with no errors")

    source = "The price is $100 for the product. Thank you!"
    translation = "Цена составляет $100 за продукт. Спасибо!"
    reference = "Цена составляет $100 за продукт. Спасибо!"

    # Calculate lightweight metrics
    metrics = LightweightMetrics()
    scores = metrics.evaluate(translation=translation, reference=reference, source=source)

    # Detect rule-based errors
    detector = ErrorDetector()
    errors = detector.detect_all_errors(source=source, translation=translation)
    rule_score = detector.calculate_rule_based_score(errors)

    # Display results
    console.print("[bold]Source:[/bold]", source)
    console.print("[bold]Translation:[/bold]", translation)
    console.print()

    print_lightweight_metrics(scores, verbose=True)
    print_rule_based_errors(errors, rule_score, verbose=True)


def test_missing_number():
    """Test case 2: Missing number error."""
    print_header("Test 2: Missing Number", "Translation missing critical information")

    source = "The total cost is $150 and tax is $15."
    translation = "Общая стоимость составляет $150."
    reference = "Общая стоимость составляет $150, а налог - $15."

    # Calculate lightweight metrics
    metrics = LightweightMetrics()
    scores = metrics.evaluate(translation=translation, reference=reference, source=source)

    # Detect rule-based errors
    detector = ErrorDetector()
    errors = detector.detect_all_errors(source=source, translation=translation)
    rule_score = detector.calculate_rule_based_score(errors)

    # Display results
    console.print("[bold]Source:[/bold]", source)
    console.print("[bold]Translation:[/bold]", translation)
    console.print()

    print_lightweight_metrics(scores, verbose=False)
    print_rule_based_errors(errors, rule_score, verbose=True)


def test_context_loss():
    """Test case 3: Context loss (negation missing)."""
    print_header("Test 3: Context Loss", "Negation removed from translation")

    source = "Do not open the door!"
    translation = "Откройте дверь."  # Missing negation!
    reference = "Не открывайте дверь!"

    # Calculate lightweight metrics
    metrics = LightweightMetrics()
    scores = metrics.evaluate(translation=translation, reference=reference, source=source)

    # Detect rule-based errors
    detector = ErrorDetector()
    errors = detector.detect_all_errors(source=source, translation=translation)
    rule_score = detector.calculate_rule_based_score(errors)

    # Display results
    console.print("[bold]Source:[/bold]", source)
    console.print("[bold]Translation:[/bold]", translation)
    console.print()

    print_lightweight_metrics(scores, verbose=False)
    print_rule_based_errors(errors, rule_score, verbose=True)


def test_punctuation_error():
    """Test case 4: Punctuation balance error."""
    print_header("Test 4: Punctuation Error", "Unbalanced quotes")

    source = 'He said "Hello" and left.'
    translation = "Он сказал «Привет и ушел."  # Missing closing quote
    reference = "Он сказал «Привет» и ушел."

    # Calculate lightweight metrics
    metrics = LightweightMetrics()
    scores = metrics.evaluate(translation=translation, reference=reference, source=source)

    # Detect rule-based errors
    detector = ErrorDetector()
    errors = detector.detect_all_errors(source=source, translation=translation)
    rule_score = detector.calculate_rule_based_score(errors)

    # Display results
    console.print("[bold]Source:[/bold]", source)
    console.print("[bold]Translation:[/bold]", translation)
    console.print()

    print_lightweight_metrics(scores, verbose=False)
    print_rule_based_errors(errors, rule_score, verbose=True)


def test_batch_evaluation():
    """Test case 5: Batch evaluation."""
    print_header("Test 5: Batch Evaluation", "Corpus-level metrics")

    translations = [
        "Привет мир",
        "Добрый день",
        "Спасибо за помощь",
    ]
    references = [
        "Привет мир",
        "Добрый день",
        "Спасибо за помощь",
    ]

    # Calculate corpus-level metrics
    metrics = LightweightMetrics()
    results = metrics.evaluate_batch(translations=translations, references=references)

    console.print("[bold]Corpus Statistics:[/bold]")
    console.print(f"  Sentences: {results['num_sentences']}")
    console.print(f"  chrF:      {results['chrf']:.2f}")
    console.print(f"  BLEU:      {results['bleu']:.2f}")
    console.print(f"  TER:       {results['ter']:.2f}")
    console.print(f"  Quality:   {results['quality_level'].title()}")
    console.print()


if __name__ == "__main__":
    console.print()
    console.print("[bold cyan]KTTC Lightweight Metrics Demo[/bold cyan]")
    console.print("[dim]CPU-based evaluation without GPU or heavy models[/dim]")
    console.print()

    # Run test cases
    test_good_translation()
    test_missing_number()
    test_context_loss()
    test_punctuation_error()
    test_batch_evaluation()

    console.print("[bold green]✓ All tests completed![/bold green]")
    console.print()
