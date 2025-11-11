"""Error Detection Accuracy Validation Suite.

Validates KTTC error detection accuracy against gold-standard annotations.
Target: 97.1% detection rate (state-of-the-art as per 2025 research).

Provides:
- Precision, Recall, F1-score calculation
- Per-category accuracy metrics
- Confusion matrix analysis
- False positive/negative analysis
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from kttc.core.models import ErrorAnnotation

logger = logging.getLogger(__name__)


@dataclass
class ErrorMatch:
    """Represents a match between predicted and gold error."""

    gold_error: dict[str, Any]
    predicted_error: ErrorAnnotation | None
    match_score: float  # 0-1, how well they match


class ValidationMetrics(BaseModel):
    """Validation metrics for error detection accuracy.

    Measures how accurately KTTC detects translation errors
    compared to human gold-standard annotations.
    """

    true_positives: int = Field(default=0, description="Correctly detected errors")
    false_positives: int = Field(default=0, description="Incorrectly flagged as errors")
    false_negatives: int = Field(default=0, description="Missed errors")
    true_negatives: int = Field(default=0, description="Correctly identified as clean")

    @property
    def precision(self) -> float:
        """Precision: TP / (TP + FP).

        Measures: Of all errors KTTC flagged, how many were correct?
        """
        denominator = self.true_positives + self.false_positives
        return self.true_positives / denominator if denominator > 0 else 0.0

    @property
    def recall(self) -> float:
        """Recall (Detection Rate): TP / (TP + FN).

        Measures: Of all real errors, how many did KTTC find?
        Target: ≥ 97.1% (state-of-the-art)
        """
        denominator = self.true_positives + self.false_negatives
        return self.true_positives / denominator if denominator > 0 else 0.0

    @property
    def f1_score(self) -> float:
        """F1 Score: Harmonic mean of precision and recall.

        Target: ≥ 92% for production readiness
        """
        p = self.precision
        r = self.recall
        return 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0

    @property
    def accuracy(self) -> float:
        """Overall accuracy: (TP + TN) / (TP + FP + FN + TN)."""
        total = (
            self.true_positives + self.false_positives + self.false_negatives + self.true_negatives
        )
        return (self.true_positives + self.true_negatives) / total if total > 0 else 0.0

    def meets_target(self, target_recall: float = 0.971, target_precision: float = 0.90) -> bool:
        """Check if metrics meet target thresholds.

        Args:
            target_recall: Minimum acceptable recall (default: 97.1%)
            target_precision: Minimum acceptable precision (default: 90%)

        Returns:
            True if both targets met
        """
        return self.recall >= target_recall and self.precision >= target_precision


class GoldStandardDataset:
    """Gold-standard dataset with human-annotated errors.

    Format:
    {
        "source": "Source text",
        "translation": "Translation with errors",
        "errors": [
            {
                "category": "accuracy",
                "subcategory": "mistranslation",
                "severity": "major",
                "location": [10, 20],
                "description": "..."
            }
        ]
    }
    """

    def __init__(self, dataset_path: str | Path | None = None):
        """Initialize gold standard dataset.

        Args:
            dataset_path: Path to JSON file with gold annotations
        """
        self.dataset_path = Path(dataset_path) if dataset_path else None
        self.samples: list[dict[str, Any]] = []

        if self.dataset_path and self.dataset_path.exists():
            self.load()

    def load(self) -> None:
        """Load gold-standard annotations from file."""
        if not self.dataset_path or not self.dataset_path.exists():
            logger.warning("No gold-standard dataset found")
            return

        with open(self.dataset_path) as f:
            data = json.load(f)

        self.samples = data.get("samples", [])
        logger.info(f"Loaded {len(self.samples)} gold-standard samples")

    def add_sample(
        self,
        source: str,
        translation: str,
        source_lang: str,
        target_lang: str,
        errors: list[dict[str, Any]],
    ) -> None:
        """Add a gold-standard sample.

        Args:
            source: Source text
            translation: Translation
            source_lang: Source language code
            target_lang: Target language code
            errors: List of error annotations (gold standard)
        """
        self.samples.append(
            {
                "source": source,
                "translation": translation,
                "source_lang": source_lang,
                "target_lang": target_lang,
                "errors": errors,
            }
        )

    def save(self, output_path: str | Path) -> None:
        """Save gold-standard dataset to file.

        Args:
            output_path: Output JSON file path
        """
        data = {"samples": self.samples, "version": "1.0"}

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved {len(self.samples)} gold-standard samples to {output_path}")

    def __len__(self) -> int:
        """Get number of samples."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get sample by index."""
        return self.samples[idx]


class ErrorDetectionAccuracyTest:
    """Error detection accuracy validation test.

    Evaluates KTTC's error detection capability against gold-standard
    human annotations. Calculates precision, recall, F1-score.

    Target metrics (based on 2025 research):
    - Detection Rate (Recall): ≥ 97.1%
    - Precision: ≥ 90%
    - F1 Score: ≥ 92%

    Example:
        >>> test = ErrorDetectionAccuracyTest()
        >>> test.load_gold_standard("tests/data/gold_annotations.json")
        >>> metrics = await test.evaluate_accuracy(orchestrator)
        >>> print(f"Detection Rate: {metrics.recall * 100:.1f}%")
        >>> print(f"Precision: {metrics.precision * 100:.1f}%")
    """

    # Matching thresholds
    LOCATION_TOLERANCE = 5  # characters
    MIN_MATCH_SCORE = 0.7  # Minimum score to consider a match

    def __init__(self, gold_dataset: GoldStandardDataset | None = None):
        """Initialize accuracy test.

        Args:
            gold_dataset: Optional gold-standard dataset
        """
        self.gold_dataset = gold_dataset or GoldStandardDataset()
        self.results: list[dict[str, Any]] = []

    def load_gold_standard(self, dataset_path: str | Path) -> None:
        """Load gold-standard dataset.

        Args:
            dataset_path: Path to gold annotations JSON file
        """
        self.gold_dataset = GoldStandardDataset(dataset_path)
        self.gold_dataset.load()

    async def evaluate_accuracy(self, orchestrator: Any) -> ValidationMetrics:
        """Evaluate error detection accuracy.

        Args:
            orchestrator: KTTC AgentOrchestrator instance

        Returns:
            ValidationMetrics with precision, recall, F1-score

        Raises:
            ValueError: If no gold-standard data loaded
        """
        if len(self.gold_dataset) == 0:
            raise ValueError("No gold-standard data loaded. Call load_gold_standard() first.")

        logger.info(f"Evaluating error detection accuracy on {len(self.gold_dataset)} samples...")

        metrics = ValidationMetrics()

        for idx, sample in enumerate(self.gold_dataset.samples):
            # Create translation task
            from kttc.core.models import TranslationTask

            task = TranslationTask(
                source_text=sample["source"],
                translation=sample["translation"],
                source_lang=sample["source_lang"],
                target_lang=sample["target_lang"],
            )

            # Get KTTC predictions
            report = await orchestrator.evaluate(task)
            predicted_errors = report.errors

            # Get gold-standard errors
            gold_errors = sample["errors"]

            # Match errors and calculate metrics
            matches = self._match_errors(predicted_errors, gold_errors)

            # Update metrics
            tp, fp, fn = self._calculate_sample_metrics(matches, predicted_errors, gold_errors)
            metrics.true_positives += tp
            metrics.false_positives += fp
            metrics.false_negatives += fn

            # Store result for detailed analysis
            self.results.append(
                {
                    "sample_id": idx,
                    "source": sample["source"],
                    "translation": sample["translation"],
                    "gold_errors_count": len(gold_errors),
                    "predicted_errors_count": len(predicted_errors),
                    "matches": len(matches),
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                }
            )

            if (idx + 1) % 10 == 0:
                logger.info(
                    f"Progress: {idx + 1}/{len(self.gold_dataset)} "
                    f"(Recall: {metrics.recall:.1%}, Precision: {metrics.precision:.1%})"
                )

        logger.info("=" * 80)
        logger.info("VALIDATION RESULTS:")
        logger.info(f"  Precision:      {metrics.precision:.1%} (target: ≥ 90%)")
        logger.info(f"  Recall:         {metrics.recall:.1%} (target: ≥ 97.1%)")
        logger.info(f"  F1-Score:       {metrics.f1_score:.1%} (target: ≥ 92%)")
        logger.info(f"  Status:         {'✅ PASS' if metrics.meets_target() else '❌ FAIL'}")
        logger.info("=" * 80)

        return metrics

    def _match_errors(
        self, predicted: list[ErrorAnnotation], gold: list[dict[str, Any]]
    ) -> list[ErrorMatch]:
        """Match predicted errors with gold-standard errors.

        Uses fuzzy matching based on:
        - Category similarity
        - Location overlap
        - Severity match

        Args:
            predicted: List of KTTC predicted errors
            gold: List of gold-standard error annotations

        Returns:
            List of ErrorMatch objects
        """
        matches: list[ErrorMatch] = []

        # For each gold error, find best matching predicted error
        for gold_error in gold:
            best_match: ErrorAnnotation | None = None
            best_score = 0.0

            for pred_error in predicted:
                score = self._calculate_match_score(pred_error, gold_error)

                if score > best_score and score >= self.MIN_MATCH_SCORE:
                    best_score = score
                    best_match = pred_error

            if best_match:
                matches.append(
                    ErrorMatch(
                        gold_error=gold_error, predicted_error=best_match, match_score=best_score
                    )
                )

        return matches

    def _calculate_match_score(self, predicted: ErrorAnnotation, gold: dict[str, Any]) -> float:
        """Calculate match score between predicted and gold error.

        Args:
            predicted: Predicted error annotation
            gold: Gold-standard error annotation

        Returns:
            Match score (0-1), higher is better match
        """
        score = 0.0

        # Category match (40% weight)
        if predicted.category == gold.get("category"):
            score += 0.4

            # Subcategory match (bonus 20%)
            if predicted.subcategory == gold.get("subcategory"):
                score += 0.2

        # Severity match (20% weight)
        if predicted.severity.value == gold.get("severity"):
            score += 0.2

        # Location overlap (20% weight)
        if predicted.location and gold.get("location"):
            overlap = self._calculate_location_overlap(predicted.location, gold["location"])
            score += 0.2 * overlap

        return score

    def _calculate_location_overlap(
        self, loc1: tuple[int, int], loc2: list[int] | tuple[int, int]
    ) -> float:
        """Calculate location overlap between two error spans.

        Args:
            loc1: First location (start, end)
            loc2: Second location (start, end)

        Returns:
            Overlap score (0-1)
        """
        start1, end1 = loc1
        start2, end2 = loc2[0], loc2[1]

        # Calculate overlap
        overlap_start = max(start1, start2)
        overlap_end = min(end1, end2)

        if overlap_end <= overlap_start:
            # No overlap - check if close enough
            distance = min(abs(start1 - end2), abs(start2 - end1))
            if distance <= self.LOCATION_TOLERANCE:
                return 0.5  # Partial credit for nearby locations
            return 0.0

        # Calculate overlap percentage
        overlap_length = overlap_end - overlap_start
        total_length = max(end1 - start1, end2 - start2)

        return overlap_length / total_length if total_length > 0 else 0.0

    def _calculate_sample_metrics(
        self,
        matches: list[ErrorMatch],
        predicted: list[ErrorAnnotation],
        gold: list[dict[str, Any]],
    ) -> tuple[int, int, int]:
        """Calculate TP, FP, FN for a single sample.

        Args:
            matches: List of matched errors
            predicted: All predicted errors
            gold: All gold-standard errors

        Returns:
            Tuple of (true_positives, false_positives, false_negatives)
        """
        # True Positives: Matched errors
        tp = len(matches)

        # False Positives: Predicted but not in gold (no match)
        matched_predictions = [m.predicted_error for m in matches if m.predicted_error]
        fp = len(predicted) - len(matched_predictions)

        # False Negatives: Gold errors that weren't detected
        fn = len(gold) - tp

        return tp, fp, fn

    def export_detailed_report(self, output_path: str | Path) -> None:
        """Export detailed validation report.

        Args:
            output_path: Output file path (JSON)
        """
        report = {
            "summary": {
                "total_samples": len(self.results),
                "total_gold_errors": sum(r["gold_errors_count"] for r in self.results),
                "total_predicted_errors": sum(r["predicted_errors_count"] for r in self.results),
            },
            "samples": self.results,
        }

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Detailed validation report saved to {output_path}")

    def analyze_failure_patterns(self) -> dict[str, Any]:
        """Analyze common failure patterns.

        Returns:
            Dictionary with failure pattern analysis
        """
        total_fp = sum(r["fp"] for r in self.results)
        total_fn = sum(r["fn"] for r in self.results)

        # Find samples with highest FP/FN
        high_fp_samples = sorted(self.results, key=lambda x: x["fp"], reverse=True)[:5]
        high_fn_samples = sorted(self.results, key=lambda x: x["fn"], reverse=True)[:5]

        return {
            "total_false_positives": total_fp,
            "total_false_negatives": total_fn,
            "high_fp_samples": [
                {
                    "sample_id": s["sample_id"],
                    "source": s["source"][:100],
                    "false_positives": s["fp"],
                }
                for s in high_fp_samples
            ],
            "high_fn_samples": [
                {
                    "sample_id": s["sample_id"],
                    "source": s["source"][:100],
                    "false_negatives": s["fn"],
                }
                for s in high_fn_samples
            ],
        }


# Helper function to create sample gold-standard dataset
def create_sample_gold_dataset(
    output_path: str | Path = "tests/data/gold_annotations.json",
) -> None:
    """Create a sample gold-standard dataset for testing.

    Args:
        output_path: Output file path
    """
    dataset = GoldStandardDataset()

    # Sample 1: Mistranslation
    dataset.add_sample(
        source="The cat sat on the mat.",
        translation="El gato se sentó en la alfombra incorrecta.",  # Added "incorrecta" (wrong)
        source_lang="en",
        target_lang="es",
        errors=[
            {
                "category": "accuracy",
                "subcategory": "addition",
                "severity": "major",
                "location": [30, 40],
                "description": "Added word 'incorrecta' not in source",
            }
        ],
    )

    # Sample 2: Omission
    dataset.add_sample(
        source="The quick brown fox jumps over the lazy dog.",
        translation="El rápido zorro salta sobre el perro.",  # Missing "brown" and "lazy"
        source_lang="en",
        target_lang="es",
        errors=[
            {
                "category": "accuracy",
                "subcategory": "omission",
                "severity": "minor",
                "location": [10, 15],
                "description": "Missing adjective 'brown'",
            },
            {
                "category": "accuracy",
                "subcategory": "omission",
                "severity": "minor",
                "location": [35, 39],
                "description": "Missing adjective 'lazy'",
            },
        ],
    )

    # Sample 3: Grammar error
    dataset.add_sample(
        source="She is reading a book.",
        translation="Ella está leyendo un libros.",  # "libros" should be "libro" (singular)
        source_lang="en",
        target_lang="es",
        errors=[
            {
                "category": "fluency",
                "subcategory": "grammar",
                "severity": "major",
                "location": [23, 29],
                "description": "Number agreement error: 'un' (singular) with 'libros' (plural)",
            }
        ],
    )

    # Sample 4: Perfect translation (no errors)
    dataset.add_sample(
        source="Hello, world!",
        translation="¡Hola, mundo!",
        source_lang="en",
        target_lang="es",
        errors=[],  # No errors
    )

    # Save dataset
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    dataset.save(output_path)
    logger.info(f"Created sample gold-standard dataset with {len(dataset)} samples")
