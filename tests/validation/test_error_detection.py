"""Tests for error detection accuracy validation.

These tests verify that the error detection accuracy validation system works correctly.
"""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest

from kttc.core.models import ErrorAnnotation, ErrorSeverity, QAReport, TranslationTask
from tests.validation.error_detection_accuracy import (
    ErrorDetectionAccuracyTest,
    ErrorMatch,
    GoldStandardDataset,
    ValidationMetrics,
)


@pytest.fixture
def mock_orchestrator():
    """Create mock orchestrator for testing."""
    orchestrator = Mock()
    orchestrator.evaluate = AsyncMock()
    return orchestrator


class TestValidationMetrics:
    """Test ValidationMetrics model."""

    def test_metrics_creation(self):
        """Test creating validation metrics."""
        metrics = ValidationMetrics(
            true_positives=90, false_positives=10, false_negatives=5, true_negatives=95
        )

        assert metrics.true_positives == 90
        assert metrics.false_positives == 10
        assert metrics.false_negatives == 5

    def test_precision_calculation(self):
        """Test precision calculation."""
        metrics = ValidationMetrics(
            true_positives=90, false_positives=10, false_negatives=0, true_negatives=0
        )

        # Precision = TP / (TP + FP) = 90 / 100 = 0.9
        assert metrics.precision == 0.9

    def test_recall_calculation(self):
        """Test recall calculation."""
        metrics = ValidationMetrics(
            true_positives=90, false_positives=0, false_negatives=10, true_negatives=0
        )

        # Recall = TP / (TP + FN) = 90 / 100 = 0.9
        assert metrics.recall == 0.9

    def test_f1_score_calculation(self):
        """Test F1 score calculation."""
        metrics = ValidationMetrics(
            true_positives=90, false_positives=10, false_negatives=10, true_negatives=90
        )

        # Precision = 90/100 = 0.9, Recall = 90/100 = 0.9
        # F1 = 2 * (0.9 * 0.9) / (0.9 + 0.9) = 0.9
        assert metrics.f1_score == pytest.approx(0.9)

    def test_zero_division_handling(self):
        """Test metrics handle zero division."""
        metrics = ValidationMetrics(
            true_positives=0, false_positives=0, false_negatives=0, true_negatives=0
        )

        assert metrics.precision == 0.0
        assert metrics.recall == 0.0
        assert metrics.f1_score == 0.0

    def test_meets_target(self):
        """Test meets_target method."""
        # Meets target
        metrics = ValidationMetrics(
            true_positives=971, false_positives=29, false_negatives=29, true_negatives=0
        )

        assert metrics.recall >= 0.971
        assert metrics.precision >= 0.90
        assert metrics.meets_target()

    def test_does_not_meet_target(self):
        """Test does not meet target."""
        # Low recall
        metrics = ValidationMetrics(
            true_positives=80, false_positives=0, false_negatives=20, true_negatives=0
        )

        assert metrics.recall < 0.971
        assert not metrics.meets_target()


class TestGoldStandardDataset:
    """Test GoldStandardDataset functionality."""

    def test_initialization_empty(self):
        """Test initializing empty dataset."""
        dataset = GoldStandardDataset()

        assert len(dataset.samples) == 0
        assert len(dataset) == 0

    def test_add_sample(self):
        """Test adding samples to dataset."""
        dataset = GoldStandardDataset()

        dataset.add_sample(
            source="Hello",
            translation="Hola",
            source_lang="en",
            target_lang="es",
            errors=[],
        )

        assert len(dataset) == 1

    def test_getitem(self):
        """Test getting sample by index."""
        dataset = GoldStandardDataset()

        dataset.add_sample(
            source="Hello",
            translation="Hola",
            source_lang="en",
            target_lang="es",
            errors=[{"category": "fluency", "severity": "minor"}],
        )

        sample = dataset[0]
        assert sample["source"] == "Hello"
        assert sample["translation"] == "Hola"
        assert len(sample["errors"]) == 1

    def test_save_and_load(self):
        """Test saving and loading dataset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir) / "gold_standard.json"

            dataset = GoldStandardDataset()
            dataset.add_sample(
                source="Test", translation="Prueba", source_lang="en", target_lang="es", errors=[]
            )

            # Save
            dataset.save(dataset_path)
            assert dataset_path.exists()

            # Load
            loaded = GoldStandardDataset(dataset_path)
            assert len(loaded) == 1
            assert loaded[0]["source"] == "Test"

    def test_load_nonexistent_file(self):
        """Test loading nonexistent file."""
        dataset = GoldStandardDataset(dataset_path="nonexistent.json")
        assert len(dataset) == 0


class TestErrorMatch:
    """Test ErrorMatch dataclass."""

    def test_error_match_creation(self):
        """Test creating error match."""
        gold_error = {"category": "accuracy", "severity": "major"}
        predicted = ErrorAnnotation(
            category="accuracy",
            subcategory="mistranslation",
            severity=ErrorSeverity.MAJOR,
            description="Error",
            location=(0, 5),
        )

        match = ErrorMatch(gold_error=gold_error, predicted_error=predicted, match_score=0.95)

        assert match.gold_error == gold_error
        assert match.predicted_error == predicted
        assert match.match_score == 0.95


class TestErrorDetectionAccuracyTest:
    """Test ErrorDetectionAccuracyTest functionality."""

    def test_initialization(self):
        """Test test initialization."""
        dataset = GoldStandardDataset()
        dataset.add_sample(
            source="Test", translation="Prueba", source_lang="en", target_lang="es", errors=[]
        )
        test = ErrorDetectionAccuracyTest(gold_dataset=dataset)

        assert len(test.gold_dataset) == 1

    def test_initialization_with_path(self):
        """Test initialization by loading dataset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir) / "gold.json"

            # Create dataset
            dataset = GoldStandardDataset()
            dataset.add_sample(
                source="Test", translation="Prueba", source_lang="en", target_lang="es", errors=[]
            )
            dataset.save(dataset_path)

            # Initialize test with path
            test = ErrorDetectionAccuracyTest()
            test.load_gold_standard(dataset_path)

            assert len(test.gold_dataset) == 1

    @pytest.mark.asyncio
    async def test_evaluate_accuracy_perfect_detection(self, mock_orchestrator):
        """Test evaluation with perfect detection."""
        dataset = GoldStandardDataset()
        dataset.add_sample(
            source="Test",
            translation="Prueba",
            source_lang="en",
            target_lang="es",
            errors=[{"category": "accuracy", "severity": "major", "location": [0, 5]}],
        )

        # Mock perfect detection
        mock_orchestrator.evaluate.return_value = QAReport(
            task=TranslationTask(
                source_text="Test", translation="Prueba", source_lang="en", target_lang="es"
            ),
            mqm_score=75.0,
            errors=[
                ErrorAnnotation(
                    category="accuracy",
                    subcategory="mistranslation",
                    severity=ErrorSeverity.MAJOR,
                    description="Error",
                    location=[0, 5],
                )
            ],
            status="fail",
        )

        test = ErrorDetectionAccuracyTest(gold_dataset=dataset)
        metrics = await test.evaluate_accuracy(mock_orchestrator)

        # Perfect detection: 1 TP, 0 FP, 0 FN
        assert metrics.true_positives >= 1
        assert metrics.precision > 0.8  # Should be high
        assert metrics.recall > 0.8  # Should be high

    @pytest.mark.asyncio
    async def test_evaluate_accuracy_with_errors(self, mock_orchestrator):
        """Test evaluation with mixed results."""
        dataset = GoldStandardDataset()

        # Add sample with error
        dataset.add_sample(
            source="Bad translation",
            translation="Mala traducción",
            source_lang="en",
            target_lang="es",
            errors=[{"category": "accuracy", "severity": "major", "location": [0, 5]}],
        )

        # Add perfect sample
        dataset.add_sample(
            source="Good translation",
            translation="Buena traducción",
            source_lang="en",
            target_lang="es",
            errors=[],
        )

        # Mock: detect first error, miss nothing on second
        mock_orchestrator.evaluate.side_effect = [
            QAReport(
                task=TranslationTask(
                    source_text="test", translation="test", source_lang="en", target_lang="es"
                ),
                mqm_score=70.0,
                errors=[
                    ErrorAnnotation(
                        category="accuracy",
                        subcategory="mistranslation",
                        severity=ErrorSeverity.MAJOR,
                        description="Error",
                        location=(0, 5),
                    )
                ],
                status="fail",
            ),
            QAReport(
                task=TranslationTask(
                    source_text="test", translation="test", source_lang="en", target_lang="es"
                ),
                mqm_score=100.0,
                errors=[],
                status="pass",
            ),
        ]

        test = ErrorDetectionAccuracyTest(gold_dataset=dataset)
        metrics = await test.evaluate_accuracy(mock_orchestrator)

        # Should have detected the error correctly
        assert metrics.true_positives >= 1

    def test_calculate_match_score(self):
        """Test match score calculation."""
        test = ErrorDetectionAccuracyTest(gold_dataset=GoldStandardDataset())

        gold = {"category": "accuracy", "severity": "major", "location": [10, 20]}

        predicted = ErrorAnnotation(
            category="accuracy",
            subcategory="mistranslation",
            severity=ErrorSeverity.MAJOR,
            description="Test",
            location=[10, 20],
        )

        score = test._calculate_match_score(predicted, gold)

        # Same category, severity, and location -> high score
        assert score > 0.5

    def test_calculate_location_overlap(self):
        """Test location overlap calculation."""
        test = ErrorDetectionAccuracyTest(gold_dataset=GoldStandardDataset())

        # Exact overlap
        overlap = test._calculate_location_overlap([10, 20], [10, 20])
        assert overlap == 1.0

        # No overlap
        overlap = test._calculate_location_overlap([10, 20], [30, 40])
        assert overlap == 0.0

        # Partial overlap
        overlap = test._calculate_location_overlap([10, 20], [15, 25])
        assert 0.0 < overlap < 1.0


@pytest.mark.asyncio
@pytest.mark.slow
async def test_full_validation_pipeline(mock_orchestrator):
    """Test complete validation pipeline."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create gold standard dataset
        dataset = GoldStandardDataset()

        for i in range(5):
            has_error = i % 2 == 0
            dataset.add_sample(
                source=f"Test {i}",
                translation=f"Prueba {i}",
                source_lang="en",
                target_lang="es",
                errors=(
                    [{"category": "accuracy", "severity": "major", "location": [0, 5]}]
                    if has_error
                    else []
                ),
            )

        dataset_path = Path(tmpdir) / "gold.json"
        dataset.save(dataset_path)

        # Mock realistic detection
        def mock_evaluate(task):
            idx = int(task.source_text.split()[-1])
            has_error = idx % 2 == 0

            if has_error:
                return QAReport(
                    task=task,
                    mqm_score=75.0,
                    errors=[
                        ErrorAnnotation(
                            category="accuracy",
                            subcategory="mistranslation",
                            severity=ErrorSeverity.MAJOR,
                            description="Error",
                            location=(0, 5),
                        )
                    ],
                    status="fail",
                )
            else:
                return QAReport(task=task, mqm_score=100.0, errors=[], status="pass")

        mock_orchestrator.evaluate.side_effect = mock_evaluate

        # Run validation
        test = ErrorDetectionAccuracyTest()
        test.load_gold_standard(dataset_path)

        metrics = await test.evaluate_accuracy(mock_orchestrator)

        # Should have perfect or near-perfect detection
        assert metrics.precision >= 0.8
        assert metrics.recall >= 0.8
