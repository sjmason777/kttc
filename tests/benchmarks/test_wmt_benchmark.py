"""Tests for WMT benchmarking functionality.

These tests verify that WMT benchmark integration works correctly.
"""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from kttc.core.models import QAReport, TranslationTask
from tests.benchmarks.wmt_benchmark import BenchmarkResult, WMTBenchmark


@pytest.fixture
def mock_orchestrator():
    """Create mock orchestrator for testing."""
    orchestrator = Mock()
    orchestrator.evaluate = AsyncMock()
    return orchestrator


class TestBenchmarkResult:
    """Test BenchmarkResult model."""

    def test_benchmark_result_creation(self):
        """Test creating benchmark result."""
        result = BenchmarkResult(
            dataset_name="test-dataset",
            system_name="kttc-test",
            sample_size=10,
            avg_mqm_score=85.5,
            std_mqm_score=5.2,
            avg_processing_time=1.5,
        )

        assert result.dataset_name == "test-dataset"
        assert result.system_name == "kttc-test"
        assert result.sample_size == 10
        assert result.avg_mqm_score == 85.5

    def test_benchmark_result_with_optional_fields(self):
        """Test benchmark result with optional fields."""
        result = BenchmarkResult(
            dataset_name="test",
            system_name="kttc",
            sample_size=100,
            avg_mqm_score=85.0,
            std_mqm_score=5.0,
            avg_comet_score=0.85,
            avg_kiwi_score=0.82,
            avg_processing_time=1.0,
            error_counts={"critical": 5, "major": 10, "minor": 20},
        )

        assert result.avg_comet_score == 0.85
        assert result.avg_kiwi_score == 0.82
        assert result.error_counts["critical"] == 5


class TestWMTBenchmark:
    """Test WMTBenchmark functionality."""

    def test_initialization(self):
        """Test benchmark initialization."""
        benchmark = WMTBenchmark(results_dir="test_results")

        assert benchmark.results_dir == Path("test_results")
        assert len(benchmark.results) == 0

    def test_available_datasets(self):
        """Test that available datasets are defined."""
        benchmark = WMTBenchmark()

        assert "flores-200" in benchmark.AVAILABLE_DATASETS
        assert "wmt23" in benchmark.AVAILABLE_DATASETS
        assert "wmt22" in benchmark.AVAILABLE_DATASETS

    @pytest.mark.asyncio
    async def test_run_benchmark_with_mock_dataset(self, mock_orchestrator):
        """Test running benchmark with mocked dataset."""
        benchmark = WMTBenchmark()

        # Mock the dataset loading
        mock_dataset = [
            {
                "id": 1,
                "sentence_eng_Latn": "Hello world",
                "sentence_spa_Latn": "Hola mundo",
            }
        ]

        # Mock orchestrator evaluation
        mock_orchestrator.evaluate.return_value = QAReport(
            task=TranslationTask(
                source_text="Hello world",
                translation="Hola mundo",
                source_lang="en",
                target_lang="es",
            ),
            mqm_score=95.0,
            status="pass",
            errors=[],
        )

        # Mock _load_dataset
        with patch.object(benchmark, "_load_dataset", return_value=mock_dataset):
            result = await benchmark.run_benchmark(
                orchestrator=mock_orchestrator,
                dataset_name="flores-200",
                language_pair="eng_Latn-spa_Latn",
                sample_size=1,
            )

            assert result.sample_size == 1
            assert result.avg_mqm_score == 95.0

    @pytest.mark.asyncio
    async def test_run_benchmark_handles_errors(self, mock_orchestrator):
        """Test benchmark handles evaluation errors gracefully."""
        benchmark = WMTBenchmark()

        mock_dataset = [{"sentence_eng_Latn": "Test", "sentence_spa_Latn": "Prueba"}]

        # Mock orchestrator to raise error
        mock_orchestrator.evaluate.side_effect = Exception("Evaluation failed")

        # Should handle error gracefully and continue (not raise exception)
        with patch.object(benchmark, "_load_dataset", return_value=mock_dataset):
            result = await benchmark.run_benchmark(
                orchestrator=mock_orchestrator,
                dataset_name="flores-200",
                language_pair="eng_Latn-spa_Latn",
                sample_size=1,
            )

            # Result should be returned with empty or zero metrics
            assert result.sample_size == 1

    def test_parse_language_pair_flores(self):
        """Test parsing FLORES language pairs."""
        benchmark = WMTBenchmark()

        source, target = benchmark._parse_language_pair("eng_Latn-spa_Latn")
        assert source == "en"
        assert target == "sp"  # Takes first 2 chars from "spa_Latn"

    def test_parse_language_pair_wmt(self):
        """Test parsing WMT language pairs."""
        benchmark = WMTBenchmark()

        source, target = benchmark._parse_language_pair("en-de")
        assert source == "en"
        assert target == "de"

    def test_extract_source_flores(self):
        """Test extracting source text from FLORES dataset."""
        benchmark = WMTBenchmark()

        sample = {"sentence_eng_Latn": "Hello world", "sentence_spa_Latn": "Hola mundo"}

        source = benchmark._extract_source(sample, "flores-200")
        assert source == "Hello world"

    def test_extract_translation_flores(self):
        """Test extracting translation from FLORES dataset."""
        benchmark = WMTBenchmark()

        sample = {"sentence_eng_Latn": "Hello world", "sentence_spa_Latn": "Hola mundo"}

        translation = benchmark._extract_translation(sample, "flores-200")
        assert translation == "Hola mundo"

    @pytest.mark.asyncio
    async def test_save_result(self):
        """Test saving benchmark result."""
        with tempfile.TemporaryDirectory() as tmpdir:
            benchmark = WMTBenchmark(results_dir=tmpdir)

            result = BenchmarkResult(
                dataset_name="test",
                system_name="kttc",
                sample_size=10,
                avg_mqm_score=85.0,
                std_mqm_score=5.0,
                avg_processing_time=1.0,
            )

            await benchmark._save_result(result)

            # Check file was created
            json_files = list(Path(tmpdir).glob("*.json"))
            assert len(json_files) > 0

    def test_export_report_markdown(self):
        """Test exporting markdown report."""
        with tempfile.TemporaryDirectory() as tmpdir:
            benchmark = WMTBenchmark(results_dir=tmpdir)

            result = BenchmarkResult(
                dataset_name="test",
                system_name="kttc",
                sample_size=10,
                avg_mqm_score=85.0,
                std_mqm_score=5.0,
                avg_processing_time=1.0,
            )

            benchmark.results.append(result)

            output_file = Path(tmpdir) / "report.md"
            benchmark.export_report(output_file, format="markdown")

            assert output_file.exists()
            content = output_file.read_text()
            assert "test" in content
            assert "85.0" in content or "85" in content

    def test_export_report_json(self):
        """Test exporting JSON report."""
        with tempfile.TemporaryDirectory() as tmpdir:
            benchmark = WMTBenchmark(results_dir=tmpdir)

            result = BenchmarkResult(
                dataset_name="test",
                system_name="kttc",
                sample_size=10,
                avg_mqm_score=85.0,
                std_mqm_score=5.0,
                avg_processing_time=1.0,
            )

            benchmark.results.append(result)

            output_file = Path(tmpdir) / "report.json"
            benchmark.export_report(output_file, format="json")

            assert output_file.exists()

            import json

            with open(output_file) as f:
                data = json.load(f)

            # JSON export returns dict with metadata and results
            assert "results" in data
            assert len(data["results"]) == 1
            assert data["results"][0]["dataset_name"] == "test"

    def test_list_available_datasets(self, capsys):
        """Test listing available datasets."""
        benchmark = WMTBenchmark()
        benchmark.list_available_datasets()

        captured = capsys.readouterr()
        assert "FLORES-200" in captured.out  # Uppercase in display
        assert "WMT23" in captured.out or "WMT 2023" in captured.out


@pytest.mark.asyncio
@pytest.mark.slow
async def test_quick_benchmark_function(mock_orchestrator):
    """Test quick_benchmark convenience function."""
    from tests.benchmarks.wmt_benchmark import quick_benchmark

    mock_dataset = [{"sentence_eng_Latn": "Test", "sentence_spa_Latn": "Prueba"}]

    mock_orchestrator.evaluate.return_value = QAReport(
        task=TranslationTask(
            source_text="Test", translation="Prueba", source_lang="en", target_lang="es"
        ),
        mqm_score=90.0,
        status="pass",
        errors=[],
    )

    with patch(
        "tests.benchmarks.wmt_benchmark.WMTBenchmark._load_dataset", return_value=mock_dataset
    ):
        result = await quick_benchmark(
            orchestrator=mock_orchestrator,
            dataset_name="flores-200",
            language_pair="eng_Latn-spa_Latn",
            sample_size=1,
        )

        assert result.sample_size == 1
        assert result.avg_mqm_score == 90.0
