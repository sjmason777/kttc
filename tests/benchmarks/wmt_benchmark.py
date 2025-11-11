"""WMT Benchmark integration for KTTC.

Provides evaluation on industry-standard WMT and FLORES datasets
to validate KTTC performance against benchmarks.

Supports:
- WMT 2023-2025 datasets
- FLORES-200 multilingual benchmarks
- Statistical analysis and reporting
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class BenchmarkResult(BaseModel):
    """Result from a single benchmark evaluation.

    Records performance metrics for a translation system on a dataset.
    """

    dataset_name: str = Field(description="Dataset identifier (e.g., wmt23-en-ru)")
    system_name: str = Field(description="System being evaluated")
    sample_size: int = Field(description="Number of samples evaluated")
    avg_mqm_score: float = Field(description="Average MQM score (0-100)")
    std_mqm_score: float = Field(description="Standard deviation of MQM scores")
    avg_comet_score: float | None = Field(default=None, description="Average COMET score (0-1)")
    avg_kiwi_score: float | None = Field(default=None, description="Average CometKiwi score (0-1)")
    error_counts: dict[str, int] = Field(
        default_factory=dict, description="Error counts by severity"
    )
    avg_processing_time: float = Field(description="Average time per translation (seconds)")
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: dict[str, Any] = Field(default_factory=dict)


class WMTBenchmark:
    """WMT and FLORES benchmark evaluation for KTTC.

    Provides standardized evaluation on public translation benchmarks
    to compare KTTC with other systems and track improvements over time.

    Example:
        >>> benchmark = WMTBenchmark()
        >>> result = await benchmark.run_benchmark(
        ...     orchestrator=my_orchestrator,
        ...     dataset_name="flores-200",
        ...     language_pair="en-es",
        ...     sample_size=100
        ... )
        >>> benchmark.export_report("results/wmt_report.md")
    """

    AVAILABLE_DATASETS = {
        # WMT datasets
        "wmt23": {
            "name": "wmt/wmt23",
            "pairs": ["en-de", "en-ru", "en-zh", "zh-en"],
            "description": "WMT 2023 News Translation Task",
        },
        "wmt22": {
            "name": "wmt/wmt22",
            "pairs": ["en-de", "en-ru", "en-cs", "en-ja"],
            "description": "WMT 2022 News Translation Task",
        },
        # FLORES-200
        "flores-200": {
            "name": "facebook/flores",
            "pairs": [
                "eng_Latn-spa_Latn",
                "eng_Latn-rus_Cyrl",
                "eng_Latn-zho_Hans",
                "eng_Latn-fra_Latn",
                "eng_Latn-deu_Latn",
            ],
            "description": "FLORES-200 Multilingual Benchmark (200 languages)",
        },
        # Custom test sets
        "custom": {
            "name": "local",
            "pairs": ["any"],
            "description": "Custom local test set",
        },
    }

    def __init__(self, results_dir: str | Path = "benchmark_results"):
        """Initialize WMT Benchmark.

        Args:
            results_dir: Directory to save benchmark results
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.results: list[BenchmarkResult] = []
        self._datasets_cache: dict[str, Any] = {}

    async def run_benchmark(
        self,
        orchestrator: Any,  # AgentOrchestrator
        dataset_name: str = "flores-200",
        language_pair: str = "eng_Latn-spa_Latn",
        sample_size: int = 100,
        subset: str = "devtest",
        seed: int = 42,
    ) -> BenchmarkResult:
        """Run KTTC evaluation on a benchmark dataset.

        Args:
            orchestrator: KTTC AgentOrchestrator instance
            dataset_name: Dataset to use (wmt23, wmt22, flores-200)
            language_pair: Language pair (format depends on dataset)
            sample_size: Number of samples to evaluate
            subset: Dataset subset (dev, test, devtest)
            seed: Random seed for sampling

        Returns:
            BenchmarkResult with evaluation metrics

        Raises:
            ValueError: If dataset or language pair not supported
            RuntimeError: If evaluation fails
        """
        logger.info(
            f"Starting benchmark: {dataset_name} ({language_pair}) with {sample_size} samples"
        )

        # Load dataset
        try:
            dataset = await self._load_dataset(dataset_name, language_pair, subset)
        except Exception as e:
            raise RuntimeError(f"Failed to load dataset {dataset_name}: {e}") from e

        # Sample data
        if len(dataset) > sample_size:
            # Shuffle and sample
            import random

            random.seed(seed)
            indices = random.sample(range(len(dataset)), sample_size)
            samples = [dataset[i] for i in indices]
        else:
            samples = list(dataset)
            sample_size = len(samples)

        logger.info(f"Evaluating {sample_size} samples...")

        # Initialize metrics
        mqm_scores = []
        comet_scores = []
        kiwi_scores = []
        error_counts = {"critical": 0, "major": 0, "minor": 0}
        processing_times = []

        # Evaluate each sample
        for idx, sample in enumerate(samples):
            try:
                # Extract source and translation
                source_text = self._extract_source(sample, dataset_name)
                translation = self._extract_translation(sample, dataset_name)
                reference = self._extract_reference(sample, dataset_name)

                # Create translation task
                from kttc.core.models import TranslationTask

                # Parse language codes
                source_lang, target_lang = self._parse_language_pair(language_pair)

                task = TranslationTask(
                    source_text=source_text,
                    translation=translation,
                    source_lang=source_lang,
                    target_lang=target_lang,
                    reference=reference,
                )

                # Run evaluation
                start_time = time.time()
                report = await orchestrator.evaluate(task)
                elapsed = time.time() - start_time

                # Collect metrics
                mqm_scores.append(report.mqm_score)
                processing_times.append(elapsed)

                # Neural metrics (if available)
                if hasattr(report, "neural_metrics") and report.neural_metrics:
                    if report.neural_metrics.comet_score:
                        comet_scores.append(report.neural_metrics.comet_score)
                    if report.neural_metrics.kiwi_score:
                        kiwi_scores.append(report.neural_metrics.kiwi_score)

                # Error counts
                for error in report.errors:
                    severity = error.severity.value
                    if severity in error_counts:
                        error_counts[severity] += 1

                if (idx + 1) % 10 == 0:
                    logger.info(f"Progress: {idx + 1}/{sample_size} samples evaluated")

            except Exception as e:
                logger.error(f"Failed to evaluate sample {idx}: {e}")
                continue

        # Calculate summary statistics
        result = BenchmarkResult(
            dataset_name=f"{dataset_name}-{language_pair}",
            system_name="kttc",
            sample_size=sample_size,
            avg_mqm_score=float(np.mean(mqm_scores)) if mqm_scores else 0.0,
            std_mqm_score=float(np.std(mqm_scores)) if mqm_scores else 0.0,
            avg_comet_score=float(np.mean(comet_scores)) if comet_scores else None,
            avg_kiwi_score=float(np.mean(kiwi_scores)) if kiwi_scores else None,
            error_counts=error_counts,
            avg_processing_time=float(np.mean(processing_times)) if processing_times else 0.0,
            metadata={
                "subset": subset,
                "seed": seed,
                "mqm_scores_distribution": {
                    "min": float(np.min(mqm_scores)) if mqm_scores else 0.0,
                    "max": float(np.max(mqm_scores)) if mqm_scores else 0.0,
                    "median": float(np.median(mqm_scores)) if mqm_scores else 0.0,
                    "p25": float(np.percentile(mqm_scores, 25)) if mqm_scores else 0.0,
                    "p75": float(np.percentile(mqm_scores, 75)) if mqm_scores else 0.0,
                },
            },
        )

        self.results.append(result)
        logger.info(
            f"Benchmark completed: MQM={result.avg_mqm_score:.2f}±{result.std_mqm_score:.2f}"
        )

        # Auto-save
        await self._save_result(result)

        return result

    async def _load_dataset(
        self, dataset_name: str, language_pair: str, subset: str
    ) -> list[dict[str, Any]]:
        """Load dataset from HuggingFace or local source.

        Args:
            dataset_name: Dataset identifier
            language_pair: Language pair
            subset: Dataset subset

        Returns:
            List of dataset samples

        Raises:
            ValueError: If dataset not available
        """
        if dataset_name not in self.AVAILABLE_DATASETS:
            raise ValueError(
                f"Dataset {dataset_name} not available. "
                f"Choose from: {list(self.AVAILABLE_DATASETS.keys())}"
            )

        cache_key = f"{dataset_name}:{language_pair}:{subset}"
        if cache_key in self._datasets_cache:
            logger.info(f"Using cached dataset: {cache_key}")
            return self._datasets_cache[cache_key]

        dataset_info = self.AVAILABLE_DATASETS[dataset_name]

        if dataset_name == "custom":
            # Load from local file
            raise NotImplementedError("Custom dataset loading not yet implemented")

        # Load from HuggingFace
        try:
            import datasets

            logger.info(f"Loading dataset from HuggingFace: {dataset_info['name']}")

            if dataset_name == "flores-200":
                # FLORES uses special naming
                dataset = datasets.load_dataset(dataset_info["name"], language_pair)
                data = list(dataset[subset])
            else:
                # WMT datasets
                dataset = datasets.load_dataset(dataset_info["name"], language_pair)
                data = list(dataset[subset] if subset in dataset else dataset["test"])

            self._datasets_cache[cache_key] = data
            logger.info(f"Loaded {len(data)} samples from {dataset_name}")
            return data

        except ImportError as e:
            raise RuntimeError(
                "datasets library not installed. Install with: pip install datasets"
            ) from e
        except Exception as e:
            raise RuntimeError(f"Failed to load dataset: {e}") from e

    def _extract_source(self, sample: dict[str, Any], dataset_name: str) -> str:
        """Extract source text from dataset sample."""
        if dataset_name == "flores-200":
            # FLORES format: sample["sentence_eng_Latn"]
            for key in sample:
                if key.startswith("sentence_") and "eng" in key:
                    return sample[key]
        # WMT format: sample["translation"]["en"]
        if "translation" in sample:
            trans = sample["translation"]
            if isinstance(trans, dict):
                # Try common source language codes
                for lang in ["en", "source", "src"]:
                    if lang in trans:
                        return trans[lang]
        # Direct field
        if "source" in sample:
            return sample["source"]

        raise ValueError(f"Could not extract source text from sample: {sample.keys()}")

    def _extract_translation(self, sample: dict[str, Any], dataset_name: str) -> str:
        """Extract translation (MT output) from dataset sample."""
        if dataset_name == "flores-200":
            # For FLORES, use target language sentence
            for key in sample:
                if key.startswith("sentence_") and "eng" not in key:
                    return sample[key]
        # WMT format
        if "translation" in sample:
            trans = sample["translation"]
            if isinstance(trans, dict):
                # Try common target language codes
                for lang in ["de", "ru", "zh", "es", "fr", "target", "tgt"]:
                    if lang in trans:
                        return trans[lang]
        # Direct field
        if "mt" in sample:
            return sample["mt"]
        if "hypothesis" in sample:
            return sample["hypothesis"]

        raise ValueError(f"Could not extract translation from sample: {sample.keys()}")

    def _extract_reference(self, sample: dict[str, Any], dataset_name: str) -> str | None:
        """Extract reference translation from dataset sample."""
        # Reference is optional
        if "reference" in sample:
            return sample["reference"]
        if "ref" in sample:
            return sample["ref"]
        # For some datasets, reference might be same as translation
        # (if we're evaluating human translations)
        return None

    def _parse_language_pair(self, language_pair: str) -> tuple[str, str]:
        """Parse language pair string into source and target codes.

        Args:
            language_pair: Language pair (e.g., "en-es" or "eng_Latn-spa_Latn")

        Returns:
            Tuple of (source_lang, target_lang) in 2-letter codes
        """
        # Handle FLORES format: eng_Latn-spa_Latn
        if "_" in language_pair:
            parts = language_pair.split("-")
            source = parts[0].split("_")[0][:2]  # eng_Latn -> en
            target = parts[1].split("_")[0][:2] if len(parts) > 1 else "es"
            return source, target

        # Handle simple format: en-es
        parts = language_pair.split("-")
        return parts[0], parts[1] if len(parts) > 1 else "en"

    async def _save_result(self, result: BenchmarkResult) -> None:
        """Save benchmark result to file."""
        filename = (
            f"benchmark_{result.dataset_name}_{result.timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        )
        filepath = self.results_dir / filename

        with open(filepath, "w") as f:
            json.dump(result.model_dump(mode="json"), f, indent=2, default=str)

        logger.info(f"Benchmark result saved to {filepath}")

    def export_report(
        self, output_file: str | Path = "benchmark_report.md", format: str = "markdown"
    ) -> None:
        """Export benchmark report.

        Args:
            output_file: Output file path
            format: Report format (markdown, json, html)
        """
        if format == "markdown":
            self._export_markdown(output_file)
        elif format == "json":
            self._export_json(output_file)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def _export_markdown(self, output_file: str | Path) -> None:
        """Export benchmark report as markdown."""
        report = f"""# WMT Benchmark Report

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Summary

Total benchmarks run: {len(self.results)}

## Results

"""

        # Add table header
        report += "| Dataset | Sample Size | MQM Score | COMET | CometKiwi | Errors (C/M/m) | Avg Time (s) |\n"
        report += "|---------|-------------|-----------|-------|-----------|----------------|---------------|\n"

        # Add results
        for result in self.results:
            errors_str = f"{result.error_counts.get('critical', 0)}/{result.error_counts.get('major', 0)}/{result.error_counts.get('minor', 0)}"
            comet_str = f"{result.avg_comet_score:.3f}" if result.avg_comet_score else "N/A"
            kiwi_str = f"{result.avg_kiwi_score:.3f}" if result.avg_kiwi_score else "N/A"

            report += (
                f"| {result.dataset_name} "
                f"| {result.sample_size} "
                f"| {result.avg_mqm_score:.2f}±{result.std_mqm_score:.2f} "
                f"| {comet_str} "
                f"| {kiwi_str} "
                f"| {errors_str} "
                f"| {result.avg_processing_time:.2f} |\n"
            )

        # Add detailed analysis
        report += "\n## Detailed Analysis\n\n"

        for result in self.results:
            report += f"### {result.dataset_name}\n\n"
            report += f"- **MQM Score:** {result.avg_mqm_score:.2f} ± {result.std_mqm_score:.2f}\n"
            if result.avg_comet_score:
                report += f"- **COMET Score:** {result.avg_comet_score:.3f}\n"
            if result.avg_kiwi_score:
                report += f"- **CometKiwi Score:** {result.avg_kiwi_score:.3f}\n"
            report += f"- **Processing Speed:** {result.avg_processing_time:.2f}s per translation\n"
            report += "- **Error Distribution:**\n"
            report += f"  - Critical: {result.error_counts.get('critical', 0)}\n"
            report += f"  - Major: {result.error_counts.get('major', 0)}\n"
            report += f"  - Minor: {result.error_counts.get('minor', 0)}\n"

            # Distribution stats
            if "mqm_scores_distribution" in result.metadata:
                dist = result.metadata["mqm_scores_distribution"]
                report += "- **MQM Distribution:**\n"
                report += f"  - Min: {dist['min']:.2f}\n"
                report += f"  - P25: {dist['p25']:.2f}\n"
                report += f"  - Median: {dist['median']:.2f}\n"
                report += f"  - P75: {dist['p75']:.2f}\n"
                report += f"  - Max: {dist['max']:.2f}\n"
            report += "\n"

        # Add interpretation section
        report += """## Interpretation

- **MQM Score ≥ 95:** Excellent quality (minimal post-editing required)
- **MQM Score 85-95:** Good quality (light post-editing required)
- **MQM Score 75-85:** Acceptable quality (moderate post-editing required)
- **MQM Score < 75:** Poor quality (significant rework required)

---

*Report generated by KTTC Benchmark Suite*
"""

        Path(output_file).write_text(report)
        logger.info(f"Markdown report exported to {output_file}")

    def _export_json(self, output_file: str | Path) -> None:
        """Export benchmark results as JSON."""
        data = {
            "generated_at": datetime.now().isoformat(),
            "total_benchmarks": len(self.results),
            "results": [result.model_dump(mode="json") for result in self.results],
        }

        with open(output_file, "w") as f:
            json.dump(data, f, indent=2, default=str)

        logger.info(f"JSON report exported to {output_file}")

    def list_available_datasets(self) -> None:
        """Print available datasets and their language pairs."""
        print("\n" + "=" * 80)
        print("AVAILABLE WMT BENCHMARK DATASETS".center(80))
        print("=" * 80)

        for dataset_id, info in self.AVAILABLE_DATASETS.items():
            print(f"\n📊 {dataset_id.upper()}")
            print(f"   Description: {info['description']}")
            print(f"   Source: {info['name']}")
            print(f"   Language pairs: {', '.join(info['pairs'])}")

        print("\n" + "=" * 80 + "\n")


# Convenience function for quick benchmarking
async def quick_benchmark(
    orchestrator: Any,
    dataset: str = "flores-200",
    language_pair: str = "eng_Latn-spa_Latn",
    sample_size: int = 50,
) -> BenchmarkResult:
    """Run a quick benchmark evaluation.

    Args:
        orchestrator: AgentOrchestrator instance
        dataset: Dataset name
        language_pair: Language pair
        sample_size: Number of samples

    Returns:
        BenchmarkResult
    """
    benchmark = WMTBenchmark()
    result = await benchmark.run_benchmark(
        orchestrator=orchestrator,
        dataset_name=dataset,
        language_pair=language_pair,
        sample_size=sample_size,
    )
    benchmark.export_report()
    return result
