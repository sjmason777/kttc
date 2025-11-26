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

"""WMT benchmark for evaluating translation quality across datasets."""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from kttc.agents.orchestrator import AgentOrchestrator
from kttc.core.models import TranslationTask


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""

    dataset_name: str
    language_pair: str
    sample_size: int
    avg_mqm_score: float
    std_mqm_score: float
    avg_comet_score: float | None = None
    avg_kiwi_score: float | None = None
    error_counts: dict[str, int] = field(default_factory=dict)
    avg_processing_time: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    individual_results: list[dict[str, Any]] = field(default_factory=list)


class WMTBenchmark:
    """Benchmark system for WMT and other translation datasets."""

    def __init__(self, results_dir: str | Path):
        """Initialize the benchmark.

        Args:
            results_dir: Directory to store benchmark results
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.results: list[BenchmarkResult] = []

    async def run_benchmark(
        self,
        orchestrator: AgentOrchestrator,
        dataset_name: str,
        language_pair: str,
        sample_size: int = 100,
    ) -> BenchmarkResult:
        """Run benchmark on a dataset.

        Args:
            orchestrator: Agent orchestrator for evaluation
            dataset_name: Name of the dataset (flores-200, wmt23, wmt22)
            language_pair: Language pair (e.g., 'eng_Latn-spa_Latn' or 'en-es')
            sample_size: Number of samples to evaluate

        Returns:
            BenchmarkResult with aggregated metrics
        """
        # Load dataset
        samples = await self._load_dataset(dataset_name, language_pair, sample_size)

        if not samples:
            raise ValueError(f"No samples found for {dataset_name} {language_pair}")

        # Run evaluation on all samples
        mqm_scores = []
        processing_times = []
        error_counts = {"critical": 0, "major": 0, "minor": 0}
        individual_results = []

        for idx, sample in enumerate(samples):
            start_time = time.time()

            # Evaluate translation
            task = TranslationTask(
                source_text=sample["source"],
                translation=sample["translation"],
                source_lang=self._parse_language(language_pair, "source"),
                target_lang=self._parse_language(language_pair, "target"),
            )

            report = await orchestrator.evaluate(task)

            processing_time = time.time() - start_time
            processing_times.append(processing_time)
            mqm_scores.append(report.mqm_score)

            # Count errors by severity
            for error in report.errors:
                severity = error.severity.lower()
                if severity in error_counts:
                    error_counts[severity] += 1

            # Store individual result
            individual_results.append(
                {
                    "sample_id": idx,
                    "source": sample["source"],
                    "translation": sample["translation"],
                    "mqm_score": report.mqm_score,
                    "status": report.status,
                    "errors": [
                        {
                            "category": e.category,
                            "severity": e.severity,
                            "description": e.description,
                        }
                        for e in report.errors
                    ],
                    "processing_time": processing_time,
                }
            )

        # Calculate aggregated metrics
        avg_mqm = statistics.mean(mqm_scores)
        std_mqm = statistics.stdev(mqm_scores) if len(mqm_scores) > 1 else 0.0
        avg_time = statistics.mean(processing_times)

        result = BenchmarkResult(
            dataset_name=dataset_name,
            language_pair=language_pair,
            sample_size=len(samples),
            avg_mqm_score=avg_mqm,
            std_mqm_score=std_mqm,
            error_counts=error_counts,
            avg_processing_time=avg_time,
            individual_results=individual_results,
        )

        self.results.append(result)

        # Save individual result file
        result_file = (
            self.results_dir
            / f"{dataset_name}_{language_pair}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        result_file.write_text(
            json.dumps(self._result_to_dict(result), indent=2, ensure_ascii=False)
        )

        return result

    async def _load_dataset(
        self, dataset_name: str, language_pair: str, sample_size: int
    ) -> list[dict[str, str]]:
        """Load samples from a dataset.

        Args:
            dataset_name: Dataset name
            language_pair: Language pair
            sample_size: Number of samples

        Returns:
            List of sample dictionaries with 'source' and 'translation'
        """
        from tests.benchmarks.dataset_loader import DatasetLoader

        # Use built-in sample data to avoid downloading large datasets in CI/CD
        # This is fast, lightweight, and doesn't require external dependencies
        return DatasetLoader._get_sample_data(language_pair, sample_size)

    def _parse_language_pair(self, language_pair: str) -> tuple[str, str]:
        """Parse language pair string.

        Args:
            language_pair: Language pair (e.g., 'eng_Latn-spa_Latn' or 'en-es')

        Returns:
            Tuple of (source_lang, target_lang)
        """
        if "-" in language_pair:
            parts = language_pair.split("-", 1)
            return (parts[0], parts[1])
        raise ValueError(
            f"Invalid language pair format (expected '<src>-<tgt>'): '{language_pair}'"
        )

    def _parse_language(self, language_pair: str, which: str) -> str:
        """Parse specific language from pair.

        Args:
            language_pair: Language pair string
            which: 'source' or 'target'

        Returns:
            Language code
        """
        src, tgt = self._parse_language_pair(language_pair)
        return src if which == "source" else tgt

    def export_report(self, filename: str, format: str = "json") -> None:
        """Export benchmark report.

        Args:
            filename: Output filename
            format: Report format ('json' or 'markdown')
        """
        output_path = Path(filename)

        if format == "json":
            self._export_json(output_path)
        elif format == "markdown":
            self._export_markdown(output_path)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def _export_json(self, output_path: Path) -> None:
        """Export results as JSON.

        Args:
            output_path: Output file path
        """
        data = {
            "benchmark_type": "wmt",
            "timestamp": datetime.now().isoformat(),
            "total_runs": len(self.results),
            "results": [self._result_to_dict(r) for r in self.results],
        }

        output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))

    def _export_markdown(self, output_path: Path) -> None:
        """Export results as Markdown.

        Args:
            output_path: Output file path
        """
        lines = [
            "# WMT Benchmark Results",
            "",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Total Runs:** {len(self.results)}",
            "",
        ]

        for result in self.results:
            lines.append(f"## {result.dataset_name} - {result.language_pair}")
            lines.append("")
            lines.append(f"- **Sample Size:** {result.sample_size}")
            lines.append(
                f"- **MQM Score:** {result.avg_mqm_score:.2f} ± {result.std_mqm_score:.2f}"
            )
            if result.avg_comet_score:
                lines.append(f"- **COMET Score:** {result.avg_comet_score:.3f}")
            if result.avg_kiwi_score:
                lines.append(f"- **CometKiwi Score:** {result.avg_kiwi_score:.3f}")
            lines.append(f"- **Avg Processing Time:** {result.avg_processing_time:.2f}s")
            lines.append("")
            lines.append("### Error Distribution")
            lines.append("")
            lines.append(f"- Critical: {result.error_counts.get('critical', 0)}")
            lines.append(f"- Major: {result.error_counts.get('major', 0)}")
            lines.append(f"- Minor: {result.error_counts.get('minor', 0)}")
            lines.append("")

        output_path.write_text("\n".join(lines))

    def _result_to_dict(self, result: BenchmarkResult) -> dict[str, Any]:
        """Convert BenchmarkResult to dictionary.

        Args:
            result: BenchmarkResult instance

        Returns:
            Dictionary representation
        """
        return {
            "dataset_name": result.dataset_name,
            "language_pair": result.language_pair,
            "sample_size": result.sample_size,
            "avg_mqm_score": result.avg_mqm_score,
            "std_mqm_score": result.std_mqm_score,
            "avg_comet_score": result.avg_comet_score,
            "avg_kiwi_score": result.avg_kiwi_score,
            "error_counts": result.error_counts,
            "avg_processing_time": result.avg_processing_time,
            "timestamp": result.timestamp,
            "individual_results": result.individual_results,
        }
