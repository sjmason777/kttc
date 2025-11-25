#!/usr/bin/env python3
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

"""Comprehensive benchmark runner for KTTC translation quality assessment."""

import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from kttc.agents.orchestrator import AgentOrchestrator
from kttc.llm import OpenAIProvider
from kttc.utils.config import get_settings
from tests.benchmarks.enhanced_dataset_loader import EnhancedDatasetLoader
from tests.benchmarks.wmt_benchmark import WMTBenchmark


class ComprehensiveBenchmark:
    """Comprehensive benchmark system for KTTC."""

    def __init__(self, results_dir: str | Path = "benchmark_results"):
        """Initialize the comprehensive benchmark.

        Args:
            results_dir: Directory to store results
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.loader = EnhancedDatasetLoader()
        self.wmt_benchmark = WMTBenchmark(self.results_dir)

        self.all_results: list[dict[str, Any]] = []

    async def run_full_benchmark(
        self,
        orchestrator: AgentOrchestrator,
        language_pairs: list[tuple[str, str]],
        sample_size: int = 100,
        include_bad_translations: bool = True,
    ) -> dict[str, Any]:
        """Run comprehensive benchmark across all language pairs and datasets.

        Args:
            orchestrator: Agent orchestrator for evaluation
            language_pairs: List of (source_lang, target_lang) tuples
            sample_size: Number of samples per language pair
            include_bad_translations: Whether to test with bad translations

        Returns:
            Comprehensive benchmark results
        """
        print("\n" + "=" * 80)
        print("KTTC COMPREHENSIVE BENCHMARK")
        print("=" * 80)
        print(f"\nStarted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Language pairs: {len(language_pairs)}")
        print(f"Samples per pair: {sample_size}")
        print(f"Results directory: {self.results_dir.absolute()}\n")
        print("=" * 80 + "\n")

        start_time = time.time()
        results_by_pair = {}

        for idx, (src_lang, tgt_lang) in enumerate(language_pairs, 1):
            print(f"\n[{idx}/{len(language_pairs)}] Running benchmark: {src_lang} → {tgt_lang}")
            print("-" * 80)

            pair_key = f"{src_lang}-{tgt_lang}"

            try:
                # Run FLORES-200 benchmark (good translations)
                flores_result = await self.run_flores200_benchmark(
                    orchestrator, src_lang, tgt_lang, sample_size
                )

                results_by_pair[pair_key] = {
                    "language_pair": pair_key,
                    "source_lang": src_lang,
                    "target_lang": tgt_lang,
                    "flores200_good": flores_result,
                }

                # Generate and test bad translations if requested
                if include_bad_translations:
                    bad_result = await self.run_bad_translation_test(
                        orchestrator, src_lang, tgt_lang, sample_size // 3
                    )
                    results_by_pair[pair_key]["synthetic_bad"] = bad_result

                print(f"✅ Completed {src_lang} → {tgt_lang}")

            except Exception as e:
                print(f"❌ Error processing {src_lang} → {tgt_lang}: {e}")
                import traceback

                traceback.print_exc()

        # Calculate summary statistics
        total_time = time.time() - start_time
        summary = self.calculate_summary(results_by_pair, total_time)

        # Save comprehensive report
        self.save_comprehensive_report(results_by_pair, summary)

        return {"summary": summary, "results_by_pair": results_by_pair}

    async def run_flores200_benchmark(
        self, orchestrator: AgentOrchestrator, src_lang: str, tgt_lang: str, sample_size: int
    ) -> dict[str, Any]:
        """Run benchmark on FLORES-200 dataset.

        Args:
            orchestrator: Agent orchestrator
            src_lang: Source language code
            tgt_lang: Target language code
            sample_size: Number of samples

        Returns:
            Benchmark results for FLORES-200
        """
        print("  📊 Loading FLORES-200 data...")

        # Load data
        samples = await self.loader.load_flores200(
            src_lang, tgt_lang, split="devtest", sample_size=sample_size
        )

        if not samples:
            print("  ⚠️  No FLORES-200 data available, using fallback")
            return {}

        print(f"  ✓ Loaded {len(samples)} samples")
        print("  🔍 Running evaluation...")

        # Run evaluation
        from kttc.core.models import TranslationTask

        mqm_scores = []
        error_counts = {"critical": 0, "major": 0, "minor": 0}
        processing_times = []

        for idx, sample in enumerate(samples):
            if (idx + 1) % 10 == 0:
                print(f"     Progress: {idx + 1}/{len(samples)}", end="\r")

            start = time.time()

            task = TranslationTask(
                source_text=sample["source"],
                translation=sample["translation"],
                source_lang=src_lang,
                target_lang=tgt_lang,
            )

            # Retry logic for API errors
            max_retries = 3
            report = None
            for attempt in range(max_retries):
                try:
                    report = await orchestrator.evaluate(task)
                    break
                except Exception:
                    if attempt < max_retries - 1:
                        print(f"\n     ⚠️  Retry {attempt + 1}/{max_retries}...", end="")
                        await asyncio.sleep(2**attempt)
                    else:
                        print(f"\n     ❌ Failed sample {idx + 1}")
                        # Skip this sample
                        continue

            if report is None:
                continue

            processing_time = time.time() - start

            mqm_scores.append(report.mqm_score)
            processing_times.append(processing_time)

            for error in report.errors:
                severity = error.severity.lower()
                if severity in error_counts:
                    error_counts[severity] += 1

        print()  # New line after progress

        # Calculate statistics
        avg_mqm = sum(mqm_scores) / len(mqm_scores) if mqm_scores else 0
        avg_time = sum(processing_times) / len(processing_times) if processing_times else 0
        pass_rate = sum(1 for s in mqm_scores if s >= 95) / len(mqm_scores) if mqm_scores else 0

        result = {
            "dataset": "flores200",
            "sample_count": len(samples),
            "avg_mqm_score": round(avg_mqm, 2),
            "pass_rate": round(pass_rate * 100, 1),
            "error_counts": error_counts,
            "avg_processing_time": round(avg_time, 3),
            "mqm_scores": mqm_scores,
        }

        print(f"  ✓ Results: MQM={avg_mqm:.2f}, Pass Rate={pass_rate*100:.1f}%")

        return result

    async def run_bad_translation_test(
        self, orchestrator: AgentOrchestrator, src_lang: str, tgt_lang: str, sample_size: int
    ) -> dict[str, Any]:
        """Test system's ability to detect bad translations.

        Args:
            orchestrator: Agent orchestrator
            src_lang: Source language code
            tgt_lang: Target language code
            sample_size: Number of samples

        Returns:
            Results for bad translation detection
        """
        print("  🔧 Testing bad translation detection...")

        # Try critical_bad first (better errors), fallback to synthetic_bad
        critical_file = self.loader.data_dir / f"critical_bad_{src_lang}_{tgt_lang}.json"
        synthetic_file = self.loader.data_dir / f"synthetic_bad_{src_lang}_{tgt_lang}.json"

        cache_file = critical_file if critical_file.exists() else synthetic_file

        if cache_file.exists():
            dataset_type = "critical" if cache_file == critical_file else "synthetic"
            print(f"  ✓ Loading cached bad translations ({dataset_type})")
            data = json.loads(cache_file.read_text(encoding="utf-8"))

            # Filter only bad quality samples
            bad_samples = [s for s in data if s.get("quality") == "bad"]
            samples = bad_samples[:sample_size] if sample_size else bad_samples
            print(f"  ✓ Found {len(samples)} bad samples")
        else:
            print("  ⚠️  No cached bad translations found")
            print("     Run: python3.11 scripts/generate_critical_bad_translations.py")
            return {}

        from kttc.core.models import TranslationTask

        mqm_scores = []
        detected = 0

        for idx, sample in enumerate(samples):
            task = TranslationTask(
                source_text=sample["source"],
                translation=sample.get("bad_translation", sample.get("translation", "")),
                source_lang=src_lang,
                target_lang=tgt_lang,
            )

            # Retry logic
            max_retries = 3
            report = None
            for attempt in range(max_retries):
                try:
                    report = await orchestrator.evaluate(task)
                    break
                except Exception:
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2**attempt)
                    else:
                        continue

            if report is None:
                continue

            mqm_scores.append(report.mqm_score)

            # Count as detected if MQM < 95
            if report.mqm_score < 95:
                detected += 1

        detection_rate = detected / len(samples) if samples else 0

        result = {
            "dataset": "synthetic_bad",
            "sample_count": len(samples),
            "avg_mqm_score": round(sum(mqm_scores) / len(mqm_scores), 2) if mqm_scores else 0,
            "detection_rate": round(detection_rate * 100, 1),
            "detected_count": detected,
        }

        print(f"  ✓ Detection rate: {detection_rate*100:.1f}%")

        return result

    def calculate_summary(
        self, results_by_pair: dict[str, Any], total_time: float
    ) -> dict[str, Any]:
        """Calculate summary statistics across all results.

        Args:
            results_by_pair: Results organized by language pair
            total_time: Total execution time

        Returns:
            Summary statistics
        """
        all_flores_scores = []
        all_bad_scores = []

        for pair_results in results_by_pair.values():
            if "flores200_good" in pair_results and pair_results["flores200_good"]:
                flores = pair_results["flores200_good"]
                if "mqm_scores" in flores:
                    all_flores_scores.extend(flores["mqm_scores"])

            if "synthetic_bad" in pair_results and pair_results["synthetic_bad"]:
                bad = pair_results["synthetic_bad"]
                if "avg_mqm_score" in bad:
                    all_bad_scores.append(bad["avg_mqm_score"])

        return {
            "total_language_pairs": len(results_by_pair),
            "total_execution_time_seconds": round(total_time, 2),
            "flores200_avg_mqm": (
                round(sum(all_flores_scores) / len(all_flores_scores), 2)
                if all_flores_scores
                else None
            ),
            "bad_translations_avg_mqm": (
                round(sum(all_bad_scores) / len(all_bad_scores), 2) if all_bad_scores else None
            ),
            "timestamp": datetime.now().isoformat(),
        }

    def save_comprehensive_report(
        self, results_by_pair: dict[str, Any], summary: dict[str, Any]
    ) -> None:
        """Save comprehensive report to files.

        Args:
            results_by_pair: Results by language pair
            summary: Summary statistics
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save JSON report
        json_file = self.results_dir / f"comprehensive_benchmark_{timestamp}.json"
        json_file.write_text(
            json.dumps(
                {"summary": summary, "results_by_pair": results_by_pair},
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        # Save Markdown report
        md_file = self.results_dir / f"comprehensive_benchmark_{timestamp}.md"
        md_content = self.generate_markdown_report(results_by_pair, summary)
        md_file.write_text(md_content, encoding="utf-8")

        print("\n" + "=" * 80)
        print("📊 BENCHMARK COMPLETE!")
        print("=" * 80)
        print(f"\n✅ JSON report saved: {json_file}")
        print(f"✅ Markdown report saved: {md_file}\n")

    def generate_markdown_report(
        self, results_by_pair: dict[str, Any], summary: dict[str, Any]
    ) -> str:
        """Generate Markdown report.

        Args:
            results_by_pair: Results by language pair
            summary: Summary statistics

        Returns:
            Markdown formatted report
        """
        lines = [
            "# KTTC Comprehensive Benchmark Report",
            "",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Summary",
            "",
            f"- **Total Language Pairs:** {summary['total_language_pairs']}",
            f"- **Total Execution Time:** {summary['total_execution_time_seconds']:.2f}s",
        ]

        if summary["flores200_avg_mqm"]:
            lines.append(f"- **FLORES-200 Avg MQM:** {summary['flores200_avg_mqm']:.2f}")

        if summary["bad_translations_avg_mqm"]:
            lines.append(
                f"- **Bad Translations Avg MQM:** {summary['bad_translations_avg_mqm']:.2f}"
            )

        lines.append("")
        lines.append("## Results by Language Pair")
        lines.append("")

        for pair_key, results in results_by_pair.items():
            lines.append(f"### {pair_key}")
            lines.append("")

            if "flores200_good" in results and results["flores200_good"]:
                flores = results["flores200_good"]
                lines.append("#### FLORES-200 (Good Translations)")
                lines.append(f"- Samples: {flores['sample_count']}")
                lines.append(f"- Avg MQM: {flores['avg_mqm_score']:.2f}")
                lines.append(f"- Pass Rate: {flores['pass_rate']:.1f}%")
                lines.append(f"- Avg Time: {flores['avg_processing_time']:.3f}s")
                lines.append("")

            if "synthetic_bad" in results and results["synthetic_bad"]:
                bad = results["synthetic_bad"]
                lines.append("#### Synthetic Bad Translations")
                lines.append(f"- Samples: {bad['sample_count']}")
                lines.append(f"- Avg MQM: {bad['avg_mqm_score']:.2f}")
                lines.append(f"- Detection Rate: {bad['detection_rate']:.1f}%")
                lines.append("")

        return "\n".join(lines)


async def main() -> None:
    """Main benchmark execution."""
    print("\n🚀 KTTC Comprehensive Benchmark Runner\n")

    # Setup orchestrator
    settings = get_settings()
    llm = None

    # Try OpenAI first
    try:
        api_key = settings.get_llm_provider_key("openai")
        llm = OpenAIProvider(api_key=api_key, model="gpt-4o")
        print("✅ Using OpenAI (gpt-4o)")
    except Exception:
        # Silently ignore OpenAI setup errors and try Anthropic instead
        pass

    # Fall back to Anthropic
    if llm is None:
        try:
            from kttc.llm import AnthropicProvider

            api_key = settings.get_llm_provider_key("anthropic")
            llm = AnthropicProvider(api_key=api_key, model="claude-3-5-haiku-20241022")
            print("✅ Using Anthropic (claude-3-5-haiku)")
        except Exception:
            # Silently ignore Anthropic setup errors and continue with error message
            pass

    if llm is None:
        print("❌ Error: Could not initialize LLM provider")
        print("   Make sure KTTC_OPENAI_API_KEY or KTTC_ANTHROPIC_API_KEY is set in .env")
        return

    orchestrator = AgentOrchestrator(
        llm,
        quality_threshold=95.0,
        agent_temperature=settings.default_temperature,
        agent_max_tokens=settings.default_max_tokens,
    )

    # Define language pairs to test
    language_pairs = [
        ("en", "ru"),  # English → Russian
        ("en", "zh"),  # English → Chinese
        ("ru", "en"),  # Russian → English
        ("zh", "en"),  # Chinese → English
        ("ru", "zh"),  # Russian → Chinese
        ("zh", "ru"),  # Chinese → Russian
    ]

    # Run benchmark
    benchmark = ComprehensiveBenchmark()
    await benchmark.run_full_benchmark(
        orchestrator,
        language_pairs=language_pairs,  # All 6 pairs
        sample_size=10,  # Full dataset
        include_bad_translations=True,  # Test bad translations too
    )

    print("\n🎉 Benchmark execution completed!")


if __name__ == "__main__":
    asyncio.run(main())
