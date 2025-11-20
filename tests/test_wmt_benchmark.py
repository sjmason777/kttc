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

"""Test script for WMT benchmark - simulates what CI workflow does."""

import asyncio
import tempfile
from pathlib import Path


async def main() -> None:
    """Test the WMT benchmark implementation."""
    print("Testing WMT Benchmark implementation...")

    # Import required modules
    import os

    from kttc.agents.orchestrator import AgentOrchestrator
    from kttc.llm import OpenAIProvider
    from tests.benchmarks.wmt_benchmark import WMTBenchmark

    # Check if API key is available
    api_key = os.getenv("KTTC_OPENAI_API_KEY")
    if not api_key:
        print("⚠️  KTTC_OPENAI_API_KEY not set, using mock test")
        print("✅ Imports work correctly")
        print("✅ Module structure is valid")
        return

    # Create temporary directory for results
    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"\n📁 Results directory: {tmpdir}")

        # Initialize
        llm = OpenAIProvider(api_key=api_key, model="gpt-4")
        orchestrator = AgentOrchestrator(llm)

        # Run benchmark
        benchmark = WMTBenchmark(tmpdir)
        print("\n🏃 Running benchmark...")

        result = await benchmark.run_benchmark(
            orchestrator=orchestrator,
            dataset_name="flores-200",
            language_pair="eng_Latn-spa_Latn",
            sample_size=3,  # Small sample for testing
        )

        print("\n✅ Benchmark completed!")
        print(f"   MQM Score: {result.avg_mqm_score:.2f}±{result.std_mqm_score:.2f}")
        print(f"   Samples: {result.sample_size}")
        print(f"   Avg time: {result.avg_processing_time:.2f}s")

        # Export reports
        report_md = Path(tmpdir) / "test_report.md"
        report_json = Path(tmpdir) / "test_report.json"

        benchmark.export_report(str(report_md), format="markdown")
        benchmark.export_report(str(report_json), format="json")

        print("\n📄 Reports exported:")
        print(f"   - {report_md}")
        print(f"   - {report_json}")

        # Verify files exist
        assert report_md.exists(), "Markdown report not created"
        assert report_json.exists(), "JSON report not created"

        print("\n✅ All tests passed!")


if __name__ == "__main__":
    asyncio.run(main())
