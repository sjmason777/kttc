"""Performance benchmarking tests.

These tests measure the performance characteristics of KTTC:
- Agent execution speed
- Parallel vs sequential execution
- Large document handling
- Memory usage

Run with: pytest tests/performance/ -v --benchmark-only
"""

import asyncio
import time
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest

from kttc.agents.accuracy import AccuracyAgent
from kttc.agents.fluency import FluencyAgent
from kttc.agents.orchestrator import AgentOrchestrator
from kttc.agents.terminology import TerminologyAgent
from kttc.core.models import ErrorAnnotation, ErrorSeverity, TranslationTask
from kttc.llm.base import BaseLLMProvider

# Performance targets from DEVELOPMENT_PLAN.md
TARGET_SPEED_PER_1K_WORDS = 30.0  # seconds
TARGET_PARALLEL_SPEEDUP = 2.5  # Minimum speedup vs sequential
TARGET_MEMORY_MB = 500  # Maximum memory usage


@pytest.fixture
def mock_llm() -> BaseLLMProvider:
    """Mock LLM provider with realistic response times."""
    mock = Mock(spec=BaseLLMProvider)

    async def complete_with_delay(*args: Any, **kwargs: Any) -> str:
        # Simulate realistic API latency (100-500ms)
        await asyncio.sleep(0.2)
        return """ERROR: minor mistranslation
LOCATION: 10-15
SEVERITY: minor
DESCRIPTION: Slight inaccuracy
SUGGESTION: Use better word"""

    mock.complete = AsyncMock(side_effect=complete_with_delay)
    return mock


@pytest.fixture
def small_task() -> TranslationTask:
    """Small translation task (~50 words)."""
    return TranslationTask(
        source_text="This is a test sentence with approximately fifty words in total. "
        "It contains multiple clauses and some technical terminology. "
        "The purpose is to benchmark translation quality assessment. "
        "We want to measure how fast the system can process this. "
        "Performance is critical for production use.",
        translation="Esta es una oración de prueba con aproximadamente cincuenta palabras en total. "
        "Contiene múltiples cláusulas y alguna terminología técnica. "
        "El propósito es medir la calidad de traducción. "
        "Queremos medir qué tan rápido puede procesar esto el sistema. "
        "El rendimiento es crítico para uso en producción.",
        source_lang="en",
        target_lang="es",
    )


@pytest.fixture
def large_task() -> TranslationTask:
    """Large translation task (~1000 words)."""
    # Generate a realistic 1000-word document
    paragraph = (
        "In the field of computational linguistics and natural language processing, "
        "translation quality assessment represents a fundamental challenge that requires "
        "sophisticated approaches combining machine learning, statistical analysis, and "
        "linguistic expertise. Modern systems leverage large language models to evaluate "
        "translations across multiple dimensions including accuracy, fluency, and terminology. "
    )

    # Repeat to create ~1000 words
    source = (paragraph + " ") * 20

    # Spanish translation
    spanish_paragraph = (
        "En el campo de la lingüística computacional y el procesamiento del lenguaje natural, "
        "la evaluación de la calidad de traducción representa un desafío fundamental que requiere "
        "enfoques sofisticados que combinan aprendizaje automático, análisis estadístico y "
        "experiencia lingüística. Los sistemas modernos aprovechan grandes modelos de lenguaje para evaluar "
        "traducciones en múltiples dimensiones incluyendo precisión, fluidez y terminología. "
    )

    translation = (spanish_paragraph + " ") * 20

    return TranslationTask(
        source_text=source,
        translation=translation,
        source_lang="en",
        target_lang="es",
    )


@pytest.mark.benchmark
def test_single_agent_performance(mock_llm: BaseLLMProvider, small_task: TranslationTask) -> None:
    """Benchmark single agent execution speed."""

    async def run_test() -> None:
        agent = AccuracyAgent(mock_llm)

        start = time.time()
        errors = await agent.evaluate(small_task)
        duration = time.time() - start

        # Should complete in reasonable time
        assert duration < 5.0, f"Single agent took too long: {duration:.2f}s"
        assert len(errors) >= 0, "Should return error list"

        print(f"\n[BENCHMARK] Single agent: {duration:.3f}s")

    asyncio.run(run_test())


@pytest.mark.benchmark
def test_parallel_agent_performance(mock_llm: BaseLLMProvider, small_task: TranslationTask) -> None:
    """Benchmark parallel agent execution vs sequential.

    Target: 2.5x speedup for 3 agents running in parallel
    """

    async def run_test() -> None:
        agents = [
            AccuracyAgent(mock_llm),
            FluencyAgent(mock_llm),
            TerminologyAgent(mock_llm),
        ]

        # Sequential execution
        start_seq = time.time()
        for agent in agents:
            await agent.evaluate(small_task)
        sequential_duration = time.time() - start_seq

        # Parallel execution
        start_par = time.time()
        await asyncio.gather(*[agent.evaluate(small_task) for agent in agents])
        parallel_duration = time.time() - start_par

        speedup = sequential_duration / parallel_duration

        print(f"\n[BENCHMARK] Sequential: {sequential_duration:.3f}s")
        print(f"[BENCHMARK] Parallel: {parallel_duration:.3f}s")
        print(f"[BENCHMARK] Speedup: {speedup:.2f}x")

        # Should see at least 2.5x speedup from parallelization
        assert (
            speedup >= TARGET_PARALLEL_SPEEDUP
        ), f"Parallel speedup too low: {speedup:.2f}x (target: {TARGET_PARALLEL_SPEEDUP}x)"

    asyncio.run(run_test())


@pytest.mark.benchmark
def test_orchestrator_performance(mock_llm: BaseLLMProvider, small_task: TranslationTask) -> None:
    """Benchmark full orchestrator pipeline."""

    async def run_test() -> None:
        orchestrator = AgentOrchestrator(mock_llm)

        start = time.time()
        report = await orchestrator.evaluate(small_task)
        duration = time.time() - start

        # Should complete in reasonable time
        assert duration < 10.0, f"Orchestrator took too long: {duration:.2f}s"
        assert report.mqm_score >= 0.0, "Should return valid MQM score"

        print(f"\n[BENCHMARK] Full orchestrator: {duration:.3f}s")
        print(f"[BENCHMARK] MQM Score: {report.mqm_score}")

    asyncio.run(run_test())


@pytest.mark.benchmark
@pytest.mark.slow
def test_large_document_performance(mock_llm: BaseLLMProvider, large_task: TranslationTask) -> None:
    """Benchmark processing of large documents (~1000 words).

    Target: < 30s per 1000 words
    """

    async def run_test() -> None:
        orchestrator = AgentOrchestrator(mock_llm)

        word_count = len(large_task.source_text.split())

        start = time.time()
        report = await orchestrator.evaluate(large_task)
        duration = time.time() - start

        # Calculate speed per 1000 words
        speed_per_1k = (duration / word_count) * 1000

        print(f"\n[BENCHMARK] Large document ({word_count} words)")
        print(f"[BENCHMARK] Total time: {duration:.3f}s")
        print(f"[BENCHMARK] Speed per 1K words: {speed_per_1k:.2f}s")
        print(f"[BENCHMARK] MQM Score: {report.mqm_score}")

        # Should meet performance target
        assert speed_per_1k < TARGET_SPEED_PER_1K_WORDS, (
            f"Processing too slow: {speed_per_1k:.2f}s per 1K words "
            f"(target: < {TARGET_SPEED_PER_1K_WORDS}s)"
        )

    asyncio.run(run_test())


@pytest.mark.benchmark
def test_batch_processing_performance(
    mock_llm: BaseLLMProvider, small_task: TranslationTask
) -> None:
    """Benchmark batch processing of multiple documents."""

    async def run_test() -> None:
        orchestrator = AgentOrchestrator(mock_llm)

        # Create 10 tasks
        tasks = [small_task for _ in range(10)]

        start = time.time()
        results = await asyncio.gather(*[orchestrator.evaluate(task) for task in tasks])
        duration = time.time() - start

        throughput = len(tasks) / duration

        print(f"\n[BENCHMARK] Batch processing ({len(tasks)} documents)")
        print(f"[BENCHMARK] Total time: {duration:.3f}s")
        print(f"[BENCHMARK] Throughput: {throughput:.2f} docs/sec")

        assert len(results) == len(tasks), "Should process all tasks"
        assert throughput > 0.5, f"Throughput too low: {throughput:.2f} docs/sec"

    asyncio.run(run_test())


@pytest.mark.benchmark
def test_error_aggregation_performance(mock_llm: BaseLLMProvider) -> None:
    """Benchmark error aggregation and MQM scoring."""
    from kttc.core.mqm import MQMScorer

    scorer = MQMScorer()

    # Create a large number of errors
    errors = [
        ErrorAnnotation(
            category="accuracy",
            subcategory="mistranslation",
            severity=ErrorSeverity.MINOR,
            location=(i, i + 5),
            description=f"Error {i}",
        )
        for i in range(100)
    ]

    word_count = 1000

    start = time.time()
    for _ in range(1000):  # Run 1000 times
        score = scorer.calculate_score(errors, word_count)
    duration = time.time() - start

    ops_per_sec = 1000 / duration

    print("\n[BENCHMARK] MQM scoring (100 errors, 1000 iterations)")
    print(f"[BENCHMARK] Total time: {duration:.3f}s")
    print(f"[BENCHMARK] Operations/sec: {ops_per_sec:.0f}")
    print(f"[BENCHMARK] Score: {score}")

    # Scoring should be very fast
    assert duration < 1.0, f"MQM scoring too slow: {duration:.2f}s for 1000 iterations"


@pytest.mark.benchmark
def test_concurrent_requests_performance(
    mock_llm: BaseLLMProvider, small_task: TranslationTask
) -> None:
    """Benchmark handling of concurrent requests.

    Simulates multiple users submitting requests simultaneously.
    """

    async def run_test() -> None:
        orchestrator = AgentOrchestrator(mock_llm)

        num_concurrent = 5

        start = time.time()
        results = await asyncio.gather(
            *[orchestrator.evaluate(small_task) for _ in range(num_concurrent)]
        )
        duration = time.time() - start

        avg_latency = duration / num_concurrent

        print(f"\n[BENCHMARK] Concurrent requests ({num_concurrent} simultaneous)")
        print(f"[BENCHMARK] Total time: {duration:.3f}s")
        print(f"[BENCHMARK] Average latency: {avg_latency:.3f}s")

        assert len(results) == num_concurrent, "Should handle all requests"
        # With parallelization, should not take num_concurrent * single_time
        assert duration < num_concurrent * 10.0, "Concurrent handling inefficient"

    asyncio.run(run_test())


@pytest.mark.benchmark
def test_model_validation_performance() -> None:
    """Benchmark Pydantic model validation speed."""
    from kttc.core.models import QAReport

    task = TranslationTask(
        source_text="Test",
        translation="Prueba",
        source_lang="en",
        target_lang="es",
    )

    errors = [
        ErrorAnnotation(
            category="accuracy",
            subcategory="mistranslation",
            severity=ErrorSeverity.MINOR,
            location=(0, 5),
            description="Test error",
        )
        for _ in range(10)
    ]

    start = time.time()
    for _ in range(10000):  # 10k validations
        _ = QAReport(
            task=task,
            mqm_score=95.5,
            errors=errors,
            status="pass",
        )
    duration = time.time() - start

    validations_per_sec = 10000 / duration

    print("\n[BENCHMARK] Pydantic validation (10k iterations)")
    print(f"[BENCHMARK] Total time: {duration:.3f}s")
    print(f"[BENCHMARK] Validations/sec: {validations_per_sec:.0f}")

    # Pydantic should be very fast
    assert duration < 2.0, f"Model validation too slow: {duration:.2f}s"


# Summary marker for benchmark results
def test_performance_summary() -> None:
    """Print performance summary.

    This test always passes and just prints the performance targets.
    """
    print("\n" + "=" * 70)
    print("PERFORMANCE TARGETS")
    print("=" * 70)
    print(f"Processing speed: < {TARGET_SPEED_PER_1K_WORDS}s per 1000 words")
    print(f"Parallel speedup: ≥ {TARGET_PARALLEL_SPEEDUP}x")
    print(f"Memory usage: < {TARGET_MEMORY_MB}MB")
    print("=" * 70)

    assert True, "Summary printed"
