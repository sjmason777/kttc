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

"""Benchmark command for comparing LLM providers."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import typer

from kttc.agents import AgentOrchestrator
from kttc.cli.formatters import ConsoleFormatter
from kttc.cli.ui import (
    console,
    create_progress,
    print_error,
    print_info,
)
from kttc.core import TranslationTask
from kttc.llm import AnthropicProvider, BaseLLMProvider, GigaChatProvider, OpenAIProvider
from kttc.utils.config import get_settings
from kttc.utils.dependencies import require_benchmark


def _init_provider(provider_name: str, settings: Any) -> BaseLLMProvider | None:
    """Initialize a single provider instance."""
    try:
        if provider_name == "openai":
            if not settings.openai_api_key:
                print_error("OpenAI API key not configured")
                return None
            return OpenAIProvider(api_key=settings.openai_api_key, model=settings.default_model)

        if provider_name == "anthropic":
            if not settings.anthropic_api_key:
                print_error("Anthropic API key not configured")
                return None
            return AnthropicProvider(
                api_key=settings.anthropic_api_key, model=settings.default_model
            )

        if provider_name == "gigachat":
            if not settings.gigachat_client_id or not settings.gigachat_client_secret:
                print_error("GigaChat credentials not configured")
                return None
            return GigaChatProvider(
                client_id=settings.gigachat_client_id, client_secret=settings.gigachat_client_secret
            )

        print_error(f"Unknown provider: {provider_name}")
        return None

    except Exception as e:
        print_error(f"Failed to initialize {provider_name}: {e}")
        return None


async def _run_benchmarks_with_progress(
    provider_instances: dict[str, BaseLLMProvider],
    source_text: str,
    source_lang: str,
    target_lang: str,
    threshold: float,
) -> list[dict[str, Any]]:
    """Run benchmarks with progress indicator."""
    results = []
    with create_progress() as progress:
        task_id = progress.add_task("Benchmarking providers...", total=len(provider_instances))

        for provider_name, provider in provider_instances.items():
            progress.update(task_id, description=f"Testing {provider_name}...")

            result = await benchmark_provider(
                provider=provider,
                provider_name=provider_name,
                source_text=source_text,
                source_lang=source_lang,
                target_lang=target_lang,
                threshold=threshold,
            )
            results.append(result)

            if result["success"]:
                status_icon = "✓" if result["status"] == "pass" else "✗"
                status_color = "green" if result["status"] == "pass" else "red"
                errors = (
                    f"{result['critical_errors']}/{result['major_errors']}/{result['minor_errors']}"
                )
                progress.console.print(
                    f"  [{status_color}]{status_icon}[/{status_color}] "
                    f"{provider_name}: MQM {result['mqm_score']:.2f}, Errors: {errors} "
                    f"({result['duration']:.2f}s)"
                )
            else:
                progress.console.print(
                    f"  [red]✗[/red] {provider_name}: Failed - {result.get('error', 'Unknown error')}"
                )

            progress.advance(task_id)

    console.print()
    return results


async def _run_benchmarks_compact(
    provider_instances: dict[str, BaseLLMProvider],
    source_text: str,
    source_lang: str,
    target_lang: str,
    threshold: float,
) -> list[dict[str, Any]]:
    """Run benchmarks without progress indicator."""
    results = []
    for provider_name, provider in provider_instances.items():
        result = await benchmark_provider(
            provider=provider,
            provider_name=provider_name,
            source_text=source_text,
            source_lang=source_lang,
            target_lang=target_lang,
            threshold=threshold,
        )
        results.append(result)
    return results


async def benchmark_provider(
    provider: BaseLLMProvider,
    provider_name: str,
    source_text: str,
    source_lang: str,
    target_lang: str,
    threshold: float,
) -> dict[str, Any]:
    """Benchmark a single provider.

    Args:
        provider: LLM provider instance
        provider_name: Name of the provider
        source_text: Source text
        source_lang: Source language code
        target_lang: Target language code
        threshold: Quality threshold

    Returns:
        Benchmark results dictionary
    """
    start_time = time.time()

    try:
        # Generate translation
        translation_prompt = f"""Translate the following text from {source_lang} to {target_lang}.
Provide only the translation without any explanation.

Text to translate:
{source_text}

Translation:"""

        translation = await provider.complete(translation_prompt, temperature=0.3, max_tokens=2000)

        # Evaluate with agents
        task = TranslationTask(
            source_text=source_text,
            translation=translation,
            source_lang=source_lang,
            target_lang=target_lang,
        )

        orchestrator = AgentOrchestrator(provider, quality_threshold=threshold)
        report = await orchestrator.evaluate(task)

        duration = time.time() - start_time

        return {
            "name": provider_name,
            "status": report.status,
            "mqm_score": report.mqm_score,
            "error_count": len(report.errors),
            "critical_errors": report.critical_error_count,
            "major_errors": report.major_error_count,
            "minor_errors": report.minor_error_count,
            "duration": duration,
            "translation": translation,
            "success": True,
        }

    except Exception as e:
        duration = time.time() - start_time
        return {
            "name": provider_name,
            "status": "error",
            "mqm_score": 0.0,
            "error_count": 0,
            "critical_errors": 0,
            "major_errors": 0,
            "minor_errors": 0,
            "duration": duration,
            "translation": "",
            "success": False,
            "error": str(e),
        }


async def run_benchmark(
    source: str,
    source_lang: str,
    target_lang: str,
    providers: list[str],
    threshold: float,
    output: str | None,
    verbose: bool,
) -> None:
    """Run benchmark comparing multiple providers.

    Args:
        source: Source text file path
        source_lang: Source language code
        target_lang: Target language code
        providers: List of provider names to benchmark
        threshold: Quality threshold
        output: Output file path for results
        verbose: Verbose output
    """
    from kttc.cli.ui import print_available_extensions
    from kttc.utils.dependencies import has_benchmark, has_webui

    settings = get_settings()

    # Show available extensions status
    if verbose and (not has_benchmark() or not has_webui()):
        print_available_extensions()
        console.print("[dim]You can continue, but some features require extensions.[/dim]")
        console.print()

    require_benchmark("benchmark")

    # Load source text
    source_path = Path(source)
    if not source_path.exists():
        print_error(f"Source file not found: {source}")
        raise typer.Exit(code=1)

    source_text = source_path.read_text(encoding="utf-8").strip()

    # Initialize providers
    provider_instances: dict[str, BaseLLMProvider] = {}
    for provider_name in providers:
        provider_instance = _init_provider(provider_name, settings)
        if provider_instance:
            provider_instances[provider_name] = provider_instance

    if not provider_instances:
        print_error("No providers available for benchmarking")
        raise typer.Exit(code=1)

    if verbose:
        print_info(f"Initialized {len(provider_instances)} provider(s)")
        console.print()

    # Run benchmarks
    if verbose:
        results = await _run_benchmarks_with_progress(
            provider_instances, source_text, source_lang, target_lang, threshold
        )
    else:
        results = await _run_benchmarks_compact(
            provider_instances, source_text, source_lang, target_lang, threshold
        )

    # Display benchmark results using compact formatter
    ConsoleFormatter.print_benchmark_result(
        source_lang=source_lang,
        target_lang=target_lang,
        results=results,
        verbose=verbose,
    )

    # Save results if requested
    if output:
        output_path = Path(output)
        output_data = {
            "source_lang": source_lang,
            "target_lang": target_lang,
            "threshold": threshold,
            "source_text": source_text,
            "results": results,
        }
        output_path.write_text(
            json.dumps(output_data, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print_info(f"Results saved to: {output}")
