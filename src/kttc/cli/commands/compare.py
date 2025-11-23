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

"""Compare command for comparing multiple translations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import typer
from rich.table import Table

from kttc.agents import AgentOrchestrator
from kttc.cli.formatters import ConsoleFormatter
from kttc.cli.ui import (
    console,
    create_progress,
    print_error,
    print_info,
)
from kttc.core import TranslationTask
from kttc.llm import AnthropicProvider, BaseLLMProvider, OpenAIProvider
from kttc.utils.config import get_settings


async def evaluate_translation(
    source_text: str,
    translation: str,
    translation_name: str,
    source_lang: str,
    target_lang: str,
    provider: BaseLLMProvider,
    threshold: float,
) -> dict[str, Any]:
    """Evaluate a single translation.

    Args:
        source_text: Source text
        translation: Translation to evaluate
        translation_name: Name/label for this translation
        source_lang: Source language code
        target_lang: Target language code
        provider: LLM provider for evaluation
        threshold: Quality threshold

    Returns:
        Evaluation results dictionary
    """
    # Create task
    task = TranslationTask(
        source_text=source_text,
        translation=translation,
        source_lang=source_lang,
        target_lang=target_lang,
    )

    # Evaluate with agents
    orchestrator = AgentOrchestrator(provider, quality_threshold=threshold)
    report = await orchestrator.evaluate(task)

    return {
        "name": translation_name,
        "translation": translation,
        "mqm_score": report.mqm_score,
        "status": report.status,
        "error_count": len(report.errors),
        "critical_errors": report.critical_error_count,
        "major_errors": report.major_error_count,
        "minor_errors": report.minor_error_count,
        "errors": report.errors,
    }


def _load_translation_files(translations: list[str]) -> list[dict[str, Any]]:
    """Load translation files and return data."""
    translation_data = []
    for trans_file in translations:
        trans_path = Path(trans_file)
        if not trans_path.exists():
            print_error(f"Translation file not found: {trans_file}")
            continue
        trans_text = trans_path.read_text(encoding="utf-8").strip()
        translation_data.append({"name": trans_path.stem, "path": trans_file, "text": trans_text})
    return translation_data


def _setup_compare_provider(provider: str | None, settings: Any, verbose: bool) -> BaseLLMProvider:
    """Setup LLM provider for comparison."""
    provider_name = provider or settings.default_llm_provider
    api_key = settings.get_llm_provider_key(provider_name)

    if provider_name == "openai":
        llm_provider: BaseLLMProvider = OpenAIProvider(
            api_key=api_key, model=settings.default_model
        )
    elif provider_name == "anthropic":
        llm_provider = AnthropicProvider(api_key=api_key, model=settings.default_model)
    else:
        print_error(f"Unknown provider: {provider_name}")
        raise typer.Exit(code=1)

    if verbose:
        print_info(f"Using {provider_name} for evaluation")
        console.print()
    return llm_provider


def _print_compare_result(result: dict[str, Any], name: str, progress_console: Any) -> None:
    """Print single comparison result."""
    status_icon = "✓" if result["status"] == "pass" else "✗"
    status_color = "green" if result["status"] == "pass" else "red"
    errors = f"{result['critical_errors']}/{result['major_errors']}/{result['minor_errors']}"
    progress_console.print(
        f"  [{status_color}]{status_icon}[/{status_color}] {name}: MQM {result['mqm_score']:.2f}, Errors: {errors}"
    )


async def run_compare(
    source: str,
    translations: list[str],
    source_lang: str,
    target_lang: str,
    threshold: float,
    provider: str | None,
    verbose: bool,
) -> None:
    """Compare multiple translations side by side."""
    settings = get_settings()

    source_path = Path(source)
    if not source_path.exists():
        print_error(f"Source file not found: {source}")
        raise typer.Exit(code=1)
    source_text = source_path.read_text(encoding="utf-8").strip()

    translation_data = _load_translation_files(translations)
    if not translation_data:
        print_error("No valid translation files found")
        raise typer.Exit(code=1)

    try:
        llm_provider = _setup_compare_provider(provider, settings, verbose)
    except Exception as e:
        print_error(f"Failed to setup LLM provider: {e}")
        raise typer.Exit(code=1)

    results = []
    if verbose:
        with create_progress() as progress:
            task_id = progress.add_task("Evaluating translations...", total=len(translation_data))
            for trans_data in translation_data:
                progress.update(task_id, description=f"Evaluating {trans_data['name']}...")
                result = await evaluate_translation(
                    source_text=source_text,
                    translation=trans_data["text"],
                    translation_name=trans_data["name"],
                    source_lang=source_lang,
                    target_lang=target_lang,
                    provider=llm_provider,
                    threshold=threshold,
                )
                results.append(result)
                _print_compare_result(result, trans_data["name"], progress.console)
                progress.advance(task_id)
        console.print()
    else:
        for trans_data in translation_data:
            result = await evaluate_translation(
                source_text=source_text,
                translation=trans_data["text"],
                translation_name=trans_data["name"],
                source_lang=source_lang,
                target_lang=target_lang,
                provider=llm_provider,
                threshold=threshold,
            )
            results.append(result)

    ConsoleFormatter.print_compare_result(
        source_lang=source_lang,
        target_lang=target_lang,
        results=results,
        verbose=verbose,
    )


def _display_comparison_results(results: list[dict[str, Any]], verbose: bool) -> None:
    """Display comparison results in a table.

    Args:
        results: List of evaluation results
        verbose: Show detailed error information
    """
    # Main comparison table
    table = Table(
        title="Translation Comparison Results",
        show_header=True,
        header_style="bold cyan",
        title_style="bold",
    )

    table.add_column("Translation", style="cyan", no_wrap=True)
    table.add_column("MQM Score", justify="right")
    table.add_column("Errors", justify="center")
    table.add_column("Status", justify="center")

    for result in results:
        # Color-code MQM score
        mqm_score = result["mqm_score"]
        if mqm_score >= 95:
            mqm_color = "green"
        elif mqm_score >= 85:
            mqm_color = "yellow"
        else:
            mqm_color = "red"

        # Color-code status
        if result["status"] == "pass":
            status_text = "[green]✓ PASS[/green]"
        else:
            status_text = "[red]✗ FAIL[/red]"

        # Error breakdown
        error_info = f"{result['error_count']} ({result['critical_errors']}C/{result['major_errors']}M/{result['minor_errors']}m)"

        table.add_row(
            result["name"],
            f"[{mqm_color}]{mqm_score:.2f}[/{mqm_color}]",
            error_info,
            status_text,
        )

    console.print(table)

    # Show detailed errors if verbose
    if verbose:
        console.print()
        for result in results:
            if result["errors"]:
                console.print(f"\n[bold cyan]{result['name']} - Errors:[/bold cyan]")
                error_table = Table(show_header=True, header_style="bold")
                error_table.add_column("Category", style="cyan")
                error_table.add_column("Severity")
                error_table.add_column("Description", max_width=60)

                for error in result["errors"]:
                    severity_color = (
                        "red"
                        if error.severity.value == "critical"
                        else "yellow" if error.severity.value == "major" else "dim"
                    )
                    error_table.add_row(
                        error.category,
                        f"[{severity_color}]{error.severity.value}[/{severity_color}]",
                        (
                            error.description[:60] + "..."
                            if len(error.description) > 60
                            else error.description
                        ),
                    )

                console.print(error_table)
