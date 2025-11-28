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

"""Self-check (proofreading) mode for translation quality assessment.

This module handles monolingual proofreading without translation,
checking text for grammar, spelling, and punctuation errors.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import typer
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from kttc.cli.ui import console, print_header
from kttc.cli.utils import auto_detect_format, setup_llm_provider
from kttc.llm import BaseLLMProvider
from kttc.utils.config import get_settings


def print_self_check_header(lang: str | None) -> None:
    """Print self-check mode Easter egg header."""
    console.print()
    console.print("[yellow]🥚 Self-Check Mode Activated![/yellow]")
    console.print('[dim]"Heal thyself before healing others" (Luke 4:23)[/dim]')
    console.print("[dim]Pro tip: Always proofread your articles about proofreading tools![/dim]")
    console.print()


def setup_self_check_llm(
    provider: str | None, settings: Any, verbose: bool, demo: bool
) -> BaseLLMProvider | None:
    """Setup LLM provider for self-check mode."""
    from kttc.cli.demo import DemoLLMProvider

    if demo:
        return DemoLLMProvider(model="demo-model")

    try:
        return setup_llm_provider(provider, settings, verbose, demo=demo)
    except Exception as e:
        if verbose:
            console.print(
                f"[yellow]⚠ LLM not available, using rule-based checks only: {e}[/yellow]\n"
            )
        return None


def run_spelling_check(language: str, text: str) -> list[Any]:
    """Run spelling check with progress indicator."""
    from kttc.agents.proofreading import SpellingAgent

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task("[cyan]Checking spelling...[/cyan]", total=None)
        spelling_agent = SpellingAgent(
            llm_provider=None,
            language=language,
            use_patterns=True,
            use_school_rules=True,
        )
        return spelling_agent.check(text)


async def run_grammar_check(
    language: str, text: str, llm_provider: BaseLLMProvider | None, existing_errors: list[Any]
) -> tuple[list[Any], list[Any]]:
    """Run grammar check with progress indicator and deduplication."""
    from kttc.agents.proofreading import GrammarAgent

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task("[cyan]Checking grammar...[/cyan]", total=None)
        grammar_agent = GrammarAgent(
            llm_provider=llm_provider,
            language=language,
            use_languagetool=True,
            use_school_rules=True,
        )
        grammar_errors = await grammar_agent.check(text)

    # Deduplicate
    all_errors = list(existing_errors)
    seen_positions: set[tuple[int, int]] = {e.location for e in all_errors}
    for err in grammar_errors:
        if err.location not in seen_positions:
            all_errors.append(err)
            seen_positions.add(err.location)

    return grammar_errors, all_errors


def calculate_self_check_score(
    all_errors: list[Any], text: str
) -> tuple[float, int, int, int, int]:
    """Calculate MQM-style score for self-check."""
    critical_count = sum(1 for e in all_errors if e.severity.value == "critical")
    major_count = sum(1 for e in all_errors if e.severity.value == "major")
    minor_count = sum(1 for e in all_errors if e.severity.value == "minor")

    penalty = critical_count * 10 + major_count * 5 + minor_count * 1
    word_count = len(text.split())
    normalized_penalty = (penalty / max(word_count, 1)) * 100
    score = max(0, 100 - normalized_penalty)

    return score, word_count, critical_count, major_count, minor_count


def display_self_check_summary(
    score: float,
    threshold: float,
    all_errors: list[Any],
    critical_count: int,
    major_count: int,
    minor_count: int,
) -> str:
    """Display self-check score summary."""
    console.print()
    score_color = "green" if score >= threshold else "yellow" if score >= 80 else "red"
    status = "PASS" if score >= threshold else "NEEDS REVISION"
    status_color = "green" if score >= threshold else "red"

    console.print(
        f"[bold {status_color}]{'✓' if score >= threshold else '✗'} {status}[/bold {status_color}]"
    )
    console.print(f"\n[bold]Quality Score:[/bold] [{score_color}]{score:.1f}[/{score_color}] / 100")
    console.print(f"[bold]Threshold:[/bold] {threshold}")
    console.print(f"\n[bold]Issues Found:[/bold] {len(all_errors)}")
    console.print(f"  Critical: {critical_count} | Major: {major_count} | Minor: {minor_count}")

    return status


def display_self_check_errors_verbose(all_errors: list[Any], text: str) -> None:
    """Display verbose error table."""
    console.print("\n[bold]Error Details:[/bold]")
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Type", style="cyan", width=12)
    table.add_column("Severity", width=10)
    table.add_column("Text", width=20)
    table.add_column("Suggestion", width=20)
    table.add_column("Description", width=40)

    for error in all_errors[:20]:
        severity_color = (
            "red"
            if error.severity.value == "critical"
            else "yellow"
            if error.severity.value == "major"
            else "dim"
        )
        start, end = error.location
        error_text = text[start:end] if start < len(text) and end <= len(text) else ""
        table.add_row(
            error.subcategory,
            f"[{severity_color}]{error.severity.value}[/{severity_color}]",
            error_text[:20],
            (error.suggestion or "")[:20],
            error.description[:40] + "..." if len(error.description) > 40 else error.description,
        )

    console.print(table)
    if len(all_errors) > 20:
        console.print(f"[dim]... and {len(all_errors) - 20} more errors[/dim]")


def _get_severity_icon(severity_value: str) -> str:
    """Map severity value to display icon."""
    severity_icons = {"critical": "🔴", "major": "🟡", "minor": "⚪"}
    return severity_icons.get(severity_value, "⚪")


def display_self_check_errors_compact(all_errors: list[Any], text: str) -> None:
    """Display compact error view."""
    console.print("\n[bold]Top Issues:[/bold]")
    for error in all_errors[:5]:
        severity_icon = _get_severity_icon(error.severity.value)
        suggestion_text = f" → '{error.suggestion}'" if error.suggestion else ""
        start, end = error.location
        error_text = text[start:end] if start < len(text) and end <= len(text) else ""
        console.print(f"  {severity_icon} '{error_text}'{suggestion_text}")
        console.print(f"     [dim]{error.description}[/dim]")

    if len(all_errors) > 5:
        console.print(f"\n[dim]Use --verbose to see all {len(all_errors)} issues[/dim]")


def save_self_check_report(
    output: str,
    output_format: str,
    language: str,
    source_path: Path,
    score: float,
    threshold: float,
    status: str,
    word_count: int,
    all_errors: list[Any],
    text: str,
    critical_count: int,
    major_count: int,
    minor_count: int,
) -> None:
    """Save self-check report to file."""
    report_data = {
        "mode": "self-check",
        "language": language,
        "file": str(source_path),
        "score": score,
        "threshold": threshold,
        "status": status.lower().replace(" ", "_"),
        "word_count": word_count,
        "errors": [
            {
                "type": e.subcategory,
                "severity": e.severity.value,
                "location": list(e.location),
                "text": text[e.location[0] : e.location[1]] if e.location[0] < len(text) else "",
                "suggestion": e.suggestion,
                "description": e.description,
            }
            for e in all_errors
        ],
        "summary": {
            "total": len(all_errors),
            "critical": critical_count,
            "major": major_count,
            "minor": minor_count,
        },
    }

    output_path = Path(output)
    output_path.write_text(json.dumps(report_data, indent=2, ensure_ascii=False), encoding="utf-8")
    console.print(f"\n[dim]Report saved to: {output}[/dim]")


async def self_check_async(
    source: str,
    language: str,
    threshold: float,
    output: str | None,
    output_format: str,
    provider: str | None,
    verbose: bool,
    demo: bool = False,
) -> None:
    """Async implementation of self-check (proofreading) mode.

    Args:
        source: Source file path to check
        language: Language code of the text
        threshold: Quality threshold (0-100)
        output: Output file path (optional)
        format: Output format
        provider: LLM provider name
        verbose: Verbose output flag
        demo: Demo mode flag
    """
    settings = get_settings()

    # Load the file
    source_path = Path(source)
    if not source_path.exists():
        raise FileNotFoundError(f"File not found: {source}")

    text = source_path.read_text(encoding="utf-8")

    if verbose:
        print_header(
            "Self-Check Mode: Proofreading",
            f"Checking {language.upper()} text for grammar, spelling, and punctuation errors",
        )
        console.print(f"[dim]Loaded {len(text)} characters from {source_path.name}[/dim]\n")

    # Setup LLM provider (optional for self-check)
    llm_provider = setup_self_check_llm(provider, settings, verbose, demo)

    # Run proofreading agents
    spelling_errors = run_spelling_check(language, text)
    console.print(f"[green]✓[/green] Spelling check: found {len(spelling_errors)} issues")

    grammar_errors, all_errors = await run_grammar_check(
        language, text, llm_provider, spelling_errors
    )
    console.print(f"[green]✓[/green] Grammar check: found {len(grammar_errors)} issues")

    # Calculate and display results
    score, word_count, critical_count, major_count, minor_count = calculate_self_check_score(
        all_errors, text
    )
    status = display_self_check_summary(
        score, threshold, all_errors, critical_count, major_count, minor_count
    )

    # Display errors
    if all_errors and verbose:
        display_self_check_errors_verbose(all_errors, text)
    elif all_errors:
        display_self_check_errors_compact(all_errors, text)

    # Save output if requested
    if output:
        save_self_check_report(
            output,
            output_format,
            language,
            source_path,
            score,
            threshold,
            status,
            word_count,
            all_errors,
            text,
            critical_count,
            major_count,
            minor_count,
        )

    # Exit code based on score
    if score < threshold:
        raise typer.Exit(code=1)


def handle_self_check_mode(
    source: str,
    source_lang: str | None,
    target_lang: str | None,
    lang: str | None,
    threshold: float,
    output: str | None,
    output_format: str | None,
    provider: str | None,
    verbose: bool,
    demo: bool,
    self_check: bool = False,
) -> bool:
    """Handle self-check mode if applicable.

    Returns:
        True if self-check mode was handled, False otherwise.
    """
    is_self_check = self_check or (source_lang and target_lang and source_lang == target_lang)
    if not is_self_check:
        return False

    # Handle --lang shortcut
    if lang:
        source_lang = lang

    if not source_lang:
        console.print("[red]Error: --lang or --source-lang required for self-check mode[/red]")
        raise typer.Exit(code=1)

    # Easter egg message
    print_self_check_header(source_lang)

    asyncio.run(
        self_check_async(
            source=source,
            language=source_lang,
            threshold=threshold,
            output=output,
            output_format=auto_detect_format(output, output_format),
            provider=provider,
            verbose=verbose,
            demo=demo,
        )
    )
    return True
