"""Main CLI application for KTTC.

KTTC - Knowledge Translation Transmutation Core
Transforming translations into gold-standard quality.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from kttc import __version__
from kttc.agents import AgentOrchestrator
from kttc.core import QAReport, TranslationTask
from kttc.llm import AnthropicProvider, BaseLLMProvider, OpenAIProvider
from kttc.utils.config import get_settings

# Create main app and console
app = typer.Typer(
    name="kttc",
    help="Knowledge Translation Transmutation Core - Transforming translations into gold-standard quality",
    add_completion=True,
)
console = Console()


def version_callback(value: bool) -> None:
    """Print version and exit."""
    if value:
        console.print(f"KTTC version: [bold cyan]{__version__}[/bold cyan]")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        "-v",
        help="Show version and exit",
        callback=version_callback,
        is_eager=True,
    ),
) -> None:
    """
    KTTC - Knowledge Translation Transmutation Core.

    Transforming translations into gold-standard quality through
    autonomous multi-agent AI systems.
    """
    pass


@app.command()
def check(
    source: str = typer.Option(..., "--source", "-s", help="Source text file path"),
    translation: str = typer.Option(..., "--translation", "-t", help="Translation file path"),
    source_lang: str = typer.Option(..., "--source-lang", help="Source language code (e.g., 'en')"),
    target_lang: str = typer.Option(..., "--target-lang", help="Target language code (e.g., 'es')"),
    threshold: float = typer.Option(95.0, "--threshold", help="Minimum MQM score to pass (0-100)"),
    output: str | None = typer.Option(None, "--output", "-o", help="Output file path (JSON)"),
    format: str = typer.Option(
        "text", "--format", "-f", help="Output format: text, json, or markdown"
    ),
    provider: str | None = typer.Option(
        None, "--provider", help="LLM provider (openai or anthropic)"
    ),
    verbose: bool = typer.Option(False, "--verbose", help="Verbose output"),
) -> None:
    """
    Check translation quality.

    Evaluates a translation using multi-agent QA system and provides
    MQM score, error annotations, and pass/fail status.

    Example:
        kttc check --source source.txt --translation trans.txt \\
                   --source-lang en --target-lang es --threshold 95
    """
    # Run async function
    try:
        asyncio.run(
            _check_async(
                source,
                translation,
                source_lang,
                target_lang,
                threshold,
                output,
                format,
                provider,
                verbose,
            )
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠ Interrupted by user[/yellow]")
        raise typer.Exit(code=130)
    except Exception as e:
        console.print(f"\n[red]✗ Error: {e}[/red]")
        if verbose:
            console.print_exception()
        raise typer.Exit(code=1)


async def _check_async(
    source: str,
    translation: str,
    source_lang: str,
    target_lang: str,
    threshold: float,
    output: str | None,
    format: str,
    provider: str | None,
    verbose: bool,
) -> None:
    """Async implementation of check command."""
    # Load settings
    settings = get_settings()

    # Display header
    console.print("\n[bold]KTTC - Translation Quality Check[/bold]\n")
    console.print(f"Source file:      [cyan]{source}[/cyan]")
    console.print(f"Translation file: [cyan]{translation}[/cyan]")
    console.print(f"Languages:        [cyan]{source_lang}[/cyan] → [cyan]{target_lang}[/cyan]")
    console.print(f"Threshold:        [cyan]{threshold}[/cyan]")
    console.print()

    # Load files
    try:
        source_path = Path(source)
        translation_path = Path(translation)

        if not source_path.exists():
            raise FileNotFoundError(f"Source file not found: {source}")
        if not translation_path.exists():
            raise FileNotFoundError(f"Translation file not found: {translation}")

        source_text = source_path.read_text(encoding="utf-8")
        translation_text = translation_path.read_text(encoding="utf-8")

        if verbose:
            console.print(f"[dim]Loaded {len(source_text)} chars from source[/dim]")
            console.print(f"[dim]Loaded {len(translation_text)} chars from translation[/dim]\n")

    except Exception as e:
        raise RuntimeError(f"Failed to load files: {e}") from e

    # Create translation task
    try:
        task = TranslationTask(
            source_text=source_text,
            translation=translation_text,
            source_lang=source_lang,
            target_lang=target_lang,
        )
        if verbose:
            console.print(f"[dim]Created task with {task.word_count} words[/dim]\n")
    except Exception as e:
        raise RuntimeError(f"Failed to create translation task: {e}") from e

    # Setup LLM provider
    try:
        provider_name = provider or settings.default_llm_provider
        api_key = settings.get_llm_provider_key(provider_name)

        llm_provider: BaseLLMProvider
        if provider_name == "openai":
            llm_provider = OpenAIProvider(api_key=api_key, model=settings.default_model)
        elif provider_name == "anthropic":
            llm_provider = AnthropicProvider(api_key=api_key, model=settings.default_model)
        else:
            raise ValueError(f"Unknown provider: {provider_name}")

        if verbose:
            console.print(f"[dim]Using {provider_name} provider[/dim]\n")
    except Exception as e:
        raise RuntimeError(f"Failed to setup LLM provider: {e}") from e

    # Run evaluation
    console.print("[yellow]⏳ Running QA agents...[/yellow]")
    try:
        orchestrator = AgentOrchestrator(
            llm_provider,
            quality_threshold=threshold,
            agent_temperature=settings.default_temperature,
            agent_max_tokens=settings.default_max_tokens,
        )
        report = await orchestrator.evaluate(task)
    except Exception as e:
        raise RuntimeError(f"Evaluation failed: {e}") from e

    # Display results
    _display_report(report, format, verbose)

    # Save output if requested
    if output:
        _save_report(report, output, format)
        console.print(f"\n[dim]Report saved to: {output}[/dim]")

    # Exit with appropriate code
    if report.status == "fail":
        raise typer.Exit(code=1)


async def _batch_async(
    source_dir: str,
    translation_dir: str,
    source_lang: str,
    target_lang: str,
    threshold: float,
    output: str,
    parallel: int,
    provider: str | None,
    verbose: bool,
) -> None:
    """Async implementation of batch command."""
    # Load settings
    settings = get_settings()

    # Display header
    console.print("\n[bold]KTTC - Batch Translation Quality Check[/bold]\n")
    console.print(f"Source directory:      [cyan]{source_dir}[/cyan]")
    console.print(f"Translation directory: [cyan]{translation_dir}[/cyan]")
    console.print(f"Languages:             [cyan]{source_lang}[/cyan] → [cyan]{target_lang}[/cyan]")
    console.print(f"Threshold:             [cyan]{threshold}[/cyan]")
    console.print(f"Parallel workers:      [cyan]{parallel}[/cyan]")
    console.print()

    # Scan directories
    try:
        source_path = Path(source_dir)
        translation_path = Path(translation_dir)

        if not source_path.exists() or not source_path.is_dir():
            raise FileNotFoundError(f"Source directory not found: {source_dir}")
        if not translation_path.exists() or not translation_path.is_dir():
            raise FileNotFoundError(f"Translation directory not found: {translation_dir}")

        # Find matching files
        source_files = sorted(source_path.glob("*.txt"))
        if not source_files:
            raise ValueError(f"No .txt files found in source directory: {source_dir}")

        # Match source and translation files
        file_pairs: list[tuple[Path, Path]] = []
        for source_file in source_files:
            translation_file = translation_path / source_file.name
            if translation_file.exists():
                file_pairs.append((source_file, translation_file))
            elif verbose:
                console.print(
                    f"[yellow]⚠ Skipping {source_file.name}: no matching translation[/yellow]"
                )

        if not file_pairs:
            raise ValueError("No matching source-translation file pairs found")

        console.print(f"Found [cyan]{len(file_pairs)}[/cyan] file pairs to process\n")

    except Exception as e:
        raise RuntimeError(f"Failed to scan directories: {e}") from e

    # Setup LLM provider
    try:
        provider_name = provider or settings.default_llm_provider
        api_key = settings.get_llm_provider_key(provider_name)

        llm_provider: BaseLLMProvider
        if provider_name == "openai":
            llm_provider = OpenAIProvider(api_key=api_key, model=settings.default_model)
        elif provider_name == "anthropic":
            llm_provider = AnthropicProvider(api_key=api_key, model=settings.default_model)
        else:
            raise ValueError(f"Unknown provider: {provider_name}")

        if verbose:
            console.print(f"[dim]Using {provider_name} provider[/dim]\n")
    except Exception as e:
        raise RuntimeError(f"Failed to setup LLM provider: {e}") from e

    # Create orchestrator
    orchestrator = AgentOrchestrator(
        llm_provider,
        quality_threshold=threshold,
        agent_temperature=settings.default_temperature,
        agent_max_tokens=settings.default_max_tokens,
    )

    # Process files in parallel
    console.print("[yellow]⏳ Processing translations...[/yellow]\n")

    results: list[tuple[str, QAReport]] = []
    semaphore = asyncio.Semaphore(parallel)

    async def process_file(source_file: Path, translation_file: Path) -> tuple[str, QAReport]:
        """Process a single file pair."""
        async with semaphore:
            # Load files
            source_text = source_file.read_text(encoding="utf-8")
            translation_text = translation_file.read_text(encoding="utf-8")

            # Create task
            task = TranslationTask(
                source_text=source_text,
                translation=translation_text,
                source_lang=source_lang,
                target_lang=target_lang,
            )

            # Evaluate
            report = await orchestrator.evaluate(task)
            return source_file.name, report

    # Process all files with progress tracking
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task_id = progress.add_task("Processing files...", total=len(file_pairs))

        tasks = [process_file(src, trg) for src, trg in file_pairs]
        for coro in asyncio.as_completed(tasks):
            try:
                filename, report = await coro
                results.append((filename, report))
                status_icon = "✓" if report.status == "pass" else "✗"
                status_color = "green" if report.status == "pass" else "red"
                progress.console.print(
                    f"  [{status_color}]{status_icon}[/{status_color}] "
                    f"{filename}: {report.mqm_score:.2f}"
                )
                progress.advance(task_id)
            except Exception as e:
                progress.console.print(f"  [red]✗ Error processing file: {e}[/red]")
                progress.advance(task_id)

    # Generate aggregated report
    console.print()
    _display_batch_summary(results, threshold)

    # Save detailed report
    _save_batch_report(results, output, threshold)
    console.print(f"\n[dim]Detailed report saved to: {output}[/dim]")

    # Exit with appropriate code
    failed_count = sum(1 for _, report in results if report.status == "fail")
    if failed_count > 0:
        raise typer.Exit(code=1)


def _display_batch_summary(results: list[tuple[str, QAReport]], threshold: float) -> None:
    """Display summary of batch processing results."""
    total = len(results)
    passed = sum(1 for _, report in results if report.status == "pass")
    failed = total - passed

    avg_score = sum(report.mqm_score for _, report in results) / total if total > 0 else 0.0
    total_errors = sum(len(report.errors) for _, report in results)

    console.print("[bold]Batch Processing Summary[/bold]\n")
    console.print(f"Total files:     [cyan]{total}[/cyan]")
    console.print(f"Passed:          [green]{passed}[/green]")
    console.print(f"Failed:          [red]{failed}[/red]")
    console.print(f"Average score:   [cyan]{avg_score:.2f}[/cyan]")
    console.print(f"Total errors:    [cyan]{total_errors}[/cyan]")

    # Overall status
    if failed == 0:
        console.print("\n[bold green]✓ ALL PASS[/bold green]")
    else:
        console.print(f"\n[bold red]✗ {failed} FILE(S) FAILED[/bold red]")


def _save_batch_report(results: list[tuple[str, QAReport]], output: str, threshold: float) -> None:
    """Save batch processing report to JSON file."""
    output_path = Path(output)

    # Prepare aggregated data
    data = {
        "summary": {
            "total_files": len(results),
            "passed": sum(1 for _, r in results if r.status == "pass"),
            "failed": sum(1 for _, r in results if r.status == "fail"),
            "average_score": (
                sum(r.mqm_score for _, r in results) / len(results) if results else 0.0
            ),
            "total_errors": sum(len(r.errors) for _, r in results),
            "threshold": threshold,
        },
        "files": [
            {
                "filename": filename,
                "status": report.status,
                "mqm_score": report.mqm_score,
                "error_count": len(report.errors),
                "errors": [
                    {
                        "category": e.category,
                        "subcategory": e.subcategory,
                        "severity": e.severity.value,
                        "location": e.location,
                        "description": e.description,
                    }
                    for e in report.errors
                ],
            }
            for filename, report in results
        ],
    }

    output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _display_report(report: QAReport, format: str, verbose: bool) -> None:
    """Display QA report in specified format."""
    console.print()

    # Status badge
    if report.status == "pass":
        console.print("[bold green]✓ PASS[/bold green]")
    else:
        console.print("[bold red]✗ FAIL[/bold red]")

    # MQM Score
    score_color = (
        "green" if report.mqm_score >= 95 else "yellow" if report.mqm_score >= 85 else "red"
    )
    console.print(
        f"\nMQM Score: [bold {score_color}]{report.mqm_score:.2f}[/bold {score_color}]/100"
    )

    # Error summary
    console.print(f"\nErrors found: [bold]{len(report.errors)}[/bold]")
    if report.errors:
        console.print(
            f"  Critical: {report.critical_error_count} | "
            f"Major: {report.major_error_count} | "
            f"Minor: {report.minor_error_count}"
        )

    # Error details
    if report.errors and verbose:
        console.print("\n[bold]Error Details:[/bold]")
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Category", style="cyan")
        table.add_column("Subcategory")
        table.add_column("Severity", style="yellow")
        table.add_column("Location")
        table.add_column("Description")

        for error in report.errors:
            severity_color = (
                "red"
                if error.severity.value == "critical"
                else "yellow" if error.severity.value == "major" else "dim"
            )
            table.add_row(
                error.category,
                error.subcategory,
                f"[{severity_color}]{error.severity.value}[/{severity_color}]",
                f"{error.location[0]}-{error.location[1]}",
                (
                    error.description[:60] + "..."
                    if len(error.description) > 60
                    else error.description
                ),
            )
        console.print(table)


def _save_report(report: QAReport, output: str, format: str) -> None:
    """Save report to file."""
    output_path = Path(output)

    if format == "json" or output.endswith(".json"):
        # Save as JSON
        data = report.model_dump(mode="json")
        output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    elif format == "markdown" or output.endswith(".md"):
        # Save as Markdown
        md_content = _generate_markdown_report(report)
        output_path.write_text(md_content, encoding="utf-8")
    else:
        # Default to JSON
        data = report.model_dump(mode="json")
        output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _generate_markdown_report(report: QAReport) -> str:
    """Generate Markdown report from QA report."""
    lines = [
        "# Translation Quality Report",
        "",
        f"**Status:** {'✓ PASS' if report.status == 'pass' else '✗ FAIL'}",
        f"**MQM Score:** {report.mqm_score:.2f}/100",
        f"**Errors Found:** {len(report.errors)}",
        "",
        "## Summary",
        "",
        f"- Critical Errors: {report.critical_error_count}",
        f"- Major Errors: {report.major_error_count}",
        f"- Minor Errors: {report.minor_error_count}",
        "",
    ]

    if report.errors:
        lines.extend(
            [
                "## Error Details",
                "",
                "| Category | Subcategory | Severity | Location | Description |",
                "|----------|-------------|----------|----------|-------------|",
            ]
        )

        for error in report.errors:
            desc = error.description.replace("|", "\\|")[:80]
            lines.append(
                f"| {error.category} | {error.subcategory} | {error.severity.value} | "
                f"{error.location[0]}-{error.location[1]} | {desc} |"
            )

    return "\n".join(lines)


@app.command()
def translate(
    text: str = typer.Option(..., "--text", help="Text to translate (or file path with @)"),
    source_lang: str = typer.Option(..., "--source-lang", help="Source language code"),
    target_lang: str = typer.Option(..., "--target-lang", help="Target language code"),
    threshold: float = typer.Option(
        95.0, "--threshold", help="Quality threshold for auto-refinement"
    ),
    max_iterations: int = typer.Option(3, "--max-iterations", help="Maximum refinement iterations"),
    output: str = typer.Option(None, "--output", "-o", help="Output file path"),
) -> None:
    """
    Translate text with automatic quality checking.

    Uses TEaR (Translate-Estimate-Refine) loop to generate and
    iteratively improve translation until quality threshold is met.

    Example:
        kttc translate --text "Hello world" \\
                      --source-lang en --target-lang es --threshold 95
    """
    console.print("[yellow]Translating with quality assurance...[/yellow]")
    console.print(f"Source language: [cyan]{source_lang}[/cyan]")
    console.print(f"Target language: [cyan]{target_lang}[/cyan]")
    console.print(f"Quality threshold: [cyan]{threshold}[/cyan]")
    console.print(f"Max iterations: [cyan]{max_iterations}[/cyan]")

    # TODO: Implement translation with TEaR loop
    console.print("\n[red]⚠ Not implemented yet - coming in future phases![/red]")


@app.command()
def batch(
    source_dir: str = typer.Option(..., "--source-dir", help="Directory with source files"),
    translation_dir: str = typer.Option(
        ..., "--translation-dir", help="Directory with translation files"
    ),
    source_lang: str = typer.Option(..., "--source-lang", help="Source language code"),
    target_lang: str = typer.Option(..., "--target-lang", help="Target language code"),
    threshold: float = typer.Option(95.0, "--threshold", help="Minimum MQM score to pass"),
    output: str = typer.Option("report.json", "--output", "-o", help="Output report file path"),
    parallel: int = typer.Option(4, "--parallel", help="Number of parallel workers"),
    provider: str | None = typer.Option(
        None, "--provider", help="LLM provider (openai or anthropic)"
    ),
    verbose: bool = typer.Option(False, "--verbose", help="Verbose output"),
) -> None:
    """
    Batch process multiple translation files.

    Evaluates multiple translations in parallel and generates
    aggregated quality report.

    Example:
        kttc batch --source-dir ./source --translation-dir ./translations \\
                   --source-lang en --target-lang es
    """
    try:
        asyncio.run(
            _batch_async(
                source_dir,
                translation_dir,
                source_lang,
                target_lang,
                threshold,
                output,
                parallel,
                provider,
                verbose,
            )
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠ Interrupted by user[/yellow]")
        raise typer.Exit(code=130)
    except Exception as e:
        console.print(f"\n[red]✗ Error: {e}[/red]")
        if verbose:
            console.print_exception()
        raise typer.Exit(code=1)


@app.command()
def report(
    input_file: str = typer.Argument(..., help="Input report file (JSON)"),
    format: str = typer.Option(
        "markdown", "--format", "-f", help="Output format: markdown or html"
    ),
    output: str | None = typer.Option(None, "--output", "-o", help="Output file path"),
) -> None:
    """
    Generate formatted report from QA results.

    Converts JSON QA results into human-readable formats.

    Example:
        kttc report results.json --format markdown -o report.md
    """
    try:
        # Display header
        console.print(f"\n[bold]Generating {format} report...[/bold]")
        console.print(f"Input: [cyan]{input_file}[/cyan]\n")

        # Load JSON report
        input_path = Path(input_file)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")

        data = json.loads(input_path.read_text(encoding="utf-8"))

        # Determine output path
        if output:
            output_path = Path(output)
        else:
            # Auto-generate output filename
            ext = ".md" if format == "markdown" else ".html"
            output_path = input_path.with_suffix(ext)

        # Generate report
        if format == "markdown":
            content = _generate_batch_markdown_report(data)
        elif format == "html":
            content = _generate_batch_html_report(data)
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'markdown' or 'html'")

        # Save report
        output_path.write_text(content, encoding="utf-8")

        console.print("[green]✓ Report generated successfully[/green]")
        console.print(f"Output: [cyan]{output_path}[/cyan]")

    except Exception as e:
        console.print(f"\n[red]✗ Error: {e}[/red]")
        raise typer.Exit(code=1)


def _generate_batch_markdown_report(data: dict[str, Any]) -> str:
    """Generate Markdown report from batch processing data."""
    summary = data.get("summary", {})
    files = data.get("files", [])

    lines = [
        "# Batch Translation Quality Report",
        "",
        "## Summary",
        "",
        f"**Total Files:** {summary.get('total_files', 0)}",
        f"**Passed:** {summary.get('passed', 0)}",
        f"**Failed:** {summary.get('failed', 0)}",
        f"**Average MQM Score:** {summary.get('average_score', 0):.2f}/100",
        f"**Total Errors:** {summary.get('total_errors', 0)}",
        f"**Quality Threshold:** {summary.get('threshold', 95.0)}",
        "",
        "## File Results",
        "",
        "| File | Status | MQM Score | Errors |",
        "|------|--------|-----------|--------|",
    ]

    for file_data in files:
        status_icon = "✓" if file_data["status"] == "pass" else "✗"
        lines.append(
            f"| {file_data['filename']} | {status_icon} {file_data['status']} | "
            f"{file_data['mqm_score']:.2f} | {file_data['error_count']} |"
        )

    # Add detailed errors section
    has_errors = any(file_data["errors"] for file_data in files)
    if has_errors:
        lines.extend(["", "## Detailed Errors", ""])

        for file_data in files:
            if file_data["errors"]:
                lines.append(f"### {file_data['filename']}")
                lines.append("")
                lines.append("| Category | Subcategory | Severity | Location | Description |")
                lines.append("|----------|-------------|----------|----------|-------------|")

                for error in file_data["errors"]:
                    desc = error["description"].replace("|", "\\|")[:80]
                    location = f"{error['location'][0]}-{error['location'][1]}"
                    lines.append(
                        f"| {error['category']} | {error['subcategory']} | "
                        f"{error['severity']} | {location} | {desc} |"
                    )
                lines.append("")

    return "\n".join(lines)


def _generate_batch_html_report(data: dict[str, Any]) -> str:
    """Generate HTML report from batch processing data."""
    summary = data.get("summary", {})
    files = data.get("files", [])

    # HTML template with embedded CSS
    html_parts = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        "    <meta charset='utf-8'>",
        "    <title>KTTC Batch Quality Report</title>",
        "    <style>",
        "        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }",
        "        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; }",
        "        h1 { color: #333; border-bottom: 3px solid #4CAF50; padding-bottom: 10px; }",
        "        h2 { color: #666; margin-top: 30px; }",
        "        .summary { display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; margin: 20px 0; }",
        "        .summary-card { background: #f9f9f9; padding: 15px; border-radius: 5px; border-left: 4px solid #4CAF50; }",
        "        .summary-card.fail { border-left-color: #f44336; }",
        "        .summary-card h3 { margin: 0; color: #666; font-size: 14px; }",
        "        .summary-card .value { font-size: 32px; font-weight: bold; color: #333; margin: 10px 0; }",
        "        table { width: 100%; border-collapse: collapse; margin: 20px 0; }",
        "        th { background: #4CAF50; color: white; padding: 12px; text-align: left; }",
        "        td { padding: 10px; border-bottom: 1px solid #ddd; }",
        "        tr:hover { background: #f5f5f5; }",
        "        .pass { color: #4CAF50; font-weight: bold; }",
        "        .fail { color: #f44336; font-weight: bold; }",
        "        .error-detail { background: #fff3cd; padding: 10px; margin: 10px 0; border-radius: 4px; border-left: 4px solid #ffc107; }",
        "        .severity-critical { color: #f44336; font-weight: bold; }",
        "        .severity-major { color: #ff9800; font-weight: bold; }",
        "        .severity-minor { color: #2196F3; }",
        "    </style>",
        "</head>",
        "<body>",
        "    <div class='container'>",
        "        <h1>Batch Translation Quality Report</h1>",
        "",
        "        <h2>Summary</h2>",
        "        <div class='summary'>",
        f"            <div class='summary-card'><h3>Total Files</h3><div class='value'>{summary.get('total_files', 0)}</div></div>",
        f"            <div class='summary-card'><h3>Passed</h3><div class='value'>{summary.get('passed', 0)}</div></div>",
        f"            <div class='summary-card fail'><h3>Failed</h3><div class='value'>{summary.get('failed', 0)}</div></div>",
        f"            <div class='summary-card'><h3>Average Score</h3><div class='value'>{summary.get('average_score', 0):.1f}</div></div>",
        f"            <div class='summary-card'><h3>Total Errors</h3><div class='value'>{summary.get('total_errors', 0)}</div></div>",
        f"            <div class='summary-card'><h3>Threshold</h3><div class='value'>{summary.get('threshold', 95.0):.0f}</div></div>",
        "        </div>",
        "",
        "        <h2>File Results</h2>",
        "        <table>",
        "            <tr><th>File</th><th>Status</th><th>MQM Score</th><th>Errors</th></tr>",
    ]

    for file_data in files:
        status_class = "pass" if file_data["status"] == "pass" else "fail"
        status_icon = "✓" if file_data["status"] == "pass" else "✗"
        html_parts.append(
            f"            <tr>"
            f"<td>{file_data['filename']}</td>"
            f"<td class='{status_class}'>{status_icon} {file_data['status'].upper()}</td>"
            f"<td>{file_data['mqm_score']:.2f}</td>"
            f"<td>{file_data['error_count']}</td>"
            f"</tr>"
        )

    html_parts.append("        </table>")

    # Add detailed errors
    has_errors = any(file_data["errors"] for file_data in files)
    if has_errors:
        html_parts.append("        <h2>Detailed Errors</h2>")

        for file_data in files:
            if file_data["errors"]:
                html_parts.append(f"        <h3>{file_data['filename']}</h3>")
                html_parts.append("        <table>")
                html_parts.append(
                    "            <tr><th>Category</th><th>Subcategory</th><th>Severity</th><th>Location</th><th>Description</th></tr>"
                )

                for error in file_data["errors"]:
                    severity_class = f"severity-{error['severity']}"
                    location = f"{error['location'][0]}-{error['location'][1]}"
                    html_parts.append(
                        f"            <tr>"
                        f"<td>{error['category']}</td>"
                        f"<td>{error['subcategory']}</td>"
                        f"<td class='{severity_class}'>{error['severity'].upper()}</td>"
                        f"<td>{location}</td>"
                        f"<td>{error['description']}</td>"
                        f"</tr>"
                    )

                html_parts.append("        </table>")

    html_parts.extend(["    </div>", "</body>", "</html>"])

    return "\n".join(html_parts)


def run() -> None:
    """Entry point for CLI."""
    app()


if __name__ == "__main__":
    run()
