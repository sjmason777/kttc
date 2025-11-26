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

"""Batch command for processing multiple translations."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import typer
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from kttc.agents import AgentOrchestrator
from kttc.cli.formatters import ConsoleFormatter
from kttc.cli.ui import console, print_header, print_startup_info
from kttc.cli.utils import setup_llm_provider
from kttc.core import BatchFileParser, BatchGrouper, QAReport, TranslationTask
from kttc.utils.config import get_settings

# Create Typer app for batch command
batch_app = typer.Typer()


def _scan_batch_directories(
    source_dir: str, translation_dir: str, verbose: bool
) -> list[tuple[Path, Path]]:
    """Scan directories and find matching source-translation file pairs.

    Args:
        source_dir: Path to source files directory
        translation_dir: Path to translation files directory
        verbose: Whether to show verbose output

    Returns:
        List of (source_file, translation_file) path pairs

    Raises:
        FileNotFoundError: If directories don't exist
        ValueError: If no matching pairs found
    """
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

    return file_pairs


async def _process_batch_files(
    file_pairs: list[tuple[Path, Path]],
    orchestrator: AgentOrchestrator,
    source_lang: str,
    target_lang: str,
    parallel: int,
) -> list[tuple[str, QAReport]]:
    """Process file pairs in parallel and collect results."""
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

    return results


def _load_and_apply_glossaries(glossary: str | None, translations: list[Any]) -> None:
    """Load glossaries and apply terms to translations."""
    if not glossary:
        return

    from kttc.core import GlossaryManager

    try:
        glossary_manager = GlossaryManager()
        glossary_names = [g.strip() for g in glossary.split(",")]
        glossary_manager.load_multiple(glossary_names)
        console.print(f"[green]✓[/green] Loaded {len(glossary_names)} glossaries\n")

        for trans in translations:
            terms = glossary_manager.find_in_text(
                trans.source_text, trans.source_lang, trans.target_lang
            )
            if terms:
                trans.context = trans.context or {}
                trans.context["glossary_terms"] = [
                    {"source": t.source, "target": t.target, "do_not_translate": t.do_not_translate}
                    for t in terms
                ]
    except Exception as e:
        console.print(f"[yellow]⚠ Warning: Failed to load glossaries: {e}[/yellow]\n")


def _build_batch_config_info(
    file_path: str,
    translations: list[Any],
    groups: dict[tuple[str, str], list[Any]],
    threshold: float,
    parallel: int,
    batch_size: int | None,
    smart_routing: bool,
    glossary: str | None,
) -> dict[str, str]:
    """Build configuration info dictionary for display."""
    config_info = {
        "Input File": file_path,
        "Total Translations": f"{len(translations)}",
        "Language Pairs": f"{len(groups)}",
        "Quality Threshold": f"{threshold}",
        "Parallel Workers": f"{parallel}",
    }
    if batch_size:
        config_info["Batch Size"] = f"{batch_size}"
    if smart_routing:
        config_info["Smart Routing"] = "Enabled"
    if glossary:
        config_info["Glossaries"] = glossary
    return config_info


def _create_batch_progress(_total: int) -> Any:  # noqa: ARG001
    """Create progress bar for batch processing."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("•"),
        TextColumn("{task.completed}/{task.total} translations"),
        TimeElapsedColumn(),
        console=console,
    )


def _get_batch_identifier(idx: int, batch_translation: Any) -> str:
    """Get identifier string for batch translation."""
    identifier = f"#{idx + 1}"
    if batch_translation.metadata and "file" in batch_translation.metadata:
        identifier = f"{Path(batch_translation.metadata['file']).name}:#{idx + 1}"
    return identifier


def _print_batch_result(identifier: str, report: QAReport, output_console: Any) -> None:
    """Print single batch result to console."""
    status_icon = "✓" if report.status == "pass" else "✗"
    status_color = "green" if report.status == "pass" else "red"
    output_console.print(
        f"  [{status_color}]{status_icon}[/{status_color}] {identifier}: {report.mqm_score:.2f}"
    )


async def _process_batch_translations(
    translations: list[Any],
    orchestrator: AgentOrchestrator,
    parallel: int,
    verbose: bool = False,
    show_progress: bool = False,
) -> list[tuple[str, QAReport]]:
    """Process batch translations in parallel."""
    results: list[tuple[str, QAReport]] = []
    semaphore = asyncio.Semaphore(parallel)

    progress = _create_batch_progress(len(translations)) if show_progress else None
    task_id = (
        progress.add_task("[cyan]Processing translations...", total=len(translations))
        if progress
        else None
    )

    async def process_one(idx: int, batch_translation: Any) -> tuple[str, QAReport]:
        async with semaphore:
            task = batch_translation.to_task()
            report = await orchestrator.evaluate(task)
            if progress and task_id is not None:
                progress.update(task_id, advance=1)
            return _get_batch_identifier(idx, batch_translation), report

    tasks = [process_one(idx, t) for idx, t in enumerate(translations)]
    output_console = progress.console if progress else console

    if progress:
        with progress:
            for coro in asyncio.as_completed(tasks):
                try:
                    identifier, report = await coro
                    results.append((identifier, report))
                    if not verbose:
                        _print_batch_result(identifier, report, output_console)
                except Exception as e:
                    output_console.print(f"  [red]✗ Error processing translation: {e}[/red]")
    else:
        for coro in asyncio.as_completed(tasks):
            try:
                identifier, report = await coro
                results.append((identifier, report))
                _print_batch_result(identifier, report, output_console)
            except Exception as e:
                output_console.print(f"  [red]✗ Error processing translation: {e}[/red]")

    return results


def _display_batch_summary(results: list[tuple[str, QAReport]], threshold: float) -> None:
    """Display summary of batch processing results."""
    total = len(results)
    passed = sum(1 for _, report in results if report.status == "pass")
    failed = total - passed

    avg_score = sum(report.mqm_score for _, report in results) / total if total > 0 else 0.0
    total_errors = sum(len(report.errors) for _, report in results)

    # Use compact formatter
    ConsoleFormatter.print_batch_result(
        total=total,
        passed=passed,
        failed=failed,
        avg_score=avg_score,
        total_errors=total_errors,
        verbose=False,
    )


def _save_batch_report(
    results: list[tuple[str, QAReport]],
    output: str,
    threshold: float,
    output_format: str | None = None,
) -> None:
    """Save batch processing report to file."""
    output_path = Path(output)

    # Auto-detect format from extension if not specified
    if output_format is None:
        ext = output_path.suffix.lower()
        format_map = {
            ".json": "json",
            ".txt": "text",
            ".md": "markdown",
            ".html": "html",
            ".htm": "html",
        }
        output_format = format_map.get(ext, "json")

    # Normalize format
    output_format = output_format.lower()
    if output_format not in ("json", "text", "markdown", "html"):
        console.print(f"[yellow]Warning: Unknown format '{output_format}', using json[/yellow]")
        output_format = "json"

    # Prepare aggregated data
    total = len(results)
    passed = sum(1 for _, r in results if r.status == "pass")
    failed = total - passed
    avg_score = sum(r.mqm_score for _, r in results) / total if total > 0 else 0.0
    total_errors = sum(len(r.errors) for _, r in results)

    if output_format == "json":
        data = {
            "summary": {
                "total_files": total,
                "passed": passed,
                "failed": failed,
                "average_score": avg_score,
                "total_errors": total_errors,
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

    elif output_format == "text":
        lines = [
            "=" * 60,
            "BATCH TRANSLATION QUALITY REPORT",
            "=" * 60,
            "",
            "SUMMARY",
            "-" * 40,
            f"Total files:    {total}",
            f"Passed:         {passed}",
            f"Failed:         {failed}",
            f"Average score:  {avg_score:.2f}",
            f"Total errors:   {total_errors}",
            f"Threshold:      {threshold}",
            "",
            "DETAILED RESULTS",
            "-" * 40,
        ]
        for filename, report in results:
            status_mark = "PASS" if report.status == "pass" else "FAIL"
            lines.append(f"\n[{status_mark}] {filename}")
            lines.append(f"  Score: {report.mqm_score:.2f} | Errors: {len(report.errors)}")
            for e in report.errors:
                lines.append(
                    f"    - [{e.severity.value}] {e.category}/{e.subcategory}: {e.description}"
                )
        output_path.write_text("\n".join(lines), encoding="utf-8")

    elif output_format == "markdown":
        lines = [
            "# Batch Translation Quality Report",
            "",
            "## Summary",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Total files | {total} |",
            f"| Passed | {passed} |",
            f"| Failed | {failed} |",
            f"| Average score | {avg_score:.2f} |",
            f"| Total errors | {total_errors} |",
            f"| Threshold | {threshold} |",
            "",
            "## Detailed Results",
            "",
        ]
        for filename, report in results:
            status_emoji = "✅" if report.status == "pass" else "❌"
            lines.append(f"### {status_emoji} {filename}")
            lines.append(f"**Score:** {report.mqm_score:.2f} | **Errors:** {len(report.errors)}")
            lines.append("")
            if report.errors:
                lines.append("| Severity | Category | Description |")
                lines.append("|----------|----------|-------------|")
                for e in report.errors:
                    desc = e.description.replace("|", "\\|")
                    lines.append(f"| {e.severity.value} | {e.category}/{e.subcategory} | {desc} |")
                lines.append("")
        output_path.write_text("\n".join(lines), encoding="utf-8")

    elif output_format == "html":
        passed_pct = (passed / total * 100) if total > 0 else 0
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Batch Translation Quality Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 40px; background: #f5f5f5; }}
        .container {{ max-width: 1000px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; border-bottom: 2px solid #007bff; padding-bottom: 10px; }}
        h2 {{ color: #555; margin-top: 30px; }}
        .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 15px; margin: 20px 0; }}
        .summary-card {{ background: #f8f9fa; padding: 15px; border-radius: 6px; text-align: center; }}
        .summary-card .value {{ font-size: 24px; font-weight: bold; color: #007bff; }}
        .summary-card .label {{ font-size: 12px; color: #666; text-transform: uppercase; }}
        .file-card {{ border: 1px solid #ddd; border-radius: 6px; margin: 15px 0; overflow: hidden; }}
        .file-header {{ padding: 12px 15px; background: #f8f9fa; font-weight: bold; display: flex; justify-content: space-between; }}
        .file-header.pass {{ border-left: 4px solid #28a745; }}
        .file-header.fail {{ border-left: 4px solid #dc3545; }}
        .file-body {{ padding: 15px; }}
        .error-table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
        .error-table th, .error-table td {{ padding: 8px; text-align: left; border-bottom: 1px solid #eee; }}
        .error-table th {{ background: #f8f9fa; }}
        .severity-critical {{ color: #dc3545; font-weight: bold; }}
        .severity-major {{ color: #fd7e14; }}
        .severity-minor {{ color: #ffc107; }}
        .pass-badge {{ background: #28a745; color: white; padding: 2px 8px; border-radius: 4px; }}
        .fail-badge {{ background: #dc3545; color: white; padding: 2px 8px; border-radius: 4px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 Batch Translation Quality Report</h1>
        <div class="summary">
            <div class="summary-card"><div class="value">{total}</div><div class="label">Total Files</div></div>
            <div class="summary-card"><div class="value" style="color:#28a745">{passed}</div><div class="label">Passed</div></div>
            <div class="summary-card"><div class="value" style="color:#dc3545">{failed}</div><div class="label">Failed</div></div>
            <div class="summary-card"><div class="value">{avg_score:.2f}</div><div class="label">Avg Score</div></div>
            <div class="summary-card"><div class="value">{total_errors}</div><div class="label">Total Errors</div></div>
            <div class="summary-card"><div class="value">{passed_pct:.0f}%</div><div class="label">Pass Rate</div></div>
        </div>
        <h2>Detailed Results</h2>
"""
        for filename, report in results:
            status_class = "pass" if report.status == "pass" else "fail"
            badge_class = "pass-badge" if report.status == "pass" else "fail-badge"
            badge_text = "PASS" if report.status == "pass" else "FAIL"
            html += f"""
        <div class="file-card">
            <div class="file-header {status_class}">
                <span>{filename}</span>
                <span><span class="{badge_class}">{badge_text}</span> Score: {report.mqm_score:.2f}</span>
            </div>
"""
            if report.errors:
                html += """            <div class="file-body">
                <table class="error-table">
                    <tr><th>Severity</th><th>Category</th><th>Description</th></tr>
"""
                for e in report.errors:
                    sev_class = f"severity-{e.severity.value}"
                    desc = e.description.replace("<", "&lt;").replace(">", "&gt;")
                    html += f'                    <tr><td class="{sev_class}">{e.severity.value}</td><td>{e.category}/{e.subcategory}</td><td>{desc}</td></tr>\n'
                html += """                </table>
            </div>
"""
            html += "        </div>\n"
        html += """    </div>
</body>
</html>"""
        output_path.write_text(html, encoding="utf-8")


async def batch_from_file_async(
    file_path: str,
    threshold: float,
    output: str,
    parallel: int,
    batch_size: int | None,
    provider: str | None,
    smart_routing: bool,
    _show_cost_savings: bool,
    show_progress: bool,
    glossary: str | None,
    verbose: bool,
    demo: bool = False,
    output_format: str | None = None,
) -> None:
    """Async implementation of batch command for file-based input."""
    settings = get_settings()

    print_header(
        "Batch Translation Quality Check (File Mode)",
        "Process translations from CSV/JSON/JSONL file",
    )

    # Parse batch file
    file_path_obj = Path(file_path)
    if not file_path_obj.exists():
        raise FileNotFoundError(f"Batch file not found: {file_path}")

    console.print(f"[cyan]Parsing batch file:[/cyan] {file_path}")
    translations = BatchFileParser.parse(file_path_obj)
    console.print(f"[green]✓[/green] Loaded [cyan]{len(translations)}[/cyan] translations\n")

    # Group by language pair (for display)
    groups = BatchGrouper.group_by_language_pair(translations)
    console.print("[bold]Language Pairs:[/bold]")
    for (src, tgt), group_translations in groups.items():
        console.print(f"  • {src} → {tgt}: [cyan]{len(group_translations)}[/cyan] translations")
    console.print()

    _load_and_apply_glossaries(glossary, translations)

    if smart_routing:
        console.print("[cyan]Smart routing enabled[/cyan] - complexity-based model selection\n")

    config_info = _build_batch_config_info(
        file_path, translations, groups, threshold, parallel, batch_size, smart_routing, glossary
    )
    print_startup_info(config_info)

    llm_provider = setup_llm_provider(provider, settings, verbose, demo=demo)
    orchestrator = AgentOrchestrator(
        llm_provider,
        quality_threshold=threshold,
        agent_temperature=settings.default_temperature,
        agent_max_tokens=settings.default_max_tokens,
    )

    console.print("[yellow]⏳ Processing translations...[/yellow]\n")
    results = await _process_batch_translations(
        translations, orchestrator, parallel, verbose, show_progress=show_progress
    )

    console.print()
    _display_batch_summary(results, threshold)
    _save_batch_report(results, output, threshold, output_format)
    console.print(f"\n[dim]Detailed report saved to: {output}[/dim]")

    failed_count = sum(1 for _, report in results if report.status == "fail")
    if failed_count > 0:
        raise typer.Exit(code=1)


async def batch_async(
    source_dir: str,
    translation_dir: str,
    source_lang: str,
    target_lang: str,
    threshold: float,
    output: str,
    parallel: int,
    provider: str | None,
    verbose: bool,
    demo: bool = False,
    output_format: str | None = None,
) -> None:
    """Async implementation of batch command."""
    # Load settings
    settings = get_settings()

    # Display header
    print_header(
        "Batch Translation Quality Check", "Process multiple translation files in parallel"
    )

    # Display configuration
    config_info = {
        "Source Directory": source_dir,
        "Translation Directory": translation_dir,
        "Languages": f"{source_lang} → {target_lang}",
        "Quality Threshold": f"{threshold}",
        "Parallel Workers": f"{parallel}",
    }
    print_startup_info(config_info)

    # Scan directories
    try:
        file_pairs = _scan_batch_directories(source_dir, translation_dir, verbose)
        console.print(f"Found [cyan]{len(file_pairs)}[/cyan] file pairs to process\n")
    except Exception as e:
        raise RuntimeError(f"Failed to scan directories: {e}") from e

    # Setup LLM provider
    try:
        llm_provider = setup_llm_provider(provider, settings, verbose, demo=demo)
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
    results = await _process_batch_files(
        file_pairs, orchestrator, source_lang, target_lang, parallel
    )

    # Generate aggregated report
    console.print()
    _display_batch_summary(results, threshold)

    # Save detailed report
    _save_batch_report(results, output, threshold, output_format)
    console.print(f"\n[dim]Detailed report saved to: {output}[/dim]")

    # Exit with appropriate code
    failed_count = sum(1 for _, report in results if report.status == "fail")
    if failed_count > 0:
        raise typer.Exit(code=1)


@batch_app.command(name="batch")
def batch(
    # File-based mode
    file: str | None = typer.Option(None, "--file", "-f", help="Batch file (CSV, JSON, or JSONL)"),
    # Directory-based mode
    source_dir: str | None = typer.Option(None, "--source-dir", help="Directory with source files"),
    translation_dir: str | None = typer.Option(
        None, "--translation-dir", help="Directory with translation files"
    ),
    source_lang: str | None = typer.Option(
        None, "--source-lang", help="Source language code (for directory mode)"
    ),
    target_lang: str | None = typer.Option(
        None, "--target-lang", help="Target language code (for directory mode)"
    ),
    # Common options
    threshold: float = typer.Option(95.0, "--threshold", help="Minimum MQM score to pass"),
    output: str = typer.Option("report.json", "--output", "-o", help="Output report file path"),
    parallel: int = typer.Option(4, "--parallel", help="Number of parallel workers"),
    batch_size: int | None = typer.Option(
        None, "--batch-size", help="Batch size for grouping (file mode only)"
    ),
    provider: str | None = typer.Option(
        None, "--provider", help="LLM provider (openai or anthropic)"
    ),
    smart_routing: bool = typer.Option(
        False, "--smart-routing", help="Enable complexity-based smart routing to optimize costs"
    ),
    show_cost_savings: bool = typer.Option(
        False, "--show-cost-savings", help="Display estimated cost savings from smart routing"
    ),
    show_progress: bool = typer.Option(
        True, "--show-progress/--no-progress", help="Show progress bar during batch processing"
    ),
    glossary: str | None = typer.Option(
        None,
        "--glossary",
        "-g",
        help="Glossaries to use (comma-separated, e.g., 'base,medical')",
    ),
    verbose: bool = typer.Option(False, "--verbose", help="Verbose output"),
    demo: bool = typer.Option(
        False, "--demo", help="Demo mode (no API calls, simulated responses)"
    ),
    output_format: str | None = typer.Option(
        None,
        "--format",
        help="Output format: json (default), text, markdown, or html.",
    ),
) -> None:
    """
    Batch process multiple translations.

    Supports two modes:

    1. FILE MODE - Process translations from CSV/JSON/JSONL file:
        kttc batch --file translations.csv --output report.json

    2. DIRECTORY MODE - Process translation files from directories:
        kttc batch --source-dir ./source --translation-dir ./translations \\
                   --source-lang en --target-lang es

    Supported file formats:
    - CSV: source,translation,source_lang,target_lang,domain
    - JSON: Array of translation objects
    - JSONL: One JSON object per line

    Examples:
        # Process CSV file
        kttc batch --file examples/batch/translations.csv

        # Process JSON with custom batch size
        kttc batch --file translations.json --batch-size 50

        # Directory mode (original behavior)
        kttc batch --source-dir ./source --translation-dir ./translations \\
                   --source-lang en --target-lang es
    """
    try:
        # Determine mode
        if file:
            # FILE MODE
            asyncio.run(
                batch_from_file_async(
                    file,
                    threshold,
                    output,
                    parallel,
                    batch_size,
                    provider,
                    smart_routing,
                    show_cost_savings,
                    show_progress,
                    glossary,
                    verbose,
                    demo,
                    output_format,
                )
            )
        elif source_dir and translation_dir:
            # DIRECTORY MODE
            if not source_lang or not target_lang:
                console.print(
                    "[red]Error: --source-lang and --target-lang required for directory mode[/red]"
                )
                raise typer.Exit(code=1)

            asyncio.run(
                batch_async(
                    source_dir,
                    translation_dir,
                    source_lang,
                    target_lang,
                    threshold,
                    output,
                    parallel,
                    provider,
                    verbose,
                    demo,
                    output_format,
                )
            )
        else:
            # Error - must specify either file or directories
            console.print(
                "[red]Error: Must specify either:[/red]\n"
                "  1. --file <path> for file-based batch processing, OR\n"
                "  2. --source-dir and --translation-dir for directory-based processing"
            )
            raise typer.Exit(code=1)

    except KeyboardInterrupt:
        console.print("\n[yellow]⚠ Interrupted by user[/yellow]")
        raise typer.Exit(code=130) from None
    except Exception as e:
        console.print(f"\n[red]✗ Error: {e}[/red]")
        if verbose:
            console.print_exception()
        raise typer.Exit(code=1)
