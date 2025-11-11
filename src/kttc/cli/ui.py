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

"""Rich UI components for beautiful CLI output.

This module provides reusable UI components using Rich library
for consistent and beautiful terminal output across all CLI commands.
"""

from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import Text

from kttc.core import QAReport

# Global console instance
console = Console()


def print_header(title: str, subtitle: str | None = None) -> None:
    """Print a beautiful header with title and optional subtitle.

    Args:
        title: Main title text
        subtitle: Optional subtitle text
    """
    console.print()
    if subtitle:
        header_text = f"[bold]{title}[/bold]\n[dim]{subtitle}[/dim]"
    else:
        header_text = f"[bold]{title}[/bold]"

    panel = Panel(
        header_text,
        border_style="cyan",
        padding=(1, 2),
    )
    console.print(panel)
    console.print()


def print_startup_info(info: dict[str, str]) -> None:
    """Print startup information in a formatted panel.

    Args:
        info: Dictionary of key-value pairs to display
    """
    lines = []
    for key, value in info.items():
        lines.append(f"[cyan]{key:20}[/cyan] {value}")

    panel = Panel(
        "\n".join(lines),
        title="KTTC Configuration",
        border_style="blue",
        padding=(1, 2),
    )
    console.print(panel)
    console.print()


def print_qa_report(report: QAReport, verbose: bool = False) -> None:
    """Print QA report with beautiful formatting.

    Args:
        report: QA report to display
        verbose: Whether to show detailed error information
    """
    # Status badge with appropriate styling
    if report.status == "pass":
        status_text = Text("✓ PASS", style="bold green")
    else:
        status_text = Text("✗ FAIL", style="bold red")

    # MQM Score with color coding
    if report.mqm_score >= 95:
        score_color = "green"
    elif report.mqm_score >= 85:
        score_color = "yellow"
    else:
        score_color = "red"

    # Create main results table
    results_table = Table(show_header=False, box=None, padding=(0, 2))
    results_table.add_column(style="bold")
    results_table.add_column()

    results_table.add_row("Status:", status_text)
    results_table.add_row(
        "MQM Score:", Text(f"{report.mqm_score:.2f}/100", style=f"bold {score_color}")
    )
    results_table.add_row("Errors Found:", str(len(report.errors)))

    if report.errors:
        error_breakdown = (
            f"Critical: {report.critical_error_count} | "
            f"Major: {report.major_error_count} | "
            f"Minor: {report.minor_error_count}"
        )
        results_table.add_row("Error Breakdown:", error_breakdown)

    # Display in a panel
    panel = Panel(
        results_table,
        title="📊 Quality Assessment Report",
        border_style="cyan",
        padding=(1, 2),
    )
    console.print(panel)

    # Show detailed errors if verbose
    if verbose and report.errors:
        console.print()
        print_error_details(report.errors)


def print_error_details(errors: list[Any]) -> None:
    """Print detailed error information in a table.

    Args:
        errors: List of error objects to display
    """
    table = Table(title="Error Details", show_header=True, header_style="bold cyan")
    table.add_column("Category", style="cyan", no_wrap=True)
    table.add_column("Subcategory", style="dim")
    table.add_column("Severity", no_wrap=True)
    table.add_column("Location", justify="center", no_wrap=True)
    table.add_column("Description", max_width=50)

    for error in errors:
        # Color-code severity
        if error.severity.value == "critical":
            severity_color = "red"
        elif error.severity.value == "major":
            severity_color = "yellow"
        else:
            severity_color = "dim"

        severity_text = Text(error.severity.value.upper(), style=f"bold {severity_color}")

        # Format location
        location = f"{error.location[0]}-{error.location[1]}"

        # Truncate description if too long
        description = error.description
        if len(description) > 50:
            description = description[:47] + "..."

        table.add_row(
            error.category,
            error.subcategory,
            severity_text,
            location,
            description,
        )

    console.print(table)


def print_comparison_table(comparisons: list[dict[str, Any]]) -> None:
    """Print comparison table for multiple translations.

    Args:
        comparisons: List of comparison dictionaries with name, score, errors, etc.
    """
    table = Table(
        title="Translation Comparison",
        show_header=True,
        header_style="bold cyan",
        title_style="bold",
    )

    table.add_column("Provider", style="cyan", no_wrap=True)
    table.add_column("COMET Score", justify="right")
    table.add_column("MQM Score", justify="right")
    table.add_column("Errors", justify="center")
    table.add_column("Status", justify="center")
    table.add_column("Duration", justify="right")

    for comp in comparisons:
        # Color-code status
        status = comp.get("status", "unknown")
        if status == "pass":
            status_text = Text("✓ PASS", style="green")
        else:
            status_text = Text("✗ FAIL", style="red")

        # Color-code MQM score
        mqm_score = comp.get("mqm_score", 0.0)
        if mqm_score >= 95:
            mqm_color = "green"
        elif mqm_score >= 85:
            mqm_color = "yellow"
        else:
            mqm_color = "red"

        table.add_row(
            comp.get("name", "Unknown"),
            f"{comp.get('comet_score', 0.0):.2f}",
            Text(f"{mqm_score:.2f}", style=mqm_color),
            str(comp.get("error_count", 0)),
            status_text,
            f"{comp.get('duration', 0.0):.2f}s",
        )

    console.print(table)


def print_benchmark_summary(results: dict[str, Any]) -> None:
    """Print benchmark summary with statistics.

    Args:
        results: Dictionary containing benchmark results and statistics
    """
    # Summary statistics
    stats_table = Table(show_header=False, box=None, padding=(0, 2))
    stats_table.add_column(style="bold cyan")
    stats_table.add_column()

    stats_table.add_row("Total Providers:", str(results.get("total_providers", 0)))
    stats_table.add_row("Test Sentences:", str(results.get("test_sentences", 0)))
    stats_table.add_row("Average COMET:", f"{results.get('avg_comet', 0.0):.2f}")
    stats_table.add_row("Average MQM:", f"{results.get('avg_mqm', 0.0):.2f}")
    stats_table.add_row("Best Provider:", results.get("best_provider", "N/A"))

    panel = Panel(
        stats_table,
        title="📊 Benchmark Summary",
        border_style="green",
        padding=(1, 2),
    )
    console.print(panel)


def create_progress() -> Progress:
    """Create a configured Progress instance for long-running tasks.

    Returns:
        Configured Progress instance
    """
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    )


def print_success(message: str) -> None:
    """Print a success message.

    Args:
        message: Success message to display
    """
    console.print(f"[bold green]✓[/bold green] {message}")


def print_error(message: str) -> None:
    """Print an error message.

    Args:
        message: Error message to display
    """
    console.print(f"[bold red]✗[/bold red] {message}")


def print_warning(message: str) -> None:
    """Print a warning message.

    Args:
        message: Warning message to display
    """
    console.print(f"[bold yellow]⚠[/bold yellow] {message}")


def print_info(message: str) -> None:
    """Print an info message.

    Args:
        message: Info message to display
    """
    console.print(f"[cyan]ℹ[/cyan] {message}")


def print_translation_preview(source: str, translation: str, max_length: int = 100) -> None:
    """Print a preview of source and translation texts.

    Args:
        source: Source text
        translation: Translation text
        max_length: Maximum length to display (will truncate if longer)
    """
    # Truncate if needed
    source_preview = source if len(source) <= max_length else source[:max_length] + "..."
    translation_preview = (
        translation if len(translation) <= max_length else translation[:max_length] + "..."
    )

    preview_table = Table(show_header=True, header_style="bold", box=None)
    preview_table.add_column("Source", style="dim")
    preview_table.add_column("Translation")

    preview_table.add_row(source_preview, translation_preview)

    console.print(preview_table)
    console.print()


def print_available_extensions() -> None:
    """Print information about all available KTTC extensions and their installation status.

    Shows a comprehensive table with:
    - Extension name
    - Current status (installed/not installed)
    - Download size
    - Features provided
    - Installation command
    """
    from kttc.utils.dependencies import has_benchmark, has_metrics, has_webui

    console.print()
    console.print("[bold cyan]📦 Available KTTC Extensions[/bold cyan]")
    console.print()

    # Create extensions table
    table = Table(show_header=True, header_style="bold cyan", box=None, padding=(0, 2))
    table.add_column("Extension", style="bold")
    table.add_column("Status", justify="center")
    table.add_column("Size", justify="right")
    table.add_column("Features")

    # Check each extension
    metrics_installed = has_metrics()
    benchmark_installed = has_benchmark()
    webui_installed = has_webui()

    # Metrics extension
    metrics_status = (
        "[green]✓ Installed[/green]" if metrics_installed else "[yellow]○ Not installed[/yellow]"
    )
    table.add_row(
        "metrics",
        metrics_status,
        "~2.5 GB",
        "COMET, XCOMET, neural quality scoring",
    )

    # Benchmark extension
    benchmark_status = (
        "[green]✓ Installed[/green]" if benchmark_installed else "[yellow]○ Not installed[/yellow]"
    )
    table.add_row(
        "benchmark",
        benchmark_status,
        "~2.5 GB",
        "Provider comparison, WMT datasets, benchmarking",
    )

    # WebUI extension
    webui_status = (
        "[green]✓ Installed[/green]" if webui_installed else "[yellow]○ Not installed[/yellow]"
    )
    table.add_row(
        "webui",
        webui_status,
        "~50 MB",
        "Web interface, API server, real-time UI",
    )

    console.print(table)
    console.print()

    # Show installation commands for missing extensions
    missing = []
    if not metrics_installed:
        missing.append("metrics")
    if not benchmark_installed:
        missing.append("benchmark")
    if not webui_installed:
        missing.append("webui")

    if missing:
        console.print("[bold]Installation Commands:[/bold]")
        console.print()

        for ext in missing:
            console.print(f"  [dim]# Install {ext}[/dim]")
            console.print(f"  pip install 'kttc[{ext}]'", style="cyan", markup=False)
            console.print()

        # Show combined install command if multiple missing
        if len(missing) > 1:
            combined = ",".join(missing)
            console.print("  [dim]# Install all missing extensions at once[/dim]")
            console.print(f"  pip install 'kttc[{combined}]'", style="cyan", markup=False)
            console.print()

        console.print(
            "[dim]Note: First install may take 5-10 minutes due to large model downloads.[/dim]"
        )
        console.print()
    else:
        console.print("[green]✓ All extensions are installed![/green]")
        console.print()
