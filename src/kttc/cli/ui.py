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

from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import Text

from kttc.core import QAReport

# Import and re-export console utilities for backward compatibility
from kttc.utils.console import (
    console,
    print_error,
    print_info,
    print_success,
    print_warning,
)

__all__ = [
    "console",
    "print_error",
    "print_info",
    "print_success",
    "print_warning",
    "print_header",
    "print_startup_info",
    "print_translation_preview",
    "print_qa_report",
    "print_lightweight_metrics",
    "print_rule_based_errors",
]


def print_header(title: str, subtitle: str | None = None) -> None:
    """Print a minimal header with title and optional subtitle.

    Args:
        title: Main title text
        subtitle: Optional subtitle text

    Best practice: Keep headers minimal and scannable (clig.dev).
    Use bold for structure, avoid large panels that waste vertical space.
    """
    console.print()
    console.print(f"[bold cyan]{title}[/bold cyan]")
    if subtitle:
        console.print(f"[dim]{subtitle}[/dim]")
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
        border_style="cyan",
        padding=(1, 2),
    )
    console.print(panel)
    console.print()


def _analyze_adj_noun_pairs(adj_noun_pairs: list[dict[str, Any]], insights: dict[str, Any]) -> None:
    """Analyze adjective-noun pairs and update insights."""
    if not adj_noun_pairs:
        return

    correct_count = sum(1 for p in adj_noun_pairs if p.get("agreement") == "correct")
    incorrect_count = len(adj_noun_pairs) - correct_count

    if incorrect_count == 0:
        insights["good_indicators"].append(f"Case agreement: {len(adj_noun_pairs)} pairs verified")
        return

    for pair in adj_noun_pairs:
        if pair.get("agreement") == "correct":
            continue
        pair_text = pair.get("text", "")
        adj = pair.get("adjective", {}).get("text", "")
        noun = pair.get("noun", {}).get("text", "")

        if not adj and not noun:
            continue

        description = f"Case mismatch: '{adj}' and '{noun}'"
        if pair_text and pair_text != "unknown":
            description = f"Case agreement issue in '{pair_text}'"

        insights["issues"].append(
            {
                "category": "Linguistic",
                "subcategory": "Case Agreement",
                "severity": "minor",
                "description": description,
                "location": pair.get("location", [0, 0]),
            }
        )


def _analyze_entities(helper: Any, translation: str, insights: dict[str, Any]) -> None:
    """Extract and analyze named entities."""
    if not hasattr(helper, "extract_entities"):
        return

    try:
        entities = helper.extract_entities(translation)
    except Exception:
        return

    if not entities:
        return

    entity_types: dict[str, int] = {}
    for e in entities:
        etype = e.get("type", "UNKNOWN")
        entity_types[etype] = entity_types.get(etype, 0) + 1

    entity_summary = ", ".join(f"{count} {type_}" for type_, count in entity_types.items())
    insights["good_indicators"].append(f"Named entities: {len(entities)} found ({entity_summary})")


def get_nlp_insights(task: Any, helper: Any) -> dict[str, Any] | None:
    """Collect NLP analysis insights for languages with NLP support.

    Args:
        task: Translation task
        helper: Language helper with NLP capabilities

    Returns:
        Dictionary with NLP insights data, or None if not available
    """
    if not helper or not helper.is_available():
        return None

    try:
        enrichment = helper.get_enrichment_data(task.translation)
        if not enrichment.get("has_morphology"):
            return None

        insights: dict[str, Any] = {
            "word_count": enrichment.get("word_count", 0),
            "issues": [],
            "good_indicators": [],
        }

        verb_aspects = enrichment.get("verb_aspects", {})
        if verb_aspects:
            insights["good_indicators"].append(f"Verb aspects: {len(verb_aspects)} verbs analyzed")

        _analyze_adj_noun_pairs(enrichment.get("adjective_noun_pairs", []), insights)
        _analyze_entities(helper, task.translation, insights)

        return insights

    except Exception:
        return None


def print_nlp_insights(task: Any, helper: Any) -> None:
    """Display NLP analysis insights for a translation task.

    Args:
        task: Translation task with translation text
        helper: Language helper with NLP capabilities

    This function displays linguistic analysis such as:
    - Word count
    - Verb aspects (for languages like Russian)
    - Case agreement (for languages with case systems)
    - Named entities
    """
    insights = get_nlp_insights(task, helper)
    if not insights:
        return

    # Create insights table
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="bold cyan", width=20)
    table.add_column()

    table.add_row("Word Count:", str(insights.get("word_count", 0)))

    # Show good indicators
    for indicator in insights.get("good_indicators", []):
        # Extract metric name and value from indicator string
        if ":" in indicator:
            metric, value = indicator.split(":", 1)
            table.add_row(metric + ":", value.strip())

    # Show issues if any
    if insights.get("issues"):
        console.print()
        console.print("[bold yellow]Linguistic Issues Found:[/bold yellow]")
        for issue in insights["issues"]:
            severity = issue.get("severity", "minor")
            severity_color = (
                "red" if severity == "critical" else "yellow" if severity == "major" else "dim"
            )
            console.print(
                f"  [{severity_color}]• {issue.get('description', 'Unknown issue')}[/{severity_color}]"
            )

    console.print()
    panel = Panel(
        table,
        title="NLP Insights",
        border_style="cyan",
        padding=(1, 2),
    )
    console.print(panel)


def _get_score_color(score: float) -> str:
    """Get color for score display."""
    if score >= 95:
        return "green"
    if score >= 85:
        return "yellow"
    return "red"


def _get_confidence_color(confidence: float) -> str:
    """Get color for confidence display."""
    if confidence >= 0.8:
        return "green"
    if confidence >= 0.6:
        return "yellow"
    return "red"


def _get_confidence_label(confidence: float) -> str:
    """Get label for confidence level."""
    if confidence >= 0.8:
        return "high"
    if confidence >= 0.6:
        return "medium"
    return "low"


def _add_confidence_to_table(results_table: Table, report: QAReport) -> None:
    """Add confidence metrics to results table."""
    if report.confidence is None:
        return

    conf_color = _get_confidence_color(report.confidence)
    conf_label = _get_confidence_label(report.confidence)

    results_table.add_row(
        "Confidence:",
        Text(f"{report.confidence:.2f}", style=f"bold {conf_color}")
        + " "
        + Text(f"({conf_label})", style="dim"),
    )

    if report.agent_agreement is not None:
        agreement_pct = report.agent_agreement * 100
        results_table.add_row("Agent Agreement:", Text(f"{agreement_pct:.0f}%", style="dim"))


def _add_domain_to_table(results_table: Table, report: QAReport, verbose: bool) -> None:
    """Add domain detection info to results table."""
    if not report.agent_details or "detected_domain" not in report.agent_details:
        return

    domain = report.agent_details["detected_domain"]
    domain_confidence = report.agent_details.get("domain_confidence", 0.0)
    domain_display = domain.replace("_", " ").title()

    if domain_confidence >= 0.75:
        domain_conf_color = "green"
    elif domain_confidence >= 0.5:
        domain_conf_color = "yellow"
    else:
        domain_conf_color = "dim"

    domain_text = Text(domain_display, style="bold cyan")
    if domain_confidence > 0.5:
        domain_text += Text(f" ({domain_confidence:.0%} confidence)", style=domain_conf_color)

    results_table.add_row("Domain:", domain_text)

    if verbose and "quality_threshold_used" in report.agent_details:
        threshold = report.agent_details["quality_threshold_used"]
        results_table.add_row("Domain Threshold:", Text(f"{threshold:.1f}", style="dim"))


def _build_error_breakdown(report: QAReport, nlp_issue_count: int, api_error_count: int) -> str:
    """Build error breakdown string."""
    parts = []
    if report.critical_error_count > 0:
        parts.append(f"Critical: {report.critical_error_count}")
    if report.major_error_count > 0:
        parts.append(f"Major: {report.major_error_count}")
    if report.minor_error_count > 0:
        parts.append(f"Minor: {report.minor_error_count}")
    if nlp_issue_count > 0:
        parts.append(f"Linguistic: {nlp_issue_count}")
    if api_error_count > 0:
        parts.append(f"System: {api_error_count}")
    return " | ".join(parts)


def _print_agent_scores(report: QAReport) -> None:
    """Print per-agent scores table."""
    if not report.agent_scores:
        return

    console.print()
    console.print("[bold]Per-Agent Scores:[/bold]")
    agent_score_table = Table(show_header=True, box=None, padding=(0, 2))
    agent_score_table.add_column("Agent", style="cyan")
    agent_score_table.add_column("MQM Score", justify="right")

    for agent_name, score in sorted(report.agent_scores.items()):
        score_style = _get_score_color(score)
        agent_score_table.add_row(
            agent_name.replace("_", " ").title(), Text(f"{score:.2f}", style=score_style)
        )

    console.print(agent_score_table)


def _collect_all_issues(
    report: QAReport,
    nlp_insights: dict[str, Any] | None,
    api_errors: list[str] | None,
) -> list[dict[str, Any]]:
    """Collect all issues from various sources into unified list."""
    all_issues: list[dict[str, Any]] = []

    # API errors (system errors)
    if api_errors:
        for api_error in api_errors:
            all_issues.append(
                {
                    "category": "System Error",
                    "subcategory": "API",
                    "severity": "critical",
                    "location": [0, 0],
                    "description": api_error,
                }
            )

    # QA report errors
    for error in report.errors:
        all_issues.append(
            {
                "category": error.category,
                "subcategory": error.subcategory,
                "severity": error.severity.value,
                "location": error.location,
                "description": error.description,
            }
        )

    # NLP issues
    if nlp_insights and nlp_insights.get("issues"):
        all_issues.extend(nlp_insights["issues"])

    return all_issues


def print_qa_report(
    report: QAReport,
    nlp_insights: dict[str, Any] | None = None,
    verbose: bool = False,
    api_errors: list[str] | None = None,
) -> None:
    """Print QA report with NLP insights and errors in unified format."""
    # Calculate issue counts
    nlp_issue_count = len(nlp_insights["issues"]) if nlp_insights else 0
    api_error_count = len(api_errors) if api_errors else 0
    total_issues = len(report.errors) + nlp_issue_count + api_error_count

    # Build results table
    results_table = _build_results_table(
        report, total_issues, nlp_issue_count, api_error_count, verbose
    )

    # Show agent scores in verbose mode
    if verbose:
        _print_agent_scores(report)

    # Show NLP good indicators
    if nlp_insights and nlp_insights.get("good_indicators"):
        console.print()
        console.print("[dim]Linguistic checks passed:[/dim]")
        for indicator in nlp_insights["good_indicators"]:
            console.print(f"  [dim]✓ {indicator}[/dim]")

    # Display results panel
    console.print()
    panel = Panel(
        results_table,
        title="Quality Assessment Report",
        border_style="cyan",
        padding=(1, 2),
    )
    console.print(panel)

    # Low confidence warning
    if report.confidence is not None and report.confidence < 0.7:
        console.print()
        console.print(
            "[bold yellow]⚠ Warning:[/bold yellow] Low confidence detected. "
            "Agents disagree on quality assessment. Human review recommended."
        )

    # Show errors
    all_issues = _collect_all_issues(report, nlp_insights, api_errors)
    if all_issues:
        console.print()
        _print_unified_error_table(all_issues, verbose)


def _build_results_table(
    report: QAReport,
    total_issues: int,
    nlp_issue_count: int,
    api_error_count: int,
    verbose: bool,
) -> Table:
    """Build the main results table."""
    # Status badge
    if report.status == "pass":
        status_text = Text("✓ PASS", style="bold green")
    else:
        status_text = Text("✗ FAIL", style="bold red")

    score_color = _get_score_color(report.mqm_score)

    # Create table
    results_table = Table(show_header=False, box=None, padding=(0, 2))
    results_table.add_column(style="bold", width=18)
    results_table.add_column()

    results_table.add_row("Status:", status_text)
    results_table.add_row(
        "MQM Score:", Text(f"{report.mqm_score:.2f}/100", style=f"bold {score_color}")
    )
    results_table.add_row("Total Issues:", str(total_issues))

    # Add optional sections
    _add_confidence_to_table(results_table, report)
    _add_domain_to_table(results_table, report, verbose)

    # Error breakdown
    if total_issues > 0:
        breakdown = _build_error_breakdown(report, nlp_issue_count, api_error_count)
        results_table.add_row("Issue Breakdown:", breakdown)

    return results_table


def _format_error_severity(severity: str) -> Text:
    """Format severity with color coding."""
    severity_colors = {"critical": "red", "major": "yellow"}
    color = severity_colors.get(severity, "dim")
    return Text(severity.upper(), style=f"bold {color}")


def _format_error_location(location: Any) -> str:
    """Format location tuple/list as string."""
    if isinstance(location, list | tuple) and len(location) >= 2:
        return f"{location[0]}-{location[1]}"
    return "N/A"


def _format_error_suggestion(suggestion: str, confidence: float | None, verbose: bool) -> str:
    """Format suggestion text with optional confidence."""
    if not suggestion:
        return "[dim]-[/dim]"
    if not verbose and len(suggestion) > 50:
        suggestion = suggestion[:47] + "..."
    if confidence is not None:
        suggestion = f"{suggestion} ({confidence:.0%})"
    return suggestion


def _print_unified_error_table(
    issues: list[dict[str, Any]], verbose: bool, show_suggestions: bool = False
) -> None:
    """Print unified error table with all issues (QA errors, NLP, API)."""
    table = Table(title="Issues Found", show_header=True, header_style="bold cyan")
    table.add_column("Category", style="cyan", no_wrap=True)
    table.add_column("Subcategory", style="dim", no_wrap=True)
    table.add_column("Severity", no_wrap=True)
    table.add_column("Location", justify="center", no_wrap=True, width=10)
    table.add_column("Description", max_width=60 if not verbose else None)
    if show_suggestions:
        table.add_column("Suggestion", max_width=50 if not verbose else None, style="green")

    for issue in issues:
        severity_text = _format_error_severity(issue.get("severity", "minor"))
        location_str = _format_error_location(issue.get("location", [0, 0]))

        description = issue.get("description", "")
        if not verbose and len(description) > 60:
            description = description[:57] + "..."

        row_data = [
            issue.get("category", "Unknown"),
            issue.get("subcategory", ""),
            severity_text,
            location_str,
            description,
        ]

        if show_suggestions:
            suggestion_text = _format_error_suggestion(
                issue.get("suggestion", ""), issue.get("confidence"), verbose
            )
            row_data.append(suggestion_text)

        table.add_row(*row_data)

    console.print(table)


def print_error_details(errors: list[Any]) -> None:
    """Print detailed error information in a table.

    DEPRECATED: Use _print_unified_error_table instead.

    Args:
        errors: List of error objects to display
    """
    issues = []
    for error in errors:
        issues.append(
            {
                "category": error.category,
                "subcategory": error.subcategory,
                "severity": error.severity.value,
                "location": error.location,
                "description": error.description,
            }
        )
    _print_unified_error_table(issues, verbose=True)


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

        # Format error breakdown (C/M/m = Critical/Major/minor)
        error_str = f"{comp.get('critical_errors', 0)}/{comp.get('major_errors', 0)}/{comp.get('minor_errors', 0)}"

        table.add_row(
            comp.get("name", "Unknown"),
            Text(f"{mqm_score:.2f}", style=mqm_color),
            error_str,
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
    stats_table.add_row("Average MQM:", f"{results.get('avg_mqm', 0.0):.2f}")
    stats_table.add_row("Average Duration:", f"{results.get('avg_duration', 0.0):.2f}s")
    stats_table.add_row("Best Provider:", results.get("best_provider", "N/A"))
    stats_table.add_row("Fastest Provider:", results.get("fastest_provider", "N/A"))
    stats_table.add_row("Pass Rate:", results.get("pass_rate", "0/0"))

    panel = Panel(
        stats_table,
        title="Benchmark Summary",
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


def create_step_progress() -> Progress:
    """Create a minimal progress spinner for multi-step operations.

    Returns:
        Configured Progress instance with spinner and text only

    Best practice: For sequential tasks, show spinner with step description
    that updates as each step completes (Evil Martians CLI UX guide).
    """
    return Progress(
        SpinnerColumn(spinner_name="dots"),
        TextColumn("{task.description}"),
        console=console,
        transient=True,  # Remove spinner when complete
    )


# Note: print_success, print_error, print_warning, print_info are now imported from kttc.utils.console


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


def check_models_with_loader() -> bool:
    """Check if neural models are downloaded, show loader and error if needed.

    Neural models are no longer required. This function always returns True
    for backward compatibility.

    Returns:
        Always True
    """
    return True


def print_available_extensions() -> None:
    """Print information about available extensions.

    Shows which optional dependencies are installed.
    """
    from kttc.utils.dependencies import has_benchmark, has_webui

    console.print("[bold]Available Extensions:[/bold]")
    console.print(f"  • Benchmark: {'✓' if has_benchmark() else '✗'}")
    console.print(f"  • WebUI: {'✓' if has_webui() else '✗'}")
    console.print()


def print_lightweight_metrics(
    scores: Any,
    verbose: bool = False,
) -> None:
    """Print lightweight translation metrics (chrF, BLEU, TER).

    Args:
        scores: MetricScores object with chrF, BLEU, TER scores
        verbose: Whether to show detailed explanations

    Example output:
        ┌─ Lightweight Metrics (CPU-based) ─┐
        │ chrF:           68.50  ✓ Good     │
        │ BLEU:           42.30  ✓ Good     │
        │ TER (inverted): 71.90  ✓ Good     │
        │ Length Ratio:    0.95  ✓          │
        │ Composite:      65.90  ✓ Good     │
        └────────────────────────────────────┘
    """
    # Color-code scores based on thresholds
    def get_score_color(score: float, metric_type: str = "default") -> str:
        """Get color for score based on thresholds."""
        if metric_type == "chrf":
            if score >= 80:
                return "green"
            if score >= 65:
                return "yellow"
            if score >= 50:
                return "dim yellow"
            return "red"
        elif metric_type == "bleu":
            if score >= 50:
                return "green"
            if score >= 40:
                return "yellow"
            if score >= 30:
                return "dim yellow"
            return "red"
        else:  # default for TER and composite
            if score >= 70:
                return "green"
            if score >= 60:
                return "yellow"
            if score >= 50:
                return "dim yellow"
            return "red"

    # Create metrics table
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="bold cyan", width=18)
    table.add_column(justify="right", width=8)
    table.add_column(width=15)

    # Add scores
    chrf_color = get_score_color(scores.chrf, "chrf")
    table.add_row(
        "chrF:",
        Text(f"{scores.chrf:.2f}", style=chrf_color),
        Text(f"✓ {scores.quality_level.title()}", style=chrf_color),
    )

    bleu_color = get_score_color(scores.bleu, "bleu")
    bleu_status = (
        "✓ Good" if scores.bleu >= 40 else "⚠ Acceptable" if scores.bleu >= 25 else "✗ Poor"
    )
    table.add_row(
        "BLEU:",
        Text(f"{scores.bleu:.2f}", style=bleu_color),
        Text(bleu_status, style=bleu_color),
    )

    ter_color = get_score_color(scores.ter)
    ter_status = "✓ Good" if scores.ter >= 70 else "⚠ Acceptable" if scores.ter >= 50 else "✗ Poor"
    table.add_row(
        "TER (inverted):",
        Text(f"{scores.ter:.2f}", style=ter_color),
        Text(ter_status, style=ter_color),
    )

    # Length ratio
    length_status = "✓" if 0.8 <= scores.length_ratio <= 1.2 else "⚠"
    length_color = "green" if 0.8 <= scores.length_ratio <= 1.2 else "yellow"
    table.add_row(
        "Length Ratio:",
        Text(f"{scores.length_ratio:.2f}", style=length_color),
        Text(length_status, style=length_color),
    )

    # Composite score
    composite = scores.composite_score
    composite_color = get_score_color(composite)
    composite_status = (
        "✓ Good" if composite >= 65 else "⚠ Acceptable" if composite >= 50 else "✗ Poor"
    )
    table.add_row(
        "Composite:",
        Text(f"{composite:.2f}", style=composite_color),
        Text(composite_status, style=composite_color),
    )

    panel = Panel(
        table,
        title="Lightweight Metrics (CPU-based)",
        border_style="cyan",
        padding=(1, 2),
    )
    console.print(panel)

    # Show interpretation if verbose
    if verbose:
        from kttc.evaluation import LightweightMetrics

        metrics = LightweightMetrics()
        interpretation = metrics.get_interpretation(scores)
        console.print(f"[dim]{interpretation}[/dim]")

    console.print()


def _get_score_color_and_icon(score: float) -> tuple[str, str]:
    """Get color and icon for score display."""
    if score >= 80:
        return "green", "✓"
    if score >= 60:
        return "yellow", "⚠"
    return "red", "✗"


def _get_severity_color(severity: str) -> str:
    """Get color for severity level."""
    if severity == "critical":
        return "red"
    if severity == "major":
        return "yellow"
    return "dim"


def _build_rule_based_summary_table(
    rule_based_score: float, errors: list[Any], severity_counts: dict[str, int]
) -> Table:
    """Build summary table for rule-based checks."""
    score_color, score_icon = _get_score_color_and_icon(rule_based_score)

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="bold cyan", width=18)
    table.add_column()

    table.add_row(
        "Score:",
        Text(f"{rule_based_score:.0f}/100", style=f"bold {score_color}")
        + " "
        + Text(score_icon, style=score_color),
    )
    table.add_row("Total Errors:", str(len(errors)))

    if errors:
        for sev, color in [("critical", "red"), ("major", "yellow"), ("minor", "dim")]:
            if severity_counts[sev] > 0:
                table.add_row(f"  {sev.title()}:", Text(str(severity_counts[sev]), style=color))

    return table


def _build_rule_based_error_table(errors: list[Any], verbose: bool) -> Table:
    """Build detailed error table for rule-based checks."""
    error_table = Table(title="Rule-Based Errors", show_header=True, header_style="bold cyan")
    error_table.add_column("Check Type", style="cyan", no_wrap=True)
    error_table.add_column("Severity", no_wrap=True)
    error_table.add_column("Description", max_width=60 if not verbose else None)

    for error in errors:
        severity_color = _get_severity_color(error.severity)
        severity_text = Text(error.severity.upper(), style=f"bold {severity_color}")

        description = error.description
        if not verbose and len(description) > 60:
            description = description[:57] + "..."

        error_table.add_row(
            error.check_type.replace("_", " ").title(),
            severity_text,
            description,
        )

    return error_table


def print_rule_based_errors(
    errors: list[Any],
    rule_based_score: float,
    verbose: bool = False,
) -> None:
    """Print rule-based error detection results.

    Args:
        errors: List of RuleBasedError objects
        rule_based_score: Overall rule-based quality score (0-100)
        verbose: Whether to show detailed explanations

    Example output:
        ┌─ Rule-Based Checks ─┐
        │ Score:   85/100  ✓  │
        │ Errors:  2          │
        │   Critical: 0       │
        │   Major:    1       │
        │   Minor:    1       │
        └─────────────────────┘
    """
    severity_counts = {"critical": 0, "major": 0, "minor": 0}
    for error in errors:
        severity_counts[error.severity] += 1

    table = _build_rule_based_summary_table(rule_based_score, errors, severity_counts)
    panel = Panel(table, title="Rule-Based Checks (No AI)", border_style="cyan", padding=(1, 2))
    console.print(panel)

    if errors and verbose:
        console.print()
        console.print(_build_rule_based_error_table(errors, verbose))

    console.print()
