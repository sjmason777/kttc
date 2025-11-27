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

"""Unified console formatter for all CLI commands.

Provides compact and verbose output modes with consistent styling
across all commands (check, translate, batch, compare, benchmark).
"""

from __future__ import annotations

import math
from typing import Any

from rich.table import Table
from rich.text import Text

from kttc.core import QAReport
from kttc.i18n import _
from kttc.utils.console import console

# Style constants
HEADER_STYLE_CYAN = "bold cyan"


class ConsoleFormatter:
    """Unified console output formatter for all CLI commands.

    Provides two modes:
    - Compact (default): ~10-15 lines, essential info only
    - Verbose: ~25-30 lines, detailed info
    """

    # Minimum widths for columns (characters)
    MIN_WIDTH = 80
    MAX_WIDTH = 200

    @classmethod
    def _get_terminal_width(cls) -> int:
        """Get terminal width, bounded by min/max limits."""
        width = console.width or 80
        return max(cls.MIN_WIDTH, min(width, cls.MAX_WIDTH))

    @classmethod
    def _calculate_column_widths(cls, columns: list[str]) -> dict[str, int | None]:
        """Calculate column widths based on terminal size.

        Args:
            columns: List of column names to calculate widths for

        Returns:
            Dictionary mapping column names to widths (None = auto)
        """
        terminal_width = cls._get_terminal_width()

        # Define column width ratios for issues table
        if "fragment" in [c.lower() for c in columns]:
            # Issues table: #(5%), Location(15%), Fragment(25%), Issue(auto)
            available = terminal_width - 10  # Padding
            return {
                "#": 4,
                "location": 14,  # Fixed: enough for [1234:5678]
                "fragment": max(20, int(available * 0.25)),
                "issue": None,  # Auto-expand to fill remaining space
            }
        if "provider" in [c.lower() for c in columns]:
            # Benchmark table
            return {
                "provider": None,  # Auto
                "mqm": 8,
                "errors": 12,
                "time": 8,
                "status": 8,
            }
        # Default: auto widths
        return {c.lower(): None for c in columns}

    @staticmethod
    def _get_status_color(status: str) -> str:
        """Get color for status."""
        return "green" if status == "pass" else "red"

    @staticmethod
    def _get_score_color(score: float) -> str:
        """Get color for MQM score."""
        if score >= 95:
            return "green"
        if score >= 85:
            return "yellow"
        return "red"

    @staticmethod
    def _format_errors(critical: int, major: int, minor: int) -> str:
        """Format error counts (C:1 M:2 m:3)."""
        return f"C:{critical} M:{major} m:{minor}"

    @staticmethod
    def print_header_compact(command: str, context: str | None = None) -> None:
        """Print compact header (1 line).

        Args:
            command: Command name
            context: Optional context (e.g., "en → ru")
        """
        if context:
            console.print(f"\n[bold cyan]{command}:[/bold cyan] {context}")
        else:
            console.print(f"\n[bold cyan]{command}[/bold cyan]")

    @staticmethod
    def _extract_fragment(text: str, location: list[int], max_length: int = 25) -> str:
        """Extract text fragment at location with ellipsis if needed.

        Args:
            text: Full text
            location: [start, end] character positions
            max_length: Maximum fragment length

        Returns:
            Fragment string with ellipsis if truncated
        """
        if not location or len(location) < 2:
            return "N/A"

        start, end = location[0], location[1]

        # Ensure valid bounds
        if start < 0 or end > len(text) or start >= end:
            return "N/A"

        # Add context (5 chars before and after)
        context_start = max(0, start - 5)
        context_end = min(len(text), end + 5)

        # Build fragment with context
        prefix = "..." if context_start > 0 else ""
        suffix = "..." if context_end < len(text) else ""
        full_fragment = f"{prefix}{text[context_start:context_end]}{suffix}"

        # Truncate if too long
        if len(full_fragment) > max_length:
            # Show beginning of fragment
            return full_fragment[: max_length - 3] + "..."

        return full_fragment

    @classmethod
    def _build_status_line(cls, report: QAReport) -> str:
        """Build main status line for check result."""
        status_color = cls._get_status_color(report.status)
        score_color = cls._get_score_color(report.mqm_score)
        status_icon = "✓" if report.status == "pass" else "✗"
        status_text = _("check_pass") if report.status == "pass" else _("check_fail")
        line = (
            f"[{status_color}]{status_icon} {status_text}[/{status_color}]  |  "
            f"MQM: [{score_color}]{report.mqm_score:.1f}[/{score_color}]/100  |  "
            f"{_('check_errors', count=len(report.errors))}"
        )
        if report.errors:
            line += f" ({cls._format_errors(report.critical_error_count, report.major_error_count, report.minor_error_count)})"
        return line

    @classmethod
    def _build_style_parts(cls, style_profile: Any) -> list[str]:
        """Build style analysis parts for display."""
        parts = []
        if style_profile.is_literary:
            parts.append("[magenta]Literary text[/magenta]")
        if style_profile.detected_pattern and style_profile.detected_pattern.value != "standard":
            parts.append(
                f"Pattern: {style_profile.detected_pattern.value.replace('_', ' ').title()}"
            )
        if style_profile.deviation_score > 0.3:
            parts.append(f"Deviation: {style_profile.deviation_score:.0%}")
        if style_profile.detected_deviations:
            dev_types = [d.type.value for d in style_profile.detected_deviations[:3]]
            parts.append(f"Features: {', '.join(dev_types)}")
        return parts

    @classmethod
    def print_check_result(
        cls,
        report: QAReport,
        source_lang: str,
        target_lang: str,
        lightweight_scores: Any | None = None,
        rule_based_score: float | None = None,
        rule_based_errors: list[Any] | None = None,
        nlp_insights: dict[str, Any] | None = None,
        style_profile: Any | None = None,
        verbose: bool = False,
    ) -> None:
        """Print check command result in compact or verbose mode.

        Args:
            report: QA report
            source_lang: Source language code
            target_lang: Target language code
            lightweight_scores: Optional lightweight metrics scores
            rule_based_score: Optional rule-based score
            rule_based_errors: Optional rule-based errors
            nlp_insights: Optional NLP insights
            style_profile: Optional StyleProfile from style analysis
            verbose: Verbose mode flag
        """
        # Header and status
        cls.print_header_compact(_("check_header"), f"{source_lang} → {target_lang}")
        console.print()
        console.print(f"● {cls._build_status_line(report)}")

        # Metrics line
        if lightweight_scores and rule_based_score is not None:
            metrics = f"chrF: {lightweight_scores.chrf:.1f} | BLEU: {lightweight_scores.bleu:.1f} | TER: {lightweight_scores.ter:.1f} | Rule-based: {rule_based_score:.0f}/100"
            console.print(f"● {_('check_metrics')}: {metrics}")

        # Style analysis line
        if style_profile:
            style_parts = cls._build_style_parts(style_profile)
            if style_parts:
                console.print(f"● Style: {' | '.join(style_parts)}")

        console.print()

        # Show errors/warnings (compact table)
        if len(report.errors) > 0 or (nlp_insights and nlp_insights.get("issues")):
            cls._print_issues_compact(report.errors, report.task.translation, nlp_insights, verbose)

        # Verbose mode: additional details
        if verbose:
            cls._print_verbose_details(
                report, lightweight_scores, rule_based_errors, nlp_insights, style_profile
            )

    @classmethod
    def print_compare_result(
        cls,
        source_lang: str,
        target_lang: str,
        results: list[dict[str, Any]],
        verbose: bool = False,
    ) -> None:
        """Print compare command result.

        Args:
            source_lang: Source language code
            target_lang: Target language code
            results: List of translation results
            verbose: Verbose mode flag
        """
        # Header
        cls.print_header_compact(_("compare_header"), f"{source_lang} → {target_lang}")
        console.print()

        # Handle empty results
        if not results:
            console.print("[yellow]No translation results to compare[/yellow]")
            return

        # Summary line
        best = max(results, key=lambda r: r["mqm_score"])
        avg_score = sum(r["mqm_score"] for r in results) / len(results)
        best_score = f"{best['mqm_score']:.1f}"

        console.print(
            f"● {_('compare_compared', count=len(results))}  |  "
            f"{_('compare_avg_mqm', score=f'{avg_score:.1f}')}  |  "
            f"{_('compare_best', name=best['name'], score=best_score)}"
        )
        console.print()

        # Results table (compact) - dynamic width
        table = Table(
            show_header=True,
            header_style=HEADER_STYLE_CYAN,
            box=None,
            padding=(0, 1),
            expand=True,
        )
        table.add_column(_("table_translation"), style="cyan", no_wrap=False)
        table.add_column(_("table_mqm"), justify="right", width=8)
        table.add_column(_("table_errors"), justify="center", width=12)
        table.add_column(_("table_status"), justify="center", width=8)

        for result in results:
            score_color = cls._get_score_color(result["mqm_score"])
            status_color = cls._get_status_color(result["status"])
            status_icon = "✓" if result["status"] == "pass" else "✗"

            table.add_row(
                result["name"],
                Text(f"{result['mqm_score']:.1f}", style=score_color),
                cls._format_errors(
                    result["critical_errors"], result["major_errors"], result["minor_errors"]
                ),
                Text(status_icon, style=status_color),
            )

        console.print(table)
        console.print()

        # Verbose: show errors for each translation
        if verbose:
            for result in results:
                if result.get("errors"):
                    console.print(f"\n[bold cyan]{result['name']}:[/bold cyan]")
                    cls._print_issues_compact(result["errors"], "", nlp_insights=None, verbose=True)

    @classmethod
    def print_benchmark_result(  # pylint: disable=unused-argument
        cls,
        source_lang: str,
        target_lang: str,
        results: list[dict[str, Any]],
        verbose: bool = False,
    ) -> None:
        """Print benchmark command result.

        Args:
            source_lang: Source language code
            target_lang: Target language code
            results: List of provider benchmark results
            verbose: Verbose mode flag
        """
        # Header
        cls.print_header_compact(_("benchmark_header"), f"{source_lang} → {target_lang}")
        console.print()

        # Summary line
        successful = [r for r in results if r["success"]]
        if successful:
            best = max(successful, key=lambda r: r["mqm_score"])
            avg_mqm = sum(r["mqm_score"] for r in successful) / len(successful)
            avg_time = sum(r["duration"] for r in successful) / len(successful)

            console.print(
                f"● {_('benchmark_tested', count=len(results))}  |  "
                f"{_('compare_avg_mqm', score=f'{avg_mqm:.1f}')}  |  "
                f"{_('benchmark_avg_time', time=f'{avg_time:.1f}')}  |  "
                f"{_('compare_best', name=best['name'], score='')}"
            )
            console.print()

            # Results table (compact) - dynamic width
            table = Table(
                show_header=True,
                header_style=HEADER_STYLE_CYAN,
                box=None,
                padding=(0, 1),
                expand=True,
            )
            table.add_column(_("table_provider"), style="cyan", no_wrap=False)
            table.add_column(_("table_mqm"), justify="right", width=8)
            table.add_column(_("table_errors"), justify="center", width=12)
            table.add_column(_("table_time"), justify="right", width=8)
            table.add_column(_("table_status"), justify="center", width=8)

            for result in results:
                if result["success"]:
                    score_color = cls._get_score_color(result["mqm_score"])
                    status_color = cls._get_status_color(result["status"])
                    status_icon = "✓" if result["status"] == "pass" else "✗"

                    table.add_row(
                        result["name"],
                        Text(f"{result['mqm_score']:.1f}", style=score_color),
                        cls._format_errors(
                            result["critical_errors"],
                            result["major_errors"],
                            result["minor_errors"],
                        ),
                        f"{result['duration']:.1f}s",
                        Text(status_icon, style=status_color),
                    )
                else:
                    table.add_row(
                        result["name"],
                        Text("ERR", style="red"),
                        "-",
                        f"{result['duration']:.1f}s",
                        Text("✗", style="red"),
                    )

            console.print(table)
            console.print()

            # Recommendation
            if best["status"] == "pass":
                console.print(
                    f"[green]✓[/green] {_('benchmark_recommendation', name=best['name'])} "
                    f"(MQM: {best['mqm_score']:.1f}, {_('table_time')}: {best['duration']:.1f}s)"
                )
            else:
                console.print(
                    f"[yellow]⚠[/yellow] {_('benchmark_no_pass', name=best['name'])} "
                    f"(MQM: {best['mqm_score']:.1f})"
                )

    @classmethod
    def print_batch_result(  # pylint: disable=unused-argument
        cls,
        total: int,
        passed: int,
        failed: int,
        avg_score: float,
        total_errors: int,
        verbose: bool = False,
    ) -> None:
        """Print batch command result.

        Args:
            total: Total files processed
            passed: Number of passed files
            failed: Number of failed files
            avg_score: Average MQM score
            total_errors: Total number of errors
            verbose: Verbose mode flag
        """
        # Header
        cls.print_header_compact(_("batch_header"))
        console.print()

        # Summary line
        pass_rate = (passed / total * 100) if total > 0 else 0
        status_icon = "✓" if failed == 0 else "✗"
        status_color = "green" if failed == 0 else "red"

        console.print(
            f"● [{status_color}]{status_icon}[/{status_color}] "
            f"{_('batch_total', total=total)}  |  "
            f"{_('batch_passed', passed=passed)}  |  "
            f"{_('batch_failed', failed=failed)}  |  "
            f"{_('batch_pass_rate', rate=int(pass_rate))}"
        )
        console.print(
            f"● {_('batch_avg_mqm', score=f'{avg_score:.1f}')}  |  "
            f"{_('batch_total_errors', count=total_errors)}"
        )
        console.print()

    @staticmethod
    def _collect_all_issues(
        errors: list[Any], nlp_insights: dict[str, Any] | None
    ) -> list[dict[str, Any]]:
        """Collect QA errors and NLP issues into a unified list.

        Args:
            errors: List of errors from QA report
            nlp_insights: Optional NLP insights with issues

        Returns:
            List of issue dictionaries with unified format
        """
        all_issues = []

        for error in errors:
            all_issues.append(
                {
                    "category": error.category,
                    "severity": error.severity.value,
                    "description": error.description,
                    "location": error.location,
                    "suggestion": error.suggestion,
                }
            )

        if nlp_insights and nlp_insights.get("issues"):
            for issue in nlp_insights["issues"]:
                all_issues.append(
                    {
                        "category": issue.get("category", "Linguistic"),
                        "severity": issue.get("severity", "minor"),
                        "description": issue.get("description", ""),
                        "location": issue.get("location", [0, 0]),
                        "suggestion": issue.get("suggestion"),
                    }
                )

        return all_issues

    @staticmethod
    def _format_location(location: Any) -> str:
        """Format location tuple/list as string."""
        if isinstance(location, list | tuple) and len(location) >= 2:
            return f"[{location[0]}:{location[1]}]"
        return "N/A"

    @staticmethod
    def _get_severity_style(severity: str) -> str:
        """Get Rich style string for severity level."""
        if severity == "critical":
            return "bold red"
        if severity == "major":
            return "bold yellow"
        return "dim"

    @classmethod
    def _render_issues_table(cls, all_issues: list[dict[str, Any]], translation_text: str) -> None:
        """Render issues as a compact table.

        Args:
            all_issues: List of issue dictionaries
            translation_text: Full translation text for extracting fragments
        """
        widths = cls._calculate_column_widths(["#", "Location", "Fragment", "Issue"])

        table = Table(
            title=_("table_issues_found"),
            show_header=True,
            header_style=HEADER_STYLE_CYAN,
            box=None,
            padding=(0, 1),
            expand=True,
        )
        table.add_column("#", style="dim", width=widths["#"], justify="right")
        table.add_column(
            _("table_location"), style="dim cyan", width=widths["location"], no_wrap=True
        )
        table.add_column(
            _("table_fragment"),
            style="yellow",
            width=widths["fragment"],
            no_wrap=False,
            overflow="fold",
        )
        table.add_column(_("table_issue"), no_wrap=False, overflow="fold")

        fragment_max = widths["fragment"] - 4 if widths["fragment"] else 25

        for idx, issue in enumerate(all_issues, 1):
            location = issue.get("location", [0, 0])
            loc_str = cls._format_location(location)
            fragment = cls._extract_fragment(translation_text, location, max_length=fragment_max)

            description = issue["description"]
            suggestion = issue.get("suggestion")
            issue_text = f"{description} → {suggestion}" if suggestion else description

            severity = issue["severity"]
            severity_style = cls._get_severity_style(severity)
            severity_badge = severity.upper()[0]
            issue_display = f"[{severity_style}]{severity_badge}[/{severity_style}] {issue_text}"

            table.add_row(str(idx), loc_str, f'"{fragment}"', issue_display)

        console.print(table)
        console.print()
        console.print(f"[dim]💡 {_('check_hint_verbose')}[/dim]")
        console.print()

    @classmethod
    def _print_issues_compact(
        cls,
        errors: list[Any],
        translation_text: str,
        nlp_insights: dict[str, Any] | None,
        verbose: bool,
    ) -> None:
        """Print issues in compact table format.

        Args:
            errors: List of errors from QA report
            translation_text: Full translation text for extracting fragments
            nlp_insights: Optional NLP insights with issues
            verbose: Show detailed descriptions
        """
        all_issues = cls._collect_all_issues(errors, nlp_insights)

        if not all_issues:
            return

        if verbose:
            cls._print_issues_verbose(all_issues, translation_text)
        else:
            cls._render_issues_table(all_issues, translation_text)

    @classmethod
    def _print_issues_verbose(
        cls,
        issues: list[dict[str, Any]],
        translation_text: str,
    ) -> None:
        """Print issues in verbose format with full context.

        Args:
            issues: List of issue dictionaries
            translation_text: Full translation text
        """
        console.print(f"[bold cyan]{_('table_issues_detailed')}[/bold cyan]\n")

        for idx, issue in enumerate(issues, 1):
            # Extract location
            location = issue.get("location", [0, 0])
            start, end = location[0], location[1]

            # Color-code severity
            severity = issue["severity"]
            if severity == "critical":
                severity_color = "bold red"
                severity_icon = "🔴"
            elif severity == "major":
                severity_color = "bold yellow"
                severity_icon = "🟡"
            else:
                severity_color = "dim"
                severity_icon = "⚪"

            # Header
            console.print(
                f"[{severity_color}]{severity_icon} [{idx}] {severity.upper()} "
                f"{issue['category']} error at {start}-{end}[/{severity_color}]"
            )

            # Extract context (50 chars before and after)
            context_start = max(0, start - 50)
            context_end = min(len(translation_text), end + 50)

            prefix = "..." if context_start > 0 else ""
            suffix = "..." if context_end < len(translation_text) else ""

            # Show text with highlighted error
            before_error = translation_text[context_start:start]
            error_text = translation_text[start:end]
            after_error = translation_text[context_end:end] if context_end > end else ""

            console.print(
                f"    Text: {prefix}{before_error}[bold yellow][{error_text}][/bold yellow]{after_error}{suffix}"
            )

            # Underline the error
            underline_pos = len(prefix) + len(before_error)
            underline = " " * (underline_pos + 10) + "^" * min(len(error_text), 20)
            console.print(f"[dim]{underline}[/dim]")

            # Issue description
            console.print(f"    [cyan]Issue:[/cyan] {issue['description']}")

            # Suggestion if available
            suggestion = issue.get("suggestion")
            if suggestion:
                console.print(f"    [green]Fix:[/green] {suggestion}")

            console.print()

        console.print()

    @classmethod
    def _print_confidence_and_agreement(cls, report: QAReport) -> None:
        """Print confidence and agent agreement details."""
        if report.confidence is not None:
            level_key = (
                "detailed_confidence_high"
                if report.confidence >= 0.8
                else (
                    "detailed_confidence_medium"
                    if report.confidence >= 0.6
                    else "detailed_confidence_low"
                )
            )
            console.print(
                f"  {_('detailed_confidence', score=f'{report.confidence:.2f}', level=_(level_key))}"
            )
        if report.agent_agreement is not None:
            console.print(f"  {_('detailed_agreement', percent=int(report.agent_agreement * 100))}")

    @classmethod
    def _print_domain_detection(cls, report: QAReport) -> None:
        """Print domain detection details."""
        if report.agent_details and "detected_domain" in report.agent_details:
            domain = report.agent_details["detected_domain"]
            domain_confidence = report.agent_details.get("domain_confidence", 0.0)
            console.print(
                f"  {_('detailed_domain', domain=domain.replace('_', ' ').title(), confidence=int(domain_confidence * 100))}"
            )

    @classmethod
    def _print_style_analysis(cls, style_profile: Any) -> None:
        """Print style analysis details."""
        console.print("\n  [bold magenta]Style Analysis[/bold magenta]")
        console.print(f"    Literary text: {'Yes' if style_profile.is_literary else 'No'}")
        console.print(
            f"    Pattern: {style_profile.detected_pattern.value.replace('_', ' ').title()}"
        )
        console.print(f"    Deviation score: {style_profile.deviation_score:.0%}")
        console.print(f"    Lexical diversity: {style_profile.lexical_diversity:.2f}")
        console.print(f"    Avg sentence length: {style_profile.avg_sentence_length:.1f} words")

        if style_profile.detected_deviations:
            console.print(f"    Detected features ({len(style_profile.detected_deviations)}):")
            for dev in style_profile.detected_deviations[:5]:
                dev_type = dev.type.value.replace("_", " ").title()
                examples = ", ".join(dev.examples[:2]) if dev.examples else ""
                console.print(f"      [dim]• {dev_type}{': ' + examples if examples else ''}[/dim]")

        adjustments = style_profile.get_agent_weight_adjustments()
        if not math.isclose(adjustments.get("fluency", 1.0), 1.0):
            console.print(
                f"    [yellow]Fluency tolerance adjusted to {adjustments['fluency']:.0%}[/yellow]"
            )
        if adjustments.get("style_preservation", 1.0) > 1.0:
            console.print(
                f"    [green]Style preservation weight increased to {adjustments['style_preservation']:.0%}[/green]"
            )

    @classmethod
    def _print_verbose_details(  # pylint: disable=unused-argument
        cls,
        report: QAReport,
        lightweight_scores: Any | None,
        rule_based_errors: list[Any] | None,
        nlp_insights: dict[str, Any] | None,
        style_profile: Any | None = None,
    ) -> None:
        """Print verbose details (confidence, agent scores, etc.)."""
        console.print(f"\n[bold]{_('detailed_metrics')}[/bold]")

        cls._print_confidence_and_agreement(report)
        cls._print_domain_detection(report)

        if report.agent_scores:
            console.print(f"\n  {_('detailed_agent_scores')}")
            for agent_name, score in sorted(report.agent_scores.items()):
                score_color = cls._get_score_color(score)
                console.print(
                    f"    {agent_name.replace('_', ' ').title()}: [{score_color}]{score:.2f}[/{score_color}]"
                )

        if nlp_insights and nlp_insights.get("good_indicators"):
            console.print(f"\n  {_('detailed_linguistic_passed')}")
            for indicator in nlp_insights["good_indicators"]:
                console.print(f"    [dim]✓ {indicator}[/dim]")

        if rule_based_errors:
            console.print(f"\n  {_('detailed_rule_errors', count=len(rule_based_errors))}")
            for error in rule_based_errors[:3]:
                console.print(f"    [dim]• {error.check_type}: {error.description[:50]}[/dim]")

        if style_profile:
            cls._print_style_analysis(style_profile)

        console.print()
