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

from typing import Any

from rich.table import Table
from rich.text import Text

from kttc.core import QAReport
from kttc.utils.console import console


class ConsoleFormatter:
    """Unified console output formatter for all CLI commands.

    Provides two modes:
    - Compact (default): ~10-15 lines, essential info only
    - Verbose: ~25-30 lines, detailed info
    """

    @staticmethod
    def _get_status_color(status: str) -> str:
        """Get color for status."""
        return "green" if status == "pass" else "red"

    @staticmethod
    def _get_score_color(score: float) -> str:
        """Get color for MQM score."""
        if score >= 95:
            return "green"
        elif score >= 85:
            return "yellow"
        else:
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
            verbose: Verbose mode flag
        """
        # Header
        cls.print_header_compact("Translation Quality Check", f"{source_lang} → {target_lang}")
        console.print()

        # Main status line
        status_color = cls._get_status_color(report.status)
        score_color = cls._get_score_color(report.mqm_score)
        status_icon = "✓" if report.status == "pass" else "✗"

        status_line = (
            f"[{status_color}]{status_icon} {report.status.upper()}[/{status_color}]  |  "
            f"MQM: [{score_color}]{report.mqm_score:.1f}[/{score_color}]/100  |  "
            f"Errors: {len(report.errors)}"
        )

        if len(report.errors) > 0:
            status_line += f" ({cls._format_errors(report.critical_error_count, report.major_error_count, report.minor_error_count)})"

        console.print(f"● {status_line}")

        # Metrics line (if available)
        if lightweight_scores and rule_based_score is not None:
            metrics_parts = [
                f"chrF: {lightweight_scores.chrf:.1f}",
                f"BLEU: {lightweight_scores.bleu:.1f}",
                f"TER: {lightweight_scores.ter:.1f}",
                f"Rule-based: {rule_based_score:.0f}/100",
            ]
            console.print(f"● Metrics: {' | '.join(metrics_parts)}")

        console.print()

        # Show errors/warnings (compact table)
        if len(report.errors) > 0 or (nlp_insights and nlp_insights.get("issues")):
            cls._print_issues_compact(report.errors, nlp_insights, verbose)

        # Verbose mode: additional details
        if verbose:
            cls._print_verbose_details(report, lightweight_scores, rule_based_errors, nlp_insights)

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
        cls.print_header_compact("Translation Comparison", f"{source_lang} → {target_lang}")
        console.print()

        # Handle empty results
        if not results:
            console.print("[yellow]No translation results to compare[/yellow]")
            return

        # Summary line
        best = max(results, key=lambda r: r["mqm_score"])
        avg_score = sum(r["mqm_score"] for r in results) / len(results)

        console.print(
            f"● Compared: {len(results)} translations  |  "
            f"Avg MQM: {avg_score:.1f}  |  "
            f"Best: [bold cyan]{best['name']}[/bold cyan] ({best['mqm_score']:.1f})"
        )
        console.print()

        # Results table (compact)
        table = Table(show_header=True, header_style="bold cyan", box=None, padding=(0, 1))
        table.add_column("Translation", style="cyan")
        table.add_column("MQM", justify="right", width=8)
        table.add_column("Errors", justify="center", width=10)
        table.add_column("Status", justify="center", width=8)

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
                    cls._print_issues_compact(result["errors"], None, verbose=True)

    @classmethod
    def print_benchmark_result(
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
        cls.print_header_compact("Provider Benchmark", f"{source_lang} → {target_lang}")
        console.print()

        # Summary line
        successful = [r for r in results if r["success"]]
        if successful:
            best = max(successful, key=lambda r: r["mqm_score"])
            avg_mqm = sum(r["mqm_score"] for r in successful) / len(successful)
            avg_time = sum(r["duration"] for r in successful) / len(successful)

            console.print(
                f"● Tested: {len(results)} providers  |  "
                f"Avg MQM: {avg_mqm:.1f}  |  "
                f"Avg time: {avg_time:.1f}s  |  "
                f"Best: [bold cyan]{best['name']}[/bold cyan]"
            )
            console.print()

            # Results table (compact)
            table = Table(show_header=True, header_style="bold cyan", box=None, padding=(0, 1))
            table.add_column("Provider", style="cyan")
            table.add_column("MQM", justify="right", width=8)
            table.add_column("Errors", justify="center", width=10)
            table.add_column("Time", justify="right", width=8)
            table.add_column("Status", justify="center", width=8)

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
                    f"[green]✓[/green] Recommendation: Use [bold cyan]{best['name']}[/bold cyan] "
                    f"(MQM: {best['mqm_score']:.1f}, Time: {best['duration']:.1f}s)"
                )
            else:
                console.print(
                    f"[yellow]⚠[/yellow] No provider passed threshold. "
                    f"Best: [bold cyan]{best['name']}[/bold cyan] (MQM: {best['mqm_score']:.1f})"
                )

    @classmethod
    def print_batch_result(
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
        cls.print_header_compact("Batch Processing Complete")
        console.print()

        # Summary line
        pass_rate = (passed / total * 100) if total > 0 else 0
        status_icon = "✓" if failed == 0 else "✗"
        status_color = "green" if failed == 0 else "red"

        console.print(
            f"● [{status_color}]{status_icon}[/{status_color}] "
            f"Total: {total}  |  "
            f"Passed: [green]{passed}[/green]  |  "
            f"Failed: [red]{failed}[/red]  |  "
            f"Pass rate: {pass_rate:.0f}%"
        )
        console.print(f"● Avg MQM: {avg_score:.1f}/100  |  " f"Total errors: {total_errors}")
        console.print()

    @classmethod
    def _print_issues_compact(
        cls,
        errors: list[Any],
        nlp_insights: dict[str, Any] | None,
        verbose: bool,
    ) -> None:
        """Print issues in compact table format.

        Args:
            errors: List of errors from QA report
            nlp_insights: Optional NLP insights with issues
            verbose: Show detailed descriptions
        """
        # Combine QA errors and NLP issues
        all_issues = []

        for error in errors:
            all_issues.append(
                {
                    "category": error.category,
                    "severity": error.severity.value,
                    "description": error.description,
                }
            )

        if nlp_insights and nlp_insights.get("issues"):
            for issue in nlp_insights["issues"]:
                all_issues.append(
                    {
                        "category": issue.get("category", "Linguistic"),
                        "severity": issue.get("severity", "minor"),
                        "description": issue.get("description", ""),
                    }
                )

        if not all_issues:
            return

        # Create compact issues table
        table = Table(
            title="Issues Found",
            show_header=True,
            header_style="bold cyan",
            box=None,
            padding=(0, 1),
        )
        table.add_column("Category", style="cyan", width=12)
        table.add_column("Severity", width=10)
        table.add_column("Description", max_width=70 if verbose else 50)

        for issue in all_issues:
            # Color-code severity
            severity = issue["severity"]
            if severity == "critical":
                severity_color = "red"
            elif severity == "major":
                severity_color = "yellow"
            else:
                severity_color = "dim"

            # Truncate description if not verbose
            description = issue["description"]
            if not verbose and len(description) > 50:
                description = description[:47] + "..."

            table.add_row(
                issue["category"],
                Text(severity.upper(), style=severity_color),
                description,
            )

        console.print(table)
        console.print()

    @classmethod
    def _print_verbose_details(
        cls,
        report: QAReport,
        lightweight_scores: Any | None,
        rule_based_errors: list[Any] | None,
        nlp_insights: dict[str, Any] | None,
    ) -> None:
        """Print verbose details (confidence, agent scores, etc.).

        Args:
            report: QA report
            lightweight_scores: Optional lightweight metrics scores
            rule_based_errors: Optional rule-based errors
            nlp_insights: Optional NLP insights
        """
        console.print("\n[bold]Detailed Metrics:[/bold]")

        # Confidence and agent agreement
        if report.confidence is not None:
            conf_color = (
                "green"
                if report.confidence >= 0.8
                else "yellow" if report.confidence >= 0.6 else "red"
            )
            console.print(
                f"  Confidence: [{conf_color}]{report.confidence:.2f}[/{conf_color}] "
                f"({'high' if report.confidence >= 0.8 else 'medium' if report.confidence >= 0.6 else 'low'})"
            )

        if report.agent_agreement is not None:
            console.print(f"  Agent Agreement: {report.agent_agreement * 100:.0f}%")

        # Domain detection
        if report.agent_details and "detected_domain" in report.agent_details:
            domain = report.agent_details["detected_domain"]
            domain_confidence = report.agent_details.get("domain_confidence", 0.0)
            console.print(
                f"  Domain: {domain.replace('_', ' ').title()} ({domain_confidence:.0%} confidence)"
            )

        # Per-agent scores
        if report.agent_scores:
            console.print("\n  Per-Agent Scores:")
            for agent_name, score in sorted(report.agent_scores.items()):
                score_color = cls._get_score_color(score)
                console.print(
                    f"    {agent_name.replace('_', ' ').title()}: [{score_color}]{score:.2f}[/{score_color}]"
                )

        # NLP good indicators
        if nlp_insights and nlp_insights.get("good_indicators"):
            console.print("\n  Linguistic Checks Passed:")
            for indicator in nlp_insights["good_indicators"]:
                console.print(f"    [dim]✓ {indicator}[/dim]")

        # Rule-based errors (if any and not shown in main table)
        if rule_based_errors:
            console.print(f"\n  Rule-Based Errors: {len(rule_based_errors)}")
            for error in rule_based_errors[:3]:  # Show first 3
                console.print(f"    [dim]• {error.check_type}: {error.description[:50]}[/dim]")

        console.print()
