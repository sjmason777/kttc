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

"""Report command for generating formatted reports from QA results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer

from kttc.cli.ui import console

# Create Typer app for report command
report_app = typer.Typer()


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


@report_app.command(name="report")
def report(
    input_file: str = typer.Argument(..., help="Input report file (JSON)"),
    output_format: str = typer.Option(
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
        console.print(f"\n[bold]Generating {output_format} report...[/bold]")
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
            ext = ".md" if output_format == "markdown" else ".html"
            output_path = input_path.with_suffix(ext)

        # Generate report
        if output_format == "markdown":
            content = _generate_batch_markdown_report(data)
        elif output_format == "html":
            content = _generate_batch_html_report(data)
        else:
            raise ValueError(f"Unsupported format: {output_format}. Use 'markdown' or 'html'")

        # Save report
        output_path.write_text(content, encoding="utf-8")

        console.print("[green]✓ Report generated successfully[/green]")
        console.print(f"Output: [cyan]{output_path}[/cyan]")

    except Exception as e:
        console.print(f"\n[red]✗ Error: {e}[/red]")
        raise typer.Exit(code=1)
