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

"""Markdown formatter for translation quality reports."""

from pathlib import Path

from kttc.core.models import QAReport


class MarkdownFormatter:
    """Format QA reports as Markdown."""

    @staticmethod
    def format_report(report: QAReport, output_path: str | Path | None = None) -> str:
        """Format a QA report as Markdown.

        Args:
            report: QA report to format
            output_path: Optional path to save the report

        Returns:
            Markdown-formatted report as string
        """
        md = ["# Translation Quality Report\n", "## Translation Details\n", f"- **Source Language**: {report.task.source_lang}", f"- **Target Language**: {report.task.target_lang}", f"- **Word Count**: {report.task.word_count}", "", "## Quality Scores\n", f"- **MQM Score**: {report.mqm_score:.2f}/100", f"- **Status**: {'✅ PASS' if report.status == 'pass' else '❌ FAIL'}", "", "## Translation Text\n", "### Source", "```", report.task.source_text, "```\n", "### Translation", "```", report.task.translation, "```\n"]

        if report.comet_score is not None:
            md.append(f"- **COMET Score**: {report.comet_score:.4f}")
        if report.kiwi_score is not None:
            md.append(f"- **KIWI Score**: {report.kiwi_score:.4f}")

        # Errors
        if report.errors:
            md.append(f"## Issues Found ({len(report.errors)})\n")
            md.append("| Category | Subcategory | Severity | Location | Description |")
            md.append("|----------|-------------|----------|----------|-------------|")

            for error in report.errors:
                location = f"{error.location[0]}-{error.location[1]}"
                # Escape pipe characters in description
                description = error.description.replace("|", "\\|")
                md.append(
                    f"| {error.category} | {error.subcategory} | "
                    f"{error.severity.value.upper()} | {location} | {description} |"
                )

            md.append("")
        else:
            md.append("## Issues Found\n")
            md.append("✅ No issues detected!\n")

        # Join and optionally save
        result = "\n".join(md)

        if output_path:
            Path(output_path).write_text(result, encoding="utf-8")

        return result
