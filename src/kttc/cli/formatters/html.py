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

"""HTML formatter for translation quality reports."""

import html as html_lib
from pathlib import Path

from kttc.core.models import QAReport


class HTMLFormatter:
    """Format QA reports as HTML."""

    @staticmethod
    def format_report(report: QAReport, output_path: str | Path | None = None) -> str:
        """Format a QA report as HTML with styling.

        Args:
            report: QA report to format
            output_path: Optional path to save the report

        Returns:
            HTML-formatted report as string
        """
        # Escape HTML
        source_text = html_lib.escape(report.task.source_text)
        translation_text = html_lib.escape(report.task.translation)

        # Determine status color
        status_color = "#28a745" if report.status == "pass" else "#dc3545"
        status_text = "✅ PASS" if report.status == "pass" else "❌ FAIL"

        # Build HTML
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Translation Quality Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f8f9fa;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 20px;
        }}
        .header h1 {{
            margin: 0;
            font-size: 2em;
        }}
        .score-card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .score-large {{
            font-size: 3em;
            font-weight: bold;
            color: {status_color};
            margin: 10px 0;
        }}
        .status {{
            font-size: 1.5em;
            color: {status_color};
            font-weight: bold;
        }}
        .details-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }}
        .detail-item {{
            background: #f8f9fa;
            padding: 10px 15px;
            border-radius: 5px;
        }}
        .detail-label {{
            font-size: 0.85em;
            color: #6c757d;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .detail-value {{
            font-size: 1.1em;
            font-weight: 600;
            margin-top: 5px;
        }}
        .errors-table {{
            width: 100%;
            border-collapse: collapse;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .errors-table th {{
            background: #343a40;
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: 600;
        }}
        .errors-table td {{
            padding: 12px;
            border-bottom: 1px solid #dee2e6;
        }}
        .errors-table tr:last-child td {{
            border-bottom: none;
        }}
        .severity-critical {{
            color: #dc3545;
            font-weight: bold;
        }}
        .severity-major {{
            color: #ffc107;
            font-weight: bold;
        }}
        .severity-minor {{
            color: #6c757d;
        }}
        .text-block {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .text-block h3 {{
            margin-top: 0;
            color: #343a40;
        }}
        .text-content {{
            font-family: 'Courier New', monospace;
            background: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            border-left: 4px solid #667eea;
            white-space: pre-wrap;
            line-height: 1.6;
        }}
        .no-issues {{
            text-align: center;
            padding: 40px;
            background: white;
            border-radius: 8px;
            color: #28a745;
            font-size: 1.2em;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Translation Quality Report</h1>
        <p style="margin: 5px 0 0 0; opacity: 0.9;">Generated by KTTC AI</p>
    </div>

    <div class="score-card">
        <div class="score-large">{report.mqm_score:.2f}/100</div>
        <div class="status">{status_text}</div>

        <div class="details-grid">
            <div class="detail-item">
                <div class="detail-label">Source Language</div>
                <div class="detail-value">{report.task.source_lang.upper()}</div>
            </div>
            <div class="detail-item">
                <div class="detail-label">Target Language</div>
                <div class="detail-value">{report.task.target_lang.upper()}</div>
            </div>
            <div class="detail-item">
                <div class="detail-label">Word Count</div>
                <div class="detail-value">{report.task.word_count}</div>
            </div>
            <div class="detail-item">
                <div class="detail-label">Issues Found</div>
                <div class="detail-value">{len(report.errors)}</div>
            </div>
"""

        if report.comet_score is not None:
            html += f"""            <div class="detail-item">
                <div class="detail-label">COMET Score</div>
                <div class="detail-value">{report.comet_score:.4f}</div>
            </div>
"""

        html += """        </div>
    </div>
"""

        # Errors table
        if report.errors:
            html += f"""    <h2 style="margin-top: 30px;">Issues Found ({len(report.errors)})</h2>
    <table class="errors-table">
        <thead>
            <tr>
                <th>Category</th>
                <th>Subcategory</th>
                <th>Severity</th>
                <th>Location</th>
                <th>Description</th>
            </tr>
        </thead>
        <tbody>
"""
            for error in report.errors:
                severity_class = f"severity-{error.severity.value}"
                location = f"{error.location[0]}-{error.location[1]}"
                description = html_lib.escape(error.description)

                html += f"""            <tr>
                <td>{error.category}</td>
                <td>{error.subcategory}</td>
                <td class="{severity_class}">{error.severity.value.upper()}</td>
                <td>{location}</td>
                <td>{description}</td>
            </tr>
"""

            html += """        </tbody>
    </table>
"""
        else:
            html += """    <div class="no-issues">
        <h2>✅ No Issues Detected!</h2>
        <p>Translation quality is excellent.</p>
    </div>
"""

        # Source and translation
        html += f"""
    <h2 style="margin-top: 30px;">Translation Text</h2>

    <div class="text-block">
        <h3>Source</h3>
        <div class="text-content">{source_text}</div>
    </div>

    <div class="text-block">
        <h3>Translation</h3>
        <div class="text-content">{translation_text}</div>
    </div>
</body>
</html>
"""

        if output_path:
            Path(output_path).write_text(html, encoding="utf-8")

        return html
