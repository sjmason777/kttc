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

"""Error span visualization for XCOMET results.

Provides utilities to visualize detected error spans in translations
with color-coded severity levels for terminal and HTML output.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .neural import ErrorSpan


class ErrorSpanVisualizer:
    """Visualize XCOMET error spans with color-coding.

    Example:
        >>> from kttc.metrics import NeuralMetrics, ErrorSpanVisualizer
        >>> metrics = NeuralMetrics(use_xcomet=True)
        >>> await metrics.initialize()
        >>> result = await metrics.evaluate(...)
        >>> visualizer = ErrorSpanVisualizer()
        >>> print(visualizer.format_terminal(result.translation, result.error_spans))
    """

    # ANSI color codes for terminal
    COLORS = {
        "critical": "\033[91m",  # Bright red
        "major": "\033[93m",  # Yellow
        "minor": "\033[94m",  # Blue
        "reset": "\033[0m",  # Reset
    }

    # HTML colors
    HTML_COLORS = {
        "critical": "#FF4444",  # Red
        "major": "#FFAA00",  # Orange
        "minor": "#4488FF",  # Blue
    }

    def format_terminal(self, text: str, error_spans: list[ErrorSpan]) -> str:
        """Format text with color-coded error spans for terminal output.

        Args:
            text: Original translation text
            error_spans: List of detected error spans

        Returns:
            Formatted string with ANSI color codes

        Example:
            >>> visualizer = ErrorSpanVisualizer()
            >>> spans = [ErrorSpan(text="error", start=0, end=5, severity="major", confidence=0.9)]
            >>> print(visualizer.format_terminal("error text", spans))
        """
        if not error_spans:
            return text

        # Sort spans by start position
        sorted_spans = sorted(error_spans, key=lambda s: s.start)

        result = []
        last_end = 0

        for span in sorted_spans:
            # Add text before span
            result.append(text[last_end : span.start])

            # Add colored span
            color = self.COLORS.get(span.severity.lower(), self.COLORS["minor"])
            result.append(f"{color}{text[span.start : span.end]}{self.COLORS['reset']}")

            last_end = span.end

        # Add remaining text
        result.append(text[last_end:])

        return "".join(result)

    def format_html(self, text: str, error_spans: list[ErrorSpan]) -> str:
        """Format text with color-coded error spans for HTML output.

        Args:
            text: Original translation text
            error_spans: List of detected error spans

        Returns:
            HTML formatted string with colored spans

        Example:
            >>> visualizer = ErrorSpanVisualizer()
            >>> spans = [ErrorSpan(text="error", start=0, end=5, severity="critical", confidence=0.95)]
            >>> html = visualizer.format_html("error text", spans)
        """
        if not error_spans:
            return f"<p>{text}</p>"

        # Sort spans by start position
        sorted_spans = sorted(error_spans, key=lambda s: s.start)

        result = []
        last_end = 0

        for span in sorted_spans:
            # Add text before span
            if span.start > last_end:
                result.append(text[last_end : span.start])

            # Add colored span with tooltip
            color = self.HTML_COLORS.get(span.severity.lower(), self.HTML_COLORS["minor"])
            title = f"{span.severity} error (confidence: {span.confidence:.2f})"
            result.append(
                f'<span style="background-color: {color}; '
                f'padding: 2px 4px; border-radius: 3px;" '
                f'title="{title}">{text[span.start : span.end]}</span>'
            )

            last_end = span.end

        # Add remaining text
        if last_end < len(text):
            result.append(text[last_end:])

        return f"<p>{''.join(result)}</p>"

    def format_markdown(self, text: str, error_spans: list[ErrorSpan]) -> str:
        """Format text with error spans for Markdown output.

        Args:
            text: Original translation text
            error_spans: List of detected error spans

        Returns:
            Markdown formatted string with error annotations

        Example:
            >>> visualizer = ErrorSpanVisualizer()
            >>> md = visualizer.format_markdown(text, spans)
        """
        if not error_spans:
            return text

        result = [f"**Translation**: {text}\n"]
        result.append("\n**Detected Errors**:\n")

        # Sort spans by severity
        severity_order = {"critical": 0, "major": 1, "minor": 2}
        sorted_spans = sorted(
            error_spans, key=lambda s: (severity_order.get(s.severity.lower(), 3), s.start)
        )

        for i, span in enumerate(sorted_spans, 1):
            emoji = {"critical": "🔴", "major": "🟡", "minor": "🔵"}.get(
                span.severity.lower(), "⚪"
            )
            result.append(
                f"{i}. {emoji} **{span.severity.upper()}** "
                f"[{span.start}:{span.end}]: `{span.text}` "
                f"(confidence: {span.confidence:.2f})\n"
            )

        return "".join(result)

    def get_summary(self, error_spans: list[ErrorSpan]) -> dict[str, int]:
        """Get summary statistics of error spans by severity.

        Args:
            error_spans: List of detected error spans

        Returns:
            Dictionary with counts by severity level

        Example:
            >>> visualizer = ErrorSpanVisualizer()
            >>> summary = visualizer.get_summary(spans)
            >>> print(f"Critical: {summary['critical']}")
        """
        summary = {"critical": 0, "major": 0, "minor": 0, "total": len(error_spans)}

        for span in error_spans:
            severity = span.severity.lower()
            if severity in summary:
                summary[severity] += 1

        return summary
