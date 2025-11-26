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

"""Lint command for quick rule-based text checking."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import typer

from kttc.cli.ui import console

# Create Typer app for lint command
lint_app = typer.Typer()


@lint_app.command(name="lint")
def lint(
    file: str = typer.Argument(..., help="File to lint"),
    lang: str = typer.Option(..., "--lang", "-l", help="Language code (e.g., 'ru', 'en', 'zh')"),
    strict: bool = typer.Option(False, "--strict", help="Strict mode: fail on any error"),
    fix: bool = typer.Option(False, "--fix", help="Show suggestions for fixing errors"),
) -> None:
    """
    Quick lint check for common errors (no LLM, fast).

    Fast rule-based checking using school curriculum rules and patterns.
    Does not use LLM - ideal for CI/CD pipelines and pre-commit hooks.

    Exit codes:
        0 - No errors found
        1 - Errors found

    Examples:
        # Quick lint check
        kttc lint article.md --lang ru

        # Strict mode (fail on any error)
        kttc lint article.md --lang ru --strict

        # Show fix suggestions
        kttc lint article.md --lang ru --fix
    """
    from kttc.agents.proofreading import SpellingAgent

    try:
        # Load file
        file_path = Path(file)
        if not file_path.exists():
            console.print(f"[red]Error: File not found: {file}[/red]")
            raise typer.Exit(code=1)

        text = file_path.read_text(encoding="utf-8")

        console.print(f"[cyan]Linting {file_path.name}...[/cyan]")

        # Run fast rule-based check only
        def run_lint() -> list[Any]:
            agent = SpellingAgent(
                llm_provider=None,
                language=lang,
                use_patterns=True,
                use_school_rules=True,
            )
            return agent.check(text)

        errors = run_lint()

        if errors:
            console.print(f"\n[yellow]Found {len(errors)} issue(s):[/yellow]\n")

            for error in errors:
                severity_icon = (
                    "🔴"
                    if error.severity.value == "critical"
                    else "🟡" if error.severity.value == "major" else "⚪"
                )
                console.print(
                    f"  {severity_icon} Line ~{error.location[0] // 50 + 1}: {error.description}"
                )

                if fix and error.suggestion:
                    console.print(f"     [green]Fix: → '{error.suggestion}'[/green]")

            if strict or any(e.severity.value == "critical" for e in errors):
                console.print("\n[red]✗ Lint failed[/red]")
                raise typer.Exit(code=1)
            console.print("\n[yellow]⚠ Lint completed with warnings[/yellow]")
        else:
            console.print("\n[green]✓ No issues found[/green]")

    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"\n[red]✗ Error: {e}[/red]")
        raise typer.Exit(code=1)
