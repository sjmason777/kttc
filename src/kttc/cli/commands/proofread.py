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

"""Proofread command for grammar, spelling, and punctuation checking."""

from __future__ import annotations

import asyncio

import typer

from kttc.cli.commands.self_check import self_check_async
from kttc.cli.ui import console

# Create Typer app for proofread command
proofread_app = typer.Typer()


@proofread_app.command(name="proofread")
def proofread(
    file: str = typer.Argument(..., help="File to proofread"),
    lang: str = typer.Option(..., "--lang", "-l", help="Language code (e.g., 'ru', 'en', 'zh')"),
    threshold: float = typer.Option(95.0, "--threshold", help="Quality threshold (0-100)"),
    output: str | None = typer.Option(None, "--output", "-o", help="Output report file path"),
    provider: str | None = typer.Option(
        None, "--provider", help="LLM provider for context-aware checking"
    ),
    verbose: bool = typer.Option(False, "--verbose", help="Verbose output"),
) -> None:
    """
    Proofread a text file for grammar, spelling, and punctuation errors.

    This is an alias for `kttc check --self` with a simpler interface.
    Uses school curriculum rules and optional LLM for context-aware checking.

    Supported languages: en, ru, zh, hi, fa

    Examples:
        # Proofread a Russian article
        kttc proofread article.md --lang ru

        # Proofread with strict threshold
        kttc proofread article.md --lang ru --threshold 98

        # Save report
        kttc proofread article.md --lang ru --output report.json --verbose
    """
    try:
        asyncio.run(
            self_check_async(
                source=file,
                language=lang,
                threshold=threshold,
                output=output,
                format="json",
                provider=provider,
                verbose=verbose,
                demo=False,
            )
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠ Interrupted by user[/yellow]")
        raise typer.Exit(code=130)
    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"\n[red]✗ Error: {e}[/red]")
        if verbose:
            console.print_exception()
        raise typer.Exit(code=1)
