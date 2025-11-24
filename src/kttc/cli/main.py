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

"""Main CLI application entry point for KTTC.

KTTC - Knowledge Translation Transmutation Core
Transforming translations into gold-standard quality.

This module serves as the central entry point for the CLI application.
All commands are organized in separate modules under `kttc.cli.commands/`.
"""

from __future__ import annotations

import asyncio
import warnings

import typer

from kttc import __version__
from kttc.cli.commands.batch import batch
from kttc.cli.commands.benchmark import run_benchmark
from kttc.cli.commands.check import check
from kttc.cli.commands.compare import run_compare as _run_compare_command
from kttc.cli.commands.glossary import glossary_app
from kttc.cli.commands.lint import lint
from kttc.cli.commands.proofread import proofread
from kttc.cli.commands.report import report
from kttc.cli.commands.terminology import terminology_app
from kttc.cli.commands.translate import translate
from kttc.cli.ui import console
from kttc.i18n import get_supported_languages, set_language

# Suppress warnings from external dependencies
warnings.filterwarnings("ignore", category=UserWarning, module="jieba._compat")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.cuda")

# Create main app
app = typer.Typer(
    name="kttc",
    help="KTTC - Knowledge Translation Transmutation Core\n\nTransforming translations into gold-standard quality through autonomous multi-agent AI systems.",
    add_completion=True,
    rich_markup_mode="rich",
    no_args_is_help=True,
)

# Register commands
app.command()(check)
app.command()(translate)
app.command()(batch)
app.command()(report)
app.command()(proofread)
app.command()(lint)

# Add sub-apps for grouped commands
app.add_typer(glossary_app, name="glossary")
app.add_typer(terminology_app, name="terminology")


def version_callback(value: bool) -> None:
    """Print version and exit."""
    if value:
        console.print(f"KTTC version: [bold cyan]{__version__}[/bold cyan]")
        raise typer.Exit()


def ui_lang_callback(value: str | None) -> None:
    """Set UI language."""
    if value:
        set_language(value)


@app.callback()
def main(
    version: bool = typer.Option(  # noqa: ARG001 - Used by Typer callback
        False,
        "--version",
        "-v",
        help="Show version and exit",
        callback=version_callback,
        is_eager=True,
    ),
    ui_lang: str | None = typer.Option(  # noqa: ARG001 - Used by Typer callback
        None,
        "--ui-lang",
        "-L",
        help=f"UI language ({', '.join(get_supported_languages())}) or 'auto' for system detection",
        callback=ui_lang_callback,
        is_eager=True,
    ),
) -> None:
    """
    KTTC - Knowledge Translation Transmutation Core.

    Transforming translations into gold-standard quality through
    autonomous multi-agent AI systems.
    """
    pass


@app.command()
def benchmark(
    source: str = typer.Option(..., "--source", "-s", help="Source text file path"),
    source_lang: str = typer.Option(..., "--source-lang", help="Source language code (e.g., 'en')"),
    target_lang: str = typer.Option(..., "--target-lang", help="Target language code (e.g., 'ru')"),
    providers: str = typer.Option(
        "gigachat,openai,anthropic",
        "--providers",
        "-p",
        help="Comma-separated list of providers to benchmark",
    ),
    threshold: float = typer.Option(95.0, "--threshold", help="Quality threshold for evaluation"),
    output: str | None = typer.Option(
        None, "--output", "-o", help="Output file path for results (JSON)"
    ),
    verbose: bool = typer.Option(False, "--verbose", help="Verbose output"),
) -> None:
    """
    Benchmark multiple LLM providers.

    Compare translation quality and performance across different LLM providers.
    Generates translations and evaluates them using MQM metrics.

    Example:
        kttc benchmark --source text.txt --source-lang en --target-lang ru \\
                      --providers gigachat,openai,anthropic
    """
    # Check models with loader
    from kttc.cli.ui import check_models_with_loader

    if not check_models_with_loader():
        raise typer.Exit(code=1)

    try:
        provider_list = [p.strip() for p in providers.split(",")]
        asyncio.run(
            run_benchmark(
                source=source,
                source_lang=source_lang,
                target_lang=target_lang,
                providers=provider_list,
                threshold=threshold,
                output=output,
                verbose=verbose,
            )
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠ Interrupted by user[/yellow]")
        raise typer.Exit(code=130)
    except Exception as e:
        console.print(f"\n[red]✗ Error: {e}[/red]")
        if verbose:
            console.print_exception()
        raise typer.Exit(code=1)


@app.command()
def compare(
    source: str = typer.Option(..., "--source", "-s", help="Source text file path"),
    translations: list[str] = typer.Option(
        ..., "--translation", "-t", help="Translation file paths (can be specified multiple times)"
    ),
    source_lang: str = typer.Option(..., "--source-lang", help="Source language code (e.g., 'en')"),
    target_lang: str = typer.Option(..., "--target-lang", help="Target language code (e.g., 'ru')"),
    threshold: float = typer.Option(95.0, "--threshold", help="Quality threshold for evaluation"),
    provider: str | None = typer.Option(
        None, "--provider", help="LLM provider for evaluation (openai or anthropic)"
    ),
    verbose: bool = typer.Option(False, "--verbose", help="Show detailed error information"),
    demo: bool = typer.Option(
        False, "--demo", help="Demo mode (no API calls, simulated responses)"
    ),
) -> None:
    """
    Compare multiple translations side by side.

    Evaluate and compare multiple translation candidates to determine
    which one has the best quality using MQM metrics.

    Example:
        kttc compare --source text.txt --translation trans1.txt --translation trans2.txt \\
                    --translation trans3.txt --source-lang en --target-lang ru --verbose
    """
    # Check models with loader
    from kttc.cli.ui import check_models_with_loader

    if not check_models_with_loader():
        raise typer.Exit(code=1)

    try:
        asyncio.run(
            _run_compare_command(
                source=source,
                translations=translations,
                source_lang=source_lang,
                target_lang=target_lang,
                threshold=threshold,
                provider=provider,
                verbose=verbose,
                demo=demo,
            )
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠ Interrupted by user[/yellow]")
        raise typer.Exit(code=130)
    except Exception as e:
        console.print(f"\n[red]✗ Error: {e}[/red]")
        if verbose:
            console.print_exception()
        raise typer.Exit(code=1)


def run() -> None:
    """Entry point for CLI."""
    app()


if __name__ == "__main__":
    run()
