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

"""Translate command for AI-powered translation with quality assurance."""

from __future__ import annotations

import asyncio
from pathlib import Path

import typer

from kttc.agents import AgentOrchestrator
from kttc.cli.ui import console, create_step_progress, print_header, print_startup_info
from kttc.cli.utils import setup_llm_provider
from kttc.core import TranslationTask
from kttc.utils.config import get_settings

# Create Typer app for translate command
translate_app = typer.Typer()


async def _translate_async(
    text: str,
    source_lang: str,
    target_lang: str,
    threshold: float,
    max_iterations: int,
    output: str | None,
    provider: str | None,
    verbose: bool,
) -> None:
    """Async implementation of translate command using TEaR loop."""
    from kttc.agents.refinement import IterativeRefinement

    # Load settings
    settings = get_settings()

    # Display header
    print_header(
        "AI Translation with Quality Assurance",
        "Generate high-quality translations using TEaR (Translate-Estimate-Refine) loop",
    )

    # Configuration info
    config_info = {
        "Languages": f"{source_lang} → {target_lang}",
        "Quality Threshold": f"{threshold}",
        "Max Iterations": f"{max_iterations}",
    }
    print_startup_info(config_info)

    # Load text (from file if starts with @)
    if text.startswith("@"):
        text_path = Path(text[1:])
        if not text_path.exists():
            raise FileNotFoundError(f"Text file not found: {text[1:]}")
        source_text = text_path.read_text(encoding="utf-8")
        if verbose:
            console.print(f"[dim]Loaded {len(source_text)} chars from {text_path}[/dim]\n")
    else:
        source_text = text

    # Setup LLM provider
    try:
        llm_provider = setup_llm_provider(provider, settings, verbose)
    except Exception as e:
        raise RuntimeError(f"Failed to setup LLM provider: {e}") from e

    # Step 1: Generate initial translation
    with create_step_progress() as progress:
        progress.add_task("[cyan]Generating initial translation...[/cyan]", total=None)
        try:
            translation_prompt = f"""Translate the following text from {source_lang} to {target_lang}.
Provide only the translation without any explanation.

Text to translate:
{source_text}

Translation:"""

            initial_translation = await llm_provider.complete(
                translation_prompt,
                temperature=settings.default_temperature,
                max_tokens=settings.default_max_tokens,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to generate translation: {e}") from e

    console.print("[green]✓[/green] Initial translation generated")
    if verbose:
        console.print(f"[dim]   Preview: {initial_translation[:100]}...[/dim]")
    console.print()

    # Step 2: Iterative refinement (TEaR loop)
    console.print("[cyan]Running TEaR (Translate-Estimate-Refine) loop...[/cyan]")
    try:
        # Create initial task
        task = TranslationTask(
            source_text=source_text,
            translation=initial_translation,
            source_lang=source_lang,
            target_lang=target_lang,
        )

        # Create orchestrator for evaluation
        orchestrator = AgentOrchestrator(
            llm_provider,
            quality_threshold=threshold,
            agent_temperature=settings.default_temperature,
            agent_max_tokens=settings.default_max_tokens,
        )

        # Create refinement engine
        refinement = IterativeRefinement(
            llm_provider=llm_provider,
            max_iterations=max_iterations,
            convergence_threshold=threshold,
            min_improvement=1.0,
        )

        # Run refinement
        result = await refinement.refine_until_convergence(task, orchestrator)

        # Display results
        console.print("\n[bold]Final Results:[/bold]")
        console.print(f"Final MQM Score: [cyan]{result.final_score:.2f}[/cyan]")
        console.print(f"Iterations: [cyan]{result.iterations}[/cyan]")
        console.print(f"Improvement: [cyan]+{result.improvement:.2f}[/cyan] points")
        console.print(
            f"Status: "
            f"[{'green' if result.converged else 'yellow'}]"
            f"{'Converged' if result.converged else 'Max iterations reached'}"
            f"[/{'green' if result.converged else 'yellow'}]"
        )
        console.print("\n[bold]Translation:[/bold]")
        console.print(f"[cyan]{result.final_translation}[/cyan]")

        # Save output if requested
        if output:
            output_path = Path(output)
            output_path.write_text(result.final_translation, encoding="utf-8")
            console.print(f"\n[dim]Translation saved to: {output}[/dim]")

        # Show iteration history if verbose
        if verbose:
            console.print("\n[bold]Iteration History:[/bold]")
            for i, report in enumerate(result.qa_reports):
                console.print(
                    f"  Iteration {i + 1}: MQM {report.mqm_score:.2f}, {len(report.errors)} errors"
                )

    except Exception as e:
        raise RuntimeError(f"Translation refinement failed: {e}") from e


@translate_app.command(name="translate")
def translate(
    text: str = typer.Option(..., "--text", help="Text to translate (or file path with @)"),
    source_lang: str = typer.Option(..., "--source-lang", help="Source language code"),
    target_lang: str = typer.Option(..., "--target-lang", help="Target language code"),
    threshold: float = typer.Option(
        95.0, "--threshold", help="Quality threshold for auto-refinement"
    ),
    max_iterations: int = typer.Option(3, "--max-iterations", help="Maximum refinement iterations"),
    output: str | None = typer.Option(None, "--output", "-o", help="Output file path"),
    provider: str | None = typer.Option(
        None, "--provider", help="LLM provider (openai or anthropic)"
    ),
    verbose: bool = typer.Option(False, "--verbose", help="Verbose output"),
) -> None:
    """
    Translate text with automatic quality checking.

    Uses TEaR (Translate-Estimate-Refine) loop to generate and
    iteratively improve translation until quality threshold is met.

    Example:
        kttc translate --text "Hello world" \\
                      --source-lang en --target-lang es --threshold 95
    """
    # Run async function
    try:
        asyncio.run(
            _translate_async(
                text,
                source_lang,
                target_lang,
                threshold,
                max_iterations,
                output,
                provider,
                verbose,
            )
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠ Interrupted by user[/yellow]")
        raise typer.Exit(code=130) from None
    except Exception as e:
        console.print(f"\n[red]✗ Error: {e}[/red]")
        if verbose:
            console.print_exception()
        raise typer.Exit(code=1) from e
