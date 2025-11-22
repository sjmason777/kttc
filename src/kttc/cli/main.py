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

"""Main CLI application for KTTC.

KTTC - Knowledge Translation Transmutation Core
Transforming translations into gold-standard quality.
"""

from __future__ import annotations

import asyncio
import json
import logging
import warnings
from pathlib import Path
from typing import Any

import typer
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from kttc import __version__
from kttc.agents import AgentOrchestrator
from kttc.cli.commands.benchmark import run_benchmark
from kttc.cli.commands.compare import run_compare as _run_compare_command
from kttc.cli.commands.glossary import glossary_app
from kttc.cli.commands.terminology import terminology_app
from kttc.cli.formatters import ConsoleFormatter, HTMLFormatter, MarkdownFormatter
from kttc.cli.ui import (
    console,
    create_step_progress,
    get_nlp_insights,
    print_header,
    print_info,
    print_startup_info,
    print_translation_preview,
)
from kttc.core import BatchFileParser, BatchGrouper, QAReport, TranslationTask
from kttc.i18n import get_supported_languages, set_language
from kttc.llm import AnthropicProvider, BaseLLMProvider, OpenAIProvider
from kttc.utils.config import get_settings

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

# Add glossary subcommands
app.add_typer(glossary_app, name="glossary")

# Add terminology subcommands
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


def _detect_check_mode(
    source: str, translations: list[str] | None
) -> tuple[str, dict[str, str | list[str]]]:
    """Detect which mode to run check command in.

    Returns:
        tuple: (mode, params) where mode is 'single', 'compare', or 'batch'
               and params contains mode-specific parameters
    """
    source_path = Path(source)

    # Check if source is CSV/JSON/JSONL file (batch mode)
    if source_path.is_file() and source_path.suffix.lower() in [".csv", ".json", ".jsonl"]:
        return "batch_file", {"file_path": source}

    # Check if source is directory (batch mode)
    if source_path.is_dir():
        if not translations or len(translations) != 1:
            raise ValueError(
                "Directory mode requires exactly one translation directory. "
                "Usage: kttc check source_dir/ translation_dir/ --source-lang en --target-lang ru"
            )
        trans_path = Path(translations[0])
        if not trans_path.is_dir():
            raise ValueError(
                f"Translation path must be a directory when source is a directory: {translations[0]}"
            )
        return "batch_dir", {"source_dir": source, "translation_dir": translations[0]}

    # Single file mode
    if not source_path.is_file():
        raise FileNotFoundError(f"Source file not found: {source}")

    # Check number of translations
    if not translations or len(translations) == 0:
        raise ValueError("At least one translation file is required")

    if len(translations) == 1:
        return "single", {"source": source, "translation": translations[0]}
    else:
        # Multiple translations - compare mode
        return "compare", {"source": source, "translations": translations}


def _auto_detect_glossary(glossary: str | None) -> str | None:
    """Auto-detect glossary to use.

    Args:
        glossary: User-specified glossary ('auto', 'none', or comma-separated names)

    Returns:
        Glossary string to use, or None
    """
    if glossary == "none":
        return None

    if glossary == "auto":
        # Check if base glossary exists
        base_paths = [
            Path("glossaries/base.json"),  # Project directory
            Path.home() / ".kttc/glossaries/base.json",  # User directory
        ]
        for path in base_paths:
            if path.exists():
                return "base"
        return None

    # User specified glossaries
    return glossary


def _auto_detect_format(output: str | None, format: str | None) -> str:
    """Auto-detect output format.

    Args:
        output: Output file path
        format: User-specified format (overrides auto-detection)

    Returns:
        Format string: 'text', 'json', 'markdown', or 'html'
    """
    if format:
        return format

    if output:
        suffix = Path(output).suffix.lower()
        if suffix == ".json":
            return "json"
        elif suffix in [".md", ".markdown"]:
            return "markdown"
        elif suffix in [".html", ".htm"]:
            return "html"

    return "text"


async def _run_compare_mode(
    source: str,
    translations: list[str],
    source_lang: str | None,
    target_lang: str | None,
    threshold: float,
    provider: str | None,
    glossary: str | None,
    verbose: bool,
    demo: bool = False,
) -> None:
    """Wrapper for compare mode called from check command.

    Args:
        source: Source file path
        translations: List of translation file paths
        source_lang: Source language code
        target_lang: Target language code
        threshold: Quality threshold
        provider: LLM provider
        glossary: Glossary to use
        verbose: Verbose output
        demo: Demo mode
    """
    # Check required languages
    if not source_lang or not target_lang:
        console.print("[red]Error: --source-lang and --target-lang required for compare mode[/red]")
        raise typer.Exit(code=1)

    # Call the existing compare command
    await _run_compare_command(
        source=source,
        translations=translations,
        source_lang=source_lang,
        target_lang=target_lang,
        threshold=threshold,
        provider=provider,
        verbose=verbose,
    )


@app.command()
def check(
    source: str = typer.Argument(..., help="Source text file path, directory, or CSV/JSON file"),
    translations: list[str] = typer.Argument(
        None, help="Translation file path(s) - can specify multiple for comparison"
    ),
    source_lang: str | None = typer.Option(
        None, "--source-lang", help="Source language code (e.g., 'en') - auto-detected from file"
    ),
    target_lang: str | None = typer.Option(
        None, "--target-lang", help="Target language code (e.g., 'es') - auto-detected from file"
    ),
    threshold: float = typer.Option(95.0, "--threshold", help="Minimum MQM score to pass (0-100)"),
    output: str | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Output file path - format auto-detected from extension (.json/.html/.md)",
    ),
    format: str | None = typer.Option(
        None,
        "--format",
        "-f",
        help="Output format (overrides auto-detection): text, json, markdown, or html",
    ),
    provider: str | None = typer.Option(
        None, "--provider", help="LLM provider (openai or anthropic)"
    ),
    auto_select_model: bool = typer.Option(
        False, "--auto-select-model", help="Automatically select best model for language pair"
    ),
    auto_correct: bool = typer.Option(
        False, "--auto-correct", help="Automatically correct detected errors"
    ),
    correction_level: str = typer.Option(
        "light",
        "--correction-level",
        help="Correction level: light (critical/major) or full (all errors)",
    ),
    smart_routing: bool = typer.Option(
        True,
        "--smart-routing/--no-smart-routing",
        help="Enable complexity-based smart routing to optimize costs (default: enabled)",
    ),
    show_routing_info: bool = typer.Option(
        False, "--show-routing-info", help="Display routing decision and complexity score"
    ),
    simple_threshold: float = typer.Option(
        0.3, "--simple-threshold", help="Complexity threshold for simple texts (0.0-1.0)"
    ),
    complex_threshold: float = typer.Option(
        0.7, "--complex-threshold", help="Complexity threshold for complex texts (0.0-1.0)"
    ),
    glossary: str | None = typer.Option(
        "auto",
        "--glossary",
        "-g",
        help="Glossaries to use (comma-separated, e.g., 'base,medical'), 'auto' to auto-detect, or 'none' to disable",
    ),
    reference: str | None = typer.Option(
        None,
        "--reference",
        "-r",
        help="Reference translation file path for metric calculation (optional)",
    ),
    verbose: bool = typer.Option(False, "--verbose", help="Verbose output"),
    demo: bool = typer.Option(
        False, "--demo", help="Demo mode (no API calls, simulated responses)"
    ),
) -> None:
    """
    Smart translation quality checker with auto-detection.

    🎯 AUTO-DETECTS MODE:
    - Single file: Simple quality check
    - Multiple translations: Automatic comparison
    - Directory/CSV/JSON: Batch processing

    🚀 SMART DEFAULTS (can disable):
    - Smart routing enabled (--no-smart-routing to disable)
    - Auto-detects glossary 'base' if exists
    - Auto-detects output format from file extension

    📝 EXAMPLES:

    # Single check
    kttc check source.txt translation.txt --source-lang en --target-lang ru

    # Compare multiple translations (auto-detected)
    kttc check source.txt trans1.txt trans2.txt trans3.txt \\
               --source-lang en --target-lang ru

    # Batch process directory (auto-detected)
    kttc check source_dir/ translation_dir/ --source-lang en --target-lang ru

    # Batch process CSV (auto-detected, langs from file)
    kttc check translations.csv

    # HTML report (auto-detected from extension)
    kttc check source.txt trans.txt --source-lang en --target-lang ru \\
               --output report.html

    # Disable smart features
    kttc check source.txt trans.txt --source-lang en --target-lang ru \\
               --no-smart-routing --glossary none
    """
    # Check models with loader
    from kttc.cli.ui import check_models_with_loader

    if not check_models_with_loader():
        raise typer.Exit(code=1)

    try:
        # 🎯 Auto-detect mode
        mode, mode_params = _detect_check_mode(source, translations)

        # 🚀 Auto-detect glossary
        detected_glossary = _auto_detect_glossary(glossary)

        # 📄 Auto-detect output format
        detected_format = _auto_detect_format(output, format)

        # Show auto-detection info if verbose
        if verbose:
            console.print(f"[dim]🎯 Mode: {mode}[/dim]")
            if detected_glossary:
                console.print(f"[dim]📚 Glossary: {detected_glossary}[/dim]")
            if smart_routing:
                console.print("[dim]🧠 Smart routing: enabled[/dim]")
            console.print(f"[dim]📄 Output format: {detected_format}[/dim]\n")

        # Route to appropriate handler
        if mode == "single":
            # Single file check
            # Check required languages
            if not source_lang or not target_lang:
                console.print("[red]Error: --source-lang and --target-lang are required[/red]")
                raise typer.Exit(code=1)

            asyncio.run(
                _check_async(
                    str(mode_params["source"]),
                    str(mode_params["translation"]),
                    source_lang,
                    target_lang,
                    threshold,
                    output,
                    detected_format,
                    provider,
                    auto_select_model,
                    auto_correct,
                    correction_level,
                    smart_routing,
                    show_routing_info,
                    simple_threshold,
                    complex_threshold,
                    detected_glossary,
                    reference,
                    verbose,
                    demo,
                )
            )
        elif mode == "compare":
            # Compare mode - delegate to compare command logic
            translations_list = mode_params["translations"]
            assert isinstance(
                translations_list, list
            ), "translations must be a list in compare mode"

            asyncio.run(
                _run_compare_mode(
                    str(mode_params["source"]),
                    translations_list,
                    source_lang,
                    target_lang,
                    threshold,
                    provider,
                    detected_glossary,
                    verbose,
                    demo,
                )
            )
        elif mode == "batch_file":
            # Batch file mode
            asyncio.run(
                _batch_from_file_async(
                    str(mode_params["file_path"]),
                    threshold,
                    output or "report.json",
                    4,  # parallel workers
                    None,  # batch_size
                    provider,
                    smart_routing,
                    False,  # show_cost_savings
                    True,  # show_progress
                    detected_glossary,
                    verbose,
                    demo,
                )
            )
        elif mode == "batch_dir":
            # Batch directory mode
            if not source_lang or not target_lang:
                console.print(
                    "[red]Error: --source-lang and --target-lang required for directory mode[/red]"
                )
                raise typer.Exit(code=1)

            asyncio.run(
                _batch_async(
                    str(mode_params["source_dir"]),
                    str(mode_params["translation_dir"]),
                    source_lang,
                    target_lang,
                    threshold,
                    output or "report.json",
                    4,  # parallel workers
                    provider,
                    verbose,
                    demo,
                )
            )

    except KeyboardInterrupt:
        console.print("\n[yellow]⚠ Interrupted by user[/yellow]")
        raise typer.Exit(code=130)
    except typer.Exit:
        # Re-raise Exit without catching it (clean exit)
        raise
    except Exception as e:
        console.print(f"\n[red]✗ Error: {e}[/red]")
        if verbose:
            console.print_exception()
        raise typer.Exit(code=1)


def _load_translation_files(source: str, translation: str, verbose: bool) -> tuple[str, str]:
    """Load source and translation text files.

    Args:
        source: Path to source file
        translation: Path to translation file
        verbose: Whether to show verbose output

    Returns:
        Tuple of (source_text, translation_text)

    Raises:
        FileNotFoundError: If files don't exist
    """
    source_path = Path(source)
    translation_path = Path(translation)

    if not source_path.exists():
        raise FileNotFoundError(f"Source file not found: {source}")
    if not translation_path.exists():
        raise FileNotFoundError(f"Translation file not found: {translation}")

    source_text = source_path.read_text(encoding="utf-8")
    translation_text = translation_path.read_text(encoding="utf-8")

    if verbose:
        console.print(f"[dim]Loaded {len(source_text)} chars from source[/dim]")
        console.print(f"[dim]Loaded {len(translation_text)} chars from translation[/dim]\n")

    return source_text, translation_text


def _create_translation_task(
    source_text: str,
    translation_text: str,
    source_lang: str,
    target_lang: str,
    verbose: bool,
) -> TranslationTask:
    """Create translation task from loaded texts.

    Args:
        source_text: Source text content
        translation_text: Translation text content
        source_lang: Source language code
        target_lang: Target language code
        verbose: Whether to show verbose output

    Returns:
        Configured TranslationTask instance
    """
    task = TranslationTask(
        source_text=source_text,
        translation=translation_text,
        source_lang=source_lang,
        target_lang=target_lang,
    )
    if verbose:
        console.print(f"[dim]Created task with {task.word_count} words[/dim]\n")
    return task


async def _check_async(
    source: str,
    translation: str,
    source_lang: str,
    target_lang: str,
    threshold: float,
    output: str | None,
    format: str,
    provider: str | None,
    auto_select_model: bool,
    auto_correct: bool,
    correction_level: str,
    smart_routing: bool,
    show_routing_info: bool,
    simple_threshold: float,
    complex_threshold: float,
    glossary: str | None,
    reference: str | None,
    verbose: bool,
    demo: bool = False,
) -> None:
    """Async implementation of check command."""
    from kttc.core import GlossaryManager
    from kttc.core.correction import AutoCorrector
    from kttc.llm import ComplexityRouter

    # Configure logging based on verbose flag
    if verbose:
        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",  # Simple format for CLI output
            force=True,  # Override any existing configuration
        )
    else:
        logging.basicConfig(level=logging.WARNING, force=True)

    # Load settings
    settings = get_settings()

    # Display header (verbose mode only)
    if verbose:
        print_header(
            "✓ Translation Quality Check",
            "Evaluating translation quality with multi-agent AI system",
        )

        # Prepare configuration info
        config_info = {
            "Source File": source,
            "Translation File": translation,
            "Languages": f"{source_lang} → {target_lang}",
            "Quality Threshold": f"{threshold}",
        }
        if auto_select_model:
            config_info["Model Selection"] = "Automatic (intelligent)"
        if auto_correct:
            config_info["Auto-Correct"] = f"Enabled ({correction_level})"

        print_startup_info(config_info)

    # Load files
    try:
        source_text, translation_text = _load_translation_files(source, translation, verbose)
    except Exception as e:
        raise RuntimeError(f"Failed to load files: {e}") from e

    # Create translation task
    try:
        task = _create_translation_task(
            source_text, translation_text, source_lang, target_lang, verbose
        )
    except Exception as e:
        raise RuntimeError(f"Failed to create translation task: {e}") from e

    # Load glossaries if specified
    if glossary:
        try:
            manager = GlossaryManager()
            glossary_names = [g.strip() for g in glossary.split(",")]
            manager.load_multiple(glossary_names)

            # Find relevant terms in source text
            terms = manager.find_in_text(source_text, source_lang, target_lang)

            # Add glossary terms to task context
            task.context = task.context or {}
            task.context["glossary_terms"] = [
                {"source": t.source, "target": t.target, "do_not_translate": t.do_not_translate}
                for t in terms
            ]

            if verbose:
                console.print(
                    f"[dim]Loaded {len(glossary_names)} glossaries, found {len(terms)} relevant terms[/dim]"
                )
        except Exception as e:
            console.print(f"[yellow]⚠ Warning: Failed to load glossaries: {e}[/yellow]")

    # Smart routing: select model based on complexity
    selected_model = None
    complexity_score = None

    if smart_routing:
        try:
            router = ComplexityRouter()
            # Get available providers to ensure smart routing only selects from configured providers
            available_providers = _get_available_providers(settings)

            # Note: ComplexityRouter doesn't have configurable thresholds yet,
            # but the parameters are accepted for future use
            selected_model, complexity_score = router.route(
                source_text,
                source_lang,
                target_lang,
                domain=task.context.get("domain") if task.context else None,
                available_providers=available_providers,
            )

            if show_routing_info:
                console.print("[dim]─ Complexity Analysis ─[/dim]")
                console.print(f"[dim]Overall: {complexity_score.overall:.2f}[/dim]")
                console.print(
                    f"[dim]  Sentence length: {complexity_score.sentence_length:.2f}[/dim]"
                )
                console.print(f"[dim]  Rare words: {complexity_score.rare_words:.2f}[/dim]")
                console.print(f"[dim]  Syntactic: {complexity_score.syntactic:.2f}[/dim]")
                console.print(
                    f"[dim]  Domain-specific: {complexity_score.domain_specific:.2f}[/dim]"
                )
                console.print(f"[dim]Selected model: {selected_model}[/dim]\n")
        except Exception as e:
            console.print(f"[yellow]⚠ Warning: Smart routing failed, using default: {e}[/yellow]")

    # Setup LLM provider with intelligent model selection
    try:
        # If smart routing selected a model AND user didn't explicitly specify provider
        # then map model to provider. Otherwise respect user's choice.
        if smart_routing and selected_model and provider is None:
            # Map model to provider
            if "gpt" in selected_model.lower() and "yandex" not in selected_model.lower():
                provider = "openai"
            elif "claude" in selected_model.lower():
                provider = "anthropic"
            elif "yandex" in selected_model.lower():
                provider = "yandex"

        llm_provider = _setup_llm_provider(
            provider, settings, verbose, task=task, auto_select_model=auto_select_model, demo=demo
        )
    except Exception as e:
        raise RuntimeError(f"Failed to setup LLM provider: {e}") from e

    # Show translation preview (verbose mode only)
    if verbose:
        print_translation_preview(source_text, translation_text)

    # Run evaluation
    nlp_insights = None
    api_errors = []

    # Step 1: NLP Analysis (if available)
    from kttc.helpers import get_helper_for_language

    helper = get_helper_for_language(task.target_lang)
    if helper and helper.is_available():
        if verbose:
            with create_step_progress() as progress:
                progress.add_task(
                    "[cyan]Step 1/3: Analyzing linguistic features...[/cyan]", total=None
                )
                try:
                    nlp_insights = get_nlp_insights(task, helper)
                except Exception as e:
                    if verbose:
                        api_errors.append(f"NLP analysis failed: {str(e)}")
            console.print("[green]✓[/green] Step 1/3: Linguistic analysis complete")
        else:
            # Compact mode: no progress output
            try:
                nlp_insights = get_nlp_insights(task, helper)
            except Exception:
                pass
    elif verbose:
        console.print(
            "[dim]⊘ Step 1/3: Linguistic analysis (not available for this language)[/dim]"
        )

    # Style analysis (optional, runs for source text to detect literary patterns)
    style_profile = None
    try:
        from kttc.style import StyleFingerprint

        style_analyzer = StyleFingerprint()
        style_profile = style_analyzer.analyze(source_text, lang=source_lang)
        if verbose and style_profile.is_literary:
            console.print(
                f"[magenta]📚 Literary text detected: "
                f"{style_profile.detected_pattern.value.replace('_', ' ').title()}[/magenta]"
            )
    except Exception:
        # Style analysis is optional, don't fail the check
        pass

    # Step 2: Quality Evaluation
    if verbose:
        with create_step_progress() as progress:
            progress.add_task(
                "[cyan]Step 2/3: Running multi-agent quality assessment...[/cyan]", total=None
            )
            try:
                orchestrator = AgentOrchestrator(
                    llm_provider,
                    quality_threshold=threshold,
                    agent_temperature=settings.default_temperature,
                    agent_max_tokens=settings.default_max_tokens,
                )
                report = await orchestrator.evaluate(task)
            except Exception as e:
                console.print("[red]✗[/red] Step 2/3: Quality assessment failed")
                api_errors.append(f"Quality assessment failed: {str(e)}")
                raise RuntimeError(f"Evaluation failed: {e}") from e
        console.print("[green]✓[/green] Step 2/3: Quality assessment complete")
        console.print("[green]✓[/green] Step 3/3: Report ready")
        console.print()
    else:
        # Compact mode: show spinner during evaluation
        with create_step_progress() as progress:
            progress.add_task(
                "[cyan]Evaluating translation quality...[/cyan]",
                total=None,
            )
            try:
                orchestrator = AgentOrchestrator(
                    llm_provider,
                    quality_threshold=threshold,
                    agent_temperature=settings.default_temperature,
                    agent_max_tokens=settings.default_max_tokens,
                )
                report = await orchestrator.evaluate(task)
            except Exception as e:
                api_errors.append(f"Quality assessment failed: {str(e)}")
                raise RuntimeError(f"Evaluation failed: {e}") from e

    # Calculate lightweight metrics and rule-based errors
    from kttc.evaluation import ErrorDetector, LightweightMetrics

    metrics_calculator = LightweightMetrics()
    error_detector = ErrorDetector()

    # Load reference translation if provided
    reference_text = None
    if reference:
        try:
            reference_path = Path(reference)
            if not reference_path.exists():
                console.print(f"[yellow]⚠ Reference file not found: {reference}[/yellow]")
            else:
                reference_text = reference_path.read_text(encoding="utf-8").strip()
                if verbose:
                    console.print(f"[dim]Loaded {len(reference_text)} chars from reference[/dim]")
        except Exception as e:
            console.print(f"[yellow]⚠ Failed to load reference: {e}[/yellow]")

    try:
        # Calculate lightweight metrics
        # If reference is provided, use it; otherwise use translation itself (baseline)
        reference_for_metrics = reference_text if reference_text else translation_text

        lightweight_scores = metrics_calculator.evaluate(
            translation=translation_text,
            reference=reference_for_metrics,
            source=source_text,
        )

        # Detect rule-based errors (source vs translation)
        rule_based_errors = error_detector.detect_all_errors(
            source=source_text, translation=translation_text
        )
        rule_based_score = error_detector.calculate_rule_based_score(rule_based_errors)

        # Show warning if no reference provided (verbose mode only)
        if not reference_text and verbose:
            console.print(
                "[dim]ℹ️  No reference translation provided. "
                "Metrics show baseline (self-comparison). "
                "Use --reference for meaningful scores.[/dim]\n"
            )
    except Exception as e:
        if verbose:
            console.print(f"[dim]⚠ Lightweight metrics calculation failed: {e}[/dim]")
        lightweight_scores = None
        rule_based_errors = None
        rule_based_score = None

    # Display results using compact formatter
    ConsoleFormatter.print_check_result(
        report=report,
        source_lang=source_lang,
        target_lang=target_lang,
        lightweight_scores=lightweight_scores,
        rule_based_score=rule_based_score,
        rule_based_errors=rule_based_errors,
        nlp_insights=nlp_insights,
        style_profile=style_profile,
        verbose=verbose,
    )

    # Auto-correct if requested and errors found
    if auto_correct and len(report.errors) > 0:
        console.print(f"\n[yellow]🔧 Applying auto-correction ({correction_level})...[/yellow]")
        try:
            corrector = AutoCorrector(llm_provider)
            corrected_text = await corrector.auto_correct(
                task=task,
                errors=report.errors,
                correction_level=correction_level,
                temperature=settings.default_temperature,
            )

            # Save corrected version
            corrected_path = Path(translation).parent / f"{Path(translation).stem}_corrected.txt"
            corrected_path.write_text(corrected_text, encoding="utf-8")

            console.print(f"[green]✓ Corrected translation saved to: {corrected_path}[/green]")

            # Re-evaluate corrected translation
            if verbose:
                console.print("\n[dim]Re-evaluating corrected translation...[/dim]")
            corrected_task = _create_translation_task(
                source_text, corrected_text, source_lang, target_lang, verbose=False
            )
            corrected_report = await orchestrator.evaluate(corrected_task)

            console.print("\n[bold]Corrected Translation Quality:[/bold]")
            console.print(f"MQM Score: [cyan]{corrected_report.mqm_score:.2f}[/cyan]")
            console.print(
                f"Errors: {len(report.errors)} → [cyan]{len(corrected_report.errors)}[/cyan]"
            )
            console.print(
                f"Status: {report.status} → "
                f"[{'green' if corrected_report.status == 'pass' else 'red'}]"
                f"{corrected_report.status}[/{'green' if corrected_report.status == 'pass' else 'red'}]"
            )

        except Exception as e:
            console.print(f"[yellow]⚠ Auto-correction failed: {e}[/yellow]")
            if verbose:
                console.print_exception()

    # Save output if requested
    if output:
        _save_report(report, output, format)
        console.print(f"\n[dim]Report saved to: {output}[/dim]")

    # Exit with appropriate code
    if report.status == "fail":
        console.print()
        print_info(f"Translation quality below threshold ({threshold}). Exiting with error code.")
        raise typer.Exit(code=1)


def _scan_batch_directories(
    source_dir: str, translation_dir: str, verbose: bool
) -> list[tuple[Path, Path]]:
    """Scan directories and find matching source-translation file pairs.

    Args:
        source_dir: Path to source files directory
        translation_dir: Path to translation files directory
        verbose: Whether to show verbose output

    Returns:
        List of (source_file, translation_file) path pairs

    Raises:
        FileNotFoundError: If directories don't exist
        ValueError: If no matching pairs found
    """
    source_path = Path(source_dir)
    translation_path = Path(translation_dir)

    if not source_path.exists() or not source_path.is_dir():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")
    if not translation_path.exists() or not translation_path.is_dir():
        raise FileNotFoundError(f"Translation directory not found: {translation_dir}")

    # Find matching files
    source_files = sorted(source_path.glob("*.txt"))
    if not source_files:
        raise ValueError(f"No .txt files found in source directory: {source_dir}")

    # Match source and translation files
    file_pairs: list[tuple[Path, Path]] = []
    for source_file in source_files:
        translation_file = translation_path / source_file.name
        if translation_file.exists():
            file_pairs.append((source_file, translation_file))
        elif verbose:
            console.print(
                f"[yellow]⚠ Skipping {source_file.name}: no matching translation[/yellow]"
            )

    if not file_pairs:
        raise ValueError("No matching source-translation file pairs found")

    return file_pairs


def _get_available_providers(settings: Any) -> list[str]:
    """Get list of available providers (those with configured API keys).

    Args:
        settings: Application settings

    Returns:
        List of available provider names
    """
    available = []

    # Check OpenAI
    try:
        settings.get_llm_provider_key("openai")
        available.append("openai")
    except (ValueError, AttributeError):
        pass

    # Check Anthropic
    try:
        settings.get_llm_provider_key("anthropic")
        available.append("anthropic")
    except (ValueError, AttributeError):
        pass

    # Check GigaChat
    try:
        settings.get_llm_provider_credentials("gigachat")
        available.append("gigachat")
    except (ValueError, AttributeError):
        pass

    # Check Yandex
    try:
        settings.get_llm_provider_credentials("yandex")
        available.append("yandex")
    except (ValueError, AttributeError):
        pass

    return available


def _setup_llm_provider(
    provider: str | None,
    settings: Any,
    verbose: bool,
    task: TranslationTask | None = None,
    auto_select_model: bool = False,
    demo: bool = False,
) -> BaseLLMProvider:
    """Setup and configure LLM provider with optional intelligent model selection.

    Args:
        provider: Provider name (openai/anthropic/gigachat/demo) or None for default
        settings: Application settings
        verbose: Whether to show verbose output
        task: Optional translation task for intelligent model selection
        auto_select_model: Whether to use ModelSelector for optimal model
        demo: Whether to use demo mode (no API calls)

    Returns:
        Configured LLM provider instance

    Raises:
        ValueError: If provider is unknown
        RuntimeError: If provider setup fails
    """
    from kttc.cli.demo import DemoLLMProvider
    from kttc.llm import GigaChatProvider
    from kttc.llm.model_selector import ModelSelector

    # Use demo provider if demo mode enabled
    if demo:
        if verbose:
            console.print(
                "[yellow]🎭 Demo mode: Using simulated responses (no API calls)[/yellow]\n"
            )
        return DemoLLMProvider(model="demo-model")

    # If no provider specified, select from available providers
    if provider is None:
        available_providers = _get_available_providers(settings)
        if not available_providers:
            raise RuntimeError(
                "No LLM providers configured. Please set at least one of:\n"
                "  - KTTC_OPENAI_API_KEY\n"
                "  - KTTC_ANTHROPIC_API_KEY\n"
                "  - KTTC_GIGACHAT_CLIENT_ID and KTTC_GIGACHAT_CLIENT_SECRET"
            )
        # Use default or first available
        provider_name = (
            settings.default_llm_provider
            if settings.default_llm_provider in available_providers
            else available_providers[0]
        )
    else:
        provider_name = provider

    # Intelligent model selection if enabled and task provided
    model = settings.default_model
    if auto_select_model and task is not None:
        selector = ModelSelector()
        recommended_model = selector.select_best_model(
            source_lang=task.source_lang,
            target_lang=task.target_lang,
            domain=task.context.get("domain") if task.context else None,
            task_type="qa",
            optimize_for="quality",
        )
        model = recommended_model
        if verbose:
            console.print(f"[dim]🤖 Auto-selected model: {model}[/dim]")

    # Setup provider based on type
    llm_provider: BaseLLMProvider
    if provider_name == "openai":
        api_key = settings.get_llm_provider_key(provider_name)
        llm_provider = OpenAIProvider(api_key=api_key, model=model)
    elif provider_name == "anthropic":
        api_key = settings.get_llm_provider_key(provider_name)
        llm_provider = AnthropicProvider(api_key=api_key, model=model)
    elif provider_name == "gigachat":
        # GigaChat uses client_id + client_secret instead of API key
        credentials = settings.get_llm_provider_credentials(provider_name)
        llm_provider = GigaChatProvider(
            client_id=credentials["client_id"],
            client_secret=credentials["client_secret"],
            model=model,
        )
    else:
        raise ValueError(
            f"Unknown provider: {provider_name}. Supported: openai, anthropic, gigachat"
        )

    if verbose:
        console.print(f"[dim]Using {provider_name} provider with model {model}[/dim]\n")

    return llm_provider


async def _process_batch_files(
    file_pairs: list[tuple[Path, Path]],
    orchestrator: AgentOrchestrator,
    source_lang: str,
    target_lang: str,
    parallel: int,
) -> list[tuple[str, QAReport]]:
    """Process file pairs in parallel and collect results.

    Args:
        file_pairs: List of (source, translation) file path pairs
        orchestrator: Agent orchestrator for evaluation
        source_lang: Source language code
        target_lang: Target language code
        parallel: Number of parallel workers

    Returns:
        List of (filename, report) tuples
    """
    results: list[tuple[str, QAReport]] = []
    semaphore = asyncio.Semaphore(parallel)

    async def process_file(source_file: Path, translation_file: Path) -> tuple[str, QAReport]:
        """Process a single file pair."""
        async with semaphore:
            # Load files
            source_text = source_file.read_text(encoding="utf-8")
            translation_text = translation_file.read_text(encoding="utf-8")

            # Create task
            task = TranslationTask(
                source_text=source_text,
                translation=translation_text,
                source_lang=source_lang,
                target_lang=target_lang,
            )

            # Evaluate
            report = await orchestrator.evaluate(task)
            return source_file.name, report

    # Process all files with progress tracking
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task_id = progress.add_task("Processing files...", total=len(file_pairs))

        tasks = [process_file(src, trg) for src, trg in file_pairs]
        for coro in asyncio.as_completed(tasks):
            try:
                filename, report = await coro
                results.append((filename, report))
                status_icon = "✓" if report.status == "pass" else "✗"
                status_color = "green" if report.status == "pass" else "red"
                progress.console.print(
                    f"  [{status_color}]{status_icon}[/{status_color}] "
                    f"{filename}: {report.mqm_score:.2f}"
                )
                progress.advance(task_id)
            except Exception as e:
                progress.console.print(f"  [red]✗ Error processing file: {e}[/red]")
                progress.advance(task_id)

    return results


async def _batch_from_file_async(
    file_path: str,
    threshold: float,
    output: str,
    parallel: int,
    batch_size: int | None,
    provider: str | None,
    smart_routing: bool,
    show_cost_savings: bool,
    show_progress: bool,
    glossary: str | None,
    verbose: bool,
    demo: bool = False,
) -> None:
    """Async implementation of batch command for file-based input.

    Args:
        file_path: Path to batch file (CSV, JSON, or JSONL)
        threshold: Quality threshold for pass/fail
        output: Output report file path
        parallel: Number of parallel workers
        batch_size: Optional batch size for grouping
        provider: LLM provider name
        smart_routing: Enable complexity-based routing
        show_cost_savings: Display cost savings
        show_progress: Show progress bar
        glossary: Comma-separated glossary names
        verbose: Verbose output flag
        demo: Demo mode flag
    """
    from kttc.core import GlossaryManager

    # Load settings
    settings = get_settings()

    # Display header
    print_header(
        "Batch Translation Quality Check (File Mode)",
        "Process translations from CSV/JSON/JSONL file",
    )

    # Parse batch file
    try:
        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            raise FileNotFoundError(f"Batch file not found: {file_path}")

        console.print(f"[cyan]Parsing batch file:[/cyan] {file_path}")
        translations = BatchFileParser.parse(file_path_obj)
        console.print(f"[green]✓[/green] Loaded [cyan]{len(translations)}[/cyan] translations\n")

    except Exception as e:
        raise RuntimeError(f"Failed to parse batch file: {e}") from e

    # Group by language pair (for display)
    groups = BatchGrouper.group_by_language_pair(translations)
    console.print("[bold]Language Pairs:[/bold]")
    for (src, tgt), group_translations in groups.items():
        console.print(f"  • {src} → {tgt}: [cyan]{len(group_translations)}[/cyan] translations")
    console.print()

    # Load glossaries if specified
    glossary_manager = None
    if glossary:
        try:
            glossary_manager = GlossaryManager()
            glossary_names = [g.strip() for g in glossary.split(",")]
            glossary_manager.load_multiple(glossary_names)
            console.print(f"[green]✓[/green] Loaded {len(glossary_names)} glossaries\n")
        except Exception as e:
            console.print(f"[yellow]⚠ Warning: Failed to load glossaries: {e}[/yellow]\n")
            glossary_manager = None

    # Apply glossary terms to translations
    if glossary_manager:
        for trans in translations:
            terms = glossary_manager.find_in_text(
                trans.source_text, trans.source_lang, trans.target_lang
            )
            if terms:
                trans.context = trans.context or {}
                trans.context["glossary_terms"] = [
                    {"source": t.source, "target": t.target, "do_not_translate": t.do_not_translate}
                    for t in terms
                ]

    # Initialize smart routing if enabled (for future use)
    # Note: Smart routing stats tracking not yet implemented
    if smart_routing:
        console.print("[cyan]Smart routing enabled[/cyan] - complexity-based model selection\n")

    # Display configuration
    config_info = {
        "Input File": file_path,
        "Total Translations": f"{len(translations)}",
        "Language Pairs": f"{len(groups)}",
        "Quality Threshold": f"{threshold}",
        "Parallel Workers": f"{parallel}",
    }
    if batch_size:
        config_info["Batch Size"] = f"{batch_size}"
    if smart_routing:
        config_info["Smart Routing"] = "Enabled"
    if glossary:
        config_info["Glossaries"] = glossary
    print_startup_info(config_info)

    # Setup LLM provider
    try:
        llm_provider = _setup_llm_provider(provider, settings, verbose, demo=demo)
    except Exception as e:
        raise RuntimeError(f"Failed to setup LLM provider: {e}") from e

    # Create orchestrator
    orchestrator = AgentOrchestrator(
        llm_provider,
        quality_threshold=threshold,
        agent_temperature=settings.default_temperature,
        agent_max_tokens=settings.default_max_tokens,
    )

    # Process translations in parallel
    console.print("[yellow]⏳ Processing translations...[/yellow]\n")
    results = await _process_batch_translations(
        translations, orchestrator, parallel, verbose, show_progress=show_progress
    )

    # Generate aggregated report
    console.print()
    _display_batch_summary(results, threshold)

    # Save detailed report
    _save_batch_report(results, output, threshold)
    console.print(f"\n[dim]Detailed report saved to: {output}[/dim]")

    # Exit with appropriate code
    failed_count = sum(1 for _, report in results if report.status == "fail")
    if failed_count > 0:
        raise typer.Exit(code=1)


async def _process_batch_translations(
    translations: list[Any],
    orchestrator: AgentOrchestrator,
    parallel: int,
    verbose: bool = False,
    show_progress: bool = False,
) -> list[tuple[str, QAReport]]:
    """Process batch translations in parallel.

    Args:
        translations: List of BatchTranslation objects
        orchestrator: Agent orchestrator for evaluation
        parallel: Number of parallel workers
        verbose: Verbose output flag
        show_progress: Show progress bar

    Returns:
        List of (identifier, report) tuples
    """
    from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

    results: list[tuple[str, QAReport]] = []
    semaphore = asyncio.Semaphore(parallel)

    # Create progress bar if requested
    progress = None
    task_id = None
    if show_progress:
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("•"),
            TextColumn("{task.completed}/{task.total} translations"),
            TimeElapsedColumn(),
            console=console,
        )
        task_id = progress.add_task("[cyan]Processing translations...", total=len(translations))

    async def process_translation(idx: int, batch_translation: Any) -> tuple[str, QAReport]:
        """Process a single translation."""
        async with semaphore:
            # Convert to TranslationTask
            task = batch_translation.to_task()

            # Evaluate
            report = await orchestrator.evaluate(task)

            # Update progress
            if progress and task_id is not None:
                progress.update(task_id, advance=1)

            # Create identifier
            identifier = f"#{idx+1}"
            if batch_translation.metadata:
                if "file" in batch_translation.metadata:
                    identifier = f"{Path(batch_translation.metadata['file']).name}:#{idx+1}"

            return identifier, report

    # Process all translations with progress tracking
    if show_progress and progress:
        with progress:
            tasks = [process_translation(idx, t) for idx, t in enumerate(translations)]
            for coro in asyncio.as_completed(tasks):
                try:
                    identifier, report = await coro
                    results.append((identifier, report))
                    status_icon = "✓" if report.status == "pass" else "✗"
                    status_color = "green" if report.status == "pass" else "red"
                    if not verbose:
                        # Show concise status with progress bar
                        progress.console.print(
                            f"  [{status_color}]{status_icon}[/{status_color}] "
                            f"{identifier}: {report.mqm_score:.2f}"
                        )
                except Exception as e:
                    progress.console.print(f"  [red]✗ Error processing translation: {e}[/red]")
                    if verbose:
                        import traceback

                        traceback.print_exc()
    else:
        # No progress bar - simple processing
        tasks = [process_translation(idx, t) for idx, t in enumerate(translations)]
        for coro in asyncio.as_completed(tasks):
            try:
                identifier, report = await coro
                results.append((identifier, report))
                status_icon = "✓" if report.status == "pass" else "✗"
                status_color = "green" if report.status == "pass" else "red"
                console.print(
                    f"  [{status_color}]{status_icon}[/{status_color}] "
                    f"{identifier}: {report.mqm_score:.2f}"
                )
            except Exception as e:
                console.print(f"  [red]✗ Error processing translation: {e}[/red]")
                if verbose:
                    import traceback

                    traceback.print_exc()

    return results


async def _batch_async(
    source_dir: str,
    translation_dir: str,
    source_lang: str,
    target_lang: str,
    threshold: float,
    output: str,
    parallel: int,
    provider: str | None,
    verbose: bool,
    demo: bool = False,
) -> None:
    """Async implementation of batch command."""
    # Load settings
    settings = get_settings()

    # Display header
    print_header(
        "Batch Translation Quality Check", "Process multiple translation files in parallel"
    )

    # Display configuration
    config_info = {
        "Source Directory": source_dir,
        "Translation Directory": translation_dir,
        "Languages": f"{source_lang} → {target_lang}",
        "Quality Threshold": f"{threshold}",
        "Parallel Workers": f"{parallel}",
    }
    print_startup_info(config_info)

    # Scan directories
    try:
        file_pairs = _scan_batch_directories(source_dir, translation_dir, verbose)
        console.print(f"Found [cyan]{len(file_pairs)}[/cyan] file pairs to process\n")
    except Exception as e:
        raise RuntimeError(f"Failed to scan directories: {e}") from e

    # Setup LLM provider
    try:
        llm_provider = _setup_llm_provider(provider, settings, verbose, demo=demo)
    except Exception as e:
        raise RuntimeError(f"Failed to setup LLM provider: {e}") from e

    # Create orchestrator
    orchestrator = AgentOrchestrator(
        llm_provider,
        quality_threshold=threshold,
        agent_temperature=settings.default_temperature,
        agent_max_tokens=settings.default_max_tokens,
    )

    # Process files in parallel
    console.print("[yellow]⏳ Processing translations...[/yellow]\n")
    results = await _process_batch_files(
        file_pairs, orchestrator, source_lang, target_lang, parallel
    )

    # Generate aggregated report
    console.print()
    _display_batch_summary(results, threshold)

    # Save detailed report
    _save_batch_report(results, output, threshold)
    console.print(f"\n[dim]Detailed report saved to: {output}[/dim]")

    # Exit with appropriate code
    failed_count = sum(1 for _, report in results if report.status == "fail")
    if failed_count > 0:
        raise typer.Exit(code=1)


def _display_batch_summary(results: list[tuple[str, QAReport]], threshold: float) -> None:
    """Display summary of batch processing results."""
    total = len(results)
    passed = sum(1 for _, report in results if report.status == "pass")
    failed = total - passed

    avg_score = sum(report.mqm_score for _, report in results) / total if total > 0 else 0.0
    total_errors = sum(len(report.errors) for _, report in results)

    # Use compact formatter
    ConsoleFormatter.print_batch_result(
        total=total,
        passed=passed,
        failed=failed,
        avg_score=avg_score,
        total_errors=total_errors,
        verbose=False,  # Batch command doesn't have verbose mode here
    )


def _save_batch_report(results: list[tuple[str, QAReport]], output: str, threshold: float) -> None:
    """Save batch processing report to JSON file."""
    output_path = Path(output)

    # Prepare aggregated data
    data = {
        "summary": {
            "total_files": len(results),
            "passed": sum(1 for _, r in results if r.status == "pass"),
            "failed": sum(1 for _, r in results if r.status == "fail"),
            "average_score": (
                sum(r.mqm_score for _, r in results) / len(results) if results else 0.0
            ),
            "total_errors": sum(len(r.errors) for _, r in results),
            "threshold": threshold,
        },
        "files": [
            {
                "filename": filename,
                "status": report.status,
                "mqm_score": report.mqm_score,
                "error_count": len(report.errors),
                "errors": [
                    {
                        "category": e.category,
                        "subcategory": e.subcategory,
                        "severity": e.severity.value,
                        "location": e.location,
                        "description": e.description,
                    }
                    for e in report.errors
                ],
            }
            for filename, report in results
        ],
    }

    output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _display_report(report: QAReport, format: str, verbose: bool) -> None:
    """Display QA report in specified format."""
    console.print()

    # Status badge
    if report.status == "pass":
        console.print("[bold green]✓ PASS[/bold green]")
    else:
        console.print("[bold red]✗ FAIL[/bold red]")

    # MQM Score
    score_color = (
        "green" if report.mqm_score >= 95 else "yellow" if report.mqm_score >= 85 else "red"
    )
    console.print(
        f"\nMQM Score: [bold {score_color}]{report.mqm_score:.2f}[/bold {score_color}]/100"
    )

    # Error summary
    console.print(f"\nErrors found: [bold]{len(report.errors)}[/bold]")
    if report.errors:
        console.print(
            f"  Critical: {report.critical_error_count} | "
            f"Major: {report.major_error_count} | "
            f"Minor: {report.minor_error_count}"
        )

    # Error details
    if report.errors and verbose:
        console.print("\n[bold]Error Details:[/bold]")
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Category", style="cyan")
        table.add_column("Subcategory")
        table.add_column("Severity", style="yellow")
        table.add_column("Location")
        table.add_column("Description")

        for error in report.errors:
            severity_color = (
                "red"
                if error.severity.value == "critical"
                else "yellow" if error.severity.value == "major" else "dim"
            )
            table.add_row(
                error.category,
                error.subcategory,
                f"[{severity_color}]{error.severity.value}[/{severity_color}]",
                f"{error.location[0]}-{error.location[1]}",
                (
                    error.description[:60] + "..."
                    if len(error.description) > 60
                    else error.description
                ),
            )
        console.print(table)


def _save_report(report: QAReport, output: str, format: str) -> None:
    """Save report to file."""
    output_path = Path(output)

    if format == "json" or output.endswith(".json"):
        # Save as JSON
        data = report.model_dump(mode="json")
        output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    elif format == "markdown" or output.endswith(".md"):
        # Save as Markdown using the new formatter
        MarkdownFormatter.format_report(report, output_path)
    elif format == "html" or output.endswith(".html"):
        # Save as HTML using the new formatter
        HTMLFormatter.format_report(report, output_path)
    else:
        # Default to JSON
        data = report.model_dump(mode="json")
        output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


@app.command()
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
        raise typer.Exit(code=130)
    except Exception as e:
        console.print(f"\n[red]✗ Error: {e}[/red]")
        if verbose:
            console.print_exception()
        raise typer.Exit(code=1)


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
        llm_provider = _setup_llm_provider(provider, settings, verbose)
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


@app.command()
def batch(
    # File-based mode
    file: str | None = typer.Option(None, "--file", "-f", help="Batch file (CSV, JSON, or JSONL)"),
    # Directory-based mode
    source_dir: str | None = typer.Option(None, "--source-dir", help="Directory with source files"),
    translation_dir: str | None = typer.Option(
        None, "--translation-dir", help="Directory with translation files"
    ),
    source_lang: str | None = typer.Option(
        None, "--source-lang", help="Source language code (for directory mode)"
    ),
    target_lang: str | None = typer.Option(
        None, "--target-lang", help="Target language code (for directory mode)"
    ),
    # Common options
    threshold: float = typer.Option(95.0, "--threshold", help="Minimum MQM score to pass"),
    output: str = typer.Option("report.json", "--output", "-o", help="Output report file path"),
    parallel: int = typer.Option(4, "--parallel", help="Number of parallel workers"),
    batch_size: int | None = typer.Option(
        None, "--batch-size", help="Batch size for grouping (file mode only)"
    ),
    provider: str | None = typer.Option(
        None, "--provider", help="LLM provider (openai or anthropic)"
    ),
    smart_routing: bool = typer.Option(
        False, "--smart-routing", help="Enable complexity-based smart routing to optimize costs"
    ),
    show_cost_savings: bool = typer.Option(
        False, "--show-cost-savings", help="Display estimated cost savings from smart routing"
    ),
    show_progress: bool = typer.Option(
        True, "--show-progress/--no-progress", help="Show progress bar during batch processing"
    ),
    glossary: str | None = typer.Option(
        None,
        "--glossary",
        "-g",
        help="Glossaries to use (comma-separated, e.g., 'base,medical')",
    ),
    verbose: bool = typer.Option(False, "--verbose", help="Verbose output"),
    demo: bool = typer.Option(
        False, "--demo", help="Demo mode (no API calls, simulated responses)"
    ),
) -> None:
    """
    Batch process multiple translations.

    Supports two modes:

    1. FILE MODE - Process translations from CSV/JSON/JSONL file:
        kttc batch --file translations.csv --output report.json

    2. DIRECTORY MODE - Process translation files from directories:
        kttc batch --source-dir ./source --translation-dir ./translations \\
                   --source-lang en --target-lang es

    Supported file formats:
    - CSV: source,translation,source_lang,target_lang,domain
    - JSON: Array of translation objects
    - JSONL: One JSON object per line

    Examples:
        # Process CSV file
        kttc batch --file examples/batch/translations.csv

        # Process JSON with custom batch size
        kttc batch --file translations.json --batch-size 50

        # Directory mode (original behavior)
        kttc batch --source-dir ./source --translation-dir ./translations \\
                   --source-lang en --target-lang es
    """
    try:
        # Determine mode
        if file:
            # FILE MODE
            asyncio.run(
                _batch_from_file_async(
                    file,
                    threshold,
                    output,
                    parallel,
                    batch_size,
                    provider,
                    smart_routing,
                    show_cost_savings,
                    show_progress,
                    glossary,
                    verbose,
                    demo,
                )
            )
        elif source_dir and translation_dir:
            # DIRECTORY MODE
            if not source_lang or not target_lang:
                console.print(
                    "[red]Error: --source-lang and --target-lang required for directory mode[/red]"
                )
                raise typer.Exit(code=1)

            asyncio.run(
                _batch_async(
                    source_dir,
                    translation_dir,
                    source_lang,
                    target_lang,
                    threshold,
                    output,
                    parallel,
                    provider,
                    verbose,
                    demo,
                )
            )
        else:
            # Error - must specify either file or directories
            console.print(
                "[red]Error: Must specify either:[/red]\n"
                "  1. --file <path> for file-based batch processing, OR\n"
                "  2. --source-dir and --translation-dir for directory-based processing"
            )
            raise typer.Exit(code=1)

    except KeyboardInterrupt:
        console.print("\n[yellow]⚠ Interrupted by user[/yellow]")
        raise typer.Exit(code=130)
    except Exception as e:
        console.print(f"\n[red]✗ Error: {e}[/red]")
        if verbose:
            console.print_exception()
        raise typer.Exit(code=1)


@app.command()
def report(
    input_file: str = typer.Argument(..., help="Input report file (JSON)"),
    format: str = typer.Option(
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
        console.print(f"\n[bold]Generating {format} report...[/bold]")
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
            ext = ".md" if format == "markdown" else ".html"
            output_path = input_path.with_suffix(ext)

        # Generate report
        if format == "markdown":
            content = _generate_batch_markdown_report(data)
        elif format == "html":
            content = _generate_batch_html_report(data)
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'markdown' or 'html'")

        # Save report
        output_path.write_text(content, encoding="utf-8")

        console.print("[green]✓ Report generated successfully[/green]")
        console.print(f"Output: [cyan]{output_path}[/cyan]")

    except Exception as e:
        console.print(f"\n[red]✗ Error: {e}[/red]")
        raise typer.Exit(code=1)


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
