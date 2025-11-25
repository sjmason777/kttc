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

"""Check command for translation quality assessment.

This module provides the main check command that routes to:
- Single file check (default)
- Compare mode (multiple translations)
- Batch mode (directory or CSV/JSON files)
- Self-check mode (monolingual proofreading)
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

import typer

from kttc.cli.commands.check_helpers import (
    calculate_lightweight_metrics,
    detect_check_mode,
    display_check_header,
    handle_auto_correction,
    perform_smart_routing,
    print_verbose_autodetect_info,
    run_nlp_analysis,
    run_quality_evaluation,
    run_style_analysis,
    save_report,
)
from kttc.cli.commands.compare import run_compare as _run_compare_command
from kttc.cli.commands.self_check import handle_self_check_mode
from kttc.cli.formatters import ConsoleFormatter
from kttc.cli.ui import (
    console,
    print_info,
    print_translation_preview,
)
from kttc.cli.utils import (
    auto_detect_format,
    auto_detect_glossary,
    create_translation_task,
    detect_languages_from_directory,
    detect_languages_from_files,
    load_glossaries_for_task,
    load_translation_files,
    map_model_to_provider,
    setup_llm_provider,
    validate_required_languages,
)
from kttc.utils.config import get_settings

# Create Typer app for check command
check_app = typer.Typer()

# Agent presets
AGENT_PRESETS = {
    "minimal": ["accuracy", "fluency"],
    "default": ["accuracy", "fluency", "terminology"],
    "full": ["accuracy", "fluency", "terminology", "hallucination", "context"],
}

VALID_AGENTS = {
    "accuracy",
    "fluency",
    "terminology",
    "hallucination",
    "context",
    "style_preservation",
    "fluency_russian",
    "fluency_chinese",
    "fluency_hindi",
    "fluency_persian",
    "fluency_english",
}


def _parse_agents_selection(agents_str: str) -> list[str]:
    """Parse agents selection string.

    Args:
        agents_str: Preset name or comma-separated list of agents

    Returns:
        List of agent IDs

    Raises:
        typer.Exit: If invalid agent specified
    """
    agents_str = agents_str.strip().lower()

    # Check if it's a preset
    if agents_str in AGENT_PRESETS:
        return AGENT_PRESETS[agents_str]

    # Parse comma-separated list
    agents = [a.strip() for a in agents_str.split(",") if a.strip()]

    # Validate agents
    invalid = [a for a in agents if a not in VALID_AGENTS]
    if invalid:
        console.print(f"[red]Error: Unknown agents: {', '.join(invalid)}[/red]")
        console.print(f"[dim]Valid agents: {', '.join(sorted(VALID_AGENTS))}[/dim]")
        console.print(f"[dim]Presets: {', '.join(AGENT_PRESETS.keys())}[/dim]")
        raise typer.Exit(code=1)

    return agents


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
    """Wrapper for compare mode called from check command."""
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
    simple_threshold: float,  # noqa: ARG001 - Reserved for future use
    complex_threshold: float,  # noqa: ARG001 - Reserved for future use
    glossary: str | None,
    reference: str | None,
    verbose: bool,
    demo: bool = False,
    quick: bool = False,
    show_cost: bool = False,
    profile: Any = None,
    selected_agents: list[str] | None = None,
) -> None:
    """Async implementation of check command."""
    # Configure logging
    log_level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(level=log_level, format="%(message)s", force=True)

    settings = get_settings()

    # Display header (verbose mode only)
    if verbose:
        display_check_header(
            source,
            translation,
            source_lang,
            target_lang,
            threshold,
            auto_select_model,
            auto_correct,
            correction_level,
        )

    # Load files and create task
    source_text, translation_text = load_translation_files(source, translation, verbose)
    task = create_translation_task(source_text, translation_text, source_lang, target_lang, verbose)

    # Load glossaries
    load_glossaries_for_task(glossary, task, source_text, source_lang, target_lang, verbose)

    # Smart routing
    selected_model = None
    if smart_routing:
        selected_model, _ = perform_smart_routing(
            source_text, source_lang, target_lang, task, settings, show_routing_info
        )
        provider = map_model_to_provider(selected_model, provider)

    # Setup LLM provider
    llm_provider = setup_llm_provider(
        provider, settings, verbose, task=task, auto_select_model=auto_select_model, demo=demo
    )

    # Show translation preview
    if verbose:
        print_translation_preview(source_text, translation_text)

    # Run analysis steps
    api_errors: list[str] = []
    nlp_insights = run_nlp_analysis(task, verbose, api_errors)
    style_profile = run_style_analysis(source_text, source_lang, verbose)

    # Quality evaluation
    report, orchestrator = await run_quality_evaluation(
        llm_provider,
        task,
        threshold,
        settings,
        verbose,
        api_errors,
        quick,
        profile,
        selected_agents,
    )

    # Calculate metrics
    lightweight_scores, rule_based_errors, rule_based_score = calculate_lightweight_metrics(
        source_text, translation_text, reference, verbose
    )

    # Display results
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

    # Add usage stats to report for JSON output
    if hasattr(llm_provider, "usage"):
        usage = llm_provider.usage
        report.usage_stats = {
            "total_tokens": usage.total_tokens,
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "total_cost_usd": usage.estimated_cost_usd,
            "api_calls": usage.call_count,
        }

        # Show cost if requested
        if show_cost:
            console.print(f"\n[dim]💰 {usage.format_summary()}[/dim]")

    # Show quick mode indicator
    if quick:
        console.print("[dim]⚡ Quick mode: 3 core agents, single pass[/dim]")

    # Auto-correction
    await handle_auto_correction(
        auto_correct,
        report,
        task,
        orchestrator,
        llm_provider,
        translation,
        source_text,
        source_lang,
        target_lang,
        correction_level,
        settings,
        verbose,
    )

    # Save output
    if output:
        save_report(report, output, format)
        console.print(f"\n[dim]Report saved to: {output}[/dim]")

    # Exit with appropriate code
    if report.status == "fail":
        console.print()
        print_info(f"Translation quality below threshold ({threshold}). Exiting with error code.")
        raise typer.Exit(code=1)


def _run_single_mode(
    mode_params: dict[str, Any],
    source_lang: str | None,
    target_lang: str | None,
    verbose: bool,
    threshold: float,
    output: str | None,
    detected_format: str,
    provider: str | None,
    auto_select_model: bool,
    auto_correct: bool,
    correction_level: str,
    smart_routing: bool,
    show_routing_info: bool,
    simple_threshold: float,
    complex_threshold: float,
    detected_glossary: str | None,
    reference: str | None,
    demo: bool,
    quick: bool = False,
    show_cost: bool = False,
    profile: Any = None,
    selected_agents: list[str] | None = None,
) -> None:
    """Run single file check mode."""
    if not source_lang or not target_lang:
        source_path = Path(str(mode_params["source"]))
        translation_path = Path(str(mode_params["translation"]))
        source_lang, target_lang = detect_languages_from_files(
            source_path, translation_path, source_lang, target_lang, verbose
        )
    validate_required_languages(source_lang, target_lang, "(auto-detection failed)")
    assert source_lang is not None and target_lang is not None  # validated above

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
            quick,
            show_cost,
            profile,
            selected_agents,
        )
    )


def _run_batch_dir_mode(
    mode_params: dict[str, Any],
    source_lang: str | None,
    target_lang: str | None,
    verbose: bool,
    threshold: float,
    output: str | None,
    provider: str | None,
    demo: bool,
) -> None:
    """Run batch directory mode."""
    # Import here to avoid circular imports
    from kttc.cli.commands.batch import batch_async

    if not source_lang or not target_lang:
        source_dir = Path(str(mode_params["source_dir"]))
        translation_dir = Path(str(mode_params["translation_dir"]))
        source_lang, target_lang = detect_languages_from_directory(
            source_dir, translation_dir, source_lang, target_lang, verbose
        )
    validate_required_languages(
        source_lang, target_lang, "for directory mode (auto-detection failed)"
    )
    assert source_lang is not None and target_lang is not None  # validated above

    asyncio.run(
        batch_async(
            str(mode_params["source_dir"]),
            str(mode_params["translation_dir"]),
            source_lang,
            target_lang,
            threshold,
            output or "report.json",
            4,
            provider,
            verbose,
            demo,
        )
    )


def _route_check_mode(
    mode: str,
    mode_params: dict[str, Any],
    source_lang: str | None,
    target_lang: str | None,
    verbose: bool,
    threshold: float,
    output: str | None,
    detected_format: str,
    provider: str | None,
    auto_select_model: bool,
    auto_correct: bool,
    correction_level: str,
    smart_routing: bool,
    show_routing_info: bool,
    simple_threshold: float,
    complex_threshold: float,
    detected_glossary: str | None,
    reference: str | None,
    demo: bool,
    quick: bool = False,
    show_cost: bool = False,
    profile: Any = None,
    selected_agents: list[str] | None = None,
) -> None:
    """Route to appropriate handler based on detected mode."""
    if mode == "single":
        _run_single_mode(
            mode_params,
            source_lang,
            target_lang,
            verbose,
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
            demo,
            quick,
            show_cost,
            profile,
            selected_agents,
        )
    elif mode == "compare":
        translations_list = mode_params["translations"]
        assert isinstance(translations_list, list), "translations must be a list"
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
        # Import here to avoid circular imports
        from kttc.cli.commands.batch import batch_from_file_async

        asyncio.run(
            batch_from_file_async(
                str(mode_params["file_path"]),
                threshold,
                output or "report.json",
                4,
                None,
                provider,
                smart_routing,
                False,
                True,
                detected_glossary,
                verbose,
                demo,
            )
        )
    elif mode == "batch_dir":
        _run_batch_dir_mode(
            mode_params,
            source_lang,
            target_lang,
            verbose,
            threshold,
            output,
            provider,
            demo,
        )


@check_app.command(name="check")
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
        help="Output format (overrides auto-detection): text, json, markdown, html, or xlsx",
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
        help="Glossaries to use (comma-separated), 'auto' to auto-detect, or 'none' to disable",
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
    self_check: bool = typer.Option(
        False,
        "--self",
        help="Self-check mode: proofread text without translation (monolingual)",
    ),
    lang: str | None = typer.Option(
        None,
        "--lang",
        "-l",
        help="Language for self-check mode (e.g., 'ru', 'en'). Sets both source and target.",
    ),
    quick: bool = typer.Option(
        False,
        "--quick",
        "-q",
        help="Quick mode: single pass with 3 core agents (accuracy, fluency, terminology). Faster and cheaper.",
    ),
    thorough: bool = typer.Option(
        False,
        "--thorough",
        help="Thorough mode: all agents including hallucination/context, extra validation. More expensive but comprehensive.",
    ),
    show_cost: bool = typer.Option(
        False,
        "--show-cost",
        help="Show token usage and estimated API cost after check.",
    ),
    profile: str | None = typer.Option(
        None,
        "--profile",
        "-p",
        help="MQM profile to use (default, strict, minimal, legal, medical, marketing, literary, technical, or path to YAML)",
    ),
    agents: str | None = typer.Option(
        None,
        "--agents",
        help="Agents to use: preset (minimal, default, full) or comma-separated list (accuracy,fluency,terminology,hallucination,context)",
    ),
    debate: bool = typer.Option(
        False,
        "--debate",
        help="Enable debate mode: agents cross-verify errors to reduce false positives.",
    ),
) -> None:
    """
    Smart translation quality checker with auto-detection.

    🎯 AUTO-DETECTS MODE:
    - Single file: Simple quality check
    - Multiple translations: Automatic comparison
    - Directory/CSV/JSON: Batch processing
    - Self-check (--self): Monolingual proofreading

    SMART DEFAULTS (can disable):
    - Smart routing enabled (--no-smart-routing to disable)
    - Auto-detects glossary 'base' if exists
    - Auto-detects output format from file extension

    📝 EXAMPLES:

    # Single check
    kttc check source.txt translation.txt --source-lang en --target-lang ru

    # Self-check mode (proofread without translation)
    kttc check article.md --self --lang ru
    kttc check article.md --source-lang ru --target-lang ru  # equivalent

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
        # 🥚 Self-check mode (Easter egg!)
        if handle_self_check_mode(
            source,
            source_lang,
            target_lang,
            lang,
            threshold,
            output,
            format,
            provider,
            verbose,
            demo,
            self_check,
        ):
            return

        # 🎯 Auto-detect mode
        mode, mode_params = detect_check_mode(source, translations)
        detected_glossary = auto_detect_glossary(glossary)
        detected_format = auto_detect_format(output, format)

        # Show auto-detection info if verbose
        if verbose:
            print_verbose_autodetect_info(mode, detected_glossary, smart_routing, detected_format)

        # Load profile if specified
        loaded_profile = None
        if profile:
            from kttc.core import load_profile as load_mqm_profile

            try:
                loaded_profile = load_mqm_profile(profile)
                if verbose:
                    console.print(
                        f"[dim]📋 Profile: {loaded_profile.name} - {loaded_profile.description}[/dim]"
                    )
                # Override threshold from profile if not explicitly set
                if threshold == 95.0:  # default value
                    threshold = loaded_profile.quality_threshold
            except ValueError as e:
                console.print(f"[red]Error loading profile: {e}[/red]")
                raise typer.Exit(code=1)

        # Parse agents selection
        selected_agents = None
        if agents:
            selected_agents = _parse_agents_selection(agents)
            if verbose:
                console.print(f"[dim]🤖 Agents: {', '.join(selected_agents)}[/dim]")
        elif thorough:
            # Thorough mode: use all agents including hallucination and context
            selected_agents = ["accuracy", "fluency", "terminology", "hallucination", "context"]
            if verbose:
                console.print(
                    "[dim]🔬 Thorough mode: using all agents (accuracy, fluency, terminology, hallucination, context)[/dim]"
                )

        # Validate quick and thorough are not both set
        if quick and thorough:
            console.print("[red]Error: Cannot use --quick and --thorough together[/red]")
            raise typer.Exit(code=1)

        # Route to appropriate handler
        _route_check_mode(
            mode,
            mode_params,
            source_lang,
            target_lang,
            verbose,
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
            demo,
            quick,
            show_cost,
            loaded_profile,
            selected_agents,
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
