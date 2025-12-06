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

"""Helper functions for check command.

This module contains helper functions for:
- Mode detection (single, compare, batch)
- Smart routing and complexity analysis
- NLP and style analysis
- Lightweight metrics calculation
- Auto-correction handling
- Report display and saving
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from kttc.agents import AgentOrchestrator
from kttc.cli.formatters import HTMLFormatter, MarkdownFormatter, XLSXFormatter
from kttc.cli.ui import (
    console,
    create_step_progress,
    get_nlp_insights,
    print_header,
    print_startup_info,
)
from kttc.cli.utils import (
    create_translation_task,
    get_available_providers,
)
from kttc.core import QAReport, TranslationTask
from kttc.llm import BaseLLMProvider

# File extension constants
JSON_EXT = ".json"
BATCH_INPUT_EXTENSIONS = (".csv", JSON_EXT, ".jsonl")


def detect_check_mode(
    source: str, translations: list[str] | None
) -> tuple[str, dict[str, str | list[str]]]:
    """Detect which mode to run check command in.

    Returns:
        tuple: (mode, params) where mode is 'single', 'compare', or 'batch'
               and params contains mode-specific parameters
    """
    source_path = Path(source)

    # Check if source is CSV/JSON/JSONL file (batch mode)
    if source_path.is_file() and source_path.suffix.lower() in BATCH_INPUT_EXTENSIONS:
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
    # Multiple translations - compare mode
    return "compare", {"source": source, "translations": translations}


def perform_smart_routing(
    source_text: str,
    source_lang: str,
    target_lang: str,
    task: TranslationTask,
    settings: Any,
    show_routing_info: bool,
    simple_threshold: float = 0.3,
    complex_threshold: float = 0.7,
) -> tuple[str | None, Any]:
    """Perform smart routing to select optimal model.

    Args:
        source_text: Source text to analyze
        source_lang: Source language code
        target_lang: Target language code
        task: Translation task with context
        settings: Application settings
        show_routing_info: Whether to display routing information
        simple_threshold: Complexity threshold for simple texts (default: 0.3)
        complex_threshold: Complexity threshold for complex texts (default: 0.7)

    Returns:
        Tuple of (selected_model, complexity_score)
    """
    from kttc.llm import ComplexityRouter

    try:
        router = ComplexityRouter(simple_threshold, complex_threshold)
        available_providers = get_available_providers(settings)

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
            console.print(f"[dim]  Sentence length: {complexity_score.sentence_length:.2f}[/dim]")
            console.print(f"[dim]  Rare words: {complexity_score.rare_words:.2f}[/dim]")
            console.print(f"[dim]  Syntactic: {complexity_score.syntactic:.2f}[/dim]")
            console.print(f"[dim]  Domain-specific: {complexity_score.domain_specific:.2f}[/dim]")
            console.print(f"[dim]Selected model: {selected_model}[/dim]\n")

        return selected_model, complexity_score
    except Exception as e:
        console.print(f"[yellow]⚠ Warning: Smart routing failed, using default: {e}[/yellow]")
        return None, None


def run_nlp_analysis(
    task: TranslationTask, verbose: bool, api_errors: list[str]
) -> dict[str, Any] | None:
    """Run NLP analysis on the translation."""
    from kttc.helpers import get_helper_for_language

    helper = get_helper_for_language(task.target_lang)
    if not helper or not helper.is_available():
        if verbose:
            console.print(
                "[dim]⊘ Step 1/3: Linguistic analysis (not available for this language)[/dim]"
            )
        return None

    if verbose:
        with create_step_progress() as progress:
            progress.add_task("[cyan]Step 1/3: Analyzing linguistic features...[/cyan]", total=None)
            try:
                nlp_insights = get_nlp_insights(task, helper)
            except Exception as e:
                api_errors.append(f"NLP analysis failed: {str(e)}")
                nlp_insights = None
        console.print("[green]✓[/green] Step 1/3: Linguistic analysis complete")
    else:
        try:
            nlp_insights = get_nlp_insights(task, helper)
        except Exception:
            nlp_insights = None

    return nlp_insights


def run_style_analysis(source_text: str, source_lang: str, verbose: bool) -> Any:
    """Run style analysis on source text."""
    try:
        from kttc.style import StyleFingerprint

        style_analyzer = StyleFingerprint()
        style_profile = style_analyzer.analyze(source_text, lang=source_lang)
        if verbose and style_profile.is_literary:
            console.print(
                f"[magenta]📚 Literary text detected: "
                f"{style_profile.detected_pattern.value.replace('_', ' ').title()}[/magenta]"
            )
        return style_profile
    except Exception:
        return None


async def run_quality_evaluation(
    llm_provider: BaseLLMProvider,
    task: TranslationTask,
    threshold: float,
    settings: Any,
    verbose: bool,
    api_errors: list[str],
    quick: bool = False,
    profile: Any = None,
    selected_agents: list[str] | None = None,
) -> tuple[QAReport, AgentOrchestrator]:
    """Run multi-agent quality evaluation.

    Args:
        llm_provider: LLM provider for API calls
        task: Translation task to evaluate
        threshold: Quality threshold
        settings: Application settings
        verbose: Enable verbose output
        api_errors: List to collect API errors
        quick: Enable quick mode (3 core agents, no iterations)
        profile: MQM profile with custom agent weights (optional)
        selected_agents: List of agent names to use (optional)
    """
    # Extract agent weights from profile if provided
    agent_weights = None
    if profile and hasattr(profile, "agent_weights"):
        agent_weights = profile.agent_weights

    orchestrator = AgentOrchestrator(
        llm_provider,
        quality_threshold=threshold,
        agent_temperature=settings.default_temperature,
        agent_max_tokens=settings.default_max_tokens,
        enable_dynamic_selection=not quick,  # Disable dynamic selection in quick mode
        quick_mode=quick,
        agent_weights=agent_weights,
        selected_agents=selected_agents,
    )

    if verbose:
        with create_step_progress() as progress:
            progress.add_task(
                "[cyan]Step 2/3: Running multi-agent quality assessment...[/cyan]", total=None
            )
            try:
                report = await orchestrator.evaluate(task)
            except Exception as e:
                console.print("[red]✗[/red] Step 2/3: Quality assessment failed")
                api_errors.append(f"Quality assessment failed: {str(e)}")
                raise RuntimeError(f"Evaluation failed: {e}") from e
        console.print("[green]✓[/green] Step 2/3: Quality assessment complete")
        console.print("[green]✓[/green] Step 3/3: Report ready")
        console.print()
    else:
        with create_step_progress() as progress:
            progress.add_task("[cyan]Evaluating translation quality...[/cyan]", total=None)
            try:
                report = await orchestrator.evaluate(task)
            except Exception as e:
                api_errors.append(f"Quality assessment failed: {str(e)}")
                raise RuntimeError(f"Evaluation failed: {e}") from e

    return report, orchestrator


def calculate_lightweight_metrics(
    source_text: str,
    translation_text: str,
    reference: str | None,
    verbose: bool,
) -> tuple[Any, list[Any] | None, float | None]:
    """Calculate lightweight metrics and detect rule-based errors."""
    from kttc.evaluation import ErrorDetector, LightweightMetrics

    metrics_calculator = LightweightMetrics()
    error_detector = ErrorDetector()

    reference_text = None
    if reference:
        try:
            reference_path = Path(reference)
            if reference_path.exists():
                reference_text = reference_path.read_text(encoding="utf-8").strip()
                if verbose:
                    console.print(f"[dim]Loaded {len(reference_text)} chars from reference[/dim]")
            else:
                console.print(f"[yellow]⚠ Reference file not found: {reference}[/yellow]")
        except Exception as e:
            console.print(f"[yellow]⚠ Failed to load reference: {e}[/yellow]")

    try:
        reference_for_metrics = reference_text if reference_text else translation_text
        lightweight_scores = metrics_calculator.evaluate(
            translation=translation_text,
            reference=reference_for_metrics,
            source=source_text,
        )

        rule_based_errors = error_detector.detect_all_errors(
            source=source_text, translation=translation_text
        )
        rule_based_score = error_detector.calculate_rule_based_score(rule_based_errors)

        if not reference_text and verbose:
            console.print(
                "[dim]ℹ️  No reference translation provided. "
                "Metrics show baseline (self-comparison). "
                "Use --reference for meaningful scores.[/dim]\n"
            )

        return lightweight_scores, rule_based_errors, rule_based_score
    except Exception as e:
        if verbose:
            console.print(f"[dim]⚠ Lightweight metrics calculation failed: {e}[/dim]")
        return None, None, None


async def handle_auto_correction(
    auto_correct: bool,
    report: QAReport,
    task: TranslationTask,
    orchestrator: AgentOrchestrator,
    llm_provider: BaseLLMProvider,
    translation: str,
    source_text: str,
    source_lang: str,
    target_lang: str,
    correction_level: str,
    settings: Any,
    verbose: bool,
) -> None:
    """Handle auto-correction if requested."""
    if not auto_correct or len(report.errors) == 0:
        return

    from kttc.core.correction import AutoCorrector

    console.print(f"\n[yellow]🔧 Applying auto-correction ({correction_level})...[/yellow]")
    try:
        corrector = AutoCorrector(llm_provider)
        corrected_text = await corrector.auto_correct(
            task=task,
            errors=report.errors,
            correction_level=correction_level,
            temperature=settings.default_temperature,
        )

        corrected_path = Path(translation).parent / f"{Path(translation).stem}_corrected.txt"
        corrected_path.write_text(corrected_text, encoding="utf-8")
        console.print(f"[green]✓ Corrected translation saved to: {corrected_path}[/green]")

        if verbose:
            console.print("\n[dim]Re-evaluating corrected translation...[/dim]")
        corrected_task = create_translation_task(
            source_text, corrected_text, source_lang, target_lang, verbose=False
        )
        corrected_report = await orchestrator.evaluate(corrected_task)

        console.print("\n[bold]Corrected Translation Quality:[/bold]")
        console.print(f"MQM Score: [cyan]{corrected_report.mqm_score:.2f}[/cyan]")
        console.print(f"Errors: {len(report.errors)} → [cyan]{len(corrected_report.errors)}[/cyan]")
        status_color = "green" if corrected_report.status == "pass" else "red"
        console.print(
            f"Status: {report.status} → [{status_color}]{corrected_report.status}[/{status_color}]"
        )
    except Exception as e:
        console.print(f"[yellow]⚠ Auto-correction failed: {e}[/yellow]")
        if verbose:
            import traceback

            traceback.print_exc()


def display_check_header(
    source: str,
    translation: str,
    source_lang: str,
    target_lang: str,
    threshold: float,
    auto_select_model: bool,
    auto_correct: bool,
    correction_level: str,
) -> None:
    """Display check command header."""
    print_header(
        "✓ Translation Quality Check",
        "Evaluating translation quality with multi-agent AI system",
    )
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


def save_report(report: QAReport, output: str, output_format: str) -> None:
    """Save report to file."""
    output_path = Path(output)

    if output_format == "json" or output.endswith(JSON_EXT):
        # Save as JSON
        data = report.model_dump(mode="json")
        output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    elif output_format == "markdown" or output.endswith(".md"):
        # Save as Markdown using the new formatter
        MarkdownFormatter.format_report(report, output_path)
    elif output_format == "html" or output.endswith(".html"):
        # Save as HTML using the new formatter
        HTMLFormatter.format_report(report, output_path)
    elif output_format == "xlsx" or output.endswith(".xlsx"):
        # Save as Excel using the XLSX formatter
        if not XLSXFormatter.is_available():
            console.print(
                "[yellow]Warning: openpyxl not installed. "
                "Install with: pip install kttc[xlsx][/yellow]"
            )
            console.print("[dim]Falling back to JSON format...[/dim]")
            data = report.model_dump(mode="json")
            json_path = output_path.with_suffix(JSON_EXT)
            json_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        else:
            XLSXFormatter.format_report(report, output_path)
    else:
        # Default to JSON
        data = report.model_dump(mode="json")
        output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def print_verbose_autodetect_info(
    mode: str,
    detected_glossary: str | None,
    smart_routing: bool,
    detected_format: str,
) -> None:
    """Print verbose auto-detection info."""
    console.print(f"[dim]🎯 Mode: {mode}[/dim]")
    if detected_glossary:
        console.print(f"[dim]📚 Glossary: {detected_glossary}[/dim]")
    if smart_routing:
        console.print("[dim]🧠 Smart routing: enabled[/dim]")
    console.print(f"[dim]📄 Output format: {detected_format}[/dim]\n")


# ============================================================================
# Ensemble Mode Helpers
# ============================================================================


def setup_ensemble_providers(
    llm_providers: str | None,
    llm_count: int,
    settings: Any,
    verbose: bool,
    demo: bool,
) -> tuple[bool, dict[str, BaseLLMProvider], list[str]]:
    """Setup providers for ensemble mode.

    Returns:
        Tuple of (use_ensemble, providers_dict, parsed_provider_names)
    """
    from kttc.cli.utils import setup_multi_llm_providers

    providers_list: list[str] = []
    if llm_providers:
        providers_list = [p.strip().lower() for p in llm_providers.split(",")]

    use_ensemble = llm_count > 1 or len(providers_list) > 1
    multi_providers: dict[str, BaseLLMProvider] = {}

    if not use_ensemble:
        return False, {}, providers_list

    # Determine which providers to use
    if len(providers_list) > 1:
        ensemble_providers = providers_list[:llm_count] if llm_count > 1 else providers_list
    else:
        default_order = ["yandex", "gigachat", "openai", "anthropic", "gemini"]
        ensemble_providers = default_order[:llm_count]

    if verbose:
        console.print(
            f"\n[bold cyan]🔀 Ensemble Mode[/bold cyan]: "
            f"Using {len(ensemble_providers)} LLM providers"
        )
        console.print(f"[dim]   Providers: {', '.join(ensemble_providers)}[/dim]")

    try:
        multi_providers = setup_multi_llm_providers(
            ensemble_providers, settings, verbose, demo=demo
        )
    except RuntimeError as e:
        console.print(f"[red]Error setting up ensemble providers: {e}[/red]")
        console.print("[yellow]Falling back to single provider mode...[/yellow]")
        return False, {}, providers_list

    return True, multi_providers, providers_list


async def run_ensemble_evaluation(
    multi_providers: dict[str, BaseLLMProvider],
    task: TranslationTask,
    threshold: float,
    quick: bool,
    selected_agents: list[str] | None,
    verbose: bool,
) -> QAReport:
    """Run evaluation in ensemble mode.

    Returns:
        QAReport with ensemble results
    """
    from kttc.agents import MultiProviderAgentOrchestrator

    consensus_threshold = min(2, len(multi_providers))
    multi_orchestrator = MultiProviderAgentOrchestrator(
        providers=multi_providers,
        quality_threshold=threshold,
        consensus_threshold=consensus_threshold,
        aggregation_strategy="weighted_vote",
        quick_mode=quick,
        selected_agents=selected_agents,
    )

    if verbose:
        console.print("\n[cyan]Running ensemble evaluation...[/cyan]")

    report = await multi_orchestrator.evaluate(task)

    if verbose and report.ensemble_metadata:
        _display_ensemble_results(report.ensemble_metadata)

    return report


def _display_ensemble_results(meta: dict[str, Any]) -> None:
    """Display ensemble evaluation results."""
    console.print("\n[bold green]✓ Ensemble Evaluation Complete[/bold green]")
    console.print(
        f"[dim]   Providers: {meta['providers_successful']}/{meta['providers_total']} successful[/dim]"
    )
    console.print(
        f"[dim]   Errors: {meta['confirmed_errors']} confirmed, "
        f"{meta['rejected_errors']} rejected by cross-validation[/dim]"
    )
    if meta.get("cross_validation_rate", 1.0) < 1.0:
        console.print(f"[dim]   Cross-validation rate: {meta['cross_validation_rate']:.0%}[/dim]")


# ============================================================================
# Debate Mode Helpers
# ============================================================================


async def run_debate_verification(
    report: QAReport,
    task: TranslationTask,
    llm_provider: BaseLLMProvider,
    threshold: float,
    verbose: bool,
) -> tuple[QAReport, dict[str, Any] | None]:
    """Run debate mode to cross-verify errors.

    Returns:
        Tuple of (updated_report, debate_summary)
    """
    from kttc.agents import DebateOrchestrator
    from kttc.core import MQMScorer

    if verbose:
        console.print("\n[bold cyan]🎭 Debate Mode[/bold cyan]: Cross-verifying errors...")
        console.print(f"[dim]   {len(report.errors)} errors to verify[/dim]")

    debate_orchestrator = DebateOrchestrator(llm_provider)
    verified_errors, debate_rounds = await debate_orchestrator.run_debate(report.errors, task)

    # Calculate how many errors were filtered
    original_count = len(report.errors)
    verified_count = len(verified_errors)
    rejected_count = original_count - verified_count

    # Update report with verified errors
    report.errors = verified_errors

    # Recalculate MQM score with filtered errors
    scorer = MQMScorer()
    report.mqm_score = scorer.calculate_score(verified_errors, task.word_count)
    report.status = "pass" if report.mqm_score >= threshold else "fail"

    # Get debate summary for display
    debate_summary = debate_orchestrator.get_debate_summary(debate_rounds)

    if verbose:
        console.print(
            f"[green]   ✓ {verified_count} errors confirmed[/green], "
            f"[yellow]{rejected_count} rejected as false positives[/yellow]"
        )
        if rejected_count > 0:
            console.print(
                f"[dim]   Precision improvement: {debate_summary['precision_improvement']}[/dim]"
            )

    return report, debate_summary


# ============================================================================
# Results Display Helpers
# ============================================================================


def display_usage_and_mode_info(
    report: QAReport,
    llm_provider: BaseLLMProvider | None,
    use_ensemble: bool,
    show_cost: bool,
    quick: bool,
) -> None:
    """Display usage statistics and mode indicators."""
    if not use_ensemble and llm_provider and hasattr(llm_provider, "usage"):
        usage = llm_provider.usage
        report.usage_stats = {
            "total_tokens": usage.total_tokens,
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "total_cost_usd": usage.estimated_cost_usd,
            "api_calls": usage.call_count,
        }
        if show_cost:
            console.print(f"\n[dim]💰 {usage.format_summary()}[/dim]")
    elif use_ensemble and show_cost and report.ensemble_metadata:
        meta = report.ensemble_metadata
        console.print(f"\n[dim]⏱ Ensemble latency: {meta['total_latency']:.2f}s total[/dim]")

    if quick:
        console.print("[dim]⚡ Quick mode: 3 core agents, single pass[/dim]")

    if use_ensemble:
        console.print("[dim]🔀 Ensemble mode: Multi-provider cross-validation[/dim]")
