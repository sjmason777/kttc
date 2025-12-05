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

"""Shared CLI utilities for KTTC commands."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import typer

from kttc.cli.ui import console
from kttc.core import TranslationTask
from kttc.helpers.detection import detect_language
from kttc.llm import AnthropicProvider, BaseLLMProvider, OpenAIProvider

# File extension constants
JSON_EXT = ".json"
TEXT_FILE_EXTENSIONS = (".txt", ".md", JSON_EXT)


def auto_detect_format(output: str | None, output_format: str | None) -> str:
    """Auto-detect output format from file extension.

    Args:
        output: Output file path
        output_format: User-specified format (overrides auto-detection)

    Returns:
        Format string: 'text', 'json', 'markdown', 'html', or 'xlsx'
    """
    if output_format:
        return output_format

    if output:
        suffix = Path(output).suffix.lower()
        if suffix == JSON_EXT:
            return "json"
        if suffix in [".md", ".markdown"]:
            return "markdown"
        if suffix in [".html", ".htm"]:
            return "html"
        if suffix in [".xlsx", ".xls"]:
            return "xlsx"

    return "text"


def auto_detect_glossary(glossary: str | None) -> str | None:
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


def get_available_providers(settings: Any) -> list[str]:
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
        # Silently ignore missing Yandex credentials and continue checking other providers
        pass

    # Check Anthropic
    try:
        settings.get_llm_provider_key("anthropic")
        available.append("anthropic")
    except (ValueError, AttributeError):
        # Silently ignore missing GigaChat credentials and continue checking other providers
        pass

    # Check GigaChat
    try:
        settings.get_llm_provider_credentials("gigachat")
        available.append("gigachat")
    except (ValueError, AttributeError):
        # Silently ignore missing Anthropic API key and continue checking other providers
        pass

    # Check Yandex
    try:
        settings.get_llm_provider_credentials("yandex")
        available.append("yandex")
    except (ValueError, AttributeError):
        # Silently ignore missing OpenAI API key and continue checking other providers
        pass

    # Check Gemini
    try:
        settings.get_llm_provider_credentials("gemini")
        available.append("gemini")
    except (ValueError, AttributeError):
        # Silently ignore missing Gemini API key and continue checking other providers
        pass

    return available


def _resolve_provider_name(provider: str | None, settings: Any) -> str:
    """Resolve provider name from user input or available providers.

    Args:
        provider: User-specified provider name or None
        settings: Application settings

    Returns:
        Resolved provider name

    Raises:
        RuntimeError: If no providers are configured
    """
    if provider is not None:
        return provider

    available_providers = get_available_providers(settings)
    if not available_providers:
        raise RuntimeError(
            "No LLM providers configured. Please set at least one of:\n"
            "  - KTTC_OPENAI_API_KEY\n"
            "  - KTTC_ANTHROPIC_API_KEY\n"
            "  - KTTC_GEMINI_API_KEY\n"
            "  - KTTC_GIGACHAT_CLIENT_ID and KTTC_GIGACHAT_CLIENT_SECRET"
        )

    if settings.default_llm_provider in available_providers:
        return str(settings.default_llm_provider)
    return str(available_providers[0])


def _create_provider_instance(provider_name: str, settings: Any, model: str) -> BaseLLMProvider:
    """Create LLM provider instance based on provider name.

    Args:
        provider_name: Name of the provider (openai, anthropic, gigachat)
        settings: Application settings
        model: Model name to use

    Returns:
        Configured provider instance

    Raises:
        ValueError: If provider is unknown
    """
    from kttc.llm import GeminiProvider, GigaChatProvider

    if provider_name == "openai":
        api_key = settings.get_llm_provider_key(provider_name)
        return OpenAIProvider(api_key=api_key, model=model)

    if provider_name == "anthropic":
        api_key = settings.get_llm_provider_key(provider_name)
        return AnthropicProvider(api_key=api_key, model=model)

    if provider_name == "gemini":
        api_key = settings.get_llm_provider_key(provider_name)
        # Use gemini-2.0-flash as default if no model specified
        gemini_model = model if "gemini" in model.lower() else "gemini-2.0-flash"
        return GeminiProvider(api_key=api_key, model=gemini_model)

    if provider_name == "gigachat":
        credentials = settings.get_llm_provider_credentials(provider_name)
        return GigaChatProvider(
            client_id=credentials["client_id"],
            client_secret=credentials["client_secret"],
            model=model,
        )

    raise ValueError(
        f"Unknown provider: {provider_name}. " "Supported: openai, anthropic, gemini, gigachat"
    )


def setup_llm_provider(
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
    from kttc.llm.model_selector import ModelSelector

    if demo:
        if verbose:
            console.print(
                "[yellow]🎭 Demo mode: Using simulated responses (no API calls)[/yellow]\n"
            )
        return DemoLLMProvider(model="demo-model")

    provider_name = _resolve_provider_name(provider, settings)
    model = settings.default_model

    if auto_select_model and task is not None:
        selector = ModelSelector()
        model = selector.select_best_model(
            source_lang=task.source_lang,
            target_lang=task.target_lang,
            domain=task.context.get("domain") if task.context else None,
            task_type="qa",
            optimize_for="quality",
        )
        if verbose:
            console.print(f"[dim]🤖 Auto-selected model: {model}[/dim]")

    llm_provider = _create_provider_instance(provider_name, settings, model)

    if verbose:
        console.print(f"[dim]Using {provider_name} provider with model {model}[/dim]\n")

    return llm_provider


def load_translation_files(source: str, translation: str, verbose: bool) -> tuple[str, str]:
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


def create_translation_task(
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


def load_glossaries_for_task(
    glossary: str | None,
    task: TranslationTask,
    source_text: str,
    source_lang: str,
    target_lang: str,
    verbose: bool,
) -> None:
    """Load glossaries and add terms to task context."""
    if not glossary:
        return

    from kttc.core import GlossaryManager

    try:
        manager = GlossaryManager()
        glossary_names = [g.strip() for g in glossary.split(",")]
        manager.load_multiple(glossary_names)

        terms = manager.find_in_text(source_text, source_lang, target_lang)
        task.context = task.context or {}
        task.context["glossary_terms"] = [
            {"source": t.source, "target": t.target, "do_not_translate": t.do_not_translate}
            for t in terms
        ]

        if verbose:
            console.print(
                f"[dim]Loaded {len(glossary_names)} glossaries, "
                f"found {len(terms)} relevant terms[/dim]"
            )
    except Exception as e:
        console.print(f"[yellow]⚠ Warning: Failed to load glossaries: {e}[/yellow]")


def detect_languages_from_files(
    source_path: Path,
    translation_path: Path,
    source_lang: str | None,
    target_lang: str | None,
    verbose: bool,
) -> tuple[str | None, str | None]:
    """Auto-detect languages from source and translation files.

    Args:
        source_path: Path to source file
        translation_path: Path to translation file
        source_lang: User-specified source language (or None for auto-detect)
        target_lang: User-specified target language (or None for auto-detect)
        verbose: Verbose output for error messages

    Returns:
        Tuple of (source_lang, target_lang) - may contain None if detection failed
    """
    try:
        if source_path.exists() and not source_lang:
            source_text_sample = source_path.read_text(encoding="utf-8")[:1000]
            source_lang = detect_language(source_text_sample)
            console.print(f"[dim]🔍 Auto-detected source language: {source_lang}[/dim]")

        if translation_path.exists() and not target_lang:
            translation_text_sample = translation_path.read_text(encoding="utf-8")[:1000]
            target_lang = detect_language(translation_text_sample)
            console.print(f"[dim]🔍 Auto-detected target language: {target_lang}[/dim]")
    except Exception as e:
        if verbose:
            console.print(f"[dim]⚠ Language auto-detection failed: {e}[/dim]")

    return source_lang, target_lang


def _detect_language_from_dir(directory: Path, label: str) -> str | None:
    """Detect language from first text file in directory.

    Args:
        directory: Path to directory to scan
        label: Label for console output (e.g., "source", "target")

    Returns:
        Detected language code or None if detection failed
    """
    if not directory.exists():
        return None

    for f in directory.iterdir():
        if f.is_file() and f.suffix in TEXT_FILE_EXTENSIONS:
            sample = f.read_text(encoding="utf-8")[:1000]
            detected = detect_language(sample)
            console.print(f"[dim]🔍 Auto-detected {label} language: {detected}[/dim]")
            return detected
    return None


def detect_languages_from_directory(
    source_dir: Path,
    translation_dir: Path,
    source_lang: str | None,
    target_lang: str | None,
    verbose: bool,
) -> tuple[str | None, str | None]:
    """Auto-detect languages from first files in directories.

    Args:
        source_dir: Path to source directory
        translation_dir: Path to translation directory
        source_lang: User-specified source language (or None for auto-detect)
        target_lang: User-specified target language (or None for auto-detect)
        verbose: Verbose output for error messages

    Returns:
        Tuple of (source_lang, target_lang) - may contain None if detection failed
    """
    try:
        if not source_lang:
            source_lang = _detect_language_from_dir(source_dir, "source")

        if not target_lang:
            target_lang = _detect_language_from_dir(translation_dir, "target")
    except Exception as e:
        if verbose:
            console.print(f"[dim]⚠ Language auto-detection failed: {e}[/dim]")

    return source_lang, target_lang


def validate_required_languages(
    source_lang: str | None,
    target_lang: str | None,
    context: str = "",
) -> None:
    """Validate that required languages are specified."""
    if not source_lang or not target_lang:
        msg = f"[red]Error: --source-lang and --target-lang are required {context}[/red]"
        console.print(msg)
        raise typer.Exit(code=1)


def map_model_to_provider(selected_model: str | None, provider: str | None) -> str | None:
    """Map selected model to provider name."""
    if selected_model is None or provider is not None:
        return provider

    model_lower = selected_model.lower()
    if "gpt" in model_lower and "yandex" not in model_lower:
        return "openai"
    if "claude" in model_lower:
        return "anthropic"
    if "gemini" in model_lower:
        return "gemini"
    if "yandex" in model_lower:
        return "yandex"
    return provider
