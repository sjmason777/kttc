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

"""Terminology system CLI commands.

Provides access to linguistic reference glossaries including:
- MQM error types and definitions
- Language-specific grammar rules (Russian cases, Chinese classifiers, etc.)
- NLP terminology and concepts
"""

from __future__ import annotations

import json
from typing import Any

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from kttc.terminology import (
    GlossaryManager,
    TermValidator,
)

console = Console()

# Create terminology subcommand app
terminology_app = typer.Typer(
    name="terminology",
    help="Access linguistic reference glossaries and validators",
    no_args_is_help=True,
)


@terminology_app.command("list")
def list_glossaries(
    lang: str | None = typer.Option(None, "--lang", "-l", help="Filter by language code")
) -> None:
    """List all available linguistic reference glossaries.

    Shows MQM error types, grammar rules, and NLP terminology across languages.

    Example:
        kttc terminology list
        kttc terminology list --lang ru
        kttc terminology list --lang zh
    """
    try:
        manager = GlossaryManager()

        # Get all available glossaries using GlossaryManager
        all_glossaries = manager.list_available_glossaries()

        if not all_glossaries:
            console.print("[yellow]No terminology glossaries found[/yellow]")
            return

        # Filter by language if specified
        if lang:
            if lang not in all_glossaries:
                console.print(f"[yellow]No glossaries found for language '{lang}'[/yellow]")
                return
            all_glossaries = {lang: all_glossaries[lang]}

        # Collect glossary information
        glossaries_info = []
        for file_lang, glossary_types in all_glossaries.items():
            for glossary_type in glossary_types:
                try:
                    # Get metadata for term count
                    metadata = manager.get_metadata(file_lang, glossary_type)
                    term_count = metadata.total_terms if metadata and metadata.total_terms else 0

                    # Build path for display
                    file_path = manager.glossaries_dir / file_lang / f"{glossary_type}.json"

                    glossaries_info.append((file_lang, glossary_type, term_count, file_path))
                except Exception:
                    # Skip invalid glossaries
                    continue

        if not glossaries_info:
            console.print("[yellow]No valid glossaries found[/yellow]")
            return

        # Create display table
        table = Table(title="Linguistic Reference Glossaries", show_lines=True)
        table.add_column("Language", style="cyan", no_wrap=True)
        table.add_column("Category", style="green")
        table.add_column("Terms/Entries", style="yellow", justify="right")
        table.add_column("Path", style="dim")

        for file_lang, glossary_type, term_count, file_path in sorted(glossaries_info):
            table.add_row(
                file_lang.upper(),
                glossary_type.replace("_", " ").title(),
                str(term_count) if term_count > 0 else "N/A",
                str(file_path.relative_to(manager.glossaries_dir.parent)),
            )

        console.print(table)
        console.print(f"\n[dim]Total: {len(glossaries_info)} glossaries[/dim]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(code=1)


def _display_mqm_glossary(glossary_data: dict[str, Any], limit: int) -> None:
    """Display MQM error types glossary."""
    error_types = glossary_data["error_types"]
    table = Table(title=f"MQM Error Types ({len(error_types)} types)", show_lines=True)
    table.add_column("Type", style="cyan", no_wrap=True)
    table.add_column("Definition", style="green", max_width=60)
    table.add_column("Severity", style="yellow")
    for i, (error_type, info) in enumerate(sorted(error_types.items())):
        if i >= limit:
            break
        table.add_row(error_type, info.get("definition", ""), info.get("severity", ""))
    console.print(table)


def _display_cases_glossary(glossary_data: dict[str, Any], key: str, title: str) -> None:
    """Display cases glossary (Russian/Hindi)."""
    cases = glossary_data[key]
    table = Table(title=f"{title} ({len(cases)} cases)", show_lines=True)
    table.add_column("Case", style="cyan", no_wrap=True)
    table.add_column("Endings/Postposition", style="green")
    table.add_column("Function", style="yellow", max_width=50)
    for case_name, case_info in sorted(cases.items()):
        endings = case_info.get("endings") or case_info.get("postposition", "")
        if isinstance(endings, list):
            endings = ", ".join(endings)
        name = (
            f"{case_name}. {case_info.get('name', '')}"
            if key == "cases_hindi"
            else case_name.title()
        )
        table.add_row(name, endings, case_info.get("function", ""))
    console.print(table)


def _display_classifiers_glossary(glossary_data: dict[str, Any], limit: int) -> None:
    """Display Chinese classifiers glossary."""
    classifiers = glossary_data["classifiers"]
    table = Table(title=f"Chinese Measure Words ({len(classifiers)} classifiers)", show_lines=True)
    table.add_column("Classifier", style="cyan", no_wrap=True)
    table.add_column("Pinyin", style="green")
    table.add_column("Category", style="yellow")
    table.add_column("Usage", style="dim", max_width=40)
    for i, (classifier, info) in enumerate(sorted(classifiers.items())):
        if i >= limit:
            break
        table.add_row(
            classifier, info.get("pinyin", ""), info.get("category", ""), info.get("usage", "")
        )
    console.print(table)


def _display_generic_glossary(glossary_data: dict[str, Any], limit: int) -> None:
    """Display generic glossary as key-value table."""
    table = Table(title="Glossary Contents", show_lines=True)
    table.add_column("Key", style="cyan")
    table.add_column("Value", style="green", max_width=80)
    for i, (key, value) in enumerate(sorted(glossary_data.items())):
        if i >= limit:
            break
        value_str = json.dumps(value, ensure_ascii=False) if isinstance(value, dict) else str(value)
        table.add_row(key, value_str[:200])
    console.print(table)


@terminology_app.command("show")
def show_glossary(
    lang: str = typer.Argument(..., help="Language code (e.g., en, ru, zh)"),
    category: str = typer.Argument(..., help="Glossary category (e.g., mqm_core, russian_cases)"),
    limit: int = typer.Option(50, "--limit", "-n", help="Maximum entries to display"),
    format: str = typer.Option("table", "--format", "-f", help="Output format: table or json"),
) -> None:
    """Show contents of a linguistic reference glossary.

    Example:
        kttc terminology show en mqm_core
        kttc terminology show ru russian_cases
        kttc terminology show zh chinese_classifiers --limit 20
        kttc terminology show en mqm_core --format json
    """
    try:
        manager = GlossaryManager()
        glossary_data = manager.load_glossary(lang, category)

        if not glossary_data:
            console.print(
                f"[yellow]No glossary found for language '{lang}' and category '{category}'[/yellow]"
            )
            return

        console.print(f"\n[bold]Terminology Glossary:[/bold] {lang.upper()} - {category}")
        console.print()

        # JSON output format
        if format == "json":
            console.print_json(data=glossary_data)
            return

        # Table output format
        if isinstance(glossary_data, dict):
            if "error_types" in glossary_data:
                _display_mqm_glossary(glossary_data, limit)
            elif "cases" in glossary_data:
                _display_cases_glossary(glossary_data, "cases", "Russian Cases")
            elif "classifiers" in glossary_data:
                _display_classifiers_glossary(glossary_data, limit)
            elif "cases_hindi" in glossary_data:
                _display_cases_glossary(glossary_data, "cases_hindi", "Hindi Cases (कारक)")
            else:
                _display_generic_glossary(glossary_data, limit)

        console.print(f"\n[dim]Showing up to {limit} entries[/dim]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(code=1)


@terminology_app.command("search")
def search_glossaries(
    query: str = typer.Argument(..., help="Search query"),
    lang: str | None = typer.Option(None, "--lang", "-l", help="Filter by language code"),
    case_sensitive: bool = typer.Option(
        False, "--case-sensitive", "-c", help="Case-sensitive search"
    ),
) -> None:
    """Search across all terminology glossaries.

    Searches in error types, definitions, grammar rules, and terminology.

    Example:
        kttc terminology search "mistranslation"
        kttc terminology search "genitive" --lang ru
        kttc terminology search "classifier" --lang zh
    """
    try:
        manager = GlossaryManager()

        # Get available glossaries
        all_glossaries = manager.list_available_glossaries()
        if not all_glossaries:
            console.print("[yellow]No terminology glossaries found[/yellow]")
            return

        # Determine which languages to search
        languages_to_search = [lang] if lang else list(all_glossaries.keys())

        # Collect all results
        all_results = []
        for search_lang in languages_to_search:
            if search_lang not in all_glossaries:
                continue

            try:
                # Use GlossaryManager's search_terms method
                results = manager.search_terms(query, language=search_lang)

                for result in results:
                    # Extract path components to get glossary type
                    path_parts = result["path"].split(" > ")
                    glossary_type = path_parts[0] if path_parts else "unknown"

                    # Format the result data
                    data = result["data"]
                    if isinstance(data, dict):
                        data_str = json.dumps(data, ensure_ascii=False)[:200]
                    else:
                        data_str = str(data)[:200]

                    all_results.append((search_lang, glossary_type, result["path"], data_str))

            except Exception as e:
                # Skip languages that fail to search
                console.print(f"[dim]Warning: Could not search in {search_lang}: {e}[/dim]")
                continue

        if not all_results:
            console.print(f"[yellow]No results found for '{query}'[/yellow]")
            return

        # Display results
        table = Table(
            title=f"Search Results for '{query}' ({len(all_results)} matches)", show_lines=True
        )
        table.add_column("Language", style="cyan", no_wrap=True)
        table.add_column("Glossary", style="green")
        table.add_column("Path", style="yellow", max_width=30)
        table.add_column("Match", style="white", max_width=50)

        # Limit to 50 results
        for search_lang, glossary_type, path, data_str in all_results[:50]:
            table.add_row(
                search_lang.upper(), glossary_type.replace("_", " ").title(), path, data_str
            )

        console.print(table)

        if len(all_results) > 50:
            console.print(f"\n[dim]Showing first 50 of {len(all_results)} results[/dim]")
        else:
            console.print(f"\n[dim]Total: {len(all_results)} results[/dim]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(code=1)


@terminology_app.command("validate-error")
def validate_error_type(
    error_type: str = typer.Argument(..., help="MQM error type to validate"),
    lang: str = typer.Option("en", "--lang", "-l", help="Language code"),
) -> None:
    """Validate an MQM error type against the terminology glossary.

    Checks if the error type is valid according to MQM standards and
    provides detailed information about the error type.

    Example:
        kttc terminology validate-error mistranslation
        kttc terminology validate-error grammar --lang ru
        kttc terminology validate-error untranslated --lang en
    """
    try:
        validator = TermValidator()

        console.print(f"\n[cyan]Validating MQM error type:[/cyan] {error_type}")
        console.print(f"[dim]Language:[/dim] {lang.upper()}\n")

        # Validate error type
        is_valid, error_info = validator.validate_mqm_error_type(error_type, lang)

        if is_valid and error_info:
            # Display success panel
            panel_content = Text()
            panel_content.append("✓ Valid MQM Error Type\n\n", style="bold green")

            if "definition" in error_info:
                panel_content.append("Definition:\n", style="bold")
                panel_content.append(f"{error_info['definition']}\n\n", style="white")

            if "severity" in error_info:
                panel_content.append("Default Severity: ", style="bold")
                panel_content.append(f"{error_info['severity']}\n\n", style="yellow")

            if "examples" in error_info:
                panel_content.append("Examples:\n", style="bold")
                for example in error_info["examples"]:
                    panel_content.append(f"  • {example}\n", style="dim")

            console.print(
                Panel(panel_content, title="MQM Error Type Information", border_style="green")
            )

        else:
            # Display error panel
            panel_content = Text()
            panel_content.append("✗ Invalid MQM Error Type\n\n", style="bold red")
            panel_content.append(
                f"The error type '{error_type}' is not recognized in the MQM glossary "
                f"for language '{lang}'.\n\n",
                style="white",
            )
            panel_content.append("Common MQM error types:\n", style="bold")
            panel_content.append("  • mistranslation\n", style="dim")
            panel_content.append("  • omission\n", style="dim")
            panel_content.append("  • addition\n", style="dim")
            panel_content.append("  • grammar\n", style="dim")
            panel_content.append("  • spelling\n", style="dim")
            panel_content.append("  • untranslated\n", style="dim")

            console.print(Panel(panel_content, title="Validation Result", border_style="red"))

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(code=1)


@terminology_app.command("validators")
def list_validators() -> None:
    """List available language-specific validators.

    Shows validators for grammar checking, measure words, cases, etc.

    Example:
        kttc terminology validators
    """
    try:
        console.print("\n[bold]Available Language Validators:[/bold]\n")

        validators_info = [
            (
                "Russian",
                "RussianCaseAspectValidator",
                "6 cases, perfective/imperfective aspect",
                "ru",
            ),
            (
                "Chinese",
                "ChineseMeasureWordValidator",
                "150+ classifiers (量词) in 6 categories",
                "zh",
            ),
            ("Hindi", "HindiPostpositionValidator", "8 cases (कारक), ergative construction", "hi"),
            (
                "Persian",
                "PersianEzafeValidator",
                "Ezafe (اضافه), compound verbs, SOV order",
                "fa",
            ),
        ]

        table = Table(show_lines=True)
        table.add_column("Language", style="cyan", no_wrap=True)
        table.add_column("Validator Class", style="green")
        table.add_column("Features", style="yellow", max_width=50)
        table.add_column("Code", style="dim")

        for lang, validator_class, features, code in validators_info:
            table.add_row(lang, validator_class, features, code)

        console.print(table)

        # Show usage example
        console.print("\n[bold]Usage Example:[/bold]")
        console.print("[dim]from kttc.terminology import RussianCaseAspectValidator[/dim]")
        console.print("[dim]validator = RussianCaseAspectValidator()[/dim]")
        console.print("[dim]case_info = validator.get_case_info('genitive')[/dim]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(code=1)


def _search_in_dict(
    data: dict[str, Any] | list[Any] | str, query: str, case_sensitive: bool, path: str = ""
) -> list[tuple[str, str]]:
    """Recursively search for query in nested dictionary.

    Args:
        data: Data to search (dict, list, or str)
        query: Search query
        case_sensitive: Whether to perform case-sensitive search
        path: Current path in the data structure

    Returns:
        List of (key_path, matched_value) tuples
    """
    results = []

    if isinstance(data, dict):
        for key, value in data.items():
            new_path = f"{path}.{key}" if path else key

            # Check if key matches
            key_str = key if case_sensitive else key.lower()
            if query in key_str:
                results.append((new_path, json.dumps(value, ensure_ascii=False)[:200]))

            # Recursively search value
            results.extend(_search_in_dict(value, query, case_sensitive, new_path))

    elif isinstance(data, list):
        for idx, item in enumerate(data):
            new_path = f"{path}[{idx}]"
            results.extend(_search_in_dict(item, query, case_sensitive, new_path))

    elif isinstance(data, str):
        # Search in string value
        search_str = data if case_sensitive else data.lower()
        if query in search_str:
            results.append((path, data[:200]))

    return results
