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

"""Glossary management CLI commands."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

from kttc.core import Glossary, GlossaryManager

console = Console()

# Help text constants
GLOSSARY_NAME_HELP = "Glossary name"

# Create glossary subcommand app
glossary_app = typer.Typer(
    name="glossary",
    help="Manage translation glossaries",
    no_args_is_help=True,
)


@glossary_app.command("list")
def list_glossaries() -> None:
    """List all available glossaries.

    Shows glossaries from both project and user directories.

    Example:
        kttc glossary list
    """
    try:
        manager = GlossaryManager()
        glossaries = manager.list_available()

        if not glossaries:
            console.print("[yellow]No glossaries found[/yellow]")
            console.print("\nCreate a glossary with:")
            console.print("  kttc glossary create <name> --from-csv <file>")
            return

        table = Table(title="Available Glossaries", show_lines=True)
        table.add_column("Name", style="cyan", no_wrap=True)
        table.add_column("Location", style="green")
        table.add_column("Terms", style="yellow", justify="right")
        table.add_column("Path", style="dim")

        for name, path, term_count in glossaries:
            # Determine location
            if path.is_relative_to(Path.cwd()):
                location = "project"
                location_style = "green"
            else:
                location = "user"
                location_style = "blue"

            table.add_row(
                name, f"[{location_style}]{location}[/{location_style}]", str(term_count), str(path)
            )

        console.print(table)
        console.print(f"\n[dim]Total: {len(glossaries)} glossaries[/dim]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(code=1)


def _filter_by_lang_pair(entries: list[Any], lang_pair: str | None) -> list[Any]:
    """Filter entries by language pair."""
    if not lang_pair:
        return entries
    try:
        src, tgt = lang_pair.split("-")
        filtered = [e for e in entries if e.source_lang == src and e.target_lang == tgt]
        console.print(f"[dim]Filtered to {src}→{tgt}[/dim]\n")
        return filtered
    except ValueError as exc:
        console.print("[red]Error: Invalid language pair format. Use 'en-ru' format[/red]")
        raise typer.Exit(code=1) from exc


def _build_entry_row(entry: Any) -> tuple[str, str, str, str, str]:
    """Build table row for glossary entry."""
    lang = f"{entry.source_lang}→{entry.target_lang}"
    domain = entry.domain or "-"
    notes = entry.notes or ""
    markers = []
    if entry.do_not_translate:
        markers.append("[red]DNT[/red]")
    if entry.case_sensitive:
        markers.append("[yellow]CS[/yellow]")
    if markers:
        notes = f"{' '.join(markers)} {notes}".strip()
    return entry.source, entry.target, lang, domain, notes


@glossary_app.command("show")
def show_glossary(
    name: str = typer.Argument(..., help=GLOSSARY_NAME_HELP),
    lang_pair: str | None = typer.Option(
        None, "--lang-pair", "-l", help="Filter by language pair (e.g., en-ru)"
    ),
    limit: int = typer.Option(20, "--limit", "-n", help="Maximum terms to display"),
) -> None:
    """Show glossary contents.

    Display terms from a specific glossary with optional filtering.

    Example:
        kttc glossary show medical
        kttc glossary show technical --lang-pair en-ru
        kttc glossary show base --limit 50
    """
    try:
        manager = GlossaryManager()
        glossary = manager.load_glossary(name)

        console.print(f"\n[bold]Glossary: {name}[/bold]")

        # Show metadata if available
        if glossary.metadata:
            console.print(f"[dim]Version: {glossary.metadata.version}[/dim]")
            if glossary.metadata.description:
                console.print(f"[dim]Description: {glossary.metadata.description}[/dim]")
        console.print()

        entries = _filter_by_lang_pair(glossary.entries, lang_pair)

        if not entries:
            console.print("[yellow]No terms found[/yellow]")
            return

        if len(entries) > limit:
            console.print(f"[yellow]Showing first {limit} of {len(entries)} terms[/yellow]\n")
            entries = entries[:limit]

        table = Table(show_header=True, show_lines=False)
        table.add_column("Source", style="cyan", max_width=30)
        table.add_column("Target", style="green", max_width=30)
        table.add_column("Lang", style="yellow", justify="center")
        table.add_column("Domain", style="blue", max_width=15)
        table.add_column("Notes", style="dim", max_width=40)

        for entry in entries:
            table.add_row(*_build_entry_row(entry))

        console.print(table)
        console.print(f"\n[dim]Total terms shown: {len(entries)}[/dim]")

    except FileNotFoundError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(code=1)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(code=1)


@glossary_app.command("create")
def create_glossary(
    name: str = typer.Argument(..., help=GLOSSARY_NAME_HELP),
    from_csv: Path | None = typer.Option(None, "--from-csv", help="Create from CSV file"),
    from_json: Path | None = typer.Option(None, "--from-json", help="Create from JSON file"),
    user: bool = typer.Option(False, "--user", help="Save to user directory (~/.kttc/glossaries)"),
) -> None:
    """Create new glossary from file.

    Example:
        kttc glossary create my-project --from-csv terms.csv
        kttc glossary create medical --from-json medical.json --user
    """
    try:
        # Load source file
        if from_csv:
            if not from_csv.exists():
                console.print(f"[red]Error: File not found: {from_csv}[/red]")
                raise typer.Exit(code=1)

            console.print(f"[cyan]Loading CSV file:[/cyan] {from_csv}")
            glossary = Glossary.from_csv(from_csv)
        elif from_json:
            if not from_json.exists():
                console.print(f"[red]Error: File not found: {from_json}[/red]")
                raise typer.Exit(code=1)

            console.print(f"[cyan]Loading JSON file:[/cyan] {from_json}")
            glossary = Glossary.from_json(from_json)
        else:
            console.print("[red]Error: Must specify --from-csv or --from-json[/red]")
            raise typer.Exit(code=1)

        # Determine output path
        if user:
            base_dir = Path.home() / ".kttc" / "glossaries"
        else:
            base_dir = Path.cwd() / "glossaries"

        base_dir.mkdir(parents=True, exist_ok=True)
        output_path = base_dir / f"{name}.json"

        # Save
        glossary.to_json(output_path)

        console.print(
            f"[green]✓[/green] Created glossary '{name}' with {len(glossary.entries)} terms"
        )
        console.print(f"[dim]Location: {output_path}[/dim]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(code=1)


@glossary_app.command("merge")
def merge_glossaries(
    glossaries: list[str] = typer.Argument(..., help="Glossaries to merge"),
    output: str = typer.Option(..., "--output", "-o", help="Output glossary name"),
    user: bool = typer.Option(False, "--user", help="Save to user directory (~/.kttc/glossaries)"),
) -> None:
    """Merge multiple glossaries into one.

    Later glossaries override earlier ones for duplicate terms.

    Example:
        kttc glossary merge base medical --output combined
        kttc glossary merge technical client-a --output project-terms
    """
    try:
        manager = GlossaryManager()

        console.print(f"[cyan]Merging glossaries:[/cyan] {', '.join(glossaries)}")

        # Merge
        merged = manager.merge_glossaries(glossaries, output)

        # Determine output path
        if user:
            base_dir = Path.home() / ".kttc" / "glossaries"
        else:
            base_dir = Path.cwd() / "glossaries"

        base_dir.mkdir(parents=True, exist_ok=True)
        output_path = base_dir / f"{output}.json"

        # Save
        merged.to_json(output_path)

        console.print(
            f"[green]✓[/green] Merged {len(glossaries)} glossaries "
            f"into '{output}' ({len(merged.entries)} terms)"
        )
        console.print(f"[dim]Location: {output_path}[/dim]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(code=1)


@glossary_app.command("export")
def export_glossary(
    name: str = typer.Argument(..., help=GLOSSARY_NAME_HELP),
    output_format: str = typer.Option("csv", "--format", "-f", help="Output format (csv or json)"),
    output: Path | None = typer.Option(None, "--output", "-o", help="Output file path"),
) -> None:
    """Export glossary to different format.

    Example:
        kttc glossary export medical --format csv --output medical.csv
        kttc glossary export technical --format json --output technical.json
    """
    try:
        manager = GlossaryManager()
        glossary = manager.load_glossary(name)

        # Determine output path
        if output:
            output_path = output
        else:
            ext = ".csv" if output_format == "csv" else ".json"
            output_path = Path(f"{name}{ext}")

        # Export
        if output_format == "csv":
            glossary.to_csv(output_path)
        elif output_format == "json":
            glossary.to_json(output_path)
        else:
            console.print(f"[red]Error: Unsupported format: {output_format}[/red]")
            raise typer.Exit(code=1)

        console.print(f"[green]✓[/green] Exported '{name}' to {output_format.upper()} format")
        console.print(f"[dim]Output: {output_path}[/dim]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(code=1)


def _find_duplicate_entries(entries: list[Any]) -> list[str]:
    """Find duplicate entries in glossary."""
    issues = []
    seen: dict[tuple[str, str, str], list[int]] = {}
    for idx, entry in enumerate(entries):
        key = (entry.source.lower(), entry.source_lang, entry.target_lang)
        if key not in seen:
            seen[key] = []
        seen[key].append(idx + 1)

    duplicates = {k: v for k, v in seen.items() if len(v) > 1}
    if duplicates:
        issues.append(f"Found {len(duplicates)} duplicate terms")
        for (source, src_lang, tgt_lang), indices in list(duplicates.items())[:5]:
            issues.append(
                f"  • '{source}' ({src_lang}→{tgt_lang}) at positions: {', '.join(map(str, indices))}"
            )
    return issues


def _find_empty_entries(entries: list[Any]) -> list[str]:
    """Find entries with empty fields."""
    issues = []
    empty_sources = sum(1 for e in entries if not e.source.strip())
    empty_targets = sum(1 for e in entries if not e.target.strip())
    if empty_sources:
        issues.append(f"Found {empty_sources} entries with empty source terms")
    if empty_targets:
        issues.append(f"Found {empty_targets} entries with empty target translations")
    return issues


@glossary_app.command("validate")
def validate_glossary(file: Path = typer.Argument(..., help="Glossary file to validate")) -> None:
    """Validate glossary file format.

    Checks for:
    - Valid JSON/CSV format
    - Required fields present
    - No duplicate terms
    - Valid language codes

    Example:
        kttc glossary validate medical.json
        kttc glossary validate terms.csv
    """
    try:
        console.print(f"[cyan]Validating glossary:[/cyan] {file}")

        # Try to load
        if file.suffix == ".json":
            glossary = Glossary.from_json(file)
        elif file.suffix == ".csv":
            glossary = Glossary.from_csv(file)
        else:
            console.print(f"[red]Error: Unsupported format: {file.suffix}[/red]")
            raise typer.Exit(code=1)

        # Validation checks
        issues = _find_duplicate_entries(glossary.entries)
        issues.extend(_find_empty_entries(glossary.entries))

        # Display results
        console.print()
        if issues:
            console.print("[yellow]⚠ Validation issues found:[/yellow]\n")
            for issue in issues:
                console.print(f"  {issue}")
            console.print()
            console.print(
                f"[yellow]Total entries: {len(glossary.entries)} (issues: {len(issues)})[/yellow]"
            )
        else:
            console.print("[green]✓ Glossary is valid[/green]")
            console.print(f"[dim]Total entries: {len(glossary.entries)}[/dim]")

    except Exception as e:
        console.print(f"[red]✗ Validation failed: {e}[/red]")
        raise typer.Exit(code=1)
