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

"""Multi-project glossary system for translation terminology management.

Supports flexible glossary management with:
- Multiple glossaries per project
- Precedence rules when glossaries conflict
- Git-friendly JSON/CSV formats
- CLI integration for easy management
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class TermEntry:
    """Single terminology entry.

    Represents a term translation with metadata and constraints.

    Attributes:
        source: Source term
        target: Target translation
        source_lang: Source language code (ISO 639-1)
        target_lang: Target language code (ISO 639-1)
        domain: Domain category (technical, medical, legal, etc.)
        definition: Term definition or explanation
        context: Usage context or example
        case_sensitive: Whether term matching is case-sensitive
        do_not_translate: If True, term should not be translated
        pos: Part of speech (noun, verb, adjective, etc.)
        variants: List of term variants (plural forms, etc.)
        notes: Additional notes for translators
        status: Term status (draft, approved, deprecated)
    """

    source: str
    target: str
    source_lang: str
    target_lang: str
    domain: str | None = None
    definition: str | None = None
    context: str | None = None
    case_sensitive: bool = False
    do_not_translate: bool = False
    pos: str | None = None
    variants: list[dict[str, str]] | None = None
    notes: str | None = None
    status: str = "approved"

    def matches(self, text: str) -> bool:
        """Check if term appears in text.

        Args:
            text: Text to search in

        Returns:
            True if term is found in text
        """
        if self.case_sensitive:
            return self.source in text
        else:
            return self.source.lower() in text.lower()


@dataclass
class GlossaryMetadata:
    """Glossary metadata.

    Contains descriptive information about the glossary.

    Attributes:
        name: Glossary name
        version: Version string (e.g., "1.2.0")
        domain: Primary domain category
        language_pair: Language pair (e.g., "en-ru")
        created_by: Creator email or name
        created_at: Creation timestamp
        description: Human-readable description
    """

    name: str
    version: str = "1.0.0"
    domain: str | None = None
    language_pair: str | None = None
    created_by: str | None = None
    created_at: str | None = None
    description: str | None = None


class Glossary:
    """Single glossary with terminology entries.

    Manages a collection of term entries with efficient lookup.

    Example:
        >>> glossary = Glossary.from_json(Path("medical.json"))
        >>> entry = glossary.lookup("adverse event", "en", "ru")
        >>> if entry:
        ...     print(entry.target)  # "нежелательное явление"
    """

    def __init__(
        self,
        entries: list[TermEntry],
        metadata: GlossaryMetadata | None = None,
        name: str = "default",
    ):
        """Initialize glossary.

        Args:
            entries: List of term entries
            metadata: Glossary metadata (optional)
            name: Glossary name
        """
        self.name = name
        self.entries = entries
        self.metadata = metadata or GlossaryMetadata(name=name)

        # Build lookup indices
        self._build_indices()

    def _build_indices(self) -> None:
        """Build efficient lookup indices."""
        self.by_source: dict[str, TermEntry] = {}
        self.by_lang_pair: dict[str, list[TermEntry]] = {}

        for entry in self.entries:
            # Index by source term (case-insensitive by default)
            key = entry.source if entry.case_sensitive else entry.source.lower()
            self.by_source[key] = entry

            # Index by language pair
            lang_pair = f"{entry.source_lang}-{entry.target_lang}"
            if lang_pair not in self.by_lang_pair:
                self.by_lang_pair[lang_pair] = []
            self.by_lang_pair[lang_pair].append(entry)

    def lookup(
        self,
        source_term: str,
        source_lang: str,
        target_lang: str,
        case_sensitive: bool | None = None,
    ) -> TermEntry | None:
        """Look up term translation.

        Args:
            source_term: Source term to look up
            source_lang: Source language code
            target_lang: Target language code
            case_sensitive: Override case sensitivity (optional)

        Returns:
            TermEntry if found, None otherwise

        Example:
            >>> entry = glossary.lookup("API", "en", "ru")
            >>> if entry:
            ...     print(entry.target)  # "API"
        """
        # Try both case-sensitive and case-insensitive lookups
        # First try exact match (case-sensitive)
        entry = self.by_source.get(source_term)
        if entry and entry.source_lang == source_lang and entry.target_lang == target_lang:
            return entry

        # Then try case-insensitive match
        entry = self.by_source.get(source_term.lower())
        if entry and entry.source_lang == source_lang and entry.target_lang == target_lang:
            return entry

        return None

    def get_all_for_language_pair(self, source_lang: str, target_lang: str) -> list[TermEntry]:
        """Get all terms for language pair.

        Args:
            source_lang: Source language code
            target_lang: Target language code

        Returns:
            List of term entries for language pair
        """
        lang_pair = f"{source_lang}-{target_lang}"
        return self.by_lang_pair.get(lang_pair, [])

    def find_in_text(self, text: str, source_lang: str, target_lang: str) -> list[TermEntry]:
        """Find all glossary terms present in text.

        Args:
            text: Text to search in
            source_lang: Source language
            target_lang: Target language

        Returns:
            List of term entries found in text
        """
        terms = self.get_all_for_language_pair(source_lang, target_lang)
        found = []

        for term in terms:
            if term.matches(text):
                found.append(term)

        return found

    @classmethod
    def from_json(cls, path: Path) -> Glossary:
        """Load glossary from JSON file.

        Args:
            path: Path to JSON file

        Returns:
            Glossary instance

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If JSON is invalid
        """
        if not path.exists():
            raise FileNotFoundError(f"Glossary file not found: {path}")

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        metadata_dict = data.get("metadata", {})
        metadata = GlossaryMetadata(**metadata_dict) if metadata_dict else None

        entries = [TermEntry(**e) for e in data.get("entries", [])]

        return cls(entries=entries, metadata=metadata, name=path.stem)

    @classmethod
    def from_csv(cls, path: Path) -> Glossary:
        """Load glossary from CSV file.

        CSV Format:
            source,target,source_lang,target_lang,domain,case_sensitive,do_not_translate,notes

        Args:
            path: Path to CSV file

        Returns:
            Glossary instance

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If CSV is invalid
        """
        if not path.exists():
            raise FileNotFoundError(f"Glossary file not found: {path}")

        entries = []

        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f)

            # Validate required columns
            required = {"source", "target", "source_lang", "target_lang"}
            if not required.issubset(set(reader.fieldnames or [])):
                missing = required - set(reader.fieldnames or [])
                raise ValueError(f"CSV missing required columns: {missing}")

            for row in reader:
                # Convert string booleans
                row["case_sensitive"] = row.get("case_sensitive", "false").lower() == "true"
                row["do_not_translate"] = row.get("do_not_translate", "false").lower() == "true"

                # Remove empty fields
                row = {k: v for k, v in row.items() if v}

                try:
                    entries.append(TermEntry(**row))  # type: ignore[arg-type]
                except TypeError as e:
                    logger.warning(f"Skipping invalid CSV row: {e}")

        return cls(entries=entries, name=path.stem)

    def to_json(self, path: Path) -> None:
        """Save glossary to JSON file.

        Args:
            path: Output file path
        """
        data = {
            "metadata": asdict(self.metadata) if self.metadata else {},
            "entries": [asdict(e) for e in self.entries],
        }

        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def to_csv(self, path: Path) -> None:
        """Save glossary to CSV file.

        Args:
            path: Output file path
        """
        if not self.entries:
            logger.warning(f"No entries to save to {path}")
            return

        path.parent.mkdir(parents=True, exist_ok=True)

        # Get all field names from first entry
        fieldnames = list(asdict(self.entries[0]).keys())

        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for entry in self.entries:
                entry_dict = asdict(entry)
                # Convert None to empty string
                entry_dict = {k: v if v is not None else "" for k, v in entry_dict.items()}
                writer.writerow(entry_dict)


class GlossaryManager:
    """Manage multiple glossaries with precedence rules.

    Handles loading, searching, and managing glossaries across
    project and user directories.

    Precedence (highest to lowest):
    1. Project-local glossaries (./glossaries/)
    2. User glossaries (~/.kttc/glossaries/)

    Example:
        >>> manager = GlossaryManager()
        >>> manager.load_multiple(["base", "medical", "client-acme"])
        >>> entry = manager.lookup("API", "en", "ru")
    """

    def __init__(self) -> None:
        """Initialize glossary manager."""
        self.glossaries: list[Glossary] = []
        self.user_glossary_dir = Path.home() / ".kttc" / "glossaries"
        self.project_glossary_dir = Path.cwd() / "glossaries"

        # Ensure user directory exists
        self.user_glossary_dir.mkdir(parents=True, exist_ok=True)

    def load_glossary(self, name: str) -> Glossary:
        """Load glossary by name.

        Searches in order:
        1. Project directory (./glossaries/)
        2. User directory (~/.kttc/glossaries/)

        Args:
            name: Glossary name (without extension)

        Returns:
            Glossary instance

        Raises:
            FileNotFoundError: If glossary not found
        """
        # Try project directory first, then user directory
        for base_dir in [self.project_glossary_dir, self.user_glossary_dir]:
            for ext in [".json", ".csv"]:
                path = base_dir / f"{name}{ext}"
                if path.exists():
                    if ext == ".json":
                        return Glossary.from_json(path)
                    else:
                        return Glossary.from_csv(path)

        raise FileNotFoundError(
            f"Glossary '{name}' not found in:\n"
            f"  - {self.project_glossary_dir}\n"
            f"  - {self.user_glossary_dir}"
        )

    def load_multiple(self, names: list[str]) -> None:
        """Load multiple glossaries in order.

        Later glossaries have higher precedence in lookups.

        Args:
            names: List of glossary names

        Example:
            >>> manager.load_multiple(["base", "medical", "client-acme"])
            # client-acme has highest precedence
        """
        self.glossaries = []
        for name in names:
            try:
                glossary = self.load_glossary(name)
                self.glossaries.append(glossary)
                logger.info(f"Loaded glossary '{name}' ({len(glossary.entries)} terms)")
            except FileNotFoundError as e:
                logger.warning(f"Failed to load glossary '{name}': {e}")

    def lookup(self, source_term: str, source_lang: str, target_lang: str) -> TermEntry | None:
        """Look up term across all loaded glossaries.

        Returns first match from highest precedence glossary.

        Args:
            source_term: Source term
            source_lang: Source language code
            target_lang: Target language code

        Returns:
            TermEntry if found, None otherwise
        """
        # Search in reverse order (highest precedence first)
        for glossary in reversed(self.glossaries):
            entry = glossary.lookup(source_term, source_lang, target_lang)
            if entry:
                return entry

        return None

    def get_all_terms(self, source_lang: str, target_lang: str) -> list[TermEntry]:
        """Get all terms for language pair from all glossaries.

        Deduplicates with precedence (later glossaries override earlier).

        Args:
            source_lang: Source language code
            target_lang: Target language code

        Returns:
            List of deduplicated term entries
        """
        terms_by_source: dict[str, TermEntry] = {}

        # Process in order (later glossaries override)
        for glossary in self.glossaries:
            for term in glossary.get_all_for_language_pair(source_lang, target_lang):
                key = term.source if term.case_sensitive else term.source.lower()
                terms_by_source[key] = term

        return list(terms_by_source.values())

    def find_in_text(self, text: str, source_lang: str, target_lang: str) -> list[TermEntry]:
        """Find all glossary terms present in text.

        Args:
            text: Text to search in
            source_lang: Source language
            target_lang: Target language

        Returns:
            List of term entries found in text (deduplicated)
        """
        all_terms = self.get_all_terms(source_lang, target_lang)
        found = []

        for term in all_terms:
            if term.matches(text):
                found.append(term)

        return found

    def list_available(self) -> list[tuple[str, Path, int]]:
        """List all available glossaries.

        Returns:
            List of (name, path, term_count) tuples
        """
        glossaries: dict[str, tuple[Path, int]] = {}

        for base_dir in [self.project_glossary_dir, self.user_glossary_dir]:
            if not base_dir.exists():
                continue

            for path in base_dir.glob("*.json"):
                if path.stem not in glossaries:
                    try:
                        glossary = Glossary.from_json(path)
                        glossaries[path.stem] = (path, len(glossary.entries))
                    except Exception as e:
                        logger.warning(f"Failed to load {path}: {e}")

            for path in base_dir.glob("*.csv"):
                if path.stem not in glossaries:
                    try:
                        glossary = Glossary.from_csv(path)
                        glossaries[path.stem] = (path, len(glossary.entries))
                    except Exception as e:
                        logger.warning(f"Failed to load {path}: {e}")

        return [(name, path, count) for name, (path, count) in sorted(glossaries.items())]

    def merge_glossaries(self, names: list[str], output_name: str) -> Glossary:
        """Merge multiple glossaries into one.

        Later glossaries override earlier ones for duplicate terms.

        Args:
            names: List of glossary names to merge
            output_name: Name for merged glossary

        Returns:
            Merged glossary

        Example:
            >>> merged = manager.merge_glossaries(["base", "medical"], "combined")
            >>> merged.to_json(Path("combined.json"))
        """
        all_terms: dict[tuple[str, str, str], TermEntry] = {}

        for name in names:
            try:
                glossary = self.load_glossary(name)
                for entry in glossary.entries:
                    # Key: (source, source_lang, target_lang)
                    key = (entry.source.lower(), entry.source_lang, entry.target_lang)
                    all_terms[key] = entry
            except FileNotFoundError:
                logger.warning(f"Glossary '{name}' not found, skipping")

        merged = Glossary(
            entries=list(all_terms.values()),
            metadata=GlossaryMetadata(
                name=output_name,
                description=f"Merged from: {', '.join(names)}",
            ),
            name=output_name,
        )

        logger.info(
            f"Merged {len(names)} glossaries into '{output_name}' ({len(merged.entries)} terms)"
        )

        return merged
