"""Glossary Manager for loading and accessing multi-lingual terminology."""

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel


class GlossaryMetadata(BaseModel):
    """Metadata for a glossary file."""

    language: str
    language_name: str
    glossary_type: str
    version: str
    created: str
    description: str
    total_terms: int | None = None
    sources: list[str] | None = None


class GlossaryManager:
    """
    Manager for loading and accessing multi-lingual glossaries.

    Supports loading MQM terminology, NLP terms, and language-specific
    grammar glossaries for English, Russian, Chinese, Persian, and Hindi.
    """

    def __init__(self, glossaries_dir: Path | None = None):
        """
        Initialize the Glossary Manager.

        Args:
            glossaries_dir: Path to glossaries directory.
                           Defaults to <project_root>/glossaries/
        """
        if glossaries_dir is None:
            # Default to project root/glossaries
            current_file = Path(__file__)
            project_root = current_file.parent.parent.parent.parent
            glossaries_dir = project_root / "glossaries"

        self.glossaries_dir = Path(glossaries_dir)
        self._glossaries: dict[str, dict[str, Any]] = {}
        self._metadata: dict[str, GlossaryMetadata] = {}

    def load_glossary(self, language: str, glossary_type: str) -> dict[str, Any]:
        """
        Load a specific glossary file.

        Args:
            language: Language code (e.g., 'en', 'ru', 'zh', 'fa', 'hi')
            glossary_type: Type of glossary (e.g., 'mqm_core', 'nlp_terms')

        Returns:
            Dictionary containing the glossary data

        Raises:
            FileNotFoundError: If glossary file doesn't exist
            json.JSONDecodeError: If glossary file is invalid JSON
        """
        glossary_path = self.glossaries_dir / language / f"{glossary_type}.json"

        if not glossary_path.exists():
            raise FileNotFoundError(
                f"Glossary not found: {glossary_path}\n"
                f"Available glossaries in {self.glossaries_dir / language}: "
                f"{list((self.glossaries_dir / language).glob('*.json')) if (self.glossaries_dir / language).exists() else 'directory does not exist'}"
            )

        with open(glossary_path, encoding="utf-8") as f:
            data: dict[str, Any] = json.load(f)

        # Store metadata
        key = f"{language}:{glossary_type}"
        if "metadata" in data:
            self._metadata[key] = GlossaryMetadata(**data["metadata"])

        # Cache glossary
        self._glossaries[key] = data

        return data

    def load_all_for_language(self, language: str) -> dict[str, dict[str, Any]]:
        """
        Load all available glossaries for a specific language.

        Args:
            language: Language code (e.g., 'en', 'ru', 'zh')

        Returns:
            Dictionary mapping glossary types to glossary data
        """
        lang_dir = self.glossaries_dir / language

        if not lang_dir.exists():
            raise FileNotFoundError(f"No glossaries directory for language: {language}")

        glossaries = {}
        for glossary_file in lang_dir.glob("*.json"):
            glossary_type = glossary_file.stem
            try:
                glossaries[glossary_type] = self.load_glossary(language, glossary_type)
            except (FileNotFoundError, json.JSONDecodeError) as e:
                print(f"Warning: Failed to load {glossary_file}: {e}")

        return glossaries

    def get_mqm_error_dimensions(self, language: str = "en") -> list[dict[str, Any]]:
        """
        Get MQM error dimensions for a language.

        Args:
            language: Language code (default: 'en')

        Returns:
            List of MQM error dimension dictionaries
        """
        mqm_glossary = self.load_glossary(language, "mqm_core")
        result: list[dict[str, Any]] = mqm_glossary.get("error_dimensions", [])
        return result

    def get_severity_levels(self, language: str = "en") -> dict[str, Any]:
        """
        Get MQM severity levels for a language.

        Args:
            language: Language code (default: 'en')

        Returns:
            Dictionary of severity levels
        """
        mqm_glossary = self.load_glossary(language, "mqm_core")
        result: dict[str, Any] = mqm_glossary.get("severity_levels", {})
        return result

    def get_nlp_term(
        self, term: str, language: str = "en", category: str | None = None
    ) -> dict[str, Any] | None:
        """
        Look up an NLP term in the glossary.

        Args:
            term: Term to look up
            language: Language code
            category: Optional category to search in

        Returns:
            Term definition or None if not found
        """
        try:
            nlp_glossary = self.load_glossary(language, "nlp_terms")
        except FileNotFoundError:
            return None

        # Search in specified category or all categories
        if category and category in nlp_glossary:
            result: dict[str, Any] | None = nlp_glossary[category].get(term)
            return result

        # Search all categories
        for cat_data in nlp_glossary.values():
            if isinstance(cat_data, dict) and term in cat_data:
                result_val: dict[str, Any] = cat_data[term]
                return result_val

        return None

    def search_terms(
        self, query: str, language: str = "en", glossary_type: str | None = None
    ) -> list[dict[str, Any]]:
        """
        Search for terms matching a query string.

        Args:
            query: Search query
            language: Language code
            glossary_type: Optional specific glossary to search

        Returns:
            List of matching terms with their definitions
        """
        results: list[dict[str, Any]] = []
        query_lower = query.lower()

        # Determine which glossaries to search
        if glossary_type:
            glossaries_to_search = {glossary_type: self.load_glossary(language, glossary_type)}
        else:
            glossaries_to_search = self.load_all_for_language(language)

        for gtype, glossary in glossaries_to_search.items():
            self._search_in_dict(glossary, query_lower, results, path=[gtype])

        return results

    def _search_in_dict(
        self,
        data: Any,
        query: str,
        results: list[dict[str, Any]],
        path: list[str],
    ) -> None:
        """Recursively search dictionary for matching terms."""
        if not isinstance(data, dict):
            return

        for key, value in data.items():
            current_path = path + [key]
            self._check_key_match(key, value, query, results, current_path)
            self._check_value_match(value, query, results, current_path)
            self._recurse_into_value(value, query, results, current_path)

    @staticmethod
    def _check_key_match(
        key: str,
        value: Any,
        query: str,
        results: list[dict[str, Any]],
        path: list[str],
    ) -> None:
        """Check if dictionary key matches the query."""
        if query in key.lower():
            results.append({"path": " > ".join(path), "data": value})

    @staticmethod
    def _check_value_match(
        value: Any,
        query: str,
        results: list[dict[str, Any]],
        path: list[str],
    ) -> None:
        """Check if string value matches the query."""
        if isinstance(value, str) and query in value.lower():
            results.append({"path": " > ".join(path), "data": value})

    def _recurse_into_value(
        self,
        value: Any,
        query: str,
        results: list[dict[str, Any]],
        path: list[str],
    ) -> None:
        """Recurse into nested dictionaries or lists."""
        if isinstance(value, dict):
            self._search_in_dict(value, query, results, path)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    self._search_in_dict(item, query, results, path)

    def get_metadata(self, language: str, glossary_type: str) -> GlossaryMetadata | None:
        """
        Get metadata for a specific glossary.

        Args:
            language: Language code
            glossary_type: Type of glossary

        Returns:
            GlossaryMetadata or None if not loaded
        """
        key = f"{language}:{glossary_type}"
        return self._metadata.get(key)

    def list_available_glossaries(self) -> dict[str, list[str]]:
        """
        List all available glossaries by language.

        Returns:
            Dictionary mapping language codes to lists of available glossary types
        """
        available = {}

        for lang_dir in self.glossaries_dir.iterdir():
            if lang_dir.is_dir() and not lang_dir.name.startswith("."):
                lang_code = lang_dir.name
                glossaries = [f.stem for f in lang_dir.glob("*.json")]
                if glossaries:
                    available[lang_code] = glossaries

        return available

    def get_term_count(self, language: str, glossary_type: str) -> int:
        """
        Get the total number of terms in a glossary.

        Args:
            language: Language code
            glossary_type: Type of glossary

        Returns:
            Number of terms
        """
        metadata = self.get_metadata(language, glossary_type)
        if metadata and metadata.total_terms:
            return metadata.total_terms

        # Count terms if not in metadata
        glossary = self.load_glossary(language, glossary_type)
        count = self._count_terms(glossary)
        return count

    def _count_terms(self, data: Any) -> int:
        """Recursively count dictionary entries as terms."""
        if isinstance(data, dict):
            # Don't count metadata
            if "metadata" in data:
                return sum(self._count_terms(v) for k, v in data.items() if k != "metadata")
            return sum(self._count_terms(v) for v in data.values())
        if isinstance(data, list):
            return sum(self._count_terms(item) for item in data)
        return 0
