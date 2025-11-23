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

"""Batch processing for multiple translation tasks.

Supports multiple input formats:
- CSV: Structured data with headers
- JSON: Array of translation objects
- JSONL: One translation per line (JSON Lines)
- XLIFF: XML Localization Interchange File Format
"""

from __future__ import annotations

import csv
import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import defusedxml.ElementTree as ET  # noqa: N817

from kttc.core.models import TranslationTask

logger = logging.getLogger(__name__)


@dataclass
class BatchTranslation:
    """Single translation entry from batch file."""

    source_text: str
    translation: str
    source_lang: str
    target_lang: str
    domain: str | None = None
    context: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None

    def to_task(self) -> TranslationTask:
        """Convert to TranslationTask."""
        return TranslationTask(
            source_text=self.source_text,
            translation=self.translation,
            source_lang=self.source_lang,
            target_lang=self.target_lang,
            context=self.context,
        )


class BatchFileParser:
    """Parse batch translation files in various formats."""

    @staticmethod
    def parse_csv(file_path: Path) -> list[BatchTranslation]:
        """Parse CSV file with translations.

        CSV Format:
            source,translation,source_lang,target_lang,domain,context

        Required columns: source, translation, source_lang, target_lang
        Optional columns: domain, context

        Args:
            file_path: Path to CSV file

        Returns:
            List of BatchTranslation objects

        Raises:
            ValueError: If required columns are missing
            FileNotFoundError: If file doesn't exist

        Example CSV:
            ```csv
            source,translation,source_lang,target_lang,domain
            Hello world,Hola mundo,en,es,general
            API endpoint,Punto final de API,en,es,technical
            ```
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        translations = []

        with open(file_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)

            # Validate required columns
            required = {"source", "translation", "source_lang", "target_lang"}
            if not required.issubset(set(reader.fieldnames or [])):
                missing = required - set(reader.fieldnames or [])
                raise ValueError(f"CSV missing required columns: {missing}")

            for row_num, row in enumerate(reader, start=2):  # Start at 2 (header is 1)
                try:
                    # Parse context if present (expects JSON string)
                    context = None
                    if row.get("context"):
                        try:
                            context = json.loads(row["context"])
                        except json.JSONDecodeError:
                            logger.warning(
                                f"Row {row_num}: Invalid JSON in context field, skipping"
                            )

                    # Create BatchTranslation
                    translation = BatchTranslation(
                        source_text=row["source"].strip(),
                        translation=row["translation"].strip(),
                        source_lang=row["source_lang"].strip(),
                        target_lang=row["target_lang"].strip(),
                        domain=row.get("domain", "").strip() or None,
                        context=context,
                        metadata={"row_number": row_num, "file": str(file_path)},
                    )

                    translations.append(translation)

                except KeyError as e:
                    logger.error(f"Row {row_num}: Missing required field {e}")
                    raise ValueError(f"Row {row_num}: Missing required field {e}") from e

        logger.info(f"Parsed {len(translations)} translations from CSV: {file_path}")
        return translations

    @staticmethod
    def parse_json(file_path: Path) -> list[BatchTranslation]:
        """Parse JSON file with translations.

        JSON Format:
            ```json
            [
                {
                    "source": "Hello world",
                    "translation": "Hola mundo",
                    "source_lang": "en",
                    "target_lang": "es",
                    "domain": "general",
                    "context": {"complexity": "simple"}
                },
                ...
            ]
            ```

        Args:
            file_path: Path to JSON file

        Returns:
            List of BatchTranslation objects

        Raises:
            ValueError: If JSON is invalid or missing required fields
            FileNotFoundError: If file doesn't exist
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError("JSON must be an array of translation objects")

        translations = []

        for idx, item in enumerate(data):
            try:
                # Validate required fields
                required = {"source", "translation", "source_lang", "target_lang"}
                missing = required - set(item.keys())
                if missing:
                    raise ValueError(f"Entry {idx}: Missing required fields: {missing}")

                # Map field names (support both 'source' and 'source_text')
                source_text = item.get("source") or item.get("source_text")
                if not source_text:
                    raise ValueError(f"Entry {idx}: Missing source text")

                translation = BatchTranslation(
                    source_text=source_text,
                    translation=item["translation"],
                    source_lang=item["source_lang"],
                    target_lang=item["target_lang"],
                    domain=item.get("domain"),
                    context=item.get("context"),
                    metadata={"index": idx, "file": str(file_path)},
                )

                translations.append(translation)

            except (KeyError, ValueError) as e:
                logger.error(f"Entry {idx}: {e}")
                raise ValueError(f"Entry {idx}: {e}") from e

        logger.info(f"Parsed {len(translations)} translations from JSON: {file_path}")
        return translations

    @staticmethod
    def parse_jsonl(file_path: Path) -> list[BatchTranslation]:
        """Parse JSONL (JSON Lines) file with translations.

        JSONL Format (one JSON object per line):
            ```jsonl
            {"source": "Hello world", "translation": "Hola mundo", "source_lang": "en", "target_lang": "es"}
            {"source": "Goodbye", "translation": "Adiós", "source_lang": "en", "target_lang": "es"}
            ```

        Args:
            file_path: Path to JSONL file

        Returns:
            List of BatchTranslation objects

        Raises:
            ValueError: If JSONL is invalid or missing required fields
            FileNotFoundError: If file doesn't exist
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        translations = []

        with open(file_path, encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue  # Skip empty lines

                try:
                    item = json.loads(line)

                    # Validate required fields
                    required = {"source", "translation", "source_lang", "target_lang"}
                    missing = required - set(item.keys())
                    if missing:
                        raise ValueError(f"Line {line_num}: Missing required fields: {missing}")

                    # Map field names
                    source_text = item.get("source") or item.get("source_text")
                    if not source_text:
                        raise ValueError(f"Line {line_num}: Missing source text")

                    translation = BatchTranslation(
                        source_text=source_text,
                        translation=item["translation"],
                        source_lang=item["source_lang"],
                        target_lang=item["target_lang"],
                        domain=item.get("domain"),
                        context=item.get("context"),
                        metadata={"line_number": line_num, "file": str(file_path)},
                    )

                    translations.append(translation)

                except json.JSONDecodeError as e:
                    logger.error(f"Line {line_num}: Invalid JSON - {e}")
                    raise ValueError(f"Line {line_num}: Invalid JSON") from e
                except ValueError as e:
                    logger.error(f"Line {line_num}: {e}")
                    raise

        logger.info(f"Parsed {len(translations)} translations from JSONL: {file_path}")
        return translations

    @staticmethod
    def _detect_xliff_namespace(root: Any) -> dict[str, str]:
        """Detect XLIFF namespace from root element."""
        if root.tag.startswith("{urn:oasis:names:tc:xliff:document:1.2}"):
            return {"xliff": "urn:oasis:names:tc:xliff:document:1.2"}
        elif root.tag.startswith("{urn:oasis:names:tc:xliff:document:2.0}"):
            return {"xliff": "urn:oasis:names:tc:xliff:document:2.0"}
        return {}

    @staticmethod
    def _find_xliff_elements(parent: Any, paths: list[str], namespace: dict[str, str]) -> list[Any]:
        """Find elements using multiple XPath patterns with namespace support."""
        for path in paths:
            elements = parent.findall(path, namespace) if namespace else parent.findall(path)
            if elements:
                return list(elements)
        return []

    @staticmethod
    def _parse_trans_unit(
        trans_unit: Any,
        idx: int,
        namespace: dict[str, str],
        source_lang: str,
        target_lang: str,
        domain: str | None,
        file_path: Path,
    ) -> BatchTranslation | None:
        """Parse a single trans-unit element."""
        unit_id = trans_unit.get("id", str(idx + 1))

        source_elem = (
            trans_unit.find("xliff:source", namespace) if namespace else trans_unit.find("source")
        )
        target_elem = (
            trans_unit.find("xliff:target", namespace) if namespace else trans_unit.find("target")
        )

        if source_elem is None:
            logger.warning(f"Trans-unit {unit_id}: Missing source element, skipping")
            return None
        if target_elem is None:
            logger.warning(f"Trans-unit {unit_id}: Missing target element, skipping")
            return None

        source_text = "".join(source_elem.itertext()).strip()
        translation_text = "".join(target_elem.itertext()).strip()

        if not source_text or not translation_text:
            logger.warning(f"Trans-unit {unit_id}: Empty source or target text, skipping")
            return None

        return BatchTranslation(
            source_text=source_text,
            translation=translation_text,
            source_lang=source_lang,
            target_lang=target_lang,
            domain=domain,
            context=None,
            metadata={"trans_unit_id": unit_id, "file": str(file_path)},
        )

    @staticmethod
    def parse_xliff(file_path: Path) -> list[BatchTranslation]:
        """Parse XLIFF (XML Localization Interchange File Format) file.

        XLIFF Format:
            ```xml
            <?xml version="1.0" encoding="UTF-8"?>
            <xliff version="1.2" xmlns="urn:oasis:names:tc:xliff:document:1.2">
              <file source-language="en" target-language="es" datatype="plaintext">
                <body>
                  <trans-unit id="1">
                    <source>Hello world</source>
                    <target>Hola mundo</target>
                  </trans-unit>
                </body>
              </file>
            </xliff>
            ```

        Supports XLIFF 1.2 and 2.0 formats.

        Args:
            file_path: Path to XLIFF file

        Returns:
            List of BatchTranslation objects

        Raises:
            ValueError: If XLIFF is invalid or missing required elements
            FileNotFoundError: If file doesn't exist
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        translations: list[BatchTranslation] = []

        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            namespace = BatchFileParser._detect_xliff_namespace(root)

            file_elements = BatchFileParser._find_xliff_elements(
                root, ["xliff:file", "file", ".//file"], namespace
            )

            for file_elem in file_elements:
                source_lang = file_elem.get("source-language", "").strip()
                target_lang = file_elem.get("target-language", "").strip()

                if not source_lang or not target_lang:
                    logger.warning(
                        "File element missing source-language or target-language, skipping"
                    )
                    continue

                domain = file_elem.get("datatype")
                trans_units = BatchFileParser._find_xliff_elements(
                    file_elem,
                    [".//xliff:trans-unit", ".//trans-unit", ".//xliff:unit", ".//unit"],
                    namespace,
                )

                for idx, trans_unit in enumerate(trans_units):
                    translation = BatchFileParser._parse_trans_unit(
                        trans_unit, idx, namespace, source_lang, target_lang, domain, file_path
                    )
                    if translation:
                        translations.append(translation)

        except ET.ParseError as e:
            logger.error(f"Failed to parse XLIFF file: {e}")
            raise ValueError(f"Invalid XLIFF format: {e}") from e
        except Exception as e:
            logger.error(f"Error parsing XLIFF file: {e}")
            raise ValueError(f"Error parsing XLIFF file: {e}") from e

        logger.info(f"Parsed {len(translations)} translations from XLIFF: {file_path}")
        return translations

    @classmethod
    def parse(cls, file_path: Path) -> list[BatchTranslation]:
        """Auto-detect format and parse batch file.

        Detects format based on file extension:
        - .csv → CSV parser
        - .json → JSON parser
        - .jsonl → JSONL parser
        - .xliff, .xlf → XLIFF parser

        Args:
            file_path: Path to batch file

        Returns:
            List of BatchTranslation objects

        Raises:
            ValueError: If file format is unsupported
        """
        suffix = file_path.suffix.lower()

        if suffix == ".csv":
            return cls.parse_csv(file_path)
        elif suffix == ".json":
            return cls.parse_json(file_path)
        elif suffix == ".jsonl":
            return cls.parse_jsonl(file_path)
        elif suffix in (".xliff", ".xlf"):
            return cls.parse_xliff(file_path)
        else:
            raise ValueError(
                f"Unsupported file format: {suffix}. "
                f"Supported: .csv, .json, .jsonl, .xliff, .xlf"
            )


class BatchGrouper:
    """Group batch translations for optimal processing."""

    @staticmethod
    def group_by_language_pair(
        translations: list[BatchTranslation],
    ) -> dict[tuple[str, str], list[BatchTranslation]]:
        """Group translations by language pair.

        Useful for batch processing with different models per language pair.

        Args:
            translations: List of BatchTranslation objects

        Returns:
            Dictionary mapping (source_lang, target_lang) to list of translations

        Example:
            ```python
            groups = BatchGrouper.group_by_language_pair(translations)
            # {
            #     ('en', 'es'): [trans1, trans2, ...],
            #     ('en', 'ru'): [trans3, trans4, ...],
            # }
            ```
        """
        groups: dict[tuple[str, str], list[BatchTranslation]] = defaultdict(list)

        for translation in translations:
            key = (translation.source_lang, translation.target_lang)
            groups[key].append(translation)

        return dict(groups)

    @staticmethod
    def group_by_domain(
        translations: list[BatchTranslation],
    ) -> dict[str, list[BatchTranslation]]:
        """Group translations by domain.

        Args:
            translations: List of BatchTranslation objects

        Returns:
            Dictionary mapping domain to list of translations
        """
        groups: dict[str, list[BatchTranslation]] = defaultdict(list)

        for translation in translations:
            domain = translation.domain or "general"
            groups[domain].append(translation)

        return dict(groups)

    @staticmethod
    def create_batches(
        translations: list[BatchTranslation], batch_size: int = 50
    ) -> list[list[BatchTranslation]]:
        """Split translations into batches of specified size.

        Args:
            translations: List of BatchTranslation objects
            batch_size: Maximum translations per batch

        Returns:
            List of batches (each batch is a list of translations)

        Example:
            ```python
            batches = BatchGrouper.create_batches(translations, batch_size=50)
            for batch in batches:
                # Process batch of up to 50 translations
                pass
            ```
        """
        batches = []

        for i in range(0, len(translations), batch_size):
            batch = translations[i : i + batch_size]
            batches.append(batch)

        return batches
