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

"""Terminology Base for domain-specific term management.

Provides centralized terminology database with:
- Domain-specific term storage
- Term validation
- Glossary support
- Multi-domain term management
"""

from __future__ import annotations

import logging
import re
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Error message constant
ERR_TERMBASE_NOT_INITIALIZED = "Terminology Base not initialized"


class TermEntry(BaseModel):
    """Terminology entry.

    Represents a term translation with metadata.
    """

    id: int | None = Field(default=None, description="Database ID")
    source_term: str = Field(..., description="Source term", min_length=1)
    target_term: str = Field(..., description="Target term", min_length=1)
    source_lang: str = Field(..., description="Source language code", pattern=r"^[a-z]{2}$")
    target_lang: str = Field(..., description="Target language code", pattern=r"^[a-z]{2}$")
    domain: str | None = Field(default=None, description="Domain (e.g., legal, medical)")
    definition: str | None = Field(default=None, description="Term definition")
    usage_note: str | None = Field(default=None, description="Usage notes or context")
    created_at: datetime | None = Field(default=None, description="Creation timestamp")


class TermViolation(BaseModel):
    """Term violation detected during validation.

    Represents a terminology issue found in translation.
    """

    source_term: str = Field(..., description="Source term that should be translated")
    expected_terms: list[str] = Field(..., description="Expected translations from termbase")
    found_in_translation: bool = Field(..., description="Whether any expected term was found")
    severity: str = Field(
        ..., description="Severity: critical, major, or minor", pattern=r"^(critical|major|minor)$"
    )


class TerminologyBase:
    """Centralized terminology management.

    Provides storage, retrieval, and validation of domain-specific terminology.

    Example:
        >>> termbase = TerminologyBase("kttc_terms.db")
        >>> await termbase.initialize()
        >>> await termbase.add_term(
        ...     source_term="API",
        ...     target_term="interfaz de programación de aplicaciones",
        ...     source_lang="en",
        ...     target_lang="es",
        ...     domain="technical"
        ... )
        >>> terms = await termbase.lookup_term(
        ...     term="API",
        ...     source_lang="en",
        ...     target_lang="es"
        ... )
    """

    def __init__(self, db_path: str | Path = "kttc_terms.db"):
        """Initialize Terminology Base.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db: sqlite3.Connection | None = None
        self._initialized = False

    def initialize(self) -> None:
        """Initialize database.

        Creates database schema.

        Raises:
            RuntimeError: If initialization fails
        """
        if self._initialized:
            return

        try:
            self.db = sqlite3.connect(str(self.db_path))
            self.db.row_factory = sqlite3.Row
            self._create_schema()

            self._initialized = True
            logger.info(f"Terminology Base initialized: {self.db_path}")

        except Exception as e:
            raise RuntimeError(f"Failed to initialize Terminology Base: {e}") from e

    def _create_schema(self) -> None:
        """Create database schema."""
        if self.db is None:
            raise RuntimeError("Database not initialized")

        self.db.execute(
            """
            CREATE TABLE IF NOT EXISTS terms (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_term TEXT NOT NULL COLLATE NOCASE,
                target_term TEXT NOT NULL,
                source_lang TEXT NOT NULL,
                target_lang TEXT NOT NULL,
                domain TEXT,
                definition TEXT,
                usage_note TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        # Create unique index that treats NULL domain as empty string
        self.db.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_unique_term
            ON terms(source_term COLLATE NOCASE, source_lang, target_lang, COALESCE(domain, ''))
        """
        )

        self.db.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_term_lookup
            ON terms(source_term COLLATE NOCASE, source_lang, target_lang)
        """
        )

        self.db.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_domain
            ON terms(domain)
        """
        )

        self.db.commit()

    def add_term(
        self,
        source_term: str,
        target_term: str,
        source_lang: str,
        target_lang: str,
        domain: str | None = None,
        definition: str | None = None,
        usage_note: str | None = None,
    ) -> int:
        """Add term to termbase.

        Args:
            source_term: Source term
            target_term: Target term translation
            source_lang: Source language code
            target_lang: Target language code
            domain: Domain category (optional)
            definition: Term definition (optional)
            usage_note: Usage notes (optional)

        Returns:
            ID of inserted term

        Raises:
            RuntimeError: If termbase not initialized
        """
        if not self._initialized or self.db is None:
            raise RuntimeError(f"{ERR_TERMBASE_NOT_INITIALIZED}. Call initialize() first.")

        try:
            cursor = self.db.execute(
                """
                INSERT INTO terms
                (source_term, target_term, source_lang, target_lang, domain, definition, usage_note)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    source_term,
                    target_term,
                    source_lang,
                    target_lang,
                    domain,
                    definition,
                    usage_note,
                ),
            )

            self.db.commit()
            return cursor.lastrowid or 0

        except sqlite3.IntegrityError:
            # Duplicate - update instead
            logger.debug(f"Duplicate term entry, updating: {source_term}")
            self.db.execute(
                """
                UPDATE terms
                SET target_term = ?, definition = ?, usage_note = ?
                WHERE source_term = ? COLLATE NOCASE
                  AND source_lang = ? AND target_lang = ?
                  AND (domain = ? OR (domain IS NULL AND ? IS NULL))
                """,
                (
                    target_term,
                    definition,
                    usage_note,
                    source_term,
                    source_lang,
                    target_lang,
                    domain,
                    domain,
                ),
            )
            self.db.commit()

            # Get ID
            cursor = self.db.execute(
                """
                SELECT id FROM terms
                WHERE source_term = ? COLLATE NOCASE
                  AND source_lang = ?
                  AND target_lang = ?
                  AND (domain = ? OR (domain IS NULL AND ? IS NULL))
                """,
                (source_term, source_lang, target_lang, domain, domain),
            )
            row = cursor.fetchone()
            return int(row["id"]) if row else 0

    def lookup_term(
        self,
        term: str,
        source_lang: str,
        target_lang: str,
        domain: str | None = None,
    ) -> list[TermEntry]:
        """Lookup term in termbase.

        Args:
            term: Term to lookup (case-insensitive)
            source_lang: Source language code
            target_lang: Target language code
            domain: Domain filter (optional)

        Returns:
            List of matching term entries

        Raises:
            RuntimeError: If termbase not initialized
        """
        if not self._initialized or self.db is None:
            raise RuntimeError(ERR_TERMBASE_NOT_INITIALIZED)

        query = """
            SELECT id, source_term, target_term, source_lang, target_lang,
                   domain, definition, usage_note, created_at
            FROM terms
            WHERE source_term = ? COLLATE NOCASE
              AND source_lang = ?
              AND target_lang = ?
        """
        params: list[str] = [term, source_lang, target_lang]

        if domain:
            query += " AND domain = ?"
            params.append(domain)

        cursor = self.db.execute(query, params)

        entries = []
        for row in cursor:
            entries.append(
                TermEntry(
                    id=row["id"],
                    source_term=row["source_term"],
                    target_term=row["target_term"],
                    source_lang=row["source_lang"],
                    target_lang=row["target_lang"],
                    domain=row["domain"],
                    definition=row["definition"],
                    usage_note=row["usage_note"],
                    created_at=(
                        datetime.fromisoformat(row["created_at"]) if row["created_at"] else None
                    ),
                )
            )

        return entries

    def validate_translation(
        self,
        source_text: str,
        translation: str,
        source_lang: str,
        target_lang: str,
        domain: str | None = None,
    ) -> list[TermViolation]:
        """Validate translation against termbase.

        Checks if terms in source are correctly translated according to termbase.

        Args:
            source_text: Source text
            translation: Translation to validate
            source_lang: Source language code
            target_lang: Target language code
            domain: Domain context (optional)

        Returns:
            List of term violations

        Raises:
            RuntimeError: If termbase not initialized
        """
        if not self._initialized or self.db is None:
            raise RuntimeError(ERR_TERMBASE_NOT_INITIALIZED)

        # Extract potential terms from source (words and phrases)
        source_terms = self._extract_terms(source_text)

        violations: list[TermViolation] = []

        for term in source_terms:
            # Lookup term in termbase
            entries = self.lookup_term(term, source_lang, target_lang, domain)

            if entries:
                # Check if any approved translation is in target
                expected_terms = [entry.target_term for entry in entries]
                found = any(
                    expected_term.lower() in translation.lower() for expected_term in expected_terms
                )

                if not found:
                    # Determine severity
                    severity = "major"  # Default
                    if any("critical" in (entry.usage_note or "").lower() for entry in entries):
                        severity = "critical"

                    violations.append(
                        TermViolation(
                            source_term=term,
                            expected_terms=expected_terms,
                            found_in_translation=False,
                            severity=severity,
                        )
                    )

        return violations

    def _extract_terms(self, text: str) -> list[str]:
        """Extract potential terms from text.

        Extracts words and multi-word phrases that might be technical terms.

        Args:
            text: Text to extract terms from

        Returns:
            List of potential terms
        """
        # Extract words (alphanumeric + hyphen)
        words = re.findall(r"\b[A-Za-z][A-Za-z0-9-]*\b", text)

        # Also extract 2-word and 3-word phrases
        phrases = re.findall(r"\b[A-Z][A-Za-z0-9-]*(?:\s+[A-Z][A-Za-z0-9-]*){1,2}\b", text)

        # Combine and deduplicate
        terms = list(set(words + phrases))

        # Filter out common words (simple heuristic: length > 3 or starts with capital)
        terms = [term for term in terms if len(term) > 3 or (term and term[0].isupper())]

        return terms

    def add_glossary(
        self,
        glossary: dict[str, str],
        source_lang: str,
        target_lang: str,
        domain: str | None = None,
    ) -> int:
        """Bulk add terms from glossary.

        Args:
            glossary: Dictionary mapping source terms to target terms
            source_lang: Source language code
            target_lang: Target language code
            domain: Domain category (optional)

        Returns:
            Number of terms added

        Raises:
            RuntimeError: If termbase not initialized
        """
        count = 0

        for source_term, target_term in glossary.items():
            try:
                self.add_term(
                    source_term=source_term,
                    target_term=target_term,
                    source_lang=source_lang,
                    target_lang=target_lang,
                    domain=domain,
                )
                count += 1
            except Exception as e:
                logger.warning(f"Failed to add term {source_term}: {e}")

        logger.info(f"Added {count} terms from glossary")
        return count

    def get_statistics(self) -> dict[str, Any]:
        """Get terminology base statistics.

        Returns:
            Dictionary with statistics
        """
        if not self._initialized or self.db is None:
            raise RuntimeError(ERR_TERMBASE_NOT_INITIALIZED)

        stats = {}

        # Total terms
        cursor = self.db.execute("SELECT COUNT(*) as count FROM terms")
        stats["total_terms"] = cursor.fetchone()["count"]

        # Language pairs
        cursor = self.db.execute("SELECT DISTINCT source_lang, target_lang FROM terms")
        stats["language_pairs"] = [f"{row['source_lang']}-{row['target_lang']}" for row in cursor]

        # Domains
        cursor = self.db.execute(
            "SELECT domain, COUNT(*) as count FROM terms WHERE domain IS NOT NULL GROUP BY domain"
        )
        stats["domains"] = {row["domain"]: row["count"] for row in cursor}

        return stats

    def cleanup(self) -> None:
        """Close database connection."""
        if self.db is not None:
            self.db.close()
            self.db = None

        self._initialized = False
        logger.info("Terminology Base cleaned up")

    def __del__(self) -> None:
        """Cleanup on object destruction."""
        if self.db is not None:
            self.db.close()
