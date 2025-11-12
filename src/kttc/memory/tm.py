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

"""Translation Memory with semantic search.

Provides database-backed translation memory with:
- Semantic similarity search using sentence embeddings
- MQM score tracking for quality filtering
- Domain categorization
- Usage statistics
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class TMSegment(BaseModel):
    """Translation Memory segment.

    Represents a stored translation with metadata.
    """

    id: int | None = Field(default=None, description="Database ID")
    source_text: str = Field(..., description="Source text", min_length=1)
    translation: str = Field(..., description="Translation", min_length=1)
    source_lang: str = Field(..., description="Source language code", pattern=r"^[a-z]{2}$")
    target_lang: str = Field(..., description="Target language code", pattern=r"^[a-z]{2}$")
    domain: str | None = Field(default=None, description="Domain (e.g., legal, medical)")
    mqm_score: float | None = Field(default=None, description="MQM quality score", ge=0.0, le=100.0)
    created_at: datetime | None = Field(default=None, description="Creation timestamp")
    usage_count: int = Field(default=0, description="Number of times used")


class TMSearchResult(BaseModel):
    """Search result from Translation Memory.

    Contains segment and similarity score.
    """

    segment: TMSegment = Field(..., description="Matching TM segment")
    similarity: float = Field(..., description="Semantic similarity score (0-1)", ge=0.0, le=1.0)


class TranslationMemory:
    """Translation Memory with semantic search.

    Provides storage and retrieval of translation segments using
    semantic similarity search based on sentence embeddings.

    Example:
        >>> tm = TranslationMemory("kttc_tm.db")
        >>> await tm.initialize()
        >>> await tm.add_translation(
        ...     source="Hello, world!",
        ...     translation="Hola, mundo!",
        ...     source_lang="en",
        ...     target_lang="es",
        ...     mqm_score=98.5
        ... )
        >>> results = await tm.search_similar(
        ...     source="Hello, everyone!",
        ...     source_lang="en",
        ...     target_lang="es"
        ... )
        >>> for result in results:
        ...     print(f"{result.segment.translation} (similarity: {result.similarity:.2f})")
    """

    def __init__(self, db_path: str | Path = "kttc_tm.db"):
        """Initialize Translation Memory.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db: sqlite3.Connection | None = None
        self.encoder: SentenceTransformer | None = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize database and embedding model.

        Creates database schema and loads sentence transformer model.

        Raises:
            RuntimeError: If initialization fails
        """
        if self._initialized:
            return

        try:
            # Initialize database
            self.db = sqlite3.connect(str(self.db_path))
            self.db.row_factory = sqlite3.Row
            self._create_schema()

            # Load sentence transformer model
            from sentence_transformers import SentenceTransformer

            logger.info("Loading sentence transformer model for TM...")
            self.encoder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

            self._initialized = True
            logger.info(f"Translation Memory initialized: {self.db_path}")

        except ImportError as e:
            raise RuntimeError(
                "Failed to import sentence-transformers. "
                "Install with: pip install sentence-transformers"
            ) from e
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Translation Memory: {e}") from e

    def _create_schema(self) -> None:
        """Create database schema."""
        if self.db is None:
            raise RuntimeError("Database not initialized")

        self.db.execute(
            """
            CREATE TABLE IF NOT EXISTS tm_segments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_text TEXT NOT NULL,
                translation TEXT NOT NULL,
                source_lang TEXT NOT NULL,
                target_lang TEXT NOT NULL,
                domain TEXT,
                mqm_score REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                source_embedding BLOB NOT NULL,
                usage_count INTEGER DEFAULT 0,
                UNIQUE(source_text, source_lang, target_lang)
            )
        """
        )

        self.db.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_lang_pair
            ON tm_segments(source_lang, target_lang)
        """
        )

        self.db.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_domain
            ON tm_segments(domain)
        """
        )

        self.db.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_mqm_score
            ON tm_segments(mqm_score DESC)
        """
        )

        self.db.commit()

    async def add_translation(
        self,
        source: str,
        translation: str,
        source_lang: str,
        target_lang: str,
        mqm_score: float | None = None,
        domain: str | None = None,
    ) -> int:
        """Add translation to memory.

        Args:
            source: Source text
            translation: Translation
            source_lang: Source language code
            target_lang: Target language code
            mqm_score: MQM quality score (optional)
            domain: Domain category (optional)

        Returns:
            ID of inserted segment

        Raises:
            RuntimeError: If TM not initialized
        """
        if not self._initialized or self.db is None or self.encoder is None:
            raise RuntimeError("Translation Memory not initialized. Call initialize() first.")

        # Generate embedding
        embedding = self.encoder.encode(source, convert_to_numpy=True)

        try:
            cursor = self.db.execute(
                """
                INSERT INTO tm_segments
                (source_text, translation, source_lang, target_lang, domain, mqm_score, source_embedding)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    source,
                    translation,
                    source_lang,
                    target_lang,
                    domain,
                    mqm_score,
                    embedding.tobytes(),
                ),
            )

            self.db.commit()
            return cursor.lastrowid or 0

        except sqlite3.IntegrityError:
            # Duplicate entry - update instead
            logger.debug(f"Duplicate TM entry, updating: {source[:50]}...")
            self.db.execute(
                """
                UPDATE tm_segments
                SET translation = ?, mqm_score = ?, domain = ?, source_embedding = ?
                WHERE source_text = ? AND source_lang = ? AND target_lang = ?
                """,
                (
                    translation,
                    mqm_score,
                    domain,
                    embedding.tobytes(),
                    source,
                    source_lang,
                    target_lang,
                ),
            )
            self.db.commit()

            # Get ID
            cursor = self.db.execute(
                "SELECT id FROM tm_segments WHERE source_text = ? AND source_lang = ? AND target_lang = ?",
                (source, source_lang, target_lang),
            )
            row = cursor.fetchone()
            return int(row["id"]) if row else 0

    async def search_similar(
        self,
        source: str,
        source_lang: str,
        target_lang: str,
        threshold: float = 0.75,
        limit: int = 5,
        min_mqm_score: float | None = 85.0,
    ) -> list[TMSearchResult]:
        """Search for similar translations in TM.

        Uses semantic similarity based on sentence embeddings.

        Args:
            source: Source text to search for
            source_lang: Source language code
            target_lang: Target language code
            threshold: Minimum similarity threshold (0-1)
            limit: Maximum number of results
            min_mqm_score: Minimum MQM score filter (optional)

        Returns:
            List of search results with similarity scores

        Raises:
            RuntimeError: If TM not initialized
        """
        if not self._initialized or self.db is None or self.encoder is None:
            raise RuntimeError("Translation Memory not initialized. Call initialize() first.")

        # Generate query embedding
        query_embedding = self.encoder.encode(source, convert_to_numpy=True)

        # Build query
        query = """
            SELECT id, source_text, translation, source_lang, target_lang,
                   domain, mqm_score, created_at, source_embedding, usage_count
            FROM tm_segments
            WHERE source_lang = ? AND target_lang = ?
        """
        params: list[str | float] = [source_lang, target_lang]

        if min_mqm_score is not None:
            query += " AND mqm_score >= ?"
            params.append(min_mqm_score)

        query += " ORDER BY mqm_score DESC LIMIT 100"

        cursor = self.db.execute(query, params)

        # Calculate similarities
        results: list[TMSearchResult] = []

        for row in cursor:
            stored_embedding = np.frombuffer(row["source_embedding"], dtype=np.float32)
            similarity = self._cosine_similarity(query_embedding, stored_embedding)

            if similarity >= threshold:
                segment = TMSegment(
                    id=row["id"],
                    source_text=row["source_text"],
                    translation=row["translation"],
                    source_lang=row["source_lang"],
                    target_lang=row["target_lang"],
                    domain=row["domain"],
                    mqm_score=row["mqm_score"],
                    created_at=(
                        datetime.fromisoformat(row["created_at"]) if row["created_at"] else None
                    ),
                    usage_count=row["usage_count"],
                )

                # Clamp similarity to [0, 1] to handle floating point precision issues
                clamped_similarity = min(1.0, max(0.0, float(similarity)))
                results.append(TMSearchResult(segment=segment, similarity=clamped_similarity))

        # Sort by similarity and limit
        results.sort(key=lambda x: x.similarity, reverse=True)
        return results[:limit]

    async def increment_usage(self, segment_id: int) -> None:
        """Increment usage counter for a segment.

        Args:
            segment_id: Database ID of segment
        """
        if not self._initialized or self.db is None:
            raise RuntimeError("Translation Memory not initialized")

        self.db.execute(
            "UPDATE tm_segments SET usage_count = usage_count + 1 WHERE id = ?", (segment_id,)
        )
        self.db.commit()

    async def get_statistics(self) -> dict[str, Any]:
        """Get Translation Memory statistics.

        Returns:
            Dictionary with statistics (total segments, languages, domains, etc.)
        """
        if not self._initialized or self.db is None:
            raise RuntimeError("Translation Memory not initialized")

        stats = {}

        # Total segments
        cursor = self.db.execute("SELECT COUNT(*) as count FROM tm_segments")
        stats["total_segments"] = cursor.fetchone()["count"]

        # Language pairs
        cursor = self.db.execute("SELECT DISTINCT source_lang, target_lang FROM tm_segments")
        stats["language_pairs"] = [f"{row['source_lang']}-{row['target_lang']}" for row in cursor]

        # Domains
        cursor = self.db.execute(
            "SELECT domain, COUNT(*) as count FROM tm_segments WHERE domain IS NOT NULL GROUP BY domain"
        )
        stats["domains"] = {row["domain"]: row["count"] for row in cursor}

        # Average MQM score
        cursor = self.db.execute(
            "SELECT AVG(mqm_score) as avg_score FROM tm_segments WHERE mqm_score IS NOT NULL"
        )
        row = cursor.fetchone()
        stats["avg_mqm_score"] = row["avg_score"] if row["avg_score"] else 0.0

        return stats

    def _cosine_similarity(
        self, vec1: npt.NDArray[np.floating[Any]], vec2: npt.NDArray[np.floating[Any]]
    ) -> float:
        """Calculate cosine similarity between two vectors.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Cosine similarity (0-1 range)
        """
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))

    async def cleanup(self) -> None:
        """Close database connection and cleanup resources."""
        if self.db is not None:
            self.db.close()
            self.db = None

        self.encoder = None
        self._initialized = False
        logger.info("Translation Memory cleaned up")

    def __del__(self) -> None:
        """Cleanup on object destruction."""
        if self.db is not None:
            self.db.close()
