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

"""Context builder for RAG-enhanced QA evaluation.

Builds context from glossaries, translation memories, and examples
for injection into LLM prompts.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from .bm25_retriever import BM25Retriever, Document

logger = logging.getLogger(__name__)


class ContextBuilder:
    """Builds RAG context for translation QA.

    Loads documents from various sources (glossaries, TMs, examples)
    and provides relevant context for LLM prompts.

    Example:
        >>> builder = ContextBuilder()
        >>> builder.load_glossary(Path("glossaries/en/technical.json"))
        >>> context = builder.get_context("API authentication error")
        >>> print(context)
    """

    def __init__(self, enabled: bool = True):
        """Initialize context builder.

        Args:
            enabled: Whether RAG is enabled (default: True, BM25 is lightweight)
        """
        self.enabled = enabled
        self.retriever = BM25Retriever()
        self._loaded_sources: list[str] = []

    def load_glossary(self, path: Path) -> int:
        """Load glossary file into RAG index.

        Args:
            path: Path to glossary JSON file

        Returns:
            Number of terms loaded

        Raises:
            FileNotFoundError: If file doesn't exist
        """
        if not self.enabled:
            logger.debug("RAG disabled, skipping glossary load: %s", path)
            return 0

        if not path.exists():
            raise FileNotFoundError(f"Glossary not found: {path}")

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            terms = data.get("terms", [])

            docs = []
            for term in terms:
                source = term.get("source", "")
                target = term.get("target", "")
                definition = term.get("definition", "")

                # Create searchable content
                content = f"{source}: {target}"
                if definition:
                    content += f" ({definition})"

                docs.append(
                    Document(
                        id=f"gloss:{path.stem}:{source}",
                        content=content,
                        metadata={
                            "type": "glossary",
                            "source_term": source,
                            "target_term": target,
                            "file": str(path),
                        },
                    )
                )

            self.retriever.add_documents(docs)
            self._loaded_sources.append(str(path))

            logger.info("Loaded %d terms from %s", len(docs), path)
            return len(docs)

        except json.JSONDecodeError as e:
            logger.error("Failed to parse glossary %s: %s", path, e)
            return 0

    def load_examples(self, examples: list[dict[str, str]]) -> int:
        """Load translation examples into RAG index.

        Args:
            examples: List of dicts with 'source' and 'target' keys

        Returns:
            Number of examples loaded
        """
        if not self.enabled:
            return 0

        docs = []
        for i, example in enumerate(examples):
            source = example.get("source", "")
            target = example.get("target", "")

            if source and target:
                docs.append(
                    Document(
                        id=f"example:{i}",
                        content=f"Source: {source}\nTarget: {target}",
                        metadata={"type": "example"},
                    )
                )

        self.retriever.add_documents(docs)
        return len(docs)

    def load_translation_memory(self, path: Path) -> int:
        """Load translation memory (TMX or JSON) into RAG index.

        Args:
            path: Path to TM file

        Returns:
            Number of segments loaded
        """
        if not self.enabled:
            return 0

        if not path.exists():
            raise FileNotFoundError(f"Translation memory not found: {path}")

        # Support JSON format
        if path.suffix == ".json":
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                segments = data.get("segments", data.get("translations", []))

                docs = []
                for i, seg in enumerate(segments):
                    source = seg.get("source", "")
                    target = seg.get("target", "")

                    if source and target:
                        docs.append(
                            Document(
                                id=f"tm:{path.stem}:{i}",
                                content=f"{source} -> {target}",
                                metadata={
                                    "type": "tm",
                                    "source": source,
                                    "target": target,
                                },
                            )
                        )

                self.retriever.add_documents(docs)
                self._loaded_sources.append(str(path))
                return len(docs)

            except json.JSONDecodeError as e:
                logger.error("Failed to parse TM %s: %s", path, e)
                return 0

        logger.warning("Unsupported TM format: %s", path.suffix)
        return 0

    def get_context(
        self,
        query: str,
        top_k: int = 3,
        max_chars: int = 1500,
    ) -> str:
        """Get relevant context for a query.

        Args:
            query: Search query (source text or terms)
            top_k: Number of results to include
            max_chars: Maximum context length

        Returns:
            Formatted context string, or empty if RAG disabled
        """
        if not self.enabled:
            return ""

        # Build index if not done
        if not self.retriever._indexed:
            self.retriever.build_index()

        return self.retriever.search_with_context(
            query=query,
            top_k=top_k,
            max_context_chars=max_chars,
        )

    def get_relevant_terms(self, query: str, top_k: int = 5) -> list[dict[str, str]]:
        """Get relevant glossary terms for a query.

        Args:
            query: Search query
            top_k: Number of terms to return

        Returns:
            List of term dictionaries with source, target, and score
        """
        if not self.enabled:
            return []

        results = self.retriever.search(query, top_k=top_k)

        terms = []
        for result in results:
            if result.document.metadata.get("type") == "glossary":
                terms.append(
                    {
                        "source": result.document.metadata.get("source_term", ""),
                        "target": result.document.metadata.get("target_term", ""),
                        "score": result.score,
                    }
                )

        return terms

    @property
    def is_enabled(self) -> bool:
        """Check if RAG is enabled."""
        return self.enabled

    @property
    def document_count(self) -> int:
        """Get total number of indexed documents."""
        return self.retriever.document_count

    @property
    def loaded_sources(self) -> list[str]:
        """Get list of loaded source files."""
        return self._loaded_sources.copy()

    def clear(self) -> None:
        """Clear all loaded documents."""
        self.retriever.clear()
        self._loaded_sources = []
