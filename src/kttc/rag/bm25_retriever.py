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

"""BM25 retriever implementation for lightweight RAG.

Implements Okapi BM25 algorithm for document retrieval without external dependencies.
Optimized for CPU-only operation on laptops.

BM25 Formula:
    score(D,Q) = Σ IDF(qi) · (f(qi,D) · (k1+1)) / (f(qi,D) + k1 · (1 - b + b · |D|/avgdl))

where:
    - f(qi,D): frequency of term qi in document D
    - |D|: document length
    - avgdl: average document length
    - k1, b: tuning parameters (typically k1=1.5, b=0.75)
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Document:
    """Document for RAG indexing.

    Attributes:
        id: Unique document identifier
        content: Document text content
        metadata: Optional metadata (source, type, etc.)
    """

    id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    """Result of document retrieval.

    Attributes:
        document: Retrieved document
        score: BM25 relevance score
        rank: Position in result ranking (1-based)
    """

    document: Document
    score: float
    rank: int


class BM25Retriever:
    """BM25-based document retriever for lightweight RAG.

    Implements Okapi BM25 algorithm for efficient, CPU-based retrieval.
    No external dependencies or GPU required.

    Example:
        >>> retriever = BM25Retriever()
        >>> retriever.add_documents([
        ...     Document(id="1", content="Translation quality assurance"),
        ...     Document(id="2", content="Machine translation evaluation"),
        ... ])
        >>> results = retriever.search("translation quality", top_k=1)
        >>> print(results[0].document.content)
        'Translation quality assurance'
    """

    # BM25 tuning parameters
    K1 = 1.5  # Term frequency saturation
    B = 0.75  # Document length normalization

    def __init__(self, stop_words: set[str] | None = None):
        """Initialize BM25 retriever.

        Args:
            stop_words: Set of words to ignore during indexing
        """
        self.documents: list[Document] = []
        self.doc_freqs: dict[str, int] = {}  # Term -> document frequency
        self.doc_lengths: list[int] = []  # Document lengths in tokens
        self.avg_doc_length: float = 0.0
        self.doc_term_freqs: list[Counter[str]] = []  # Per-document term frequencies

        # Default English stop words (lightweight set)
        self.stop_words = stop_words or {
            "a",
            "an",
            "and",
            "are",
            "as",
            "at",
            "be",
            "by",
            "for",
            "from",
            "has",
            "he",
            "in",
            "is",
            "it",
            "its",
            "of",
            "on",
            "that",
            "the",
            "to",
            "was",
            "were",
            "will",
            "with",
        }

        self._indexed = False

    @property
    def is_indexed(self) -> bool:
        """Check if documents have been indexed."""
        return self._indexed

    def tokenize(self, text: str) -> list[str]:
        """Tokenize text into words.

        Performs basic normalization:
        - Lowercase conversion
        - Non-alphanumeric removal
        - Stop word filtering

        Args:
            text: Text to tokenize

        Returns:
            List of tokens
        """
        # Lowercase and split on non-alphanumeric
        tokens = re.findall(r"\b\w+\b", text.lower())
        # Filter stop words and short tokens
        return [t for t in tokens if t not in self.stop_words and len(t) > 1]

    def add_document(self, doc: Document) -> None:
        """Add single document to index.

        Args:
            doc: Document to add
        """
        self.documents.append(doc)
        self._indexed = False

    def add_documents(self, docs: list[Document]) -> None:
        """Add multiple documents to index.

        Args:
            docs: List of documents to add
        """
        self.documents.extend(docs)
        self._indexed = False

    def build_index(self) -> None:
        """Build BM25 index from added documents.

        Calculates:
        - Document frequencies for IDF
        - Term frequencies per document
        - Average document length
        """
        self.doc_freqs = {}
        self.doc_lengths = []
        self.doc_term_freqs = []

        total_length = 0

        for doc in self.documents:
            tokens = self.tokenize(doc.content)
            term_freq = Counter(tokens)

            self.doc_term_freqs.append(term_freq)
            self.doc_lengths.append(len(tokens))
            total_length += len(tokens)

            # Update document frequencies
            for term in set(tokens):
                self.doc_freqs[term] = self.doc_freqs.get(term, 0) + 1

        # Calculate average document length
        self.avg_doc_length = total_length / len(self.documents) if self.documents else 0
        self._indexed = True

    def _idf(self, term: str) -> float:
        """Calculate Inverse Document Frequency for a term.

        Uses smoothed IDF: log((N - df + 0.5) / (df + 0.5))

        Args:
            term: Term to calculate IDF for

        Returns:
            IDF value
        """
        n = len(self.documents)
        df = self.doc_freqs.get(term, 0)

        # Smoothed IDF (Robertson-Sparck Jones formula)
        return math.log((n - df + 0.5) / (df + 0.5) + 1)

    def _score_document(self, query_terms: list[str], doc_idx: int) -> float:
        """Calculate BM25 score for a document.

        Args:
            query_terms: Tokenized query terms
            doc_idx: Document index

        Returns:
            BM25 score
        """
        score = 0.0
        doc_len = self.doc_lengths[doc_idx]
        term_freqs = self.doc_term_freqs[doc_idx]

        for term in query_terms:
            if term not in term_freqs:
                continue

            tf = term_freqs[term]
            idf = self._idf(term)

            # BM25 formula
            numerator = tf * (self.K1 + 1)
            denominator = tf + self.K1 * (1 - self.B + self.B * doc_len / self.avg_doc_length)

            score += idf * numerator / denominator

        return score

    def search(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """Search for documents matching query.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of RetrievalResult sorted by relevance

        Raises:
            ValueError: If index not built
        """
        if not self._indexed:
            self.build_index()

        if not self.documents:
            return []

        query_terms = self.tokenize(query)
        if not query_terms:
            return []

        # Score all documents
        scores: list[tuple[int, float]] = []
        for idx in range(len(self.documents)):
            score = self._score_document(query_terms, idx)
            if score > 0:
                scores.append((idx, score))

        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)

        # Build results
        results = []
        for rank, (idx, score) in enumerate(scores[:top_k], start=1):
            results.append(
                RetrievalResult(
                    document=self.documents[idx],
                    score=round(score, 4),
                    rank=rank,
                )
            )

        return results

    def search_with_context(
        self,
        query: str,
        top_k: int = 3,
        max_context_chars: int = 2000,
    ) -> str:
        """Search and format results as context string.

        Useful for injecting into LLM prompts.

        Args:
            query: Search query
            top_k: Number of results to include
            max_context_chars: Maximum characters in context

        Returns:
            Formatted context string
        """
        results = self.search(query, top_k=top_k)

        if not results:
            return ""

        context_parts = []
        total_chars = 0

        for result in results:
            content = result.document.content
            metadata = result.document.metadata

            # Format entry
            entry = f"[{result.document.id}]"
            if metadata.get("type"):
                entry += f" ({metadata['type']})"
            entry += f": {content}"

            # Check character limit
            if total_chars + len(entry) > max_context_chars:
                break

            context_parts.append(entry)
            total_chars += len(entry)

        return "\n".join(context_parts)

    def clear(self) -> None:
        """Clear all documents and reset index."""
        self.documents = []
        self.doc_freqs = {}
        self.doc_lengths = []
        self.doc_term_freqs = []
        self.avg_doc_length = 0.0
        self._indexed = False

    @property
    def document_count(self) -> int:
        """Get number of indexed documents."""
        return len(self.documents)
