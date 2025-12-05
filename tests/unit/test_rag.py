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

"""Strict tests for RAG (Retrieval-Augmented Generation) module.

Tests BM25 retriever and context builder functionality.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kttc.rag import BM25Retriever, ContextBuilder, Document, RetrievalResult


class TestDocument:
    """Tests for Document dataclass."""

    def test_document_creation(self) -> None:
        """Test basic document creation."""
        doc = Document(id="test1", content="Hello world")
        assert doc.id == "test1"
        assert doc.content == "Hello world"
        assert doc.metadata == {}

    def test_document_with_metadata(self) -> None:
        """Test document creation with metadata."""
        doc = Document(
            id="test2",
            content="Test content",
            metadata={"type": "glossary", "source": "en"},
        )
        assert doc.metadata["type"] == "glossary"
        assert doc.metadata["source"] == "en"

    def test_document_empty_content(self) -> None:
        """Test document with empty content is valid."""
        doc = Document(id="empty", content="")
        assert doc.content == ""


class TestRetrievalResult:
    """Tests for RetrievalResult dataclass."""

    def test_retrieval_result_creation(self) -> None:
        """Test basic result creation."""
        doc = Document(id="doc1", content="Test")
        result = RetrievalResult(document=doc, score=0.85, rank=1)
        assert result.document.id == "doc1"
        assert result.score == 0.85
        assert result.rank == 1

    def test_retrieval_result_negative_score(self) -> None:
        """Test that negative scores are valid (BM25 can have negative scores)."""
        doc = Document(id="doc1", content="Test")
        result = RetrievalResult(document=doc, score=-0.5, rank=1)
        assert result.score == -0.5


class TestBM25Retriever:
    """Comprehensive tests for BM25Retriever."""

    def test_empty_retriever(self) -> None:
        """Test retriever with no documents."""
        retriever = BM25Retriever()
        assert retriever.document_count == 0
        results = retriever.search("test query")
        assert len(results) == 0

    def test_add_single_document(self) -> None:
        """Test adding a single document."""
        retriever = BM25Retriever()
        doc = Document(id="doc1", content="machine translation quality")
        retriever.add_documents([doc])
        assert retriever.document_count == 1

    def test_add_multiple_documents(self) -> None:
        """Test adding multiple documents."""
        retriever = BM25Retriever()
        docs = [
            Document(id="1", content="translation quality assurance"),
            Document(id="2", content="machine translation errors"),
            Document(id="3", content="natural language processing"),
        ]
        retriever.add_documents(docs)
        assert retriever.document_count == 3

    def test_tokenization(self) -> None:
        """Test that tokenization works correctly."""
        retriever = BM25Retriever()
        tokens = retriever.tokenize("Hello, World! Testing 123.")
        assert "hello" in tokens
        assert "world" in tokens
        assert "testing" in tokens
        # Numbers are kept, punctuation is filtered
        assert "123" in tokens  # Numbers are valid tokens
        assert "," not in tokens

    def test_tokenization_unicode(self) -> None:
        """Test tokenization with Unicode characters."""
        retriever = BM25Retriever()
        tokens = retriever.tokenize("Привет мир translation")
        assert "привет" in tokens
        assert "мир" in tokens
        assert "translation" in tokens

    def test_build_index(self) -> None:
        """Test that index is built correctly."""
        retriever = BM25Retriever()
        docs = [
            Document(id="1", content="apple banana cherry"),
            Document(id="2", content="banana cherry date"),
        ]
        retriever.add_documents(docs)
        retriever.build_index()

        assert retriever._indexed is True
        assert "apple" in retriever.doc_freqs
        assert "banana" in retriever.doc_freqs
        # banana appears in 2 docs
        assert retriever.doc_freqs["banana"] == 2
        # apple appears in 1 doc
        assert retriever.doc_freqs["apple"] == 1

    def test_search_returns_relevant_results(self) -> None:
        """Test that search returns relevant documents first."""
        retriever = BM25Retriever()
        docs = [
            Document(id="1", content="translation quality metrics MQM"),
            Document(id="2", content="cooking recipes food"),
            Document(id="3", content="translation errors accuracy"),
        ]
        retriever.add_documents(docs)

        results = retriever.search("translation quality", top_k=3)

        # Translation-related docs should score higher
        assert len(results) > 0
        result_ids = [r.document.id for r in results]
        assert "1" in result_ids[:2]  # Doc 1 should be in top 2
        assert "3" in result_ids[:2]  # Doc 3 should be in top 2

    def test_search_top_k_limit(self) -> None:
        """Test that top_k limits results correctly."""
        retriever = BM25Retriever()
        docs = [Document(id=str(i), content=f"document {i}") for i in range(10)]
        retriever.add_documents(docs)

        results = retriever.search("document", top_k=3)
        assert len(results) == 3

        results = retriever.search("document", top_k=5)
        assert len(results) == 5

    def test_search_with_context(self) -> None:
        """Test search_with_context formatting."""
        retriever = BM25Retriever()
        docs = [
            Document(id="1", content="API: Application Programming Interface"),
            Document(id="2", content="REST: Representational State Transfer"),
        ]
        retriever.add_documents(docs)

        context = retriever.search_with_context("API REST", top_k=2)

        assert "API" in context
        assert "Application Programming Interface" in context
        assert len(context) > 0

    def test_search_with_context_max_chars(self) -> None:
        """Test that max_context_chars is respected."""
        retriever = BM25Retriever()
        docs = [Document(id=str(i), content="A" * 100) for i in range(10)]
        retriever.add_documents(docs)

        context = retriever.search_with_context("A", top_k=10, max_context_chars=50)
        assert len(context) <= 100  # Some buffer for formatting

    def test_clear(self) -> None:
        """Test clearing the retriever."""
        retriever = BM25Retriever()
        docs = [Document(id="1", content="test")]
        retriever.add_documents(docs)
        retriever.build_index()

        assert retriever.document_count == 1

        retriever.clear()

        assert retriever.document_count == 0
        assert retriever._indexed is False

    def test_bm25_scoring_formula(self) -> None:
        """Test that BM25 scoring follows expected behavior."""
        retriever = BM25Retriever()
        docs = [
            Document(id="1", content="cat cat cat"),  # High term frequency
            Document(id="2", content="cat dog"),  # Lower term frequency
        ]
        retriever.add_documents(docs)

        results = retriever.search("cat", top_k=2)

        # Doc with more "cat" occurrences should score higher
        assert results[0].document.id == "1"
        assert results[0].score > results[1].score

    def test_idf_calculation(self) -> None:
        """Test IDF calculation for rare vs common terms."""
        retriever = BM25Retriever()
        docs = [
            Document(id="1", content="common rare"),
            Document(id="2", content="common word"),
            Document(id="3", content="common text"),
        ]
        retriever.add_documents(docs)
        retriever.build_index()

        # Rare word appears in 1 doc, common appears in 3
        # IDF for rare should be higher than for common
        rare_idf = retriever._idf("rare")
        common_idf = retriever._idf("common")

        assert rare_idf > common_idf

    def test_empty_query(self) -> None:
        """Test search with empty query."""
        retriever = BM25Retriever()
        docs = [Document(id="1", content="test document")]
        retriever.add_documents(docs)

        results = retriever.search("")
        assert len(results) == 0

    def test_query_not_in_documents(self) -> None:
        """Test search for terms not in any document."""
        retriever = BM25Retriever()
        docs = [Document(id="1", content="apple banana")]
        retriever.add_documents(docs)

        results = retriever.search("xyznotexist")
        # Results may be empty or have zero/negative scores
        for result in results:
            assert result.score <= 0


class TestContextBuilder:
    """Comprehensive tests for ContextBuilder."""

    def test_default_enabled(self) -> None:
        """Test that ContextBuilder is enabled by default."""
        builder = ContextBuilder()
        assert builder.is_enabled is True

    def test_disabled_builder(self) -> None:
        """Test disabled ContextBuilder."""
        builder = ContextBuilder(enabled=False)
        assert builder.is_enabled is False

        # Should return empty results when disabled
        context = builder.get_context("test")
        assert context == ""

    def test_load_glossary(self, tmp_path: Path) -> None:
        """Test loading glossary from JSON file."""
        glossary_data = {
            "terms": [
                {
                    "source": "API",
                    "target": "АПИ",
                    "definition": "Application Programming Interface",
                },
                {"source": "bug", "target": "ошибка", "definition": "Software defect"},
            ]
        }
        glossary_file = tmp_path / "test_glossary.json"
        glossary_file.write_text(json.dumps(glossary_data))

        builder = ContextBuilder(enabled=True)
        count = builder.load_glossary(glossary_file)

        assert count == 2
        assert builder.document_count == 2
        assert str(glossary_file) in builder.loaded_sources

    def test_load_glossary_disabled(self, tmp_path: Path) -> None:
        """Test that glossary is not loaded when disabled."""
        glossary_data = {"terms": [{"source": "test", "target": "тест"}]}
        glossary_file = tmp_path / "test.json"
        glossary_file.write_text(json.dumps(glossary_data))

        builder = ContextBuilder(enabled=False)
        count = builder.load_glossary(glossary_file)

        assert count == 0
        assert builder.document_count == 0

    def test_load_glossary_not_found(self) -> None:
        """Test loading non-existent glossary."""
        builder = ContextBuilder(enabled=True)

        with pytest.raises(FileNotFoundError):
            builder.load_glossary(Path("/nonexistent/path.json"))

    def test_load_glossary_invalid_json(self, tmp_path: Path) -> None:
        """Test loading invalid JSON glossary."""
        glossary_file = tmp_path / "invalid.json"
        glossary_file.write_text("not valid json {{{")

        builder = ContextBuilder(enabled=True)
        count = builder.load_glossary(glossary_file)

        assert count == 0

    def test_load_examples(self) -> None:
        """Test loading translation examples."""
        builder = ContextBuilder(enabled=True)
        examples = [
            {"source": "Hello world", "target": "Привет мир"},
            {"source": "Good morning", "target": "Доброе утро"},
        ]

        count = builder.load_examples(examples)

        assert count == 2
        assert builder.document_count == 2

    def test_load_examples_disabled(self) -> None:
        """Test that examples are not loaded when disabled."""
        builder = ContextBuilder(enabled=False)
        examples = [{"source": "test", "target": "тест"}]

        count = builder.load_examples(examples)
        assert count == 0

    def test_load_translation_memory(self, tmp_path: Path) -> None:
        """Test loading translation memory from JSON."""
        tm_data = {
            "segments": [
                {"source": "Save file", "target": "Сохранить файл"},
                {"source": "Open document", "target": "Открыть документ"},
            ]
        }
        tm_file = tmp_path / "test_tm.json"
        tm_file.write_text(json.dumps(tm_data))

        builder = ContextBuilder(enabled=True)
        count = builder.load_translation_memory(tm_file)

        assert count == 2
        assert builder.document_count == 2

    def test_load_translation_memory_not_found(self) -> None:
        """Test loading non-existent TM."""
        builder = ContextBuilder(enabled=True)

        with pytest.raises(FileNotFoundError):
            builder.load_translation_memory(Path("/nonexistent/tm.json"))

    def test_load_translation_memory_unsupported_format(self, tmp_path: Path) -> None:
        """Test loading unsupported TM format."""
        tm_file = tmp_path / "test.tmx"
        tm_file.write_text("<tmx>...</tmx>")

        builder = ContextBuilder(enabled=True)
        count = builder.load_translation_memory(tm_file)

        assert count == 0

    def test_get_context(self, tmp_path: Path) -> None:
        """Test getting context for a query."""
        glossary_data = {
            "terms": [
                {"source": "translation", "target": "перевод"},
                {"source": "quality", "target": "качество"},
                {"source": "error", "target": "ошибка"},
            ]
        }
        glossary_file = tmp_path / "glossary.json"
        glossary_file.write_text(json.dumps(glossary_data))

        builder = ContextBuilder(enabled=True)
        builder.load_glossary(glossary_file)

        context = builder.get_context("translation quality", top_k=2)

        assert len(context) > 0
        # Should contain relevant terms
        assert "translation" in context.lower() or "перевод" in context.lower()

    def test_get_context_disabled(self) -> None:
        """Test that get_context returns empty when disabled."""
        builder = ContextBuilder(enabled=False)
        context = builder.get_context("test query")
        assert context == ""

    def test_get_relevant_terms(self, tmp_path: Path) -> None:
        """Test getting relevant glossary terms."""
        glossary_data = {
            "terms": [
                {"source": "API", "target": "АПИ"},
                {"source": "endpoint", "target": "конечная точка"},
            ]
        }
        glossary_file = tmp_path / "glossary.json"
        glossary_file.write_text(json.dumps(glossary_data))

        builder = ContextBuilder(enabled=True)
        builder.load_glossary(glossary_file)

        terms = builder.get_relevant_terms("API endpoint", top_k=2)

        assert len(terms) > 0
        assert all("source" in t and "target" in t and "score" in t for t in terms)

    def test_get_relevant_terms_disabled(self) -> None:
        """Test that get_relevant_terms returns empty when disabled."""
        builder = ContextBuilder(enabled=False)
        terms = builder.get_relevant_terms("test")
        assert terms == []

    def test_clear(self, tmp_path: Path) -> None:
        """Test clearing the context builder."""
        glossary_data = {"terms": [{"source": "test", "target": "тест"}]}
        glossary_file = tmp_path / "glossary.json"
        glossary_file.write_text(json.dumps(glossary_data))

        builder = ContextBuilder(enabled=True)
        builder.load_glossary(glossary_file)

        assert builder.document_count == 1
        assert len(builder.loaded_sources) == 1

        builder.clear()

        assert builder.document_count == 0
        assert len(builder.loaded_sources) == 0

    def test_document_count_property(self) -> None:
        """Test document_count property."""
        builder = ContextBuilder(enabled=True)
        assert builder.document_count == 0

        builder.load_examples([{"source": "a", "target": "b"}])
        assert builder.document_count == 1

    def test_loaded_sources_immutability(self, tmp_path: Path) -> None:
        """Test that loaded_sources returns a copy."""
        glossary_data = {"terms": [{"source": "test", "target": "тест"}]}
        glossary_file = tmp_path / "glossary.json"
        glossary_file.write_text(json.dumps(glossary_data))

        builder = ContextBuilder(enabled=True)
        builder.load_glossary(glossary_file)

        sources = builder.loaded_sources
        sources.append("fake_source")

        # Original should not be modified
        assert len(builder.loaded_sources) == 1


class TestRAGIntegration:
    """Integration tests for RAG functionality."""

    def test_full_workflow(self, tmp_path: Path) -> None:
        """Test complete RAG workflow."""
        # Create glossary
        glossary_data = {
            "terms": [
                {"source": "machine translation", "target": "машинный перевод"},
                {"source": "neural network", "target": "нейронная сеть"},
                {"source": "deep learning", "target": "глубокое обучение"},
            ]
        }
        glossary_file = tmp_path / "ml_glossary.json"
        glossary_file.write_text(json.dumps(glossary_data))

        # Create TM
        tm_data = {
            "segments": [
                {
                    "source": "Machine translation uses neural networks",
                    "target": "Машинный перевод использует нейронные сети",
                }
            ]
        }
        tm_file = tmp_path / "tm.json"
        tm_file.write_text(json.dumps(tm_data))

        # Build context
        builder = ContextBuilder(enabled=True)
        builder.load_glossary(glossary_file)
        builder.load_translation_memory(tm_file)

        # Test retrieval
        context = builder.get_context("neural network translation", top_k=3)

        assert len(context) > 0
        assert builder.document_count == 4  # 3 glossary + 1 TM

    def test_multiple_glossaries(self, tmp_path: Path) -> None:
        """Test loading multiple glossaries."""
        glossary1 = {"terms": [{"source": "term1", "target": "термин1"}]}
        glossary2 = {"terms": [{"source": "term2", "target": "термин2"}]}

        file1 = tmp_path / "glossary1.json"
        file2 = tmp_path / "glossary2.json"
        file1.write_text(json.dumps(glossary1))
        file2.write_text(json.dumps(glossary2))

        builder = ContextBuilder(enabled=True)
        builder.load_glossary(file1)
        builder.load_glossary(file2)

        assert builder.document_count == 2
        assert len(builder.loaded_sources) == 2
