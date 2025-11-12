"""Tests for Translation Memory module."""

import tempfile
from pathlib import Path

import pytest

from kttc.memory.tm import TMSegment, TranslationMemory


class TestTMSegment:
    """Tests for TMSegment model."""

    def test_create_segment(self):
        """Test creating a TM segment."""
        segment = TMSegment(
            source_text="Hello, world!",
            translation="¡Hola, mundo!",
            source_lang="en",
            target_lang="es",
            domain="general",
            mqm_score=98.5,
        )

        assert segment.source_text == "Hello, world!"
        assert segment.translation == "¡Hola, mundo!"
        assert segment.source_lang == "en"
        assert segment.target_lang == "es"
        assert segment.domain == "general"
        assert segment.mqm_score == 98.5

    def test_segment_with_minimal_fields(self):
        """Test segment with only required fields."""
        segment = TMSegment(
            source_text="Test",
            translation="Prueba",
            source_lang="en",
            target_lang="es",
        )

        assert segment.id is None
        assert segment.domain is None
        assert segment.mqm_score is None
        assert segment.usage_count == 0


@pytest.mark.asyncio
@pytest.mark.metrics
class TestTranslationMemory:
    """Tests for TranslationMemory class."""

    @pytest.fixture
    async def tm(self):
        """Create temporary TM database."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        tm = TranslationMemory(db_path)
        await tm.initialize()

        yield tm

        await tm.cleanup()
        Path(db_path).unlink(missing_ok=True)

    async def test_initialization(self, tm):
        """Test TM initialization."""
        assert tm._initialized is True
        assert tm.db is not None
        assert tm.encoder is not None

    async def test_add_translation(self, tm):
        """Test adding translation to TM."""
        segment_id = await tm.add_translation(
            source="Hello, world!",
            translation="¡Hola, mundo!",
            source_lang="en",
            target_lang="es",
            mqm_score=98.5,
            domain="general",
        )

        assert segment_id > 0

    async def test_add_duplicate_translation_updates(self, tm):
        """Test that adding duplicate updates existing entry."""
        # Add first time
        id1 = await tm.add_translation(
            source="Test",
            translation="Prueba v1",
            source_lang="en",
            target_lang="es",
            mqm_score=90.0,
        )

        # Add again with different translation
        id2 = await tm.add_translation(
            source="Test",
            translation="Prueba v2",
            source_lang="en",
            target_lang="es",
            mqm_score=95.0,
        )

        # Should update, not create new
        assert id1 == id2

    async def test_search_similar_exact_match(self, tm):
        """Test searching for similar translations with exact match."""
        # Add translation
        await tm.add_translation(
            source="Hello, world!",
            translation="¡Hola, mundo!",
            source_lang="en",
            target_lang="es",
            mqm_score=98.5,
        )

        # Search for exact match
        results = await tm.search_similar(
            source="Hello, world!",
            source_lang="en",
            target_lang="es",
            threshold=0.95,
        )

        assert len(results) == 1
        assert results[0].segment.translation == "¡Hola, mundo!"
        assert results[0].similarity >= 0.95

    async def test_search_similar_fuzzy_match(self, tm):
        """Test searching with fuzzy match."""
        # Add translation
        await tm.add_translation(
            source="Hello, world!",
            translation="¡Hola, mundo!",
            source_lang="en",
            target_lang="es",
            mqm_score=98.5,
        )

        # Search for similar text
        results = await tm.search_similar(
            source="Hello, everyone!",  # Similar but not exact
            source_lang="en",
            target_lang="es",
            threshold=0.70,  # Lower threshold
        )

        # Should find similar match
        assert len(results) >= 1

    async def test_search_similar_filters_by_language(self, tm):
        """Test that search filters by language pair."""
        # Add English-Spanish
        await tm.add_translation(
            source="Test",
            translation="Prueba",
            source_lang="en",
            target_lang="es",
        )

        # Search for English-French (should not match)
        results = await tm.search_similar(
            source="Test", source_lang="en", target_lang="fr", threshold=0.70
        )

        assert len(results) == 0

    async def test_search_similar_filters_by_mqm_score(self, tm):
        """Test MQM score filtering."""
        # Add low-quality translation
        await tm.add_translation(
            source="Test",
            translation="Prueba",
            source_lang="en",
            target_lang="es",
            mqm_score=70.0,
        )

        # Search with high MQM threshold
        results = await tm.search_similar(
            source="Test",
            source_lang="en",
            target_lang="es",
            min_mqm_score=85.0,
        )

        # Should not match due to low quality
        assert len(results) == 0

    async def test_search_similar_respects_limit(self, tm):
        """Test that search respects limit parameter."""
        # Add multiple translations
        for i in range(10):
            await tm.add_translation(
                source=f"Test {i}",
                translation=f"Prueba {i}",
                source_lang="en",
                target_lang="es",
                mqm_score=90.0 + i,
            )

        # Search with limit
        results = await tm.search_similar(
            source="Test 0",
            source_lang="en",
            target_lang="es",
            threshold=0.60,
            limit=3,
        )

        assert len(results) <= 3

    async def test_increment_usage(self, tm):
        """Test incrementing usage counter."""
        # Add translation
        segment_id = await tm.add_translation(
            source="Test",
            translation="Prueba",
            source_lang="en",
            target_lang="es",
        )

        # Increment usage
        await tm.increment_usage(segment_id)
        await tm.increment_usage(segment_id)

        # Search to verify usage count (disable MQM filter for test data without scores)
        results = await tm.search_similar(
            source="Test", source_lang="en", target_lang="es", min_mqm_score=None
        )

        assert results[0].segment.usage_count == 2

    async def test_get_statistics(self, tm):
        """Test getting TM statistics."""
        # Add some translations
        await tm.add_translation(
            source="Test 1",
            translation="Prueba 1",
            source_lang="en",
            target_lang="es",
            mqm_score=90.0,
            domain="technical",
        )

        await tm.add_translation(
            source="Test 2",
            translation="Prueba 2",
            source_lang="en",
            target_lang="es",
            mqm_score=95.0,
            domain="medical",
        )

        stats = await tm.get_statistics()

        assert stats["total_segments"] == 2
        assert "en-es" in stats["language_pairs"]
        assert "technical" in stats["domains"]
        assert "medical" in stats["domains"]
        assert stats["avg_mqm_score"] == 92.5

    async def test_cosine_similarity_identical_vectors(self, tm):
        """Test cosine similarity with identical vectors."""
        import numpy as np

        vec = np.array([1.0, 2.0, 3.0])
        similarity = tm._cosine_similarity(vec, vec)

        assert similarity == pytest.approx(1.0)

    async def test_cosine_similarity_zero_vectors(self, tm):
        """Test cosine similarity with zero vectors."""
        import numpy as np

        vec1 = np.array([0.0, 0.0, 0.0])
        vec2 = np.array([1.0, 2.0, 3.0])

        similarity = tm._cosine_similarity(vec1, vec2)
        assert similarity == 0.0

    async def test_cleanup_closes_database(self, tm):
        """Test cleanup closes database connection."""
        await tm.cleanup()

        assert tm.db is None
        assert tm.encoder is None
        assert tm._initialized is False

    async def test_initialize_already_initialized(self):
        """Test that initialize() skips if already initialized."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        tm = TranslationMemory(db_path)
        await tm.initialize()

        # Mark as already initialized
        assert tm._initialized is True

        # Initialize again - should skip
        await tm.initialize()

        # Should still be initialized
        assert tm._initialized is True

        await tm.cleanup()
        Path(db_path).unlink(missing_ok=True)

    async def test_initialize_import_error(self):
        """Test initialization with missing sentence-transformers."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        tm = TranslationMemory(db_path)

        # Mock SentenceTransformer to raise ImportError
        import sys

        old_modules = sys.modules.copy()

        try:
            # Remove sentence_transformers from modules to simulate ImportError
            if "sentence_transformers" in sys.modules:
                del sys.modules["sentence_transformers"]

            # Mock import to fail
            import builtins

            original_import = builtins.__import__

            def mock_import(name, *args, **kwargs):
                if "sentence_transformers" in name:
                    raise ImportError("No module named 'sentence_transformers'")
                return original_import(name, *args, **kwargs)

            builtins.__import__ = mock_import

            with pytest.raises(RuntimeError, match="Failed to import sentence-transformers"):
                await tm.initialize()

        finally:
            builtins.__import__ = original_import
            sys.modules.update(old_modules)
            Path(db_path).unlink(missing_ok=True)

    async def test_initialize_general_error(self):
        """Test initialization with general error."""
        # Use invalid path to trigger error
        tm = TranslationMemory("/invalid/path/that/does/not/exist/tm.db")

        with pytest.raises(RuntimeError, match="Failed to initialize Translation Memory"):
            await tm.initialize()

    async def test_not_initialized_error_add_translation(self):
        """Test add_translation fails when not initialized."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        tm = TranslationMemory(db_path)

        # Try to add without initializing
        with pytest.raises(RuntimeError, match="not initialized"):
            await tm.add_translation("test", "prueba", "en", "es")

        Path(db_path).unlink(missing_ok=True)

    async def test_not_initialized_error_search(self):
        """Test search_similar fails when not initialized."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        tm = TranslationMemory(db_path)

        # Try to search without initializing
        with pytest.raises(RuntimeError, match="not initialized"):
            await tm.search_similar("test", "en", "es")

        Path(db_path).unlink(missing_ok=True)

    async def test_not_initialized_error_increment_usage(self):
        """Test increment_usage fails when not initialized."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        tm = TranslationMemory(db_path)

        # Try to increment without initializing
        with pytest.raises(RuntimeError, match="not initialized"):
            await tm.increment_usage(1)

        Path(db_path).unlink(missing_ok=True)

    async def test_not_initialized_error_get_statistics(self):
        """Test get_statistics fails when not initialized."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        tm = TranslationMemory(db_path)

        # Try to get stats without initializing
        with pytest.raises(RuntimeError, match="not initialized"):
            await tm.get_statistics()

        Path(db_path).unlink(missing_ok=True)

    async def test_del_method_cleanup(self):
        """Test __del__ method closes database."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        tm = TranslationMemory(db_path)
        await tm.initialize()

        # Manually call __del__
        tm.__del__()

        # Database should be closed
        # Note: we can't easily verify this, but we ensure it doesn't crash

        Path(db_path).unlink(missing_ok=True)

    async def test_search_similar_no_min_mqm_score(self, tm):
        """Test search without minimum MQM score filter."""
        # Add translation without MQM score
        await tm.add_translation(
            source="Test without score",
            translation="Prueba sin puntuación",
            source_lang="en",
            target_lang="es",
        )

        # Search without MQM filter
        results = await tm.search_similar(
            source="Test without score",
            source_lang="en",
            target_lang="es",
            min_mqm_score=None,  # No MQM filtering
            threshold=0.9,
        )

        # Should find the result even though it has no MQM score
        assert len(results) >= 1

    async def test_cosine_similarity_orthogonal_vectors(self, tm):
        """Test cosine similarity with orthogonal vectors."""
        import numpy as np

        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([0.0, 1.0, 0.0])

        similarity = tm._cosine_similarity(vec1, vec2)

        # Orthogonal vectors should have 0 similarity
        assert similarity == pytest.approx(0.0, abs=1e-6)

    async def test_search_similar_sorts_by_similarity(self, tm):
        """Test that search results are sorted by similarity."""
        # Add multiple translations
        await tm.add_translation(
            source="exact match",
            translation="coincidencia exacta",
            source_lang="en",
            target_lang="es",
            mqm_score=95.0,
        )

        await tm.add_translation(
            source="close match test",
            translation="coincidencia cercana",
            source_lang="en",
            target_lang="es",
            mqm_score=90.0,
        )

        await tm.add_translation(
            source="distant match example",
            translation="ejemplo lejano",
            source_lang="en",
            target_lang="es",
            mqm_score=85.0,
        )

        # Search for "exact match" - should return results sorted by similarity
        results = await tm.search_similar(
            source="exact match",
            source_lang="en",
            target_lang="es",
            threshold=0.3,
            limit=10,
            min_mqm_score=None,
        )

        # Results should be sorted by similarity (descending)
        if len(results) > 1:
            for i in range(len(results) - 1):
                assert results[i].similarity >= results[i + 1].similarity

    def test_create_schema_not_initialized_error(self):
        """Test _create_schema fails when db is None."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        tm = TranslationMemory(db_path)
        # Don't initialize, so db will be None

        with pytest.raises(RuntimeError, match="Database not initialized"):
            tm._create_schema()

        Path(db_path).unlink(missing_ok=True)
