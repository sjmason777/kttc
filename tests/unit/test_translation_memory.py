"""Tests for Translation Memory module."""

import tempfile
from pathlib import Path

import pytest

from kttc.memory.tm import TMSegment, TranslationMemory

pytestmark = pytest.mark.asyncio


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

        # Search to verify usage count
        results = await tm.search_similar(source="Test", source_lang="en", target_lang="es")

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
