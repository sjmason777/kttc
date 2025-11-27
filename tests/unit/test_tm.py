"""Unit tests for translation memory module.

Tests translation memory functionality with temporary database.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from kttc.memory.tm import TMSearchResult, TMSegment, TranslationMemory

# Check if sentence_transformers is available
try:
    import sentence_transformers  # noqa: F401

    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

requires_sentence_transformers = pytest.mark.skipif(
    not HAS_SENTENCE_TRANSFORMERS,
    reason="sentence_transformers not installed",
)


@pytest.mark.unit
@requires_sentence_transformers
class TestTranslationMemory:
    """Test TranslationMemory functionality."""

    @pytest.fixture
    def temp_tm(self) -> TranslationMemory:
        """Provide a temporary translation memory for testing."""
        # Create temporary database file
        db_file = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        db_file.close()

        # Create TM with temporary database
        tm = TranslationMemory(db_file.name)

        # Mock the sentence transformer
        with patch("sentence_transformers.SentenceTransformer") as mock_encoder:
            mock_model = MagicMock()
            mock_model.encode.return_value = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
            mock_encoder.return_value = mock_model
            tm.initialize()

        yield tm

        # Cleanup
        tm.cleanup()
        Path(db_file.name).unlink(missing_ok=True)

    def test_initialization(self) -> None:
        """Test TranslationMemory initialization."""
        # Arrange
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            tmp.close()
            db_path = tmp.name

        # Act
        tm = TranslationMemory(db_path)

        # Mock the sentence transformer
        with patch("sentence_transformers.SentenceTransformer") as mock_encoder:
            mock_model = MagicMock()
            mock_model.encode.return_value = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
            mock_encoder.return_value = mock_model

            tm.initialize()

        # Assert
        assert tm._initialized is True
        assert tm.db is not None
        assert tm.encoder is not None

        # Cleanup
        tm.cleanup()
        Path(db_path).unlink(missing_ok=True)

    def test_initialization_import_error(self) -> None:
        """Test TranslationMemory initialization with import error."""
        # Arrange
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            tmp.close()
            db_path = tmp.name

        tm = TranslationMemory(db_path)

        # Act & Assert
        with patch.dict("sys.modules", {"sentence_transformers": None}), pytest.raises(RuntimeError, match="Failed to import sentence-transformers"):
            tm.initialize()

        # Cleanup
        Path(db_path).unlink(missing_ok=True)

    def test_add_translation(self, temp_tm: TranslationMemory) -> None:
        """Test adding a translation to TM."""
        # Act
        segment_id = temp_tm.add_translation(
            source="Hello world",
            translation="Hola mundo",
            source_lang="en",
            target_lang="es",
            mqm_score=95.0,
        )

        # Assert
        assert segment_id > 0

    def test_add_translation_with_domain(self, temp_tm: TranslationMemory) -> None:
        """Test adding a translation with domain to TM."""
        # Act
        segment_id = temp_tm.add_translation(
            source="Hello world",
            translation="Hola mundo",
            source_lang="en",
            target_lang="es",
            domain="general",
        )

        # Assert
        assert segment_id > 0

    def test_add_duplicate_translation(self, temp_tm: TranslationMemory) -> None:
        """Test adding a duplicate translation updates existing entry."""
        # Arrange
        temp_tm.add_translation(
            source="Hello world", translation="Hola mundo", source_lang="en", target_lang="es"
        )

        # Act
        segment_id = temp_tm.add_translation(
            source="Hello world", translation="¡Hola mundo!", source_lang="en", target_lang="es"
        )

        # Assert
        assert segment_id > 0

    def test_search_similar(self, temp_tm: TranslationMemory) -> None:
        """Test searching for similar translations."""
        # Arrange
        temp_tm.add_translation(
            source="Hello world",
            translation="Hola mundo",
            source_lang="en",
            target_lang="es",
            mqm_score=95.0,
        )

        # Mock cosine similarity to return a high value
        with patch.object(temp_tm, "_cosine_similarity", return_value=0.9):
            # Act
            results = temp_tm.search_similar(
                source="Hello world!", source_lang="en", target_lang="es"
            )

        # Assert
        assert len(results) == 1
        assert results[0].segment.source_text == "Hello world"
        assert results[0].segment.translation == "Hola mundo"
        assert results[0].similarity == 0.9

    def test_search_similar_below_threshold(self, temp_tm: TranslationMemory) -> None:
        """Test searching for similar translations below threshold."""
        # Arrange
        temp_tm.add_translation(
            source="Hello world", translation="Hola mundo", source_lang="en", target_lang="es"
        )

        # Mock cosine similarity to return a low value
        with patch.object(temp_tm, "_cosine_similarity", return_value=0.1):
            # Act
            results = temp_tm.search_similar(
                source="Hello world!", source_lang="en", target_lang="es", threshold=0.75
            )

        # Assert
        assert len(results) == 0

    def test_search_similar_with_mqm_filter(self, temp_tm: TranslationMemory) -> None:
        """Test searching with MQM score filter."""
        # Arrange
        temp_tm.add_translation(
            source="Hello world",
            translation="Hola mundo",
            source_lang="en",
            target_lang="es",
            mqm_score=95.0,
        )

        temp_tm.add_translation(
            source="Goodbye world",
            translation="Adiós mundo",
            source_lang="en",
            target_lang="es",
            mqm_score=80.0,
        )

        # Mock cosine similarity to return high values
        with patch.object(temp_tm, "_cosine_similarity", return_value=0.9):
            # Act
            results = temp_tm.search_similar(
                source="Hello world!", source_lang="en", target_lang="es", min_mqm_score=90.0
            )

        # Assert
        assert len(results) == 1
        assert results[0].segment.mqm_score == 95.0

    def test_search_similar_limit(self, temp_tm: TranslationMemory) -> None:
        """Test searching with limit."""
        # Arrange - add entries with mqm_score to pass the default filter
        for i in range(10):
            temp_tm.add_translation(
                source=f"Hello world {i}",
                translation=f"Hola mundo {i}",
                source_lang="en",
                target_lang="es",
                mqm_score=90.0,  # Above default min_mqm_score of 85.0
            )

        # Mock cosine similarity to return high values
        with patch.object(temp_tm, "_cosine_similarity", return_value=0.9):
            # Act
            results = temp_tm.search_similar(
                source="Hello world", source_lang="en", target_lang="es", limit=3
            )

        # Assert
        assert len(results) == 3

    def test_increment_usage(self, temp_tm: TranslationMemory) -> None:
        """Test incrementing usage counter."""
        # Arrange
        segment_id = temp_tm.add_translation(
            source="Hello world", translation="Hola mundo", source_lang="en", target_lang="es"
        )

        # Act
        temp_tm.increment_usage(segment_id)

        # Assert
        # We can't directly check the usage count without a getter method,
        # but we can verify the method doesn't raise an exception

    def test_get_statistics(self, temp_tm: TranslationMemory) -> None:
        """Test getting TM statistics."""
        # Arrange
        temp_tm.add_translation(
            source="Hello world",
            translation="Hola mundo",
            source_lang="en",
            target_lang="es",
            domain="general",
        )

        # Act
        stats = temp_tm.get_statistics()

        # Assert
        assert stats["total_segments"] == 1
        assert "en-es" in stats["language_pairs"]
        assert "general" in stats["domains"]

    def test_cosine_similarity(self, temp_tm: TranslationMemory) -> None:
        """Test cosine similarity calculation."""
        # Arrange
        vec1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        vec2 = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        orthogonal = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)

        # Act
        similarity_same = temp_tm._cosine_similarity(vec1, vec1)
        similarity_orthogonal = temp_tm._cosine_similarity(vec1, vec2)
        similarity_perpendicular = temp_tm._cosine_similarity(vec1, orthogonal)

        # Assert
        assert similarity_same == 1.0
        assert similarity_orthogonal == 0.0
        assert similarity_perpendicular == 0.0

    def test_cosine_similarity_zero_vectors(self, temp_tm: TranslationMemory) -> None:
        """Test cosine similarity with zero vectors."""
        # Arrange
        zero_vec = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

        # Act
        similarity = temp_tm._cosine_similarity(zero_vec, vec)

        # Assert
        assert similarity == 0.0

    def test_cleanup(self) -> None:
        """Test cleanup method."""
        # Arrange
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            tmp.close()
            db_path = tmp.name

        tm = TranslationMemory(db_path)

        # Mock the sentence transformer
        with patch("sentence_transformers.SentenceTransformer") as mock_encoder:
            mock_model = MagicMock()
            mock_model.encode.return_value = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
            mock_encoder.return_value = mock_model
            tm.initialize()

        # Act
        tm.cleanup()

        # Assert
        assert tm._initialized is False
        assert tm.db is None
        assert tm.encoder is None

        # Cleanup
        Path(db_path).unlink(missing_ok=True)

    def test_add_translation_without_initialization(self) -> None:
        """Test adding translation without initialization."""
        # Arrange
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            tmp.close()
            db_path = tmp.name

        tm = TranslationMemory(db_path)
        # Not initialized

        # Act & Assert
        with pytest.raises(RuntimeError):
            tm.add_translation("Hello", "Hola", "en", "es")

        # Cleanup
        Path(db_path).unlink(missing_ok=True)

    def test_search_similar_without_initialization(self) -> None:
        """Test searching without initialization."""
        # Arrange
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            tmp.close()
            db_path = tmp.name

        tm = TranslationMemory(db_path)
        # Not initialized

        # Act & Assert
        with pytest.raises(RuntimeError):
            tm.search_similar("Hello", "en", "es")

        # Cleanup
        Path(db_path).unlink(missing_ok=True)

    def test_increment_usage_without_initialization(self) -> None:
        """Test incrementing usage without initialization."""
        # Arrange
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            tmp.close()
            db_path = tmp.name

        tm = TranslationMemory(db_path)
        # Not initialized

        # Act & Assert
        with pytest.raises(RuntimeError):
            tm.increment_usage(1)

        # Cleanup
        Path(db_path).unlink(missing_ok=True)

    def test_get_statistics_without_initialization(self) -> None:
        """Test getting statistics without initialization."""
        # Arrange
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            tmp.close()
            db_path = tmp.name

        tm = TranslationMemory(db_path)
        # Not initialized

        # Act & Assert
        with pytest.raises(RuntimeError):
            tm.get_statistics()

        # Cleanup
        Path(db_path).unlink(missing_ok=True)


@pytest.mark.unit
class TestTMSegment:
    """Test TMSegment model."""

    def test_segment_creation(self) -> None:
        """Test creating TMSegment."""
        # Act
        segment = TMSegment(
            id=1,
            source_text="Hello world",
            translation="Hola mundo",
            source_lang="en",
            target_lang="es",
            domain="general",
            mqm_score=95.0,
            usage_count=3,
        )

        # Assert
        assert segment.source_text == "Hello world"
        assert segment.translation == "Hola mundo"
        assert segment.source_lang == "en"
        assert segment.target_lang == "es"
        assert segment.domain == "general"
        assert segment.mqm_score == 95.0
        assert segment.usage_count == 3

    def test_segment_invalid_language(self) -> None:
        """Test creating TMSegment with invalid language."""
        # Act & Assert
        with pytest.raises(ValueError):
            TMSegment(
                source_text="Hello world",
                translation="Hola mundo",
                source_lang="eng",  # Too long
                target_lang="es",
            )

    def test_segment_invalid_mqm_score(self) -> None:
        """Test creating TMSegment with invalid MQM score."""
        # Act & Assert
        with pytest.raises(ValueError):
            TMSegment(
                source_text="Hello world",
                translation="Hola mundo",
                source_lang="en",
                target_lang="es",
                mqm_score=150.0,  # Too high
            )


@pytest.mark.unit
class TestTMSearchResult:
    """Test TMSearchResult model."""

    def test_search_result_creation(self) -> None:
        """Test creating TMSearchResult."""
        # Arrange
        segment = TMSegment(
            source_text="Hello world", translation="Hola mundo", source_lang="en", target_lang="es"
        )

        # Act
        result = TMSearchResult(segment=segment, similarity=0.95)

        # Assert
        assert result.segment.source_text == "Hello world"
        assert result.similarity == 0.95

    def test_search_result_invalid_similarity(self) -> None:
        """Test creating TMSearchResult with invalid similarity."""
        # Arrange
        segment = TMSegment(
            source_text="Hello world", translation="Hola mundo", source_lang="en", target_lang="es"
        )

        # Act & Assert
        with pytest.raises(ValueError):
            TMSearchResult(segment=segment, similarity=1.5)  # Too high
