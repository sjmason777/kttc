"""Unit tests for memory modules (termbase and translation memory).

Tests key functionality with mocked database operations.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from kttc.memory.termbase import TermEntry, TerminologyBase, TermViolation
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
class TestTermEntry:
    """Test TermEntry model."""

    def test_term_entry_creation(self) -> None:
        """Test creating a valid term entry."""
        # Arrange & Act
        entry = TermEntry(
            source_term="API",
            target_term="interfaz de programación",
            source_lang="en",
            target_lang="es",
            domain="technical",
        )

        # Assert
        assert entry.source_term == "API"
        assert entry.target_term == "interfaz de programación"
        assert entry.source_lang == "en"
        assert entry.target_lang == "es"
        assert entry.domain == "technical"

    def test_term_entry_invalid_lang_code(self) -> None:
        """Test that invalid language codes are rejected."""
        # Act & Assert
        with pytest.raises(Exception):  # Pydantic validation error
            TermEntry(
                source_term="test",
                target_term="prueba",
                source_lang="eng",  # Invalid: should be 2 letters
                target_lang="es",
            )


@pytest.mark.unit
class TestTermViolation:
    """Test TermViolation model."""

    def test_term_violation_creation(self) -> None:
        """Test creating a term violation."""
        # Arrange & Act
        violation = TermViolation(
            source_term="API",
            expected_terms=["interfaz de programación", "API"],
            found_in_translation=False,
            severity="major",
        )

        # Assert
        assert violation.source_term == "API"
        assert len(violation.expected_terms) == 2
        assert violation.found_in_translation is False
        assert violation.severity == "major"


@pytest.mark.unit
class TestTerminologyBase:
    """Test TerminologyBase with mocked database."""

    def test_initialization(self) -> None:
        """Test termbase initialization."""
        # Arrange
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"

            # Act
            termbase = TerminologyBase(db_path)

            # Assert
            assert termbase.db_path == db_path
            assert termbase.db is None
            assert termbase._initialized is False

    def test_add_term_creates_entry(self) -> None:
        """Test adding a term to the termbase."""
        # Arrange
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            termbase = TerminologyBase(db_path)
            termbase.initialize()

            # Act
            result = termbase.add_term(
                source_term="test",
                target_term="prueba",
                source_lang="en",
                target_lang="es",
                domain="general",
            )

            # Assert
            assert result == 1

            # Cleanup
            termbase.cleanup()

    def test_db_path_conversion(self) -> None:
        """Test that string paths are converted to Path objects."""
        # Arrange & Act
        termbase = TerminologyBase("test.db")

        # Assert
        assert isinstance(termbase.db_path, Path)
        assert termbase.db_path.name == "test.db"


# Translation Memory tests


@pytest.mark.unit
class TestTMSearchResult:
    """Test TMSearchResult model."""

    def test_search_result_creation(self) -> None:
        """Test creating a TM search result."""
        # Arrange
        segment = TMSegment(
            source_text="Hello world",
            translation="Hola mundo",
            source_lang="en",
            target_lang="es",
        )

        # Act
        result = TMSearchResult(
            segment=segment,
            similarity=0.95,
        )

        # Assert
        assert result.segment.source_text == "Hello world"
        assert result.segment.translation == "Hola mundo"
        assert result.similarity == 0.95


@pytest.mark.unit
@requires_sentence_transformers
class TestTranslationMemory:
    """Test TranslationMemory with mocked database."""

    def test_initialization(self) -> None:
        """Test TM initialization."""
        # Arrange
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_tm.db"

            # Act
            tm = TranslationMemory(db_path)

            # Assert
            assert tm.db_path == db_path
            assert tm.db is None
            assert tm._initialized is False

    def test_add_translation_stores_entry(self) -> None:
        """Test adding translation to memory."""
        # Arrange
        import numpy as np

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_tm.db"
            tm = TranslationMemory(db_path)

            # Mock the sentence transformer
            with patch("sentence_transformers.SentenceTransformer") as mock_encoder_cls:
                mock_encoder = MagicMock()
                mock_encoder.encode.return_value = np.array([[0.1, 0.2, 0.3]])
                mock_encoder_cls.return_value = mock_encoder

                tm.initialize()

                # Act
                result = tm.add_translation(
                    source="Hello",
                    translation="Hola",
                    source_lang="en",
                    target_lang="es",
                    mqm_score=95.0,
                )

                # Assert
                assert result == 1

            # Cleanup
            tm.cleanup()
