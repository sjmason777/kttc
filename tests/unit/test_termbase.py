"""Unit tests for termbase module.

Tests terminology base functionality with temporary database.
"""

import tempfile
from collections.abc import Generator
from pathlib import Path

import pytest

from kttc.memory.termbase import TermEntry, TerminologyBase, TermViolation


@pytest.mark.unit
class TestTerminologyBase:
    """Test TerminologyBase functionality."""

    @pytest.fixture
    def temp_termbase(self) -> Generator[TerminologyBase, None, None]:
        """Provide a temporary terminology base for testing."""
        # Create temporary database file
        db_file = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        db_file.close()

        # Create termbase with temporary database
        termbase = TerminologyBase(db_file.name)
        termbase.initialize()

        yield termbase

        # Cleanup
        termbase.cleanup()
        Path(db_file.name).unlink(missing_ok=True)

    def test_initialization(self) -> None:
        """Test TerminologyBase initialization."""
        # Arrange
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            tmp.close()
            db_path = tmp.name

        # Act
        termbase = TerminologyBase(db_path)
        termbase.initialize()

        # Assert
        assert termbase._initialized is True
        assert termbase.db is not None

        # Cleanup
        termbase.cleanup()
        Path(db_path).unlink(missing_ok=True)

    def test_add_term(self, temp_termbase: TerminologyBase) -> None:
        """Test adding a term to termbase."""
        # Act
        term_id = temp_termbase.add_term(
            source_term="API",
            target_term="интерфейс программирования приложений",
            source_lang="en",
            target_lang="ru",
        )

        # Assert
        assert term_id > 0

    def test_add_term_with_domain(self, temp_termbase: TerminologyBase) -> None:
        """Test adding a term with domain to termbase."""
        # Act
        term_id = temp_termbase.add_term(
            source_term="API",
            target_term="интерфейс программирования приложений",
            source_lang="en",
            target_lang="ru",
            domain="technical",
        )

        # Assert
        assert term_id > 0

    def test_add_duplicate_term(self, temp_termbase: TerminologyBase) -> None:
        """Test adding a duplicate term updates existing entry."""
        # Arrange
        temp_termbase.add_term(
            source_term="API",
            target_term="интерфейс программирования приложений",
            source_lang="en",
            target_lang="ru",
        )

        # Act
        term_id = temp_termbase.add_term(
            source_term="API",
            target_term="API (Application Programming Interface)",
            source_lang="en",
            target_lang="ru",
        )

        # Assert
        assert term_id > 0

    def test_lookup_term(self, temp_termbase: TerminologyBase) -> None:
        """Test looking up a term in termbase."""
        # Arrange
        temp_termbase.add_term(
            source_term="API",
            target_term="интерфейс программирования приложений",
            source_lang="en",
            target_lang="ru",
        )

        # Act
        entries = temp_termbase.lookup_term("API", "en", "ru")

        # Assert
        assert len(entries) == 1
        assert entries[0].source_term == "API"
        assert entries[0].target_term == "интерфейс программирования приложений"
        assert entries[0].source_lang == "en"
        assert entries[0].target_lang == "ru"

    def test_lookup_term_with_domain(self, temp_termbase: TerminologyBase) -> None:
        """Test looking up a term with domain filter."""
        # Arrange
        temp_termbase.add_term(
            source_term="API",
            target_term="интерфейс программирования приложений",
            source_lang="en",
            target_lang="ru",
            domain="technical",
        )

        # Act
        entries = temp_termbase.lookup_term("API", "en", "ru", domain="technical")

        # Assert
        assert len(entries) == 1
        assert entries[0].domain == "technical"

    def test_lookup_nonexistent_term(self, temp_termbase: TerminologyBase) -> None:
        """Test looking up a term that doesn't exist."""
        # Act
        entries = temp_termbase.lookup_term("NONEXISTENT", "en", "ru")

        # Assert
        assert len(entries) == 0

    def test_validate_translation(self, temp_termbase: TerminologyBase) -> None:
        """Test validating a translation against termbase."""
        # Arrange
        temp_termbase.add_term(
            source_term="API",
            target_term="интерфейс программирования приложений",
            source_lang="en",
            target_lang="ru",
        )

        # Act
        violations = temp_termbase.validate_translation(
            source_text="Use the API to access data",
            translation="Используйте API для доступа к данным",
            source_lang="en",
            target_lang="ru",
        )

        # Assert
        assert len(violations) == 1
        assert violations[0].source_term == "API"
        assert violations[0].expected_terms == ["интерфейс программирования приложений"]
        assert violations[0].found_in_translation is False
        assert violations[0].severity == "major"

    def test_validate_translation_correct(self, temp_termbase: TerminologyBase) -> None:
        """Test validating a correct translation."""
        # Arrange
        temp_termbase.add_term(
            source_term="API",
            target_term="интерфейс программирования приложений",
            source_lang="en",
            target_lang="ru",
        )

        # Act
        violations = temp_termbase.validate_translation(
            source_text="Use the API to access data",
            translation="Используйте интерфейс программирования приложений для доступа к данным",
            source_lang="en",
            target_lang="ru",
        )

        # Assert
        assert len(violations) == 0

    def test_validate_translation_no_terms(self, temp_termbase: TerminologyBase) -> None:
        """Test validating translation with no terms in source."""
        # Act
        violations = temp_termbase.validate_translation(
            source_text="Hello world", translation="Привет мир", source_lang="en", target_lang="ru"
        )

        # Assert
        assert len(violations) == 0

    def test_add_glossary(self, temp_termbase: TerminologyBase) -> None:
        """Test adding terms from glossary."""
        # Arrange
        glossary = {
            "API": "интерфейс программирования приложений",
            "SDK": "набор средств разработки",
        }

        # Act
        count = temp_termbase.add_glossary(glossary, "en", "ru")

        # Assert
        assert count == 2

        # Verify terms were added
        entries = temp_termbase.lookup_term("API", "en", "ru")
        assert len(entries) == 1
        assert entries[0].target_term == "интерфейс программирования приложений"

    def test_get_statistics(self, temp_termbase: TerminologyBase) -> None:
        """Test getting termbase statistics."""
        # Arrange
        temp_termbase.add_term(
            source_term="API",
            target_term="интерфейс программирования приложений",
            source_lang="en",
            target_lang="ru",
        )
        temp_termbase.add_term(
            source_term="SDK",
            target_term="набор средств разработки",
            source_lang="en",
            target_lang="ru",
            domain="technical",
        )

        # Act
        stats = temp_termbase.get_statistics()

        # Assert
        assert stats["total_terms"] == 2
        assert "en-ru" in stats["language_pairs"]
        assert "technical" in stats["domains"]

    def test_extract_terms(self, temp_termbase: TerminologyBase) -> None:
        """Test extracting terms from text."""
        # Act
        terms = temp_termbase._extract_terms("Use the API and SDK to develop applications")

        # Assert
        assert "API" in terms
        assert "SDK" in terms
        assert "applications" in terms

    def test_extract_terms_with_phrases(self, temp_termbase: TerminologyBase) -> None:
        """Test extracting multi-word terms."""
        # Act
        terms = temp_termbase._extract_terms("Use the Application Programming Interface API")

        # Assert
        assert "Application Programming Interface" in terms
        assert "API" in terms

    def test_term_entry_model(self) -> None:
        """Test TermEntry model creation."""
        # Act
        entry = TermEntry(
            id=1,
            source_term="API",
            target_term="интерфейс программирования приложений",
            source_lang="en",
            target_lang="ru",
            domain="technical",
            definition="Application Programming Interface",
            usage_note="Use in technical contexts",
        )

        # Assert
        assert entry.source_term == "API"
        assert entry.target_term == "интерфейс программирования приложений"
        assert entry.source_lang == "en"
        assert entry.target_lang == "ru"
        assert entry.domain == "technical"

    def test_term_violation_model(self) -> None:
        """Test TermViolation model creation."""
        # Act
        violation = TermViolation(
            source_term="API",
            expected_terms=["интерфейс программирования приложений"],
            found_in_translation=False,
            severity="major",
        )

        # Assert
        assert violation.source_term == "API"
        assert violation.expected_terms == ["интерфейс программирования приложений"]
        assert violation.found_in_translation is False
        assert violation.severity == "major"

    def test_term_violation_invalid_severity(self) -> None:
        """Test TermViolation with invalid severity."""
        # Act & Assert
        with pytest.raises(ValueError):
            TermViolation(
                source_term="API",
                expected_terms=["интерфейс программирования приложений"],
                found_in_translation=False,
                severity="invalid",
            )

    def test_cleanup(self) -> None:
        """Test cleanup method."""
        # Arrange
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            tmp.close()
            db_path = tmp.name

        termbase = TerminologyBase(db_path)
        termbase.initialize()

        # Act
        termbase.cleanup()

        # Assert
        assert termbase._initialized is False
        assert termbase.db is None

        # Cleanup
        Path(db_path).unlink(missing_ok=True)

    def test_initialization_failure(self) -> None:
        """Test initialization with invalid path."""
        # Arrange
        termbase = TerminologyBase("/invalid/path/to/database.db")

        # Act & Assert
        with pytest.raises(RuntimeError):
            termbase.initialize()

    def test_add_term_without_initialization(self) -> None:
        """Test adding term without initialization."""
        # Arrange
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            tmp.close()
            db_path = tmp.name

        termbase = TerminologyBase(db_path)
        # Not initialized

        # Act & Assert
        with pytest.raises(RuntimeError):
            termbase.add_term("API", "интерфейс", "en", "ru")

        # Cleanup
        Path(db_path).unlink(missing_ok=True)
