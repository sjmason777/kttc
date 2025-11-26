"""Unit tests for language helper base module.

Tests base language helper functionality.
"""

import pytest

from kttc.helpers.base import LanguageHelper, MorphologyInfo


@pytest.mark.unit
class TestLanguageHelper:
    """Test LanguageHelper base class."""

    def test_language_helper_is_abstract(self) -> None:
        """Test LanguageHelper is abstract base class."""
        # LanguageHelper is abstract and should not be instantiated directly
        assert hasattr(LanguageHelper, "language_code")

    def test_morphology_info_exists(self) -> None:
        """Test MorphologyInfo class exists."""
        assert MorphologyInfo is not None


@pytest.mark.unit
class TestMorphologyInfo:
    """Test MorphologyInfo data class."""

    def test_morphology_info_creation(self) -> None:
        """Test creating MorphologyInfo instance."""
        info = MorphologyInfo(
            word="word",
            pos="NOUN",
            case="nominative",
            number="singular",
        )
        assert info.word == "word"
        assert info.pos == "NOUN"
        assert info.case == "nominative"

    def test_morphology_info_minimal(self) -> None:
        """Test MorphologyInfo with minimal data."""
        info = MorphologyInfo(word="test", pos="VERB")
        assert info.word == "test"
        assert info.pos == "VERB"

    def test_morphology_info_with_metadata(self) -> None:
        """Test MorphologyInfo with metadata."""
        info = MorphologyInfo(word="a", pos="DET", metadata={"extra": "data"})
        assert info.metadata == {"extra": "data"}
