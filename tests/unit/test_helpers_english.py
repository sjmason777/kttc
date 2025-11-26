"""Unit tests for English language helper module.

Tests English-specific NLP functionality.
"""

import pytest

from kttc.helpers.english import EnglishLanguageHelper


@pytest.mark.unit
class TestEnglishLanguageHelper:
    """Test EnglishLanguageHelper functionality."""

    @pytest.fixture
    def helper(self) -> EnglishLanguageHelper:
        """Create a helper instance."""
        return EnglishLanguageHelper()

    def test_helper_initialization(self, helper: EnglishLanguageHelper) -> None:
        """Test helper initializes correctly."""
        assert helper is not None
        assert helper.language_code == "en"

    def test_helper_language_code(self, helper: EnglishLanguageHelper) -> None:
        """Test language code is 'en'."""
        assert helper.language_code == "en"

    def test_helper_is_available(self, helper: EnglishLanguageHelper) -> None:
        """Test is_available method."""
        result = helper.is_available()
        assert isinstance(result, bool)

    def test_tokenize_simple_text(self, helper: EnglishLanguageHelper) -> None:
        """Test tokenizing simple English text."""
        text = "Hello world"
        if helper.is_available():
            tokens = helper.tokenize(text)
            assert isinstance(tokens, list)

    def test_check_grammar_simple(self, helper: EnglishLanguageHelper) -> None:
        """Test grammar check on simple text."""
        text = "This is a test sentence."
        if helper.is_available():
            errors = helper.check_grammar(text)
            assert isinstance(errors, list)


@pytest.mark.unit
class TestEnglishHelperMethods:
    """Test EnglishLanguageHelper methods."""

    @pytest.fixture
    def helper(self) -> EnglishLanguageHelper:
        """Create a helper instance."""
        return EnglishLanguageHelper()

    def test_analyze_morphology(self, helper: EnglishLanguageHelper) -> None:
        """Test morphological analysis."""
        word = "running"
        if helper.is_available():
            result = helper.analyze_morphology(word)
            assert result is not None
