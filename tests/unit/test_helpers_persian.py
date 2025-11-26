"""Unit tests for Persian language helper module.

Tests Persian-specific NLP functionality.
"""

import pytest

from kttc.helpers.persian import PersianLanguageHelper


@pytest.mark.unit
class TestPersianLanguageHelper:
    """Test PersianLanguageHelper functionality."""

    @pytest.fixture
    def helper(self) -> PersianLanguageHelper:
        """Create a helper instance."""
        return PersianLanguageHelper()

    def test_helper_initialization(self, helper: PersianLanguageHelper) -> None:
        """Test helper initializes correctly."""
        assert helper is not None
        assert helper.language_code == "fa"

    def test_helper_language_code(self, helper: PersianLanguageHelper) -> None:
        """Test language code is 'fa'."""
        assert helper.language_code == "fa"

    def test_helper_is_available(self, helper: PersianLanguageHelper) -> None:
        """Test is_available method."""
        result = helper.is_available()
        assert isinstance(result, bool)

    def test_tokenize_returns_list(self, helper: PersianLanguageHelper) -> None:
        """Test tokenize always returns a list."""
        text = "من به مدرسه می‌روم"
        tokens = helper.tokenize(text)
        assert isinstance(tokens, list)

    def test_check_grammar_returns_list(self, helper: PersianLanguageHelper) -> None:
        """Test check_grammar returns a list."""
        text = "این یک آزمایش است"
        errors = helper.check_grammar(text)
        assert isinstance(errors, list)

    def test_verify_word_exists(self, helper: PersianLanguageHelper) -> None:
        """Test verify_word_exists method."""
        text = "من به خانه می‌روم"
        # Word that exists
        assert helper.verify_word_exists("خانه", text) is True
        # Word that doesn't exist
        assert helper.verify_word_exists("مدرسه", text) is False

    def test_analyze_morphology(self, helper: PersianLanguageHelper) -> None:
        """Test morphological analysis."""
        text = "کتاب"
        result = helper.analyze_morphology(text)
        assert isinstance(result, list)


@pytest.mark.unit
class TestPersianHelperConstants:
    """Test Persian helper constants."""

    def test_helper_has_language_code(self) -> None:
        """Test helper defines language code."""
        helper = PersianLanguageHelper()
        assert hasattr(helper, "language_code")
        assert helper.language_code == "fa"


@pytest.mark.unit
class TestPersianHelperEdgeCases:
    """Test PersianLanguageHelper edge cases."""

    @pytest.fixture
    def helper(self) -> PersianLanguageHelper:
        """Create a helper instance."""
        return PersianLanguageHelper()

    def test_empty_text_tokenize(self, helper: PersianLanguageHelper) -> None:
        """Test tokenizing empty text."""
        tokens = helper.tokenize("")
        assert isinstance(tokens, list)
        assert len(tokens) == 0

    def test_empty_text_grammar_check(self, helper: PersianLanguageHelper) -> None:
        """Test grammar check on empty text."""
        errors = helper.check_grammar("")
        assert isinstance(errors, list)

    def test_whitespace_only_text(self, helper: PersianLanguageHelper) -> None:
        """Test text with only whitespace."""
        tokens = helper.tokenize("   ")
        assert isinstance(tokens, list)

    def test_mixed_script_text(self, helper: PersianLanguageHelper) -> None:
        """Test text with mixed Persian and English."""
        text = "من Python یاد می‌گیرم"
        tokens = helper.tokenize(text)
        assert isinstance(tokens, list)

    def test_rtl_text_handling(self, helper: PersianLanguageHelper) -> None:
        """Test RTL (right-to-left) text handling."""
        text = "متن فارسی"
        tokens = helper.tokenize(text)
        assert isinstance(tokens, list)
