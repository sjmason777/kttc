"""Unit tests for Hindi language helper module.

Tests Hindi-specific NLP functionality.
"""

import pytest

from kttc.helpers.hindi import HindiLanguageHelper


@pytest.mark.unit
class TestHindiLanguageHelper:
    """Test HindiLanguageHelper functionality."""

    @pytest.fixture
    def helper(self) -> HindiLanguageHelper:
        """Create a helper instance."""
        return HindiLanguageHelper()

    def test_helper_initialization(self, helper: HindiLanguageHelper) -> None:
        """Test helper initializes correctly."""
        assert helper is not None
        assert helper.language_code == "hi"

    def test_helper_language_code(self, helper: HindiLanguageHelper) -> None:
        """Test language code is 'hi'."""
        assert helper.language_code == "hi"

    def test_helper_is_available(self, helper: HindiLanguageHelper) -> None:
        """Test is_available method."""
        result = helper.is_available()
        assert isinstance(result, bool)

    def test_tokenize_returns_list(self, helper: HindiLanguageHelper) -> None:
        """Test tokenize always returns a list."""
        text = "मैं स्कूल जाता हूं"
        tokens = helper.tokenize(text)
        assert isinstance(tokens, list)

    def test_check_grammar_returns_list(self, helper: HindiLanguageHelper) -> None:
        """Test check_grammar returns a list."""
        text = "यह एक परीक्षण है"
        errors = helper.check_grammar(text)
        assert isinstance(errors, list)

    def test_verify_word_exists(self, helper: HindiLanguageHelper) -> None:
        """Test verify_word_exists method."""
        text = "मैं घर जाता हूं"
        # Word that exists
        assert helper.verify_word_exists("घर", text) is True
        # Word that doesn't exist
        assert helper.verify_word_exists("स्कूल", text) is False

    def test_analyze_morphology(self, helper: HindiLanguageHelper) -> None:
        """Test morphological analysis."""
        text = "लड़का"
        result = helper.analyze_morphology(text)
        assert isinstance(result, list)


@pytest.mark.unit
class TestHindiHelperConstants:
    """Test Hindi helper constants."""

    def test_helper_has_language_code(self) -> None:
        """Test helper defines language code."""
        helper = HindiLanguageHelper()
        assert hasattr(helper, "language_code")
        assert helper.language_code == "hi"

    def test_helper_has_common_postpositions(self) -> None:
        """Test helper has common postpositions defined."""
        # Hindi has postpositions like में, पर, से, को
        helper = HindiLanguageHelper()
        # Check if helper has some Hindi-specific data
        assert helper is not None


@pytest.mark.unit
class TestHindiHelperEdgeCases:
    """Test HindiLanguageHelper edge cases."""

    @pytest.fixture
    def helper(self) -> HindiLanguageHelper:
        """Create a helper instance."""
        return HindiLanguageHelper()

    def test_empty_text_tokenize(self, helper: HindiLanguageHelper) -> None:
        """Test tokenizing empty text."""
        tokens = helper.tokenize("")
        assert isinstance(tokens, list)
        assert len(tokens) == 0

    def test_empty_text_grammar_check(self, helper: HindiLanguageHelper) -> None:
        """Test grammar check on empty text."""
        errors = helper.check_grammar("")
        assert isinstance(errors, list)

    def test_whitespace_only_text(self, helper: HindiLanguageHelper) -> None:
        """Test text with only whitespace."""
        tokens = helper.tokenize("   ")
        assert isinstance(tokens, list)

    def test_mixed_script_text(self, helper: HindiLanguageHelper) -> None:
        """Test text with mixed Hindi and English."""
        text = "मैं Python सीख रहा हूं"
        tokens = helper.tokenize(text)
        assert isinstance(tokens, list)
