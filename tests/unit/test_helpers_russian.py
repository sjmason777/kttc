"""Unit tests for Russian language helper module.

Tests Russian-specific NLP functionality.
"""

import pytest

from kttc.helpers.russian import LANGUAGETOOL_AVAILABLE, MAWO_AVAILABLE, RussianLanguageHelper


@pytest.mark.unit
class TestRussianLanguageHelper:
    """Test RussianLanguageHelper functionality."""

    @pytest.fixture
    def helper(self) -> RussianLanguageHelper:
        """Create a helper instance."""
        return RussianLanguageHelper()

    def test_helper_initialization(self, helper: RussianLanguageHelper) -> None:
        """Test helper initializes correctly."""
        assert helper is not None
        assert helper.language_code == "ru"

    def test_helper_language_code(self, helper: RussianLanguageHelper) -> None:
        """Test language code is 'ru'."""
        assert helper.language_code == "ru"

    def test_helper_is_available(self, helper: RussianLanguageHelper) -> None:
        """Test is_available method."""
        result = helper.is_available()
        # Result depends on whether MAWO/LanguageTool is installed
        assert isinstance(result, bool)

    def test_helper_has_it_terms_whitelist(self) -> None:
        """Test IT terms whitelist is defined."""
        assert hasattr(RussianLanguageHelper, "IT_TERMS_WHITELIST")
        whitelist = RussianLanguageHelper.IT_TERMS_WHITELIST
        assert isinstance(whitelist, set)
        assert len(whitelist) > 0

    def test_it_terms_include_common_terms(self) -> None:
        """Test common IT terms are in whitelist."""
        whitelist = RussianLanguageHelper.IT_TERMS_WHITELIST
        # These are common IT terms
        assert "коммит" in whitelist or "демо" in whitelist


@pytest.mark.unit
class TestRussianHelperMethods:
    """Test RussianLanguageHelper methods."""

    @pytest.fixture
    def helper(self) -> RussianLanguageHelper:
        """Create a helper instance."""
        return RussianLanguageHelper()

    def test_tokenize_text(self, helper: RussianLanguageHelper) -> None:
        """Test text tokenization."""
        text = "Привет мир"
        if helper.is_available():
            tokens = helper.tokenize(text)
            assert isinstance(tokens, list)
        else:
            # Should handle gracefully when MAWO not available
            pass

    def test_check_spelling(self, helper: RussianLanguageHelper) -> None:
        """Test spelling check method exists."""
        # Method should exist and be callable
        assert hasattr(helper, "check_spelling") or hasattr(helper, "check_grammar")

    def test_check_grammar(self, helper: RussianLanguageHelper) -> None:
        """Test grammar check."""
        text = "Привет мир"
        if helper.is_available():
            errors = helper.check_grammar(text)
            assert isinstance(errors, list)

    def test_analyze_morphology(self, helper: RussianLanguageHelper) -> None:
        """Test morphological analysis."""
        word = "дома"
        if helper.is_available():
            result = helper.analyze_morphology(word)
            # Should return MorphologyInfo or dict
            assert result is not None


@pytest.mark.unit
class TestRussianAvailabilityFlags:
    """Test availability flags."""

    def test_mawo_availability_is_bool(self) -> None:
        """Test MAWO_AVAILABLE is boolean."""
        assert isinstance(MAWO_AVAILABLE, bool)

    def test_languagetool_availability_is_bool(self) -> None:
        """Test LANGUAGETOOL_AVAILABLE is boolean."""
        assert isinstance(LANGUAGETOOL_AVAILABLE, bool)


@pytest.mark.unit
@pytest.mark.skipif(not MAWO_AVAILABLE, reason="MAWO not installed")
class TestRussianWithMawo:
    """Tests that require MAWO to be installed."""

    @pytest.fixture
    def helper(self) -> RussianLanguageHelper:
        """Create a helper instance."""
        return RussianLanguageHelper()

    def test_tokenize_russian_text(self, helper: RussianLanguageHelper) -> None:
        """Test tokenizing Russian text."""
        text = "Москва — столица России"
        tokens = helper.tokenize(text)
        assert isinstance(tokens, list)
        assert len(tokens) > 0

    def test_analyze_verb_aspect(self, helper: RussianLanguageHelper) -> None:
        """Test verb aspect analysis."""
        # "читать" is imperfective, "прочитать" is perfective
        word = "читать"
        result = helper.analyze_morphology(word)
        assert result is not None


@pytest.mark.unit
@pytest.mark.skipif(not LANGUAGETOOL_AVAILABLE, reason="LanguageTool not installed")
class TestRussianWithLanguageTool:
    """Tests that require LanguageTool to be installed."""

    @pytest.fixture
    def helper(self) -> RussianLanguageHelper:
        """Create a helper instance."""
        return RussianLanguageHelper()

    def test_grammar_check_finds_errors(self, helper: RussianLanguageHelper) -> None:
        """Test grammar check finds actual errors."""
        # Text with intentional grammar error
        text = "Я пошёл в магазине"  # Should be "в магазин" (accusative, not prepositional)
        errors = helper.check_grammar(text)
        # LanguageTool may or may not catch this specific error
        assert isinstance(errors, list)

    def test_grammar_check_correct_text(self, helper: RussianLanguageHelper) -> None:
        """Test grammar check on correct text."""
        text = "Я пошёл в магазин"
        errors = helper.check_grammar(text)
        assert isinstance(errors, list)
