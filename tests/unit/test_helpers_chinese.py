"""Unit tests for Chinese language helper module.

Tests Chinese-specific NLP functionality.
"""

import pytest

from kttc.helpers.chinese import ChineseLanguageHelper


@pytest.mark.unit
class TestChineseLanguageHelper:
    """Test ChineseLanguageHelper functionality."""

    @pytest.fixture
    def helper(self) -> ChineseLanguageHelper:
        """Create a helper instance."""
        return ChineseLanguageHelper()

    def test_helper_initialization(self, helper: ChineseLanguageHelper) -> None:
        """Test helper initializes correctly."""
        assert helper is not None
        assert helper.language_code == "zh"

    def test_helper_language_code(self, helper: ChineseLanguageHelper) -> None:
        """Test language code is 'zh'."""
        assert helper.language_code == "zh"

    def test_helper_is_available(self, helper: ChineseLanguageHelper) -> None:
        """Test is_available method."""
        result = helper.is_available()
        assert isinstance(result, bool)

    @pytest.mark.xfail(reason="Flaky in full test suite due to jieba state pollution")
    def test_tokenize_simple_text(self, helper: ChineseLanguageHelper) -> None:
        """Test tokenizing simple Chinese text."""
        text = "你好世界"
        # Tokenize always returns a list (even if empty when deps unavailable)
        tokens = helper.tokenize(text)
        assert isinstance(tokens, list)
        # If dependencies are available, we should get non-empty tokens
        if helper.is_available():
            assert len(tokens) > 0

    def test_check_grammar_simple(self, helper: ChineseLanguageHelper) -> None:
        """Test grammar check on simple text."""
        text = "这是一个测试"
        if helper.is_available():
            errors = helper.check_grammar(text)
            assert isinstance(errors, list)


@pytest.mark.unit
class TestChineseHelperConstants:
    """Test Chinese helper constants."""

    def test_helper_has_language_code(self) -> None:
        """Test helper defines language code."""
        helper = ChineseLanguageHelper()
        assert hasattr(helper, "language_code")
        assert helper.language_code == "zh"
