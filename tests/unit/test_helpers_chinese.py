"""Unit tests for Chinese language helper module.

Tests Chinese-specific NLP functionality.
"""

from unittest.mock import patch

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

    def test_tokenize_simple_text(self) -> None:
        """Test tokenizing simple Chinese text with mocked jieba."""
        # Use mocking to avoid jieba state pollution issues
        with patch("kttc.helpers.chinese.HANLP_AVAILABLE", False):
            with patch("kttc.helpers.chinese.JIEBA_AVAILABLE", True):
                with patch("kttc.helpers.chinese.SPACY_AVAILABLE", False):
                    with patch("kttc.helpers.chinese.jieba") as mock_jieba:
                        # Mock jieba.cut to return expected tokens
                        mock_jieba.cut.return_value = ["你好", "世界"]

                        helper = ChineseLanguageHelper()
                        text = "你好世界"
                        tokens = helper.tokenize(text)

                        assert isinstance(tokens, list)
                        # With mocked jieba, we should get tokens
                        assert len(tokens) == 2
                        # Verify jieba.cut was called with the text
                        mock_jieba.cut.assert_called_once_with(text)

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
