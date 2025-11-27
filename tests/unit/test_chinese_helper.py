"""Unit tests for ChineseLanguageHelper.

Tests Chinese language helper with mocked HanLP, jieba, and spaCy.
Focus: Fast, isolated tests that improve coverage.
"""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from kttc.core.models import ErrorAnnotation, ErrorSeverity

# ============================================================================
# Mock spaCy-like objects
# ============================================================================


class MockToken:
    """Mock spaCy Token."""

    def __init__(
        self,
        text: str,
        idx: int,
        pos_: str = "NOUN",
        is_punct: bool = False,
    ):
        self.text = text
        self.idx = idx
        self.pos_ = pos_
        self.is_punct = is_punct


class MockEntity:
    """Mock spaCy Entity."""

    def __init__(self, text: str, label_: str, start_char: int, end_char: int):
        self.text = text
        self.label_ = label_
        self.start_char = start_char
        self.end_char = end_char


class MockDoc:
    """Mock spaCy Doc."""

    def __init__(self, tokens: list[MockToken], ents: list[MockEntity] | None = None):
        self._tokens = tokens
        self.ents = ents or []

    def __iter__(self):
        return iter(self._tokens)

    def __len__(self):
        return len(self._tokens)

    @property
    def sents(self):
        """Mock sentence iterator."""
        return [self]


class MockSpacyNLP:
    """Mock spaCy NLP pipeline."""

    def __init__(self, doc: MockDoc | None = None):
        self._doc = doc

    def __call__(self, text: str) -> MockDoc:
        if self._doc:
            return self._doc
        # Default: character-level tokenization for Chinese
        tokens = [MockToken(char, i) for i, char in enumerate(text) if char.strip()]
        return MockDoc(tokens)


class MockHanLP:
    """Mock HanLP pipeline."""

    def __init__(self, result: dict[str, Any] | None = None):
        self._result = result or {"tok": [], "pos": [], "ner": []}

    def __call__(self, text: str) -> dict[str, Any]:
        return self._result


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def mock_all_unavailable():
    """Mock all Chinese NLP tools as unavailable."""
    with patch("kttc.helpers.chinese.HANLP_AVAILABLE", False):
        with patch("kttc.helpers.chinese.JIEBA_AVAILABLE", False):
            with patch("kttc.helpers.chinese.SPACY_AVAILABLE", False):
                yield


@pytest.fixture
def mock_jieba_only():
    """Mock jieba as the only available tool."""
    with patch("kttc.helpers.chinese.HANLP_AVAILABLE", False):
        with patch("kttc.helpers.chinese.JIEBA_AVAILABLE", True):
            with patch("kttc.helpers.chinese.SPACY_AVAILABLE", False):
                # Mock jieba.cut
                with patch("kttc.helpers.chinese.jieba") as mock_jieba:
                    mock_jieba.cut = lambda text: list(text)
                    yield mock_jieba


@pytest.fixture
def mock_spacy_nlp():
    """Create a mock spaCy NLP pipeline."""
    return MockSpacyNLP()


@pytest.fixture
def mock_hanlp():
    """Create a mock HanLP pipeline."""
    return MockHanLP()


# ============================================================================
# Tests: Initialization
# ============================================================================


@pytest.mark.unit
class TestChineseHelperInitialization:
    """Test ChineseLanguageHelper initialization."""

    def test_init_without_any_tools(self, mock_all_unavailable: None) -> None:
        """Test initialization when no tools are available."""
        from kttc.helpers.chinese import ChineseLanguageHelper

        helper = ChineseLanguageHelper()

        assert helper.language_code == "zh"
        assert helper.is_available() is False
        assert helper._nlp is None
        assert helper._hanlp is None

    def test_init_with_jieba_only(self) -> None:
        """Test initialization with jieba only."""
        with patch("kttc.helpers.chinese.HANLP_AVAILABLE", False):
            with patch("kttc.helpers.chinese.JIEBA_AVAILABLE", True):
                with patch("kttc.helpers.chinese.SPACY_AVAILABLE", False):
                    from kttc.helpers.chinese import ChineseLanguageHelper

                    helper = ChineseLanguageHelper()

                    assert helper.is_available() is True
                    assert helper._initialized is True
                    assert helper._hanlp_available is False

    def test_init_with_spacy_success(self) -> None:
        """Test initialization with spaCy."""
        mock_nlp = MockSpacyNLP()

        with patch("kttc.helpers.chinese.HANLP_AVAILABLE", False):
            with patch("kttc.helpers.chinese.JIEBA_AVAILABLE", False):
                with patch("kttc.helpers.chinese.SPACY_AVAILABLE", True):
                    with patch("kttc.helpers.chinese.spacy") as mock_spacy:
                        mock_spacy.load.return_value = mock_nlp

                        from kttc.helpers.chinese import ChineseLanguageHelper

                        helper = ChineseLanguageHelper()

                        assert helper.is_available() is True
                        assert helper._nlp is not None

    def test_init_with_hanlp_success(self) -> None:
        """Test initialization with HanLP."""
        mock_hanlp_pipeline = MockHanLP()

        with patch("kttc.helpers.chinese.HANLP_AVAILABLE", True):
            with patch("kttc.helpers.chinese.JIEBA_AVAILABLE", True):
                with patch("kttc.helpers.chinese.SPACY_AVAILABLE", False):
                    with patch("kttc.helpers.chinese.hanlp") as mock_hanlp_module:
                        mock_hanlp_module.load.return_value = mock_hanlp_pipeline

                        from kttc.helpers.chinese import ChineseLanguageHelper

                        helper = ChineseLanguageHelper()

                        assert helper._hanlp_available is True

    def test_init_hanlp_failure(self) -> None:
        """Test initialization when HanLP fails."""
        with patch("kttc.helpers.chinese.HANLP_AVAILABLE", True):
            with patch("kttc.helpers.chinese.JIEBA_AVAILABLE", True):
                with patch("kttc.helpers.chinese.SPACY_AVAILABLE", False):
                    with patch("kttc.helpers.chinese.hanlp") as mock_hanlp_module:
                        mock_hanlp_module.load.side_effect = Exception("HanLP init failed")

                        from kttc.helpers.chinese import ChineseLanguageHelper

                        helper = ChineseLanguageHelper()

                        assert helper._hanlp_available is False
                        # Should still work with jieba
                        assert helper.is_available() is True


# ============================================================================
# Tests: Basic Methods
# ============================================================================


@pytest.mark.unit
class TestChineseHelperBasicMethods:
    """Test basic helper methods."""

    def test_language_code(self, mock_all_unavailable: None) -> None:
        """Test language_code property returns 'zh'."""
        from kttc.helpers.chinese import ChineseLanguageHelper

        helper = ChineseLanguageHelper()
        assert helper.language_code == "zh"

    def test_verify_word_exists_without_tools(self, mock_all_unavailable: None) -> None:
        """Test verify_word_exists falls back to simple search."""
        from kttc.helpers.chinese import ChineseLanguageHelper

        helper = ChineseLanguageHelper()

        assert helper.verify_word_exists("我", "我爱中文") is True
        assert helper.verify_word_exists("你", "我爱中文") is False

    def test_verify_word_exists_with_spacy(self) -> None:
        """Test verify_word_exists with spaCy tokenization."""
        tokens = [
            MockToken("我", 0),
            MockToken("爱", 1),
            MockToken("中文", 2),
        ]
        mock_doc = MockDoc(tokens)
        mock_nlp = MockSpacyNLP(mock_doc)

        with patch("kttc.helpers.chinese.HANLP_AVAILABLE", False):
            with patch("kttc.helpers.chinese.JIEBA_AVAILABLE", False):
                with patch("kttc.helpers.chinese.SPACY_AVAILABLE", True):
                    with patch("kttc.helpers.chinese.spacy") as mock_spacy:
                        mock_spacy.load.return_value = mock_nlp

                        from kttc.helpers.chinese import ChineseLanguageHelper

                        helper = ChineseLanguageHelper()

                        assert helper.verify_word_exists("我", "我爱中文") is True
                        assert helper.verify_word_exists("你", "我爱中文") is False

    def test_verify_error_position_valid(self, mock_all_unavailable: None) -> None:
        """Test verify_error_position with valid position."""
        from kttc.helpers.chinese import ChineseLanguageHelper

        helper = ChineseLanguageHelper()
        text = "我爱中文"

        error = ErrorAnnotation(
            category="fluency",
            subcategory="test",
            severity=ErrorSeverity.MINOR,
            location=(0, 2),
            description="test",
        )

        assert helper.verify_error_position(error, text) is True

    def test_verify_error_position_invalid_bounds(self, mock_all_unavailable: None) -> None:
        """Test verify_error_position with invalid bounds."""
        from kttc.helpers.chinese import ChineseLanguageHelper

        helper = ChineseLanguageHelper()
        text = "我爱"

        # Start < 0
        error1 = ErrorAnnotation(
            category="fluency",
            subcategory="test",
            severity=ErrorSeverity.MINOR,
            location=(-1, 2),
            description="test",
        )
        assert helper.verify_error_position(error1, text) is False

        # End > len(text)
        error2 = ErrorAnnotation(
            category="fluency",
            subcategory="test",
            severity=ErrorSeverity.MINOR,
            location=(0, 100),
            description="test",
        )
        assert helper.verify_error_position(error2, text) is False

        # Start >= End
        error3 = ErrorAnnotation(
            category="fluency",
            subcategory="test",
            severity=ErrorSeverity.MINOR,
            location=(2, 2),
            description="test",
        )
        assert helper.verify_error_position(error3, text) is False


# ============================================================================
# Tests: Tokenization
# ============================================================================


@pytest.mark.unit
class TestChineseHelperTokenization:
    """Test tokenization methods."""

    def test_tokenize_without_tools(self, mock_all_unavailable: None) -> None:
        """Test tokenize falls back to character-level tokenization."""
        from kttc.helpers.chinese import ChineseLanguageHelper

        helper = ChineseLanguageHelper()
        tokens = helper.tokenize("我爱")

        # Character-level tokenization
        assert len(tokens) == 2
        assert tokens[0][0] == "我"
        assert tokens[1][0] == "爱"

    def test_tokenize_with_spacy(self) -> None:
        """Test tokenize with spaCy."""
        tokens = [
            MockToken("我", 0),
            MockToken("爱", 1),
            MockToken("中文", 2),
        ]
        mock_doc = MockDoc(tokens)
        mock_nlp = MockSpacyNLP(mock_doc)

        with patch("kttc.helpers.chinese.HANLP_AVAILABLE", False):
            with patch("kttc.helpers.chinese.JIEBA_AVAILABLE", False):
                with patch("kttc.helpers.chinese.SPACY_AVAILABLE", True):
                    with patch("kttc.helpers.chinese.spacy") as mock_spacy:
                        mock_spacy.load.return_value = mock_nlp

                        from kttc.helpers.chinese import ChineseLanguageHelper

                        helper = ChineseLanguageHelper()
                        result = helper.tokenize("我爱中文")

                        assert len(result) == 3
                        assert result[0] == ("我", 0, 1)

    def test_tokenize_with_jieba(self) -> None:
        """Test tokenize with jieba."""
        with patch("kttc.helpers.chinese.HANLP_AVAILABLE", False):
            with patch("kttc.helpers.chinese.JIEBA_AVAILABLE", True):
                with patch("kttc.helpers.chinese.SPACY_AVAILABLE", False):
                    with patch("kttc.helpers.chinese.jieba") as mock_jieba:
                        mock_jieba.cut.return_value = ["我", "爱", "中文"]

                        from kttc.helpers.chinese import ChineseLanguageHelper

                        helper = ChineseLanguageHelper()
                        result = helper.tokenize("我爱中文")

                        assert len(result) >= 1


# ============================================================================
# Tests: Morphology Analysis
# ============================================================================


@pytest.mark.unit
class TestChineseHelperMorphology:
    """Test morphology analysis methods."""

    def test_analyze_morphology_without_spacy(self, mock_all_unavailable: None) -> None:
        """Test analyze_morphology returns empty without spaCy."""
        from kttc.helpers.chinese import ChineseLanguageHelper

        helper = ChineseLanguageHelper()
        result = helper.analyze_morphology("我爱中文")

        assert result == []

    def test_analyze_morphology_with_spacy(self) -> None:
        """Test analyze_morphology with spaCy."""
        tokens = [
            MockToken("我", 0, pos_="PRON"),
            MockToken("爱", 1, pos_="VERB"),
            MockToken("中文", 2, pos_="NOUN"),
        ]
        mock_doc = MockDoc(tokens)
        mock_nlp = MockSpacyNLP(mock_doc)

        with patch("kttc.helpers.chinese.HANLP_AVAILABLE", False):
            with patch("kttc.helpers.chinese.JIEBA_AVAILABLE", False):
                with patch("kttc.helpers.chinese.SPACY_AVAILABLE", True):
                    with patch("kttc.helpers.chinese.spacy") as mock_spacy:
                        mock_spacy.load.return_value = mock_nlp

                        from kttc.helpers.chinese import ChineseLanguageHelper

                        helper = ChineseLanguageHelper()
                        result = helper.analyze_morphology("我爱中文")

                        assert len(result) == 3
                        assert result[0].word == "我"
                        assert result[0].pos == "PRON"
                        assert result[1].word == "爱"
                        assert result[1].pos == "VERB"


# ============================================================================
# Tests: Grammar Checking
# ============================================================================


@pytest.mark.unit
class TestChineseHelperGrammar:
    """Test grammar checking methods."""

    def test_check_grammar_without_tools(self, mock_all_unavailable: None) -> None:
        """Test check_grammar returns empty without tools."""
        from kttc.helpers.chinese import ChineseLanguageHelper

        helper = ChineseLanguageHelper()
        result = helper.check_grammar("三个书")

        assert result == []

    def test_check_grammar_measure_word_error(self) -> None:
        """Test check_grammar detects measure word error."""
        # HanLP result: 三 (CD) + 个 (M) + 书 (NN)
        hanlp_result = {
            "tok": ["三", "个", "书"],
            "pos": ["CD", "M", "NN"],
            "ner": [],
        }
        mock_hanlp = MockHanLP(hanlp_result)

        with patch("kttc.helpers.chinese.HANLP_AVAILABLE", True):
            with patch("kttc.helpers.chinese.JIEBA_AVAILABLE", True):
                with patch("kttc.helpers.chinese.SPACY_AVAILABLE", False):
                    with patch("kttc.helpers.chinese.hanlp") as mock_hanlp_module:
                        mock_hanlp_module.load.return_value = mock_hanlp

                        from kttc.helpers.chinese import ChineseLanguageHelper

                        helper = ChineseLanguageHelper()
                        errors = helper.check_grammar("三个书")

                        # Should find measure word error (个 should be 本 for books)
                        assert len(errors) >= 1
                        assert any("measure word" in e.description.lower() for e in errors)

    def test_check_grammar_particle_error(self) -> None:
        """Test check_grammar detects particle error."""
        # HanLP result: 我 (PRON) + 了 (AS) - particle not following verb
        hanlp_result = {
            "tok": ["我", "了"],
            "pos": ["PN", "AS"],
            "ner": [],
        }
        mock_hanlp = MockHanLP(hanlp_result)

        with patch("kttc.helpers.chinese.HANLP_AVAILABLE", True):
            with patch("kttc.helpers.chinese.JIEBA_AVAILABLE", True):
                with patch("kttc.helpers.chinese.SPACY_AVAILABLE", False):
                    with patch("kttc.helpers.chinese.hanlp") as mock_hanlp_module:
                        mock_hanlp_module.load.return_value = mock_hanlp

                        from kttc.helpers.chinese import ChineseLanguageHelper

                        helper = ChineseLanguageHelper()
                        errors = helper.check_grammar("我了")

                        # Should find particle error
                        assert len(errors) >= 1
                        assert any("particle" in e.description.lower() for e in errors)

    def test_check_measure_words_exception(self) -> None:
        """Test _check_measure_words handles exceptions."""
        mock_hanlp = MagicMock()
        mock_hanlp.side_effect = Exception("HanLP error")

        with patch("kttc.helpers.chinese.HANLP_AVAILABLE", True):
            with patch("kttc.helpers.chinese.JIEBA_AVAILABLE", True):
                with patch("kttc.helpers.chinese.SPACY_AVAILABLE", False):
                    with patch("kttc.helpers.chinese.hanlp") as mock_hanlp_module:
                        mock_hanlp_module.load.return_value = mock_hanlp

                        from kttc.helpers.chinese import ChineseLanguageHelper

                        helper = ChineseLanguageHelper()
                        errors = helper._check_measure_words("三个书")

                        assert errors == []

    def test_check_particles_exception(self) -> None:
        """Test _check_particles handles exceptions."""
        mock_hanlp = MagicMock()
        mock_hanlp.side_effect = Exception("HanLP error")

        with patch("kttc.helpers.chinese.HANLP_AVAILABLE", True):
            with patch("kttc.helpers.chinese.JIEBA_AVAILABLE", True):
                with patch("kttc.helpers.chinese.SPACY_AVAILABLE", False):
                    with patch("kttc.helpers.chinese.hanlp") as mock_hanlp_module:
                        mock_hanlp_module.load.return_value = mock_hanlp

                        from kttc.helpers.chinese import ChineseLanguageHelper

                        helper = ChineseLanguageHelper()
                        errors = helper._check_particles("我吃了饭")

                        assert errors == []


# ============================================================================
# Tests: Measure Word Helpers
# ============================================================================


@pytest.mark.unit
class TestChineseHelperMeasureWords:
    """Test measure word helper methods."""

    def test_get_appropriate_measures_known_noun(self, mock_all_unavailable: None) -> None:
        """Test _get_appropriate_measures for known noun."""
        from kttc.helpers.chinese import ChineseLanguageHelper

        helper = ChineseLanguageHelper()

        # 书 (book) should use 本
        measures = helper._get_appropriate_measures("书")
        assert "本" in measures

        # 车 (car) should use 辆
        measures = helper._get_appropriate_measures("车")
        assert "辆" in measures

        # 猫 (cat) should use 只
        measures = helper._get_appropriate_measures("猫")
        assert "只" in measures

    def test_get_appropriate_measures_unknown_noun(self, mock_all_unavailable: None) -> None:
        """Test _get_appropriate_measures for unknown noun."""
        from kttc.helpers.chinese import ChineseLanguageHelper

        helper = ChineseLanguageHelper()

        # Unknown noun should return empty list
        measures = helper._get_appropriate_measures("未知名词")
        assert measures == []

    def test_find_measure_position_found(self, mock_all_unavailable: None) -> None:
        """Test _find_measure_position when pattern found."""
        from kttc.helpers.chinese import ChineseLanguageHelper

        position = ChineseLanguageHelper._find_measure_position("三个书", "三", "个", "书")

        assert position is not None
        assert position == (1, 2)  # Position of 个

    def test_find_measure_position_not_found(self, mock_all_unavailable: None) -> None:
        """Test _find_measure_position when pattern not found."""
        from kttc.helpers.chinese import ChineseLanguageHelper

        position = ChineseLanguageHelper._find_measure_position("我爱中文", "三", "个", "书")

        assert position is None

    def test_find_measure_position_without_number(self, mock_all_unavailable: None) -> None:
        """Test _find_measure_position when number not present."""
        from kttc.helpers.chinese import ChineseLanguageHelper

        position = ChineseLanguageHelper._find_measure_position("个书", "三", "个", "书")

        # Should find "个书" and return position of 个
        assert position is not None
        assert position == (0, 1)

    def test_create_measure_word_error(self, mock_all_unavailable: None) -> None:
        """Test _create_measure_word_error creates correct annotation."""
        from kttc.helpers.chinese import ChineseLanguageHelper

        error = ChineseLanguageHelper._create_measure_word_error(
            measure="个",
            noun="书",
            location=(1, 2),
            suggested=["本"],
        )

        assert error.category == "fluency"
        assert error.subcategory == "measure_word"
        assert error.severity == ErrorSeverity.MINOR
        assert error.location == (1, 2)
        assert "个" in error.description
        assert "书" in error.description
        assert error.suggestion == "本"


# ============================================================================
# Tests: Pattern Finding
# ============================================================================


@pytest.mark.unit
class TestChineseHelperPatterns:
    """Test pattern finding methods."""

    def test_find_measure_patterns(self, mock_all_unavailable: None) -> None:
        """Test _find_measure_patterns finds CD + M + NN patterns."""
        from kttc.helpers.chinese import ChineseLanguageHelper

        helper = ChineseLanguageHelper()

        tokens = ["三", "本", "书"]
        pos_tags = ["CD", "M", "NN"]

        patterns = helper._find_measure_patterns(tokens, pos_tags)

        assert len(patterns) == 1
        assert patterns[0]["number"] == "三"
        assert patterns[0]["measure"] == "本"
        assert patterns[0]["noun"] == "书"

    def test_find_aspect_particles(self, mock_all_unavailable: None) -> None:
        """Test _find_aspect_particles finds AS tags."""
        from kttc.helpers.chinese import ChineseLanguageHelper

        helper = ChineseLanguageHelper()

        tokens = ["吃", "了", "饭"]
        pos_tags = ["VV", "AS", "NN"]

        particles = helper._find_aspect_particles(tokens, pos_tags)

        assert len(particles) == 1
        assert particles[0]["particle"] == "了"
        assert particles[0]["verb"] == "吃"


# ============================================================================
# Tests: Enrichment Data
# ============================================================================


@pytest.mark.unit
class TestChineseHelperEnrichment:
    """Test enrichment data generation."""

    def test_get_enrichment_data_without_tools(self, mock_all_unavailable: None) -> None:
        """Test get_enrichment_data returns minimal data without tools."""
        from kttc.helpers.chinese import ChineseLanguageHelper

        helper = ChineseLanguageHelper()
        result = helper.get_enrichment_data("我爱中文")

        assert result == {"has_morphology": False}

    def test_get_enrichment_data_with_hanlp(self) -> None:
        """Test get_enrichment_data with HanLP."""
        hanlp_result = {
            "tok": ["我", "爱", "中文"],
            "pos": ["PN", "VV", "NN"],
            "ner": [("中文", "LANGUAGE", 2, 4)],
        }
        mock_hanlp = MockHanLP(hanlp_result)

        with patch("kttc.helpers.chinese.HANLP_AVAILABLE", True):
            with patch("kttc.helpers.chinese.JIEBA_AVAILABLE", True):
                with patch("kttc.helpers.chinese.SPACY_AVAILABLE", False):
                    with patch("kttc.helpers.chinese.hanlp") as mock_hanlp_module:
                        mock_hanlp_module.load.return_value = mock_hanlp

                        from kttc.helpers.chinese import ChineseLanguageHelper

                        helper = ChineseLanguageHelper()
                        result = helper.get_enrichment_data("我爱中文")

                        assert result["has_morphology"] is True
                        assert result["word_count"] == 3
                        assert "PN" in result["pos_distribution"]
                        assert result["has_hanlp"] is True

    def test_get_enrichment_data_with_spacy(self) -> None:
        """Test get_enrichment_data with spaCy."""
        tokens = [
            MockToken("我", 0, pos_="PRON"),
            MockToken("爱", 1, pos_="VERB"),
            MockToken("中文", 2, pos_="NOUN"),
        ]
        entities = [MockEntity("中文", "LANGUAGE", 2, 4)]
        mock_doc = MockDoc(tokens, entities)
        mock_nlp = MockSpacyNLP(mock_doc)

        with patch("kttc.helpers.chinese.HANLP_AVAILABLE", False):
            with patch("kttc.helpers.chinese.JIEBA_AVAILABLE", False):
                with patch("kttc.helpers.chinese.SPACY_AVAILABLE", True):
                    with patch("kttc.helpers.chinese.spacy") as mock_spacy:
                        mock_spacy.load.return_value = mock_nlp

                        from kttc.helpers.chinese import ChineseLanguageHelper

                        helper = ChineseLanguageHelper()
                        result = helper.get_enrichment_data("我爱中文")

                        assert result["has_morphology"] is True
                        assert result["word_count"] == 3
                        assert "PRON" in result["pos_distribution"]

    def test_get_enrichment_data_with_jieba(self) -> None:
        """Test get_enrichment_data with jieba only."""
        with patch("kttc.helpers.chinese.HANLP_AVAILABLE", False):
            with patch("kttc.helpers.chinese.JIEBA_AVAILABLE", True):
                with patch("kttc.helpers.chinese.SPACY_AVAILABLE", False):
                    with patch("kttc.helpers.chinese.jieba") as mock_jieba:
                        mock_jieba.cut.return_value = ["我", "爱", "中文"]

                        from kttc.helpers.chinese import ChineseLanguageHelper

                        helper = ChineseLanguageHelper()
                        result = helper.get_enrichment_data("我爱中文")

                        assert result["has_morphology"] is True
                        assert result["segmentation_method"] == "jieba"

    def test_get_hanlp_enrichment_exception(self) -> None:
        """Test _get_hanlp_enrichment handles exceptions."""
        mock_hanlp = MagicMock()
        mock_hanlp.side_effect = Exception("HanLP error")

        with patch("kttc.helpers.chinese.HANLP_AVAILABLE", True):
            with patch("kttc.helpers.chinese.JIEBA_AVAILABLE", True):
                with patch("kttc.helpers.chinese.SPACY_AVAILABLE", False):
                    with patch("kttc.helpers.chinese.hanlp") as mock_hanlp_module:
                        mock_hanlp_module.load.return_value = mock_hanlp

                        from kttc.helpers.chinese import ChineseLanguageHelper

                        helper = ChineseLanguageHelper()
                        result = helper._get_hanlp_enrichment("我爱中文")

                        assert result is None


# ============================================================================
# Tests: Entity Extraction
# ============================================================================


@pytest.mark.unit
class TestChineseHelperEntities:
    """Test entity extraction and preservation."""

    def test_extract_entities_without_spacy(self, mock_all_unavailable: None) -> None:
        """Test extract_entities returns empty without spaCy."""
        from kttc.helpers.chinese import ChineseLanguageHelper

        helper = ChineseLanguageHelper()
        result = helper.extract_entities("北京是中国的首都")

        assert result == []

    def test_extract_entities_with_spacy(self) -> None:
        """Test extract_entities returns entities."""
        entities = [
            MockEntity("北京", "GPE", 0, 2),
            MockEntity("中国", "GPE", 3, 5),
        ]
        mock_doc = MockDoc([MockToken("北京", 0)], entities)
        mock_nlp = MockSpacyNLP(mock_doc)

        with patch("kttc.helpers.chinese.HANLP_AVAILABLE", False):
            with patch("kttc.helpers.chinese.JIEBA_AVAILABLE", False):
                with patch("kttc.helpers.chinese.SPACY_AVAILABLE", True):
                    with patch("kttc.helpers.chinese.spacy") as mock_spacy:
                        mock_spacy.load.return_value = mock_nlp

                        from kttc.helpers.chinese import ChineseLanguageHelper

                        helper = ChineseLanguageHelper()
                        result = helper.extract_entities("北京是中国的首都")

                        assert len(result) == 2
                        assert result[0]["text"] == "北京"
                        assert result[0]["type"] == "GPE"

    def test_extract_entities_exception(self) -> None:
        """Test extract_entities handles exceptions."""
        mock_nlp = MagicMock()
        mock_nlp.side_effect = Exception("NER failed")

        with patch("kttc.helpers.chinese.HANLP_AVAILABLE", False):
            with patch("kttc.helpers.chinese.JIEBA_AVAILABLE", False):
                with patch("kttc.helpers.chinese.SPACY_AVAILABLE", True):
                    with patch("kttc.helpers.chinese.spacy") as mock_spacy:
                        mock_spacy.load.return_value = mock_nlp

                        from kttc.helpers.chinese import ChineseLanguageHelper

                        helper = ChineseLanguageHelper()
                        result = helper.extract_entities("北京")

                        assert result == []

    def test_check_entity_preservation_without_spacy(self, mock_all_unavailable: None) -> None:
        """Test check_entity_preservation returns empty without spaCy."""
        from kttc.helpers.chinese import ChineseLanguageHelper

        helper = ChineseLanguageHelper()
        result = helper.check_entity_preservation("John went to Beijing", "约翰去了北京")

        assert result == []

    def test_check_entity_preservation_missing_entities(self) -> None:
        """Test check_entity_preservation detects missing entities."""
        # Translation has no entities but source has names
        mock_doc = MockDoc([MockToken("他", 0)], [])  # No entities
        mock_nlp = MockSpacyNLP(mock_doc)

        with patch("kttc.helpers.chinese.HANLP_AVAILABLE", False):
            with patch("kttc.helpers.chinese.JIEBA_AVAILABLE", False):
                with patch("kttc.helpers.chinese.SPACY_AVAILABLE", True):
                    with patch("kttc.helpers.chinese.spacy") as mock_spacy:
                        mock_spacy.load.return_value = mock_nlp

                        from kttc.helpers.chinese import ChineseLanguageHelper

                        helper = ChineseLanguageHelper()
                        errors = helper.check_entity_preservation(
                            "John Smith went to Paris",  # Source with names
                            "他去了",  # Translation without entities
                        )

                        assert len(errors) >= 1
                        assert errors[0].subcategory == "entity_omission"

    def test_check_entity_preservation_exception(self) -> None:
        """Test check_entity_preservation handles exceptions."""
        mock_nlp = MagicMock()
        mock_nlp.side_effect = Exception("Entity check failed")

        with patch("kttc.helpers.chinese.HANLP_AVAILABLE", False):
            with patch("kttc.helpers.chinese.JIEBA_AVAILABLE", False):
                with patch("kttc.helpers.chinese.SPACY_AVAILABLE", True):
                    with patch("kttc.helpers.chinese.spacy") as mock_spacy:
                        mock_spacy.load.return_value = mock_nlp

                        from kttc.helpers.chinese import ChineseLanguageHelper

                        helper = ChineseLanguageHelper()
                        # Use lowercase source to avoid regex match
                        errors = helper.check_entity_preservation("hello world", "你好世界")

                        assert errors == []
