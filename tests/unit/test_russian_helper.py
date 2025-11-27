"""Unit tests for Russian language helper.

Tests RussianLanguageHelper with mocked MAWO core and LanguageTool dependencies.
"""

from unittest.mock import MagicMock, patch

import pytest

from kttc.core.models import ErrorAnnotation, ErrorSeverity
from kttc.helpers.base import MorphologyInfo


# Mock classes for MAWO core
class MockToken:
    """Mock MAWO Token object."""

    def __init__(
        self,
        text: str,
        start: int,
        end: int,
        pos: str = "NOUN",
        gender: str | None = None,
        case: str | None = None,
        number: str | None = None,
        aspect: str | None = None,
    ):
        self.text = text
        self.start = start
        self.end = end
        self.pos = pos
        self.gender = gender
        self.case = case
        self.number = number
        self.aspect = aspect


class MockEntity:
    """Mock MAWO Entity object."""

    def __init__(self, text: str, label: str, start: int, end: int):
        self.text = text
        self.label = label
        self.start = start
        self.end = end


class MockVerb:
    """Mock MAWO Verb object."""

    def __init__(self, text: str, aspect: str, start: int, end: int):
        self.text = text
        self.aspect = aspect
        self.start = start
        self.end = end


class MockAdjNounPair:
    """Mock MAWO AdjNounPair object."""

    def __init__(self, adj_token: MockToken, noun_token: MockToken, agreement: str):
        self.adjective = adj_token
        self.noun = noun_token
        self.agreement = agreement


class MockEntityMatch:
    """Mock MAWO EntityMatch object."""

    def __init__(self, source: MockEntity, target: MockEntity):
        self.source = source
        self.target = target


class MockDoc:
    """Mock MAWO Document object."""

    def __init__(
        self,
        tokens: list[MockToken] | None = None,
        entities: list[MockEntity] | None = None,
        verbs: list[MockVerb] | None = None,
        adj_noun_pairs: list[MockAdjNounPair] | None = None,
    ):
        self.tokens = tokens or []
        self.entities = entities or []
        self.verbs = verbs or []
        self.adjective_noun_pairs = adj_noun_pairs or []


class MockRussianNLP:
    """Mock MAWO Russian NLP pipeline."""

    def __init__(self, doc: MockDoc | None = None):
        self._doc = doc or MockDoc()

    def __call__(self, text: str) -> MockDoc:
        return self._doc

    def match_entities(self, source_doc: MockDoc, target_doc: MockDoc) -> list[MockEntityMatch]:
        """Mock entity matching."""
        return []


class MockLanguageToolMatch:
    """Mock LanguageTool Match object."""

    def __init__(
        self,
        rule_id: str,
        message: str,
        offset: int,
        error_length: int,
        replacements: list[str] | None = None,
    ):
        self.ruleId = rule_id
        self.message = message
        self.offset = offset
        self.errorLength = error_length
        self.replacements = replacements or []


class MockLanguageTool:
    """Mock LanguageTool instance."""

    def __init__(self, matches: list[MockLanguageToolMatch] | None = None):
        self._matches = matches or []

    def check(self, text: str) -> list[MockLanguageToolMatch]:
        return self._matches


@pytest.mark.unit
class TestRussianHelperInitialization:
    """Tests for RussianLanguageHelper initialization."""

    def test_init_without_any_tools(self) -> None:
        """Test initialization without MAWO or LanguageTool."""
        with (
            patch("kttc.helpers.russian.MAWO_AVAILABLE", False),
            patch("kttc.helpers.russian.LANGUAGETOOL_AVAILABLE", False),
        ):
            from kttc.helpers.russian import RussianLanguageHelper

            helper = RussianLanguageHelper()
            assert helper.language_code == "ru"
            assert not helper.is_available()

    def test_init_with_mawo_success(self) -> None:
        """Test initialization with MAWO core."""
        mock_nlp = MockRussianNLP()

        with (
            patch("kttc.helpers.russian.MAWO_AVAILABLE", True),
            patch("kttc.helpers.russian.LANGUAGETOOL_AVAILABLE", False),
            patch("kttc.helpers.russian.Russian", return_value=mock_nlp),
        ):
            from kttc.helpers.russian import RussianLanguageHelper

            helper = RussianLanguageHelper()
            assert helper.is_available()

    def test_init_with_mawo_failure(self) -> None:
        """Test initialization when MAWO fails."""
        with (
            patch("kttc.helpers.russian.MAWO_AVAILABLE", True),
            patch("kttc.helpers.russian.LANGUAGETOOL_AVAILABLE", False),
            patch("kttc.helpers.russian.Russian", side_effect=Exception("MAWO init failed")),
        ):
            from kttc.helpers.russian import RussianLanguageHelper

            helper = RussianLanguageHelper()
            assert not helper.is_available()

    def test_init_with_languagetool_success(self) -> None:
        """Test initialization with LanguageTool."""
        mock_nlp = MockRussianNLP()
        mock_lt = MockLanguageTool()

        with (
            patch("kttc.helpers.russian.MAWO_AVAILABLE", True),
            patch("kttc.helpers.russian.LANGUAGETOOL_AVAILABLE", True),
            patch("kttc.helpers.russian.Russian", return_value=mock_nlp),
            patch("kttc.helpers.russian.language_tool_python") as mock_lt_module,
        ):
            mock_lt_module.LanguageTool.return_value = mock_lt

            from kttc.helpers.russian import RussianLanguageHelper

            helper = RussianLanguageHelper()
            assert helper._lt_available
            mock_lt_module.LanguageTool.assert_called_once_with("ru")

    def test_init_with_languagetool_failure(self) -> None:
        """Test initialization when LanguageTool fails."""
        mock_nlp = MockRussianNLP()

        with (
            patch("kttc.helpers.russian.MAWO_AVAILABLE", True),
            patch("kttc.helpers.russian.LANGUAGETOOL_AVAILABLE", True),
            patch("kttc.helpers.russian.Russian", return_value=mock_nlp),
            patch("kttc.helpers.russian.language_tool_python") as mock_lt_module,
        ):
            mock_lt_module.LanguageTool.side_effect = Exception("LT init failed")

            from kttc.helpers.russian import RussianLanguageHelper

            helper = RussianLanguageHelper()
            assert not helper._lt_available


@pytest.mark.unit
class TestRussianHelperBasicMethods:
    """Tests for basic helper methods."""

    def test_language_code(self) -> None:
        """Test language code property."""
        with (
            patch("kttc.helpers.russian.MAWO_AVAILABLE", False),
            patch("kttc.helpers.russian.LANGUAGETOOL_AVAILABLE", False),
        ):
            from kttc.helpers.russian import RussianLanguageHelper

            helper = RussianLanguageHelper()
            assert helper.language_code == "ru"

    def test_verify_word_exists_without_mawo(self) -> None:
        """Test verify_word_exists without MAWO (fallback)."""
        with (
            patch("kttc.helpers.russian.MAWO_AVAILABLE", False),
            patch("kttc.helpers.russian.LANGUAGETOOL_AVAILABLE", False),
        ):
            from kttc.helpers.russian import RussianLanguageHelper

            helper = RussianLanguageHelper()
            assert helper.verify_word_exists("привет", "Привет мир")
            assert helper.verify_word_exists("МИР", "Привет мир")
            assert not helper.verify_word_exists("foo", "Привет мир")

    def test_verify_word_exists_with_mawo(self) -> None:
        """Test verify_word_exists with MAWO tokenization."""
        mock_tokens = [
            MockToken("Привет", 0, 6),
            MockToken("мир", 7, 10),
        ]
        mock_doc = MockDoc(tokens=mock_tokens)
        mock_nlp = MockRussianNLP(mock_doc)

        with (
            patch("kttc.helpers.russian.MAWO_AVAILABLE", True),
            patch("kttc.helpers.russian.LANGUAGETOOL_AVAILABLE", False),
            patch("kttc.helpers.russian.Russian", return_value=mock_nlp),
        ):
            from kttc.helpers.russian import RussianLanguageHelper

            helper = RussianLanguageHelper()
            assert helper.verify_word_exists("привет", "Привет мир")
            assert not helper.verify_word_exists("foo", "Привет мир")

    def test_verify_error_position_valid(self) -> None:
        """Test verify_error_position with valid position."""
        with (
            patch("kttc.helpers.russian.MAWO_AVAILABLE", False),
            patch("kttc.helpers.russian.LANGUAGETOOL_AVAILABLE", False),
        ):
            from kttc.helpers.russian import RussianLanguageHelper

            helper = RussianLanguageHelper()
            error = ErrorAnnotation(
                category="fluency",
                subcategory="grammar",
                severity=ErrorSeverity.MINOR,
                location=(0, 6),
                description="Test error",
            )
            assert helper.verify_error_position(error, "Привет мир")

    def test_verify_error_position_invalid_bounds(self) -> None:
        """Test verify_error_position with invalid bounds."""
        with (
            patch("kttc.helpers.russian.MAWO_AVAILABLE", False),
            patch("kttc.helpers.russian.LANGUAGETOOL_AVAILABLE", False),
        ):
            from kttc.helpers.russian import RussianLanguageHelper

            helper = RussianLanguageHelper()
            text = "Короткий"

            # Start < 0
            error1 = ErrorAnnotation(
                category="fluency",
                subcategory="test",
                severity=ErrorSeverity.MINOR,
                location=(-1, 5),
                description="Test",
            )
            assert not helper.verify_error_position(error1, text)

            # End > len(text)
            error2 = ErrorAnnotation(
                category="fluency",
                subcategory="test",
                severity=ErrorSeverity.MINOR,
                location=(0, 100),
                description="Test",
            )
            assert not helper.verify_error_position(error2, text)

    def test_verify_error_position_with_quoted_word(self) -> None:
        """Test verify_error_position checks quoted words in description."""
        with (
            patch("kttc.helpers.russian.MAWO_AVAILABLE", False),
            patch("kttc.helpers.russian.LANGUAGETOOL_AVAILABLE", False),
        ):
            from kttc.helpers.russian import RussianLanguageHelper

            helper = RussianLanguageHelper()
            text = "Привет мир"

            # Word is in substring
            error1 = ErrorAnnotation(
                category="fluency",
                subcategory="test",
                severity=ErrorSeverity.MINOR,
                location=(0, 6),
                description="Error with 'Привет'",
            )
            assert helper.verify_error_position(error1, text)

            # Word is NOT in substring
            error2 = ErrorAnnotation(
                category="fluency",
                subcategory="test",
                severity=ErrorSeverity.MINOR,
                location=(0, 6),
                description="Error with 'мир'",
            )
            assert not helper.verify_error_position(error2, text)


@pytest.mark.unit
class TestRussianHelperTokenization:
    """Tests for tokenization methods."""

    def test_tokenize_without_mawo(self) -> None:
        """Test tokenization without MAWO (fallback)."""
        with (
            patch("kttc.helpers.russian.MAWO_AVAILABLE", False),
            patch("kttc.helpers.russian.LANGUAGETOOL_AVAILABLE", False),
        ):
            from kttc.helpers.russian import RussianLanguageHelper

            helper = RussianLanguageHelper()
            tokens = helper.tokenize("Привет мир тест")

            assert len(tokens) == 3
            assert tokens[0][0] == "Привет"
            assert tokens[1][0] == "мир"

    def test_tokenize_with_mawo(self) -> None:
        """Test tokenization with MAWO core."""
        mock_tokens = [
            MockToken("Привет", 0, 6),
            MockToken(",", 6, 7),
            MockToken("мир", 8, 11),
        ]
        mock_doc = MockDoc(tokens=mock_tokens)
        mock_nlp = MockRussianNLP(mock_doc)

        with (
            patch("kttc.helpers.russian.MAWO_AVAILABLE", True),
            patch("kttc.helpers.russian.LANGUAGETOOL_AVAILABLE", False),
            patch("kttc.helpers.russian.Russian", return_value=mock_nlp),
        ):
            from kttc.helpers.russian import RussianLanguageHelper

            helper = RussianLanguageHelper()
            tokens = helper.tokenize("Привет, мир")

            assert len(tokens) == 3
            assert tokens[0] == ("Привет", 0, 6)


@pytest.mark.unit
class TestRussianHelperMorphology:
    """Tests for morphology analysis."""

    def test_analyze_morphology_without_mawo(self) -> None:
        """Test morphology analysis without MAWO."""
        with (
            patch("kttc.helpers.russian.MAWO_AVAILABLE", False),
            patch("kttc.helpers.russian.LANGUAGETOOL_AVAILABLE", False),
        ):
            from kttc.helpers.russian import RussianLanguageHelper

            helper = RussianLanguageHelper()
            result = helper.analyze_morphology("Привет мир")
            assert result == []

    def test_analyze_morphology_with_mawo(self) -> None:
        """Test morphology analysis with MAWO core."""
        mock_tokens = [
            MockToken("красивый", 0, 8, pos="ADJF", gender="masc", case="nomn", number="sing"),
            MockToken("дом", 9, 12, pos="NOUN", gender="masc", case="nomn", number="sing"),
        ]
        mock_doc = MockDoc(tokens=mock_tokens)
        mock_nlp = MockRussianNLP(mock_doc)

        with (
            patch("kttc.helpers.russian.MAWO_AVAILABLE", True),
            patch("kttc.helpers.russian.LANGUAGETOOL_AVAILABLE", False),
            patch("kttc.helpers.russian.Russian", return_value=mock_nlp),
        ):
            from kttc.helpers.russian import RussianLanguageHelper

            helper = RussianLanguageHelper()
            result = helper.analyze_morphology("красивый дом")

            assert len(result) == 2
            assert isinstance(result[0], MorphologyInfo)
            assert result[0].word == "красивый"
            assert result[0].pos == "ADJF"
            assert result[0].gender == "masc"


@pytest.mark.unit
class TestRussianHelperGrammar:
    """Tests for grammar checking."""

    def test_check_grammar_without_tools(self) -> None:
        """Test grammar check without any tools."""
        with (
            patch("kttc.helpers.russian.MAWO_AVAILABLE", False),
            patch("kttc.helpers.russian.LANGUAGETOOL_AVAILABLE", False),
        ):
            from kttc.helpers.russian import RussianLanguageHelper

            helper = RussianLanguageHelper()
            errors = helper.check_grammar("Привет мир")
            assert errors == []

    def test_check_grammar_with_languagetool(self) -> None:
        """Test grammar check with LanguageTool."""
        mock_match = MockLanguageToolMatch(
            rule_id="GRAMMAR_ERROR",
            message="Grammar error found",
            offset=0,
            error_length=5,
            replacements=["исправление"],
        )
        mock_lt = MockLanguageTool([mock_match])

        with (
            patch("kttc.helpers.russian.MAWO_AVAILABLE", False),
            patch("kttc.helpers.russian.LANGUAGETOOL_AVAILABLE", True),
            patch("kttc.helpers.russian.language_tool_python") as mock_lt_module,
        ):
            mock_lt_module.LanguageTool.return_value = mock_lt

            from kttc.helpers.russian import RussianLanguageHelper

            helper = RussianLanguageHelper()
            errors = helper.check_grammar("тест текст")

            assert len(errors) == 1
            assert errors[0].category == "fluency"
            assert "GRAMMAR" in errors[0].subcategory

    def test_check_grammar_filters_style_errors(self) -> None:
        """Test that style errors are filtered."""
        mock_matches = [
            MockLanguageToolMatch(
                rule_id="GRAMMAR_ERROR",
                message="Grammar error",
                offset=0,
                error_length=5,
            ),
            MockLanguageToolMatch(
                rule_id="STYLE_SUGGESTION",
                message="Style suggestion",
                offset=10,
                error_length=5,
            ),
        ]
        mock_lt = MockLanguageTool(mock_matches)

        with (
            patch("kttc.helpers.russian.MAWO_AVAILABLE", False),
            patch("kttc.helpers.russian.LANGUAGETOOL_AVAILABLE", True),
            patch("kttc.helpers.russian.language_tool_python") as mock_lt_module,
        ):
            mock_lt_module.LanguageTool.return_value = mock_lt

            from kttc.helpers.russian import RussianLanguageHelper

            helper = RussianLanguageHelper()
            errors = helper.check_grammar("Test text here")

            assert len(errors) == 1
            assert "GRAMMAR" in errors[0].subcategory

    def test_check_grammar_skips_it_terms(self) -> None:
        """Test that IT terms are not flagged."""
        # "демо" is 4 chars, so error_length should be 4
        mock_match = MockLanguageToolMatch(
            rule_id="SPELLING_ERROR",
            message="Spelling error",
            offset=0,
            error_length=4,  # Length of "демо"
        )
        mock_lt = MockLanguageTool([mock_match])

        with (
            patch("kttc.helpers.russian.MAWO_AVAILABLE", False),
            patch("kttc.helpers.russian.LANGUAGETOOL_AVAILABLE", True),
            patch("kttc.helpers.russian.language_tool_python") as mock_lt_module,
        ):
            mock_lt_module.LanguageTool.return_value = mock_lt

            from kttc.helpers.russian import RussianLanguageHelper

            helper = RussianLanguageHelper()
            # "демо" is in IT_TERMS_WHITELIST
            errors = helper.check_grammar("демо режим")

            assert len(errors) == 0

    def test_check_grammar_adj_noun_agreement(self) -> None:
        """Test adjective-noun agreement checking."""
        # Create tokens with gender mismatch
        adj_token = MockToken("быстрый", 0, 7, pos="ADJF", gender="masc")
        noun_token = MockToken("лиса", 8, 12, pos="NOUN", gender="femn")
        mock_doc = MockDoc(tokens=[adj_token, noun_token])
        mock_nlp = MockRussianNLP(mock_doc)

        with (
            patch("kttc.helpers.russian.MAWO_AVAILABLE", True),
            patch("kttc.helpers.russian.LANGUAGETOOL_AVAILABLE", False),
            patch("kttc.helpers.russian.Russian", return_value=mock_nlp),
        ):
            from kttc.helpers.russian import RussianLanguageHelper

            helper = RussianLanguageHelper()
            errors = helper.check_grammar("быстрый лиса")

            assert len(errors) >= 1
            agreement_error = next(
                (e for e in errors if "agreement" in e.subcategory.lower()), None
            )
            assert agreement_error is not None

    def test_check_grammar_exception_handling(self) -> None:
        """Test grammar check handles exceptions."""
        mock_lt = MagicMock()
        mock_lt.check.side_effect = Exception("LT check failed")

        with (
            patch("kttc.helpers.russian.MAWO_AVAILABLE", False),
            patch("kttc.helpers.russian.LANGUAGETOOL_AVAILABLE", True),
            patch("kttc.helpers.russian.language_tool_python") as mock_lt_module,
        ):
            mock_lt_module.LanguageTool.return_value = mock_lt

            from kttc.helpers.russian import RussianLanguageHelper

            helper = RussianLanguageHelper()
            errors = helper.check_grammar("Test text")
            assert errors == []


@pytest.mark.unit
class TestRussianHelperSeverityMapping:
    """Tests for severity mapping."""

    def test_severity_spelling_critical(self) -> None:
        """Test spelling errors are CRITICAL."""
        # Use "слово" (word) - 5 chars, not in IT_TERMS_WHITELIST
        mock_match = MockLanguageToolMatch(
            rule_id="SPELLING_TYPO",
            message="Spelling error",
            offset=0,
            error_length=5,
        )
        mock_lt = MockLanguageTool([mock_match])

        with (
            patch("kttc.helpers.russian.MAWO_AVAILABLE", False),
            patch("kttc.helpers.russian.LANGUAGETOOL_AVAILABLE", True),
            patch("kttc.helpers.russian.language_tool_python") as mock_lt_module,
        ):
            mock_lt_module.LanguageTool.return_value = mock_lt

            from kttc.helpers.russian import RussianLanguageHelper

            helper = RussianLanguageHelper()
            errors = helper.check_grammar("слово")

            assert len(errors) == 1
            assert errors[0].severity == ErrorSeverity.CRITICAL

    def test_severity_grammar_major(self) -> None:
        """Test grammar errors are MAJOR."""
        # Use "слово" (word) - 5 chars, not in IT_TERMS_WHITELIST
        mock_match = MockLanguageToolMatch(
            rule_id="GRAMMAR_CASE_ERROR",
            message="Case error",
            offset=0,
            error_length=5,
        )
        mock_lt = MockLanguageTool([mock_match])

        with (
            patch("kttc.helpers.russian.MAWO_AVAILABLE", False),
            patch("kttc.helpers.russian.LANGUAGETOOL_AVAILABLE", True),
            patch("kttc.helpers.russian.language_tool_python") as mock_lt_module,
        ):
            mock_lt_module.LanguageTool.return_value = mock_lt

            from kttc.helpers.russian import RussianLanguageHelper

            helper = RussianLanguageHelper()
            errors = helper.check_grammar("слово")

            assert len(errors) == 1
            assert errors[0].severity == ErrorSeverity.MAJOR

    def test_severity_other_minor(self) -> None:
        """Test other errors are MINOR."""
        # Use "слово" (word) - 5 chars, not in IT_TERMS_WHITELIST
        mock_match = MockLanguageToolMatch(
            rule_id="SOME_OTHER_RULE",
            message="Some issue",
            offset=0,
            error_length=5,
        )
        mock_lt = MockLanguageTool([mock_match])

        with (
            patch("kttc.helpers.russian.MAWO_AVAILABLE", False),
            patch("kttc.helpers.russian.LANGUAGETOOL_AVAILABLE", True),
            patch("kttc.helpers.russian.language_tool_python") as mock_lt_module,
        ):
            mock_lt_module.LanguageTool.return_value = mock_lt

            from kttc.helpers.russian import RussianLanguageHelper

            helper = RussianLanguageHelper()
            errors = helper.check_grammar("слово")

            assert len(errors) == 1
            assert errors[0].severity == ErrorSeverity.MINOR


@pytest.mark.unit
class TestRussianHelperITTerms:
    """Tests for IT terminology whitelist."""

    def test_is_it_term_direct_match(self) -> None:
        """Test direct IT term match."""
        with (
            patch("kttc.helpers.russian.MAWO_AVAILABLE", False),
            patch("kttc.helpers.russian.LANGUAGETOOL_AVAILABLE", False),
        ):
            from kttc.helpers.russian import RussianLanguageHelper

            helper = RussianLanguageHelper()
            assert helper._is_it_term("демо")
            assert helper._is_it_term("деплой")
            assert helper._is_it_term("микросервис")
            assert not helper._is_it_term("обычное_слово")

    def test_is_it_term_with_punctuation(self) -> None:
        """Test IT term match with punctuation."""
        with (
            patch("kttc.helpers.russian.MAWO_AVAILABLE", False),
            patch("kttc.helpers.russian.LANGUAGETOOL_AVAILABLE", False),
        ):
            from kttc.helpers.russian import RussianLanguageHelper

            helper = RussianLanguageHelper()
            assert helper._is_it_term("демо.")
            assert helper._is_it_term("«деплой»")
            assert helper._is_it_term("(микросервис)")

    def test_is_it_term_prefix_match(self) -> None:
        """Test IT term prefix match."""
        with (
            patch("kttc.helpers.russian.MAWO_AVAILABLE", False),
            patch("kttc.helpers.russian.LANGUAGETOOL_AVAILABLE", False),
        ):
            from kttc.helpers.russian import RussianLanguageHelper

            helper = RussianLanguageHelper()
            assert helper._is_it_term("демо-версия")
            assert helper._is_it_term("веб-приложение")
            assert helper._is_it_term("код-ревью")


@pytest.mark.unit
class TestRussianHelperEnrichment:
    """Tests for enrichment data methods."""

    def test_get_enrichment_data_without_mawo(self) -> None:
        """Test enrichment without MAWO."""
        with (
            patch("kttc.helpers.russian.MAWO_AVAILABLE", False),
            patch("kttc.helpers.russian.LANGUAGETOOL_AVAILABLE", False),
        ):
            from kttc.helpers.russian import RussianLanguageHelper

            helper = RussianLanguageHelper()
            result = helper.get_enrichment_data("тест")
            assert result == {"has_morphology": False}

    def test_get_enrichment_data_with_mawo(self) -> None:
        """Test enrichment with MAWO core."""
        mock_tokens = [MockToken("тест", 0, 4, pos="NOUN")]
        mock_verbs = [MockVerb("пришёл", "perf", 0, 6)]
        mock_adj = MockToken(
            "красивый", 0, 8, pos="ADJF", gender="masc", case="nomn", number="sing"
        )
        mock_noun = MockToken("дом", 9, 12, pos="NOUN", gender="masc", case="nomn", number="sing")
        mock_pairs = [MockAdjNounPair(mock_adj, mock_noun, "correct")]
        mock_doc = MockDoc(tokens=mock_tokens, verbs=mock_verbs, adj_noun_pairs=mock_pairs)
        mock_nlp = MockRussianNLP(mock_doc)

        with (
            patch("kttc.helpers.russian.MAWO_AVAILABLE", True),
            patch("kttc.helpers.russian.LANGUAGETOOL_AVAILABLE", False),
            patch("kttc.helpers.russian.Russian", return_value=mock_nlp),
        ):
            from kttc.helpers.russian import RussianLanguageHelper

            helper = RussianLanguageHelper()
            result = helper.get_enrichment_data("красивый дом")

            assert result["has_morphology"]
            assert "word_count" in result
            assert "verb_aspects" in result
            assert "adjective_noun_pairs" in result
            assert "pos_distribution" in result


@pytest.mark.unit
class TestRussianHelperEntities:
    """Tests for entity extraction and preservation."""

    def test_extract_entities_without_mawo(self) -> None:
        """Test entity extraction without MAWO."""
        with (
            patch("kttc.helpers.russian.MAWO_AVAILABLE", False),
            patch("kttc.helpers.russian.LANGUAGETOOL_AVAILABLE", False),
        ):
            from kttc.helpers.russian import RussianLanguageHelper

            helper = RussianLanguageHelper()
            entities = helper.extract_entities("Иван пошёл в Москву")
            assert entities == []

    def test_extract_entities_with_mawo(self) -> None:
        """Test entity extraction with MAWO NER."""
        mock_entities = [
            MockEntity("Иван", "PER", 0, 4),
            MockEntity("Москва", "LOC", 14, 20),
        ]
        mock_doc = MockDoc(entities=mock_entities)
        mock_nlp = MockRussianNLP(mock_doc)

        with (
            patch("kttc.helpers.russian.MAWO_AVAILABLE", True),
            patch("kttc.helpers.russian.LANGUAGETOOL_AVAILABLE", False),
            patch("kttc.helpers.russian.Russian", return_value=mock_nlp),
        ):
            from kttc.helpers.russian import RussianLanguageHelper

            helper = RussianLanguageHelper()
            entities = helper.extract_entities("Иван пошёл в Москву")

            assert len(entities) == 2
            assert entities[0]["text"] == "Иван"
            assert entities[0]["type"] == "PER"

    def test_extract_entities_exception(self) -> None:
        """Test entity extraction handles exceptions."""
        mock_nlp = MagicMock()
        mock_nlp.side_effect = Exception("NER failed")

        with (
            patch("kttc.helpers.russian.MAWO_AVAILABLE", True),
            patch("kttc.helpers.russian.LANGUAGETOOL_AVAILABLE", False),
            patch("kttc.helpers.russian.Russian", return_value=mock_nlp),
        ):
            from kttc.helpers.russian import RussianLanguageHelper

            helper = RussianLanguageHelper()
            entities = helper.extract_entities("тест")
            assert entities == []

    def test_check_entity_preservation_without_mawo(self) -> None:
        """Test entity preservation without MAWO."""
        with (
            patch("kttc.helpers.russian.MAWO_AVAILABLE", False),
            patch("kttc.helpers.russian.LANGUAGETOOL_AVAILABLE", False),
        ):
            from kttc.helpers.russian import RussianLanguageHelper

            helper = RussianLanguageHelper()
            errors = helper.check_entity_preservation("John Smith", "Иван Смит")
            assert errors == []

    def test_check_entity_preservation_missing_entities(self) -> None:
        """Test entity preservation detects missing entities."""
        source_entity = MockEntity("John", "PER", 0, 4)
        source_doc = MockDoc(entities=[source_entity])
        target_doc = MockDoc(entities=[])  # No entities in translation

        mock_nlp = MagicMock()
        mock_nlp.side_effect = [source_doc, target_doc]
        mock_nlp.match_entities.return_value = []  # No matches

        with (
            patch("kttc.helpers.russian.MAWO_AVAILABLE", True),
            patch("kttc.helpers.russian.LANGUAGETOOL_AVAILABLE", False),
            patch("kttc.helpers.russian.Russian", return_value=mock_nlp),
        ):
            from kttc.helpers.russian import RussianLanguageHelper

            helper = RussianLanguageHelper()
            errors = helper.check_entity_preservation(
                "John Smith visited New York", "он пошёл туда"
            )

            assert len(errors) >= 1
            assert errors[0].subcategory == "entity_omission"

    def test_check_entity_preservation_exception_fallback(self) -> None:
        """Test entity preservation falls back to basic check on exception."""
        mock_nlp = MagicMock()
        mock_nlp.side_effect = Exception("NER failed")

        with (
            patch("kttc.helpers.russian.MAWO_AVAILABLE", True),
            patch("kttc.helpers.russian.LANGUAGETOOL_AVAILABLE", False),
            patch("kttc.helpers.russian.Russian", return_value=mock_nlp),
        ):
            from kttc.helpers.russian import RussianLanguageHelper

            helper = RussianLanguageHelper()
            # Use lowercase to avoid regex match on capitalized words
            errors = helper.check_entity_preservation("hello world", "привет мир")
            assert errors == []


@pytest.mark.unit
class TestRussianHelperDeduplication:
    """Tests for error deduplication."""

    def test_deduplicate_errors_empty(self) -> None:
        """Test deduplication with empty list."""
        with (
            patch("kttc.helpers.russian.MAWO_AVAILABLE", False),
            patch("kttc.helpers.russian.LANGUAGETOOL_AVAILABLE", False),
        ):
            from kttc.helpers.russian import RussianLanguageHelper

            helper = RussianLanguageHelper()
            result = helper._deduplicate_errors([])
            assert result == []

    def test_deduplicate_errors_keeps_higher_severity(self) -> None:
        """Test deduplication keeps higher severity error."""
        with (
            patch("kttc.helpers.russian.MAWO_AVAILABLE", False),
            patch("kttc.helpers.russian.LANGUAGETOOL_AVAILABLE", False),
        ):
            from kttc.helpers.russian import RussianLanguageHelper

            helper = RussianLanguageHelper()

            error1 = ErrorAnnotation(
                category="fluency",
                subcategory="test1",
                severity=ErrorSeverity.MINOR,
                location=(0, 5),
                description="Minor error",
            )
            error2 = ErrorAnnotation(
                category="fluency",
                subcategory="test2",
                severity=ErrorSeverity.CRITICAL,
                location=(0, 5),
                description="Critical error",
            )

            result = helper._deduplicate_errors([error1, error2])

            assert len(result) == 1
            assert result[0].severity == ErrorSeverity.CRITICAL


@pytest.mark.unit
class TestRussianHelperCSVContext:
    """Tests for CSV/code context detection."""

    def test_is_csv_context_detected(self) -> None:
        """Test CSV context is detected."""
        with (
            patch("kttc.helpers.russian.MAWO_AVAILABLE", False),
            patch("kttc.helpers.russian.LANGUAGETOOL_AVAILABLE", False),
        ):
            from kttc.helpers.russian import RussianLanguageHelper

            helper = RussianLanguageHelper()

            mock_match = MagicMock()
            mock_match.ruleId = "COMMA_PARENTHESIS_WHITESPACE"
            mock_match.offset = 10
            mock_match.errorLength = 5

            # CSV-like text
            csv_text = "source,target,lang,translation"
            assert helper._is_csv_or_code_context(csv_text, mock_match)

    def test_is_code_context_detected(self) -> None:
        """Test code context is detected."""
        with (
            patch("kttc.helpers.russian.MAWO_AVAILABLE", False),
            patch("kttc.helpers.russian.LANGUAGETOOL_AVAILABLE", False),
        ):
            from kttc.helpers.russian import RussianLanguageHelper

            helper = RussianLanguageHelper()

            mock_match = MagicMock()
            mock_match.ruleId = "WHITESPACE_RULE"
            mock_match.offset = 10
            mock_match.errorLength = 5

            # Code-like text
            code_text = "```python\nimport os\n```"
            assert helper._is_csv_or_code_context(code_text, mock_match)

    def test_normal_context_not_detected(self) -> None:
        """Test normal text is not flagged as CSV/code."""
        with (
            patch("kttc.helpers.russian.MAWO_AVAILABLE", False),
            patch("kttc.helpers.russian.LANGUAGETOOL_AVAILABLE", False),
        ):
            from kttc.helpers.russian import RussianLanguageHelper

            helper = RussianLanguageHelper()

            mock_match = MagicMock()
            mock_match.ruleId = "COMMA_PARENTHESIS_WHITESPACE"
            mock_match.offset = 10
            mock_match.errorLength = 5

            # Normal Russian text
            normal_text = "Привет, мир! Как дела?"
            assert not helper._is_csv_or_code_context(normal_text, mock_match)
