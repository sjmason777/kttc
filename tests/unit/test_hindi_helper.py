"""Unit tests for Hindi language helper.

Tests HindiLanguageHelper with mocked Indic NLP Library, Stanza, and Spello dependencies.
"""

from unittest.mock import MagicMock, patch

import pytest

from kttc.core.models import ErrorAnnotation, ErrorSeverity
from kttc.helpers.base import MorphologyInfo


# Mock classes for Stanza
class MockStanzaWord:
    """Mock Stanza Word object."""

    def __init__(
        self,
        text: str,
        upos: str = "NOUN",
        feats: str | None = None,
        deprel: str | None = None,
        id: int = 1,
        start_char: int = 0,
        end_char: int = 5,
    ):
        self.text = text
        self.upos = upos
        self.feats = feats
        self.deprel = deprel
        self.id = id
        self.start_char = start_char
        self.end_char = end_char


class MockStanzaToken:
    """Mock Stanza Token object."""

    def __init__(self, word: MockStanzaWord, start_char: int = 0, end_char: int = 5):
        self.words = [word]
        self.start_char = start_char
        self.end_char = end_char


class MockStanzaEntity:
    """Mock Stanza Entity object."""

    def __init__(self, text: str, entity_type: str, start_char: int, end_char: int):
        self.text = text
        self.type = entity_type
        self.start_char = start_char
        self.end_char = end_char


class MockStanzaSentence:
    """Mock Stanza Sentence object."""

    def __init__(
        self,
        tokens: list[MockStanzaToken],
        words: list[MockStanzaWord],
        ents: list[MockStanzaEntity],
    ):
        self.tokens = tokens
        self.words = words
        self.ents = ents


class MockStanzaDoc:
    """Mock Stanza Document object."""

    def __init__(self, sentences: list[MockStanzaSentence]):
        self.sentences = sentences


class MockStanzaPipeline:
    """Mock Stanza Pipeline."""

    def __init__(self, doc: MockStanzaDoc | None = None):
        self._doc = doc

    def __call__(self, text: str) -> MockStanzaDoc:
        if self._doc:
            return self._doc
        # Return empty document by default
        return MockStanzaDoc([])


class MockNormalizer:
    """Mock Indic NLP Normalizer."""

    def normalize(self, text: str) -> str:
        return text


class MockSpellCorrectionModel:
    """Mock Spello SpellCorrectionModel."""

    def __init__(self, corrected_text: str | None = None):
        self._corrected = corrected_text

    def spell_correct(self, text: str) -> str:
        return self._corrected if self._corrected is not None else text


@pytest.mark.unit
class TestHindiHelperInitialization:
    """Tests for HindiLanguageHelper initialization."""

    def test_init_without_any_tools(self) -> None:
        """Test initialization without any NLP tools."""
        with (
            patch.dict("sys.modules", {"indicnlp": None, "stanza": None, "spello": None}),
            patch("kttc.helpers.hindi.INDIC_NLP_AVAILABLE", False),
            patch("kttc.helpers.hindi.STANZA_AVAILABLE", False),
            patch("kttc.helpers.hindi.SPELLO_AVAILABLE", False),
        ):
            from kttc.helpers.hindi import HindiLanguageHelper

            helper = HindiLanguageHelper()
            assert helper.language_code == "hi"
            assert not helper.is_available()

    def test_init_with_indic_nlp_success(self) -> None:
        """Test initialization with Indic NLP Library."""
        mock_normalizer = MockNormalizer()
        mock_factory = MagicMock()
        mock_factory.get_normalizer.return_value = mock_normalizer

        with (
            patch("kttc.helpers.hindi.INDIC_NLP_AVAILABLE", True),
            patch("kttc.helpers.hindi.STANZA_AVAILABLE", False),
            patch("kttc.helpers.hindi.SPELLO_AVAILABLE", False),
            patch("kttc.helpers.hindi.IndicNormalizerFactory", return_value=mock_factory),
        ):
            from kttc.helpers.hindi import HindiLanguageHelper

            helper = HindiLanguageHelper()
            assert helper.is_available()
            mock_factory.get_normalizer.assert_called_once_with("hi")

    def test_init_with_indic_nlp_failure(self) -> None:
        """Test initialization when Indic NLP initialization fails."""
        mock_factory = MagicMock()
        mock_factory.get_normalizer.side_effect = Exception("Init failed")

        with (
            patch("kttc.helpers.hindi.INDIC_NLP_AVAILABLE", True),
            patch("kttc.helpers.hindi.STANZA_AVAILABLE", False),
            patch("kttc.helpers.hindi.SPELLO_AVAILABLE", False),
            patch("kttc.helpers.hindi.IndicNormalizerFactory", return_value=mock_factory),
        ):
            from kttc.helpers.hindi import HindiLanguageHelper

            helper = HindiLanguageHelper()
            assert not helper.is_available()

    def test_init_with_stanza_success(self) -> None:
        """Test initialization with Stanza pipeline."""
        mock_normalizer = MockNormalizer()
        mock_factory = MagicMock()
        mock_factory.get_normalizer.return_value = mock_normalizer
        mock_pipeline = MockStanzaPipeline()

        with (
            patch("kttc.helpers.hindi.INDIC_NLP_AVAILABLE", True),
            patch("kttc.helpers.hindi.STANZA_AVAILABLE", True),
            patch("kttc.helpers.hindi.SPELLO_AVAILABLE", False),
            patch("kttc.helpers.hindi.IndicNormalizerFactory", return_value=mock_factory),
            patch("kttc.helpers.hindi.stanza") as mock_stanza_module,
        ):
            mock_stanza_module.Pipeline.return_value = mock_pipeline
            mock_stanza_module.DownloadMethod.REUSE_RESOURCES = "reuse"

            from kttc.helpers.hindi import HindiLanguageHelper

            helper = HindiLanguageHelper()
            assert helper._stanza_available

    def test_init_with_stanza_failure(self) -> None:
        """Test initialization when Stanza fails."""
        mock_normalizer = MockNormalizer()
        mock_factory = MagicMock()
        mock_factory.get_normalizer.return_value = mock_normalizer

        with (
            patch("kttc.helpers.hindi.INDIC_NLP_AVAILABLE", True),
            patch("kttc.helpers.hindi.STANZA_AVAILABLE", True),
            patch("kttc.helpers.hindi.SPELLO_AVAILABLE", False),
            patch("kttc.helpers.hindi.IndicNormalizerFactory", return_value=mock_factory),
            patch("kttc.helpers.hindi.stanza") as mock_stanza_module,
        ):
            mock_stanza_module.Pipeline.side_effect = Exception("Stanza init failed")
            mock_stanza_module.DownloadMethod.REUSE_RESOURCES = "reuse"

            from kttc.helpers.hindi import HindiLanguageHelper

            helper = HindiLanguageHelper()
            assert not helper._stanza_available

    def test_init_with_spello_success(self) -> None:
        """Test initialization with Spello spell checker."""
        mock_normalizer = MockNormalizer()
        mock_factory = MagicMock()
        mock_factory.get_normalizer.return_value = mock_normalizer
        mock_spellchecker = MockSpellCorrectionModel()

        with (
            patch("kttc.helpers.hindi.INDIC_NLP_AVAILABLE", True),
            patch("kttc.helpers.hindi.STANZA_AVAILABLE", False),
            patch("kttc.helpers.hindi.SPELLO_AVAILABLE", True),
            patch("kttc.helpers.hindi.IndicNormalizerFactory", return_value=mock_factory),
            patch("kttc.helpers.hindi.SpellCorrectionModel", return_value=mock_spellchecker),
        ):
            from kttc.helpers.hindi import HindiLanguageHelper

            helper = HindiLanguageHelper()
            assert helper._spello_available

    def test_init_with_spello_failure(self) -> None:
        """Test initialization when Spello fails."""
        mock_normalizer = MockNormalizer()
        mock_factory = MagicMock()
        mock_factory.get_normalizer.return_value = mock_normalizer

        with (
            patch("kttc.helpers.hindi.INDIC_NLP_AVAILABLE", True),
            patch("kttc.helpers.hindi.STANZA_AVAILABLE", False),
            patch("kttc.helpers.hindi.SPELLO_AVAILABLE", True),
            patch("kttc.helpers.hindi.IndicNormalizerFactory", return_value=mock_factory),
            patch("kttc.helpers.hindi.SpellCorrectionModel") as mock_spello_cls,
        ):
            mock_spello_cls.side_effect = Exception("Spello init failed")

            from kttc.helpers.hindi import HindiLanguageHelper

            helper = HindiLanguageHelper()
            assert not helper._spello_available


@pytest.mark.unit
class TestHindiHelperBasicMethods:
    """Tests for basic helper methods."""

    def test_language_code(self) -> None:
        """Test language code property."""
        with (
            patch("kttc.helpers.hindi.INDIC_NLP_AVAILABLE", False),
            patch("kttc.helpers.hindi.STANZA_AVAILABLE", False),
            patch("kttc.helpers.hindi.SPELLO_AVAILABLE", False),
        ):
            from kttc.helpers.hindi import HindiLanguageHelper

            helper = HindiLanguageHelper()
            assert helper.language_code == "hi"

    def test_verify_word_exists_without_tools(self) -> None:
        """Test verify_word_exists without NLP tools (fallback)."""
        with (
            patch("kttc.helpers.hindi.INDIC_NLP_AVAILABLE", False),
            patch("kttc.helpers.hindi.STANZA_AVAILABLE", False),
            patch("kttc.helpers.hindi.SPELLO_AVAILABLE", False),
        ):
            from kttc.helpers.hindi import HindiLanguageHelper

            helper = HindiLanguageHelper()
            assert helper.verify_word_exists("मैं", "मैं स्कूल जाता हूं")
            assert not helper.verify_word_exists("गया", "मैं स्कूल जाता हूं")

    def test_verify_word_exists_with_tokenization(self) -> None:
        """Test verify_word_exists with tokenization."""
        mock_normalizer = MockNormalizer()
        mock_factory = MagicMock()
        mock_factory.get_normalizer.return_value = mock_normalizer

        with (
            patch("kttc.helpers.hindi.INDIC_NLP_AVAILABLE", True),
            patch("kttc.helpers.hindi.STANZA_AVAILABLE", False),
            patch("kttc.helpers.hindi.SPELLO_AVAILABLE", False),
            patch("kttc.helpers.hindi.IndicNormalizerFactory", return_value=mock_factory),
            patch("kttc.helpers.hindi.trivial_tokenize", return_value=["मैं", "स्कूल", "जाता", "हूं"]),
        ):
            from kttc.helpers.hindi import HindiLanguageHelper

            helper = HindiLanguageHelper()
            assert helper.verify_word_exists("स्कूल", "मैं स्कूल जाता हूं")
            assert not helper.verify_word_exists("गया", "मैं स्कूल जाता हूं")

    def test_verify_error_position_valid(self) -> None:
        """Test verify_error_position with valid position."""
        with (
            patch("kttc.helpers.hindi.INDIC_NLP_AVAILABLE", False),
            patch("kttc.helpers.hindi.STANZA_AVAILABLE", False),
            patch("kttc.helpers.hindi.SPELLO_AVAILABLE", False),
        ):
            from kttc.helpers.hindi import HindiLanguageHelper

            helper = HindiLanguageHelper()
            error = ErrorAnnotation(
                category="fluency",
                subcategory="grammar",
                severity=ErrorSeverity.MINOR,
                location=(0, 5),
                description="Test error",
            )
            text = "मैं स्कूल जाता हूं"
            assert helper.verify_error_position(error, text)

    def test_verify_error_position_invalid_bounds(self) -> None:
        """Test verify_error_position with invalid bounds."""
        with (
            patch("kttc.helpers.hindi.INDIC_NLP_AVAILABLE", False),
            patch("kttc.helpers.hindi.STANZA_AVAILABLE", False),
            patch("kttc.helpers.hindi.SPELLO_AVAILABLE", False),
        ):
            from kttc.helpers.hindi import HindiLanguageHelper

            helper = HindiLanguageHelper()
            text = "Short"

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

            # Start >= end
            error3 = ErrorAnnotation(
                category="fluency",
                subcategory="test",
                severity=ErrorSeverity.MINOR,
                location=(5, 5),
                description="Test",
            )
            assert not helper.verify_error_position(error3, text)

    def test_verify_error_position_empty_substring(self) -> None:
        """Test verify_error_position with whitespace-only substring."""
        with (
            patch("kttc.helpers.hindi.INDIC_NLP_AVAILABLE", False),
            patch("kttc.helpers.hindi.STANZA_AVAILABLE", False),
            patch("kttc.helpers.hindi.SPELLO_AVAILABLE", False),
        ):
            from kttc.helpers.hindi import HindiLanguageHelper

            helper = HindiLanguageHelper()
            text = "Hello   World"  # Multiple spaces

            error = ErrorAnnotation(
                category="fluency",
                subcategory="test",
                severity=ErrorSeverity.MINOR,
                location=(5, 8),
                description="Test",  # Just spaces
            )
            assert not helper.verify_error_position(error, text)


@pytest.mark.unit
class TestHindiHelperTokenization:
    """Tests for tokenization methods."""

    def test_tokenize_empty_text(self) -> None:
        """Test tokenization with empty text."""
        with (
            patch("kttc.helpers.hindi.INDIC_NLP_AVAILABLE", False),
            patch("kttc.helpers.hindi.STANZA_AVAILABLE", False),
            patch("kttc.helpers.hindi.SPELLO_AVAILABLE", False),
        ):
            from kttc.helpers.hindi import HindiLanguageHelper

            helper = HindiLanguageHelper()
            assert helper.tokenize("") == []
            assert helper.tokenize("   ") == []

    def test_tokenize_without_tools(self) -> None:
        """Test tokenization without NLP tools (fallback)."""
        with (
            patch("kttc.helpers.hindi.INDIC_NLP_AVAILABLE", False),
            patch("kttc.helpers.hindi.STANZA_AVAILABLE", False),
            patch("kttc.helpers.hindi.SPELLO_AVAILABLE", False),
        ):
            from kttc.helpers.hindi import HindiLanguageHelper

            helper = HindiLanguageHelper()
            tokens = helper.tokenize("मैं स्कूल जाता हूं")
            assert len(tokens) > 0
            # Check token structure (word, start, end)
            for token in tokens:
                assert len(token) == 3
                assert isinstance(token[0], str)
                assert isinstance(token[1], int)
                assert isinstance(token[2], int)

    def test_tokenize_with_indic_nlp(self) -> None:
        """Test tokenization with Indic NLP Library."""
        mock_normalizer = MockNormalizer()
        mock_factory = MagicMock()
        mock_factory.get_normalizer.return_value = mock_normalizer

        with (
            patch("kttc.helpers.hindi.INDIC_NLP_AVAILABLE", True),
            patch("kttc.helpers.hindi.STANZA_AVAILABLE", False),
            patch("kttc.helpers.hindi.SPELLO_AVAILABLE", False),
            patch("kttc.helpers.hindi.IndicNormalizerFactory", return_value=mock_factory),
            patch("kttc.helpers.hindi.trivial_tokenize", return_value=["मैं", "स्कूल", "जाता", "हूं"]),
        ):
            from kttc.helpers.hindi import HindiLanguageHelper

            helper = HindiLanguageHelper()
            text = "मैं स्कूल जाता हूं"
            tokens = helper.tokenize(text)

            assert len(tokens) == 4
            assert tokens[0][0] == "मैं"

    def test_tokenize_exception_handling(self) -> None:
        """Test tokenization handles exceptions gracefully."""
        mock_normalizer = MagicMock()
        mock_normalizer.normalize.side_effect = Exception("Normalization failed")
        mock_factory = MagicMock()
        mock_factory.get_normalizer.return_value = mock_normalizer

        with (
            patch("kttc.helpers.hindi.INDIC_NLP_AVAILABLE", True),
            patch("kttc.helpers.hindi.STANZA_AVAILABLE", False),
            patch("kttc.helpers.hindi.SPELLO_AVAILABLE", False),
            patch("kttc.helpers.hindi.IndicNormalizerFactory", return_value=mock_factory),
        ):
            from kttc.helpers.hindi import HindiLanguageHelper

            helper = HindiLanguageHelper()
            tokens = helper.tokenize("test text")
            assert tokens == []

    def test_find_word_positions_static_method(self) -> None:
        """Test _find_word_positions static method."""
        with (
            patch("kttc.helpers.hindi.INDIC_NLP_AVAILABLE", False),
            patch("kttc.helpers.hindi.STANZA_AVAILABLE", False),
            patch("kttc.helpers.hindi.SPELLO_AVAILABLE", False),
        ):
            from kttc.helpers.hindi import HindiLanguageHelper

            text = "Hello World Test"
            words = ["Hello", "World", "Test"]
            positions = HindiLanguageHelper._find_word_positions(text, words)

            assert len(positions) == 3
            assert positions[0] == ("Hello", 0, 5)
            assert positions[1] == ("World", 6, 11)
            assert positions[2] == ("Test", 12, 16)

    def test_find_word_positions_with_empty_words(self) -> None:
        """Test _find_word_positions handles empty words."""
        with (
            patch("kttc.helpers.hindi.INDIC_NLP_AVAILABLE", False),
            patch("kttc.helpers.hindi.STANZA_AVAILABLE", False),
            patch("kttc.helpers.hindi.SPELLO_AVAILABLE", False),
        ):
            from kttc.helpers.hindi import HindiLanguageHelper

            text = "Hello World"
            words = ["Hello", "", "  ", "World"]
            positions = HindiLanguageHelper._find_word_positions(text, words)

            assert len(positions) == 2


@pytest.mark.unit
class TestHindiHelperMorphology:
    """Tests for morphology analysis."""

    def test_analyze_morphology_without_stanza(self) -> None:
        """Test morphology analysis without Stanza."""
        with (
            patch("kttc.helpers.hindi.INDIC_NLP_AVAILABLE", False),
            patch("kttc.helpers.hindi.STANZA_AVAILABLE", False),
            patch("kttc.helpers.hindi.SPELLO_AVAILABLE", False),
        ):
            from kttc.helpers.hindi import HindiLanguageHelper

            helper = HindiLanguageHelper()
            result = helper.analyze_morphology("मैं स्कूल जाता हूं")
            assert result == []

    def test_analyze_morphology_with_stanza(self) -> None:
        """Test morphology analysis with Stanza."""
        mock_normalizer = MockNormalizer()
        mock_factory = MagicMock()
        mock_factory.get_normalizer.return_value = mock_normalizer

        # Create mock Stanza document
        mock_word = MockStanzaWord(text="मैं", upos="PRON", feats="Gender=Masc|Number=Sing|Case=Nom")
        mock_token = MockStanzaToken(mock_word, start_char=0, end_char=4)
        mock_sentence = MockStanzaSentence([mock_token], [mock_word], [])
        mock_doc = MockStanzaDoc([mock_sentence])
        mock_pipeline = MockStanzaPipeline(mock_doc)

        with (
            patch("kttc.helpers.hindi.INDIC_NLP_AVAILABLE", True),
            patch("kttc.helpers.hindi.STANZA_AVAILABLE", True),
            patch("kttc.helpers.hindi.SPELLO_AVAILABLE", False),
            patch("kttc.helpers.hindi.IndicNormalizerFactory", return_value=mock_factory),
            patch("kttc.helpers.hindi.stanza") as mock_stanza_module,
        ):
            mock_stanza_module.Pipeline.return_value = mock_pipeline
            mock_stanza_module.DownloadMethod.REUSE_RESOURCES = "reuse"

            from kttc.helpers.hindi import HindiLanguageHelper

            helper = HindiLanguageHelper()
            result = helper.analyze_morphology("मैं")

            assert len(result) == 1
            assert isinstance(result[0], MorphologyInfo)
            assert result[0].word == "मैं"
            assert result[0].pos == "PRON"
            assert result[0].gender == "Masc"
            assert result[0].number == "Sing"

    def test_analyze_morphology_exception(self) -> None:
        """Test morphology analysis handles exceptions."""
        mock_normalizer = MockNormalizer()
        mock_factory = MagicMock()
        mock_factory.get_normalizer.return_value = mock_normalizer
        mock_pipeline = MagicMock()
        mock_pipeline.side_effect = Exception("Stanza failed")

        with (
            patch("kttc.helpers.hindi.INDIC_NLP_AVAILABLE", True),
            patch("kttc.helpers.hindi.STANZA_AVAILABLE", True),
            patch("kttc.helpers.hindi.SPELLO_AVAILABLE", False),
            patch("kttc.helpers.hindi.IndicNormalizerFactory", return_value=mock_factory),
            patch("kttc.helpers.hindi.stanza") as mock_stanza_module,
        ):
            mock_stanza_module.Pipeline.return_value = mock_pipeline
            mock_stanza_module.DownloadMethod.REUSE_RESOURCES = "reuse"

            from kttc.helpers.hindi import HindiLanguageHelper

            helper = HindiLanguageHelper()
            result = helper.analyze_morphology("test")
            assert result == []

    def test_parse_feats_string_static_method(self) -> None:
        """Test _parse_feats_string static method."""
        with (
            patch("kttc.helpers.hindi.INDIC_NLP_AVAILABLE", False),
            patch("kttc.helpers.hindi.STANZA_AVAILABLE", False),
            patch("kttc.helpers.hindi.SPELLO_AVAILABLE", False),
        ):
            from kttc.helpers.hindi import HindiLanguageHelper

            # Normal case
            result = HindiLanguageHelper._parse_feats_string("Gender=Masc|Number=Sing")
            assert result == {"Gender": "Masc", "Number": "Sing"}

            # Empty string
            assert HindiLanguageHelper._parse_feats_string("") == {}

            # None
            assert HindiLanguageHelper._parse_feats_string(None) == {}


@pytest.mark.unit
class TestHindiHelperGrammar:
    """Tests for grammar checking."""

    def test_check_grammar_without_stanza(self) -> None:
        """Test grammar check without Stanza returns empty list."""
        with (
            patch("kttc.helpers.hindi.INDIC_NLP_AVAILABLE", False),
            patch("kttc.helpers.hindi.STANZA_AVAILABLE", False),
            patch("kttc.helpers.hindi.SPELLO_AVAILABLE", False),
        ):
            from kttc.helpers.hindi import HindiLanguageHelper

            helper = HindiLanguageHelper()
            errors = helper.check_grammar("मैं स्कूल जाता हूं")
            assert errors == []

    def test_check_grammar_ergative_case_error(self) -> None:
        """Test ergative case marker detection."""
        mock_normalizer = MockNormalizer()
        mock_factory = MagicMock()
        mock_factory.get_normalizer.return_value = mock_normalizer

        # Create document with ने but non-past tense verb
        ne_word = MockStanzaWord(text="ने", upos="ADP", start_char=0, end_char=2)
        verb_word = MockStanzaWord(
            text="खाता", upos="VERB", feats="Tense=Pres", start_char=3, end_char=8
        )
        mock_sentence = MockStanzaSentence([], [ne_word, verb_word], [])
        mock_doc = MockStanzaDoc([mock_sentence])
        mock_pipeline = MockStanzaPipeline(mock_doc)

        with (
            patch("kttc.helpers.hindi.INDIC_NLP_AVAILABLE", True),
            patch("kttc.helpers.hindi.STANZA_AVAILABLE", True),
            patch("kttc.helpers.hindi.SPELLO_AVAILABLE", False),
            patch("kttc.helpers.hindi.IndicNormalizerFactory", return_value=mock_factory),
            patch("kttc.helpers.hindi.stanza") as mock_stanza_module,
        ):
            mock_stanza_module.Pipeline.return_value = mock_pipeline
            mock_stanza_module.DownloadMethod.REUSE_RESOURCES = "reuse"

            from kttc.helpers.hindi import HindiLanguageHelper

            helper = HindiLanguageHelper()
            errors = helper.check_grammar("ने खाता")

            assert len(errors) >= 1
            assert errors[0].subcategory == "hindi_grammar_case"

    def test_check_grammar_subject_verb_agreement_error(self) -> None:
        """Test subject-verb agreement detection."""
        mock_normalizer = MockNormalizer()
        mock_factory = MagicMock()
        mock_factory.get_normalizer.return_value = mock_normalizer

        # Create document with gender mismatch
        subject_word = MockStanzaWord(
            text="लड़की",
            upos="NOUN",
            feats="Gender=Fem|Number=Sing",
            deprel="nsubj",
            id=1,
            start_char=0,
            end_char=5,
        )
        verb_word = MockStanzaWord(
            text="गया",
            upos="VERB",
            feats="Gender=Masc|Number=Sing",
            id=2,
            start_char=6,
            end_char=10,
        )
        mock_sentence = MockStanzaSentence([], [subject_word, verb_word], [])
        mock_doc = MockStanzaDoc([mock_sentence])
        mock_pipeline = MockStanzaPipeline(mock_doc)

        with (
            patch("kttc.helpers.hindi.INDIC_NLP_AVAILABLE", True),
            patch("kttc.helpers.hindi.STANZA_AVAILABLE", True),
            patch("kttc.helpers.hindi.SPELLO_AVAILABLE", False),
            patch("kttc.helpers.hindi.IndicNormalizerFactory", return_value=mock_factory),
            patch("kttc.helpers.hindi.stanza") as mock_stanza_module,
        ):
            mock_stanza_module.Pipeline.return_value = mock_pipeline
            mock_stanza_module.DownloadMethod.REUSE_RESOURCES = "reuse"

            from kttc.helpers.hindi import HindiLanguageHelper

            helper = HindiLanguageHelper()
            errors = helper.check_grammar("लड़की गया")

            assert len(errors) >= 1
            error = next((e for e in errors if e.subcategory == "hindi_grammar_agreement"), None)
            assert error is not None
            assert "gender" in error.description.lower()

    def test_check_grammar_word_order_error(self) -> None:
        """Test SOV word order detection."""
        mock_normalizer = MockNormalizer()
        mock_factory = MagicMock()
        mock_factory.get_normalizer.return_value = mock_normalizer

        # Create document with verb before subject (unusual in Hindi)
        verb_word = MockStanzaWord(text="खाता", upos="VERB", id=1, start_char=0, end_char=5)
        subject_word = MockStanzaWord(
            text="वह", upos="PRON", deprel="nsubj", id=5, start_char=20, end_char=22
        )
        mock_sentence = MockStanzaSentence([], [verb_word, subject_word], [])
        mock_doc = MockStanzaDoc([mock_sentence])
        mock_pipeline = MockStanzaPipeline(mock_doc)

        with (
            patch("kttc.helpers.hindi.INDIC_NLP_AVAILABLE", True),
            patch("kttc.helpers.hindi.STANZA_AVAILABLE", True),
            patch("kttc.helpers.hindi.SPELLO_AVAILABLE", False),
            patch("kttc.helpers.hindi.IndicNormalizerFactory", return_value=mock_factory),
            patch("kttc.helpers.hindi.stanza") as mock_stanza_module,
        ):
            mock_stanza_module.Pipeline.return_value = mock_pipeline
            mock_stanza_module.DownloadMethod.REUSE_RESOURCES = "reuse"

            from kttc.helpers.hindi import HindiLanguageHelper

            helper = HindiLanguageHelper()
            errors = helper.check_grammar("खाता ... वह")

            # Should detect unusual word order
            word_order_error = next(
                (e for e in errors if e.subcategory == "hindi_grammar_word_order"), None
            )
            assert word_order_error is not None

    def test_check_grammar_multiple_case_markers(self) -> None:
        """Test consecutive case markers detection."""
        mock_normalizer = MockNormalizer()
        mock_factory = MagicMock()
        mock_factory.get_normalizer.return_value = mock_normalizer

        # Create document with consecutive case markers
        case1 = MockStanzaWord(text="में", upos="ADP", start_char=0, end_char=3)
        case2 = MockStanzaWord(text="पर", upos="ADP", start_char=4, end_char=6)
        mock_sentence = MockStanzaSentence([], [case1, case2], [])
        mock_doc = MockStanzaDoc([mock_sentence])
        mock_pipeline = MockStanzaPipeline(mock_doc)

        with (
            patch("kttc.helpers.hindi.INDIC_NLP_AVAILABLE", True),
            patch("kttc.helpers.hindi.STANZA_AVAILABLE", True),
            patch("kttc.helpers.hindi.SPELLO_AVAILABLE", False),
            patch("kttc.helpers.hindi.IndicNormalizerFactory", return_value=mock_factory),
            patch("kttc.helpers.hindi.stanza") as mock_stanza_module,
        ):
            mock_stanza_module.Pipeline.return_value = mock_pipeline
            mock_stanza_module.DownloadMethod.REUSE_RESOURCES = "reuse"

            from kttc.helpers.hindi import HindiLanguageHelper

            helper = HindiLanguageHelper()
            errors = helper.check_grammar("में पर")

            case_error = next((e for e in errors if "case" in e.subcategory.lower()), None)
            assert case_error is not None

    def test_check_grammar_exception(self) -> None:
        """Test grammar check handles exceptions."""
        mock_normalizer = MockNormalizer()
        mock_factory = MagicMock()
        mock_factory.get_normalizer.return_value = mock_normalizer
        mock_pipeline = MagicMock()
        mock_pipeline.side_effect = Exception("Stanza failed")

        with (
            patch("kttc.helpers.hindi.INDIC_NLP_AVAILABLE", True),
            patch("kttc.helpers.hindi.STANZA_AVAILABLE", True),
            patch("kttc.helpers.hindi.SPELLO_AVAILABLE", False),
            patch("kttc.helpers.hindi.IndicNormalizerFactory", return_value=mock_factory),
            patch("kttc.helpers.hindi.stanza") as mock_stanza_module,
        ):
            mock_stanza_module.Pipeline.return_value = mock_pipeline
            mock_stanza_module.DownloadMethod.REUSE_RESOURCES = "reuse"

            from kttc.helpers.hindi import HindiLanguageHelper

            helper = HindiLanguageHelper()
            errors = helper.check_grammar("test")
            assert errors == []


@pytest.mark.unit
class TestHindiHelperSpelling:
    """Tests for spelling checking."""

    def test_check_spelling_without_spello(self) -> None:
        """Test spelling check without Spello returns empty list."""
        with (
            patch("kttc.helpers.hindi.INDIC_NLP_AVAILABLE", False),
            patch("kttc.helpers.hindi.STANZA_AVAILABLE", False),
            patch("kttc.helpers.hindi.SPELLO_AVAILABLE", False),
        ):
            from kttc.helpers.hindi import HindiLanguageHelper

            helper = HindiLanguageHelper()
            errors = helper.check_spelling("मैं स्कूल जाता हूं")
            assert errors == []

    def test_check_spelling_no_errors(self) -> None:
        """Test spelling check with no errors."""
        mock_normalizer = MockNormalizer()
        mock_factory = MagicMock()
        mock_factory.get_normalizer.return_value = mock_normalizer
        mock_spellchecker = MockSpellCorrectionModel("मैं स्कूल जाता हूं")

        with (
            patch("kttc.helpers.hindi.INDIC_NLP_AVAILABLE", True),
            patch("kttc.helpers.hindi.STANZA_AVAILABLE", False),
            patch("kttc.helpers.hindi.SPELLO_AVAILABLE", True),
            patch("kttc.helpers.hindi.IndicNormalizerFactory", return_value=mock_factory),
            patch("kttc.helpers.hindi.SpellCorrectionModel", return_value=mock_spellchecker),
            patch("kttc.helpers.hindi.trivial_tokenize", return_value=["मैं", "स्कूल", "जाता", "हूं"]),
        ):
            from kttc.helpers.hindi import HindiLanguageHelper

            helper = HindiLanguageHelper()
            errors = helper.check_spelling("मैं स्कूल जाता हूं")
            assert errors == []

    def test_check_spelling_with_replace_error(self) -> None:
        """Test spelling check detects replacement errors."""
        mock_normalizer = MockNormalizer()
        mock_factory = MagicMock()
        mock_factory.get_normalizer.return_value = mock_normalizer
        # Misspelled "सकूल" -> corrected "स्कूल"
        mock_spellchecker = MockSpellCorrectionModel("मैं स्कूल जाता हूं")

        call_count = [0]

        def mock_tokenize(text: str, lang: str = "hi") -> list[str]:
            call_count[0] += 1
            if call_count[0] == 1:  # Original text
                return ["मैं", "सकूल", "जाता", "हूं"]
            return ["मैं", "स्कूल", "जाता", "हूं"]  # Corrected text

        with (
            patch("kttc.helpers.hindi.INDIC_NLP_AVAILABLE", True),
            patch("kttc.helpers.hindi.STANZA_AVAILABLE", False),
            patch("kttc.helpers.hindi.SPELLO_AVAILABLE", True),
            patch("kttc.helpers.hindi.IndicNormalizerFactory", return_value=mock_factory),
            patch("kttc.helpers.hindi.SpellCorrectionModel", return_value=mock_spellchecker),
            patch("kttc.helpers.hindi.trivial_tokenize", side_effect=mock_tokenize),
        ):
            from kttc.helpers.hindi import HindiLanguageHelper

            helper = HindiLanguageHelper()
            errors = helper.check_spelling("मैं सकूल जाता हूं")

            assert len(errors) >= 1
            assert errors[0].subcategory == "hindi_spelling"

    def test_check_spelling_exception(self) -> None:
        """Test spelling check handles exceptions."""
        mock_normalizer = MockNormalizer()
        mock_factory = MagicMock()
        mock_factory.get_normalizer.return_value = mock_normalizer
        mock_spellchecker = MagicMock()
        mock_spellchecker.spell_correct.side_effect = Exception("Spell check failed")

        with (
            patch("kttc.helpers.hindi.INDIC_NLP_AVAILABLE", True),
            patch("kttc.helpers.hindi.STANZA_AVAILABLE", False),
            patch("kttc.helpers.hindi.SPELLO_AVAILABLE", True),
            patch("kttc.helpers.hindi.IndicNormalizerFactory", return_value=mock_factory),
            patch("kttc.helpers.hindi.SpellCorrectionModel", return_value=mock_spellchecker),
        ):
            from kttc.helpers.hindi import HindiLanguageHelper

            helper = HindiLanguageHelper()
            errors = helper.check_spelling("test")
            assert errors == []


@pytest.mark.unit
class TestHindiHelperEnrichment:
    """Tests for enrichment data methods."""

    def test_get_enrichment_data_without_tools(self) -> None:
        """Test enrichment without NLP tools."""
        with (
            patch("kttc.helpers.hindi.INDIC_NLP_AVAILABLE", False),
            patch("kttc.helpers.hindi.STANZA_AVAILABLE", False),
            patch("kttc.helpers.hindi.SPELLO_AVAILABLE", False),
        ):
            from kttc.helpers.hindi import HindiLanguageHelper

            helper = HindiLanguageHelper()
            result = helper.get_enrichment_data("test")
            assert result == {"has_morphology": False}

    def test_get_enrichment_data_with_normalization(self) -> None:
        """Test enrichment with normalization."""
        mock_normalizer = MockNormalizer()
        mock_factory = MagicMock()
        mock_factory.get_normalizer.return_value = mock_normalizer

        with (
            patch("kttc.helpers.hindi.INDIC_NLP_AVAILABLE", True),
            patch("kttc.helpers.hindi.STANZA_AVAILABLE", False),
            patch("kttc.helpers.hindi.SPELLO_AVAILABLE", False),
            patch("kttc.helpers.hindi.IndicNormalizerFactory", return_value=mock_factory),
        ):
            from kttc.helpers.hindi import HindiLanguageHelper

            helper = HindiLanguageHelper()
            result = helper.get_enrichment_data("मैं स्कूल जाता हूं")

            assert result["has_morphology"]
            assert "normalized_text" in result

    def test_get_enrichment_data_with_stanza(self) -> None:
        """Test enrichment with Stanza analysis."""
        mock_normalizer = MockNormalizer()
        mock_factory = MagicMock()
        mock_factory.get_normalizer.return_value = mock_normalizer

        mock_word = MockStanzaWord(text="मैं", upos="PRON")
        mock_token = MockStanzaToken(mock_word)
        mock_entity = MockStanzaEntity("नरेंद्र मोदी", "PER", 0, 12)
        mock_sentence = MockStanzaSentence([mock_token], [mock_word], [mock_entity])
        mock_doc = MockStanzaDoc([mock_sentence])
        mock_pipeline = MockStanzaPipeline(mock_doc)

        with (
            patch("kttc.helpers.hindi.INDIC_NLP_AVAILABLE", True),
            patch("kttc.helpers.hindi.STANZA_AVAILABLE", True),
            patch("kttc.helpers.hindi.SPELLO_AVAILABLE", False),
            patch("kttc.helpers.hindi.IndicNormalizerFactory", return_value=mock_factory),
            patch("kttc.helpers.hindi.stanza") as mock_stanza_module,
        ):
            mock_stanza_module.Pipeline.return_value = mock_pipeline
            mock_stanza_module.DownloadMethod.REUSE_RESOURCES = "reuse"

            from kttc.helpers.hindi import HindiLanguageHelper

            helper = HindiLanguageHelper()
            result = helper.get_enrichment_data("नरेंद्र मोदी")

            assert result["has_morphology"]
            assert result.get("has_stanza")
            assert "word_count" in result
            assert "sentence_count" in result
            assert "pos_distribution" in result
            assert "entities" in result

    def test_get_enrichment_data_normalization_exception(self) -> None:
        """Test enrichment handles normalization exception."""
        mock_normalizer = MagicMock()
        mock_normalizer.normalize.side_effect = Exception("Normalize failed")
        mock_factory = MagicMock()
        mock_factory.get_normalizer.return_value = mock_normalizer

        with (
            patch("kttc.helpers.hindi.INDIC_NLP_AVAILABLE", True),
            patch("kttc.helpers.hindi.STANZA_AVAILABLE", False),
            patch("kttc.helpers.hindi.SPELLO_AVAILABLE", False),
            patch("kttc.helpers.hindi.IndicNormalizerFactory", return_value=mock_factory),
        ):
            from kttc.helpers.hindi import HindiLanguageHelper

            helper = HindiLanguageHelper()
            result = helper.get_enrichment_data("test")

            # Should still return basic data
            assert result["has_morphology"]
            assert "normalized_text" not in result

    def test_get_enrichment_data_stanza_exception(self) -> None:
        """Test enrichment handles Stanza exception."""
        mock_normalizer = MockNormalizer()
        mock_factory = MagicMock()
        mock_factory.get_normalizer.return_value = mock_normalizer
        mock_pipeline = MagicMock()
        mock_pipeline.side_effect = Exception("Stanza failed")

        with (
            patch("kttc.helpers.hindi.INDIC_NLP_AVAILABLE", True),
            patch("kttc.helpers.hindi.STANZA_AVAILABLE", True),
            patch("kttc.helpers.hindi.SPELLO_AVAILABLE", False),
            patch("kttc.helpers.hindi.IndicNormalizerFactory", return_value=mock_factory),
            patch("kttc.helpers.hindi.stanza") as mock_stanza_module,
        ):
            mock_stanza_module.Pipeline.return_value = mock_pipeline
            mock_stanza_module.DownloadMethod.REUSE_RESOURCES = "reuse"

            from kttc.helpers.hindi import HindiLanguageHelper

            helper = HindiLanguageHelper()
            result = helper.get_enrichment_data("test")

            # Should still return basic data without stanza
            assert result["has_morphology"]


@pytest.mark.unit
class TestHindiHelperEntities:
    """Tests for entity extraction and preservation."""

    def test_extract_entities_without_stanza(self) -> None:
        """Test entity extraction without Stanza."""
        with (
            patch("kttc.helpers.hindi.INDIC_NLP_AVAILABLE", False),
            patch("kttc.helpers.hindi.STANZA_AVAILABLE", False),
            patch("kttc.helpers.hindi.SPELLO_AVAILABLE", False),
        ):
            from kttc.helpers.hindi import HindiLanguageHelper

            helper = HindiLanguageHelper()
            entities = helper.extract_entities("नरेंद्र मोदी दिल्ली में हैं")
            assert entities == []

    def test_extract_entities_with_stanza(self) -> None:
        """Test entity extraction with Stanza NER."""
        mock_normalizer = MockNormalizer()
        mock_factory = MagicMock()
        mock_factory.get_normalizer.return_value = mock_normalizer

        mock_entity1 = MockStanzaEntity("नरेंद्र मोदी", "PER", 0, 12)
        mock_entity2 = MockStanzaEntity("दिल्ली", "LOC", 13, 19)
        mock_sentence = MockStanzaSentence([], [], [mock_entity1, mock_entity2])
        mock_doc = MockStanzaDoc([mock_sentence])
        mock_pipeline = MockStanzaPipeline(mock_doc)

        with (
            patch("kttc.helpers.hindi.INDIC_NLP_AVAILABLE", True),
            patch("kttc.helpers.hindi.STANZA_AVAILABLE", True),
            patch("kttc.helpers.hindi.SPELLO_AVAILABLE", False),
            patch("kttc.helpers.hindi.IndicNormalizerFactory", return_value=mock_factory),
            patch("kttc.helpers.hindi.stanza") as mock_stanza_module,
        ):
            mock_stanza_module.Pipeline.return_value = mock_pipeline
            mock_stanza_module.DownloadMethod.REUSE_RESOURCES = "reuse"

            from kttc.helpers.hindi import HindiLanguageHelper

            helper = HindiLanguageHelper()
            entities = helper.extract_entities("नरेंद्र मोदी दिल्ली में हैं")

            assert len(entities) == 2
            assert entities[0]["text"] == "नरेंद्र मोदी"
            assert entities[0]["type"] == "PER"
            assert entities[1]["text"] == "दिल्ली"
            assert entities[1]["type"] == "LOC"

    def test_extract_entities_exception(self) -> None:
        """Test entity extraction handles exceptions."""
        mock_normalizer = MockNormalizer()
        mock_factory = MagicMock()
        mock_factory.get_normalizer.return_value = mock_normalizer
        mock_pipeline = MagicMock()
        mock_pipeline.side_effect = Exception("NER failed")

        with (
            patch("kttc.helpers.hindi.INDIC_NLP_AVAILABLE", True),
            patch("kttc.helpers.hindi.STANZA_AVAILABLE", True),
            patch("kttc.helpers.hindi.SPELLO_AVAILABLE", False),
            patch("kttc.helpers.hindi.IndicNormalizerFactory", return_value=mock_factory),
            patch("kttc.helpers.hindi.stanza") as mock_stanza_module,
        ):
            mock_stanza_module.Pipeline.return_value = mock_pipeline
            mock_stanza_module.DownloadMethod.REUSE_RESOURCES = "reuse"

            from kttc.helpers.hindi import HindiLanguageHelper

            helper = HindiLanguageHelper()
            entities = helper.extract_entities("test")
            assert entities == []

    def test_check_entity_preservation_without_stanza(self) -> None:
        """Test entity preservation check without Stanza."""
        with (
            patch("kttc.helpers.hindi.INDIC_NLP_AVAILABLE", False),
            patch("kttc.helpers.hindi.STANZA_AVAILABLE", False),
            patch("kttc.helpers.hindi.SPELLO_AVAILABLE", False),
        ):
            from kttc.helpers.hindi import HindiLanguageHelper

            helper = HindiLanguageHelper()
            errors = helper.check_entity_preservation("John Smith", "जॉन स्मिथ")
            assert errors == []

    def test_check_entity_preservation_missing_entities(self) -> None:
        """Test entity preservation detects missing entities."""
        mock_normalizer = MockNormalizer()
        mock_factory = MagicMock()
        mock_factory.get_normalizer.return_value = mock_normalizer

        # Source has capitalized names, but translation has no entities
        mock_sentence = MockStanzaSentence([], [], [])  # No entities
        mock_doc = MockStanzaDoc([mock_sentence])
        mock_pipeline = MockStanzaPipeline(mock_doc)

        with (
            patch("kttc.helpers.hindi.INDIC_NLP_AVAILABLE", True),
            patch("kttc.helpers.hindi.STANZA_AVAILABLE", True),
            patch("kttc.helpers.hindi.SPELLO_AVAILABLE", False),
            patch("kttc.helpers.hindi.IndicNormalizerFactory", return_value=mock_factory),
            patch("kttc.helpers.hindi.stanza") as mock_stanza_module,
        ):
            mock_stanza_module.Pipeline.return_value = mock_pipeline
            mock_stanza_module.DownloadMethod.REUSE_RESOURCES = "reuse"

            from kttc.helpers.hindi import HindiLanguageHelper

            helper = HindiLanguageHelper()
            errors = helper.check_entity_preservation(
                "John Smith visited New York", "वह शहर गया"  # No entities preserved
            )

            assert len(errors) >= 1
            assert errors[0].subcategory == "entity_omission"

    def test_check_entity_preservation_entities_preserved(self) -> None:
        """Test entity preservation when entities are preserved."""
        mock_normalizer = MockNormalizer()
        mock_factory = MagicMock()
        mock_factory.get_normalizer.return_value = mock_normalizer

        # Translation has entities
        mock_entity = MockStanzaEntity("जॉन स्मिथ", "PER", 0, 10)
        mock_sentence = MockStanzaSentence([], [], [mock_entity])
        mock_doc = MockStanzaDoc([mock_sentence])
        mock_pipeline = MockStanzaPipeline(mock_doc)

        with (
            patch("kttc.helpers.hindi.INDIC_NLP_AVAILABLE", True),
            patch("kttc.helpers.hindi.STANZA_AVAILABLE", True),
            patch("kttc.helpers.hindi.SPELLO_AVAILABLE", False),
            patch("kttc.helpers.hindi.IndicNormalizerFactory", return_value=mock_factory),
            patch("kttc.helpers.hindi.stanza") as mock_stanza_module,
        ):
            mock_stanza_module.Pipeline.return_value = mock_pipeline
            mock_stanza_module.DownloadMethod.REUSE_RESOURCES = "reuse"

            from kttc.helpers.hindi import HindiLanguageHelper

            helper = HindiLanguageHelper()
            errors = helper.check_entity_preservation("John Smith arrived", "जॉन स्मिथ आ गए")

            # Should not report error if entities are preserved
            assert len(errors) == 0

    def test_check_entity_preservation_exception(self) -> None:
        """Test entity preservation handles exceptions."""
        mock_normalizer = MockNormalizer()
        mock_factory = MagicMock()
        mock_factory.get_normalizer.return_value = mock_normalizer
        mock_pipeline = MagicMock()
        mock_pipeline.side_effect = Exception("NER failed")

        with (
            patch("kttc.helpers.hindi.INDIC_NLP_AVAILABLE", True),
            patch("kttc.helpers.hindi.STANZA_AVAILABLE", True),
            patch("kttc.helpers.hindi.SPELLO_AVAILABLE", False),
            patch("kttc.helpers.hindi.IndicNormalizerFactory", return_value=mock_factory),
            patch("kttc.helpers.hindi.stanza") as mock_stanza_module,
        ):
            mock_stanza_module.Pipeline.return_value = mock_pipeline
            mock_stanza_module.DownloadMethod.REUSE_RESOURCES = "reuse"

            from kttc.helpers.hindi import HindiLanguageHelper

            helper = HindiLanguageHelper()
            # Use lowercase to avoid regex matching capitalized words
            errors = helper.check_entity_preservation("hello world", "नमस्ते दुनिया")
            assert errors == []
