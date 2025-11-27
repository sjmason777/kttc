"""Unit tests for PersianLanguageHelper.

Tests Persian language helper with mocked DadmaTools.
Focus: Fast, isolated tests that improve coverage.
"""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from kttc.core.models import ErrorAnnotation, ErrorSeverity

# ============================================================================
# Mock DadmaTools spaCy-like objects
# ============================================================================


class MockToken:
    """Mock spaCy Token for DadmaTools."""

    def __init__(
        self,
        text: str,
        idx: int,
        pos_: str = "NOUN",
        dep_: str = "nsubj",
        i: int = 0,
        morph: Any = None,
        is_punct: bool = False,
    ):
        self.text = text
        self.idx = idx
        self.pos_ = pos_
        self.dep_ = dep_
        self.i = i
        self.morph = morph
        self.is_punct = is_punct
        self.head = self  # Default to self
        self.children = []


class MockMorph:
    """Mock morphology features."""

    def __init__(self, features: dict[str, str]):
        self._features = features

    def to_dict(self) -> dict[str, str]:
        return self._features


class MockEntity:
    """Mock spaCy Entity."""

    def __init__(self, text: str, label_: str, start_char: int, end_char: int):
        self.text = text
        self.label_ = label_
        self.start_char = start_char
        self.end_char = end_char


class MockDoc:
    """Mock spaCy Doc from DadmaTools."""

    def __init__(self, tokens: list[MockToken], ents: list[MockEntity] | None = None):
        self._tokens = tokens
        self.ents = ents or []
        self._ = MockDocExtensions()

    def __iter__(self):
        return iter(self._tokens)

    def __len__(self):
        return len(self._tokens)

    def __getitem__(self, idx: int) -> MockToken:
        return self._tokens[idx]

    @property
    def sents(self):
        """Mock sentence iterator."""
        return [self]


class MockDocExtensions:
    """Mock doc._ extensions from DadmaTools."""

    def __init__(self):
        self.spell_corrected: str | None = None
        self.sentiment: str | None = None
        self.formal_text: str | None = None


class MockPipeline:
    """Mock DadmaTools Pipeline."""

    def __init__(self, config: str = ""):
        self.config = config
        self._doc: MockDoc | None = None

    def __call__(self, text: str) -> MockDoc:
        if self._doc:
            return self._doc
        # Default: tokenize by whitespace
        tokens = []
        idx = 0
        for i, word in enumerate(text.split()):
            start = text.find(word, idx)
            tokens.append(MockToken(word, start, i=i))
            idx = start + len(word)
        return MockDoc(tokens)


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def mock_dadmatools_available():
    """Mock DadmaTools as available."""
    with patch.dict("sys.modules", {"dadmatools.pipeline.language": MagicMock()}):
        with patch("kttc.helpers.persian.DADMATOOLS_AVAILABLE", True):
            yield


@pytest.fixture
def mock_dadmatools_unavailable():
    """Mock DadmaTools as unavailable."""
    with patch("kttc.helpers.persian.DADMATOOLS_AVAILABLE", False):
        yield


@pytest.fixture
def mock_pipeline():
    """Create a mock DadmaTools pipeline."""
    return MockPipeline()


# ============================================================================
# Tests: Initialization
# ============================================================================


@pytest.mark.unit
class TestPersianHelperInitialization:
    """Test PersianLanguageHelper initialization."""

    def test_init_without_dadmatools(self, mock_dadmatools_unavailable: None) -> None:
        """Test initialization when DadmaTools is not available."""
        from kttc.helpers.persian import PersianLanguageHelper

        helper = PersianLanguageHelper()

        assert helper.language_code == "fa"
        assert helper.is_available() is False
        assert helper._nlp is None
        assert helper._initialized is False

    def test_init_with_dadmatools_success(self) -> None:
        """Test successful initialization with DadmaTools."""
        mock_pipeline = MockPipeline()

        with patch("kttc.helpers.persian.DADMATOOLS_AVAILABLE", True):
            with patch(
                "kttc.helpers.persian.dadma_language.Pipeline",
                return_value=mock_pipeline,
            ):
                from kttc.helpers.persian import PersianLanguageHelper

                helper = PersianLanguageHelper()

                assert helper.is_available() is True
                assert helper._initialized is True

    def test_init_with_dadmatools_failure(self) -> None:
        """Test initialization when DadmaTools fails to initialize."""
        with patch("kttc.helpers.persian.DADMATOOLS_AVAILABLE", True):
            with patch(
                "kttc.helpers.persian.dadma_language.Pipeline",
                side_effect=Exception("Pipeline init failed"),
            ):
                from kttc.helpers.persian import PersianLanguageHelper

                helper = PersianLanguageHelper()

                assert helper.is_available() is False
                assert helper._initialized is False


# ============================================================================
# Tests: Basic Methods
# ============================================================================


@pytest.mark.unit
class TestPersianHelperBasicMethods:
    """Test basic helper methods."""

    def test_language_code(self, mock_dadmatools_unavailable: None) -> None:
        """Test language_code property returns 'fa'."""
        from kttc.helpers.persian import PersianLanguageHelper

        helper = PersianLanguageHelper()
        assert helper.language_code == "fa"

    def test_verify_word_exists_without_dadmatools(self, mock_dadmatools_unavailable: None) -> None:
        """Test verify_word_exists falls back to simple search."""
        from kttc.helpers.persian import PersianLanguageHelper

        helper = PersianLanguageHelper()

        assert helper.verify_word_exists("سلام", "سلام دنیا") is True
        assert helper.verify_word_exists("خداحافظ", "سلام دنیا") is False

    def test_verify_word_exists_with_dadmatools(self) -> None:
        """Test verify_word_exists with DadmaTools tokenization."""
        mock_pipeline = MockPipeline()
        # Set up tokens
        tokens = [
            MockToken("سلام", 0, i=0),
            MockToken("دنیا", 5, i=1),
        ]
        mock_pipeline._doc = MockDoc(tokens)

        with patch("kttc.helpers.persian.DADMATOOLS_AVAILABLE", True):
            with patch(
                "kttc.helpers.persian.dadma_language.Pipeline",
                return_value=mock_pipeline,
            ):
                from kttc.helpers.persian import PersianLanguageHelper

                helper = PersianLanguageHelper()

                assert helper.verify_word_exists("سلام", "سلام دنیا") is True
                assert helper.verify_word_exists("خداحافظ", "سلام دنیا") is False

    def test_verify_error_position_valid(self, mock_dadmatools_unavailable: None) -> None:
        """Test verify_error_position with valid position."""
        from kttc.helpers.persian import PersianLanguageHelper

        helper = PersianLanguageHelper()
        text = "سلام دنیا"

        error = ErrorAnnotation(
            category="fluency",
            subcategory="test",
            severity=ErrorSeverity.MINOR,
            location=(0, 4),
            description="test",
        )

        assert helper.verify_error_position(error, text) is True

    def test_verify_error_position_invalid_bounds(self, mock_dadmatools_unavailable: None) -> None:
        """Test verify_error_position with invalid bounds."""
        from kttc.helpers.persian import PersianLanguageHelper

        helper = PersianLanguageHelper()
        text = "سلام"

        # Start < 0
        error1 = ErrorAnnotation(
            category="fluency",
            subcategory="test",
            severity=ErrorSeverity.MINOR,
            location=(-1, 4),
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
            location=(5, 5),
            description="test",
        )
        assert helper.verify_error_position(error3, text) is False

    def test_verify_error_position_empty_substring(self, mock_dadmatools_unavailable: None) -> None:
        """Test verify_error_position with whitespace-only substring."""
        from kttc.helpers.persian import PersianLanguageHelper

        helper = PersianLanguageHelper()
        text = "سلام   دنیا"  # Multiple spaces

        error = ErrorAnnotation(
            category="fluency",
            subcategory="test",
            severity=ErrorSeverity.MINOR,
            location=(4, 7),  # Points to spaces
            description="test",
        )

        assert helper.verify_error_position(error, text) is False


# ============================================================================
# Tests: Tokenization
# ============================================================================


@pytest.mark.unit
class TestPersianHelperTokenization:
    """Test tokenization methods."""

    def test_tokenize_without_dadmatools(self, mock_dadmatools_unavailable: None) -> None:
        """Test tokenize falls back to whitespace split."""
        from kttc.helpers.persian import PersianLanguageHelper

        helper = PersianLanguageHelper()
        tokens = helper.tokenize("سلام دنیا")

        assert len(tokens) == 2
        assert tokens[0][0] == "سلام"
        assert tokens[1][0] == "دنیا"

    def test_tokenize_with_dadmatools(self) -> None:
        """Test tokenize with DadmaTools."""
        mock_pipeline = MockPipeline()
        tokens = [
            MockToken("سلام", 0, i=0),
            MockToken("دنیا", 5, i=1),
        ]
        mock_pipeline._doc = MockDoc(tokens)

        with patch("kttc.helpers.persian.DADMATOOLS_AVAILABLE", True):
            with patch(
                "kttc.helpers.persian.dadma_language.Pipeline",
                return_value=mock_pipeline,
            ):
                from kttc.helpers.persian import PersianLanguageHelper

                helper = PersianLanguageHelper()
                result = helper.tokenize("سلام دنیا")

                assert len(result) == 2
                assert result[0] == ("سلام", 0, 4)
                assert result[1] == ("دنیا", 5, 9)

    def test_tokenize_exception_handling(self) -> None:
        """Test tokenize handles exceptions gracefully."""
        mock_pipeline = MagicMock()
        mock_pipeline.side_effect = Exception("Tokenization error")

        with patch("kttc.helpers.persian.DADMATOOLS_AVAILABLE", True):
            with patch(
                "kttc.helpers.persian.dadma_language.Pipeline",
                return_value=mock_pipeline,
            ):
                from kttc.helpers.persian import PersianLanguageHelper

                helper = PersianLanguageHelper()
                result = helper.tokenize("سلام دنیا")

                assert result == []


# ============================================================================
# Tests: Morphology Analysis
# ============================================================================


@pytest.mark.unit
class TestPersianHelperMorphology:
    """Test morphology analysis methods."""

    def test_analyze_morphology_without_dadmatools(self, mock_dadmatools_unavailable: None) -> None:
        """Test analyze_morphology returns empty without DadmaTools."""
        from kttc.helpers.persian import PersianLanguageHelper

        helper = PersianLanguageHelper()
        result = helper.analyze_morphology("سلام دنیا")

        assert result == []

    def test_analyze_morphology_with_dadmatools(self) -> None:
        """Test analyze_morphology with DadmaTools."""
        mock_pipeline = MockPipeline()
        tokens = [
            MockToken("سلام", 0, pos_="NOUN", i=0),
            MockToken("دنیا", 5, pos_="NOUN", i=1),
        ]
        mock_pipeline._doc = MockDoc(tokens)

        with patch("kttc.helpers.persian.DADMATOOLS_AVAILABLE", True):
            with patch(
                "kttc.helpers.persian.dadma_language.Pipeline",
                return_value=mock_pipeline,
            ):
                from kttc.helpers.persian import PersianLanguageHelper

                helper = PersianLanguageHelper()
                result = helper.analyze_morphology("سلام دنیا")

                assert len(result) == 2
                assert result[0].word == "سلام"
                assert result[0].pos == "NOUN"
                assert result[1].word == "دنیا"

    def test_analyze_morphology_exception(self) -> None:
        """Test analyze_morphology handles exceptions."""
        mock_pipeline = MagicMock()
        mock_pipeline.side_effect = Exception("Analysis failed")

        with patch("kttc.helpers.persian.DADMATOOLS_AVAILABLE", True):
            with patch(
                "kttc.helpers.persian.dadma_language.Pipeline",
                return_value=mock_pipeline,
            ):
                from kttc.helpers.persian import PersianLanguageHelper

                helper = PersianLanguageHelper()
                result = helper.analyze_morphology("سلام دنیا")

                assert result == []


# ============================================================================
# Tests: Grammar Checking
# ============================================================================


@pytest.mark.unit
class TestPersianHelperGrammar:
    """Test grammar checking methods."""

    def test_check_grammar_without_dadmatools(self, mock_dadmatools_unavailable: None) -> None:
        """Test check_grammar returns empty without DadmaTools."""
        from kttc.helpers.persian import PersianLanguageHelper

        helper = PersianLanguageHelper()
        result = helper.check_grammar("سلام دنیا")

        assert result == []

    def test_check_grammar_subject_verb_agreement_person(self) -> None:
        """Test grammar check detects person disagreement."""
        mock_pipeline = MockPipeline()

        # Create subject with Person=1
        subject = MockToken(
            "من", 0, pos_="PRON", dep_="nsubj", i=0, morph=MockMorph({"Person": "1"})
        )

        # Create verb with Person=3 (mismatch)
        verb = MockToken(
            "می‌رود", 3, pos_="VERB", dep_="ROOT", i=1, morph=MockMorph({"Person": "3"})
        )

        # Link subject to verb
        subject.head = verb

        mock_pipeline._doc = MockDoc([subject, verb])

        with patch("kttc.helpers.persian.DADMATOOLS_AVAILABLE", True):
            with patch(
                "kttc.helpers.persian.dadma_language.Pipeline",
                return_value=mock_pipeline,
            ):
                from kttc.helpers.persian import PersianLanguageHelper

                helper = PersianLanguageHelper()
                errors = helper.check_grammar("من می‌رود")

                # Should find person disagreement
                assert len(errors) >= 1
                assert any("person" in e.description.lower() for e in errors)

    def test_check_grammar_subject_verb_agreement_number(self) -> None:
        """Test grammar check detects number disagreement."""
        mock_pipeline = MockPipeline()

        # Create subject with Number=Plur
        subject = MockToken(
            "آنها", 0, pos_="PRON", dep_="nsubj", i=0, morph=MockMorph({"Number": "Plur"})
        )

        # Create verb with Number=Sing (mismatch)
        verb = MockToken(
            "می‌رود", 5, pos_="VERB", dep_="ROOT", i=1, morph=MockMorph({"Number": "Sing"})
        )

        subject.head = verb

        mock_pipeline._doc = MockDoc([subject, verb])

        with patch("kttc.helpers.persian.DADMATOOLS_AVAILABLE", True):
            with patch(
                "kttc.helpers.persian.dadma_language.Pipeline",
                return_value=mock_pipeline,
            ):
                from kttc.helpers.persian import PersianLanguageHelper

                helper = PersianLanguageHelper()
                errors = helper.check_grammar("آنها می‌رود")

                assert len(errors) >= 1
                assert any("number" in e.description.lower() for e in errors)

    def test_check_grammar_word_order_violation(self) -> None:
        """Test grammar check detects word order violations."""
        mock_pipeline = MockPipeline()

        # Verb before subject (unusual in Persian)
        verb = MockToken("می‌رود", 0, pos_="VERB", dep_="ROOT", i=0)
        subject = MockToken("او", 8, pos_="PRON", dep_="nsubj", i=5)

        subject.head = verb

        mock_pipeline._doc = MockDoc([verb, subject])

        with patch("kttc.helpers.persian.DADMATOOLS_AVAILABLE", True):
            with patch(
                "kttc.helpers.persian.dadma_language.Pipeline",
                return_value=mock_pipeline,
            ):
                from kttc.helpers.persian import PersianLanguageHelper

                helper = PersianLanguageHelper()
                errors = helper.check_grammar("می‌رود او")

                assert len(errors) >= 1
                assert any("word order" in e.description.lower() for e in errors)

    def test_check_grammar_preposition_missing_object(self) -> None:
        """Test grammar check detects preposition without object."""
        mock_pipeline = MockPipeline()

        # Preposition "به" followed by verb (unusual)
        prep = MockToken("به", 0, pos_="ADP", dep_="case", i=0)
        prep.children = []  # No object

        verb = MockToken("می‌رود", 3, pos_="VERB", dep_="ROOT", i=1)

        mock_pipeline._doc = MockDoc([prep, verb])

        with patch("kttc.helpers.persian.DADMATOOLS_AVAILABLE", True):
            with patch(
                "kttc.helpers.persian.dadma_language.Pipeline",
                return_value=mock_pipeline,
            ):
                from kttc.helpers.persian import PersianLanguageHelper

                helper = PersianLanguageHelper()
                errors = helper.check_grammar("به می‌رود")

                assert len(errors) >= 1
                assert any("preposition" in e.description.lower() for e in errors)

    def test_check_grammar_non_verb_root(self) -> None:
        """Test grammar check detects non-verb as ROOT."""
        mock_pipeline = MockPipeline()

        # Noun as ROOT (unusual sentence structure)
        noun = MockToken("کتاب", 0, pos_="NOUN", dep_="ROOT", i=0)

        mock_pipeline._doc = MockDoc([noun])

        with patch("kttc.helpers.persian.DADMATOOLS_AVAILABLE", True):
            with patch(
                "kttc.helpers.persian.dadma_language.Pipeline",
                return_value=mock_pipeline,
            ):
                from kttc.helpers.persian import PersianLanguageHelper

                helper = PersianLanguageHelper()
                errors = helper.check_grammar("کتاب")

                assert len(errors) >= 1
                assert any("structure" in e.description.lower() for e in errors)

    def test_check_grammar_exception(self) -> None:
        """Test check_grammar handles exceptions gracefully."""
        mock_pipeline = MagicMock()
        mock_pipeline.side_effect = Exception("Grammar check failed")

        with patch("kttc.helpers.persian.DADMATOOLS_AVAILABLE", True):
            with patch(
                "kttc.helpers.persian.dadma_language.Pipeline",
                return_value=mock_pipeline,
            ):
                from kttc.helpers.persian import PersianLanguageHelper

                helper = PersianLanguageHelper()
                errors = helper.check_grammar("سلام دنیا")

                assert errors == []


# ============================================================================
# Tests: Spelling
# ============================================================================


@pytest.mark.unit
class TestPersianHelperSpelling:
    """Test spelling checking methods."""

    def test_check_spelling_without_dadmatools(self, mock_dadmatools_unavailable: None) -> None:
        """Test check_spelling returns empty without DadmaTools."""
        from kttc.helpers.persian import PersianLanguageHelper

        helper = PersianLanguageHelper()
        result = helper.check_spelling("سلام دنیا")

        assert result == []

    def test_check_spelling_no_errors(self) -> None:
        """Test check_spelling with no spelling errors."""
        mock_pipeline = MockPipeline()
        text = "سلام دنیا"

        tokens = [
            MockToken("سلام", 0, i=0),
            MockToken("دنیا", 5, i=1),
        ]
        doc = MockDoc(tokens)
        doc._.spell_corrected = text  # Same as original = no errors

        mock_pipeline._doc = doc

        with patch("kttc.helpers.persian.DADMATOOLS_AVAILABLE", True):
            with patch(
                "kttc.helpers.persian.dadma_language.Pipeline",
                return_value=mock_pipeline,
            ):
                from kttc.helpers.persian import PersianLanguageHelper

                helper = PersianLanguageHelper()
                errors = helper.check_spelling(text)

                assert errors == []

    def test_check_spelling_with_replace_error(self) -> None:
        """Test check_spelling detects replacement errors."""
        original_text = "سالم دنیا"  # سالم misspelled (should be سلام)
        corrected_text = "سلام دنیا"

        # Original tokens
        original_tokens = [
            MockToken("سالم", 0, i=0),
            MockToken("دنیا", 5, i=1),
        ]

        # Corrected tokens
        corrected_tokens = [
            MockToken("سلام", 0, i=0),
            MockToken("دنیا", 5, i=1),
        ]

        # Create docs for both original and corrected
        original_doc = MockDoc(original_tokens)
        original_doc._.spell_corrected = corrected_text

        corrected_doc = MockDoc(corrected_tokens)
        corrected_doc._.spell_corrected = corrected_text

        call_count = [0]

        def mock_call(text: str) -> MockDoc:
            call_count[0] += 1
            if call_count[0] == 1:
                return original_doc  # First call for spelling check
            elif "سلام" in text:
                return corrected_doc  # Tokenize corrected text
            return original_doc  # Tokenize original text

        mock_pipeline = MagicMock()
        mock_pipeline.side_effect = mock_call

        with patch("kttc.helpers.persian.DADMATOOLS_AVAILABLE", True):
            with patch(
                "kttc.helpers.persian.dadma_language.Pipeline",
                return_value=mock_pipeline,
            ):
                from kttc.helpers.persian import PersianLanguageHelper

                helper = PersianLanguageHelper()
                errors = helper.check_spelling(original_text)

                assert len(errors) >= 1
                assert any("سالم" in e.description for e in errors)

    def test_check_spelling_no_spell_corrected_attr(self) -> None:
        """Test check_spelling when spell_corrected is not available."""
        mock_pipeline = MockPipeline()
        tokens = [MockToken("سلام", 0, i=0)]
        doc = MockDoc(tokens)
        # Remove spell_corrected attribute
        delattr(doc._, "spell_corrected")

        mock_pipeline._doc = doc

        with patch("kttc.helpers.persian.DADMATOOLS_AVAILABLE", True):
            with patch(
                "kttc.helpers.persian.dadma_language.Pipeline",
                return_value=mock_pipeline,
            ):
                from kttc.helpers.persian import PersianLanguageHelper

                helper = PersianLanguageHelper()
                errors = helper.check_spelling("سلام")

                assert errors == []

    def test_check_spelling_exception(self) -> None:
        """Test check_spelling handles exceptions."""
        mock_pipeline = MagicMock()
        mock_pipeline.side_effect = Exception("Spell check failed")

        with patch("kttc.helpers.persian.DADMATOOLS_AVAILABLE", True):
            with patch(
                "kttc.helpers.persian.dadma_language.Pipeline",
                return_value=mock_pipeline,
            ):
                from kttc.helpers.persian import PersianLanguageHelper

                helper = PersianLanguageHelper()
                errors = helper.check_spelling("سلام")

                assert errors == []


# ============================================================================
# Tests: Sentiment and Formal Conversion
# ============================================================================


@pytest.mark.unit
class TestPersianHelperSentimentAndFormal:
    """Test sentiment analysis and formal conversion."""

    def test_check_sentiment_without_dadmatools(self, mock_dadmatools_unavailable: None) -> None:
        """Test check_sentiment returns None without DadmaTools."""
        from kttc.helpers.persian import PersianLanguageHelper

        helper = PersianLanguageHelper()
        result = helper.check_sentiment("سلام")

        assert result is None

    def test_check_sentiment_positive(self) -> None:
        """Test check_sentiment returns sentiment."""
        mock_pipeline = MockPipeline()
        doc = MockDoc([MockToken("عالی", 0)])
        doc._.sentiment = "positive"

        mock_pipeline._doc = doc

        with patch("kttc.helpers.persian.DADMATOOLS_AVAILABLE", True):
            with patch(
                "kttc.helpers.persian.dadma_language.Pipeline",
                return_value=mock_pipeline,
            ):
                from kttc.helpers.persian import PersianLanguageHelper

                helper = PersianLanguageHelper()
                result = helper.check_sentiment("عالی")

                assert result == "positive"

    def test_check_sentiment_exception(self) -> None:
        """Test check_sentiment handles exceptions."""
        mock_pipeline = MagicMock()
        mock_pipeline.side_effect = Exception("Sentiment failed")

        with patch("kttc.helpers.persian.DADMATOOLS_AVAILABLE", True):
            with patch(
                "kttc.helpers.persian.dadma_language.Pipeline",
                return_value=mock_pipeline,
            ):
                from kttc.helpers.persian import PersianLanguageHelper

                helper = PersianLanguageHelper()
                result = helper.check_sentiment("سلام")

                assert result is None

    def test_convert_to_formal_without_dadmatools(self, mock_dadmatools_unavailable: None) -> None:
        """Test convert_to_formal returns None without DadmaTools."""
        from kttc.helpers.persian import PersianLanguageHelper

        helper = PersianLanguageHelper()
        result = helper.convert_to_formal("چطوری")

        assert result is None

    def test_convert_to_formal_success(self) -> None:
        """Test convert_to_formal returns formal text."""
        mock_pipeline = MockPipeline()
        doc = MockDoc([MockToken("چطوری", 0)])
        doc._.formal_text = "حالتان چطور است"

        mock_pipeline._doc = doc

        with patch("kttc.helpers.persian.DADMATOOLS_AVAILABLE", True):
            with patch(
                "kttc.helpers.persian.dadma_language.Pipeline",
                return_value=mock_pipeline,
            ):
                from kttc.helpers.persian import PersianLanguageHelper

                helper = PersianLanguageHelper()
                result = helper.convert_to_formal("چطوری")

                assert result == "حالتان چطور است"

    def test_convert_to_formal_exception(self) -> None:
        """Test convert_to_formal handles exceptions."""
        mock_pipeline = MagicMock()
        mock_pipeline.side_effect = Exception("Conversion failed")

        with patch("kttc.helpers.persian.DADMATOOLS_AVAILABLE", True):
            with patch(
                "kttc.helpers.persian.dadma_language.Pipeline",
                return_value=mock_pipeline,
            ):
                from kttc.helpers.persian import PersianLanguageHelper

                helper = PersianLanguageHelper()
                result = helper.convert_to_formal("چطوری")

                assert result is None


# ============================================================================
# Tests: Enrichment Data
# ============================================================================


@pytest.mark.unit
class TestPersianHelperEnrichment:
    """Test enrichment data generation."""

    def test_get_enrichment_data_without_dadmatools(
        self, mock_dadmatools_unavailable: None
    ) -> None:
        """Test get_enrichment_data returns minimal data without DadmaTools."""
        from kttc.helpers.persian import PersianLanguageHelper

        helper = PersianLanguageHelper()
        result = helper.get_enrichment_data("سلام")

        assert result == {"has_morphology": False}

    def test_get_enrichment_data_with_dadmatools(self) -> None:
        """Test get_enrichment_data returns full data."""
        mock_pipeline = MockPipeline()

        tokens = [
            MockToken("سلام", 0, pos_="NOUN", i=0),
            MockToken("دنیا", 5, pos_="NOUN", i=1),
        ]
        entities = [
            MockEntity("تهران", "LOC", 0, 5),
        ]
        doc = MockDoc(tokens, entities)
        doc._.sentiment = "positive"
        doc._.formal_text = "سلام دنیا"

        mock_pipeline._doc = doc

        with patch("kttc.helpers.persian.DADMATOOLS_AVAILABLE", True):
            with patch(
                "kttc.helpers.persian.dadma_language.Pipeline",
                return_value=mock_pipeline,
            ):
                from kttc.helpers.persian import PersianLanguageHelper

                helper = PersianLanguageHelper()
                result = helper.get_enrichment_data("سلام دنیا")

                assert result["has_morphology"] is True
                assert result["word_count"] == 2
                assert result["pos_distribution"]["NOUN"] == 2
                assert len(result["entities"]) == 1
                assert result["sentiment"] == "positive"
                assert result["has_dadmatools"] is True

    def test_get_enrichment_data_exception(self) -> None:
        """Test get_enrichment_data handles exceptions."""
        mock_pipeline = MagicMock()
        mock_pipeline.side_effect = Exception("Enrichment failed")

        with patch("kttc.helpers.persian.DADMATOOLS_AVAILABLE", True):
            with patch(
                "kttc.helpers.persian.dadma_language.Pipeline",
                return_value=mock_pipeline,
            ):
                from kttc.helpers.persian import PersianLanguageHelper

                helper = PersianLanguageHelper()
                result = helper.get_enrichment_data("سلام")

                # Should return basic structure even on error
                assert "has_morphology" in result


# ============================================================================
# Tests: Entity Extraction
# ============================================================================


@pytest.mark.unit
class TestPersianHelperEntities:
    """Test entity extraction and preservation."""

    def test_extract_entities_without_dadmatools(self, mock_dadmatools_unavailable: None) -> None:
        """Test extract_entities returns empty without DadmaTools."""
        from kttc.helpers.persian import PersianLanguageHelper

        helper = PersianLanguageHelper()
        result = helper.extract_entities("تهران پایتخت ایران است")

        assert result == []

    def test_extract_entities_with_dadmatools(self) -> None:
        """Test extract_entities returns entities."""
        mock_pipeline = MockPipeline()

        entities = [
            MockEntity("تهران", "LOC", 0, 5),
            MockEntity("ایران", "LOC", 14, 19),
        ]
        doc = MockDoc([MockToken("تهران", 0)], entities)

        mock_pipeline._doc = doc

        with patch("kttc.helpers.persian.DADMATOOLS_AVAILABLE", True):
            with patch(
                "kttc.helpers.persian.dadma_language.Pipeline",
                return_value=mock_pipeline,
            ):
                from kttc.helpers.persian import PersianLanguageHelper

                helper = PersianLanguageHelper()
                result = helper.extract_entities("تهران پایتخت ایران است")

                assert len(result) == 2
                assert result[0]["text"] == "تهران"
                assert result[0]["type"] == "LOC"

    def test_extract_entities_exception(self) -> None:
        """Test extract_entities handles exceptions."""
        mock_pipeline = MagicMock()
        mock_pipeline.side_effect = Exception("NER failed")

        with patch("kttc.helpers.persian.DADMATOOLS_AVAILABLE", True):
            with patch(
                "kttc.helpers.persian.dadma_language.Pipeline",
                return_value=mock_pipeline,
            ):
                from kttc.helpers.persian import PersianLanguageHelper

                helper = PersianLanguageHelper()
                result = helper.extract_entities("تهران")

                assert result == []

    def test_check_entity_preservation_without_dadmatools(
        self, mock_dadmatools_unavailable: None
    ) -> None:
        """Test check_entity_preservation returns empty without DadmaTools."""
        from kttc.helpers.persian import PersianLanguageHelper

        helper = PersianLanguageHelper()
        result = helper.check_entity_preservation("John went to Paris", "جان به پاریس رفت")

        assert result == []

    def test_check_entity_preservation_missing_entities(self) -> None:
        """Test check_entity_preservation detects missing entities."""
        mock_pipeline = MockPipeline()

        # Translation has no entities but source has names
        doc = MockDoc([MockToken("او", 0)], [])  # No entities

        mock_pipeline._doc = doc

        with patch("kttc.helpers.persian.DADMATOOLS_AVAILABLE", True):
            with patch(
                "kttc.helpers.persian.dadma_language.Pipeline",
                return_value=mock_pipeline,
            ):
                from kttc.helpers.persian import PersianLanguageHelper

                helper = PersianLanguageHelper()
                errors = helper.check_entity_preservation(
                    "John Smith went to Paris",  # Source with names
                    "او رفت",  # Translation without entities
                )

                assert len(errors) >= 1
                assert errors[0].subcategory == "entity_omission"

    def test_check_entity_preservation_entities_present(self) -> None:
        """Test check_entity_preservation passes when entities preserved."""
        mock_pipeline = MockPipeline()

        # Translation has entities
        entities = [MockEntity("جان", "PER", 0, 3)]
        doc = MockDoc([MockToken("جان", 0)], entities)

        mock_pipeline._doc = doc

        with patch("kttc.helpers.persian.DADMATOOLS_AVAILABLE", True):
            with patch(
                "kttc.helpers.persian.dadma_language.Pipeline",
                return_value=mock_pipeline,
            ):
                from kttc.helpers.persian import PersianLanguageHelper

                helper = PersianLanguageHelper()
                errors = helper.check_entity_preservation(
                    "John went home",
                    "جان به خانه رفت",
                )

                # Should not flag error since translation has entity
                assert len(errors) == 0

    def test_check_entity_preservation_exception(self) -> None:
        """Test check_entity_preservation handles exceptions in entity extraction."""
        _ = MockPipeline()  # Initialize mock pipeline

        # First call works (to initialize), then fails
        call_count = [0]

        def mock_call(text: str) -> MockDoc:
            call_count[0] += 1
            if call_count[0] == 1:
                # Initialization call
                return MockDoc([MockToken("test", 0)])
            # Subsequent calls raise exception
            raise Exception("Entity check failed")

        mock_pipeline_obj = MagicMock()
        mock_pipeline_obj.side_effect = mock_call

        with patch("kttc.helpers.persian.DADMATOOLS_AVAILABLE", True):
            with patch(
                "kttc.helpers.persian.dadma_language.Pipeline",
                return_value=mock_pipeline_obj,
            ):
                from kttc.helpers.persian import PersianLanguageHelper

                helper = PersianLanguageHelper()
                # Use source without capitalized words to avoid fallback regex match
                errors = helper.check_entity_preservation("hello world", "سلام دنیا")

                assert errors == []
