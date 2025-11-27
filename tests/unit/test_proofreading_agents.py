# Copyright 2025 KTTC AI (https://github.com/kttc-ai)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for proofreading agents (self-check mode)."""

from __future__ import annotations

import json
import re
from unittest.mock import AsyncMock, MagicMock

import pytest

from kttc.agents.proofreading import GrammarAgent, SpellingAgent
from kttc.core import ErrorSeverity, TranslationTask

# ============================================================================
# Mock Classes
# ============================================================================


class MockLLMProvider:
    """Mock LLM provider for testing."""

    def __init__(self, response: str = ""):
        self.response = response

    async def complete(self, prompt: str, **kwargs) -> str:
        """Return mock response."""
        return self.response


# ============================================================================
# Unit Tests for SpellingAgent
# ============================================================================


@pytest.mark.unit
class TestSpellingAgentInit:
    """Unit tests for SpellingAgent initialization."""

    def test_init_defaults(self) -> None:
        """Test initialization with defaults."""
        agent = SpellingAgent()

        assert agent.llm_provider is None
        assert agent.language == "en"
        assert agent.use_patterns is True
        assert agent.use_school_rules is True

    def test_init_custom(self) -> None:
        """Test initialization with custom params."""
        mock_llm = MockLLMProvider()
        agent = SpellingAgent(
            llm_provider=mock_llm,
            language="ru",
            use_patterns=False,
            use_school_rules=False,
        )

        assert agent.llm_provider == mock_llm
        assert agent.language == "ru"
        assert agent.use_patterns is False

    def test_category_property(self) -> None:
        """Test category property."""
        agent = SpellingAgent()
        assert agent.category == "spelling"


@pytest.mark.unit
class TestSpellingAgentSeverity:
    """Unit tests for severity parsing."""

    def test_parse_severity_values(self) -> None:
        """Test all severity values."""
        agent = SpellingAgent()

        assert agent._parse_severity("critical") == ErrorSeverity.CRITICAL
        assert agent._parse_severity("minor") == ErrorSeverity.MINOR
        assert agent._parse_severity("major") == ErrorSeverity.MAJOR
        assert agent._parse_severity("unknown") == ErrorSeverity.MAJOR


@pytest.mark.unit
class TestSpellingAgentPatterns:
    """Unit tests for regex pattern checking."""

    def test_check_russian_ne_pattern(self) -> None:
        """Test Russian НЕ with verb pattern."""
        agent = SpellingAgent(language="ru", use_school_rules=False)
        errors = agent._check_with_patterns("Нехочу идти домой")

        assert len(errors) == 1
        assert errors[0].severity == ErrorSeverity.MAJOR
        assert "раздельно" in errors[0].description

    def test_check_russian_indefinite_pronouns(self) -> None:
        """Test Russian indefinite pronoun pattern."""
        agent = SpellingAgent(language="ru", use_school_rules=False)
        errors = agent._check_with_patterns("какой то человек пришёл")

        assert len(errors) == 1
        assert "Дефис" in errors[0].description

    def test_check_english_modal_of(self) -> None:
        """Test English modal + of pattern."""
        agent = SpellingAgent(language="en", use_school_rules=False)
        errors = agent._check_with_patterns("I should of known better")

        assert len(errors) == 1
        assert "have" in errors[0].description

    def test_check_english_your_youre(self) -> None:
        """Test English your/you're pattern."""
        agent = SpellingAgent(language="en", use_school_rules=False)
        errors = agent._check_with_patterns("Your welcome to join us")

        assert len(errors) == 1
        assert "you're" in errors[0].description

    def test_check_english_its_pattern(self) -> None:
        """Test English its/it's pattern."""
        agent = SpellingAgent(language="en", use_school_rules=False)
        errors = agent._check_with_patterns("Its a great day")

        assert len(errors) == 1
        assert "it's" in errors[0].description

    def test_check_chinese_punctuation(self) -> None:
        """Test Chinese punctuation pattern."""
        agent = SpellingAgent(language="zh", use_school_rules=False)
        errors = agent._check_with_patterns("你好,世界")

        assert len(errors) == 1
        assert errors[0].severity == ErrorSeverity.MINOR

    def test_check_no_patterns_unknown_language(self) -> None:
        """Test no patterns for unknown language."""
        agent = SpellingAgent(language="xx", use_school_rules=False)
        errors = agent._check_with_patterns("Any text here")

        assert errors == []


@pytest.mark.unit
class TestSpellingAgentSchoolRulesUnit:
    """Unit tests for school rules methods."""

    def test_load_school_rules_caching(self) -> None:
        """Test school rules are cached."""
        agent = SpellingAgent(language="nonexistent_lang_xyz")

        _ = agent._load_school_rules()  # First call to populate cache
        agent._school_rules["cached"] = True
        rules2 = agent._load_school_rules()

        assert rules2.get("cached") is True

    def test_check_school_spelling_rules(self) -> None:
        """Test school spelling rules checking."""
        agent = SpellingAgent()

        spelling_rules = {
            "rule1": {
                "name": "Spelling test",
                "severity": "major",
                "examples": {
                    "incorrect": ["seperate"],
                    "correct": ["separate"],
                },
            }
        }

        errors = agent._check_school_spelling_rules("seperate items", spelling_rules)

        assert len(errors) == 1
        assert "seperate" in errors[0].description
        assert errors[0].suggestion == "separate"

    def test_check_school_common_mistakes(self) -> None:
        """Test school common mistakes checking."""
        agent = SpellingAgent()

        common_mistakes = {
            "mistake1": {
                "examples": ["definately"],
                "correct_forms": ["definitely"],
                "description": "Common misspelling",
            }
        }

        errors = agent._check_school_common_mistakes("I definately agree", common_mistakes)

        assert len(errors) == 1
        assert "definately" in errors[0].description
        assert errors[0].suggestion == "definitely"

    def test_check_school_rules_invalid_data(self) -> None:
        """Test handling of invalid rule data."""
        agent = SpellingAgent()

        spelling_rules = {
            "invalid": "not a dict",
            "valid": {
                "name": "Valid rule",
                "examples": {
                    "incorrect": ["writting"],
                    "correct": ["writing"],
                },
            },
        }

        errors = agent._check_school_spelling_rules("writting test", spelling_rules)

        assert len(errors) == 1


@pytest.mark.unit
class TestSpellingAgentEvaluate:
    """Unit tests for evaluate method."""

    def test_evaluate_uses_translation(self) -> None:
        """Test evaluate uses translation field."""
        agent = SpellingAgent(
            language="en",
            use_patterns=False,
            use_school_rules=False,
        )

        task = TranslationTask(
            source_text="Source",
            translation="Translation",
            source_lang="en",
            target_lang="ru",
        )

        agent.evaluate(task)

        assert agent.language == "ru"

    def test_evaluate_with_valid_task(self) -> None:
        """Test evaluate with a valid task sets language correctly."""
        agent = SpellingAgent(
            language="en",
            use_patterns=False,
            use_school_rules=False,
        )

        task = TranslationTask(
            source_text="Source text",
            translation="Target text",
            source_lang="en",
            target_lang="ru",
        )

        errors = agent.evaluate(task)

        assert isinstance(errors, list)
        # Language should be set from target_lang
        assert agent.language == "ru"


@pytest.mark.unit
class TestSpellingPatternsConstant:
    """Tests for SPELLING_PATTERNS constant."""

    def test_patterns_have_required_fields(self) -> None:
        """Test all patterns have required fields."""
        from kttc.agents.proofreading.spelling_agent import SPELLING_PATTERNS

        for lang, patterns in SPELLING_PATTERNS.items():
            for pattern_def in patterns:
                assert "pattern" in pattern_def, f"Pattern missing in {lang}"
                assert "description" in pattern_def, f"Description missing in {lang}"

    def test_patterns_are_valid_regex(self) -> None:
        """Test all patterns are valid regex."""
        from kttc.agents.proofreading.spelling_agent import SPELLING_PATTERNS

        for lang, patterns in SPELLING_PATTERNS.items():
            for pattern_def in patterns:
                try:
                    re.compile(pattern_def["pattern"])
                except re.error as e:
                    pytest.fail(f"Invalid regex in {lang}: {pattern_def['pattern']} - {e}")


# ============================================================================
# Unit Tests for GrammarAgent
# ============================================================================


@pytest.mark.unit
class TestGrammarAgentInit:
    """Unit tests for GrammarAgent initialization."""

    def test_init_defaults(self) -> None:
        """Test initialization with defaults."""
        agent = GrammarAgent()

        assert agent.llm_provider is None
        assert agent.language == "en"
        assert agent.use_languagetool is True
        assert agent.use_school_rules is True
        assert agent.temperature == 0.1
        assert agent.max_tokens == 2000

    def test_init_custom(self) -> None:
        """Test initialization with custom params."""
        mock_llm = MockLLMProvider()
        agent = GrammarAgent(
            llm_provider=mock_llm,
            language="ru",
            use_languagetool=False,
            use_school_rules=False,
            temperature=0.5,
            max_tokens=1000,
        )

        assert agent.llm_provider == mock_llm
        assert agent.language == "ru"
        assert agent.use_languagetool is False
        assert agent.temperature == 0.5

    def test_category_property(self) -> None:
        """Test category property."""
        agent = GrammarAgent()
        assert agent.category == "grammar"


@pytest.mark.unit
class TestGrammarAgentSeverityParsing:
    """Unit tests for severity parsing."""

    def test_parse_severity_critical(self) -> None:
        """Test parsing critical severity."""
        agent = GrammarAgent()
        assert agent._parse_severity("critical") == ErrorSeverity.CRITICAL

    def test_parse_severity_minor(self) -> None:
        """Test parsing minor severity."""
        agent = GrammarAgent()
        assert agent._parse_severity("minor") == ErrorSeverity.MINOR

    def test_parse_severity_major_default(self) -> None:
        """Test parsing defaults to major."""
        agent = GrammarAgent()
        assert agent._parse_severity("major") == ErrorSeverity.MAJOR
        assert agent._parse_severity("unknown") == ErrorSeverity.MAJOR


@pytest.mark.unit
class TestGrammarAgentSchoolRulesUnit:
    """Unit tests for school rules methods."""

    def test_load_school_rules_no_dir(self) -> None:
        """Test loading when glossary dir doesn't exist."""
        agent = GrammarAgent(language="nonexistent_lang_xyz")
        rules = agent._load_school_rules()

        assert rules == {
            "spelling_rules": {},
            "punctuation_rules": {},
            "common_mistakes": {},
        }

    def test_load_school_rules_caching(self) -> None:
        """Test that rules are cached."""
        agent = GrammarAgent(language="nonexistent_lang_xyz")

        _ = agent._load_school_rules()  # First call to populate cache
        agent._school_rules["test"] = "value"
        rules2 = agent._load_school_rules()

        assert rules2.get("test") == "value"

    def test_check_common_mistakes(self) -> None:
        """Test common mistakes checking."""
        agent = GrammarAgent()

        common_mistakes = {
            "test_mistake": {
                "examples": ["writting"],
                "correct_forms": ["writing"],
                "description": "Common spelling mistake",
            }
        }

        errors = agent._check_common_mistakes("I love writting code", common_mistakes)

        assert len(errors) == 1
        assert errors[0].severity == ErrorSeverity.MAJOR
        assert errors[0].location[0] == 7
        assert "writting" in errors[0].description

    def test_check_spelling_rules(self) -> None:
        """Test spelling rules checking."""
        agent = GrammarAgent()

        spelling_rules = {
            "test_rule": {
                "name": "Test spelling rule",
                "severity": "critical",
                "examples": {
                    "incorrect": ["recieve"],
                    "correct": ["receive"],
                },
            }
        }

        errors = agent._check_spelling_rules("I recieve mail daily", spelling_rules)

        assert len(errors) == 1
        assert errors[0].severity == ErrorSeverity.CRITICAL
        assert "recieve" in errors[0].description


@pytest.mark.unit
class TestGrammarAgentLLMCheck:
    """Unit tests for LLM-based grammar checking."""

    @pytest.mark.asyncio
    async def test_check_with_llm_no_provider(self) -> None:
        """Test LLM check without provider."""
        agent = GrammarAgent(llm_provider=None)
        errors = await agent._check_with_llm("Test text")

        assert errors == []

    @pytest.mark.asyncio
    async def test_check_with_llm_valid_response(self) -> None:
        """Test LLM check with valid JSON response."""
        response = json.dumps(
            {
                "errors": [
                    {
                        "text": "writting",
                        "suggestion": "writing",
                        "type": "spelling",
                        "severity": "major",
                        "explanation": "Incorrect spelling",
                    }
                ]
            }
        )

        mock_llm = MockLLMProvider(response)
        agent = GrammarAgent(llm_provider=mock_llm, language="en")

        errors = await agent._check_with_llm("I love writting")

        assert len(errors) == 1
        assert errors[0].subcategory == "spelling"
        assert errors[0].severity == ErrorSeverity.MAJOR
        assert "writting" in errors[0].description

    @pytest.mark.asyncio
    async def test_check_with_llm_no_errors(self) -> None:
        """Test LLM check with no errors found."""
        response = json.dumps({"errors": []})

        mock_llm = MockLLMProvider(response)
        agent = GrammarAgent(llm_provider=mock_llm)

        errors = await agent._check_with_llm("Perfect text.")

        assert errors == []

    @pytest.mark.asyncio
    async def test_check_with_llm_invalid_json(self) -> None:
        """Test LLM check with invalid JSON."""
        mock_llm = MockLLMProvider("Not valid JSON at all")
        agent = GrammarAgent(llm_provider=mock_llm)

        errors = await agent._check_with_llm("Test text")

        assert errors == []

    @pytest.mark.asyncio
    async def test_check_with_llm_handles_llm_error(self) -> None:
        """Test LLM check handles LLMError."""
        from kttc.llm import LLMError

        mock_llm = MagicMock()
        mock_llm.complete = AsyncMock(side_effect=LLMError("API error"))

        agent = GrammarAgent(llm_provider=mock_llm)
        errors = await agent._check_with_llm("Test")

        assert errors == []

    @pytest.mark.asyncio
    async def test_check_with_llm_severity_mapping(self) -> None:
        """Test severity mapping from LLM response."""
        response = json.dumps(
            {
                "errors": [
                    {
                        "text": "error",
                        "type": "grammar",
                        "severity": "critical",
                        "explanation": "Test",
                    }
                ]
            }
        )

        mock_llm = MockLLMProvider(response)
        agent = GrammarAgent(llm_provider=mock_llm)

        errors = await agent._check_with_llm("error text")

        assert errors[0].severity == ErrorSeverity.CRITICAL

    @pytest.mark.asyncio
    async def test_check_with_llm_invalid_subcategory(self) -> None:
        """Test LLM check normalizes invalid subcategory."""
        response = json.dumps(
            {
                "errors": [
                    {
                        "text": "error",
                        "type": "invalid_type",
                        "severity": "major",
                        "explanation": "Test",
                    }
                ]
            }
        )

        mock_llm = MockLLMProvider(response)
        agent = GrammarAgent(llm_provider=mock_llm)

        errors = await agent._check_with_llm("error text")

        assert errors[0].subcategory == "grammar"


@pytest.mark.unit
class TestGrammarAgentCheckUnit:
    """Unit tests for main check method."""

    @pytest.mark.asyncio
    async def test_check_without_any_tools(self) -> None:
        """Test check without school rules or LLM."""
        agent = GrammarAgent(
            llm_provider=None,
            use_school_rules=False,
            use_languagetool=False,
        )

        errors = await agent.check("Test text")

        assert errors == []


@pytest.mark.unit
class TestGrammarAgentEvaluateUnit:
    """Unit tests for evaluate method."""

    @pytest.mark.asyncio
    async def test_evaluate_uses_translation(self) -> None:
        """Test evaluate uses translation field."""
        agent = GrammarAgent(
            llm_provider=None,
            use_school_rules=False,
            use_languagetool=False,
        )

        task = TranslationTask(
            source_text="Source text",
            translation="Translation text to check",
            source_lang="en",
            target_lang="ru",
        )

        await agent.evaluate(task)

        assert agent.language == "ru"

    @pytest.mark.asyncio
    async def test_evaluate_with_valid_task(self) -> None:
        """Test evaluate with a valid task sets language correctly."""
        agent = GrammarAgent(
            llm_provider=None,
            use_school_rules=False,
            use_languagetool=False,
            language="en",  # Initial language
        )

        task = TranslationTask(
            source_text="Source text",
            translation="Target text",
            source_lang="en",
            target_lang="ru",
        )

        errors = await agent.evaluate(task)

        assert isinstance(errors, list)
        # Language should be set from target_lang
        assert agent.language == "ru"


# ============================================================================
# Integration/Functional Tests
# ============================================================================


class TestSpellingAgent:
    """Tests for SpellingAgent."""

    def test_russian_ne_with_verbs(self) -> None:
        """Test detection of НЕ with verbs written together."""
        agent = SpellingAgent(language="ru")

        # Test incorrect forms
        text = "Нехочу идти в школу."
        errors = agent.check(text)

        # Should detect "нехочу" as incorrect
        assert len(errors) >= 1
        ne_errors = [
            e
            for e in errors
            if (e.suggestion and "не хочу" in e.suggestion.lower()) or "не" in e.description.lower()
        ]
        assert len(ne_errors) >= 1

    def test_russian_hyphen_indefinite_pronouns(self) -> None:
        """Test detection of missing hyphen in indefinite pronouns."""
        agent = SpellingAgent(language="ru")

        text = "Какой то человек пришёл."
        errors = agent.check(text)

        # Should detect "какой то" as incorrect
        assert len(errors) >= 1
        hyphen_errors = [
            e
            for e in errors
            if (e.suggestion and "какой-то" in e.suggestion.lower())
            or "дефис" in e.description.lower()
        ]
        assert len(hyphen_errors) >= 1

    def test_english_should_of_error(self) -> None:
        """Test detection of 'should of' instead of 'should have'."""
        agent = SpellingAgent(language="en")

        text = "I should of gone to the store."
        errors = agent.check(text)

        # Should detect "should of" as incorrect
        assert len(errors) >= 1
        should_errors = [
            e
            for e in errors
            if (e.suggestion and "have" in e.suggestion.lower()) or "have" in e.description.lower()
        ]
        assert len(should_errors) >= 1

    def test_no_false_positives_correct_text(self) -> None:
        """Test that correct text doesn't trigger false positives."""
        agent = SpellingAgent(language="ru")

        # Correct Russian text
        text = "Не хочу идти в школу. Какой-то человек пришёл."
        errors = agent.check(text)

        # Should not detect errors in correct text
        assert len(errors) == 0

    def test_school_rules_loading(self) -> None:
        """Test that school curriculum rules are loaded."""
        agent = SpellingAgent(language="ru", use_school_rules=True)

        # Access internal method to check rules loaded
        rules = agent._load_school_rules()

        # Should have loaded some rules
        assert rules is not None
        # Check that at least some keys exist
        assert "spelling_rules" in rules or "common_mistakes" in rules


class TestGrammarAgent:
    """Tests for GrammarAgent."""

    def test_grammar_agent_initialization(self) -> None:
        """Test GrammarAgent can be initialized."""
        agent = GrammarAgent(language="ru")
        assert agent.language == "ru"
        assert agent.category == "grammar"

    @pytest.mark.asyncio
    async def test_grammar_agent_check_text(self) -> None:
        """Test GrammarAgent can check text."""
        agent = GrammarAgent(language="ru", llm_provider=None)

        # Text with known errors
        text = "Нехочу идти. Какой то человек."
        errors = await agent.check(text)

        # Should find some errors (from school rules)
        assert isinstance(errors, list)

    def test_grammar_agent_school_rules(self) -> None:
        """Test GrammarAgent loads school rules."""
        agent = GrammarAgent(language="en", use_school_rules=True)

        rules = agent._load_school_rules()

        # Should have rules for English
        assert rules is not None

    def test_error_annotation_format(self) -> None:
        """Test that errors have correct annotation format."""
        agent = SpellingAgent(language="ru")

        text = "Нехочу идти."
        errors = agent.check(text)

        if errors:
            error = errors[0]
            # Check error has required fields (no source_text/target_text in ErrorAnnotation)
            assert hasattr(error, "category")
            assert hasattr(error, "subcategory")
            assert hasattr(error, "severity")
            assert hasattr(error, "location")
            assert hasattr(error, "description")
            assert hasattr(error, "suggestion")
            assert isinstance(error.location, tuple)
            assert len(error.location) == 2


class TestMultiLanguageSupport:
    """Tests for multi-language support in proofreading."""

    def test_chinese_de_particles(self) -> None:
        """Test Chinese 的/地/得 detection (placeholder)."""
        agent = SpellingAgent(language="zh")

        # Chinese text - basic test that agent works
        text = "这是一个测试。"
        errors = agent.check(text)

        # Should return a list (may be empty for correct text)
        assert isinstance(errors, list)

    def test_persian_nim_fasele(self) -> None:
        """Test Persian nim-fasele detection (placeholder)."""
        agent = SpellingAgent(language="fa")

        # Persian text - basic test
        text = "سلام"
        errors = agent.check(text)

        assert isinstance(errors, list)

    def test_hindi_support(self) -> None:
        """Test Hindi language support (placeholder)."""
        agent = SpellingAgent(language="hi")

        # Hindi text - basic test
        text = "नमस्ते"
        errors = agent.check(text)

        assert isinstance(errors, list)


class TestGlossaryIntegration:
    """Tests for glossary integration with proofreading."""

    def test_glossary_files_exist(self) -> None:
        """Test that school curriculum glossary files exist."""
        from pathlib import Path

        glossaries_dir = Path(__file__).parent.parent.parent / "glossaries"

        # Check Russian glossaries
        ru_school = glossaries_dir / "ru" / "school_curriculum"
        assert ru_school.exists(), "Russian school curriculum directory should exist"
        assert (
            ru_school / "orthography_fgos.json"
        ).exists(), "Russian orthography glossary should exist"

        # Check English glossaries
        en_school = glossaries_dir / "en" / "school_curriculum"
        assert en_school.exists(), "English school curriculum directory should exist"
        assert (
            en_school / "spelling_uk_gps.json"
        ).exists(), "English spelling glossary should exist"

    def test_glossary_json_valid(self) -> None:
        """Test that glossary JSON files are valid."""
        import json
        from pathlib import Path

        glossaries_dir = Path(__file__).parent.parent.parent / "glossaries"

        for lang_dir in glossaries_dir.iterdir():
            if lang_dir.is_dir() and not lang_dir.name.startswith("."):
                school_dir = lang_dir / "school_curriculum"
                if school_dir.exists():
                    for json_file in school_dir.glob("*.json"):
                        with open(json_file, encoding="utf-8") as f:
                            data = json.load(f)
                            assert "metadata" in data, f"{json_file} should have metadata"
