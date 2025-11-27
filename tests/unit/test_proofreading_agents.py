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

import pytest

from kttc.agents.proofreading import GrammarAgent, SpellingAgent


class TestSpellingAgent:
    """Tests for SpellingAgent."""

    @pytest.mark.asyncio
    async def test_russian_ne_with_verbs(self) -> None:
        """Test detection of НЕ with verbs written together."""
        agent = SpellingAgent(language="ru")

        # Test incorrect forms
        text = "Нехочу идти в школу."
        errors = await agent.check(text)

        # Should detect "нехочу" as incorrect
        assert len(errors) >= 1
        ne_errors = [
            e
            for e in errors
            if (e.suggestion and "не хочу" in e.suggestion.lower()) or "не" in e.description.lower()
        ]
        assert len(ne_errors) >= 1

    @pytest.mark.asyncio
    async def test_russian_hyphen_indefinite_pronouns(self) -> None:
        """Test detection of missing hyphen in indefinite pronouns."""
        agent = SpellingAgent(language="ru")

        text = "Какой то человек пришёл."
        errors = await agent.check(text)

        # Should detect "какой то" as incorrect
        assert len(errors) >= 1
        hyphen_errors = [
            e
            for e in errors
            if (e.suggestion and "какой-то" in e.suggestion.lower())
            or "дефис" in e.description.lower()
        ]
        assert len(hyphen_errors) >= 1

    @pytest.mark.asyncio
    async def test_english_should_of_error(self) -> None:
        """Test detection of 'should of' instead of 'should have'."""
        agent = SpellingAgent(language="en")

        text = "I should of gone to the store."
        errors = await agent.check(text)

        # Should detect "should of" as incorrect
        assert len(errors) >= 1
        should_errors = [
            e
            for e in errors
            if (e.suggestion and "have" in e.suggestion.lower()) or "have" in e.description.lower()
        ]
        assert len(should_errors) >= 1

    @pytest.mark.asyncio
    async def test_no_false_positives_correct_text(self) -> None:
        """Test that correct text doesn't trigger false positives."""
        agent = SpellingAgent(language="ru")

        # Correct Russian text
        text = "Не хочу идти в школу. Какой-то человек пришёл."
        errors = await agent.check(text)

        # Should not detect errors in correct text
        assert len(errors) == 0

    @pytest.mark.asyncio
    async def test_school_rules_loading(self) -> None:
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

    @pytest.mark.asyncio
    async def test_grammar_agent_initialization(self) -> None:
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

    @pytest.mark.asyncio
    async def test_grammar_agent_school_rules(self) -> None:
        """Test GrammarAgent loads school rules."""
        agent = GrammarAgent(language="en", use_school_rules=True)

        rules = agent._load_school_rules()

        # Should have rules for English
        assert rules is not None

    @pytest.mark.asyncio
    async def test_error_annotation_format(self) -> None:
        """Test that errors have correct annotation format."""
        agent = SpellingAgent(language="ru")

        text = "Нехочу идти."
        errors = await agent.check(text)

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

    @pytest.mark.asyncio
    async def test_chinese_de_particles(self) -> None:
        """Test Chinese 的/地/得 detection (placeholder)."""
        agent = SpellingAgent(language="zh")

        # Chinese text - basic test that agent works
        text = "这是一个测试。"
        errors = await agent.check(text)

        # Should return a list (may be empty for correct text)
        assert isinstance(errors, list)

    @pytest.mark.asyncio
    async def test_persian_nim_fasele(self) -> None:
        """Test Persian nim-fasele detection (placeholder)."""
        agent = SpellingAgent(language="fa")

        # Persian text - basic test
        text = "سلام"
        errors = await agent.check(text)

        assert isinstance(errors, list)

    @pytest.mark.asyncio
    async def test_hindi_support(self) -> None:
        """Test Hindi language support (placeholder)."""
        agent = SpellingAgent(language="hi")

        # Hindi text - basic test
        text = "नमस्ते"
        errors = await agent.check(text)

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
