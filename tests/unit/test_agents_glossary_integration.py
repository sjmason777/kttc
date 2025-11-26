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

"""Unit tests for agent glossary integration."""

import pytest

from kttc.agents.fluency_chinese import ChineseFluencyAgent
from kttc.agents.fluency_hindi import HindiFluencyAgent
from kttc.agents.fluency_persian import PersianFluencyAgent
from kttc.agents.fluency_russian import RussianFluencyAgent
from kttc.agents.terminology import TerminologyAgent
from kttc.core import TranslationTask
from kttc.terminology import (
    ChineseMeasureWordValidator,
    HindiPostpositionValidator,
    PersianEzafeValidator,
    RussianCaseAspectValidator,
    TermValidator,
)


# Mock LLM Provider for testing
class MockLLMProvider:
    """Mock LLM provider for testing."""

    async def complete(self, prompt, temperature=0.1, max_tokens=2000):
        """Return empty response (no errors)."""
        return "No errors found."


class TestRussianFluencyAgentGlossaryIntegration:
    """Test Russian fluency agent glossary integration."""

    def test_agent_has_case_validator(self):
        """Test agent initializes with case validator."""
        provider = MockLLMProvider()
        agent = RussianFluencyAgent(provider)

        assert hasattr(agent, "case_validator")
        assert agent.case_validator is not None
        assert isinstance(agent.case_validator, RussianCaseAspectValidator)

    def test_agent_has_glossary_check_method(self):
        """Test agent has _glossary_check_sync method."""
        provider = MockLLMProvider()
        agent = RussianFluencyAgent(provider)

        assert hasattr(agent, "_glossary_check_sync")
        assert callable(agent._glossary_check_sync)

    def test_glossary_check_returns_error_list(self):
        """Test _glossary_check_sync returns list of errors."""
        provider = MockLLMProvider()
        agent = RussianFluencyAgent(provider)

        task = TranslationTask(
            source_text="Hello",
            translation="Привет",
            source_lang="en",
            target_lang="ru",
        )

        errors = agent._glossary_check_sync(task)

        assert isinstance(errors, list)
        # For now, should return empty list (reference data loaded but no active validation yet)
        assert len(errors) == 0

    def test_case_validator_loads_all_cases(self):
        """Test case validator can load all 6 Russian cases."""
        provider = MockLLMProvider()
        agent = RussianFluencyAgent(provider)

        cases = ["nominative", "genitive", "dative", "accusative", "instrumental", "prepositional"]

        for case_name in cases:
            case_info = agent.case_validator.get_case_info(case_name)
            assert case_info is not None
            assert isinstance(case_info, dict)

    def test_case_validator_loads_aspect_rules(self):
        """Test case validator can load aspect rules."""
        provider = MockLLMProvider()
        agent = RussianFluencyAgent(provider)

        perfective = agent.case_validator.get_aspect_usage_rules("perfective")
        imperfective = agent.case_validator.get_aspect_usage_rules("imperfective")

        assert perfective is not None
        assert imperfective is not None
        assert isinstance(perfective, dict)
        assert isinstance(imperfective, dict)


class TestChineseFluencyAgentGlossaryIntegration:
    """Test Chinese fluency agent glossary integration."""

    def test_agent_has_measure_validator(self):
        """Test agent initializes with measure word validator."""
        provider = MockLLMProvider()
        agent = ChineseFluencyAgent(provider)

        assert hasattr(agent, "measure_validator")
        assert agent.measure_validator is not None
        assert isinstance(agent.measure_validator, ChineseMeasureWordValidator)

    def test_agent_has_glossary_check_method(self):
        """Test agent has _glossary_check_sync method."""
        provider = MockLLMProvider()
        agent = ChineseFluencyAgent(provider)

        assert hasattr(agent, "_glossary_check_sync")
        assert callable(agent._glossary_check_sync)

    def test_glossary_check_returns_error_list(self):
        """Test _glossary_check_sync returns list of errors."""
        provider = MockLLMProvider()
        agent = ChineseFluencyAgent(provider)

        task = TranslationTask(
            source_text="Hello",
            translation="你好",
            source_lang="en",
            target_lang="zh",
        )

        errors = agent._glossary_check_sync(task)

        assert isinstance(errors, list)
        assert len(errors) == 0  # Reference data loaded but no active validation yet

    def test_measure_validator_loads_classifiers(self):
        """Test measure validator can load classifier categories."""
        provider = MockLLMProvider()
        agent = ChineseFluencyAgent(provider)

        # Test loading different classifier categories
        categories = [
            "individual_classifiers",
            "collective_classifiers",
            "container_classifiers",
            "measurement_classifiers",
            "temporal_classifiers",
            "verbal_classifiers",
        ]

        for category in categories:
            classifiers = agent.measure_validator.get_classifier_by_category(category)
            assert classifiers is not None
            assert isinstance(classifiers, dict)

    def test_measure_validator_gets_common_classifiers(self):
        """Test measure validator can get most common classifiers."""
        provider = MockLLMProvider()
        agent = ChineseFluencyAgent(provider)

        common = agent.measure_validator.get_most_common_classifiers(limit=5)

        assert isinstance(common, list)
        assert len(common) <= 5


class TestHindiFluencyAgentGlossaryIntegration:
    """Test Hindi fluency agent glossary integration."""

    def test_agent_has_case_validator(self):
        """Test agent initializes with case/postposition validator."""
        provider = MockLLMProvider()
        agent = HindiFluencyAgent(provider)

        assert hasattr(agent, "case_validator")
        assert agent.case_validator is not None
        assert isinstance(agent.case_validator, HindiPostpositionValidator)

    def test_agent_has_glossary_check_method(self):
        """Test agent has _glossary_check_sync method."""
        provider = MockLLMProvider()
        agent = HindiFluencyAgent(provider)

        assert hasattr(agent, "_glossary_check_sync")
        assert callable(agent._glossary_check_sync)

    def test_glossary_check_returns_error_list(self):
        """Test _glossary_check_sync returns list of errors."""
        provider = MockLLMProvider()
        agent = HindiFluencyAgent(provider)

        task = TranslationTask(
            source_text="Hello",
            translation="नमस्ते",
            source_lang="en",
            target_lang="hi",
        )

        errors = agent._glossary_check_sync(task)

        assert isinstance(errors, list)
        assert len(errors) == 0  # Reference data loaded but no active validation yet

    def test_case_validator_loads_all_hindi_cases(self):
        """Test case validator can load all 8 Hindi cases."""
        provider = MockLLMProvider()
        agent = HindiFluencyAgent(provider)

        # Hindi has 8 cases (कारक)
        for case_num in range(1, 9):
            case_info = agent.case_validator.get_case_info(case_num)
            assert case_info is not None
            assert isinstance(case_info, dict)

    def test_case_validator_loads_oblique_rules(self):
        """Test case validator can load oblique form rules."""
        provider = MockLLMProvider()
        agent = HindiFluencyAgent(provider)

        oblique_rules = agent.case_validator.get_oblique_form_rule()

        assert oblique_rules is not None
        assert isinstance(oblique_rules, dict)


class TestPersianFluencyAgentGlossaryIntegration:
    """Test Persian fluency agent glossary integration."""

    def test_agent_has_ezafe_validator(self):
        """Test agent initializes with ezafe validator."""
        provider = MockLLMProvider()
        agent = PersianFluencyAgent(provider)

        assert hasattr(agent, "ezafe_validator")
        assert agent.ezafe_validator is not None
        assert isinstance(agent.ezafe_validator, PersianEzafeValidator)

    def test_agent_has_glossary_check_method(self):
        """Test agent has _glossary_check_sync method."""
        provider = MockLLMProvider()
        agent = PersianFluencyAgent(provider)

        assert hasattr(agent, "_glossary_check_sync")
        assert callable(agent._glossary_check_sync)

    def test_glossary_check_returns_error_list(self):
        """Test _glossary_check_sync returns list of errors."""
        provider = MockLLMProvider()
        agent = PersianFluencyAgent(provider)

        task = TranslationTask(
            source_text="Hello",
            translation="سلام",
            source_lang="en",
            target_lang="fa",
        )

        errors = agent._glossary_check_sync(task)

        assert isinstance(errors, list)
        assert len(errors) == 0  # Reference data loaded but no active validation yet

    def test_ezafe_validator_loads_ezafe_rules(self):
        """Test ezafe validator can load ezafe construction rules."""
        provider = MockLLMProvider()
        agent = PersianFluencyAgent(provider)

        ezafe_rules = agent.ezafe_validator.get_ezafe_rules()

        assert ezafe_rules is not None
        assert isinstance(ezafe_rules, dict)

    def test_ezafe_validator_loads_compound_verbs(self):
        """Test ezafe validator can load compound verb information."""
        provider = MockLLMProvider()
        agent = PersianFluencyAgent(provider)

        # Test with common light verb
        verb_info = agent.ezafe_validator.get_compound_verb_info("کردن")

        assert verb_info is not None
        assert isinstance(verb_info, dict)


class TestTerminologyAgentGlossaryIntegration:
    """Test terminology agent glossary integration."""

    def test_agent_has_term_validator(self):
        """Test agent initializes with term validator."""
        provider = MockLLMProvider()
        agent = TerminologyAgent(provider)

        assert hasattr(agent, "term_validator")
        assert agent.term_validator is not None
        assert isinstance(agent.term_validator, TermValidator)

    def test_agent_has_glossary_manager(self):
        """Test agent initializes with glossary manager."""
        provider = MockLLMProvider()
        agent = TerminologyAgent(provider)

        assert hasattr(agent, "glossary_manager")
        assert agent.glossary_manager is not None

    @pytest.mark.asyncio
    async def test_agent_validates_mqm_error_types(self):
        """Test agent can validate MQM error types."""
        provider = MockLLMProvider()
        agent = TerminologyAgent(provider)

        # Test with standard MQM error type
        is_valid, info = agent.term_validator.validate_mqm_error_type("mistranslation", "en")

        # Should either be valid or return gracefully
        assert isinstance(is_valid, bool)
        if is_valid:
            assert info is not None
            assert isinstance(info, dict)

    def test_category_property(self):
        """Test agent category property."""
        provider = MockLLMProvider()
        agent = TerminologyAgent(provider)

        assert agent.category == "terminology"
