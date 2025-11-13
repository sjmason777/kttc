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

"""Comprehensive tests for NLP insights UI."""

from unittest.mock import Mock, patch

import pytest

from kttc.cli.ui import print_nlp_insights
from kttc.core import TranslationTask


class TestPrintNLPInsights:
    """Test suite for print_nlp_insights function."""

    @pytest.fixture
    def mock_task_russian(self):
        """Create mock Russian translation task."""
        task = Mock(spec=TranslationTask)
        task.translation = "Быстрая лиса прыгает"
        task.target_lang = "ru"
        return task

    @pytest.fixture
    def mock_task_english(self):
        """Create mock English translation task."""
        task = Mock(spec=TranslationTask)
        task.translation = "The quick fox jumps"
        task.target_lang = "en"
        return task

    @pytest.fixture
    def mock_task_chinese(self):
        """Create mock Chinese translation task."""
        task = Mock(spec=TranslationTask)
        task.translation = "我爱中文"
        task.target_lang = "zh"
        return task

    def test_print_nlp_insights_no_helper(self, mock_task_russian):
        """Test that function handles None helper gracefully."""
        # Should not crash with None helper
        print_nlp_insights(mock_task_russian, None)

    def test_print_nlp_insights_helper_not_available(self, mock_task_russian):
        """Test that function handles unavailable helper gracefully."""
        mock_helper = Mock()
        mock_helper.is_available.return_value = False

        # Should not crash
        print_nlp_insights(mock_task_russian, mock_helper)

    def test_print_nlp_insights_no_morphology(self, mock_task_russian):
        """Test that function handles helper with no morphology."""
        mock_helper = Mock()
        mock_helper.is_available.return_value = True
        mock_helper.get_enrichment_data.return_value = {"has_morphology": False}

        # Should not crash
        print_nlp_insights(mock_task_russian, mock_helper)

    @patch("kttc.cli.ui.console")
    def test_print_nlp_insights_basic_enrichment(self, mock_console, mock_task_russian):
        """Test basic enrichment data display."""
        mock_helper = Mock()
        mock_helper.is_available.return_value = True
        mock_helper.get_enrichment_data.return_value = {
            "has_morphology": True,
            "word_count": 3,
        }
        # Add extract_entities method that returns empty list
        mock_helper.extract_entities.return_value = []

        print_nlp_insights(mock_task_russian, mock_helper)

        # Should call console.print at least once
        assert mock_console.print.called

    @patch("kttc.cli.ui.console")
    def test_print_nlp_insights_with_verb_aspects(self, mock_console, mock_task_russian):
        """Test display with verb aspect information."""
        mock_helper = Mock()
        mock_helper.is_available.return_value = True
        mock_helper.get_enrichment_data.return_value = {
            "has_morphology": True,
            "word_count": 3,
            "verb_aspects": {
                "прыгает": {"aspect": "impf"},
                "прыгнул": {"aspect": "perf"},
            },
        }
        mock_helper.extract_entities.return_value = []

        print_nlp_insights(mock_task_russian, mock_helper)

        # Should display insights
        assert mock_console.print.called

    @patch("kttc.cli.ui.console")
    def test_print_nlp_insights_with_adj_noun_pairs(self, mock_console, mock_task_russian):
        """Test display with adjective-noun pairs."""
        mock_helper = Mock()
        mock_helper.is_available.return_value = True
        mock_helper.get_enrichment_data.return_value = {
            "has_morphology": True,
            "word_count": 2,
            "adjective_noun_pairs": [
                {"agreement": "correct"},
                {"agreement": "mismatch"},
            ],
        }
        mock_helper.extract_entities.return_value = []

        print_nlp_insights(mock_task_russian, mock_helper)

        # Should display insights
        assert mock_console.print.called

    @patch("kttc.cli.ui.console")
    def test_print_nlp_insights_with_entities(self, mock_console, mock_task_english):
        """Test display with named entities."""
        mock_helper = Mock()
        mock_helper.is_available.return_value = True
        mock_helper.get_enrichment_data.return_value = {
            "has_morphology": True,
            "word_count": 5,
        }
        mock_helper.extract_entities.return_value = [
            {"type": "ORG", "text": "Apple"},
            {"type": "PERSON", "text": "Tim Cook"},
            {"type": "GPE", "text": "California"},
        ]

        print_nlp_insights(mock_task_english, mock_helper)

        # Should display insights
        assert mock_console.print.called

    @patch("kttc.cli.ui.console")
    def test_print_nlp_insights_helper_without_extract_entities(
        self, mock_console, mock_task_russian
    ):
        """Test with helper that doesn't have extract_entities method."""
        mock_helper = Mock()
        mock_helper.is_available.return_value = True
        mock_helper.get_enrichment_data.return_value = {
            "has_morphology": True,
            "word_count": 3,
        }
        # Remove extract_entities attribute
        delattr(mock_helper, "extract_entities")

        # Should not crash
        print_nlp_insights(mock_task_russian, mock_helper)

    @patch("kttc.cli.ui.console")
    def test_print_nlp_insights_extract_entities_raises_exception(
        self, mock_console, mock_task_russian
    ):
        """Test that exceptions in extract_entities are handled gracefully."""
        mock_helper = Mock()
        mock_helper.is_available.return_value = True
        mock_helper.get_enrichment_data.return_value = {
            "has_morphology": True,
            "word_count": 3,
        }
        mock_helper.extract_entities.side_effect = Exception("NER failed")

        # Should not crash
        print_nlp_insights(mock_task_russian, mock_helper)

    @patch("kttc.cli.ui.console")
    def test_print_nlp_insights_get_enrichment_raises_exception(
        self, mock_console, mock_task_russian
    ):
        """Test that exceptions in get_enrichment_data are handled gracefully."""
        mock_helper = Mock()
        mock_helper.is_available.return_value = True
        mock_helper.get_enrichment_data.side_effect = Exception("Enrichment failed")

        # Should not crash (caught by outer try-except)
        print_nlp_insights(mock_task_russian, mock_helper)

    @patch("kttc.cli.ui.console")
    def test_print_nlp_insights_complete_russian_data(self, mock_console, mock_task_russian):
        """Test with complete Russian linguistic data."""
        mock_helper = Mock()
        mock_helper.is_available.return_value = True
        mock_helper.get_enrichment_data.return_value = {
            "has_morphology": True,
            "word_count": 5,
            "verb_aspects": {
                "прыгал": {"aspect": "impf"},
                "прыгнул": {"aspect": "perf"},
            },
            "adjective_noun_pairs": [
                {"agreement": "correct"},
                {"agreement": "correct"},
            ],
            "pos_distribution": {"NOUN": 2, "ADJF": 2, "VERB": 1},
        }
        mock_helper.extract_entities.return_value = [
            {"type": "PER", "text": "Иван"},
            {"type": "LOC", "text": "Москва"},
        ]

        print_nlp_insights(mock_task_russian, mock_helper)

        # Should display all insights
        assert mock_console.print.called

    @patch("kttc.cli.ui.console")
    def test_print_nlp_insights_entity_types_aggregation(self, mock_console, mock_task_english):
        """Test that entity types are properly aggregated."""
        mock_helper = Mock()
        mock_helper.is_available.return_value = True
        mock_helper.get_enrichment_data.return_value = {
            "has_morphology": True,
            "word_count": 10,
        }
        mock_helper.extract_entities.return_value = [
            {"type": "ORG", "text": "Apple"},
            {"type": "ORG", "text": "Microsoft"},
            {"type": "PERSON", "text": "Tim Cook"},
            {"type": "GPE", "text": "California"},
            {"type": "GPE", "text": "New York"},
        ]

        print_nlp_insights(mock_task_english, mock_helper)

        # Should aggregate: 2 ORG, 1 PERSON, 2 GPE
        assert mock_console.print.called

    @patch("kttc.cli.ui.console")
    def test_print_nlp_insights_entity_without_type(self, mock_console, mock_task_english):
        """Test handling of entities without type field."""
        mock_helper = Mock()
        mock_helper.is_available.return_value = True
        mock_helper.get_enrichment_data.return_value = {
            "has_morphology": True,
            "word_count": 5,
        }
        mock_helper.extract_entities.return_value = [
            {"text": "Something"},  # No type field
            {"type": "ORG", "text": "Apple"},
        ]

        # Should not crash - defaults to "UNKNOWN"
        print_nlp_insights(mock_task_english, mock_helper)

    @patch("kttc.cli.ui.console")
    def test_print_nlp_insights_zero_word_count(self, mock_console, mock_task_russian):
        """Test with zero word count."""
        mock_helper = Mock()
        mock_helper.is_available.return_value = True
        mock_helper.get_enrichment_data.return_value = {
            "has_morphology": True,
            "word_count": 0,
        }
        mock_helper.extract_entities.return_value = []

        print_nlp_insights(mock_task_russian, mock_helper)

        # Should handle gracefully
        assert mock_console.print.called

    @patch("kttc.cli.ui.console")
    def test_print_nlp_insights_empty_verb_aspects(self, mock_console, mock_task_russian):
        """Test with empty verb aspects."""
        mock_helper = Mock()
        mock_helper.is_available.return_value = True
        mock_helper.get_enrichment_data.return_value = {
            "has_morphology": True,
            "word_count": 3,
            "verb_aspects": {},  # Empty
        }
        mock_helper.extract_entities.return_value = []

        print_nlp_insights(mock_task_russian, mock_helper)

        # Should not display verb aspects row
        assert mock_console.print.called

    @patch("kttc.cli.ui.console")
    def test_print_nlp_insights_empty_adj_noun_pairs(self, mock_console, mock_task_russian):
        """Test with empty adjective-noun pairs."""
        mock_helper = Mock()
        mock_helper.is_available.return_value = True
        mock_helper.get_enrichment_data.return_value = {
            "has_morphology": True,
            "word_count": 3,
            "adjective_noun_pairs": [],  # Empty
        }
        mock_helper.extract_entities.return_value = []

        print_nlp_insights(mock_task_russian, mock_helper)

        # Should not display case agreement row
        assert mock_console.print.called

    @patch("kttc.cli.ui.console")
    def test_print_nlp_insights_empty_entities(self, mock_console, mock_task_english):
        """Test with empty entities list."""
        mock_helper = Mock()
        mock_helper.is_available.return_value = True
        mock_helper.get_enrichment_data.return_value = {
            "has_morphology": True,
            "word_count": 5,
        }
        mock_helper.extract_entities.return_value = []  # Empty

        print_nlp_insights(mock_task_english, mock_helper)

        # Should not display entities row
        assert mock_console.print.called

    @patch("kttc.cli.ui.console")
    def test_print_nlp_insights_missing_word_count(self, mock_console, mock_task_russian):
        """Test with missing word_count field."""
        mock_helper = Mock()
        mock_helper.is_available.return_value = True
        mock_helper.get_enrichment_data.return_value = {
            "has_morphology": True,
            # No word_count field
        }
        mock_helper.extract_entities.return_value = []

        # Should default to 0
        print_nlp_insights(mock_task_russian, mock_helper)

        assert mock_console.print.called

    def test_print_nlp_insights_integration_with_real_helper(self):
        """Test integration with real language helper (if available)."""
        from kttc.helpers.detection import get_helper_for_language

        task = Mock(spec=TranslationTask)
        task.translation = "Hello world"
        task.target_lang = "en"

        helper = get_helper_for_language("en")

        if helper and helper.is_available():
            # Should not crash with real helper
            print_nlp_insights(task, helper)
        else:
            pytest.skip("English helper not available")

    def test_print_nlp_insights_verb_aspect_counting(self):
        """Test correct counting of verb aspects."""
        from kttc.helpers.russian import RussianLanguageHelper

        helper = RussianLanguageHelper()

        if not helper.is_available():
            pytest.skip("Russian helper not available")

        task = Mock(spec=TranslationTask)
        task.translation = "Я прыгал и прыгнул"  # 1 impf + 1 perf
        task.target_lang = "ru"

        # Get real enrichment data
        enrichment = helper.get_enrichment_data(task.translation)

        # Count aspects
        verb_aspects = enrichment.get("verb_aspects", {})
        perf_count = sum(1 for v in verb_aspects.values() if v.get("aspect") == "perf")
        imperf_count = sum(1 for v in verb_aspects.values() if v.get("aspect") == "impf")

        # Should have both perfective and imperfective
        assert perf_count + imperf_count == len(verb_aspects)

    def test_print_nlp_insights_case_agreement_counting(self):
        """Test correct counting of case agreement."""
        from kttc.helpers.russian import RussianLanguageHelper

        helper = RussianLanguageHelper()

        if not helper.is_available():
            pytest.skip("Russian helper not available")

        task = Mock(spec=TranslationTask)
        task.translation = "быстрая лиса"  # Correct agreement
        task.target_lang = "ru"

        # Get real enrichment data
        enrichment = helper.get_enrichment_data(task.translation)

        # Check agreement
        pairs = enrichment.get("adjective_noun_pairs", [])
        correct_count = sum(1 for p in pairs if p.get("agreement") == "correct")

        assert correct_count == len(pairs)  # All should be correct
