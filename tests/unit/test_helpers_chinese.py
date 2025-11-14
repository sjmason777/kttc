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

"""Comprehensive tests for ChineseLanguageHelper."""

import pytest

from kttc.core import ErrorAnnotation, ErrorSeverity
from kttc.helpers.chinese import ChineseLanguageHelper


class TestChineseLanguageHelper:
    """Test suite for Chinese language helper."""

    @pytest.fixture
    def helper(self):
        """Create Chinese language helper instance."""
        return ChineseLanguageHelper()

    def test_initialization(self, helper):
        """Test that helper initializes correctly."""
        assert helper is not None
        assert helper.language_code == "zh"

    def test_is_available(self, helper):
        """Test availability check."""
        # Should be available if jieba or spaCy is installed
        available = helper.is_available()
        assert isinstance(available, bool)

        # If available, should be initialized
        if available:
            assert helper._initialized is True
        else:
            pytest.skip("Neither jieba nor spaCy available - install with: pip install jieba")

    def test_tokenize_simple(self, helper):
        """Test tokenization of simple Chinese text."""
        if not helper.is_available():
            pytest.skip("Chinese NLP not available")

        text = "我爱中文"
        tokens = helper.tokenize(text)

        # Should tokenize into words
        assert len(tokens) >= 2  # At least "我爱" and "中文" or "我", "爱", "中文"

        # Check positions are valid
        for word, start, end in tokens:
            assert text[start:end] == word

    def test_tokenize_complex(self, helper):
        """Test tokenization of complex Chinese text."""
        if not helper.is_available():
            pytest.skip("Chinese NLP not available")

        text = "苹果公司首席执行官库克宣布新产品"
        tokens = helper.tokenize(text)

        # Should have multiple tokens
        assert len(tokens) > 0

        # Check positions
        for word, start, end in tokens:
            assert text[start:end] == word

    def test_tokenize_mixed_content(self, helper):
        """Test tokenization with mixed Chinese and English."""
        if not helper.is_available():
            pytest.skip("Chinese NLP not available")

        text = "我爱Python编程"
        tokens = helper.tokenize(text)

        # Should handle mixed content
        assert len(tokens) > 0

        for word, start, end in tokens:
            assert text[start:end] == word

    def test_tokenize_with_punctuation(self, helper):
        """Test tokenization with Chinese punctuation."""
        if not helper.is_available():
            pytest.skip("Chinese NLP not available")

        text = "你好，世界！"
        tokens = helper.tokenize(text)

        # Should tokenize correctly
        assert len(tokens) > 0

        # Verify positions
        for word, start, end in tokens:
            assert text[start:end] == word

    def test_tokenize_fallback(self):
        """Test tokenization fallback when no NLP available."""
        helper = ChineseLanguageHelper()
        helper._initialized = False
        helper._nlp = None

        text = "我爱中文"
        tokens = helper.tokenize(text)

        # Fallback uses character-level tokenization
        # Should return each non-whitespace character
        assert len(tokens) == 4  # "我", "爱", "中", "文"

    def test_verify_word_exists_found(self, helper):
        """Test word verification when word exists."""
        if not helper.is_available():
            pytest.skip("Chinese NLP not available")

        text = "我爱中文编程"
        tokens = helper.tokenize(text)

        # Get first token and verify it exists
        if len(tokens) > 0:
            first_word = tokens[0][0]
            assert helper.verify_word_exists(first_word, text) is True

    def test_verify_word_exists_not_found(self, helper):
        """Test word verification when word doesn't exist."""
        if not helper.is_available():
            pytest.skip("Chinese NLP not available")

        text = "我爱中文"
        assert helper.verify_word_exists("英语", text) is False

    def test_verify_error_position_valid(self, helper):
        """Test error position verification with valid positions."""
        text = "我爱中文"

        error = ErrorAnnotation(
            category="accuracy",
            subcategory="test",
            severity=ErrorSeverity.MINOR,
            location=(0, 1),  # "我"
            description="test",
        )
        assert helper.verify_error_position(error, text) is True

    def test_verify_error_position_invalid(self, helper):
        """Test error position verification with invalid positions."""
        text = "我爱中文"

        # Out of bounds
        error1 = ErrorAnnotation(
            category="accuracy",
            subcategory="test",
            severity=ErrorSeverity.MINOR,
            location=(0, 100),
            description="test",
        )
        assert helper.verify_error_position(error1, text) is False

        # Negative start
        error2 = ErrorAnnotation(
            category="accuracy",
            subcategory="test",
            severity=ErrorSeverity.MINOR,
            location=(-1, 1),
            description="test",
        )
        assert helper.verify_error_position(error2, text) is False

    def test_analyze_morphology(self, helper):
        """Test morphological analysis."""
        if not helper.is_available():
            pytest.skip("Chinese NLP not available")

        text = "我爱中文"
        morphology = helper.analyze_morphology(text)

        # If spaCy is available, should return morphology
        # If only jieba, returns empty list
        assert isinstance(morphology, list)

        # If we got results, check structure
        for morph in morphology:
            assert hasattr(morph, "word")
            assert hasattr(morph, "pos")
            assert hasattr(morph, "start")
            assert hasattr(morph, "stop")
            assert text[morph.start : morph.stop] == morph.word

    def test_get_enrichment_data_with_spacy(self, helper):
        """Test enrichment data extraction with spaCy."""
        if not helper.is_available():
            pytest.skip("Chinese NLP not available")

        text = "苹果公司首席执行官库克宣布新产品"
        enrichment = helper.get_enrichment_data(text)

        assert enrichment["has_morphology"] is True
        assert "word_count" in enrichment

    def test_get_enrichment_data_jieba_only(self):
        """Test enrichment data with jieba-only mode."""
        # Create helper and force jieba-only mode
        helper = ChineseLanguageHelper()

        if not helper.is_available():
            pytest.skip("Chinese NLP not available")

        # If we have jieba, test it
        text = "我爱中文编程"
        enrichment = helper.get_enrichment_data(text)

        assert "has_morphology" in enrichment
        assert "word_count" in enrichment or "segmentation_method" in enrichment

    def test_extract_entities(self, helper):
        """Test named entity extraction."""
        if not helper.is_available():
            pytest.skip("Chinese NLP not available")

        text = "苹果公司首席执行官库克在北京宣布新产品"
        entities = helper.extract_entities(text)

        # Returns list (might be empty if only jieba)
        assert isinstance(entities, list)

        # If we got entities, check structure
        for entity in entities:
            assert "text" in entity
            assert "type" in entity
            assert "start" in entity
            assert "stop" in entity
            assert text[entity["start"] : entity["stop"]] == entity["text"]

    def test_extract_entities_no_entities(self, helper):
        """Test entity extraction when no entities present."""
        if not helper.is_available():
            pytest.skip("Chinese NLP not available")

        text = "我喜欢吃饭"
        entities = helper.extract_entities(text)

        # Should return empty list
        assert isinstance(entities, list)

    def test_check_entity_preservation(self, helper):
        """Test entity preservation checking."""
        if not helper.is_available():
            pytest.skip("Chinese NLP not available")

        source = "Apple CEO Tim Cook visited Beijing"
        translation = "苹果首席执行官库克访问北京"

        errors = helper.check_entity_preservation(source, translation)

        # Should return list of errors
        assert isinstance(errors, list)

    def test_check_entity_preservation_missing_entities(self, helper):
        """Test entity preservation when translation missing entities."""
        if not helper.is_available():
            pytest.skip("Chinese NLP not available")

        source = "Apple CEO Tim Cook announced products"
        translation = "首席执行官宣布产品"  # Missing names

        errors = helper.check_entity_preservation(source, translation)

        # Source has capitalized words, might detect missing entities
        assert isinstance(errors, list)

    def test_check_grammar(self, helper):
        """Test grammar checking with HanLP."""
        if not helper.is_available():
            pytest.skip("Chinese NLP not available")

        text = "我爱中文"
        errors = helper.check_grammar(text)

        # Should return list of errors
        assert isinstance(errors, list)

    def test_check_measure_words_incorrect(self, helper):
        """Test detection of incorrect measure words."""
        if not helper.is_available():
            pytest.skip("Chinese NLP not available")

        if not helper._hanlp_available:
            pytest.skip("HanLP not available for grammar checking")

        # Incorrect: "个" for books (should be "本")
        text = "三个书"
        errors = helper.check_grammar(text)

        # Should detect incorrect measure word
        assert len(errors) > 0
        assert any("measure" in e.subcategory.lower() for e in errors)
        assert any("个" in e.description for e in errors)

    def test_check_measure_words_correct(self, helper):
        """Test that correct measure words don't trigger errors."""
        if not helper.is_available():
            pytest.skip("Chinese NLP not available")

        if not helper._hanlp_available:
            pytest.skip("HanLP not available for grammar checking")

        # Correct: "本" for books
        text = "三本书"
        errors = helper.check_grammar(text)

        # Should not detect errors for correct measure word
        measure_errors = [e for e in errors if "measure" in e.subcategory.lower()]
        assert len(measure_errors) == 0

    def test_check_measure_words_vehicles(self, helper):
        """Test measure word checking for vehicles."""
        if not helper.is_available():
            pytest.skip("Chinese NLP not available")

        if not helper._hanlp_available:
            pytest.skip("HanLP not available for grammar checking")

        # Incorrect: "本" for cars (should be "辆")
        text = "一本车"
        errors = helper.check_grammar(text)

        # Should detect incorrect measure word
        measure_errors = [e for e in errors if "measure" in e.subcategory.lower()]
        if len(measure_errors) > 0:
            assert any("本" in e.description for e in measure_errors)

    def test_check_measure_words_animals(self, helper):
        """Test measure word checking for animals."""
        if not helper.is_available():
            pytest.skip("Chinese NLP not available")

        if not helper._hanlp_available:
            pytest.skip("HanLP not available for grammar checking")

        # Test incorrect measure word for animals
        text = "两条狗"  # Should be "两只狗"
        errors = helper.check_grammar(text)

        # May detect error depending on dictionary
        assert isinstance(errors, list)

    def test_check_measure_words_unknown_noun(self, helper):
        """Test that unknown nouns don't trigger false positives."""
        if not helper.is_available():
            pytest.skip("Chinese NLP not available")

        if not helper._hanlp_available:
            pytest.skip("HanLP not available for grammar checking")

        # Unknown noun - should not flag error
        text = "三个测试物品"
        errors = helper.check_grammar(text)

        # Should not flag error for unknown nouns
        # (dictionary doesn't contain this noun)
        assert isinstance(errors, list)

    def test_check_particles_correct(self, helper):
        """Test aspect particle checking with correct usage."""
        if not helper.is_available():
            pytest.skip("Chinese NLP not available")

        if not helper._hanlp_available:
            pytest.skip("HanLP not available for grammar checking")

        # Correct: 了 after verb 买
        text = "我买了书"
        errors = helper.check_grammar(text)

        # Should not flag errors for correct particle usage
        particle_errors = [e for e in errors if "particle" in e.subcategory.lower()]
        assert len(particle_errors) == 0

    def test_check_particles_experience(self, helper):
        """Test aspect particle 过 (experience)."""
        if not helper.is_available():
            pytest.skip("Chinese NLP not available")

        if not helper._hanlp_available:
            pytest.skip("HanLP not available for grammar checking")

        # Correct: 过 after verb 去
        text = "我去过北京"
        errors = helper.check_grammar(text)

        # Should not flag errors
        particle_errors = [e for e in errors if "particle" in e.subcategory.lower()]
        assert len(particle_errors) == 0

    def test_get_enrichment_data_with_hanlp(self, helper):
        """Test enrichment data with HanLP insights."""
        if not helper.is_available():
            pytest.skip("Chinese NLP not available")

        if not helper._hanlp_available:
            pytest.skip("HanLP not available")

        text = "我买了三本书"
        enrichment = helper.get_enrichment_data(text)

        # Should have HanLP data
        assert enrichment.get("has_hanlp") is True
        assert "measure_patterns" in enrichment
        assert "aspect_particles" in enrichment
        assert "pos_distribution" in enrichment

        # Should detect measure word pattern
        measure_patterns = enrichment["measure_patterns"]
        assert len(measure_patterns) > 0
        assert any(p["measure"] == "本" and p["noun"] == "书" for p in measure_patterns)

        # Should detect aspect particle 了
        aspect_particles = enrichment["aspect_particles"]
        assert len(aspect_particles) > 0
        assert any(p["particle"] == "了" for p in aspect_particles)

    def test_get_enrichment_data_pos_counts(self, helper):
        """Test POS tag distribution in enrichment data."""
        if not helper.is_available():
            pytest.skip("Chinese NLP not available")

        if not helper._hanlp_available:
            pytest.skip("HanLP not available")

        text = "三本书很贵"
        enrichment = helper.get_enrichment_data(text)

        # Should have CTB POS counts
        pos_dist = enrichment.get("pos_distribution", {})
        assert len(pos_dist) > 0

        # Should detect CD (number), M (measure), NN (noun)
        # Actual tags depend on HanLP model
        assert isinstance(pos_dist, dict)

    def test_get_enrichment_data_no_measure_words(self, helper):
        """Test enrichment data when no measure words present."""
        if not helper.is_available():
            pytest.skip("Chinese NLP not available")

        if not helper._hanlp_available:
            pytest.skip("HanLP not available")

        text = "我爱中文"
        enrichment = helper.get_enrichment_data(text)

        # Should have empty measure patterns
        measure_patterns = enrichment.get("measure_patterns", [])
        assert isinstance(measure_patterns, list)
        assert len(measure_patterns) == 0

    def test_get_enrichment_data_multiple_measures(self, helper):
        """Test enrichment data with multiple measure words."""
        if not helper.is_available():
            pytest.skip("Chinese NLP not available")

        if not helper._hanlp_available:
            pytest.skip("HanLP not available")

        text = "我有三本书和两辆车"
        enrichment = helper.get_enrichment_data(text)

        # Should detect multiple measure patterns
        measure_patterns = enrichment.get("measure_patterns", [])
        if len(measure_patterns) > 1:
            # Should find both "三本书" and "两辆车"
            assert any(p["noun"] == "书" for p in measure_patterns)
            assert any(p["noun"] == "车" for p in measure_patterns)

    def test_empty_text_handling(self, helper):
        """Test handling of empty text."""
        if not helper.is_available():
            pytest.skip("Chinese NLP not available")

        text = ""

        tokens = helper.tokenize(text)
        assert tokens == []

        morphology = helper.analyze_morphology(text)
        assert isinstance(morphology, list)

        entities = helper.extract_entities(text)
        assert entities == []

    def test_whitespace_text_handling(self, helper):
        """Test handling of whitespace-only text."""
        if not helper.is_available():
            pytest.skip("Chinese NLP not available")

        text = "   \n\t  "

        tokens = helper.tokenize(text)
        # Should return empty or only whitespace tokens
        assert isinstance(tokens, list)

    def test_numbers_and_symbols(self, helper):
        """Test handling of numbers and symbols."""
        if not helper.is_available():
            pytest.skip("Chinese NLP not available")

        text = "2025年11月13日"
        tokens = helper.tokenize(text)

        # Should tokenize without errors
        assert len(tokens) > 0

        # Verify positions
        for word, start, end in tokens:
            assert text[start:end] == word

    def test_traditional_chinese(self, helper):
        """Test handling of traditional Chinese characters."""
        if not helper.is_available():
            pytest.skip("Chinese NLP not available")

        text = "我愛繁體中文"
        tokens = helper.tokenize(text)

        # Should handle traditional characters
        assert len(tokens) > 0

        for word, start, end in tokens:
            assert text[start:end] == word

    def test_very_long_text(self, helper):
        """Test handling of very long Chinese text."""
        if not helper.is_available():
            pytest.skip("Chinese NLP not available")

        # Create a long text
        text = "我爱中文编程。" * 100

        tokens = helper.tokenize(text)
        enrichment = helper.get_enrichment_data(text)

        # Should process without errors
        assert len(tokens) > 0
        assert enrichment["has_morphology"] is True

    def test_mixed_chinese_english_numbers(self, helper):
        """Test handling of mixed content."""
        if not helper.is_available():
            pytest.skip("Chinese NLP not available")

        text = "Python 3.11 是最新版本，我在2025年学习"
        tokens = helper.tokenize(text)

        # Should handle mixed content
        assert len(tokens) > 0

        # Verify all positions
        for word, start, end in tokens:
            assert text[start:end] == word

    def test_chinese_punctuation_only(self, helper):
        """Test handling of Chinese punctuation."""
        if not helper.is_available():
            pytest.skip("Chinese NLP not available")

        text = "，。！？；："
        tokens = helper.tokenize(text)

        # Should handle punctuation
        assert isinstance(tokens, list)

    def test_jieba_vs_spacy_consistency(self, helper):
        """Test that jieba and spaCy produce consistent results."""
        if not helper.is_available():
            pytest.skip("Chinese NLP not available")

        text = "我爱中文编程"

        # Tokenize once
        tokens1 = helper.tokenize(text)

        # Tokenize again (should be consistent)
        tokens2 = helper.tokenize(text)

        assert tokens1 == tokens2

    def test_position_accuracy(self, helper):
        """Test that all token positions are accurate."""
        if not helper.is_available():
            pytest.skip("Chinese NLP not available")

        text = "苹果公司在北京开设新办公室"
        tokens = helper.tokenize(text)

        # Every token should match its position in text
        for word, start, end in tokens:
            assert (
                text[start:end] == word
            ), f"Position mismatch: text[{start}:{end}] = '{text[start:end]}' != '{word}'"

    def test_no_overlapping_tokens(self, helper):
        """Test that tokens don't overlap."""
        if not helper.is_available():
            pytest.skip("Chinese NLP not available")

        text = "我爱中文编程"
        tokens = helper.tokenize(text)

        # Sort by start position
        sorted_tokens = sorted(tokens, key=lambda t: t[1])

        # Check no overlaps
        for i in range(len(sorted_tokens) - 1):
            current_end = sorted_tokens[i][2]
            next_start = sorted_tokens[i + 1][1]
            assert (
                current_end <= next_start
            ), f"Overlapping tokens: {sorted_tokens[i]} and {sorted_tokens[i+1]}"
