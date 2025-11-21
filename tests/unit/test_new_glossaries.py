"""
Comprehensive unit tests for new glossaries (2025-11-21 expansion).

Tests all 7 new glossary files:
- en/mqm_error_taxonomy.json
- en/translation_metrics.json
- en/transformer_nlp_terms.json
- en/llm_terminology.json
- zh/idioms_expressions_zh.json
- ru/morphology_ru.json (verification)
"""

import json
from pathlib import Path

import pytest

from kttc.terminology import GlossaryManager


@pytest.fixture
def glossary_manager():
    """Fixture providing GlossaryManager instance."""
    return GlossaryManager()


@pytest.fixture
def glossaries_dir():
    """Fixture providing path to glossaries directory."""
    current_file = Path(__file__)
    project_root = current_file.parent.parent.parent
    return project_root / "glossaries"


class TestMQMErrorTaxonomy:
    """Tests for en/mqm_error_taxonomy.json."""

    def test_loads_successfully(self, glossary_manager):
        """Test that MQM error taxonomy loads without errors."""
        glossary = glossary_manager.load_glossary("en", "mqm_error_taxonomy")
        assert glossary is not None
        assert "metadata" in glossary

    def test_metadata_structure(self, glossary_manager):
        """Test metadata has correct structure."""
        glossary = glossary_manager.load_glossary("en", "mqm_error_taxonomy")
        metadata = glossary["metadata"]

        assert metadata["language"] == "en"
        assert metadata["language_name"] == "English"
        assert metadata["total_terms"] == 47
        assert "MQM" in metadata["glossary_type"]
        assert len(metadata["sources"]) > 0

    def test_severity_levels_present(self, glossary_manager):
        """Test that all severity levels are defined."""
        glossary = glossary_manager.load_glossary("en", "mqm_error_taxonomy")

        assert "severity_levels" in glossary
        assert "minor" in glossary["severity_levels"]
        assert "major" in glossary["severity_levels"]
        assert "critical" in glossary["severity_levels"]

        # Check weights
        assert glossary["severity_levels"]["minor"]["weight"] == 1
        assert glossary["severity_levels"]["major"]["weight"] == 5
        assert glossary["severity_levels"]["critical"]["weight"] == 10

    def test_accuracy_category(self, glossary_manager):
        """Test accuracy category is complete."""
        glossary = glossary_manager.load_glossary("en", "mqm_error_taxonomy")

        assert "accuracy" in glossary
        accuracy = glossary["accuracy"]

        assert accuracy["category_weight"] == 1.5
        assert "subcategories" in accuracy

        # Check key error types
        subcats = accuracy["subcategories"]
        assert "addition" in subcats
        assert "mistranslation" in subcats
        assert "omission" in subcats
        assert "untranslated" in subcats

        # Check structure
        addition = subcats["addition"]
        assert "error_code" in addition
        assert addition["error_code"] == "ACC-ADD"
        assert "examples" in addition
        assert len(addition["examples"]) > 0

    def test_fluency_category(self, glossary_manager):
        """Test fluency category is complete."""
        glossary = glossary_manager.load_glossary("en", "mqm_error_taxonomy")

        assert "fluency" in glossary
        fluency = glossary["fluency"]

        subcats = fluency["subcategories"]
        assert "grammar" in subcats
        assert "spelling" in subcats
        assert "punctuation" in subcats
        assert "inconsistency" in subcats
        assert "register" in subcats

    def test_scoring_formula(self, glossary_manager):
        """Test scoring formula is documented."""
        glossary = glossary_manager.load_glossary("en", "mqm_error_taxonomy")

        assert "scoring_formula" in glossary
        formula = glossary["scoring_formula"]

        assert "severity_weights" in formula
        assert "category_weights" in formula
        assert "quality_thresholds" in formula

        # Check thresholds
        thresholds = formula["quality_thresholds"]
        assert "excellent" in thresholds
        assert "≥95" in thresholds["excellent"]

    def test_all_categories_present(self, glossary_manager):
        """Test all MQM categories are present."""
        glossary = glossary_manager.load_glossary("en", "mqm_error_taxonomy")

        required_categories = [
            "accuracy",
            "fluency",
            "terminology",
            "style",
            "locale_convention",
            "verity",
        ]

        for category in required_categories:
            assert category in glossary, f"Missing category: {category}"


class TestTranslationMetrics:
    """Tests for en/translation_metrics.json."""

    def test_loads_successfully(self, glossary_manager):
        """Test that translation metrics loads without errors."""
        glossary = glossary_manager.load_glossary("en", "translation_metrics")
        assert glossary is not None
        assert "metadata" in glossary

    def test_metadata_structure(self, glossary_manager):
        """Test metadata has correct structure."""
        glossary = glossary_manager.load_glossary("en", "translation_metrics")
        metadata = glossary["metadata"]

        assert metadata["language"] == "en"
        assert metadata["total_terms"] == 32
        assert "metrics" in metadata["glossary_type"].lower()

    def test_automated_metrics_present(self, glossary_manager):
        """Test key automated metrics are documented."""
        glossary = glossary_manager.load_glossary("en", "translation_metrics")

        assert "automated_metrics" in glossary
        metrics = glossary["automated_metrics"]

        # Check key metrics
        required_metrics = ["bleu", "ter", "comet", "bertscore"]
        for metric in required_metrics:
            assert metric in metrics, f"Missing metric: {metric}"

    def test_bleu_metric_details(self, glossary_manager):
        """Test BLEU metric has complete information."""
        glossary = glossary_manager.load_glossary("en", "translation_metrics")
        bleu = glossary["automated_metrics"]["bleu"]

        assert bleu["acronym"] == "BLEU"
        assert "Bilingual Evaluation Understudy" in bleu["full_name"]
        assert "range" in bleu
        assert "advantages" in bleu
        assert "limitations" in bleu
        assert len(bleu["advantages"]) > 0
        assert len(bleu["limitations"]) > 0

    def test_comet_metric_details(self, glossary_manager):
        """Test COMET metric has version information."""
        glossary = glossary_manager.load_glossary("en", "translation_metrics")
        comet = glossary["automated_metrics"]["comet"]

        assert "versions" in comet
        versions = comet["versions"]
        assert "comet_22" in versions
        assert "COMET-22" in versions["comet_22"]

    def test_evaluation_approaches(self, glossary_manager):
        """Test evaluation approaches are documented."""
        glossary = glossary_manager.load_glossary("en", "translation_metrics")

        assert "evaluation_approaches" in glossary
        approaches = glossary["evaluation_approaches"]

        assert "direct_assessment" in approaches
        assert "mqm" in approaches
        assert "quality_estimation" in approaches


class TestTransformerNLPTerms:
    """Tests for en/transformer_nlp_terms.json."""

    def test_loads_successfully(self, glossary_manager):
        """Test that transformer NLP terms loads without errors."""
        glossary = glossary_manager.load_glossary("en", "transformer_nlp_terms")
        assert glossary is not None

    def test_metadata_structure(self, glossary_manager):
        """Test metadata has correct structure."""
        glossary = glossary_manager.load_glossary("en", "transformer_nlp_terms")
        metadata = glossary["metadata"]

        assert metadata["language"] == "en"
        assert metadata["total_terms"] == 58
        assert "transformer" in metadata["glossary_type"].lower()

    def test_tokenization_section(self, glossary_manager):
        """Test tokenization concepts are documented."""
        glossary = glossary_manager.load_glossary("en", "transformer_nlp_terms")

        assert "tokenization" in glossary
        tokenization = glossary["tokenization"]

        # Check key tokenization methods
        assert "bpe" in tokenization
        assert "wordpiece" in tokenization
        assert "sentencepiece" in tokenization

        # Check BPE details
        bpe = tokenization["bpe"]
        assert bpe["acronym"] == "BPE"
        assert "Byte-Pair Encoding" in bpe["full_name"]
        assert "algorithm" in bpe
        assert "used_in" in bpe

    def test_embeddings_section(self, glossary_manager):
        """Test embeddings concepts are documented."""
        glossary = glossary_manager.load_glossary("en", "transformer_nlp_terms")

        assert "embeddings" in glossary
        embeddings = glossary["embeddings"]

        assert "token_embeddings" in embeddings
        assert "positional_embeddings" in embeddings
        assert "contextual_embeddings" in embeddings

    def test_attention_mechanisms(self, glossary_manager):
        """Test attention mechanisms are documented."""
        glossary = glossary_manager.load_glossary("en", "transformer_nlp_terms")

        assert "attention_mechanisms" in glossary
        attention = glossary["attention_mechanisms"]

        required = [
            "self_attention",
            "multi_head_attention",
            "causal_attention",
            "cross_attention",
        ]

        for mechanism in required:
            assert mechanism in attention, f"Missing: {mechanism}"

    def test_model_architectures(self, glossary_manager):
        """Test model architectures are documented."""
        glossary = glossary_manager.load_glossary("en", "transformer_nlp_terms")

        assert "model_architectures" in glossary
        models = glossary["model_architectures"]

        # Check key models
        assert "bert" in models
        assert "gpt" in models
        assert "t5" in models

        # Check BERT details
        bert = models["bert"]
        assert bert["acronym"] == "BERT"
        assert "Bidirectional" in bert["full_name"]
        assert bert["type"] == "Encoder-only"


class TestLLMTerminology:
    """Tests for en/llm_terminology.json."""

    def test_loads_successfully(self, glossary_manager):
        """Test that LLM terminology loads without errors."""
        glossary = glossary_manager.load_glossary("en", "llm_terminology")
        assert glossary is not None

    def test_metadata_structure(self, glossary_manager):
        """Test metadata has correct structure."""
        glossary = glossary_manager.load_glossary("en", "llm_terminology")
        metadata = glossary["metadata"]

        assert metadata["language"] == "en"
        assert metadata["total_terms"] == 45
        assert "LLM" in metadata["glossary_type"]

    def test_hallucination_documented(self, glossary_manager):
        """Test hallucination concept is well-documented."""
        glossary = glossary_manager.load_glossary("en", "llm_terminology")

        assert "hallucination" in glossary
        hallucination = glossary["hallucination"]

        assert "definition" in hallucination
        assert "causes" in hallucination
        assert "mitigation" in hallucination
        assert "detection" in hallucination

        # Check mitigation strategies
        mitigation = hallucination["mitigation"]
        assert "retrieval_augmentation" in mitigation
        assert "fact_checking" in mitigation

    def test_prompt_engineering(self, glossary_manager):
        """Test prompt engineering techniques are documented."""
        glossary = glossary_manager.load_glossary("en", "llm_terminology")

        assert "prompt_engineering" in glossary
        pe = glossary["prompt_engineering"]

        assert "techniques" in pe
        techniques = pe["techniques"]

        # Check key techniques
        assert "zero_shot" in techniques
        assert "few_shot" in techniques
        assert "chain_of_thought" in techniques
        assert "system_prompt" in techniques

        # Check CoT details
        cot = techniques["chain_of_thought"]
        assert cot["acronym"] == "CoT"
        assert "variants" in cot

    def test_rlhf_documented(self, glossary_manager):
        """Test RLHF is comprehensively documented."""
        glossary = glossary_manager.load_glossary("en", "llm_terminology")

        assert "rlhf" in glossary
        rlhf = glossary["rlhf"]

        assert rlhf["acronym"] == "RLHF"
        assert "Reinforcement Learning from Human Feedback" in rlhf["full_name"]
        assert "process" in rlhf

        # Check 3-step process
        process = rlhf["process"]
        assert "step_1" in process  # SFT
        assert "step_2" in process  # Reward Model
        assert "step_3" in process  # RL Fine-Tuning

    def test_alignment_documented(self, glossary_manager):
        """Test alignment concept is documented."""
        glossary = glossary_manager.load_glossary("en", "llm_terminology")

        assert "alignment" in glossary
        alignment = glossary["alignment"]

        assert "definition" in alignment
        assert "goals" in alignment["definition"]

        # Check 3H framework
        goals = alignment["definition"]["goals"]
        assert any("Helpful" in str(g) for g in goals)
        assert any("Honest" in str(g) for g in goals)
        assert any("Harmless" in str(g) for g in goals)

    def test_rag_documented(self, glossary_manager):
        """Test RAG is documented."""
        glossary = glossary_manager.load_glossary("en", "llm_terminology")

        assert "retrieval_augmented_generation" in glossary
        rag = glossary["retrieval_augmented_generation"]

        assert rag["acronym"] == "RAG"
        assert "workflow" in rag
        assert "advantages" in rag


class TestChineseIdiomsExpressions:
    """Tests for zh/idioms_expressions_zh.json."""

    def test_loads_successfully(self, glossary_manager):
        """Test that Chinese idioms loads without errors."""
        glossary = glossary_manager.load_glossary("zh", "idioms_expressions_zh")
        assert glossary is not None

    def test_metadata_structure(self, glossary_manager):
        """Test metadata has correct structure."""
        glossary = glossary_manager.load_glossary("zh", "idioms_expressions_zh")
        metadata = glossary["metadata"]

        assert metadata["language"] == "zh"
        assert metadata["language_name"] == "中文 (Chinese)"
        assert metadata["total_terms"] == 120

    def test_chengyu_section(self, glossary_manager):
        """Test chengyu (成语) section is complete."""
        glossary = glossary_manager.load_glossary("zh", "idioms_expressions_zh")

        assert "chengyu" in glossary
        chengyu = glossary["chengyu"]

        assert "definition" in chengyu
        assert "examples" in chengyu

        # Check examples structure
        examples = chengyu["examples"]
        assert "historical_origin" in examples
        assert "common_usage" in examples
        assert "philosophical" in examples

        # Check one example
        historical = examples["historical_origin"]
        assert len(historical) > 0
        first_example = historical[0]
        assert "chengyu" in first_example
        assert "pinyin" in first_example
        assert "meaning" in first_example

    def test_guanyongyu_section(self, glossary_manager):
        """Test guanyongyu (惯用语) section is complete."""
        glossary = glossary_manager.load_glossary("zh", "idioms_expressions_zh")

        assert "guanyongyu" in glossary
        guanyongyu = glossary["guanyongyu"]

        assert "definition" in guanyongyu
        assert "structure_patterns" in guanyongyu

        # Check verb-object pattern
        patterns = guanyongyu["structure_patterns"]
        assert "verb_object" in patterns

        vo = patterns["verb_object"]
        assert "examples" in vo
        assert len(vo["examples"]) > 0

        # Check example structure
        example = vo["examples"][0]
        assert "phrase" in example
        assert "pinyin" in example
        assert "meaning" in example

    def test_xiehouyu_section(self, glossary_manager):
        """Test xiehouyu (歇后语) section is complete."""
        glossary = glossary_manager.load_glossary("zh", "idioms_expressions_zh")

        assert "xiehouyu" in glossary
        xiehouyu = glossary["xiehouyu"]

        assert "definition" in xiehouyu
        assert "examples" in xiehouyu

        examples = xiehouyu["examples"]
        assert "homophone_type" in examples
        assert "metaphor_type" in examples

    def test_yanyu_section(self, glossary_manager):
        """Test yanyu (谚语) section is complete."""
        glossary = glossary_manager.load_glossary("zh", "idioms_expressions_zh")

        assert "yanyu" in glossary
        yanyu = glossary["yanyu"]

        assert "definition" in yanyu
        assert "examples" in yanyu

    def test_translation_strategies(self, glossary_manager):
        """Test translation strategies are documented."""
        glossary = glossary_manager.load_glossary("zh", "idioms_expressions_zh")

        assert "usage_in_translation" in glossary
        usage = glossary["usage_in_translation"]

        assert "challenges" in usage
        assert "best_practices" in usage
        assert "common_errors" in usage


class TestRussianMorphology:
    """Tests for ru/morphology_ru.json (verification)."""

    def test_loads_successfully(self, glossary_manager):
        """Test that Russian morphology loads without errors."""
        glossary = glossary_manager.load_glossary("ru", "morphology_ru")
        assert glossary is not None

    def test_metadata_structure(self, glossary_manager):
        """Test metadata has correct structure."""
        glossary = glossary_manager.load_glossary("ru", "morphology_ru")
        metadata = glossary["metadata"]

        assert metadata["language"] == "ru"
        assert metadata["language_name"] == "Русский"
        assert metadata["total_terms"] >= 70

    def test_core_concepts(self, glossary_manager):
        """Test core morphology concepts are present."""
        glossary = glossary_manager.load_glossary("ru", "morphology_ru")

        assert "core_concepts" in glossary
        concepts = glossary["core_concepts"]

        assert "morphology" in concepts
        assert "morpheme" in concepts
        assert "allomorph" in concepts
        assert "zero_morpheme" in concepts

    def test_grammatical_categories(self, glossary_manager):
        """Test grammatical categories are complete."""
        glossary = glossary_manager.load_glossary("ru", "morphology_ru")

        assert "grammatical_categories" in glossary
        categories = glossary["grammatical_categories"]

        assert "case" in categories
        assert "gender" in categories
        assert "aspect" in categories

        # Check cases
        cases = categories["case"]["russian_cases"]
        assert len(cases) == 6  # Russian has 6 cases


class TestJSONValidity:
    """Tests for JSON syntax and structure validity."""

    def test_all_new_glossaries_valid_json(self, glossaries_dir):
        """Test that all new glossary files are valid JSON."""
        new_files = [
            "en/mqm_error_taxonomy.json",
            "en/translation_metrics.json",
            "en/transformer_nlp_terms.json",
            "en/llm_terminology.json",
            "zh/idioms_expressions_zh.json",
        ]

        for file_path in new_files:
            full_path = glossaries_dir / file_path
            assert full_path.exists(), f"File not found: {file_path}"

            with open(full_path, encoding="utf-8") as f:
                try:
                    data = json.load(f)
                    assert isinstance(data, dict)
                    assert "metadata" in data
                except json.JSONDecodeError as e:
                    pytest.fail(f"Invalid JSON in {file_path}: {e}")

    def test_metadata_required_fields(self, glossaries_dir):
        """Test that all glossaries have required metadata fields."""
        new_files = [
            "en/mqm_error_taxonomy.json",
            "en/translation_metrics.json",
            "en/transformer_nlp_terms.json",
            "en/llm_terminology.json",
            "zh/idioms_expressions_zh.json",
        ]

        required_fields = [
            "language",
            "language_name",
            "glossary_type",
            "version",
            "created",
            "description",
        ]

        for file_path in new_files:
            full_path = glossaries_dir / file_path

            with open(full_path, encoding="utf-8") as f:
                data = json.load(f)
                metadata = data["metadata"]

                for field in required_fields:
                    assert field in metadata, f"Missing field '{field}' in {file_path} metadata"


class TestGlossaryManagerIntegration:
    """Tests for GlossaryManager with new glossaries."""

    def test_list_available_includes_new_glossaries(self, glossary_manager):
        """Test that new glossaries appear in available list."""
        available = glossary_manager.list_available_glossaries()

        assert "en" in available
        en_glossaries = available["en"]

        # Check new English glossaries
        assert "mqm_error_taxonomy" in en_glossaries
        assert "translation_metrics" in en_glossaries
        assert "transformer_nlp_terms" in en_glossaries
        assert "llm_terminology" in en_glossaries

        # Check new Chinese glossary
        assert "zh" in available
        zh_glossaries = available["zh"]
        assert "idioms_expressions_zh" in zh_glossaries

    def test_get_metadata_for_new_glossaries(self, glossary_manager):
        """Test metadata retrieval for new glossaries."""
        test_cases = [
            ("en", "mqm_error_taxonomy", 47),
            ("en", "translation_metrics", 32),
            ("en", "transformer_nlp_terms", 58),
            ("en", "llm_terminology", 45),
            ("zh", "idioms_expressions_zh", 120),
        ]

        for lang, gtype, expected_terms in test_cases:
            glossary_manager.load_glossary(lang, gtype)
            metadata = glossary_manager.get_metadata(lang, gtype)

            assert metadata is not None
            assert metadata.total_terms == expected_terms
            assert metadata.language == lang

    def test_search_in_new_glossaries(self, glossary_manager):
        """Test search functionality works with new glossaries."""
        # Search for hallucination in LLM glossary
        results = glossary_manager.search_terms("hallucination", language="en")
        assert len(results) > 0

        # Search for BLEU in metrics glossary
        results = glossary_manager.search_terms("BLEU", language="en")
        assert len(results) > 0

        # Search for attention in transformer glossary
        results = glossary_manager.search_terms("attention", language="en")
        assert len(results) > 0

        # Search for 成语 in Chinese glossary
        results = glossary_manager.search_terms("成语", language="zh")
        assert len(results) > 0

    def test_load_all_for_language_includes_new(self, glossary_manager):
        """Test load_all_for_language includes new glossaries."""
        # Load all English glossaries
        en_glossaries = glossary_manager.load_all_for_language("en")

        assert "mqm_error_taxonomy" in en_glossaries
        assert "translation_metrics" in en_glossaries
        assert "transformer_nlp_terms" in en_glossaries
        assert "llm_terminology" in en_glossaries

        # Load all Chinese glossaries
        zh_glossaries = glossary_manager.load_all_for_language("zh")
        assert "idioms_expressions_zh" in zh_glossaries


class TestErrorHandling:
    """Tests for error handling with glossaries."""

    def test_load_nonexistent_glossary(self, glossary_manager):
        """Test loading nonexistent glossary raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            glossary_manager.load_glossary("en", "nonexistent_glossary")

    def test_load_invalid_language(self, glossary_manager):
        """Test loading from invalid language raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            glossary_manager.load_glossary("invalid", "mqm_core")
