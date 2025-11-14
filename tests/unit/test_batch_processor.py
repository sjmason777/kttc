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

"""Unit tests for batch processing functionality."""

import json
from pathlib import Path

import pytest

from kttc.core import BatchFileParser, BatchGrouper, BatchTranslation


class TestBatchTranslation:
    """Tests for BatchTranslation dataclass."""

    def test_to_task(self):
        """Test conversion to TranslationTask."""
        batch_trans = BatchTranslation(
            source_text="Hello, world!",
            translation="¡Hola, mundo!",
            source_lang="en",
            target_lang="es",
            domain="general",
        )

        task = batch_trans.to_task()

        assert task.source_text == "Hello, world!"
        assert task.translation == "¡Hola, mundo!"
        assert task.source_lang == "en"
        assert task.target_lang == "es"


class TestBatchFileParser:
    """Tests for BatchFileParser."""

    def test_parse_csv_valid(self, tmp_path):
        """Test parsing valid CSV file."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text(
            "source,translation,source_lang,target_lang,domain\n"
            '"Hello, world!","¡Hola, mundo!",en,es,general\n'
            "Goodbye,Adiós,en,es,general\n"
        )

        translations = BatchFileParser.parse_csv(csv_file)

        assert len(translations) == 2
        assert translations[0].source_text == "Hello, world!"
        assert translations[0].translation == "¡Hola, mundo!"
        assert translations[0].source_lang == "en"
        assert translations[0].target_lang == "es"
        assert translations[0].domain == "general"

    def test_parse_csv_missing_columns(self, tmp_path):
        """Test parsing CSV with missing required columns."""
        csv_file = tmp_path / "invalid.csv"
        csv_file.write_text("source,translation\n" "Hello,Hola\n")

        with pytest.raises(ValueError, match="missing required columns"):
            BatchFileParser.parse_csv(csv_file)

    def test_parse_csv_with_context(self, tmp_path):
        """Test parsing CSV with JSON context field."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text(
            "source,translation,source_lang,target_lang,context\n"
            'API,API,en,ru,"{""complexity"": ""simple""}"\n'
        )

        translations = BatchFileParser.parse_csv(csv_file)

        assert len(translations) == 1
        assert translations[0].context == {"complexity": "simple"}

    def test_parse_json_valid(self, tmp_path):
        """Test parsing valid JSON file."""
        json_file = tmp_path / "test.json"
        data = [
            {
                "source": "Hello, world!",
                "translation": "¡Hola, mundo!",
                "source_lang": "en",
                "target_lang": "es",
                "domain": "general",
            },
            {
                "source": "Goodbye",
                "translation": "Adiós",
                "source_lang": "en",
                "target_lang": "es",
            },
        ]
        json_file.write_text(json.dumps(data))

        translations = BatchFileParser.parse_json(json_file)

        assert len(translations) == 2
        assert translations[0].source_text == "Hello, world!"
        assert translations[1].domain is None

    def test_parse_json_invalid_format(self, tmp_path):
        """Test parsing JSON with invalid format (not an array)."""
        json_file = tmp_path / "invalid.json"
        json_file.write_text('{"source": "Hello"}')

        with pytest.raises(ValueError, match="must be an array"):
            BatchFileParser.parse_json(json_file)

    def test_parse_json_missing_fields(self, tmp_path):
        """Test parsing JSON with missing required fields."""
        json_file = tmp_path / "invalid.json"
        data = [{"source": "Hello", "translation": "Hola"}]  # Missing lang fields
        json_file.write_text(json.dumps(data))

        with pytest.raises(ValueError, match="Missing required fields"):
            BatchFileParser.parse_json(json_file)

    def test_parse_jsonl_valid(self, tmp_path):
        """Test parsing valid JSONL file."""
        jsonl_file = tmp_path / "test.jsonl"
        jsonl_file.write_text(
            '{"source": "Hello", "translation": "Hola", "source_lang": "en", "target_lang": "es"}\n'
            '{"source": "Goodbye", "translation": "Adiós", "source_lang": "en", "target_lang": "es"}\n'
        )

        translations = BatchFileParser.parse_jsonl(jsonl_file)

        assert len(translations) == 2
        assert translations[0].source_text == "Hello"
        assert translations[1].source_text == "Goodbye"

    def test_parse_jsonl_skip_empty_lines(self, tmp_path):
        """Test JSONL parser skips empty lines."""
        jsonl_file = tmp_path / "test.jsonl"
        jsonl_file.write_text(
            '{"source": "Hello", "translation": "Hola", "source_lang": "en", "target_lang": "es"}\n'
            "\n"  # Empty line
            '{"source": "Goodbye", "translation": "Adiós", "source_lang": "en", "target_lang": "es"}\n'
        )

        translations = BatchFileParser.parse_jsonl(jsonl_file)

        assert len(translations) == 2

    def test_parse_jsonl_invalid_json(self, tmp_path):
        """Test JSONL parser handles invalid JSON lines."""
        jsonl_file = tmp_path / "invalid.jsonl"
        jsonl_file.write_text('{"source": "Hello"')  # Invalid JSON

        with pytest.raises(ValueError, match="Invalid JSON"):
            BatchFileParser.parse_jsonl(jsonl_file)

    def test_parse_auto_detect_csv(self, tmp_path):
        """Test auto-detection of CSV format."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("source,translation,source_lang,target_lang\n" "Hello,Hola,en,es\n")

        translations = BatchFileParser.parse(csv_file)

        assert len(translations) == 1
        assert translations[0].source_text == "Hello"

    def test_parse_auto_detect_json(self, tmp_path):
        """Test auto-detection of JSON format."""
        json_file = tmp_path / "test.json"
        data = [
            {"source": "Hello", "translation": "Hola", "source_lang": "en", "target_lang": "es"}
        ]
        json_file.write_text(json.dumps(data))

        translations = BatchFileParser.parse(json_file)

        assert len(translations) == 1

    def test_parse_unsupported_format(self, tmp_path):
        """Test auto-detection rejects unsupported format."""
        xml_file = tmp_path / "test.xml"
        xml_file.write_text("<root></root>")

        with pytest.raises(ValueError, match="Unsupported file format"):
            BatchFileParser.parse(xml_file)

    def test_parse_file_not_found(self):
        """Test parsing non-existent file."""
        with pytest.raises(FileNotFoundError):
            BatchFileParser.parse_csv(Path("/nonexistent/file.csv"))


class TestBatchGrouper:
    """Tests for BatchGrouper."""

    @pytest.fixture
    def sample_translations(self):
        """Create sample translations for testing."""
        return [
            BatchTranslation("Hello", "Hola", "en", "es", "general"),
            BatchTranslation("Goodbye", "Adiós", "en", "es", "general"),
            BatchTranslation("API", "API", "en", "ru", "technical"),
            BatchTranslation("Hello", "Привет", "en", "ru", "general"),
            BatchTranslation("Database", "База данных", "en", "ru", "technical"),
        ]

    def test_group_by_language_pair(self, sample_translations):
        """Test grouping by language pair."""
        groups = BatchGrouper.group_by_language_pair(sample_translations)

        assert len(groups) == 2
        assert ("en", "es") in groups
        assert ("en", "ru") in groups
        assert len(groups[("en", "es")]) == 2
        assert len(groups[("en", "ru")]) == 3

    def test_group_by_domain(self, sample_translations):
        """Test grouping by domain."""
        groups = BatchGrouper.group_by_domain(sample_translations)

        assert len(groups) == 2
        assert "general" in groups
        assert "technical" in groups
        assert len(groups["general"]) == 3
        assert len(groups["technical"]) == 2

    def test_group_by_domain_default(self):
        """Test grouping with None domain defaults to 'general'."""
        translations = [
            BatchTranslation("Hello", "Hola", "en", "es", None),
        ]

        groups = BatchGrouper.group_by_domain(translations)

        assert "general" in groups
        assert len(groups["general"]) == 1

    def test_create_batches(self, sample_translations):
        """Test creating batches of specified size."""
        batches = BatchGrouper.create_batches(sample_translations, batch_size=2)

        assert len(batches) == 3  # 5 translations / 2 per batch = 3 batches
        assert len(batches[0]) == 2
        assert len(batches[1]) == 2
        assert len(batches[2]) == 1  # Last batch has remainder

    def test_create_batches_exact_fit(self):
        """Test creating batches when total divides evenly."""
        translations = [
            BatchTranslation("T1", "T1", "en", "es"),
            BatchTranslation("T2", "T2", "en", "es"),
            BatchTranslation("T3", "T3", "en", "es"),
            BatchTranslation("T4", "T4", "en", "es"),
        ]

        batches = BatchGrouper.create_batches(translations, batch_size=2)

        assert len(batches) == 2
        assert all(len(batch) == 2 for batch in batches)

    def test_create_batches_single_batch(self):
        """Test creating single batch when size exceeds total."""
        translations = [
            BatchTranslation("T1", "T1", "en", "es"),
            BatchTranslation("T2", "T2", "en", "es"),
        ]

        batches = BatchGrouper.create_batches(translations, batch_size=10)

        assert len(batches) == 1
        assert len(batches[0]) == 2


@pytest.mark.integration
class TestBatchProcessorIntegration:
    """Integration tests using example files."""

    def test_parse_example_csv(self):
        """Test parsing example CSV file."""
        csv_path = Path("examples/batch/translations.csv")

        if not csv_path.exists():
            pytest.skip("Example CSV not found")

        translations = BatchFileParser.parse_csv(csv_path)

        assert len(translations) > 0
        assert all(hasattr(t, "source_text") for t in translations)
        assert all(hasattr(t, "translation") for t in translations)

    def test_parse_example_json(self):
        """Test parsing example JSON file."""
        json_path = Path("examples/batch/translations.json")

        if not json_path.exists():
            pytest.skip("Example JSON not found")

        translations = BatchFileParser.parse_json(json_path)

        assert len(translations) > 0

    def test_parse_example_jsonl(self):
        """Test parsing example JSONL file."""
        jsonl_path = Path("examples/batch/translations.jsonl")

        if not jsonl_path.exists():
            pytest.skip("Example JSONL not found")

        translations = BatchFileParser.parse_jsonl(jsonl_path)

        assert len(translations) > 0
