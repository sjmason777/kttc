"""Unit tests for batch processor module.

Tests batch file parsing for CSV, JSON, and JSONL formats.
"""

import json
from pathlib import Path

import pytest

from kttc.core.batch_processor import BatchFileParser, BatchGrouper, BatchTranslation


@pytest.mark.unit
class TestBatchTranslation:
    """Test BatchTranslation dataclass."""

    def test_basic_creation(self) -> None:
        """Test basic BatchTranslation creation."""
        bt = BatchTranslation(
            source_text="Hello",
            translation="Hola",
            source_lang="en",
            target_lang="es",
        )

        assert bt.source_text == "Hello"
        assert bt.translation == "Hola"
        assert bt.source_lang == "en"
        assert bt.target_lang == "es"
        assert bt.domain is None
        assert bt.context is None
        assert bt.metadata is None

    def test_creation_with_all_fields(self) -> None:
        """Test BatchTranslation with all fields."""
        bt = BatchTranslation(
            source_text="Hello",
            translation="Hola",
            source_lang="en",
            target_lang="es",
            domain="technical",
            context={"complexity": "simple"},
            metadata={"file": "test.csv", "row": 1},
        )

        assert bt.domain == "technical"
        assert bt.context == {"complexity": "simple"}
        assert bt.metadata["file"] == "test.csv"

    def test_to_task_conversion(self) -> None:
        """Test conversion to TranslationTask."""
        bt = BatchTranslation(
            source_text="Hello world",
            translation="Hola mundo",
            source_lang="en",
            target_lang="es",
            context={"key": "value"},
        )

        task = bt.to_task()

        assert task.source_text == "Hello world"
        assert task.translation == "Hola mundo"
        assert task.source_lang == "en"
        assert task.target_lang == "es"
        assert task.context == {"key": "value"}


@pytest.mark.unit
class TestBatchFileParserCSV:
    """Test CSV parsing."""

    def test_parse_valid_csv(self, tmp_path: Path) -> None:
        """Test parsing valid CSV file."""
        csv_content = """source,translation,source_lang,target_lang,domain
Hello,Hola,en,es,general
World,Mundo,en,es,general
"""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text(csv_content)

        translations = BatchFileParser.parse_csv(csv_file)

        assert len(translations) == 2
        assert translations[0].source_text == "Hello"
        assert translations[0].translation == "Hola"
        assert translations[1].source_text == "World"

    def test_parse_csv_with_context(self, tmp_path: Path) -> None:
        """Test parsing CSV with JSON context field."""
        csv_content = """source,translation,source_lang,target_lang,context
Hello,Hola,en,es,"{""key"": ""value""}"
"""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text(csv_content)

        translations = BatchFileParser.parse_csv(csv_file)

        assert len(translations) == 1
        assert translations[0].context == {"key": "value"}

    def test_parse_csv_invalid_context(self, tmp_path: Path) -> None:
        """Test parsing CSV with invalid JSON context (should warn, not fail)."""
        csv_content = """source,translation,source_lang,target_lang,context
Hello,Hola,en,es,not valid json
"""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text(csv_content)

        translations = BatchFileParser.parse_csv(csv_file)

        assert len(translations) == 1
        assert translations[0].context is None  # Should be None, not fail

    def test_parse_csv_missing_columns(self, tmp_path: Path) -> None:
        """Test parsing CSV with missing required columns."""
        csv_content = """source,translation
Hello,Hola
"""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text(csv_content)

        with pytest.raises(ValueError, match="missing required columns"):
            BatchFileParser.parse_csv(csv_file)

    def test_parse_csv_file_not_found(self, tmp_path: Path) -> None:
        """Test parsing nonexistent CSV file."""
        with pytest.raises(FileNotFoundError):
            BatchFileParser.parse_csv(tmp_path / "nonexistent.csv")

    def test_parse_csv_metadata_included(self, tmp_path: Path) -> None:
        """Test that metadata includes row number and file path."""
        csv_content = """source,translation,source_lang,target_lang
Hello,Hola,en,es
"""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text(csv_content)

        translations = BatchFileParser.parse_csv(csv_file)

        assert translations[0].metadata["row_number"] == 2  # Header is row 1
        assert str(csv_file) in translations[0].metadata["file"]


@pytest.mark.unit
class TestBatchFileParserJSON:
    """Test JSON parsing."""

    def test_parse_valid_json(self, tmp_path: Path) -> None:
        """Test parsing valid JSON file."""
        data = [
            {
                "source": "Hello",
                "translation": "Hola",
                "source_lang": "en",
                "target_lang": "es",
            },
            {
                "source": "World",
                "translation": "Mundo",
                "source_lang": "en",
                "target_lang": "es",
            },
        ]
        json_file = tmp_path / "test.json"
        json_file.write_text(json.dumps(data))

        translations = BatchFileParser.parse_json(json_file)

        assert len(translations) == 2
        assert translations[0].source_text == "Hello"
        assert translations[1].source_text == "World"

    def test_parse_json_with_domain_and_context(self, tmp_path: Path) -> None:
        """Test parsing JSON with optional fields."""
        data = [
            {
                "source": "API",
                "translation": "API",
                "source_lang": "en",
                "target_lang": "es",
                "domain": "technical",
                "context": {"type": "acronym"},
            }
        ]
        json_file = tmp_path / "test.json"
        json_file.write_text(json.dumps(data))

        translations = BatchFileParser.parse_json(json_file)

        assert translations[0].domain == "technical"
        assert translations[0].context == {"type": "acronym"}

    def test_parse_json_not_array(self, tmp_path: Path) -> None:
        """Test parsing JSON that is not an array."""
        json_file = tmp_path / "test.json"
        json_file.write_text('{"source": "Hello"}')

        with pytest.raises(ValueError, match="must be an array"):
            BatchFileParser.parse_json(json_file)

    def test_parse_json_missing_fields(self, tmp_path: Path) -> None:
        """Test parsing JSON with missing required fields."""
        data = [{"source": "Hello"}]  # Missing translation, langs
        json_file = tmp_path / "test.json"
        json_file.write_text(json.dumps(data))

        with pytest.raises(ValueError, match="Missing required fields"):
            BatchFileParser.parse_json(json_file)

    def test_parse_json_file_not_found(self, tmp_path: Path) -> None:
        """Test parsing nonexistent JSON file."""
        with pytest.raises(FileNotFoundError):
            BatchFileParser.parse_json(tmp_path / "nonexistent.json")

    def test_parse_json_requires_source_key(self, tmp_path: Path) -> None:
        """Test parsing JSON requires 'source' key."""
        data = [
            {
                "source": "Hello",
                "translation": "Hola",
                "source_lang": "en",
                "target_lang": "es",
            }
        ]
        json_file = tmp_path / "test.json"
        json_file.write_text(json.dumps(data))

        translations = BatchFileParser.parse_json(json_file)

        assert translations[0].source_text == "Hello"


@pytest.mark.unit
class TestBatchFileParserJSONL:
    """Test JSONL parsing."""

    def test_parse_valid_jsonl(self, tmp_path: Path) -> None:
        """Test parsing valid JSONL file."""
        lines = [
            '{"source": "Hello", "translation": "Hola", "source_lang": "en", "target_lang": "es"}',
            '{"source": "World", "translation": "Mundo", "source_lang": "en", "target_lang": "es"}',
        ]
        jsonl_file = tmp_path / "test.jsonl"
        jsonl_file.write_text("\n".join(lines))

        translations = BatchFileParser.parse_jsonl(jsonl_file)

        assert len(translations) == 2
        assert translations[0].source_text == "Hello"
        assert translations[1].source_text == "World"

    def test_parse_jsonl_skips_empty_lines(self, tmp_path: Path) -> None:
        """Test parsing JSONL skips empty lines."""
        lines = [
            '{"source": "Hello", "translation": "Hola", "source_lang": "en", "target_lang": "es"}',
            "",
            '{"source": "World", "translation": "Mundo", "source_lang": "en", "target_lang": "es"}',
        ]
        jsonl_file = tmp_path / "test.jsonl"
        jsonl_file.write_text("\n".join(lines))

        translations = BatchFileParser.parse_jsonl(jsonl_file)

        assert len(translations) == 2

    def test_parse_jsonl_file_not_found(self, tmp_path: Path) -> None:
        """Test parsing nonexistent JSONL file."""
        with pytest.raises(FileNotFoundError):
            BatchFileParser.parse_jsonl(tmp_path / "nonexistent.jsonl")


@pytest.mark.unit
class TestBatchFileParserAutoDetect:
    """Test automatic format detection."""

    def test_parse_auto_detects_csv(self, tmp_path: Path) -> None:
        """Test parse() auto-detects CSV format."""
        csv_content = """source,translation,source_lang,target_lang
Hello,Hola,en,es
"""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text(csv_content)

        translations = BatchFileParser.parse(csv_file)

        assert len(translations) == 1

    def test_parse_auto_detects_json(self, tmp_path: Path) -> None:
        """Test parse() auto-detects JSON format."""
        data = [
            {"source": "Hello", "translation": "Hola", "source_lang": "en", "target_lang": "es"}
        ]
        json_file = tmp_path / "test.json"
        json_file.write_text(json.dumps(data))

        translations = BatchFileParser.parse(json_file)

        assert len(translations) == 1

    def test_parse_auto_detects_jsonl(self, tmp_path: Path) -> None:
        """Test parse() auto-detects JSONL format."""
        line = (
            '{"source": "Hello", "translation": "Hola", "source_lang": "en", "target_lang": "es"}'
        )
        jsonl_file = tmp_path / "test.jsonl"
        jsonl_file.write_text(line)

        translations = BatchFileParser.parse(jsonl_file)

        assert len(translations) == 1

    def test_parse_unsupported_format(self, tmp_path: Path) -> None:
        """Test parse() raises for unsupported format."""
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("plain text")

        with pytest.raises(ValueError, match="Unsupported file format"):
            BatchFileParser.parse(txt_file)


@pytest.mark.unit
class TestBatchGrouper:
    """Test batch grouping utilities."""

    def test_group_by_language_pair(self) -> None:
        """Test grouping translations by language pair."""
        translations = [
            BatchTranslation("Hello", "Hola", "en", "es"),
            BatchTranslation("World", "Mundo", "en", "es"),
            BatchTranslation("Bonjour", "Hello", "fr", "en"),
        ]

        groups = BatchGrouper.group_by_language_pair(translations)

        assert len(groups) == 2
        assert len(groups[("en", "es")]) == 2
        assert len(groups[("fr", "en")]) == 1

    def test_group_by_domain(self) -> None:
        """Test grouping translations by domain."""
        translations = [
            BatchTranslation("Hello", "Hola", "en", "es", domain="general"),
            BatchTranslation("API", "API", "en", "es", domain="technical"),
            BatchTranslation("World", "Mundo", "en", "es", domain="general"),
        ]

        groups = BatchGrouper.group_by_domain(translations)

        assert len(groups) == 2
        assert len(groups["general"]) == 2
        assert len(groups["technical"]) == 1

    def test_group_by_domain_none_defaults_to_general(self) -> None:
        """Test that None domains default to 'general' key."""
        translations = [
            BatchTranslation("Hello", "Hola", "en", "es"),  # No domain = None
            BatchTranslation("World", "Mundo", "en", "es"),  # No domain = None
        ]

        groups = BatchGrouper.group_by_domain(translations)

        # None domains should be grouped under "general"
        assert "general" in groups
        assert len(groups["general"]) == 2

    def test_group_empty_list(self) -> None:
        """Test grouping empty list."""
        groups = BatchGrouper.group_by_language_pair([])
        assert len(groups) == 0

        groups = BatchGrouper.group_by_domain([])
        assert len(groups) == 0
