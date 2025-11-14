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

"""Unit tests for glossary system."""

import json
from pathlib import Path

import pytest

from kttc.core import Glossary, GlossaryManager, TermEntry


class TestTermEntry:
    """Tests for TermEntry dataclass."""

    def test_matches_case_insensitive(self):
        """Test case-insensitive matching."""
        term = TermEntry(
            source="API",
            target="API",
            source_lang="en",
            target_lang="ru",
            case_sensitive=False,
        )

        assert term.matches("The API endpoint")
        assert term.matches("the api endpoint")
        assert term.matches("using API")

    def test_matches_case_sensitive(self):
        """Test case-sensitive matching."""
        term = TermEntry(
            source="API",
            target="API",
            source_lang="en",
            target_lang="ru",
            case_sensitive=True,
        )

        assert term.matches("The API endpoint")
        assert not term.matches("the api endpoint")

    def test_matches_not_found(self):
        """Test matching when term not in text."""
        term = TermEntry(
            source="Docker",
            target="Docker",
            source_lang="en",
            target_lang="ru",
        )

        assert not term.matches("The API endpoint")


class TestGlossary:
    """Tests for Glossary class."""

    @pytest.fixture
    def sample_entries(self):
        """Create sample term entries."""
        return [
            TermEntry("API", "API", "en", "ru", "technical", do_not_translate=True),
            TermEntry("Docker", "Docker", "en", "ru", "devops", do_not_translate=True),
            TermEntry("endpoint", "эндпоинт", "en", "ru", "technical"),
            TermEntry("database", "база данных", "en", "ru", "technical"),
        ]

    def test_init(self, sample_entries):
        """Test glossary initialization."""
        glossary = Glossary(sample_entries, name="test")

        assert glossary.name == "test"
        assert len(glossary.entries) == 4
        assert len(glossary.by_source) > 0
        assert len(glossary.by_lang_pair) > 0

    def test_lookup_found(self, sample_entries):
        """Test successful term lookup."""
        glossary = Glossary(sample_entries)

        entry = glossary.lookup("API", "en", "ru")

        assert entry is not None
        assert entry.source == "API"
        assert entry.target == "API"

    def test_lookup_case_insensitive(self, sample_entries):
        """Test case-insensitive lookup."""
        glossary = Glossary(sample_entries)

        entry = glossary.lookup("api", "en", "ru")  # lowercase

        assert entry is not None
        assert entry.source == "API"

    def test_lookup_not_found(self, sample_entries):
        """Test lookup of non-existent term."""
        glossary = Glossary(sample_entries)

        entry = glossary.lookup("nonexistent", "en", "ru")

        assert entry is None

    def test_lookup_wrong_language_pair(self, sample_entries):
        """Test lookup with wrong language pair."""
        glossary = Glossary(sample_entries)

        entry = glossary.lookup("API", "en", "es")  # Wrong target language

        assert entry is None

    def test_get_all_for_language_pair(self, sample_entries):
        """Test getting all terms for language pair."""
        glossary = Glossary(sample_entries)

        terms = glossary.get_all_for_language_pair("en", "ru")

        assert len(terms) == 4
        assert all(t.source_lang == "en" for t in terms)
        assert all(t.target_lang == "ru" for t in terms)

    def test_get_all_for_language_pair_empty(self, sample_entries):
        """Test getting terms for non-existent language pair."""
        glossary = Glossary(sample_entries)

        terms = glossary.get_all_for_language_pair("fr", "de")

        assert len(terms) == 0

    def test_find_in_text(self, sample_entries):
        """Test finding terms in text."""
        glossary = Glossary(sample_entries)

        text = "The API endpoint returns database records"
        found = glossary.find_in_text(text, "en", "ru")

        assert len(found) == 3  # API, endpoint, database
        sources = {t.source for t in found}
        assert "API" in sources
        assert "endpoint" in sources
        assert "database" in sources

    def test_find_in_text_no_matches(self, sample_entries):
        """Test finding terms when none match."""
        glossary = Glossary(sample_entries)

        text = "Hello world"
        found = glossary.find_in_text(text, "en", "ru")

        assert len(found) == 0

    def test_from_json(self, tmp_path):
        """Test loading glossary from JSON."""
        json_file = tmp_path / "test.json"
        data = {
            "metadata": {"name": "Test", "version": "1.0.0"},
            "entries": [
                {
                    "source": "API",
                    "target": "API",
                    "source_lang": "en",
                    "target_lang": "ru",
                    "domain": "technical",
                    "do_not_translate": True,
                    "case_sensitive": True,
                    "notes": None,
                    "definition": None,
                    "context": None,
                    "pos": None,
                    "variants": None,
                    "status": "approved",
                }
            ],
        }
        json_file.write_text(json.dumps(data))

        glossary = Glossary.from_json(json_file)

        assert glossary.name == "test"
        assert len(glossary.entries) == 1
        assert glossary.metadata.name == "Test"
        assert glossary.metadata.version == "1.0.0"

    def test_from_json_file_not_found(self):
        """Test loading from non-existent JSON file."""
        with pytest.raises(FileNotFoundError):
            Glossary.from_json(Path("/nonexistent/file.json"))

    def test_from_csv(self, tmp_path):
        """Test loading glossary from CSV."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text(
            "source,target,source_lang,target_lang,domain,case_sensitive,do_not_translate,notes\n"
            "API,API,en,ru,technical,true,true,Keep uppercase\n"
            "endpoint,эндпоинт,en,ru,technical,false,false,\n"
        )

        glossary = Glossary.from_csv(csv_file)

        assert glossary.name == "test"
        assert len(glossary.entries) == 2
        assert glossary.entries[0].source == "API"
        assert glossary.entries[0].case_sensitive is True
        assert glossary.entries[1].source == "endpoint"

    def test_to_json(self, tmp_path, sample_entries):
        """Test saving glossary to JSON."""
        glossary = Glossary(sample_entries, name="test")
        json_file = tmp_path / "output.json"

        glossary.to_json(json_file)

        assert json_file.exists()

        # Load and verify
        with open(json_file) as f:
            data = json.load(f)

        assert "entries" in data
        assert len(data["entries"]) == 4

    def test_to_csv(self, tmp_path, sample_entries):
        """Test saving glossary to CSV."""
        glossary = Glossary(sample_entries, name="test")
        csv_file = tmp_path / "output.csv"

        glossary.to_csv(csv_file)

        assert csv_file.exists()

        # Verify can be loaded back
        loaded = Glossary.from_csv(csv_file)
        assert len(loaded.entries) == 4


class TestGlossaryManager:
    """Tests for GlossaryManager."""

    @pytest.fixture
    def temp_glossary_dir(self, tmp_path):
        """Create temporary glossary directory."""
        glossary_dir = tmp_path / "glossaries"
        glossary_dir.mkdir()

        # Create test glossary
        glossary_data = {
            "entries": [
                {
                    "source": "API",
                    "target": "API",
                    "source_lang": "en",
                    "target_lang": "ru",
                    "domain": None,
                    "definition": None,
                    "context": None,
                    "case_sensitive": True,
                    "do_not_translate": True,
                    "pos": None,
                    "variants": None,
                    "notes": None,
                    "status": "approved",
                }
            ]
        }

        (glossary_dir / "test.json").write_text(json.dumps(glossary_data))

        return glossary_dir

    def test_load_glossary(self, temp_glossary_dir, monkeypatch):
        """Test loading single glossary."""
        # Mock project directory
        monkeypatch.setattr(Path, "cwd", lambda: temp_glossary_dir.parent)

        manager = GlossaryManager()
        manager.project_glossary_dir = temp_glossary_dir

        glossary = manager.load_glossary("test")

        assert glossary.name == "test"
        assert len(glossary.entries) == 1

    def test_load_glossary_not_found(self):
        """Test loading non-existent glossary."""
        manager = GlossaryManager()

        with pytest.raises(FileNotFoundError, match="not found"):
            manager.load_glossary("nonexistent")

    def test_load_multiple(self, temp_glossary_dir, monkeypatch):
        """Test loading multiple glossaries."""
        monkeypatch.setattr(Path, "cwd", lambda: temp_glossary_dir.parent)

        manager = GlossaryManager()
        manager.project_glossary_dir = temp_glossary_dir

        manager.load_multiple(["test"])

        assert len(manager.glossaries) == 1

    def test_lookup_single_glossary(self, temp_glossary_dir, monkeypatch):
        """Test lookup across loaded glossaries."""
        monkeypatch.setattr(Path, "cwd", lambda: temp_glossary_dir.parent)

        manager = GlossaryManager()
        manager.project_glossary_dir = temp_glossary_dir
        manager.load_multiple(["test"])

        entry = manager.lookup("API", "en", "ru")

        assert entry is not None
        assert entry.source == "API"

    def test_lookup_not_found(self, temp_glossary_dir, monkeypatch):
        """Test lookup of non-existent term."""
        monkeypatch.setattr(Path, "cwd", lambda: temp_glossary_dir.parent)

        manager = GlossaryManager()
        manager.project_glossary_dir = temp_glossary_dir
        manager.load_multiple(["test"])

        entry = manager.lookup("nonexistent", "en", "ru")

        assert entry is None

    def test_get_all_terms(self, temp_glossary_dir, monkeypatch):
        """Test getting all terms for language pair."""
        monkeypatch.setattr(Path, "cwd", lambda: temp_glossary_dir.parent)

        manager = GlossaryManager()
        manager.project_glossary_dir = temp_glossary_dir
        manager.load_multiple(["test"])

        terms = manager.get_all_terms("en", "ru")

        assert len(terms) == 1

    def test_find_in_text(self, temp_glossary_dir, monkeypatch):
        """Test finding terms in text."""
        monkeypatch.setattr(Path, "cwd", lambda: temp_glossary_dir.parent)

        manager = GlossaryManager()
        manager.project_glossary_dir = temp_glossary_dir
        manager.load_multiple(["test"])

        found = manager.find_in_text("The API endpoint", "en", "ru")

        assert len(found) == 1
        assert found[0].source == "API"

    def test_merge_glossaries(self, temp_glossary_dir, monkeypatch):
        """Test merging multiple glossaries."""
        monkeypatch.setattr(Path, "cwd", lambda: temp_glossary_dir.parent)

        # Create second glossary
        glossary_data = {
            "entries": [
                {
                    "source": "Docker",
                    "target": "Docker",
                    "source_lang": "en",
                    "target_lang": "ru",
                    "domain": None,
                    "definition": None,
                    "context": None,
                    "case_sensitive": True,
                    "do_not_translate": True,
                    "pos": None,
                    "variants": None,
                    "notes": None,
                    "status": "approved",
                }
            ]
        }
        (temp_glossary_dir / "test2.json").write_text(json.dumps(glossary_data))

        manager = GlossaryManager()
        manager.project_glossary_dir = temp_glossary_dir

        merged = manager.merge_glossaries(["test", "test2"], "combined")

        assert len(merged.entries) == 2
        sources = {e.source for e in merged.entries}
        assert "API" in sources
        assert "Docker" in sources


@pytest.mark.integration
class TestGlossaryIntegration:
    """Integration tests using example glossaries."""

    def test_load_base_glossary(self):
        """Test loading base glossary."""
        glossary_path = Path("glossaries/base.json")

        if not glossary_path.exists():
            pytest.skip("Base glossary not found")

        glossary = Glossary.from_json(glossary_path)

        assert len(glossary.entries) > 0
        assert glossary.metadata.name == "Base Terminology"

    def test_load_technical_glossary(self):
        """Test loading technical glossary."""
        glossary_path = Path("glossaries/technical.json")

        if not glossary_path.exists():
            pytest.skip("Technical glossary not found")

        glossary = Glossary.from_json(glossary_path)

        assert len(glossary.entries) > 0
