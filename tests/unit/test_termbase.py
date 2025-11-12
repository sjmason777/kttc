"""Tests for TerminologyBase module."""

import tempfile
from pathlib import Path

import pytest

from kttc.memory.termbase import TermEntry, TerminologyBase, TermViolation

pytestmark = pytest.mark.asyncio


class TestTermEntry:
    """Tests for TermEntry model."""

    def test_create_entry(self):
        """Test creating a term entry."""
        entry = TermEntry(
            source_term="API",
            target_term="interfaz de programación de aplicaciones",
            source_lang="en",
            target_lang="es",
            domain="technical",
            definition="Application Programming Interface",
            usage_note="Always translate in full form",
        )

        assert entry.source_term == "API"
        assert entry.target_term == "interfaz de programación de aplicaciones"
        assert entry.source_lang == "en"
        assert entry.target_lang == "es"
        assert entry.domain == "technical"
        assert entry.definition == "Application Programming Interface"
        assert entry.usage_note == "Always translate in full form"

    def test_entry_with_minimal_fields(self):
        """Test entry with only required fields."""
        entry = TermEntry(
            source_term="Test",
            target_term="Prueba",
            source_lang="en",
            target_lang="es",
        )

        assert entry.id is None
        assert entry.domain is None
        assert entry.definition is None
        assert entry.usage_note is None
        assert entry.created_at is None


class TestTermViolation:
    """Tests for TermViolation model."""

    def test_create_violation(self):
        """Test creating a term violation."""
        violation = TermViolation(
            source_term="API",
            expected_terms=["API", "interfaz de programación"],
            found_in_translation=False,
            severity="major",
        )

        assert violation.source_term == "API"
        assert violation.expected_terms == ["API", "interfaz de programación"]
        assert violation.found_in_translation is False
        assert violation.severity == "major"


class TestTerminologyBase:
    """Tests for TerminologyBase class."""

    @pytest.fixture
    async def termbase(self):
        """Create temporary terminology database."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        tb = TerminologyBase(db_path)
        await tb.initialize()

        yield tb

        await tb.cleanup()
        Path(db_path).unlink(missing_ok=True)

    async def test_initialization(self, termbase):
        """Test termbase initialization."""
        assert termbase._initialized is True
        assert termbase.db is not None

    async def test_add_term(self, termbase):
        """Test adding term to termbase."""
        term_id = await termbase.add_term(
            source_term="API",
            target_term="interfaz de programación de aplicaciones",
            source_lang="en",
            target_lang="es",
            domain="technical",
            definition="Application Programming Interface",
            usage_note="Use full form in formal documents",
        )

        assert term_id > 0

    async def test_add_duplicate_term_updates(self, termbase):
        """Test that adding duplicate updates existing entry."""
        # Add first time
        id1 = await termbase.add_term(
            source_term="API",
            target_term="API",
            source_lang="en",
            target_lang="es",
            domain="technical",
        )

        # Add again with different translation
        id2 = await termbase.add_term(
            source_term="API",
            target_term="interfaz de programación de aplicaciones",
            source_lang="en",
            target_lang="es",
            domain="technical",
        )

        # Should update, not create new
        assert id1 == id2

    async def test_add_term_case_insensitive(self, termbase):
        """Test that term matching is case-insensitive."""
        # Add with lowercase
        id1 = await termbase.add_term(
            source_term="api",
            target_term="API",
            source_lang="en",
            target_lang="es",
            domain="technical",
        )

        # Add with uppercase should update
        id2 = await termbase.add_term(
            source_term="API",
            target_term="interfaz",
            source_lang="en",
            target_lang="es",
            domain="technical",
        )

        assert id1 == id2

    async def test_add_term_different_domains(self, termbase):
        """Test that same term can exist in different domains."""
        # Add in technical domain
        id1 = await termbase.add_term(
            source_term="cell",
            target_term="célula",
            source_lang="en",
            target_lang="es",
            domain="biology",
        )

        # Add in telecom domain (should be different entry)
        id2 = await termbase.add_term(
            source_term="cell",
            target_term="celda",
            source_lang="en",
            target_lang="es",
            domain="telecom",
        )

        assert id1 != id2

    async def test_add_term_null_domain_handling(self, termbase):
        """Test that terms without domain are handled correctly."""
        # Add term without domain
        id1 = await termbase.add_term(
            source_term="hello",
            target_term="hola",
            source_lang="en",
            target_lang="es",
            domain=None,
        )

        # Add again without domain should update
        id2 = await termbase.add_term(
            source_term="hello",
            target_term="hola!",
            source_lang="en",
            target_lang="es",
            domain=None,
        )

        assert id1 == id2

    async def test_lookup_term_basic(self, termbase):
        """Test looking up a term."""
        await termbase.add_term(
            source_term="API",
            target_term="interfaz",
            source_lang="en",
            target_lang="es",
        )

        entries = await termbase.lookup_term(
            term="API",
            source_lang="en",
            target_lang="es",
        )

        assert len(entries) == 1
        assert entries[0].source_term == "API"
        assert entries[0].target_term == "interfaz"

    async def test_lookup_term_case_insensitive(self, termbase):
        """Test lookup is case-insensitive."""
        await termbase.add_term(
            source_term="API",
            target_term="interfaz",
            source_lang="en",
            target_lang="es",
        )

        # Lookup with different case
        entries = await termbase.lookup_term(
            term="api",  # lowercase
            source_lang="en",
            target_lang="es",
        )

        assert len(entries) == 1
        assert entries[0].source_term == "API"

    async def test_lookup_term_with_domain_filter(self, termbase):
        """Test looking up term with domain filter."""
        await termbase.add_term(
            source_term="cell",
            target_term="célula",
            source_lang="en",
            target_lang="es",
            domain="biology",
        )

        await termbase.add_term(
            source_term="cell",
            target_term="celda",
            source_lang="en",
            target_lang="es",
            domain="telecom",
        )

        # Lookup with domain filter
        entries = await termbase.lookup_term(
            term="cell",
            source_lang="en",
            target_lang="es",
            domain="biology",
        )

        assert len(entries) == 1
        assert entries[0].target_term == "célula"

    async def test_lookup_term_not_found(self, termbase):
        """Test looking up non-existent term."""
        entries = await termbase.lookup_term(
            term="nonexistent",
            source_lang="en",
            target_lang="es",
        )

        assert entries == []

    async def test_validate_translation_no_violations(self, termbase):
        """Test validation with correct terminology."""
        await termbase.add_term(
            source_term="API",
            target_term="API",
            source_lang="en",
            target_lang="es",
            domain="technical",
        )

        violations = await termbase.validate_translation(
            source_text="The API works well.",
            translation="El API funciona bien.",
            source_lang="en",
            target_lang="es",
            domain="technical",
        )

        assert len(violations) == 0

    async def test_validate_translation_with_violations(self, termbase):
        """Test validation detecting term violations."""
        # Add approved term
        await termbase.add_term(
            source_term="API",
            target_term="interfaz de programación",
            source_lang="en",
            target_lang="es",
            domain="technical",
        )

        # Validate translation that doesn't use approved term
        violations = await termbase.validate_translation(
            source_text="The API is fast.",
            translation="El sistema es rápido.",  # Doesn't contain "interfaz"
            source_lang="en",
            target_lang="es",
            domain="technical",
        )

        assert len(violations) == 1
        assert violations[0].source_term == "API"
        assert "interfaz de programación" in violations[0].expected_terms
        assert violations[0].found_in_translation is False

    async def test_validate_translation_critical_severity(self, termbase):
        """Test validation with critical severity."""
        await termbase.add_term(
            source_term="critical",
            target_term="crítico",
            source_lang="en",
            target_lang="es",
            usage_note="This is CRITICAL term",
        )

        violations = await termbase.validate_translation(
            source_text="This is critical.",
            translation="Esto es importante.",  # Wrong translation
            source_lang="en",
            target_lang="es",
        )

        # Should detect critical severity from usage_note
        assert len(violations) > 0
        critical_violations = [v for v in violations if v.severity == "critical"]
        assert len(critical_violations) > 0

    async def test_validate_translation_case_insensitive(self, termbase):
        """Test validation is case-insensitive."""
        await termbase.add_term(
            source_term="API",
            target_term="interfaz",
            source_lang="en",
            target_lang="es",
        )

        # Translation has term in different case
        violations = await termbase.validate_translation(
            source_text="The API works.",
            translation="La INTERFAZ funciona.",  # Uppercase INTERFAZ
            source_lang="en",
            target_lang="es",
        )

        # Should NOT be a violation (case-insensitive match)
        assert len(violations) == 0

    @pytest.mark.asyncio(False)  # Override module-level asyncio marker
    def test_extract_terms_basic(self, termbase):
        """Test extracting terms from text."""
        text = "The API uses HTTP protocol for REST communication."

        terms = termbase._extract_terms(text)

        assert "API" in terms
        assert "HTTP" in terms
        assert "REST" in terms

    def test_extract_terms_capitalized(self, termbase):
        """Test extracting capitalized terms."""
        text = "The Database connects to Server."

        terms = termbase._extract_terms(text)

        assert "Database" in terms
        assert "Server" in terms

    def test_extract_terms_hyphenated(self, termbase):
        """Test extracting hyphenated terms."""
        text = "Using multi-threading and load-balancing."

        terms = termbase._extract_terms(text)

        hyphenated = [t for t in terms if "-" in t]
        assert len(hyphenated) > 0

    def test_extract_terms_filters_short(self, termbase):
        """Test that short common words are filtered."""
        text = "The is at on in"

        terms = termbase._extract_terms(text)

        # Short words should be filtered
        assert "is" not in terms
        assert "at" not in terms
        assert "on" not in terms

    def test_extract_terms_multi_word_phrases(self, termbase):
        """Test extracting multi-word phrases."""
        text = "Application Programming Interface works with Remote Procedure Call"

        terms = termbase._extract_terms(text)

        # Should extract multi-word technical phrases
        multi_word = [t for t in terms if " " in t]
        assert len(multi_word) > 0

    async def test_add_glossary(self, termbase):
        """Test bulk adding terms from glossary."""
        glossary = {
            "API": "interfaz de programación",
            "HTTP": "HTTP",
            "REST": "REST",
        }

        count = await termbase.add_glossary(
            glossary=glossary,
            source_lang="en",
            target_lang="es",
            domain="technical",
        )

        assert count == 3

        # Verify all terms were added
        api_entries = await termbase.lookup_term("API", "en", "es")
        assert len(api_entries) == 1

    async def test_add_glossary_with_errors(self, termbase):
        """Test glossary addition handles individual failures gracefully."""
        # Create a glossary where one term might cause issues
        glossary = {
            "term1": "traducción1",
            "term2": "traducción2",
        }

        # Should not raise exception even if some terms fail
        count = await termbase.add_glossary(
            glossary=glossary,
            source_lang="en",
            target_lang="es",
        )

        assert count >= 0

    async def test_get_statistics(self, termbase):
        """Test getting termbase statistics."""
        # Add some terms
        await termbase.add_term(
            source_term="term1",
            target_term="término1",
            source_lang="en",
            target_lang="es",
            domain="technical",
        )

        await termbase.add_term(
            source_term="term2",
            target_term="término2",
            source_lang="en",
            target_lang="es",
            domain="medical",
        )

        stats = await termbase.get_statistics()

        assert stats["total_terms"] == 2
        assert "en-es" in stats["language_pairs"]
        assert "technical" in stats["domains"]
        assert "medical" in stats["domains"]
        assert stats["domains"]["technical"] == 1
        assert stats["domains"]["medical"] == 1

    async def test_cleanup_closes_database(self, termbase):
        """Test cleanup closes database connection."""
        await termbase.cleanup()

        assert termbase.db is None
        assert termbase._initialized is False

    async def test_not_initialized_error(self):
        """Test operations fail when not initialized."""
        tb = TerminologyBase(":memory:")

        # Should raise error when not initialized
        with pytest.raises(RuntimeError, match="not initialized"):
            await tb.add_term("test", "prueba", "en", "es")

    async def test_validate_translation_not_initialized(self):
        """Test validation fails when not initialized."""
        tb = TerminologyBase(":memory:")

        with pytest.raises(RuntimeError, match="not initialized"):
            await tb.validate_translation("test", "prueba", "en", "es")

    async def test_lookup_term_not_initialized(self):
        """Test lookup fails when not initialized."""
        tb = TerminologyBase(":memory:")

        with pytest.raises(RuntimeError, match="not initialized"):
            await tb.lookup_term("test", "en", "es")

    async def test_get_statistics_not_initialized(self):
        """Test statistics fails when not initialized."""
        tb = TerminologyBase(":memory:")

        with pytest.raises(RuntimeError, match="not initialized"):
            await tb.get_statistics()
