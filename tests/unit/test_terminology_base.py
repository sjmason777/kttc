"""Tests for Terminology Base module."""

import tempfile
from pathlib import Path

import pytest

from kttc.memory.termbase import TermEntry, TerminologyBase, TermViolation


class TestTermEntry:
    """Tests for TermEntry model."""

    def test_create_term_entry(self):
        """Test creating a term entry."""
        entry = TermEntry(
            source_term="API",
            target_term="interfaz de programación de aplicaciones",
            source_lang="en",
            target_lang="es",
            domain="technical",
            definition="Application Programming Interface",
        )

        assert entry.source_term == "API"
        assert entry.target_term == "interfaz de programación de aplicaciones"
        assert entry.domain == "technical"


@pytest.mark.asyncio
class TestTerminologyBase:
    """Tests for TerminologyBase class."""

    @pytest.fixture
    async def termbase(self):
        """Create temporary termbase."""
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
            target_term="API",
            source_lang="en",
            target_lang="es",
            domain="technical",
            definition="Application Programming Interface",
        )

        assert term_id > 0

    async def test_add_duplicate_term_updates(self, termbase):
        """Test that adding duplicate term updates existing entry."""
        # Add first time
        id1 = await termbase.add_term(
            source_term="API",
            target_term="interfaz",
            source_lang="en",
            target_lang="es",
        )

        # Add again with different translation
        id2 = await termbase.add_term(
            source_term="API",
            target_term="API",
            source_lang="en",
            target_lang="es",
        )

        assert id1 == id2

    async def test_lookup_term_exact_match(self, termbase):
        """Test looking up term with exact match."""
        await termbase.add_term(
            source_term="API",
            target_term="API",
            source_lang="en",
            target_lang="es",
            domain="technical",
        )

        results = await termbase.lookup_term(term="API", source_lang="en", target_lang="es")

        assert len(results) == 1
        assert results[0].source_term == "API"
        assert results[0].target_term == "API"

    async def test_lookup_term_case_insensitive(self, termbase):
        """Test that lookup is case-insensitive."""
        await termbase.add_term(
            source_term="API",
            target_term="API",
            source_lang="en",
            target_lang="es",
        )

        # Lookup with lowercase
        results = await termbase.lookup_term(term="api", source_lang="en", target_lang="es")

        assert len(results) == 1
        assert results[0].source_term == "API"

    async def test_lookup_term_filters_by_domain(self, termbase):
        """Test domain filtering in lookup."""
        # Add same term in different domains
        await termbase.add_term(
            source_term="cell",
            target_term="célula",
            source_lang="en",
            target_lang="es",
            domain="medical",
        )

        await termbase.add_term(
            source_term="cell",
            target_term="celda",
            source_lang="en",
            target_lang="es",
            domain="technical",
        )

        # Lookup with domain filter
        results = await termbase.lookup_term(
            term="cell", source_lang="en", target_lang="es", domain="medical"
        )

        assert len(results) == 1
        assert results[0].target_term == "célula"

    async def test_validate_translation_detects_violations(self, termbase):
        """Test translation validation detects term violations."""
        # Add term to termbase
        await termbase.add_term(
            source_term="API",
            target_term="interfaz de programación",
            source_lang="en",
            target_lang="es",
            domain="technical",
        )

        # Validate translation that doesn't use correct term
        violations = await termbase.validate_translation(
            source_text="The API is available",
            translation="La API está disponible",  # Uses "API" instead of "interfaz de programación"
            source_lang="en",
            target_lang="es",
            domain="technical",
        )

        # Should detect violation
        assert len(violations) == 1
        assert violations[0].source_term == "API"
        assert not violations[0].found_in_translation

    async def test_validate_translation_no_violations(self, termbase):
        """Test validation with correct terminology."""
        # Add term
        await termbase.add_term(
            source_term="API",
            target_term="API",
            source_lang="en",
            target_lang="es",
        )

        # Validate correct translation
        violations = await termbase.validate_translation(
            source_text="The API is available",
            translation="La API está disponible",
            source_lang="en",
            target_lang="es",
        )

        # Should have no violations
        assert len(violations) == 0

    async def test_extract_terms(self, termbase):
        """Test term extraction from text."""
        text = "The API uses REST architecture for HTTP requests."

        terms = termbase._extract_terms(text)

        # Should extract capitalized words and acronyms
        assert "API" in terms
        assert "REST" in terms
        assert "HTTP" in terms

    async def test_add_glossary_bulk(self, termbase):
        """Test bulk adding terms from glossary."""
        glossary = {
            "API": "API",
            "REST": "REST",
            "HTTP": "HTTP",
        }

        count = await termbase.add_glossary(
            glossary=glossary,
            source_lang="en",
            target_lang="es",
            domain="technical",
        )

        assert count == 3

        # Verify all added
        results = await termbase.lookup_term(term="API", source_lang="en", target_lang="es")
        assert len(results) == 1

    async def test_get_statistics(self, termbase):
        """Test getting termbase statistics."""
        # Add some terms
        await termbase.add_term(
            source_term="API",
            target_term="API",
            source_lang="en",
            target_lang="es",
            domain="technical",
        )

        await termbase.add_term(
            source_term="cell",
            target_term="célula",
            source_lang="en",
            target_lang="es",
            domain="medical",
        )

        stats = await termbase.get_statistics()

        assert stats["total_terms"] == 2
        assert "en-es" in stats["language_pairs"]
        assert "technical" in stats["domains"]
        assert "medical" in stats["domains"]

    async def test_cleanup_closes_database(self, termbase):
        """Test cleanup closes database."""
        await termbase.cleanup()

        assert termbase.db is None
        assert termbase._initialized is False


class TestTermViolation:
    """Tests for TermViolation model."""

    def test_create_violation(self):
        """Test creating term violation."""
        violation = TermViolation(
            source_term="API",
            expected_terms=["interfaz de programación"],
            found_in_translation=False,
            severity="major",
        )

        assert violation.source_term == "API"
        assert len(violation.expected_terms) == 1
        assert not violation.found_in_translation
        assert violation.severity == "major"
