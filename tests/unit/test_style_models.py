"""Unit tests for style data models.

Tests StyleDeviationType, StylePattern, StyleDeviation, StyleProfile,
and StyleComparisonResult classes.
"""

import pytest

from kttc.style.models import (
    StyleComparisonResult,
    StyleDeviation,
    StyleDeviationType,
    StylePattern,
    StyleProfile,
)


@pytest.mark.unit
class TestStyleDeviationType:
    """Test StyleDeviationType enum."""

    def test_lexical_deviations_exist(self) -> None:
        """Test lexical deviation types exist."""
        assert StyleDeviationType.PLEONASM.value == "pleonasm"
        assert StyleDeviationType.NEOLOGISM.value == "neologism"
        assert StyleDeviationType.ARCHAISM.value == "archaism"
        assert StyleDeviationType.FOLK_ETYMOLOGY.value == "folk_etymology"
        assert StyleDeviationType.DIALECTISM.value == "dialectism"
        assert StyleDeviationType.COLLOQUIALISM.value == "colloquialism"

    def test_syntactic_deviations_exist(self) -> None:
        """Test syntactic deviation types exist."""
        assert StyleDeviationType.INVERSION.value == "syntactic_inversion"
        assert StyleDeviationType.FRAGMENTATION.value == "fragmentation"
        assert StyleDeviationType.RUN_ON.value == "run_on"
        assert StyleDeviationType.ELLIPSIS.value == "ellipsis"

    def test_stylistic_deviations_exist(self) -> None:
        """Test stylistic deviation types exist."""
        assert StyleDeviationType.REGISTER_MIXING.value == "register_mixing"
        assert StyleDeviationType.STREAM_OF_CONSCIOUSNESS.value == "stream_of_consciousness"
        assert StyleDeviationType.SKAZ.value == "skaz"
        assert StyleDeviationType.FREE_INDIRECT_DISCOURSE.value == "free_indirect_discourse"

    def test_phonetic_deviations_exist(self) -> None:
        """Test phonetic/rhythmic deviation types exist."""
        assert StyleDeviationType.ALLITERATION.value == "alliteration"
        assert StyleDeviationType.RHYTHM_BREAK.value == "rhythm_break"

    def test_other_deviations_exist(self) -> None:
        """Test other deviation types exist."""
        assert StyleDeviationType.WORDPLAY.value == "wordplay"
        assert StyleDeviationType.IRONY_MARKER.value == "irony_marker"

    def test_is_str_enum(self) -> None:
        """Test that StyleDeviationType is a string enum."""
        assert isinstance(StyleDeviationType.PLEONASM.value, str)
        assert str(StyleDeviationType.PLEONASM) == "StyleDeviationType.PLEONASM"


@pytest.mark.unit
class TestStylePattern:
    """Test StylePattern enum."""

    def test_all_patterns_exist(self) -> None:
        """Test all style patterns exist."""
        assert StylePattern.STANDARD.value == "standard"
        assert StylePattern.TECHNICAL.value == "technical"
        assert StylePattern.SKAZ_NARRATIVE.value == "skaz_narrative"
        assert StylePattern.MODERNIST.value == "modernist"
        assert StylePattern.STREAM.value == "stream_of_consciousness"
        assert StylePattern.POETIC.value == "poetic"
        assert StylePattern.COLLOQUIAL.value == "colloquial"
        assert StylePattern.ARCHAIC.value == "archaic"
        assert StylePattern.MIXED.value == "mixed"

    def test_is_str_enum(self) -> None:
        """Test that StylePattern is a string enum."""
        assert isinstance(StylePattern.STANDARD.value, str)


@pytest.mark.unit
class TestStyleDeviation:
    """Test StyleDeviation dataclass."""

    def test_minimal_creation(self) -> None:
        """Test creating deviation with minimal arguments."""
        deviation = StyleDeviation(type=StyleDeviationType.PLEONASM)

        assert deviation.type == StyleDeviationType.PLEONASM
        assert deviation.examples == []
        assert deviation.locations == []
        assert deviation.confidence == 0.8
        assert deviation.interpretation == ""
        assert deviation.is_intentional is True

    def test_full_creation(self) -> None:
        """Test creating deviation with all arguments."""
        deviation = StyleDeviation(
            type=StyleDeviationType.SKAZ,
            examples=["батюшк", "голубчик"],
            locations=[(10, 18), (30, 38)],
            confidence=0.9,
            interpretation="Folk speech markers detected",
            is_intentional=True,
        )

        assert deviation.type == StyleDeviationType.SKAZ
        assert len(deviation.examples) == 2
        assert len(deviation.locations) == 2
        assert deviation.confidence == 0.9
        assert "Folk speech" in deviation.interpretation

    def test_to_dict(self) -> None:
        """Test converting deviation to dictionary."""
        deviation = StyleDeviation(
            type=StyleDeviationType.PLEONASM,
            examples=["жить жизнью"],
            locations=[(0, 12)],
            confidence=0.85,
            interpretation="Platanov-style redundancy",
            is_intentional=True,
        )

        result = deviation.to_dict()

        assert result["type"] == "pleonasm"
        assert result["examples"] == ["жить жизнью"]
        assert result["locations"] == [(0, 12)]
        assert result["confidence"] == 0.85
        assert result["interpretation"] == "Platanov-style redundancy"
        assert result["is_intentional"] is True

    def test_to_dict_with_empty_lists(self) -> None:
        """Test to_dict with empty lists."""
        deviation = StyleDeviation(type=StyleDeviationType.INVERSION)
        result = deviation.to_dict()

        assert result["examples"] == []
        assert result["locations"] == []


@pytest.mark.unit
class TestStyleProfile:
    """Test StyleProfile dataclass."""

    def test_default_creation(self) -> None:
        """Test creating profile with defaults."""
        profile = StyleProfile()

        assert profile.deviation_score == 0.0
        assert profile.detected_pattern == StylePattern.STANDARD
        assert profile.detected_deviations == []
        assert profile.lexical_diversity == 0.0
        assert profile.avg_sentence_length == 0.0
        assert profile.sentence_length_variance == 0.0
        assert profile.punctuation_density == 0.0
        assert profile.is_literary is False
        assert profile.is_technical is False
        assert profile.recommended_fluency_tolerance == 0.0
        assert profile.metadata == {}

    def test_full_creation(self) -> None:
        """Test creating profile with all fields."""
        deviation = StyleDeviation(type=StyleDeviationType.PLEONASM)
        profile = StyleProfile(
            deviation_score=0.6,
            detected_pattern=StylePattern.MODERNIST,
            detected_deviations=[deviation],
            lexical_diversity=0.75,
            avg_sentence_length=25.5,
            sentence_length_variance=150.0,
            punctuation_density=12.5,
            is_literary=True,
            is_technical=False,
            recommended_fluency_tolerance=0.4,
            metadata={"language": "ru", "word_count": 500},
        )

        assert profile.deviation_score == 0.6
        assert profile.detected_pattern == StylePattern.MODERNIST
        assert len(profile.detected_deviations) == 1
        assert profile.is_literary is True

    def test_deviation_types_property(self) -> None:
        """Test deviation_types property."""
        deviations = [
            StyleDeviation(type=StyleDeviationType.PLEONASM),
            StyleDeviation(type=StyleDeviationType.INVERSION),
            StyleDeviation(type=StyleDeviationType.PLEONASM),  # Duplicate
        ]
        profile = StyleProfile(detected_deviations=deviations)

        types = profile.deviation_types

        assert len(types) == 2
        assert StyleDeviationType.PLEONASM in types
        assert StyleDeviationType.INVERSION in types

    def test_deviation_types_empty(self) -> None:
        """Test deviation_types with no deviations."""
        profile = StyleProfile()
        assert profile.deviation_types == set()

    def test_has_significant_deviations_by_score(self) -> None:
        """Test has_significant_deviations based on deviation_score."""
        profile = StyleProfile(deviation_score=0.35)
        assert profile.has_significant_deviations is True

        profile = StyleProfile(deviation_score=0.25)
        assert profile.has_significant_deviations is False

    def test_has_significant_deviations_by_count(self) -> None:
        """Test has_significant_deviations based on deviation count."""
        deviations = [
            StyleDeviation(type=StyleDeviationType.PLEONASM),
            StyleDeviation(type=StyleDeviationType.INVERSION),
            StyleDeviation(type=StyleDeviationType.SKAZ),
        ]
        profile = StyleProfile(deviation_score=0.1, detected_deviations=deviations)

        assert profile.has_significant_deviations is True

    def test_has_significant_deviations_false(self) -> None:
        """Test has_significant_deviations returns False for standard text."""
        profile = StyleProfile(
            deviation_score=0.1,
            detected_deviations=[StyleDeviation(type=StyleDeviationType.PLEONASM)],
        )
        assert profile.has_significant_deviations is False

    def test_get_agent_weight_adjustments_standard(self) -> None:
        """Test weight adjustments for standard style."""
        profile = StyleProfile()
        adjustments = profile.get_agent_weight_adjustments()

        assert adjustments["accuracy"] == 1.0
        assert adjustments["fluency"] == 1.0
        assert adjustments["terminology"] == 1.0
        assert adjustments["style_preservation"] == 1.0

    def test_get_agent_weight_adjustments_technical(self) -> None:
        """Test weight adjustments for technical documentation."""
        profile = StyleProfile(
            is_technical=True,
            detected_pattern=StylePattern.TECHNICAL,
        )
        adjustments = profile.get_agent_weight_adjustments()

        assert adjustments["style_preservation"] == 0.5
        assert adjustments["terminology"] == 1.5

    def test_get_agent_weight_adjustments_skaz(self) -> None:
        """Test weight adjustments for skaz narrative."""
        deviations = [
            StyleDeviation(type=StyleDeviationType.SKAZ),
            StyleDeviation(type=StyleDeviationType.COLLOQUIALISM),
            StyleDeviation(type=StyleDeviationType.DIALECTISM),
        ]
        profile = StyleProfile(
            deviation_score=0.5,
            detected_pattern=StylePattern.SKAZ_NARRATIVE,
            detected_deviations=deviations,
        )
        adjustments = profile.get_agent_weight_adjustments()

        assert adjustments["fluency"] < 1.0
        assert adjustments["style_preservation"] > 1.0

    def test_get_agent_weight_adjustments_modernist(self) -> None:
        """Test weight adjustments for modernist style."""
        deviations = [
            StyleDeviation(type=StyleDeviationType.PLEONASM),
            StyleDeviation(type=StyleDeviationType.INVERSION),
            StyleDeviation(type=StyleDeviationType.FRAGMENTATION),
        ]
        profile = StyleProfile(
            deviation_score=0.6,
            detected_pattern=StylePattern.MODERNIST,
            detected_deviations=deviations,
        )
        adjustments = profile.get_agent_weight_adjustments()

        assert adjustments["fluency"] < 0.8
        assert adjustments["style_preservation"] > 1.3

    def test_get_agent_weight_adjustments_stream(self) -> None:
        """Test weight adjustments for stream of consciousness."""
        deviations = [
            StyleDeviation(type=StyleDeviationType.STREAM_OF_CONSCIOUSNESS),
            StyleDeviation(type=StyleDeviationType.RUN_ON),
            StyleDeviation(type=StyleDeviationType.FRAGMENTATION),
        ]
        profile = StyleProfile(
            deviation_score=0.7,
            detected_pattern=StylePattern.STREAM,
            detected_deviations=deviations,
        )
        adjustments = profile.get_agent_weight_adjustments()

        assert adjustments["fluency"] == 0.2
        assert adjustments["style_preservation"] == 1.8

    def test_to_dict(self) -> None:
        """Test converting profile to dictionary."""
        deviation = StyleDeviation(type=StyleDeviationType.PLEONASM)
        profile = StyleProfile(
            deviation_score=0.5,
            detected_pattern=StylePattern.MODERNIST,
            detected_deviations=[deviation],
            lexical_diversity=0.7,
            avg_sentence_length=20.0,
            sentence_length_variance=100.0,
            punctuation_density=10.0,
            is_literary=True,
            is_technical=False,
            recommended_fluency_tolerance=0.3,
            metadata={"language": "ru"},
        )

        result = profile.to_dict()

        assert result["deviation_score"] == 0.5
        assert result["detected_pattern"] == "modernist"
        assert len(result["detected_deviations"]) == 1
        assert result["lexical_diversity"] == 0.7
        assert result["is_literary"] is True
        assert result["is_technical"] is False
        assert result["has_significant_deviations"] is True
        assert "agent_weight_adjustments" in result
        assert result["metadata"]["language"] == "ru"


@pytest.mark.unit
class TestStyleComparisonResult:
    """Test StyleComparisonResult dataclass."""

    def test_default_creation(self) -> None:
        """Test creating result with defaults."""
        result = StyleComparisonResult()

        assert result.style_preservation_score == 0.0
        assert result.deviation_transfer_rate == 0.0
        assert result.preserved_deviations == []
        assert result.lost_deviations == []
        assert result.new_errors == []
        assert result.recommendations == []

    def test_full_creation(self) -> None:
        """Test creating result with all fields."""
        preserved = [StyleDeviation(type=StyleDeviationType.PLEONASM)]
        lost = [StyleDeviation(type=StyleDeviationType.SKAZ)]

        result = StyleComparisonResult(
            style_preservation_score=0.75,
            deviation_transfer_rate=0.5,
            preserved_deviations=preserved,
            lost_deviations=lost,
            new_errors=["Normalized some style"],
            recommendations=["Preserve folk speech"],
        )

        assert result.style_preservation_score == 0.75
        assert result.deviation_transfer_rate == 0.5
        assert len(result.preserved_deviations) == 1
        assert len(result.lost_deviations) == 1
        assert len(result.new_errors) == 1
        assert len(result.recommendations) == 1

    def test_to_dict(self) -> None:
        """Test converting result to dictionary."""
        preserved = [StyleDeviation(type=StyleDeviationType.PLEONASM)]
        lost = [StyleDeviation(type=StyleDeviationType.INVERSION)]

        result = StyleComparisonResult(
            style_preservation_score=0.8,
            deviation_transfer_rate=0.6,
            preserved_deviations=preserved,
            lost_deviations=lost,
            new_errors=["Some normalization"],
            recommendations=["Keep inversions"],
        )

        result_dict = result.to_dict()

        assert result_dict["style_preservation_score"] == 0.8
        assert result_dict["deviation_transfer_rate"] == 0.6
        assert len(result_dict["preserved_deviations"]) == 1
        assert len(result_dict["lost_deviations"]) == 1
        assert result_dict["new_errors"] == ["Some normalization"]
        assert result_dict["recommendations"] == ["Keep inversions"]

    def test_to_dict_empty(self) -> None:
        """Test to_dict with empty result."""
        result = StyleComparisonResult()
        result_dict = result.to_dict()

        assert result_dict["preserved_deviations"] == []
        assert result_dict["lost_deviations"] == []
        assert result_dict["new_errors"] == []
        assert result_dict["recommendations"] == []
