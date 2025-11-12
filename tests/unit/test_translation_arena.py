"""Tests for Translation Arena ELO benchmarking system."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Add parent directory to path to allow imports from tests.benchmarks
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.translation_arena import ArenaSystem, TranslationArena


class TestArenaSystem:
    """Test ArenaSystem model."""

    def test_create_system(self) -> None:
        """Test creating arena system."""
        system = ArenaSystem(system_id="test-system", display_name="Test System", elo_rating=1500.0)

        assert system.system_id == "test-system"
        assert system.display_name == "Test System"
        assert system.elo_rating == 1500.0
        assert system.battles_count == 0
        assert system.wins == 0
        assert system.losses == 0
        assert system.ties == 0

    def test_win_rate_no_battles(self) -> None:
        """Test win rate with no battles."""
        system = ArenaSystem(system_id="test", display_name="Test")
        assert system.win_rate == 0.0

    def test_win_rate_calculation(self) -> None:
        """Test win rate calculation."""
        system = ArenaSystem(
            system_id="test", display_name="Test", battles_count=10, wins=7, losses=2, ties=1
        )
        assert system.win_rate == 70.0


class TestTranslationArena:
    """Test TranslationArena functionality."""

    @pytest.fixture
    def arena(self, tmp_path: Path) -> TranslationArena:
        """Create arena instance with temp file."""
        return TranslationArena(tmp_path / "arena_test.json")

    def test_register_system(self, arena: TranslationArena) -> None:
        """Test registering systems."""
        arena.register_system("kttc-v1", "KTTC v1.0")

        assert "kttc-v1" in arena.systems
        assert arena.systems["kttc-v1"].display_name == "KTTC v1.0"
        assert arena.systems["kttc-v1"].elo_rating == 1500.0

    def test_register_duplicate_system(
        self, arena: TranslationArena, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test registering duplicate system."""
        arena.register_system("kttc-v1", "KTTC v1.0")
        arena.register_system("kttc-v1", "KTTC v1.0")  # Duplicate

        assert "already registered" in caplog.text

    def test_calculate_expected_score_equal_ratings(self, arena: TranslationArena) -> None:
        """Test expected score for equal ratings."""
        expected = arena.calculate_expected_score(1500.0, 1500.0)
        assert expected == 0.5

    def test_calculate_expected_score_higher_rating(self, arena: TranslationArena) -> None:
        """Test expected score for higher rated system."""
        expected = arena.calculate_expected_score(1600.0, 1500.0)
        assert expected > 0.5

    def test_calculate_expected_score_lower_rating(self, arena: TranslationArena) -> None:
        """Test expected score for lower rated system."""
        expected = arena.calculate_expected_score(1400.0, 1500.0)
        assert expected < 0.5

    def test_record_battle_winner_a(self, arena: TranslationArena) -> None:
        """Test recording battle with system A winning."""
        arena.register_system("sys-a", "System A")
        arena.register_system("sys-b", "System B")

        initial_rating_a = arena.systems["sys-a"].elo_rating
        initial_rating_b = arena.systems["sys-b"].elo_rating

        result = arena.record_battle(
            system_a_id="sys-a",
            system_b_id="sys-b",
            source="Test source",
            translation_a="Translation A",
            translation_b="Translation B",
            winner="system_a",
            judge="human",
        )

        assert result.winner == "system_a"
        assert len(arena.battles) == 1

        # Check rating updates
        assert arena.systems["sys-a"].elo_rating > initial_rating_a
        assert arena.systems["sys-b"].elo_rating < initial_rating_b

        # Check battle counts
        assert arena.systems["sys-a"].battles_count == 1
        assert arena.systems["sys-b"].battles_count == 1
        assert arena.systems["sys-a"].wins == 1
        assert arena.systems["sys-b"].losses == 1

    def test_record_battle_tie(self, arena: TranslationArena) -> None:
        """Test recording battle with tie."""
        arena.register_system("sys-a", "System A")
        arena.register_system("sys-b", "System B")

        initial_rating_a = arena.systems["sys-a"].elo_rating
        initial_rating_b = arena.systems["sys-b"].elo_rating

        arena.record_battle(
            system_a_id="sys-a",
            system_b_id="sys-b",
            source="Test",
            translation_a="A",
            translation_b="B",
            winner="tie",
        )

        # Ratings should remain roughly equal
        assert abs(arena.systems["sys-a"].elo_rating - initial_rating_a) < 1
        assert abs(arena.systems["sys-b"].elo_rating - initial_rating_b) < 1

        assert arena.systems["sys-a"].ties == 1
        assert arena.systems["sys-b"].ties == 1

    def test_record_battle_invalid_system(self, arena: TranslationArena) -> None:
        """Test recording battle with non-existent system."""
        arena.register_system("sys-a", "System A")

        with pytest.raises(ValueError, match="not registered"):
            arena.record_battle(
                system_a_id="sys-a",
                system_b_id="non-existent",
                source="Test",
                translation_a="A",
                translation_b="B",
                winner="system_a",
            )

    def test_get_leaderboard(self, arena: TranslationArena) -> None:
        """Test getting leaderboard."""
        # Register systems with different ratings
        arena.register_system("sys-a", "System A")
        arena.register_system("sys-b", "System B")
        arena.register_system("sys-c", "System C")

        # Manually set ratings for testing
        arena.systems["sys-a"].elo_rating = 1600
        arena.systems["sys-b"].elo_rating = 1500
        arena.systems["sys-c"].elo_rating = 1400

        leaderboard = arena.get_leaderboard()

        assert len(leaderboard) == 3
        assert leaderboard[0].system_id == "sys-a"
        assert leaderboard[1].system_id == "sys-b"
        assert leaderboard[2].system_id == "sys-c"

    def test_get_statistics(self, arena: TranslationArena) -> None:
        """Test getting arena statistics."""
        arena.register_system("sys-a", "System A")
        arena.register_system("sys-b", "System B")

        arena.record_battle(
            system_a_id="sys-a",
            system_b_id="sys-b",
            source="Test",
            translation_a="A",
            translation_b="B",
            winner="system_a",
        )

        stats = arena.get_statistics()

        assert stats["total_systems"] == 2
        assert stats["total_battles"] == 1
        assert len(stats["leaderboard"]) == 2
        assert stats["leaderboard"][0]["system_id"] == "sys-a"

    def test_save_and_load_state(self, arena: TranslationArena) -> None:
        """Test saving and loading arena state."""
        arena.register_system("sys-a", "System A")
        arena.register_system("sys-b", "System B")

        arena.record_battle(
            system_a_id="sys-a",
            system_b_id="sys-b",
            source="Test",
            translation_a="A",
            translation_b="B",
            winner="system_a",
        )

        arena.save_state()

        # Load into new arena
        new_arena = TranslationArena(arena.results_file)

        assert len(new_arena.systems) == 2
        assert len(new_arena.battles) == 1
        assert "sys-a" in new_arena.systems
        assert "sys-b" in new_arena.systems

    def test_export_report(self, arena: TranslationArena, tmp_path: Path) -> None:
        """Test exporting arena report."""
        arena.register_system("sys-a", "System A")
        arena.register_system("sys-b", "System B")

        arena.record_battle(
            system_a_id="sys-a",
            system_b_id="sys-b",
            source="Test",
            translation_a="A",
            translation_b="B",
            winner="system_a",
        )

        report_file = tmp_path / "report.md"
        arena.export_report(report_file)

        assert report_file.exists()
        content = report_file.read_text()
        assert "Translation Arena Report" in content
        assert "System A" in content
        assert "System B" in content
        assert "Leaderboard" in content

    def test_elo_convergence_multiple_battles(self, arena: TranslationArena) -> None:
        """Test ELO ratings converge with multiple battles."""
        arena.register_system("strong", "Strong System")
        arena.register_system("weak", "Weak System")

        # Strong system wins 10 battles
        for _ in range(10):
            arena.record_battle(
                system_a_id="strong",
                system_b_id="weak",
                source="Test",
                translation_a="A",
                translation_b="B",
                winner="system_a",
            )

        # Strong system should have higher rating
        assert arena.systems["strong"].elo_rating > arena.systems["weak"].elo_rating
        assert arena.systems["strong"].elo_rating > 1550  # Should increase
        assert arena.systems["weak"].elo_rating < 1450  # Should decrease
