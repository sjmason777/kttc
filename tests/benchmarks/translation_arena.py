"""Translation Arena - ELO-based benchmarking system.

Provides competitive evaluation of translation systems using ELO ratings,
similar to LMSYS Chatbot Arena approach.
"""

from __future__ import annotations

import json
import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ArenaSystem(BaseModel):
    """Translation system in arena.

    Represents a system (KTTC configuration, etc.)
    with its ELO rating and battle history.
    """

    system_id: str = Field(description="Unique system identifier")
    display_name: str = Field(description="Human-readable name")
    elo_rating: float = Field(default=1500.0, description="ELO rating (starts at 1500)")
    battles_count: int = Field(default=0, description="Total number of battles")
    wins: int = Field(default=0, description="Number of wins")
    losses: int = Field(default=0, description="Number of losses")
    ties: int = Field(default=0, description="Number of ties")

    @property
    def win_rate(self) -> float:
        """Calculate win rate percentage."""
        if self.battles_count == 0:
            return 0.0
        return (self.wins / self.battles_count) * 100


class BattleResult(BaseModel):
    """Result from an arena battle.

    Records outcome of head-to-head comparison between two systems.
    """

    battle_id: str = Field(description="Unique battle identifier")
    system_a: str = Field(description="First system ID")
    system_b: str = Field(description="Second system ID")
    source_text: str = Field(description="Source text being translated")
    translation_a: str = Field(description="Translation from system A")
    translation_b: str = Field(description="Translation from system B")
    winner: str = Field(
        description="Winner system ID, or 'tie'", pattern=r"^(system_a|system_b|tie)$"
    )
    judge: str = Field(description="Judge type: human, llm, automatic")
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: dict[str, Any] = Field(default_factory=dict)


class TranslationArena:
    """ELO-based benchmarking arena for translation systems.

    Implements competitive evaluation using ELO rating system.
    Systems battle head-to-head and ratings are updated based on outcomes.

    Example:
        >>> arena = TranslationArena("arena_results.json")
        >>>
        >>> # Register systems
        >>> arena.register_system("kttc-v1", "KTTC v1.0")
        >>> arena.register_system("google-translate", "Google Translate")
        >>>
        >>> # Run battle
        >>> result = await arena.battle(
        ...     system_a="kttc-v1",
        ...     system_b="google-translate",
        ...     source="Hello, world!",
        ...     judge_type="llm"
        ... )
        >>>
        >>> # View leaderboard
        >>> arena.print_leaderboard()
    """

    K_FACTOR = 32  # ELO K-factor (higher = more volatile ratings)

    def __init__(self, results_file: str | Path = "arena_results.json"):
        """Initialize Translation Arena.

        Args:
            results_file: Path to save/load arena state
        """
        self.results_file = Path(results_file)
        self.systems: dict[str, ArenaSystem] = {}
        self.battles: list[BattleResult] = []

        # Load existing state if available
        if self.results_file.exists():
            self.load_state()

    def register_system(
        self, system_id: str, display_name: str, initial_rating: float = 1500.0
    ) -> None:
        """Register a new system in the arena.

        Args:
            system_id: Unique identifier
            display_name: Human-readable name
            initial_rating: Starting ELO rating (default: 1500)
        """
        if system_id in self.systems:
            logger.warning(f"System {system_id} already registered")
            return

        self.systems[system_id] = ArenaSystem(
            system_id=system_id, display_name=display_name, elo_rating=initial_rating
        )
        logger.info(f"Registered system: {display_name} (ID: {system_id})")

    def calculate_expected_score(self, rating_a: float, rating_b: float) -> float:
        """Calculate expected score for system A vs system B.

        Args:
            rating_a: ELO rating of system A
            rating_b: ELO rating of system B

        Returns:
            Expected score (0-1) for system A
        """
        return 1 / (1 + math.pow(10, (rating_b - rating_a) / 400))

    def update_elo_ratings(self, system_a_id: str, system_b_id: str, winner: str) -> None:
        """Update ELO ratings after battle.

        Args:
            system_a_id: First system ID
            system_b_id: Second system ID
            winner: Winner ('system_a', 'system_b', or 'tie')
        """
        system_a = self.systems[system_a_id]
        system_b = self.systems[system_b_id]

        # Calculate expected scores
        expected_a = self.calculate_expected_score(system_a.elo_rating, system_b.elo_rating)
        expected_b = 1 - expected_a

        # Actual scores
        if winner == "system_a":
            actual_a, actual_b = 1.0, 0.0
            system_a.wins += 1
            system_b.losses += 1
        elif winner == "system_b":
            actual_a, actual_b = 0.0, 1.0
            system_a.losses += 1
            system_b.wins += 1
        else:  # tie
            actual_a, actual_b = 0.5, 0.5
            system_a.ties += 1
            system_b.ties += 1

        # Update ratings
        new_rating_a = system_a.elo_rating + self.K_FACTOR * (actual_a - expected_a)
        new_rating_b = system_b.elo_rating + self.K_FACTOR * (actual_b - expected_b)

        logger.info(
            f"ELO update: {system_a.display_name} "
            f"{system_a.elo_rating:.0f} → {new_rating_a:.0f}, "
            f"{system_b.display_name} "
            f"{system_b.elo_rating:.0f} → {new_rating_b:.0f}"
        )

        system_a.elo_rating = new_rating_a
        system_b.elo_rating = new_rating_b
        system_a.battles_count += 1
        system_b.battles_count += 1

    def record_battle(
        self,
        system_a_id: str,
        system_b_id: str,
        source: str,
        translation_a: str,
        translation_b: str,
        winner: str,
        judge: str = "human",
        metadata: dict[str, Any] | None = None,
    ) -> BattleResult:
        """Record battle result and update ratings.

        Args:
            system_a_id: First system ID
            system_b_id: Second system ID
            source: Source text
            translation_a: Translation from system A
            translation_b: Translation from system B
            winner: Winner ('system_a', 'system_b', or 'tie')
            judge: Judge type (human, llm, automatic)
            metadata: Additional battle metadata

        Returns:
            BattleResult object
        """
        # Validate systems exist
        if system_a_id not in self.systems:
            raise ValueError(f"System {system_a_id} not registered")
        if system_b_id not in self.systems:
            raise ValueError(f"System {system_b_id} not registered")

        # Create battle result
        battle_id = f"battle_{len(self.battles) + 1:06d}"
        result = BattleResult(
            battle_id=battle_id,
            system_a=system_a_id,
            system_b=system_b_id,
            source_text=source,
            translation_a=translation_a,
            translation_b=translation_b,
            winner=winner,
            judge=judge,
            metadata=metadata or {},
        )

        # Update ELO ratings
        self.update_elo_ratings(system_a_id, system_b_id, winner)

        # Record battle
        self.battles.append(result)

        # Auto-save
        self.save_state()

        return result

    def get_leaderboard(self) -> list[ArenaSystem]:
        """Get leaderboard sorted by ELO rating.

        Returns:
            List of systems sorted by rating (highest first)
        """
        return sorted(self.systems.values(), key=lambda s: s.elo_rating, reverse=True)

    def print_leaderboard(self, top_n: int = 10) -> None:
        """Print formatted leaderboard.

        Args:
            top_n: Number of top systems to display
        """
        leaderboard = self.get_leaderboard()[:top_n]

        print("\n" + "=" * 80)
        print("🏆 TRANSLATION ARENA LEADERBOARD".center(80))
        print("=" * 80)
        print(f"{'Rank':<6} {'System':<30} {'ELO':<8} {'Battles':<10} {'Win Rate':<10}")
        print("-" * 80)

        for rank, system in enumerate(leaderboard, 1):
            print(
                f"{rank:<6} {system.display_name:<30} "
                f"{system.elo_rating:<8.0f} {system.battles_count:<10} "
                f"{system.win_rate:<10.1f}%"
            )

        print("=" * 80 + "\n")

    def get_statistics(self) -> dict[str, Any]:
        """Get arena statistics.

        Returns:
            Dictionary with arena stats
        """
        return {
            "total_systems": len(self.systems),
            "total_battles": len(self.battles),
            "leaderboard": [
                {
                    "rank": rank,
                    "system_id": system.system_id,
                    "display_name": system.display_name,
                    "elo_rating": round(system.elo_rating, 1),
                    "battles": system.battles_count,
                    "wins": system.wins,
                    "losses": system.losses,
                    "ties": system.ties,
                    "win_rate": round(system.win_rate, 2),
                }
                for rank, system in enumerate(self.get_leaderboard(), 1)
            ],
        }

    def save_state(self) -> None:
        """Save arena state to file."""
        state = {
            "systems": [system.model_dump() for system in self.systems.values()],
            "battles": [battle.model_dump(mode="json") for battle in self.battles],
            "updated_at": datetime.now().isoformat(),
        }

        self.results_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.results_file, "w") as f:
            json.dump(state, f, indent=2, default=str)

        logger.info(f"Arena state saved to {self.results_file}")

    def load_state(self) -> None:
        """Load arena state from file."""
        if not self.results_file.exists():
            logger.info("No existing arena state found")
            return

        with open(self.results_file) as f:
            state = json.load(f)

        # Load systems
        self.systems = {s["system_id"]: ArenaSystem(**s) for s in state.get("systems", [])}

        # Load battles
        self.battles = [BattleResult(**b) for b in state.get("battles", [])]

        logger.info(f"Loaded arena state: {len(self.systems)} systems, {len(self.battles)} battles")

    def export_report(self, output_file: str | Path = "arena_report.md") -> None:
        """Export arena report as markdown.

        Args:
            output_file: Path to output markdown file
        """
        report = f"""# Translation Arena Report

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## 📊 Statistics

- **Total Systems:** {len(self.systems)}
- **Total Battles:** {len(self.battles)}

## 🏆 Leaderboard

| Rank | System | ELO Rating | Battles | W-L-T | Win Rate |
|------|--------|-----------|---------|-------|----------|
"""

        for rank, system in enumerate(self.get_leaderboard(), 1):
            wlt = f"{system.wins}-{system.losses}-{system.ties}"
            report += (
                f"| {rank} | {system.display_name} | "
                f"{system.elo_rating:.0f} | {system.battles_count} | "
                f"{wlt} | {system.win_rate:.1f}% |\n"
            )

        report += "\n## 📈 Recent Battles\n\n"

        for battle in self.battles[-10:]:
            system_a_name = self.systems[battle.system_a].display_name
            system_b_name = self.systems[battle.system_b].display_name

            if battle.winner == "system_a":
                winner_name = system_a_name
            elif battle.winner == "system_b":
                winner_name = system_b_name
            else:
                winner_name = "Tie"

            report += f"- **{battle.battle_id}:** {system_a_name} vs {system_b_name}\n"
            report += f"  - Winner: {winner_name}\n"
            report += f"  - Judge: {battle.judge}\n"
            report += f"  - Date: {battle.timestamp.strftime('%Y-%m-%d %H:%M')}\n\n"

        Path(output_file).write_text(report)
        logger.info(f"Arena report exported to {output_file}")
