"""
curriculum.py — Auto-Curriculum for ARJUNA Environment.

Tracks agent performance across episodes and automatically
adjusts scene difficulty based on recent reward history.

Difficulty progression:
  easy   → agent is struggling (mean reward < 0.60)
  medium → agent is learning  (mean reward 0.60–0.85)
  hard   → agent is performing well (mean reward > 0.85)

Uses a sliding window of recent episode rewards to decide
when to increase or decrease difficulty.

STABILITY NOTE: Window is cleared on every difficulty change to prevent 
"Policy Oscillation" and ensure the agent stabilizes at the new level.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Deque

logger = logging.getLogger(__name__)

# Thresholds for difficulty transitions
PROMOTE_THRESHOLD = 0.85   # above this → increase difficulty
DEMOTE_THRESHOLD  = 0.60   # below this → decrease difficulty
WINDOW_SIZE       = 5      # episodes to consider
MIN_EPISODES      = 3      # minimum before adjusting

DIFFICULTY_LEVELS = ["easy", "medium", "hard"]


class AutoCurriculum:
    """
    Tracks agent performance and adjusts difficulty automatically.

    Usage::

        curriculum = AutoCurriculum(initial_difficulty="easy")

        # After each episode:
        result = curriculum.record(episode_reward=0.92)
        next_difficulty = curriculum.current_difficulty

        # Check what difficulty to use next reset:
        difficulty = curriculum.get_difficulty()
    """

    def __init__(
        self,
        initial_difficulty: str = "easy",
        window_size: int = WINDOW_SIZE,
        promote_threshold: float = PROMOTE_THRESHOLD,
        demote_threshold: float = DEMOTE_THRESHOLD,
        min_episodes: int = MIN_EPISODES,
    ) -> None:
        if initial_difficulty not in DIFFICULTY_LEVELS:
            raise ValueError(
                f"difficulty must be one of {DIFFICULTY_LEVELS}, "
                f"got {initial_difficulty!r}"
            )

        self._difficulty          = initial_difficulty
        self._window: Deque[float] = deque(maxlen=window_size)
        self._promote_threshold   = promote_threshold
        self._demote_threshold    = demote_threshold
        self._min_episodes        = min_episodes

        self.total_episodes: int       = 0
        self.promotions:     int       = 0
        self.demotions:      int       = 0
        self.history:        list[dict] = []

    # ── read-only properties ───────────────────────────────

    @property
    def current_difficulty(self) -> str:
        return self._difficulty

    def get_difficulty(self) -> str:
        """Return current difficulty level."""
        return self._difficulty

    def get_recent_mean(self) -> float:
        """Return mean reward over recent window (0.0 if no data)."""
        if not self._window:
            return 0.0
        return sum(self._window) / len(self._window)

    # ── core record / adjust ───────────────────────────────

    def record(self, episode_reward: float) -> dict:
        """
        Record an episode reward and potentially adjust difficulty.

        Returns a dict with keys:

        * ``difficulty_before``  – str
        * ``difficulty_after``   – str
        * ``changed``            – bool
        * ``reason``             – str (human-readable explanation)
        * ``recent_mean``        – float
        * ``window_size``        – int
        """
        old_difficulty = self._difficulty
        self._window.append(episode_reward)
        self.total_episodes += 1

        result: dict = {
            "difficulty_before": old_difficulty,
            "difficulty_after":  old_difficulty,
            "changed":  False,
            "reason":   "insufficient data",
            "recent_mean":  self.get_recent_mean(),
            "window_size":  len(self._window),
        }

        # Need minimum episodes before adjusting
        if len(self._window) < self._min_episodes:
            result["reason"] = (
                f"collecting data "
                f"({len(self._window)}/{self._min_episodes})"
            )
            self.history.append(
                {"episode": self.total_episodes, "reward": episode_reward, **result}
            )
            return result

        mean           = self.get_recent_mean()
        current_idx    = DIFFICULTY_LEVELS.index(self._difficulty)

        # Promotion (increase difficulty)
        if mean >= self._promote_threshold and current_idx < len(DIFFICULTY_LEVELS) - 1:
            self._difficulty = DIFFICULTY_LEVELS[current_idx + 1]
            self._window.clear()   # fresh window after change
            self.promotions += 1
            result.update(
                difficulty_after=self._difficulty,
                changed=True,
                reason=(
                    f"promoted: mean={mean:.3f} >= "
                    f"threshold={self._promote_threshold}"
                ),
            )
            logger.info(
                "Curriculum PROMOTED: %s → %s (mean=%.3f)",
                old_difficulty, self._difficulty, mean,
            )

        # Demotion (decrease difficulty)
        elif mean < self._demote_threshold and current_idx > 0:
            self._difficulty = DIFFICULTY_LEVELS[current_idx - 1]
            self._window.clear()
            self.demotions += 1
            result.update(
                difficulty_after=self._difficulty,
                changed=True,
                reason=(
                    f"demoted: mean={mean:.3f} < "
                    f"threshold={self._demote_threshold}"
                ),
            )
            logger.info(
                "Curriculum DEMOTED: %s → %s (mean=%.3f)",
                old_difficulty, self._difficulty, mean,
            )

        else:
            result["reason"] = (
                f"stable at {self._difficulty} (mean={mean:.3f})"
            )

        self.history.append(
            {"episode": self.total_episodes, "reward": episode_reward, **result}
        )
        return result

    # ── stats / reset ──────────────────────────────────────

    def get_stats(self) -> dict:
        """Return full curriculum statistics."""
        return {
            "current_difficulty": self._difficulty,
            "total_episodes":     self.total_episodes,
            "recent_mean":        self.get_recent_mean(),
            "promotions":         self.promotions,
            "demotions":          self.demotions,
            "window":             list(self._window),
            "promote_threshold":  self._promote_threshold,
            "demote_threshold":   self._demote_threshold,
        }

    def reset_stats(self) -> None:
        """Reset curriculum to initial state (easy, empty window)."""
        self._difficulty    = "easy"
        self._window.clear()
        self.total_episodes = 0
        self.promotions     = 0
        self.demotions      = 0
        self.history        = []


# ── Module-level singleton ─────────────────────────────────
# Shared across all sessions in the same server process.
# Resets when the server process restarts (e.g. HF Spaces rebuild).

_global_curriculum = AutoCurriculum(initial_difficulty="easy")


def get_curriculum() -> AutoCurriculum:
    """Get the global curriculum instance."""
    return _global_curriculum


def get_current_difficulty() -> str:
    """Shortcut: current difficulty from the global curriculum."""
    return _global_curriculum.get_difficulty()


def record_episode(reward: float) -> dict:
    """Shortcut: record an episode reward in the global curriculum."""
    return _global_curriculum.record(reward)


def get_curriculum_stats() -> dict:
    """Shortcut: stats from the global curriculum."""
    return _global_curriculum.get_stats()
