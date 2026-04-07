"""
ARJUNA perception environment: reset → observe → step (grade) → done.
Each episode has 3 sequential steps (identify → triage → low-confidence decision).
"""

from __future__ import annotations

import logging
import os
import random
import uuid
from dataclasses import dataclass
from statistics import mean
from typing import Any

from pathlib import Path

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import EnvironmentMetadata

from models import ArjunaAction, ArjunaObservation, ArjunaState

from . import synthetic_data as sd
from .tasks import (
    format_step_observation,
    format_task1_prompt,
    format_task2_prompt,
    format_task3_prompt,
    grade_task1_identification,
    grade_task2_triage,
    grade_task3_low_confidence,
)

logger = logging.getLogger(__name__)


def _load_readme_text() -> str | None:
    """Load README for /metadata; supports Docker (/app) and local dev layouts."""
    candidates = (
        Path("/app/README.md"),
        Path(__file__).resolve().parent.parent / "README.md",
    )
    for path in candidates:
        if path.is_file():
            try:
                raw = path.read_text(encoding="utf-8")
            except OSError:
                continue
            if raw.lstrip().startswith("---"):
                parts = raw.split("---", 2)
                if len(parts) >= 3:
                    return parts[2].lstrip("\n")
            return raw
    return None


@dataclass
class _EpisodeSession:
    """Server-side episode state for HTTP (survives across per-request env instances)."""

    next_step: int  # 1, 2, or 3 — which step the incoming action answers
    step_rewards: list[float]
    bundle: sd.EpisodeBundle


# Keyed by episode_id; updated across steps, removed when the episode completes.
SESSIONS: dict[str, _EpisodeSession] = {}


class ArjunaEnvironment(Environment):
    """Simulated perception episodes with three sequential tasks per episode."""

    SUPPORTS_CONCURRENT_SESSIONS = True

    _META_NAME = "arjuna-perception-env"
    _META_DESCRIPTION = (
        "A 3-step sequential robot perception environment. Each episode progresses through "
        "object identification (easy), multi-object triage (medium), and low-confidence "
        "decision making (hard). The agent receives YOLO-style scene descriptions and must "
        "make correct perception decisions. Rewards range 0.0–1.0 with partial credit and "
        "semantic grading."
    )
    _META_VERSION = "1.0.0"
    _META_AUTHOR = "Calpol500mg"
    _META_DOCS_URL = "https://huggingface.co/spaces/Calpol500mg/arjuna-env"

    def __init__(self) -> None:
        super().__init__()
        self._rng = random.Random()
        self._state = ArjunaState(
            episode_id=str(uuid.uuid4()),
            step_count=0,
            task_type=None,
            scene_id=None,
            awaiting_action=False,
        )

    def get_metadata(self) -> EnvironmentMetadata:
        """Expose name, description, and docs for GET /metadata (OpenEnv)."""
        readme_raw = _load_readme_text()
        return EnvironmentMetadata(
            name=self._META_NAME,
            description=self._META_DESCRIPTION,
            readme_content=readme_raw,
            version=self._META_VERSION,
            author=self._META_AUTHOR,
            documentation_url=self._META_DOCS_URL,
        )

    def _pick_bundle(self, seed: int | None) -> sd.EpisodeBundle:
        """Pick a bundle: try LLM-generated first, fall back to hardcoded."""
        difficulty = "medium"  # safe default
        generated_bundle = None

        # Level 1: try dynamic generation if enabled
        if os.environ.get("ENABLE_DYNAMIC_SCENES", "false").lower() in ("true", "1", "yes"):
            try:
                from server.scene_generator import generate_episode_bundle
                from server.curriculum import get_current_difficulty
                difficulty = get_current_difficulty()
                generated_bundle = generate_episode_bundle(difficulty=difficulty, seed=seed)
                if generated_bundle is not None:
                    logger.info(
                        "Using generated bundle: %s  difficulty=%s",
                        generated_bundle.bundle_id, difficulty,
                    )
                    return generated_bundle
            except Exception as exc:  # pragma: no cover
                logger.warning("Dynamic scene generation skipped: %s. Using fallback.", exc)
                generated_bundle = None

        # Fallback: hardcoded synthetic_data.py bundles
        if seed is not None:
            self._rng.seed(seed)
            
        # Map static bundles to matching offline curriculum tiers
        from server.curriculum import get_current_difficulty
        fallback_difficulty = get_current_difficulty()
        
        easy_names = {"Forest Trail", "Office Lobby", "Airport"}
        medium_names = {"Urban Street", "Parking Lot", "School Zone", "Hospital Entrance"}
        hard_names = {"Construction Site", "Night Street", "Rainy Street", "Shopping Mall", "Warehouse"}
        
        valid_bundles = []
        for b in sd.EPISODE_BUNDLES:
            if fallback_difficulty == "easy" and b.name in easy_names:
                valid_bundles.append(b)
            elif fallback_difficulty == "hard" and b.name in hard_names:
                valid_bundles.append(b)
            elif fallback_difficulty == "medium" and b.name in medium_names:
                valid_bundles.append(b)
                
        # If something went wrong, failsafe to all bundles
        if not valid_bundles:
            valid_bundles = sd.EPISODE_BUNDLES
            
        bundle = self._rng.choice(valid_bundles)
        logger.debug("Using synthetic bundle: %s mapped to tier: %s", bundle.bundle_id, fallback_difficulty)
        return bundle

    def _observation_for_active_step(self, bundle: sd.EpisodeBundle, active: int) -> tuple[str, str, int]:
        if active == 1:
            inner = format_task1_prompt(bundle.task1)
            sid = bundle.task1.scene_id
            tt = 1
        elif active == 2:
            inner = format_task2_prompt(bundle.task2)
            sid = bundle.task2.scene_id
            tt = 2
        else:
            inner = format_task3_prompt(bundle.task3)
            sid = bundle.task3.scene_id
            tt = 3
        text = format_step_observation(bundle.name, active, inner)
        return text, sid, tt

    def _grader_detail(self, step: int, reward: float, bundle: sd.EpisodeBundle, action: ArjunaAction) -> str:
        if step == 1:
            return (
                f"Expected '{bundle.task1.expected_label}', "
                f"got {action.task1_label!r}. Score={reward:.2f}."
            )
        if step == 2:
            return (
                f"Ground truth order: {list(bundle.task2.expected_priority)}; "
                f"agent: {action.ranked_objects}. Score={reward:.2f}."
            )
        return (
            f"Band expects {bundle.task3.expected_action!r}; "
            f"decision={action.decision!r}. Score={reward:.2f}."
        )

    def _grade_action(self, step: int, bundle: sd.EpisodeBundle, action: ArjunaAction) -> float:
        if step == 1:
            return float(grade_task1_identification(action.task1_label, bundle.task1))
        if step == 2:
            return float(grade_task2_triage(action.ranked_objects, bundle.task2))
        return float(
            grade_task3_low_confidence(
                action.decision,
                action.reasoning,
                bundle.task3,
            )
        )

    def _step_banner(self, step: int, reward: float, done: bool, overall: float | None) -> str:
        if not done:
            return f"Step {step}/3 complete. Reward: {reward:.3f}. Move to next step."
        assert overall is not None
        return (
            f"Step 3/3 complete. Reward: {reward:.3f}. Episode done. Overall: {overall:.3f}"
        )

    def _run_step(
        self,
        sess: _EpisodeSession,
        action: ArjunaAction,
        episode_id: str,
        sync_state: bool,
    ) -> ArjunaObservation:
        step = sess.next_step
        bundle = sess.bundle
        reward = self._grade_action(step, bundle, action)
        detail = self._grader_detail(step, reward, bundle, action)
        sess.step_rewards.append(reward)

        if step < 3:
            sess.next_step = step + 1
            active = sess.next_step
            obs_text, sid, task_type = self._observation_for_active_step(bundle, active)
            banner = self._step_banner(step, reward, done=False, overall=None)
            feedback = f"{banner}\n{detail}"
            if sync_state and self._state.episode_id == episode_id:
                self._state.step_count += 1
                self._state.steps_completed = step
                self._state.step_rewards = list(sess.step_rewards)
                self._state.task_type = task_type
                self._state.scene_id = sid
                self._state.awaiting_action = True
                self._state.bundle_theme = bundle.name
            SESSIONS[episode_id] = sess
            return ArjunaObservation(
                episode_id=episode_id,
                task_type=task_type,  # type: ignore[arg-type]
                scene_id=sid,
                step_number=active,  # type: ignore[arg-type]
                bundle_name=bundle.name,
                observation_text=obs_text,
                feedback=feedback,
                reward=float(reward),
                overall_reward=None,
                done=False,
            )

        # Final step
        overall = float(mean(sess.step_rewards))
        banner = self._step_banner(3, reward, done=True, overall=overall)
        feedback = f"{banner}\n{detail}"
        obs_text, sid, task_type = self._observation_for_active_step(bundle, 3)
        SESSIONS.pop(episode_id, None)

        # Level 2: record episode reward for auto-curriculum
        try:
            from server.curriculum import record_episode
            curriculum_result = record_episode(overall)
            logger.info(
                "Curriculum update: %s → difficulty now: %s",
                curriculum_result["reason"],
                curriculum_result["difficulty_after"],
            )
        except Exception as exc:  # pragma: no cover
            logger.warning("Curriculum record skipped: %s", exc)

        if sync_state and self._state.episode_id == episode_id:
            self._state.step_count += 1
            self._state.steps_completed = 3
            self._state.step_rewards = list(sess.step_rewards)
            self._state.task_type = task_type
            self._state.scene_id = sid
            self._state.awaiting_action = False
            self._state.bundle_theme = bundle.name
        return ArjunaObservation(
            episode_id=episode_id,
            task_type=task_type,  # type: ignore[arg-type]
            scene_id=sid,
            step_number=3,  # type: ignore[arg-type]
            bundle_name=bundle.name,
            observation_text=obs_text,
            feedback=feedback,
            reward=float(reward),
            overall_reward=overall,
            done=True,
        )

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        **kwargs: Any,
    ) -> ArjunaObservation:
        _ = kwargs.get("task_type")
        eid = episode_id or str(uuid.uuid4())
        bundle = self._pick_bundle(seed)

        self._state = ArjunaState(
            episode_id=eid,
            step_count=0,
            awaiting_action=True,
            steps_completed=0,
            step_rewards=[],
            bundle_theme=bundle.name,
        )

        sess = _EpisodeSession(next_step=1, step_rewards=[], bundle=bundle)
        SESSIONS[eid] = sess

        obs_text, sid, task_type = self._observation_for_active_step(bundle, 1)
        self._state.task_type = task_type
        self._state.scene_id = sid

        return ArjunaObservation(
            episode_id=eid,
            task_type=task_type,  # type: ignore[arg-type]
            scene_id=sid,
            step_number=1,  # type: ignore[arg-type]
            bundle_name=bundle.name,
            observation_text=obs_text,
            feedback="",
            reward=0.0,
            overall_reward=None,
            done=False,
        )

    def _step_error_obs(self) -> ArjunaObservation:
        return ArjunaObservation(
            episode_id=None,
            task_type=1,
            scene_id="invalid",
            step_number=1,
            bundle_name=None,
            observation_text="",
            feedback="Call reset() before step().",
            reward=0.0,
            overall_reward=None,
            done=True,
        )

    def step(
        self,
        action: ArjunaAction,
        timeout_s: float | None = None,
        **kwargs: Any,
    ) -> ArjunaObservation:
        eid = kwargs.get("episode_id")
        if eid is None and action.metadata:
            eid = action.metadata.get("episode_id")
        if eid is None:
            eid = self._state.episode_id
        if not isinstance(eid, str) or not eid:
            return self._step_error_obs()
        if eid not in SESSIONS:
            return self._step_error_obs()
        return self._run_step(SESSIONS[eid], action, eid, sync_state=True)

    @property
    def state(self) -> ArjunaState:
        return self._state
