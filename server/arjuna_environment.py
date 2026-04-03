"""
ARJUNA perception environment: reset → observe → step (grade) → done.
"""

from __future__ import annotations

import random
import uuid
from typing import Any

from pathlib import Path

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import EnvironmentMetadata

from models import ArjunaAction, ArjunaObservation, ArjunaState

from . import synthetic_data as sd
from .tasks import (
    format_task1_prompt,
    format_task2_prompt,
    format_task3_prompt,
    grade_task1_identification,
    grade_task2_triage,
    grade_task3_low_confidence,
)


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


class ArjunaEnvironment(Environment):
    """Simulated perception episodes with three task families."""

    SUPPORTS_CONCURRENT_SESSIONS = True

    _META_NAME = "arjuna-perception-env"
    _META_DESCRIPTION = (
        "A simulated robot perception RL environment where an AI agent acts as the decision brain "
        "of ARJUNA autonomous robot. The agent receives camera scene descriptions and must identify "
        "objects, triage multi-object scenes, and make low-confidence decisions across 3 tasks of "
        "increasing difficulty."
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
        self._task: int | None = None
        self._scene1: sd.Task1Scene | None = None
        self._scene2: sd.Task2Scene | None = None
        self._scene3: sd.Task3Scene | None = None

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

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        **kwargs: Any,
    ) -> ArjunaObservation:
        if seed is not None:
            self._rng.seed(seed)
        eid = episode_id or str(uuid.uuid4())
        self._state = ArjunaState(
            episode_id=eid,
            step_count=0,
            awaiting_action=True,
        )

        forced = kwargs.get("task_type")
        if forced in (1, 2, 3):
            task = int(forced)
        else:
            task = self._rng.choice((1, 2, 3))
        self._task = task
        self._scene1 = self._scene2 = self._scene3 = None

        if task == 1:
            self._scene1 = self._rng.choice(sd.TASK1_SCENES)
            self._state.task_type = 1
            self._state.scene_id = self._scene1.scene_id
            text = format_task1_prompt(self._scene1)
            sid = self._scene1.scene_id
        elif task == 2:
            self._scene2 = self._rng.choice(sd.TASK2_SCENES)
            self._state.task_type = 2
            self._state.scene_id = self._scene2.scene_id
            text = format_task2_prompt(self._scene2)
            sid = self._scene2.scene_id
        else:
            self._scene3 = self._rng.choice(sd.TASK3_SCENES)
            self._state.task_type = 3
            self._state.scene_id = self._scene3.scene_id
            text = format_task3_prompt(self._scene3)
            sid = self._scene3.scene_id

        return ArjunaObservation(
            task_type=task,  # type: ignore[arg-type]
            scene_id=sid,
            observation_text=text,
            feedback="",
            reward=0.0,
            done=False,
        )

    def step(
        self,
        action: ArjunaAction,
        timeout_s: float | None = None,
        **kwargs: Any,
    ) -> ArjunaObservation:
        if not self._state.awaiting_action or self._task is None:
            return ArjunaObservation(
                task_type=1,
                scene_id="invalid",
                observation_text="",
                feedback="Call reset() before step().",
                reward=0.0,
                done=True,
            )

        self._state.step_count += 1
        task = self._task
        reward = 0.0
        feedback = ""

        if task == 1 and self._scene1 is not None:
            reward = grade_task1_identification(action.task1_label, self._scene1)
            feedback = (
                f"Expected '{self._scene1.expected_label}', "
                f"got {action.task1_label!r}. Score={reward:.2f}."
            )
        elif task == 2 and self._scene2 is not None:
            reward = grade_task2_triage(action.ranked_objects, self._scene2)
            feedback = (
                f"Ground truth order: {list(self._scene2.expected_priority)}; "
                f"agent: {action.ranked_objects}. Score={reward:.2f}."
            )
        elif task == 3 and self._scene3 is not None:
            reward = grade_task3_low_confidence(
                action.decision,
                action.reasoning,
                self._scene3,
            )
            feedback = (
                f"Band expects {self._scene3.expected_action!r}; "
                f"decision={action.decision!r}. Score={reward:.2f}."
            )
        else:
            feedback = "Internal error: missing scene."

        self._state.awaiting_action = False
        obs_text = ""
        if task == 1 and self._scene1:
            obs_text = format_task1_prompt(self._scene1)
        elif task == 2 and self._scene2:
            obs_text = format_task2_prompt(self._scene2)
        elif task == 3 and self._scene3:
            obs_text = format_task3_prompt(self._scene3)

        return ArjunaObservation(
            task_type=task,  # type: ignore[arg-type]
            scene_id=self._state.scene_id or "unknown",
            observation_text=obs_text,
            feedback=feedback,
            reward=float(reward),
            done=True,
        )

    @property
    def state(self) -> ArjunaState:
        return self._state
