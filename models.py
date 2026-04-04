"""
Pydantic models for ARJUNA perception OpenEnv (actions, observations, state).
"""

from __future__ import annotations

import json
from typing import Any, Literal

from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field, field_validator


class ArjunaState(State):
    """Episode state exposed via state(); extra fields are allowed on State."""

    task_type: int | None = Field(
        default=None,
        description="Active task id: 1=identification, 2=triage, 3=low-confidence policy.",
    )
    scene_id: str | None = Field(default=None, description="Synthetic scene identifier.")
    awaiting_action: bool = Field(
        default=False,
        description="True after reset until the episode completes (after 3 steps).",
    )
    steps_completed: int = Field(
        default=0,
        ge=0,
        description="How many graded steps finished in the current episode (0–3).",
    )
    step_rewards: list[float] = Field(
        default_factory=list,
        description="Per-step rewards accumulated so far in the episode.",
    )
    bundle_theme: str | None = Field(
        default=None,
        description="Human-readable name of the episode bundle (shared location/theme).",
    )


class ArjunaAction(Action):
    """
    Unified action: set fields matching the active task (see task_type on observation/state).
    """

    task1_label: str | None = Field(
        default=None,
        description="Task 1: predicted YOLO class for the single visible object.",
    )
    ranked_objects: list[str] | None = Field(
        default=None,
        description="Task 2: ordered list of class labels, most important first.",
    )
    decision: str | None = Field(
        default=None,
        description="Task 3: one of log_and_continue, discard, request_rescan.",
    )
    reasoning: str | None = Field(
        default=None,
        description="Task 3 optional explanation (used for partial credit).",
    )

    @field_validator("ranked_objects", mode="before")
    @classmethod
    def _coerce_ranked_objects(cls, v: Any) -> list[str] | None:
        """Gradio sends a single text line; accept JSON array or comma-separated labels."""
        if v is None or v == "":
            return None
        if isinstance(v, list):
            return [str(x).strip() for x in v if str(x).strip()]
        if isinstance(v, str):
            s = v.strip()
            if not s:
                return None
            if s.startswith("["):
                try:
                    parsed = json.loads(s)
                except json.JSONDecodeError:
                    return None
                if isinstance(parsed, list):
                    return [str(x).strip() for x in parsed if str(x).strip()]
                return None
            return [p.strip() for p in s.split(",") if p.strip()]
        return v


class ArjunaObservation(Observation):
    """What the agent sees after reset or step."""

    episode_id: str | None = Field(
        default=None,
        description="Episode id from reset; pass HTTP /step as episode_id to correlate state.",
    )
    task_type: Literal[1, 2, 3] = Field(..., description="Which task is active.")
    scene_id: str = Field(..., description="Synthetic scene id.")
    step_number: Literal[1, 2, 3] = Field(
        default=1,
        description="Which step of the 3-step episode this observation corresponds to.",
    )
    bundle_name: str | None = Field(
        default=None,
        description="Episode bundle theme (coherent location across the three tasks).",
    )
    observation_text: str = Field(
        ...,
        description="Natural-language scene + detections for the LLM or planner.",
    )
    feedback: str = Field(
        default="",
        description="Human-readable grader message (populated after step).",
    )
    reward: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Reward for the step just graded, or 0.0 on reset before any step.",
    )
    overall_reward: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Mean of the three step rewards; set only when done=True after step 3.",
    )
    done: bool = Field(default=False, description="True when the episode ended.")
