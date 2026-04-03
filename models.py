"""
Pydantic models for ARJUNA perception OpenEnv (actions, observations, state).
"""

from __future__ import annotations

from typing import Literal

from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field


class ArjunaState(State):
    """Episode state exposed via state(); extra fields are allowed on State."""

    task_type: int | None = Field(
        default=None,
        description="Active task id: 1=identification, 2=triage, 3=low-confidence policy.",
    )
    scene_id: str | None = Field(default=None, description="Synthetic scene identifier.")
    awaiting_action: bool = Field(
        default=False,
        description="True after reset until the first step completes.",
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


class ArjunaObservation(Observation):
    """What the agent sees after reset or step."""

    episode_id: str | None = Field(
        default=None,
        description="Episode id from reset; pass HTTP /step as episode_id to correlate state.",
    )
    task_type: Literal[1, 2, 3] = Field(..., description="Which task is active.")
    scene_id: str = Field(..., description="Synthetic scene id.")
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
        description="Last reward in [0,1].",
    )
    done: bool = Field(default=False, description="True when the episode ended.")
