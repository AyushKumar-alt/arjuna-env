"""
Typed WebSocket client for the ARJUNA perception environment.
"""

from __future__ import annotations

from typing import Any, Dict

from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient

from models import ArjunaAction, ArjunaObservation, ArjunaState


class ArjunaEnv(EnvClient[ArjunaAction, ArjunaObservation, ArjunaState]):
    """Connects to a running arjuna-perception-env server (local or Hugging Face Space)."""

    def _step_payload(self, action: ArjunaAction) -> Dict[str, Any]:
        return action.model_dump(exclude_none=True)

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[ArjunaObservation]:
        obs_raw = payload.get("observation") or {}
        return StepResult(
            observation=ArjunaObservation.model_validate(obs_raw),
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> ArjunaState:
        return ArjunaState.model_validate(payload)
