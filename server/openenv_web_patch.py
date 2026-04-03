"""
Patch OpenEnv's Gradio WebInterfaceManager so step() receives episode_id.

The default Playground calls env.step(action) only. ARJUNA keys episode state in
SESSIONS by episode_id (same contract as HTTP /step). Passing episode_id from
episode_state keeps /web aligned with Invoke-RestMethod / curl flows.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict


def apply() -> None:
    from openenv.core.env_server.serialization import (
        deserialize_action_with_preprocessing,
        serialize_observation,
    )
    from openenv.core.env_server.web_interface import ActionLog, WebInterfaceManager

    async def step_environment_with_episode_id(
        self: WebInterfaceManager, action_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        action = deserialize_action_with_preprocessing(action_data, self.action_cls)
        eid = self.episode_state.episode_id
        if isinstance(eid, str) and eid.strip():
            observation = await self._run_sync_in_thread_pool(
                self.env.step,
                action,
                episode_id=eid,
            )
        else:
            observation = await self._run_sync_in_thread_pool(
                self.env.step,
                action,
            )
        state = self.env.state
        serialized = serialize_observation(observation)
        action_log = ActionLog(
            timestamp=datetime.now().isoformat(),
            action=action.model_dump(exclude={"metadata"}),
            observation=serialized["observation"],
            reward=observation.reward,
            done=observation.done,
            step_count=state.step_count,
        )
        self.episode_state.episode_id = state.episode_id
        self.episode_state.step_count = state.step_count
        self.episode_state.current_observation = serialized["observation"]
        self.episode_state.action_logs.append(action_log)
        if len(self.episode_state.action_logs) > self.MAX_ACTION_LOGS:
            self.episode_state.action_logs = self.episode_state.action_logs[
                -self.MAX_ACTION_LOGS :
            ]
        self.episode_state.is_reset = False
        await self._send_state_update()
        return serialized

    WebInterfaceManager.step_environment = step_environment_with_episode_id
