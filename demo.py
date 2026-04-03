from __future__ import annotations

"""
Simple offline demo for the ARJUNA perception environment.

This script:
- starts a short episode loop against a running ARJUNA HTTP server
- uses a trivial, hand-written policy (no external LLMs)
- prints observations, actions, rewards, and final scores

Usage (with local server already running on port 7860):

    # In one terminal
    docker build -t arjuna-env .
    docker run -p 7860:7860 arjuna-env

    # In another terminal (inside this repo)
    python demo.py
"""

import os
from typing import Any, Dict

from client import ArjunaClient


def _heuristic_policy(task_type: int, obs_text: str) -> Dict[str, Any]:
    """
    Tiny, deterministic policy purely for demonstration.

    - Task 1: guess a generic label based on keywords
    - Task 2: returns a fixed ordering that often makes sense
    - Task 3: picks a decision based on the numeric confidence in the text
    """
    text = obs_text.lower()

    if task_type == 1:
        if "person" in text or "worker" in text or "pedestrian" in text:
            label = "person"
        elif "car" in text:
            label = "car"
        elif "truck" in text or "bus" in text:
            label = "truck"
        elif "bicycle" in text or "bike" in text:
            label = "bicycle"
        else:
            label = "object"
        return {"task1_label": label}

    if task_type == 2:
        # Very rough heuristic: always prioritize people, then vehicles, then others.
        ordered: list[str] = []
        if "person" in text or "worker" in text or "pedestrian" in text or "child" in text:
            ordered.append("person")
        if "truck" in text or "bus" in text:
            ordered.append("truck")
        if "car" in text:
            ordered.append("car")
        if "bicycle" in text or "bike" in text:
            ordered.append("bicycle")
        if not ordered:
            ordered.append("object")
        return {"ranked_objects": ordered}

    if task_type == 3:
        # Extract first floating-point number as "confidence".
        import re

        match = re.search(r"(\d\.\d+)", text)
        confidence = float(match.group(1)) if match else 0.3

        if confidence < 0.35:
            decision = "discard"
        elif confidence < 0.5:
            decision = "request_rescan"
        else:
            decision = "log_and_continue"

        return {
            "decision": decision,
            "reasoning": f"confidence {confidence:.2f} mapped to this band",
        }

    # Fallback empty action (should not happen with valid tasks)
    return {}


def run_demo(episodes: int = 5) -> None:
    base_url = os.environ.get("ARJUNA_ENV_BASE_URL", "http://127.0.0.1:7860")
    client = ArjunaClient(base_url=base_url)

    print(f"Connecting to ARJUNA environment at {base_url}")
    total_reward = 0.0

    for i in range(episodes):
        print(f"\n=== Episode {i + 1}/{episodes} ===")
        obs, _ = client.reset()
        episode_id = obs.episode_id

        print(f"task_type={obs.task_type} scene_id={obs.scene_id}")
        print(f"observation_text={obs.observation_text!r}")

        action_payload = _heuristic_policy(obs.task_type, obs.observation_text)
        print(f"action={action_payload}")

        step_obs, _ = client.step(action_payload, episode_id=episode_id)
        total_reward += step_obs.reward or 0.0

        print(f"reward={step_obs.reward} done={step_obs.done}")
        if step_obs.feedback:
            print(f"feedback={step_obs.feedback}")

    mean_reward = total_reward / max(episodes, 1)
    print(f"\nDemo finished: mean reward over {episodes} episodes = {mean_reward:.3f}")


if __name__ == "__main__":
    run_demo()

