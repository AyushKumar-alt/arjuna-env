from __future__ import annotations

"""
Offline demo for the ARJUNA perception environment (no LLM API key required).

Runs one full 3-step episode against a running ARJUNA HTTP server using a
small heuristic that parses observation_text (regex + confidence sorting).

Usage (with local server on port 7860):

    docker build -t arjuna-env .
    docker run -p 7860:7860 arjuna-env

    python demo.py
"""

import os
import re
from statistics import mean
from typing import List

from models import ArjunaAction

from client import ArjunaEnv


def _task1_label_from_obs(text: str) -> str:
    """Primary YOLO line: label='person' or label=\"person\"."""
    m = re.search(r"label\s*=\s*['\"]([^'\"]+)['\"]", text)
    if m:
        return m.group(1).strip().lower()
    return "object"


def _task2_ranked_from_obs(text: str) -> List[str]:
    """Extract label/confidence pairs and sort by confidence descending."""
    pairs: list[tuple[str, float]] = []
    for m in re.finditer(
        r"label\s*=\s*['\"]([^'\"]+)['\"]\s*,\s*confidence\s*=\s*([0-9.]+)",
        text,
        flags=re.IGNORECASE,
    ):
        pairs.append((m.group(1).strip().lower(), float(m.group(2))))
    if not pairs:
        for m in re.finditer(r"label\s*=\s*['\"]([^'\"]+)['\"]", text):
            pairs.append((m.group(1).strip().lower(), 0.5))
    pairs.sort(key=lambda x: x[1], reverse=True)
    return [p[0] for p in pairs]


def _task3_decision_from_obs(text: str) -> tuple[str, str]:
    m = re.search(r"confidence\s*=\s*([0-9.]+)", text, flags=re.IGNORECASE)
    confidence = float(m.group(1)) if m else 0.3
    if confidence < 0.35:
        decision = "discard"
    elif confidence < 0.5:
        decision = "request_rescan"
    else:
        decision = "log_and_continue"
    reasoning = f"parsed confidence {confidence:.3f} against policy bands"
    return decision, reasoning


def _heuristic_action(task_type: int, obs_text: str, episode_id: str | None) -> ArjunaAction:
    meta: dict[str, str] = {}
    if episode_id:
        meta["episode_id"] = episode_id

    if task_type == 1:
        return ArjunaAction(task1_label=_task1_label_from_obs(obs_text), metadata=meta)
    if task_type == 2:
        return ArjunaAction(ranked_objects=_task2_ranked_from_obs(obs_text), metadata=meta)
    decision, reasoning = _task3_decision_from_obs(obs_text)
    return ArjunaAction(decision=decision, reasoning=reasoning, metadata=meta)


def run_demo(episodes: int = 1) -> None:
    base_url = os.environ.get("ARJUNA_ENV_BASE_URL", "http://127.0.0.1:7860")

    print(f"Connecting to ARJUNA environment at {base_url}")

    with ArjunaEnv(base_url=base_url).sync() as client:
        for i in range(episodes):
            reset_out = client.reset()
            obs = reset_out.observation
            episode_id = obs.episode_id
            bundle = obs.bundle_name or "unknown"

            print(f"\n=== Episode Start (Bundle: {bundle}) ===")

            step_rewards: list[float] = []

            for sub in (1, 2, 3):
                task = obs.task_type
                scene = obs.scene_id
                title = (
                    "Single Object ID"
                    if task == 1
                    else "Multi Object Triage"
                    if task == 2
                    else "Low Confidence Decision"
                )

                action = _heuristic_action(task, obs.observation_text, episode_id)

                print(f"\nStep {sub}/3 — {title}")
                short_scene = obs.observation_text.replace("\n", " ")[:120]
                print(f"  Scene: {short_scene}…")
                if task == 1:
                    print(f"  Action: task1_label={action.task1_label!r}")
                elif task == 2:
                    print(f"  Action: ranked_objects={action.ranked_objects!r}")
                else:
                    print(f"  Action: decision={action.decision!r}")

                step_out = client.step(action)
                r = step_out.reward
                if r is None:
                    r = step_out.observation.reward
                r = float(r or 0.0)
                step_rewards.append(r)
                fb = step_out.observation.feedback or ""
                first_line = fb.splitlines()[0] if fb else ""
                print(f"  Reward: {r:.3f} | Feedback: {first_line}")

                obs = step_out.observation
                if step_out.done:
                    overall = (
                        obs.overall_reward
                        if obs.overall_reward is not None
                        else float(mean(step_rewards))
                    )
                    print(f"\n=== Episode Complete ===")
                    print(f"Overall reward: {overall:.3f}")
                    print("==================")
                    break


if __name__ == "__main__":
    run_demo(1)
