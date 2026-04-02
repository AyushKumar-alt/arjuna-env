"""
Baseline LLM agent for ARJUNA perception OpenEnv (hackathon mandatory).

Requires:
  API_BASE_URL   — OpenAI-compatible base URL (e.g. Hugging Face Inference)
  MODEL_NAME     — model id (e.g. meta-llama/Llama-3.3-70B-Instruct)
  HF_TOKEN       — API key for the provider

Optional:
  ARJUNA_ENV_BASE_URL — OpenEnv server (default http://127.0.0.1:7860)
"""

from __future__ import annotations

import json
import os
import re
from statistics import mean
from typing import List

from openai import OpenAI
from openenv.core.client_types import StepResult

from client import ArjunaEnv
from models import ArjunaAction, ArjunaObservation
from server.tasks import extract_json_list

SYSTEM = (
    "You are the perception policy for autonomous robot ARJUNA. "
    "Follow each TASK block exactly. Be concise."
)


def _client() -> OpenAI:
    return OpenAI(
        base_url=os.environ["API_BASE_URL"],
        api_key=os.environ["HF_TOKEN"],
    )


def _chat(llm: OpenAI, user: str) -> str:
    model = os.environ["MODEL_NAME"]
    resp = llm.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
        ],
        temperature=0.1,
        max_tokens=512,
    )
    return (resp.choices[0].message.content or "").strip()


def parse_task1_label(text: str) -> str:
    text = text.strip()
    if text.startswith("{"):
        try:
            data = json.loads(text)
            if isinstance(data, dict) and "label" in data:
                return str(data["label"]).strip().lower()
        except json.JSONDecodeError:
            pass
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    candidate = lines[-1] if lines else text
    candidate = candidate.strip('"`')
    # strip trailing punctuation
    candidate = re.sub(r"^[^\w]+|[^\w]+$", "", candidate, flags=re.UNICODE)
    if not candidate:
        return "unknown"
    parts = candidate.split()
    if len(parts) == 1:
        return parts[0].lower()
    # Prefer last token if it looks like a class name (single word answer)
    return parts[-1].strip(",.").lower()


def parse_task2_ranking(text: str) -> List[str]:
    parsed = extract_json_list(text)
    if parsed:
        return [x.strip().lower() for x in parsed]
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for line in reversed(lines):
        if "," in line:
            return [p.strip().lower().strip("'\"") for p in line.split(",") if p.strip()]
    if "," in text:
        return [p.strip().lower() for p in text.split(",") if p.strip()]
    return [text.strip().lower()]


def parse_task3_decision(text: str) -> tuple[str, str]:
    norm = text.lower().replace("-", "_")
    decision: str | None = None
    for token in ("log_and_continue", "request_rescan", "discard"):
        if token in norm:
            decision = token
            break
    if decision is None:
        decision = "discard"
    return decision, text.strip()


def episode_reward(step_result: StepResult[ArjunaObservation]) -> float:
    r = step_result.reward
    if r is not None:
        return float(r)
    return float(step_result.observation.reward or 0.0)


def main() -> None:
    _ = os.environ["API_BASE_URL"]
    _ = os.environ["MODEL_NAME"]
    _ = os.environ["HF_TOKEN"]

    base_url = os.environ.get("ARJUNA_ENV_BASE_URL", "http://127.0.0.1:7860")
    llm = _client()

    per_task: dict[int, list[float]] = {1: [], 2: [], 3: []}
    all_rewards: list[float] = []

    with ArjunaEnv(base_url=base_url).sync() as env:
        for task in (1, 2, 3):
            for seed in (0, 1, 2):
                reset_out = env.reset(task_type=task, seed=seed)
                obs = reset_out.observation
                reply = _chat(llm, obs.observation_text)

                if task == 1:
                    label = parse_task1_label(reply)
                    action = ArjunaAction(task1_label=label)
                elif task == 2:
                    ranked = parse_task2_ranking(reply)
                    action = ArjunaAction(ranked_objects=ranked)
                else:
                    dec, reason = parse_task3_decision(reply)
                    action = ArjunaAction(decision=dec, reasoning=reason)

                step_out = env.step(action)
                rw = episode_reward(step_out)
                per_task[task].append(rw)
                all_rewards.append(rw)
                print(
                    f"task={task} seed={seed} scene={obs.scene_id} "
                    f"reward={rw:.3f} done={step_out.done}"
                )

    print("---")
    for t in (1, 2, 3):
        m = mean(per_task[t]) if per_task[t] else 0.0
        print(f"task {t} mean reward: {m:.3f}")
    print(f"overall mean reward: {mean(all_rewards) if all_rewards else 0.0:.3f}")


if __name__ == "__main__":
    main()
