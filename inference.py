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
from typing import Any, List

from openai import OpenAI
from openenv.core.client_types import StepResult

from client import ArjunaEnv
from models import ArjunaAction, ArjunaObservation
from server.tasks import extract_json_list

TASK1_SYSTEM = """
You are ARJUNA, an autonomous robot vision system.
You are given a camera scene with one YOLO detection.
Identify the object and respond with its class label only.
One word or short phrase. No explanation.
""".strip()

TASK2_SYSTEM = """
You are ARJUNA, an autonomous robot at an industrial or urban site.
Your camera detected multiple objects simultaneously.
Prioritize which objects need attention first based on:
- How safety-critical the object type is
- How reliable the detection appears to be

Return a JSON array of object labels, most important first.
Example: ["person", "car", "bicycle"]
Return the array only. No explanation.
""".strip()

TASK3_SYSTEM = """
You are ARJUNA, an autonomous robot making real-time perception decisions.
Your camera produced a detection with a confidence score between 0.0 and 1.0.

Use this scale to decide:

CONFIDENCE SCALE (0.0 to 1.0):
- 0.0 to 0.4  → VERY LOW  → action: discard
- 0.4 to 0.5  → MODERATE  → action: request_rescan
- 0.5 to 1.0  → HIGH      → action: log_and_continue

Steps:
1. Find the confidence number in the scene description
2. Look at the scale above
3. Pick the matching action

Return JSON only: 
{"decision": "<choice>", "reasoning": "<one sentence>"}
No other text.
""".strip()


def _client() -> OpenAI:
    return OpenAI(
        base_url=os.environ["API_BASE_URL"],
        api_key=os.environ["HF_TOKEN"],
    )


def _chat(llm: OpenAI, system: str, user: str) -> str:
    model = os.environ["MODEL_NAME"]
    resp = llm.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.3,
        max_tokens=200,
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
    text = text.strip()
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "decision" in data:
            decision_raw = str(data["decision"])
            reasoning = str(data.get("reasoning", "")).strip()
            decision = decision_raw.strip().lower().replace("-", "_")
            if decision not in ("log_and_continue", "request_rescan", "discard"):
                decision = "discard"
            return decision, reasoning
    except json.JSONDecodeError:
        pass

    norm = text.lower().replace("-", "_")
    for token in ("log_and_continue", "request_rescan", "discard"):
        if token in norm:
            return token, text.strip()

    match = re.search(r"(log_and_continue|discard|request_rescan)", norm)
    if match:
        return match.group(1), text.strip()

    return "discard", text.strip()


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

    def run_episode(
        env: ArjunaEnv,
        llm_client: OpenAI,
        task: int,
        seed: int,
    ) -> float:
        reset_out = env.reset(task_type=task, seed=seed)
        obs = reset_out.observation

        user_prompt: str
        system_prompt: str

        if task == 1:
            system_prompt = TASK1_SYSTEM
            user_prompt = obs.observation_text
        elif task == 2:
            system_prompt = TASK2_SYSTEM
            user_prompt = obs.observation_text
        else:
            system_prompt = TASK3_SYSTEM
            user_prompt = obs.observation_text

        ep_meta: dict[str, str] = {}
        if obs.episode_id:
            ep_meta["episode_id"] = obs.episode_id

        reply = _chat(llm_client, system_prompt, user_prompt)
        print(f"task={task} seed={seed} scene={obs.scene_id}")
        print(f"  LLM raw response: {reply[:200]}")

        action: ArjunaAction
        if task == 1:
            label = reply.strip().lower()
            if not label:
                label = parse_task1_label(reply)
            action = ArjunaAction(task1_label=label, metadata=ep_meta)
        elif task == 2:
            ranked: List[str]
            try:
                parsed = json.loads(reply)
                if isinstance(parsed, list):
                    ranked = [str(x).strip().lower() for x in parsed if str(x).strip()]
                else:
                    ranked = parse_task2_ranking(reply)
            except json.JSONDecodeError:
                labels = re.findall(r'"([a-z_ ]+)"', reply, flags=re.IGNORECASE)
                if labels:
                    ranked = [lbl.strip().lower() for lbl in labels if lbl.strip()]
                else:
                    ranked = parse_task2_ranking(reply)
            action = ArjunaAction(ranked_objects=ranked, metadata=ep_meta)
        else:
            decision, reasoning = parse_task3_decision(reply)
            action = ArjunaAction(
                decision=decision,
                reasoning=reasoning,
                metadata=ep_meta,
            )

        step_out = env.step(action)
        rw = episode_reward(step_out)
        print(f"  reward={rw:.3f} done={step_out.done}")
        return rw

    seeds = [0, 1, 2, 3, 4]

    with ArjunaEnv(base_url=base_url).sync() as env:
        for task in (1, 2, 3):
            for seed in seeds:
                try:
                    rw = run_episode(env, llm, task, seed)
                except Exception as exc:
                    print(f"  episode error (task={task}, seed={seed}): {exc!r}, retrying once")
                    rw = run_episode(env, llm, task, seed)
                per_task[task].append(rw)
                all_rewards.append(rw)

    print("---")
    for t in (1, 2, 3):
        m = mean(per_task[t]) if per_task[t] else 0.0
        print(f"task {t} mean reward: {m:.3f}")
    print(f"overall mean reward: {mean(all_rewards) if all_rewards else 0.0:.3f}")


if __name__ == "__main__":
    main()
