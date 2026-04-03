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
import random
import re
from statistics import mean
from typing import Any, List

from openai import OpenAI
from openai import APIStatusError
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
    max_tokens = int(os.environ.get("MAX_TOKENS", "80"))
    resp = llm.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.3,
        max_tokens=max_tokens,
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


def _extract_first_json_array_block(text: str) -> str | None:
    """Return the first balanced JSON array block from text, if present."""
    start = text.find("[")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


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
            array_block = _extract_first_json_array_block(reply)
            parsed_any = False
            ranked = []
            if array_block is not None:
                try:
                    parsed = json.loads(array_block)
                    parsed_any = True
                    if isinstance(parsed, list):
                        for item in parsed:
                            # Preferred shape: ["person", "car", ...]
                            if isinstance(item, str):
                                label = item.strip().lower()
                                if label:
                                    ranked.append(label)
                                continue
                            # Also accept: [{"label": "person", ...}, ...]
                            if isinstance(item, dict):
                                raw_label = item.get("label")
                                if raw_label is not None:
                                    label = str(raw_label).strip().lower()
                                    if label:
                                        ranked.append(label)
                except json.JSONDecodeError:
                    parsed_any = False

            if not ranked:
                if not parsed_any:
                    # Regex fallback intentionally captures only quoted values,
                    # not object keys like "label" / "confidence".
                    labels = re.findall(r':\s*"([a-z_ ]+)"', reply, flags=re.IGNORECASE)
                    if labels:
                        ranked = [lbl.strip().lower() for lbl in labels if lbl.strip()]
                if not ranked:
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

    n_seeds = int(os.environ.get("N_SEEDS", "3"))
    seeds = random.sample(range(100), n_seeds)
    exhausted_quota = False
    completed_episodes = 0
    enable_retry = os.environ.get("ENABLE_RETRY", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )

    with ArjunaEnv(base_url=base_url).sync() as env:
        for task in (1, 2, 3):
            if exhausted_quota:
                break
            for seed in seeds:
                if exhausted_quota:
                    break
                try:
                    rw = run_episode(env, llm, task, seed)
                except APIStatusError as exc:
                    if exc.status_code == 402:
                        exhausted_quota = True
                        print(
                            "Quota exhausted (HTTP 402 from inference provider). "
                            "Stopping run and reporting partial results."
                        )
                        break
                    raise
                except Exception as exc:
                    if not enable_retry:
                        raise
                    print(f"  episode error (task={task}, seed={seed}): {exc!r}, retrying once")
                    try:
                        rw = run_episode(env, llm, task, seed)
                    except APIStatusError as retry_exc:
                        if retry_exc.status_code == 402:
                            exhausted_quota = True
                            print(
                                "Quota exhausted (HTTP 402 from inference provider) "
                                "during retry. Stopping run and reporting partial results."
                            )
                            break
                        raise
                per_task[task].append(rw)
                all_rewards.append(rw)
                completed_episodes += 1

    print("---")
    for t in (1, 2, 3):
        m = mean(per_task[t]) if per_task[t] else 0.0
        print(f"task {t} mean reward: {m:.3f}")
    print(f"overall mean reward: {mean(all_rewards) if all_rewards else 0.0:.3f}")
    min_task_scores = {t: (min(per_task[t]) if per_task[t] else 0.0) for t in (1, 2, 3)}
    max_task_scores = {t: (max(per_task[t]) if per_task[t] else 0.0) for t in (1, 2, 3)}
    print(f"Run seeds used: {seeds}")
    print(f"Score variation: min={min_task_scores}, max={max_task_scores}")
    print(f"Episodes completed: {completed_episodes}/{len(seeds) * 3}")
    if exhausted_quota:
        print("Run ended early due to inference quota exhaustion.")


if __name__ == "__main__":
    main()
