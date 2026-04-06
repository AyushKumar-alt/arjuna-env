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

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

# Optional - if you use from_docker_image():
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")


TASK1_SYSTEM = """
You are ARJUNA, an autonomous robot vision system.
You are given a camera scene with one YOLO detection.
Identify the object and respond with its class label only.
One word or short phrase. No explanation.
""".strip()

TASK2_SYSTEM = """
You are ARJUNA, an autonomous robot at an industrial or urban site.
Your camera detected multiple objects simultaneously.

Rank them from most to least important to act on. Rules:
1. Higher confidence first (primary rule).
2. If confidence is identical: person > vehicle (bicycle, car, bus, truck, motorcycle, airplane, train, boat) > other.

Return ONLY a JSON array of label strings:
["label_a", "label_b", "label_c"]

MANDATORY: Answer with the list ONLY. No preamble, no explanation, no "To solve this..." text. Just the array.
""".strip()

TASK3_SYSTEM = """
You are ARJUNA, an autonomous robot making real-time perception decisions.
Your camera produced a low-confidence detection. You must decide what to do next.

Your three options:
- discard         — confidence is too low to trust; ignore this detection entirely
- request_rescan  — borderline confidence; ask for another sensor pass before deciding
- log_and_continue — confidence is acceptable; record and proceed with caution

Use the confidence score in the scene. Mandatory rules:
- confidence < 0.35: discard
- 0.35 <= confidence < 0.50: request_rescan
- confidence >= 0.50: log_and_continue

Return JSON only:
{"decision": "<choice>", "reasoning": "<short sentence mentioning the numeric value>"}
No other text.
""".strip()


def _client() -> OpenAI:
    return OpenAI(
        base_url=API_BASE_URL,
        api_key=HF_TOKEN,
    )


def _chat(llm: OpenAI, system: str, user: str) -> str:
    max_tokens = int(os.environ.get("MAX_TOKENS", "80"))
    resp = llm.chat.completions.create(
        model=MODEL_NAME,
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
    if not HF_TOKEN:
        print("Error: HF_TOKEN is not set.")
        return

    base_url = os.environ.get("ARJUNA_ENV_BASE_URL", "http://127.0.0.1:7860")
    llm = _client()

    per_task: dict[int, list[float]] = {1: [], 2: [], 3: []}
    all_episode_means: list[float] = []

    def run_episode(env: ArjunaEnv, llm_client: OpenAI, seed: int, ep_idx: int) -> float:
        reset_out = env.reset(seed=seed)
        obs = reset_out.observation

        print(f"START Episode {ep_idx} (seed={seed})")

        ep_meta: dict[str, Any] = {}
        if obs.episode_id:
            ep_meta["episode_id"] = obs.episode_id


        step_rewards: list[float] = []

        for sub in (1, 2, 3):
            scene_id = obs.scene_id
            task = obs.task_type
            user_prompt = obs.observation_text
            print(f"\n--- [STEP {sub}/3] AI IS READING ---\n{user_prompt}")

            if task == 1:
                system_prompt = TASK1_SYSTEM
            elif task == 2:
                system_prompt = TASK2_SYSTEM
            else:
                system_prompt = TASK3_SYSTEM

            reply = _chat(llm_client, system_prompt, user_prompt)
            print(f"\n--- [STEP {sub}/3] AI RAW REPLY ---\n{reply}")

            action: ArjunaAction
            if task == 1:
                label = reply.strip().lower()
                if not label:
                    label = parse_task1_label(reply)
                action = ArjunaAction(task1_label=label, metadata=ep_meta)
            elif task == 2:
                ranked: List[str] = []
                array_block = _extract_first_json_array_block(reply)
                parsed_any = False
                if array_block is not None:
                    try:
                        parsed = json.loads(array_block)
                        parsed_any = True
                        if isinstance(parsed, list):
                            for item in parsed:
                                if isinstance(item, str):
                                    label = item.strip().lower()
                                    if label:
                                        ranked.append(label)
                                    continue
                                if isinstance(item, dict) and "label" in item:
                                    raw_label = item.get("label")
                                    if raw_label is not None:
                                        label = str(raw_label).strip().lower()
                                        if label:
                                            ranked.append(label)
                    except json.JSONDecodeError:
                        parsed_any = False

                if not ranked:
                    if not parsed_any:
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

            print(f"\n--- [STEP {sub}/3] SUBMITTED ACTION ---\n{action.model_dump_json(indent=2)}\n")
            step_out = env.step(action)
            rw = episode_reward(step_out)
            step_rewards.append(rw)
            per_task[task].append(rw)
            print(f"STEP {sub}/3 reward={rw:.3f}")

            obs = step_out.observation
            if step_out.done:
                overall = (
                    obs.overall_reward
                    if obs.overall_reward is not None
                    else float(mean(step_rewards))
                )
                print(f"END Episode {ep_idx} reward={overall:.3f}")
                
                # Level 2: Fetch and print curriculum status
                try:
                    import requests
                    # Hack to parse base URL correctly if using standard format 
                    # Note: this works when the server is local, but we don't assume `base_url` format strictly,
                    # mostly used for local testing `http://127.0.0.1:7860/curriculum`
                    curr_url = "http://127.0.0.1:7860/curriculum"
                    if "localhost" in base_url or "127.0.0.1" in base_url:
                        host_port = base_url.replace("http://", "").split("/")[0]
                        curr_url = f"http://{host_port}/curriculum"
                        
                    curr_resp = requests.get(curr_url, timeout=2)
                    if curr_resp.status_code == 200:
                        curr = curr_resp.json()
                        print(
                            f"  Curriculum: difficulty={curr['current_difficulty']} "
                            f"| recent_mean={curr['recent_mean_reward']:.3f}"
                        )
                except Exception:
                    pass  # curriculum endpoint optional or unreachable

                return float(overall)

        overall = float(mean(step_rewards))
        print(f"END Episode {ep_idx} reward={overall:.3f}")
        return overall

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
        for ep_idx, seed in enumerate(seeds, start=1):
            if exhausted_quota:
                break
            try:
                overall = run_episode(env, llm, seed, ep_idx)
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
                print(f"  episode error (seed={seed}): {exc!r}, retrying once")
                try:
                    overall = run_episode(env, llm, seed, ep_idx)
                except APIStatusError as retry_exc:
                    if retry_exc.status_code == 402:
                        exhausted_quota = True
                        print(
                            "Quota exhausted (HTTP 402 from inference provider) "
                            "during retry. Stopping run and reporting partial results."
                        )
                        break
                    raise
            all_episode_means.append(overall)
            completed_episodes += 1

    print("---")
    for t in (1, 2, 3):
        m = mean(per_task[t]) if per_task[t] else 0.0
        print(f"task {t} mean reward: {m:.3f}")
    print(f"overall mean reward: {mean(all_episode_means) if all_episode_means else 0.0:.3f}")
    min_task_scores = {t: (min(per_task[t]) if per_task[t] else 0.0) for t in (1, 2, 3)}
    max_task_scores = {t: (max(per_task[t]) if per_task[t] else 0.0) for t in (1, 2, 3)}
    print(f"Run seeds used: {seeds}")
    print(f"Score variation: min={min_task_scores}, max={max_task_scores}")
    print(f"Episodes completed: {completed_episodes}/{len(seeds)}")
    if exhausted_quota:
        print("Run ended early due to inference quota exhaustion.")


if __name__ == "__main__":
    main()
