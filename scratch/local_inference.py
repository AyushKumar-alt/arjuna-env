"""
Local (offline) inference test — no API key, no HF token required.

Simulates an LLM agent with INTENTIONAL variation to show the full reward
range across tasks and episodes.  Runs like inference.py but uses a built-in
stochastic mock policy instead of a real model call.

Usage:
    python local_inference.py            # 5 random episodes
    python local_inference.py --seeds 3  # custom episode count

The mock policy randomly mixes:
  - Task 1: exact match, same-category, wrong-category, unknown
  - Task 2: all-correct, n-1 correct, n-2 correct, fully wrong
  - Task 3: correct decision with strong/weak/no reasoning, adjacent-band, wrong
"""

from __future__ import annotations

import argparse
import random
from statistics import mean
from typing import Any

from client import ArjunaEnv
from models import ArjunaAction, ArjunaObservation
from server.synthetic_data import (
    EPISODE_BUNDLES,
    VEHICLE_LABELS,
)
from server.tasks import (
    extract_json_list,
    grade_task1_identification,
    grade_task2_triage,
    grade_task3_low_confidence,
)

# ---------------------------------------------------------------------------
# Mock policy helpers
# ---------------------------------------------------------------------------

_PERSON_LABELS = ["person", "man", "woman", "pedestrian"]
_ANIMAL_LABELS = ["dog", "cat", "bird", "horse"]
_WRONG_LABELS = ["hydrant", "clock", "toothbrush", "hair drier"]


def _mock_task1(obs_text: str, rng: random.Random) -> str:
    """
    Parse the gold label from obs_text, then return a varied response.

    Distribution (configurable):
      50 % exact match
      20 % same category (vehicle→car, person→man, animal→dog)
      15 % known but wrong category (vehicle instead of person etc.)
      15 % completely wrong label
    """
    import re
    m = re.search(r"label=['\"]([^'\"]+)['\"]", obs_text)
    gold = m.group(1).lower() if m else "object"

    roll = rng.random()
    if roll < 0.50:
        return gold                      # exact match → 1.0

    # same category synonym
    if roll < 0.70:
        if gold in VEHICLE_LABELS:
            alts = [v for v in VEHICLE_LABELS if v != gold]
            return rng.choice(alts) if alts else "car"
        if gold in _PERSON_LABELS or gold == "person":
            return rng.choice([p for p in _PERSON_LABELS if p != gold] or ["man"])
        if gold in _ANIMAL_LABELS:
            return rng.choice([a for a in _ANIMAL_LABELS if a != gold] or ["cat"])
        return gold  # unknown group → exact

    # wrong category
    if roll < 0.85:
        if gold in VEHICLE_LABELS:
            return "person"
        if gold in _PERSON_LABELS or gold == "person":
            return "car"
        return "car"

    # completely off
    return rng.choice(_WRONG_LABELS)


def _mock_task2(obs_text: str, rng: random.Random) -> list[str]:
    """
    Parse all (label, confidence) pairs from obs_text, sort by confidence,
    then optionally shuffle some positions to produce partial credit.

    Distribution:
      35 % all correct
      25 % n-1 correct (swap last two)
      20 % n-2 correct (swap two middle positions)
      20 % completely reversed (0 correct)
    """
    import re

    pairs: list[tuple[str, float]] = []
    for m in re.finditer(
        r"label=['\"]([^'\"]+)['\"].*?confidence=([\d.]+)", obs_text, re.DOTALL
    ):
        pairs.append((m.group(1).lower(), float(m.group(2))))

    if not pairs:
        return ["person"]

    # gold order
    ordered = [p[0] for p in sorted(pairs, key=lambda x: x[1], reverse=True)]
    n = len(ordered)

    roll = rng.random()
    if roll < 0.35 or n == 1:
        return ordered                   # all correct → 1.0

    result = list(ordered)
    if roll < 0.60 and n >= 2:
        # swap last two → n-1 correct → 0.85
        result[-1], result[-2] = result[-2], result[-1]
        return result

    if roll < 0.80 and n >= 3:
        # swap two random middle positions → n-2 correct → 0.65
        i = rng.randint(0, n - 2)
        result[i], result[i + 1] = result[i + 1], result[i]
        return result

    # fully reverse → 0 correct → 0.0
    result.reverse()
    return result


def _mock_task3(obs_text: str, rng: random.Random) -> tuple[str, str]:
    """
    Apply the correct band rule, then vary the reasoning quality or decision.

    Distribution:
      30 % correct + strong reasoning (mentions confidence) → 1.0
      25 % correct + weak reasoning  → 0.85
      20 % correct + no reasoning   → 0.70
      15 % adjacent band + strong   → 0.50
      10 % two bands off            → 0.0
    """
    import re

    m = re.search(r"confidence=([\d.]+)", obs_text, re.IGNORECASE)
    conf = float(m.group(1)) if m else 0.3

    # Ground-truth decision
    if conf < 0.35:
        correct = "discard"
    elif conf < 0.50:
        correct = "request_rescan"
    else:
        correct = "log_and_continue"

    bands = ["discard", "request_rescan", "log_and_continue"]
    correct_idx = bands.index(correct)

    roll = rng.random()

    if roll < 0.30:
        # correct + strong (mentions number)
        return correct, f"confidence {conf:.2f} is in the {correct} band per policy."

    if roll < 0.55:
        # correct + weak
        return correct, "the detection seems uncertain, applying caution."

    if roll < 0.75:
        # correct + no reasoning
        return correct, ""

    if roll < 0.90:
        # adjacent band
        adj_idx = correct_idx + 1 if correct_idx < 2 else correct_idx - 1
        adjacent = bands[adj_idx]
        return adjacent, f"confidence {conf:.2f} suggests {adjacent}."

    # two bands off
    far_idx = (correct_idx + 2) % 3
    return bands[far_idx], ""


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(
    env: ArjunaEnv,
    seed: int,
    ep_idx: int,
    rng: random.Random,
    per_task: dict[int, list[float]],
) -> float:
    reset_out = env.reset(seed=seed)
    obs = reset_out.observation
    episode_id = obs.episode_id
    bundle_name = obs.bundle_name or "unknown"

    ep_meta: dict[str, Any] = {}
    if episode_id:
        ep_meta["episode_id"] = episode_id

    print(f"\n=== Episode {ep_idx} (seed={seed}, bundle={bundle_name!r}) ===")

    step_rewards: list[float] = []

    for sub in (1, 2, 3):
        task = obs.task_type
        scene_id = obs.scene_id
        obs_text = obs.observation_text

        # Build mock action
        if task == 1:
            label = _mock_task1(obs_text, rng)
            action = ArjunaAction(task1_label=label, metadata=ep_meta)
            action_desc = f"task1_label={label!r}"
        elif task == 2:
            ranked = _mock_task2(obs_text, rng)
            action = ArjunaAction(ranked_objects=ranked, metadata=ep_meta)
            action_desc = f"ranked_objects={ranked}"
        else:
            decision, reasoning = _mock_task3(obs_text, rng)
            action = ArjunaAction(decision=decision, reasoning=reasoning or None, metadata=ep_meta)
            action_desc = f"decision={decision!r}, reasoning={reasoning!r}"

        step_out = env.step(action)
        rw = step_out.reward
        if rw is None:
            rw = step_out.observation.reward
        rw = float(rw or 0.0)

        step_rewards.append(rw)
        per_task[task].append(rw)

        fb_first = (step_out.observation.feedback or "").splitlines()[0]
        print(f"  Step {sub}/3: task{task} | scene={scene_id} | reward={rw:.3f}")
        print(f"    action : {action_desc}")
        print(f"    feedback: {fb_first}")

        obs = step_out.observation
        if step_out.done:
            overall = (
                obs.overall_reward
                if obs.overall_reward is not None
                else float(mean(step_rewards))
            )
            print(f"  Episode reward: {overall:.3f}")
            return float(overall)

    overall = float(mean(step_rewards))
    print(f"  Episode reward: {overall:.3f}")
    return overall


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Offline ARJUNA inference test (no API key)")
    parser.add_argument("--seeds", type=int, default=5, help="Number of episodes to run")
    parser.add_argument("--url", default="http://127.0.0.1:7860", help="Server base URL")
    parser.add_argument("--seed-offset", type=int, default=0, help="Starting seed value")
    args = parser.parse_args()

    rng = random.Random()  # seeded from system time → different results each run
    seeds = list(range(args.seed_offset, args.seed_offset + args.seeds))

    per_task: dict[int, list[float]] = {1: [], 2: [], 3: []}
    all_episode_means: list[float] = []

    print(f"Running {args.seeds} episodes against {args.url}")
    print("Mock policy: intentional variation to show reward diversity\n")

    with ArjunaEnv(base_url=args.url).sync() as env:
        for ep_idx, seed in enumerate(seeds, start=1):
            overall = run_episode(env, seed, ep_idx, rng, per_task)
            all_episode_means.append(overall)

    print("\n" + "=" * 50)
    print("RESULTS SUMMARY")
    print("=" * 50)
    for t in (1, 2, 3):
        rewards = per_task[t]
        if rewards:
            task_names = {1: "identification", 2: "triage", 3: "low-confidence"}
            print(
                f"task {t} ({task_names[t]:>17}): "
                f"mean={mean(rewards):.3f}  "
                f"min={min(rewards):.3f}  "
                f"max={max(rewards):.3f}  "
                f"scores={[round(r, 2) for r in rewards]}"
            )
    if all_episode_means:
        print(f"\noverall mean reward : {mean(all_episode_means):.3f}")
        print(f"overall min / max   : {min(all_episode_means):.3f} / {max(all_episode_means):.3f}")
    print(f"seeds used          : {seeds}")


if __name__ == "__main__":
    main()
