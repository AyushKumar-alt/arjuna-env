"""
scene_generator.py — Dynamic LLM-powered scene generation for ARJUNA.

Generates fresh perception scenes on demand using an LLM.
Falls back to synthetic_data.py hardcoded scenes if the LLM is
unavailable, quota-exhausted, or returns invalid output.

Difficulty levels:
  easy   — high confidence, unambiguous, few objects
  medium — moderate confidence, some ambiguity, 3-4 objects
  hard   — boundary confidence values, many objects, edge cases
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
from typing import Optional

try:
    from openai import OpenAI
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False

from .synthetic_data import (
    EpisodeBundle,
    SyntheticDetection,
    Task1Scene,
    Task2Scene,
    Task3Scene,
    expected_low_confidence_action,
)

logger = logging.getLogger(__name__)

# ── LLM Client (optional) ─────────────────────────────────

def _get_client() -> Optional['OpenAI']:
    """Return an OpenAI-compatible client, or None if creds are missing."""
    if not _OPENAI_AVAILABLE:
        return None
    base_url = os.environ.get("API_BASE_URL")
    token    = os.environ.get("HF_TOKEN")
    if not base_url or not token:
        return None
    try:
        return OpenAI(base_url=base_url, api_key=token)
    except Exception:
        return None

# ── Prompt templates per task × difficulty ────────────────

TASK1_PROMPTS: dict[str, str] = {
    "easy": """Generate a Task 1 robot perception scene.
Rules:
- ONE object detected, confidence between 0.85-0.98
- Object must be from thematic categories: person, car, truck (ambulance/forklift), bus,
  bicycle, motorcycle, airplane, dog, bear, backpack, umbrella,
  fire hydrant, stop sign, bench, chair, potted plant, laptop
- Clear unambiguous scene description (daytime, good lighting)

Return ONLY valid JSON, no other text:
{
  "scene_id": "gen_t1_easy_<random 4 digit number>",
  "description": "<2-3 word location description>",
  "label": "<single COCO class name>",
  "confidence": <float 0.85-0.98>,
  "bbox_xyxy": [<x1>, <y1>, <x2>, <y2>],
  "observation_text": "TASK: single_object_identification\\n\\nSCENE: <scene description>\\n\\nYOLO detections (simulated):\\n- label='<label>', confidence=<confidence>, bbox_xyxy=[<bbox>]\\n\\nRespond with the object class label only (e.g. person, car, bicycle)."
}""",

    "medium": """Generate a Task 1 robot perception scene.
Rules:
- ONE primary object detected, confidence between 0.72-0.84
- Some scene ambiguity (partial occlusion, side angle, shadows)
- Object from COCO classes

Return ONLY valid JSON:
{
  "scene_id": "gen_t1_med_<random 4 digit number>",
  "description": "<location>",
  "label": "<COCO class>",
  "confidence": <float 0.72-0.84>,
  "bbox_xyxy": [<x1>, <y1>, <x2>, <y2>],
  "observation_text": "TASK: single_object_identification\\n\\nSCENE: <scene with some ambiguity>\\n\\nYOLO detections (simulated):\\n- label='<label>', confidence=<confidence>, bbox_xyxy=[<bbox>]\\n\\nRespond with the object class label only."
}""",

    "hard": """Generate a Task 1 robot perception scene.
Rules:
- ONE object detected, confidence between 0.60-0.71
- Challenging conditions: night, rain, fog, motion blur,
  partial view, unusual angle, occlusion
- Object from COCO classes

Return ONLY valid JSON:
{
  "scene_id": "gen_t1_hard_<random 4 digit number>",
  "description": "<challenging location>",
  "label": "<COCO class>",
  "confidence": <float 0.60-0.71>,
  "bbox_xyxy": [<x1>, <y1>, <x2>, <y2>],
  "observation_text": "TASK: single_object_identification\\n\\nSCENE: <challenging scene description>\\n\\nYOLO detections (simulated):\\n- label='<label>', confidence=<confidence>, bbox_xyxy=[<bbox>]\\n\\nRespond with the object class label only."
}""",
}

TASK2_PROMPTS: dict[str, str] = {
    "easy": """Generate a Task 2 multi-object triage scene.
Rules:
- 3 objects, clearly different confidence scores (no ties)
- Confidences spread: one high (0.85-0.95), one mid (0.65-0.80),
  one low (0.45-0.60)
- Include at least one person
- Correct priority ordering obvious from confidence alone

Return ONLY valid JSON:
{
  "scene_id": "gen_t2_easy_<random 4 digit number>",
  "description": "<location>",
  "detections": [
    {"label": "<class>", "confidence": <float>, "bbox_xyxy": [x1,y1,x2,y2]},
    {"label": "<class>", "confidence": <float>, "bbox_xyxy": [x1,y1,x2,y2]},
    {"label": "<class>", "confidence": <float>, "bbox_xyxy": [x1,y1,x2,y2]}
  ],
  "expected_priority": ["<highest conf label>", "<mid conf label>", "<lowest conf label>"],
  "observation_text": "TASK: multi_object_triage\\n\\nSCENE: <description>\\n\\nYOLO detections (simulated), unsorted:\\n<one line per detection: - label='X', confidence=Y, bbox_xyxy=[...]>\\n\\nReturn a priority list from most important to least important.\\nPriority rules: (1) higher confidence first; (2) ties: person > vehicle > other classes.\\nAnswer with a JSON array of labels in order."
}""",

    "medium": """Generate a Task 2 multi-object triage scene.
Rules:
- 4 objects
- EXACTLY ONE tie: two objects with same confidence (0.80-0.88)
- Tie must be broken by class priority (person > vehicle > other)
- Other two objects have clearly different confidences

Return ONLY valid JSON:
{
  "scene_id": "gen_t2_med_<random 4 digit number>",
  "description": "<location>",
  "detections": [
    {"label": "<class>", "confidence": <float>, "bbox_xyxy": [x1,y1,x2,y2]},
    {"label": "<class>", "confidence": <float>, "bbox_xyxy": [x1,y1,x2,y2]},
    {"label": "<class>", "confidence": <float>, "bbox_xyxy": [x1,y1,x2,y2]},
    {"label": "<class>", "confidence": <float>, "bbox_xyxy": [x1,y1,x2,y2]}
  ],
  "expected_priority": ["<label>", "<label>", "<label>", "<label>"],
  "observation_text": "TASK: multi_object_triage\\n\\nSCENE: <description>\\n\\nYOLO detections (simulated), unsorted:\\n<one line per detection>\\n\\nReturn a priority list from most important to least important.\\nPriority rules: (1) higher confidence first; (2) ties: person > vehicle > other classes.\\nAnswer with a JSON array of labels in order."
}""",

    "hard": """Generate a Task 2 multi-object triage scene.
Rules:
- 5 objects
- MULTIPLE ties: at least 2 pairs with same confidence
- Mix of persons, vehicles, and other objects
- Ties require knowing full priority hierarchy to resolve correctly
- Confidences mostly clustered in 0.75-0.88 range

Return ONLY valid JSON:
{
  "scene_id": "gen_t2_hard_<random 4 digit number>",
  "description": "<busy location like intersection/station>",
  "detections": [
    {"label": "<class>", "confidence": <float>, "bbox_xyxy": [x1,y1,x2,y2]},
    {"label": "<class>", "confidence": <float>, "bbox_xyxy": [x1,y1,x2,y2]},
    {"label": "<class>", "confidence": <float>, "bbox_xyxy": [x1,y1,x2,y2]},
    {"label": "<class>", "confidence": <float>, "bbox_xyxy": [x1,y1,x2,y2]},
    {"label": "<class>", "confidence": <float>, "bbox_xyxy": [x1,y1,x2,y2]}
  ],
  "expected_priority": ["<label>","<label>","<label>","<label>","<label>"],
  "observation_text": "TASK: multi_object_triage\\n\\nSCENE: <description>\\n\\nYOLO detections (simulated), unsorted:\\n<one line per detection>\\n\\nReturn a priority list from most important to least important.\\nPriority rules: (1) higher confidence first; (2) ties: person > vehicle > other classes.\\nAnswer with a JSON array of labels in order."
}""",
}

TASK3_PROMPTS: dict[str, str] = {
    "easy": """Generate a Task 3 low-confidence decision scene.
Rules:
- Confidence CLEARLY in one band, NOT near boundary:
  - discard band:        0.15-0.29
  - request_rescan band: 0.38-0.46
  - log_and_continue:    0.55-0.72
- Pick one band randomly, generate confidence within it
- Scene context should match the confidence level

Band rules (use when setting correct_action):
  confidence < 0.35          → correct_action = "discard"
  0.35 <= confidence < 0.50  → correct_action = "request_rescan"
  confidence >= 0.50         → correct_action = "log_and_continue"

Return ONLY valid JSON:
{
  "scene_id": "gen_t3_easy_<random 4 digit number>",
  "description": "<scene context>",
  "label": "<detected object class>",
  "confidence": <float — clearly in one band>,
  "correct_action": "<discard|request_rescan|log_and_continue>",
  "bbox_xyxy": [x1,y1,x2,y2],
  "observation_text": "TASK: low_confidence_decision\\n\\nSCENE: <scene>\\n\\nPRIMARY YOLO detection (low confidence band):\\n- label='<label>', confidence=<confidence>, bbox_xyxy=[<bbox>]\\n\\nChoose exactly one action: log_and_continue | discard | request_rescan.\\nOptionally add a short reasoning string about uncertainty and safety."
}""",

    "medium": """Generate a Task 3 low-confidence decision scene.
Rules:
- Confidence in MODERATE range: 0.31-0.34 or 0.50-0.54
  (near-boundary but still technically in one band)
- Scene has some contextual ambiguity
- correct_action follows band rules:
  confidence < 0.35          → discard
  0.35 <= confidence < 0.50  → request_rescan
  confidence >= 0.50         → log_and_continue

Return ONLY valid JSON with the same structure:
{
  "scene_id": "gen_t3_med_<random 4 digit number>",
  "description": "<scene context with ambiguity>",
  "label": "<COCO class>",
  "confidence": <float 0.31-0.34 OR 0.50-0.54>,
  "correct_action": "<discard|request_rescan|log_and_continue>",
  "bbox_xyxy": [x1,y1,x2,y2],
  "observation_text": "TASK: low_confidence_decision\\n\\nSCENE: <scene>\\n\\nPRIMARY YOLO detection (low confidence band):\\n- label='<label>', confidence=<confidence>, bbox_xyxy=[<bbox>]\\n\\nChoose exactly one action: log_and_continue | discard | request_rescan.\\nOptionally add a short reasoning string about uncertainty and safety."
}""",

    "hard": """Generate a Task 3 low-confidence decision scene.
Rules:
- Confidence sits very close to a boundary (within 0.005):
  - near 0.350 (boundary between discard and request_rescan)
  - near 0.500 (boundary between request_rescan and log_and_continue)
- Challenging visual conditions: heavy fog, night, motion blur,
  severe occlusion, sensor noise
- correct_action MUST still follow strict band rules

Return ONLY valid JSON with the same structure:
{
  "scene_id": "gen_t3_hard_<random 4 digit number>",
  "description": "<very challenging scene>",
  "label": "<COCO class>",
  "confidence": <float near 0.350 or 0.500>,
  "correct_action": "<discard|request_rescan|log_and_continue>",
  "bbox_xyxy": [x1,y1,x2,y2],
  "observation_text": "TASK: low_confidence_decision\\n\\nSCENE: <scene>\\n\\nPRIMARY YOLO detection (low confidence band):\\n- label='<label>', confidence=<confidence>, bbox_xyxy=[<bbox>]\\n\\nChoose exactly one action: log_and_continue | discard | request_rescan.\\nOptionally add a short reasoning string about uncertainty and safety."
}""",
}

# ── JSON parsing ──────────────────────────────────────────

def _parse_json_response(text: str) -> Optional[dict]:
    """Extract JSON from LLM response, handling markdown fences."""
    text = re.sub(r"^```[a-z]*\n?", "", text.strip())
    text = text.rstrip("`").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None

# ── Validators ────────────────────────────────────────────

def _validate_task1_scene(scene: dict) -> bool:
    required = ["scene_id", "label", "confidence", "observation_text"]
    return (
        all(k in scene for k in required)
        and isinstance(scene["label"], str)
        and bool(scene["label"].strip())
        and 0.0 < float(scene["confidence"]) < 1.0
        and len(scene["observation_text"]) > 50
    )

def _validate_task2_scene(scene: dict) -> bool:
    required = ["scene_id", "detections", "expected_priority", "observation_text"]
    return (
        all(k in scene for k in required)
        and isinstance(scene["detections"], list)
        and len(scene["detections"]) >= 2
        and isinstance(scene["expected_priority"], list)
        and len(scene["expected_priority"]) >= 2
    )

def _validate_task3_scene(scene: dict) -> bool:
    required = ["scene_id", "label", "confidence", "correct_action", "observation_text"]
    valid_actions = {"discard", "request_rescan", "log_and_continue"}
    if not all(k in scene for k in required):
        return False
    if scene["correct_action"] not in valid_actions:
        return False
    try:
        c = float(scene["confidence"])
    except (ValueError, TypeError):
        return False
    if not (0.0 < c < 1.0):
        return False
    # Ensure correct_action matches band rules
    expected = expected_low_confidence_action(c)
    return expected == scene["correct_action"]

# ── Converter: dict → dataclass bundle ───────────────────

def _make_bbox(raw: object) -> tuple[float, float, float, float]:
    """Safely parse a bbox from list or tuple; returns a 4-tuple."""
    try:
        lst = list(raw)  # type: ignore[arg-type]
        if len(lst) == 4:
            return (float(lst[0]), float(lst[1]), float(lst[2]), float(lst[3]))
    except Exception:
        pass
    return (0.0, 0.0, 100.0, 100.0)

def _convert_to_bundle(
    t1: dict,
    t2: dict,
    t3: dict,
    bundle_id: str,
    theme: str,
    difficulty: str,
) -> EpisodeBundle:
    """
    Convert three LLM-generated scene dicts into a real ``EpisodeBundle``
    dataclass so the existing environment code needs no changes.
    """
    # Task 1
    task1 = Task1Scene(
        scene_id=t1["scene_id"],
        description=t1.get("description", "Generated scene"),
        detection=SyntheticDetection(
            label=t1["label"],
            confidence=float(t1["confidence"]),
            bbox_xyxy=_make_bbox(t1.get("bbox_xyxy", [0, 0, 100, 100])),
        ),
        expected_label=t1["label"],
        obs_text_override=t1.get("observation_text", ""),
    )

    # Task 2 – build SyntheticDetection list
    det_list = []
    for d in t2["detections"]:
        det_list.append(
            SyntheticDetection(
                label=str(d["label"]),
                confidence=float(d["confidence"]),
                bbox_xyxy=_make_bbox(d.get("bbox_xyxy", [0, 0, 100, 100])),
            )
        )
    task2 = Task2Scene(
        scene_id=t2["scene_id"],
        description=t2.get("description", "Generated scene"),
        detections=tuple(det_list),
        expected_priority=tuple(str(x) for x in t2["expected_priority"]),
        obs_text_override=t2.get("observation_text", ""),
    )

    # Task 3
    task3 = Task3Scene(
        scene_id=t3["scene_id"],
        description=t3.get("description", "Generated scene"),
        primary_detection=SyntheticDetection(
            label=t3["label"],
            confidence=float(t3["confidence"]),
            bbox_xyxy=_make_bbox(t3.get("bbox_xyxy", [0, 0, 100, 100])),
        ),
        expected_action=t3["correct_action"],  # type: ignore[arg-type]
        notes=f"Generated ({difficulty})",
        obs_text_override=t3.get("observation_text", ""),
    )

    return EpisodeBundle(
        bundle_id=bundle_id,
        name=f"{theme.title()} [{difficulty}]",
        task1=task1,
        task2=task2,
        task3=task3,
    )

# ── Scene generation ──────────────────────────────────────

def generate_scene(
    task_type: int,
    difficulty: str = "medium",
    seed: Optional[int] = None,
) -> Optional[dict]:
    """
    Generate a single scene dict using the LLM.
    Returns ``None`` if generation fails (caller should use fallback).
    """
    client = _get_client()
    if client is None:
        logger.debug("No LLM client available for scene generation")
        return None

    prompts_map = {1: TASK1_PROMPTS, 2: TASK2_PROMPTS, 3: TASK3_PROMPTS}
    if task_type not in prompts_map:
        return None
    prompt = prompts_map[task_type].get(difficulty)
    if not prompt:
        return None

    seed_hint = seed if seed is not None else random.randint(1000, 9999)

    try:
        response = client.chat.completions.create(
            model=os.environ.get("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct"),
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a robotics simulation data generator. "
                        "Generate realistic perception scenes for an autonomous robot "
                        "training environment. Always return valid JSON only. "
                        f"Use seed hint {seed_hint} for variety."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.9,
            max_tokens=800,
        )
        raw   = response.choices[0].message.content or ""
        scene = _parse_json_response(raw)

        if scene is None:
            logger.warning(
                "Failed to parse LLM scene for task%d/%s", task_type, difficulty
            )
            return None

        validators = {
            1: _validate_task1_scene,
            2: _validate_task2_scene,
            3: _validate_task3_scene,
        }
        if not validators[task_type](scene):
            logger.warning(
                "Generated scene failed validation: task%d/%s", task_type, difficulty
            )
            return None

        scene["_generated"] = True
        scene["_difficulty"] = difficulty
        scene["_seed"] = seed_hint
        logger.info(
            "Generated scene: %s  task%d/%s", scene.get("scene_id"), task_type, difficulty
        )
        return scene

    except Exception as exc:
        logger.warning("Scene generation failed: %s", exc)
        return None


def generate_episode_bundle(
    difficulty: str = "medium",
    seed: Optional[int] = None,
) -> Optional[EpisodeBundle]:
    """
    Generate a complete 3-task episode bundle using the LLM.
    Returns a real ``EpisodeBundle`` dataclass, or ``None`` on any failure.
    On failure the caller should fall back to hardcoded bundles.
    """
    location_themes = [
        "urban street intersection",
        "warehouse loading dock",
        "hospital entrance",
        "school zone",
        "airport runway area",
        "construction site",
        "shopping mall parking",
        "train station platform",
        "highway rest stop",
        "industrial factory floor",
        "suburban neighborhood",
        "university campus",
        "sports stadium entrance",
        "ferry terminal dock",
        "forest research station",
    ]

    rng   = random.Random(seed)
    theme = rng.choice(location_themes)

    t1 = generate_scene(1, difficulty, seed)
    t2 = generate_scene(2, difficulty, seed + 1 if seed is not None else None)
    t3 = generate_scene(3, difficulty, seed + 2 if seed is not None else None)

    if not all([t1, t2, t3]):
        logger.warning(
            "generate_episode_bundle: one or more tasks failed — using fallback"
        )
        return None

    bundle_id = f"gen_bundle_{difficulty}_{seed if seed is not None else rng.randint(1000, 9999)}"

    try:
        return _convert_to_bundle(
            t1=t1,  # type: ignore[arg-type]
            t2=t2,  # type: ignore[arg-type]
            t3=t3,  # type: ignore[arg-type]
            bundle_id=bundle_id,
            theme=theme,
            difficulty=difficulty,
        )
    except Exception as exc:
        logger.warning("Bundle conversion failed: %s", exc)
        return None
