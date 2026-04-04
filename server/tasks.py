"""
Task definitions and grader functions for ARJUNA perception (scores in [0.0, 1.0]).
"""

from __future__ import annotations

import json
import re
from typing import Any, Iterable

from .synthetic_data import (
    LowConfidenceAction,
    Task1Scene,
    Task2Scene,
    Task3Scene,
    expected_low_confidence_action,
)

VEHICLE_LABELS = {
    "car",
    "truck",
    "bus",
    "motorcycle",
    "bicycle",
    "airplane",
    "train",
    "boat",
}
PERSON_LABELS = {"person", "man", "woman", "child", "pedestrian"}
ANIMAL_LABELS = {
    "dog",
    "cat",
    "bird",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
}


def _norm_label(s: str) -> str:
    return s.strip().lower()


def _norm_list(labels: Iterable[str]) -> tuple[str, ...]:
    return tuple(_norm_label(x) for x in labels)


def _task1_group(label: str) -> str | None:
    if label in VEHICLE_LABELS:
        return "vehicle"
    if label in PERSON_LABELS:
        return "person"
    if label in ANIMAL_LABELS:
        return "animal"
    return None


def grade_task1_identification(
    predicted: str | None,
    scene: Task1Scene,
    metadata: dict[str, Any] | None = None,
) -> float:
    """
    Task 1 with semantic partial credit.

    - exact match -> 1.0 (metadata may repeat the YOLO confidence; score stays capped at 1.0)
    - same category group -> 0.7
    - predicted is a known grouped class but wrong group -> 0.2
    - unrelated/unknown -> 0.0
    """
    _ = metadata
    if predicted is None:
        return 0.0
    pred = _norm_label(predicted)
    expected = _norm_label(scene.expected_label)
    if pred == expected:
        return 1.0

    pred_group = _task1_group(pred)
    expected_group = _task1_group(expected)
    if pred_group is not None and pred_group == expected_group:
        return 0.7
    if pred_group is not None and pred_group != expected_group:
        return 0.2
    return 0.0


def grade_task2_triage(predicted_rank: list[str] | None, scene: Task2Scene) -> float:
    """
    Granular reward by how many positions match ground truth (including tie-break order).

    - all positions correct -> 1.0
    - n-1 positions correct -> 0.85
    - n-2 positions correct -> 0.65
    - exactly 1 position correct -> 0.33
    - none correct -> 0.0

    Length must match ground truth or the score is 0.0.
    """
    expected = _norm_list(scene.expected_priority)
    if not predicted_rank:
        return 0.0
    pred = _norm_list(predicted_rank)
    if len(pred) != len(expected):
        return 0.0
    n = len(expected)
    correct = sum(1 for i in range(n) if pred[i] == expected[i])
    if correct == n:
        return 1.0
    if n >= 2 and correct == n - 1:
        return 0.85
    if n >= 3 and correct == n - 2:
        return 0.65
    if correct == 1:
        return 0.33
    return 0.0


def _normalize_decision(raw: str | None) -> LowConfidenceAction | None:
    if raw is None:
        return None
    s = raw.strip().lower().replace("-", "_")
    for a in ("log_and_continue", "discard", "request_rescan"):
        if s == a:
            return a  # type: ignore[return-value]
    # allow minor phrasing
    if "log" in s and "continue" in s:
        return "log_and_continue"
    if "discard" in s or "drop" in s:
        return "discard"
    if "rescan" in s or "re-scan" in s or ("request" in s and "scan" in s):
        return "request_rescan"
    return None


def _reasoning_quality(reasoning: str | None) -> str:
    """Classify reasoning quality as strong/weak/none."""
    if reasoning is None:
        return "none"
    text = reasoning.strip()
    if not text:
        return "none"
    # Strong reasoning explicitly references a confidence value.
    if re.search(r"\b(?:0(?:\.\d+)?|1(?:\.0+)?)\b", text):
        return "strong"
    return "weak"


def _decision_band_index(decision: LowConfidenceAction | None) -> int | None:
    if decision == "discard":
        return 0
    if decision == "request_rescan":
        return 1
    if decision == "log_and_continue":
        return 2
    return None


def grade_task3_low_confidence(
    decision: str | None,
    reasoning: str | None,
    scene: Task3Scene,
) -> float:
    """
    Richer low-confidence grading:
    - Correct + strong reasoning (mentions confidence number) -> 1.0
    - Correct + weak reasoning -> 0.85
    - Correct + no reasoning -> 0.7
    - Adjacent-band + strong reasoning -> 0.5
    - Adjacent-band + weak/no reasoning -> 0.3
    - Two bands off / invalid -> 0.0
    """
    conf = scene.primary_detection.confidence
    correct = expected_low_confidence_action(conf)
    got = _normalize_decision(decision)
    quality = _reasoning_quality(reasoning)

    if got is None:
        return 0.0

    if got == correct:
        if quality == "strong":
            return 1.0
        if quality == "weak":
            return 0.85
        return 0.7

    correct_idx = _decision_band_index(correct)
    got_idx = _decision_band_index(got)
    if correct_idx is None or got_idx is None:
        return 0.0

    if abs(correct_idx - got_idx) == 1:
        return 0.5 if quality == "strong" else 0.3
    return 0.0


def format_step_observation(bundle_name: str, step: int, inner_prompt: str) -> str:
    """Wrap a task prompt with step index and bundle theme for multi-step episodes."""
    titles = {
        1: "Step 1/3 — Single Object ID",
        2: "Step 2/3 — Multi Object Triage",
        3: "Step 3/3 — Low Confidence Decision",
    }
    title = titles.get(step, f"Step {step}/3")
    return f"{title}\n\nBundle: {bundle_name}\n\n{inner_prompt}"


def format_task1_prompt(scene: Task1Scene) -> str:
    if scene.obs_text_override:
        return scene.obs_text_override
    d = scene.detection
    return (
        "TASK: single_object_identification\n\n"
        f"SCENE: {scene.description}\n\n"
        "YOLO detections (simulated):\n"
        f"- label={d.label!r}, confidence={d.confidence:.3f}, "
        f"bbox_xyxy={list(map(int, d.bbox_xyxy))}\n\n"
        "Respond with the object class label only (e.g. person, car, bicycle)."
    )


def format_task2_prompt(scene: Task2Scene) -> str:
    if scene.obs_text_override:
        return scene.obs_text_override
    lines = [
        "TASK: multi_object_triage",
        "",
        f"SCENE: {scene.description}",
        "",
        "YOLO detections (simulated), unsorted:",
    ]
    for det in scene.detections:
        lines.append(
            f"- label={det.label!r}, confidence={det.confidence:.3f}, "
            f"bbox_xyxy={list(map(int, det.bbox_xyxy))}"
        )
    lines.extend(
        [
            "",
            "Return a priority list from most important to least important.",
            "Priority rules: (1) higher confidence first; (2) ties — person > vehicle "
            "(bicycle, car, bus, truck, motorcycle, airplane, train, boat) > other classes.",
            "Answer with a comma-separated list of labels in order, or JSON array in your tool output.",
        ]
    )
    return "\n".join(lines)


def format_task3_prompt(scene: Task3Scene) -> str:
    if scene.obs_text_override:
        return scene.obs_text_override
    d = scene.primary_detection
    opts = "log_and_continue | discard | request_rescan"
    return (
        "TASK: low_confidence_decision\n\n"
        f"SCENE: {scene.description}\n\n"
        "PRIMARY YOLO detection (low confidence band):\n"
        f"- label={d.label!r}, confidence={d.confidence:.3f}, "
        f"bbox_xyxy={list(map(int, d.bbox_xyxy))}\n\n"
        f"Choose exactly one action: {opts}.\n"
        "Optionally add a short reasoning string about uncertainty and safety."
    )


def extract_json_list(text: str) -> list[str] | None:
    """Best-effort parse of a JSON array of strings from model output."""
    m = re.search(r"\[[^\]]+\]", text, re.DOTALL)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
        if isinstance(data, list) and all(isinstance(x, str) for x in data):
            return list(data)
    except json.JSONDecodeError:
        return None
    return None
