"""
Task definitions and grader functions for ARJUNA perception (scores in [0.0, 1.0]).
"""

from __future__ import annotations

import json
import re
from typing import Iterable

from .synthetic_data import (
    LowConfidenceAction,
    Task1Scene,
    Task2Scene,
    Task3Scene,
    expected_low_confidence_action,
)


def _norm_label(s: str) -> str:
    return s.strip().lower()


def _norm_list(labels: Iterable[str]) -> tuple[str, ...]:
    return tuple(_norm_label(x) for x in labels)


def grade_task1_identification(predicted: str | None, scene: Task1Scene) -> float:
    """1.0 if predicted class matches expected label (case-insensitive)."""
    if predicted is None:
        return 0.0
    return 1.0 if _norm_label(predicted) == _norm_label(scene.expected_label) else 0.0


def grade_task2_triage(predicted_rank: list[str] | None, scene: Task2Scene) -> float:
    """
    Fraction of positions where the agent's ranking matches ground truth.
    Ground truth is scene.expected_priority (most important first).
    """
    expected = _norm_list(scene.expected_priority)
    if not predicted_rank:
        return 0.0
    pred = _norm_list(predicted_rank)
    if len(pred) != len(expected):
        return 0.0
    correct = sum(1 for i in range(len(expected)) if pred[i] == expected[i])
    return correct / len(expected)


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


def _reasoning_suggests_engagement(reasoning: str | None) -> bool:
    """Heuristic for partial credit when the chosen action is wrong."""
    if not reasoning:
        return False
    t = reasoning.strip().lower()
    if len(t) < 12:
        return False
    hints = (
        "confidence",
        "uncertain",
        "uncertainty",
        "ambiguous",
        "low conf",
        "noisy",
        "blur",
        "occlusion",
        "rescan",
        "discard",
        "risk",
        "sensor",
        "yolo",
    )
    return any(h in t for h in hints)


def grade_task3_low_confidence(
    decision: str | None,
    reasoning: str | None,
    scene: Task3Scene,
) -> float:
    """
    1.0 if decision matches band rule; 0.5 if wrong decision but reasoning shows
    sound uncertainty analysis; 0.0 otherwise.
    """
    conf = scene.primary_detection.confidence
    correct = expected_low_confidence_action(conf)
    got = _normalize_decision(decision)
    if got == correct:
        return 1.0
    if _reasoning_suggests_engagement(reasoning):
        return 0.5
    return 0.0


def format_task1_prompt(scene: Task1Scene) -> str:
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
