"""
Synthetic camera scenes for ARJUNA perception tasks.

All values simulate YOLO-style detections: class label, confidence score,
and optional bounding box for narrative richness (not used by graders unless
you extend the environment).

Task 1: single high-confidence object identification.
Task 2: multi-object triage with confidence + class priority tie-breaks.
Task 3: low-confidence policy decisions keyed to confidence bands.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

# ---------------------------------------------------------------------------
# Shared types
# ---------------------------------------------------------------------------

LowConfidenceAction = Literal["log_and_continue", "discard", "request_rescan"]

# COCO-style labels used across tasks; "vehicle" tier matches hackathon spec.
VEHICLE_LABELS: Final[frozenset[str]] = frozenset(
    {
        "bicycle",
        "car",
        "motorcycle",
        "airplane",
        "bus",
        "train",
        "truck",
        "boat",
    }
)


def class_priority_rank(label: str) -> int:
    """
    Tie-break for Task 2 when confidences are equal or need ordering:
    person > vehicle > others.
    """
    if label == "person":
        return 2
    if label in VEHICLE_LABELS:
        return 1
    return 0


@dataclass(frozen=True)
class SyntheticDetection:
    """One simulated YOLO detection."""

    label: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)


@dataclass(frozen=True)
class Task1Scene:
    """Single-object identification: one clear detection, confidence > 0.85."""

    scene_id: str
    description: str
    detection: SyntheticDetection
    expected_label: str


@dataclass(frozen=True)
class Task2Scene:
    """
    Multi-object triage: 3–5 detections.
    `expected_priority` is most important first (ground truth for grader).
    """

    scene_id: str
    description: str
    detections: tuple[SyntheticDetection, ...]
    expected_priority: tuple[str, ...]


@dataclass(frozen=True)
class Task3Scene:
    """
    Low-confidence decision: primary detection in [0.25, 0.55].
    `expected_action` follows band rules from hackathon spec.
    """

    scene_id: str
    description: str
    primary_detection: SyntheticDetection
    expected_action: LowConfidenceAction
    notes: str = ""


# ---------------------------------------------------------------------------
# Task 1 — Single object identification (confidence > 0.85)
# ---------------------------------------------------------------------------

TASK1_SCENES: tuple[Task1Scene, ...] = (
    Task1Scene(
        scene_id="t1_001",
        description=(
            "Indoor corridor: single full-body human figure centered in frame, "
            "lighting even, minimal occlusion."
        ),
        detection=SyntheticDetection("person", 0.92, (120.0, 80.0, 520.0, 420.0)),
        expected_label="person",
    ),
    Task1Scene(
        scene_id="t1_002",
        description=(
            "Parking lot: one sedan facing the camera, license plate region visible."
        ),
        detection=SyntheticDetection("car", 0.88, (200.0, 210.0, 600.0, 380.0)),
        expected_label="car",
    ),
    Task1Scene(
        scene_id="t1_003",
        description="Warehouse aisle: one pallet jack with operator at distance.",
        detection=SyntheticDetection("person", 0.91, (300.0, 50.0, 400.0, 350.0)),
        expected_label="person",
    ),
    Task1Scene(
        scene_id="t1_004",
        description="Street crossing: one bicycle side profile, clear silhouette.",
        detection=SyntheticDetection("bicycle", 0.87, (50.0, 200.0, 350.0, 450.0)),
        expected_label="bicycle",
    ),
    Task1Scene(
        scene_id="t1_005",
        description="Sidewalk: medium-sized dog on leash, unobstructed.",
        detection=SyntheticDetection("dog", 0.89, (400.0, 300.0, 580.0, 480.0)),
        expected_label="dog",
    ),
    Task1Scene(
        scene_id="t1_006",
        description="Highway overpass: single delivery truck, dominant in frame.",
        detection=SyntheticDetection("truck", 0.86, (100.0, 120.0, 700.0, 400.0)),
        expected_label="truck",
    ),
    Task1Scene(
        scene_id="t1_007",
        description="Airport perimeter: one commercial jet parked, tail visible.",
        detection=SyntheticDetection("airplane", 0.90, (150.0, 100.0, 650.0, 280.0)),
        expected_label="airplane",
    ),
    Task1Scene(
        scene_id="t1_008",
        description="Bus stop: single public transit bus, front fascia visible.",
        detection=SyntheticDetection("bus", 0.88, (80.0, 60.0, 720.0, 460.0)),
        expected_label="bus",
    ),
)


# ---------------------------------------------------------------------------
# Task 2 — Multi-object triage (3–5 objects, varying confidence)
# ---------------------------------------------------------------------------
#
# Ground-truth order: sort by confidence descending, then person > vehicle > other.


def _expected_triage_order(detections: tuple[SyntheticDetection, ...]) -> tuple[str, ...]:
    ranked = sorted(
        detections,
        key=lambda d: (d.confidence, class_priority_rank(d.label)),
        reverse=True,
    )
    return tuple(d.label for d in ranked)


TASK2_SCENES: tuple[Task2Scene, ...] = (
    Task2Scene(
        scene_id="t2_001",
        description=(
            "Crosswalk: pedestrian close, car behind, traffic light small in frame."
        ),
        detections=(
            SyntheticDetection("person", 0.91, (200.0, 100.0, 380.0, 460.0)),
            SyntheticDetection("car", 0.78, (400.0, 220.0, 680.0, 380.0)),
            SyntheticDetection("traffic light", 0.62, (600.0, 50.0, 640.0, 120.0)),
        ),
        expected_priority=("person", "car", "traffic light"),
    ),
    Task2Scene(
        scene_id="t2_002",
        description="Loading dock: worker, forklift (mapped as truck in COCO-like set), cone.",
        detections=(
            SyntheticDetection("truck", 0.85, (100.0, 150.0, 500.0, 400.0)),
            SyntheticDetection("person", 0.85, (420.0, 180.0, 500.0, 380.0)),
            SyntheticDetection("traffic cone", 0.55, (300.0, 380.0, 330.0, 450.0)),
        ),
        expected_priority=("person", "truck", "traffic cone"),
    ),
    Task2Scene(
        scene_id="t2_003",
        description="Bike lane: cyclist, parked van, dog on leash.",
        detections=(
            SyntheticDetection("bicycle", 0.82, (150.0, 200.0, 400.0, 420.0)),
            SyntheticDetection("car", 0.79, (450.0, 230.0, 700.0, 360.0)),
            SyntheticDetection("dog", 0.70, (520.0, 300.0, 600.0, 380.0)),
            SyntheticDetection("person", 0.68, (530.0, 250.0, 590.0, 360.0)),
        ),
        expected_priority=("bicycle", "car", "dog", "person"),
    ),
    Task2Scene(
        scene_id="t2_004",
        description="Construction: excavator, worker, safety barrier, distant sign.",
        detections=(
            SyntheticDetection("person", 0.88, (300.0, 120.0, 380.0, 400.0)),
            SyntheticDetection("truck", 0.72, (50.0, 180.0, 280.0, 380.0)),
            SyntheticDetection("stop sign", 0.58, (620.0, 80.0, 660.0, 130.0)),
            SyntheticDetection("traffic cone", 0.52, (400.0, 360.0, 430.0, 420.0)),
        ),
        expected_priority=("person", "truck", "stop sign", "traffic cone"),
    ),
    Task2Scene(
        scene_id="t2_005",
        description="Crowded market: partial person, motorcycle, umbrella, fruit stand.",
        detections=(
            SyntheticDetection("person", 0.77, (100.0, 90.0, 220.0, 420.0)),
            SyntheticDetection("motorcycle", 0.77, (380.0, 240.0, 560.0, 400.0)),
            SyntheticDetection("umbrella", 0.64, (250.0, 50.0, 340.0, 140.0)),
            SyntheticDetection("apple", 0.50, (480.0, 380.0, 520.0, 420.0)),
            SyntheticDetection("handbag", 0.45, (200.0, 300.0, 240.0, 360.0)),
        ),
        expected_priority=("person", "motorcycle", "umbrella", "apple", "handbag"),
    ),
    Task2Scene(
        scene_id="t2_006",
        description="Airfield: plane, service truck, fuel hose cart, marshaller.",
        detections=(
            SyntheticDetection("airplane", 0.90, (50.0, 40.0, 750.0, 320.0)),
            SyntheticDetection("person", 0.74, (320.0, 280.0, 360.0, 380.0)),
            SyntheticDetection("truck", 0.68, (200.0, 260.0, 340.0, 340.0)),
        ),
        expected_priority=("airplane", "person", "truck"),
    ),
)

# Sanity: every Task2 scene must match computed triage order (catches data bugs).
for _scene in TASK2_SCENES:
    _computed = _expected_triage_order(_scene.detections)
    if _computed != _scene.expected_priority:
        raise RuntimeError(
            f"Task2 scene {_scene.scene_id} expected_priority mismatch: "
            f"{_computed} != {_scene.expected_priority}"
        )


# ---------------------------------------------------------------------------
# Task 3 — Low-confidence decision (0.25–0.55)
# ---------------------------------------------------------------------------
#
# Bands (from spec):
#   confidence < 0.35           → discard
#   0.35 ≤ confidence < 0.50    → request_rescan
#   confidence ≥ 0.50           → log_and_continue


def expected_low_confidence_action(confidence: float) -> LowConfidenceAction:
    """Ground-truth action for a single low-confidence primary detection."""
    if confidence < 0.35:
        return "discard"
    if confidence < 0.50:
        return "request_rescan"
    return "log_and_continue"


TASK3_SCENES: tuple[Task3Scene, ...] = (
    Task3Scene(
        scene_id="t3_001",
        description="Dusk scene: ambiguous blob at road edge, possible debris or animal.",
        primary_detection=SyntheticDetection("cat", 0.28, (410.0, 300.0, 470.0, 360.0)),
        expected_action="discard",
        notes="Below 0.35 — too uncertain to trust.",
    ),
    Task3Scene(
        scene_id="t3_002",
        description="Fog: faint vertical shape, could be person or post.",
        primary_detection=SyntheticDetection("person", 0.32, (300.0, 100.0, 340.0, 380.0)),
        expected_action="discard",
        notes="0.32 < 0.35",
    ),
    Task3Scene(
        scene_id="t3_003",
        description="Glare on windshield: weak car hypothesis.",
        primary_detection=SyntheticDetection("car", 0.34, (120.0, 200.0, 580.0, 360.0)),
        expected_action="discard",
        notes="0.34 < 0.35",
    ),
    Task3Scene(
        scene_id="t3_004",
        description="Partial occlusion: person-like contour behind fence.",
        primary_detection=SyntheticDetection("person", 0.38, (220.0, 140.0, 280.0, 360.0)),
        expected_action="request_rescan",
        notes="0.35–0.50 band",
    ),
    Task3Scene(
        scene_id="t3_005",
        description="Motion blur: bicycle shape, unstable box.",
        primary_detection=SyntheticDetection("bicycle", 0.42, (180.0, 220.0, 420.0, 400.0)),
        expected_action="request_rescan",
    ),
    Task3Scene(
        scene_id="t3_006",
        description="Rain: truck at long range, noisy depth alignment.",
        primary_detection=SyntheticDetection("truck", 0.48, (50.0, 160.0, 420.0, 340.0)),
        expected_action="request_rescan",
    ),
    Task3Scene(
        scene_id="t3_007",
        description="Night IR: warm blob, likely vehicle but not confirmed.",
        primary_detection=SyntheticDetection("car", 0.52, (200.0, 240.0, 560.0, 380.0)),
        expected_action="log_and_continue",
        notes="≥ 0.50 but still in low-confidence evaluation range.",
    ),
    Task3Scene(
        scene_id="t3_008",
        description="Crowd edge: shoulder and bag visible, person class uncertain.",
        primary_detection=SyntheticDetection("person", 0.54, (400.0, 200.0, 520.0, 440.0)),
        expected_action="log_and_continue",
    ),
    Task3Scene(
        scene_id="t3_009",
        description="Drone shot: small bus in corner, low pixels on target.",
        primary_detection=SyntheticDetection("bus", 0.55, (620.0, 280.0, 710.0, 360.0)),
        expected_action="log_and_continue",
        notes="Upper bound of stated 0.25–0.55 evaluation range.",
    ),
)

for _scene in TASK3_SCENES:
    if not (0.25 <= _scene.primary_detection.confidence <= 0.55):
        raise RuntimeError(
            f"Task3 scene {_scene.scene_id} confidence out of band: "
            f"{_scene.primary_detection.confidence}"
        )
    _exp = expected_low_confidence_action(_scene.primary_detection.confidence)
    if _exp != _scene.expected_action:
        raise RuntimeError(
            f"Task3 scene {_scene.scene_id} expected_action mismatch: "
            f"{_exp} != {_scene.expected_action}"
        )


# ---------------------------------------------------------------------------
# Lookups and helpers for the environment
# ---------------------------------------------------------------------------

def task1_scene_by_id(scene_id: str) -> Task1Scene:
    for s in TASK1_SCENES:
        if s.scene_id == scene_id:
            return s
    raise KeyError(f"Unknown Task1 scene_id: {scene_id}")


def task2_scene_by_id(scene_id: str) -> Task2Scene:
    for s in TASK2_SCENES:
        if s.scene_id == scene_id:
            return s
    raise KeyError(f"Unknown Task2 scene_id: {scene_id}")


def task3_scene_by_id(scene_id: str) -> Task3Scene:
    for s in TASK3_SCENES:
        if s.scene_id == scene_id:
            return s
    raise KeyError(f"Unknown Task3 scene_id: {scene_id}")


def all_scene_ids() -> dict[str, tuple[str, ...]]:
    """Convenience for tests and manifests."""
    return {
        "task1": tuple(s.scene_id for s in TASK1_SCENES),
        "task2": tuple(s.scene_id for s in TASK2_SCENES),
        "task3": tuple(s.scene_id for s in TASK3_SCENES),
    }
