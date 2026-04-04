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
    obs_text_override: str = ""  # set by scene_generator for LLM-generated scenes


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
    obs_text_override: str = ""  # set by scene_generator for LLM-generated scenes


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
    obs_text_override: str = ""  # set by scene_generator for LLM-generated scenes


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
    Task1Scene(
        scene_id="t1_009",
        description="Campus walkway: student with backpack centered, background defocused.",
        detection=SyntheticDetection("backpack", 0.90, (260.0, 200.0, 420.0, 420.0)),
        expected_label="backpack",
    ),
    Task1Scene(
        scene_id="t1_010",
        description="Rainy sidewalk: open umbrella dominating the upper half of the frame.",
        detection=SyntheticDetection("umbrella", 0.93, (180.0, 40.0, 520.0, 360.0)),
        expected_label="umbrella",
    ),
    Task1Scene(
        scene_id="t1_011",
        description="City park trail: a dog near a bench under morning light.",
        detection=SyntheticDetection("dog", 0.78, (220.0, 260.0, 420.0, 430.0)),
        expected_label="dog",
    ),
    Task1Scene(
        scene_id="t1_012",
        description="Forest edge: a cat sitting on a fallen log with clear silhouette.",
        detection=SyntheticDetection("cat", 0.79, (300.0, 220.0, 430.0, 340.0)),
        expected_label="cat",
    ),
    Task1Scene(
        scene_id="t1_013",
        description="Airport waiting area: unattended backpack by the charging station.",
        detection=SyntheticDetection("backpack", 0.91, (260.0, 240.0, 360.0, 380.0)),
        expected_label="backpack",
    ),
    Task1Scene(
        scene_id="t1_014",
        description="Beach boardwalk: bright umbrella opened beside a kiosk.",
        detection=SyntheticDetection("umbrella", 0.87, (120.0, 70.0, 420.0, 330.0)),
        expected_label="umbrella",
    ),
    Task1Scene(
        scene_id="t1_015",
        description="Train station concourse: rolling suitcase near platform sign.",
        detection=SyntheticDetection("suitcase", 0.84, (330.0, 230.0, 430.0, 410.0)),
        expected_label="suitcase",
    ),
    Task1Scene(
        scene_id="t1_016",
        description="Open-plan office: a single laptop on a desk under desk lamp.",
        detection=SyntheticDetection("laptop", 0.95, (250.0, 210.0, 470.0, 340.0)),
        expected_label="laptop",
    ),
    Task1Scene(
        scene_id="t1_017",
        description="Mall food court: plastic bottle on a table near the aisle.",
        detection=SyntheticDetection("bottle", 0.82, (360.0, 220.0, 410.0, 330.0)),
        expected_label="bottle",
    ),
    Task1Scene(
        scene_id="t1_018",
        description="Office lobby: a chair facing the reception desk.",
        detection=SyntheticDetection("chair", 0.76, (240.0, 180.0, 460.0, 420.0)),
        expected_label="chair",
    ),
    Task1Scene(
        scene_id="t1_019",
        description="Living room showroom: one couch centered under warm lighting.",
        detection=SyntheticDetection("couch", 0.89, (140.0, 190.0, 620.0, 430.0)),
        expected_label="couch",
    ),
    Task1Scene(
        scene_id="t1_020",
        description="Electronics store aisle: wall-mounted TV panel in full view.",
        detection=SyntheticDetection("tv", 0.97, (180.0, 70.0, 620.0, 320.0)),
        expected_label="tv",
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
        description="Construction site: hard hat worker, excavator, safety barrier",
        detections=(
            SyntheticDetection("person", 0.72, (100.0, 90.0, 220.0, 420.0)),
            SyntheticDetection("truck", 0.72, (380.0, 240.0, 560.0, 400.0)),
            SyntheticDetection("stop sign", 0.81, (250.0, 50.0, 340.0, 140.0)),
        ),
        expected_priority=("stop sign", "person", "truck"),
    ),
    Task2Scene(
        scene_id="t2_006",
        description="School zone: children crossing, school bus, crossing guard",
        detections=(
            SyntheticDetection("person", 0.65, (120.0, 220.0, 420.0, 360.0)),
            SyntheticDetection("bus", 0.89, (40.0, 150.0, 640.0, 360.0)),
            SyntheticDetection("person", 0.65, (560.0, 180.0, 620.0, 360.0)),
        ),
        expected_priority=("bus", "person", "person"),
    ),
    Task2Scene(
        scene_id="t2_007",
        description="Highway: motorcycle, car, pedestrian bridge overhead",
        detections=(
            SyntheticDetection("car", 0.78, (140.0, 220.0, 520.0, 360.0)),
            SyntheticDetection("motorcycle", 0.78, (260.0, 140.0, 340.0, 340.0)),
            SyntheticDetection("bridge", 0.62, (380.0, 260.0, 430.0, 330.0)),
        ),
        expected_priority=("car", "motorcycle", "bridge"),
    ),
    Task2Scene(
        scene_id="t2_008",
        description="Night intersection: person at crosswalk, bus approaching, traffic light.",
        detections=(
            SyntheticDetection("person", 0.79, (220.0, 120.0, 320.0, 420.0)),
            SyntheticDetection("bus", 0.76, (40.0, 150.0, 640.0, 360.0)),
            SyntheticDetection("traffic light", 0.80, (620.0, 40.0, 660.0, 120.0)),
            SyntheticDetection("bicycle", 0.65, (360.0, 260.0, 460.0, 380.0)),
        ),
        expected_priority=("traffic light", "person", "bus", "bicycle"),
    ),
    Task2Scene(
        scene_id="t2_009",
        description="Train platform: commuter, stroller, suitcase, train door partially open.",
        detections=(
            SyntheticDetection("person", 0.88, (150.0, 120.0, 230.0, 420.0)),
            SyntheticDetection("train", 0.84, (260.0, 80.0, 740.0, 360.0)),
            SyntheticDetection("suitcase", 0.71, (320.0, 260.0, 380.0, 360.0)),
            SyntheticDetection("stroller", 0.69, (210.0, 260.0, 280.0, 360.0)),
        ),
        expected_priority=("person", "train", "suitcase", "stroller"),
    ),
    Task2Scene(
        scene_id="t2_010",
        description="Shopping mall atrium: multiple people, escalator, decorative car display.",
        detections=(
            SyntheticDetection("person", 0.92, (300.0, 140.0, 380.0, 420.0)),
            SyntheticDetection("person", 0.86, (420.0, 160.0, 500.0, 430.0)),
            SyntheticDetection("car", 0.80, (120.0, 220.0, 420.0, 380.0)),
            SyntheticDetection("bench", 0.74, (480.0, 320.0, 620.0, 380.0)),
            SyntheticDetection("potted plant", 0.60, (80.0, 260.0, 140.0, 360.0)),
        ),
        expected_priority=("person", "person", "car", "bench", "potted plant"),
    ),
    Task2Scene(
        scene_id="t2_011",
        description="Rainy shoulder: stopped truck, traffic cone, distant person in high-vis.",
        detections=(
            SyntheticDetection("truck", 0.87, (40.0, 200.0, 520.0, 380.0)),
            SyntheticDetection("traffic cone", 0.72, (260.0, 260.0, 300.0, 340.0)),
            SyntheticDetection("person", 0.69, (560.0, 180.0, 620.0, 360.0)),
            SyntheticDetection("umbrella", 0.66, (540.0, 150.0, 620.0, 260.0)),
        ),
        expected_priority=("truck", "traffic cone", "person", "umbrella"),
    ),
    Task2Scene(
        scene_id="t2_012",
        description="Warehouse lane: forklift crossing with one worker nearby.",
        detections=(
            SyntheticDetection("truck", 0.93, (80.0, 170.0, 460.0, 380.0)),
            SyntheticDetection("person", 0.75, (500.0, 150.0, 560.0, 360.0)),
            SyntheticDetection("pallet", 0.59, (290.0, 300.0, 360.0, 390.0)),
        ),
        expected_priority=("truck", "person", "pallet"),
    ),
    Task2Scene(
        scene_id="t2_013",
        description="City curb: person stepping out, car approaching, umbrella overhead.",
        detections=(
            SyntheticDetection("person", 0.81, (300.0, 120.0, 380.0, 410.0)),
            SyntheticDetection("car", 0.81, (120.0, 220.0, 520.0, 380.0)),
            SyntheticDetection("umbrella", 0.67, (280.0, 40.0, 390.0, 180.0)),
        ),
        expected_priority=("person", "car", "umbrella"),
    ),
    Task2Scene(
        scene_id="t2_014",
        description="Airport apron: airplane parked, bus service vehicle, two ground crew.",
        detections=(
            SyntheticDetection("airplane", 0.92, (40.0, 60.0, 760.0, 300.0)),
            SyntheticDetection("bus", 0.84, (170.0, 260.0, 360.0, 360.0)),
            SyntheticDetection("person", 0.84, (390.0, 250.0, 430.0, 360.0)),
            SyntheticDetection("person", 0.72, (460.0, 250.0, 500.0, 360.0)),
            SyntheticDetection("luggage cart", 0.69, (520.0, 280.0, 620.0, 360.0)),
        ),
        expected_priority=("airplane", "person", "bus", "person", "luggage cart"),
    ),
    Task2Scene(
        scene_id="t2_015",
        description="School drop-off: bus, bicycle, backpack, child crossing sign.",
        detections=(
            SyntheticDetection("bus", 0.86, (60.0, 120.0, 680.0, 380.0)),
            SyntheticDetection("bicycle", 0.73, (420.0, 240.0, 530.0, 360.0)),
            SyntheticDetection("backpack", 0.73, (510.0, 230.0, 560.0, 330.0)),
            SyntheticDetection("stop sign", 0.75, (700.0, 120.0, 740.0, 180.0)),
        ),
        expected_priority=("bus", "stop sign", "bicycle", "backpack"),
    ),
    Task2Scene(
        scene_id="t2_016",
        description="Downtown junction: traffic light, person, truck, motorcycle, dog.",
        detections=(
            SyntheticDetection("traffic light", 0.82, (620.0, 50.0, 660.0, 120.0)),
            SyntheticDetection("person", 0.82, (250.0, 130.0, 320.0, 410.0)),
            SyntheticDetection("truck", 0.82, (80.0, 210.0, 300.0, 360.0)),
            SyntheticDetection("motorcycle", 0.82, (350.0, 250.0, 470.0, 360.0)),
            SyntheticDetection("dog", 0.61, (500.0, 300.0, 580.0, 380.0)),
        ),
        expected_priority=("person", "truck", "motorcycle", "traffic light", "dog"),
    ),
    Task2Scene(
        scene_id="t2_017",
        description="Port entry gate: boat trailer, car, security guard, cone.",
        detections=(
            SyntheticDetection("boat", 0.77, (70.0, 170.0, 400.0, 320.0)),
            SyntheticDetection("car", 0.77, (430.0, 200.0, 660.0, 340.0)),
            SyntheticDetection("person", 0.77, (355.0, 130.0, 410.0, 350.0)),
            SyntheticDetection("traffic cone", 0.65, (300.0, 330.0, 330.0, 390.0)),
        ),
        expected_priority=("person", "boat", "car", "traffic cone"),
    ),
    Task2Scene(
        scene_id="t2_018",
        description="Mall atrium: person near escalator, chair cluster, potted plant.",
        detections=(
            SyntheticDetection("person", 0.74, (320.0, 130.0, 390.0, 410.0)),
            SyntheticDetection("chair", 0.74, (190.0, 250.0, 290.0, 390.0)),
            SyntheticDetection("potted plant", 0.74, (500.0, 220.0, 560.0, 360.0)),
            SyntheticDetection("bench", 0.69, (120.0, 300.0, 260.0, 380.0)),
        ),
        expected_priority=("person", "chair", "potted plant", "bench"),
    ),
    Task2Scene(
        scene_id="t2_019",
        description="Rail crossing: train approaching, warning sign, cyclist waiting.",
        detections=(
            SyntheticDetection("train", 0.91, (50.0, 120.0, 760.0, 330.0)),
            SyntheticDetection("stop sign", 0.79, (620.0, 110.0, 670.0, 180.0)),
            SyntheticDetection("bicycle", 0.79, (390.0, 230.0, 500.0, 360.0)),
            SyntheticDetection("person", 0.79, (430.0, 180.0, 480.0, 350.0)),
        ),
        expected_priority=("train", "person", "bicycle", "stop sign"),
    ),
    Task2Scene(
        scene_id="t2_020",
        description="Beach access road: bus, person, dog, surfboard rack, bicycle.",
        detections=(
            SyntheticDetection("bus", 0.83, (80.0, 130.0, 650.0, 330.0)),
            SyntheticDetection("person", 0.83, (360.0, 150.0, 420.0, 370.0)),
            SyntheticDetection("dog", 0.83, (430.0, 300.0, 510.0, 380.0)),
            SyntheticDetection("surfboard", 0.70, (540.0, 120.0, 600.0, 330.0)),
            SyntheticDetection("bicycle", 0.70, (250.0, 250.0, 340.0, 360.0)),
        ),
        expected_priority=("person", "bus", "dog", "bicycle", "surfboard"),
    ),
    Task2Scene(
        scene_id="t2_021",
        description="Office parking deck: motorcycle, car, laptop bag, worker.",
        detections=(
            SyntheticDetection("motorcycle", 0.76, (170.0, 230.0, 340.0, 360.0)),
            SyntheticDetection("car", 0.76, (360.0, 200.0, 670.0, 360.0)),
            SyntheticDetection("backpack", 0.76, (520.0, 240.0, 560.0, 320.0)),
            SyntheticDetection("person", 0.76, (470.0, 150.0, 530.0, 360.0)),
        ),
        expected_priority=("person", "motorcycle", "car", "backpack"),
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
        description="Night vision: faint silhouette near a curb in poor visibility.",
        primary_detection=SyntheticDetection("person", 0.28, (400.0, 200.0, 520.0, 440.0)),
        expected_action="discard",
    ),
    Task3Scene(
        scene_id="t3_009",
        description="Rain + headlight glare: low-confidence candidate detected by the sensor.",
        primary_detection=SyntheticDetection("car", 0.42, (620.0, 280.0, 710.0, 360.0)),
        expected_action="request_rescan",
    ),
    Task3Scene(
        scene_id="t3_010",
        description="Tunnel entrance: bright backlight, sensor returns moderately-to-strong signal.",
        primary_detection=SyntheticDetection("truck", 0.65, (80.0, 200.0, 520.0, 360.0)),
        expected_action="log_and_continue",
    ),
    Task3Scene(
        scene_id="t3_011",
        description="Dust + haze: extremely low confidence detection on the roadway.",
        primary_detection=SyntheticDetection("motorcycle", 0.18, (320.0, 260.0, 420.0, 340.0)),
        expected_action="discard",
    ),
    Task3Scene(
        scene_id="t3_012",
        description="Bright glare on wet asphalt: confidence is moderate-high but still uncertain.",
        primary_detection=SyntheticDetection("bicycle", 0.58, (220.0, 180.0, 280.0, 360.0)),
        expected_action="log_and_continue",
    ),
    Task3Scene(
        scene_id="t3_013",
        description="Snow at night: parked truck partially occluded by drifting snow.",
        primary_detection=SyntheticDetection("truck", 0.47, (80.0, 200.0, 520.0, 360.0)),
        expected_action="request_rescan",
    ),
    Task3Scene(
        scene_id="t3_014",
        description="Tunnel exit: strong backlight, single sedan emerging into brighter area.",
        primary_detection=SyntheticDetection("car", 0.53, (260.0, 220.0, 520.0, 360.0)),
        expected_action="log_and_continue",
        notes="Confidence above 0.50 despite tricky lighting.",
    ),
    Task3Scene(
        scene_id="t3_015",
        description="Night rain on highway: faint object at shoulder in low visibility.",
        primary_detection=SyntheticDetection("person", 0.21, (420.0, 260.0, 470.0, 360.0)),
        expected_action="discard",
    ),
    Task3Scene(
        scene_id="t3_016",
        description="Dense morning fog: possible bicycle silhouette near lane divider.",
        primary_detection=SyntheticDetection("bicycle", 0.15, (330.0, 260.0, 410.0, 340.0)),
        expected_action="discard",
    ),
    Task3Scene(
        scene_id="t3_017",
        description="Dusty construction route: ambiguous box close to rubble pile.",
        primary_detection=SyntheticDetection("truck", 0.29, (120.0, 220.0, 430.0, 360.0)),
        expected_action="discard",
    ),
    Task3Scene(
        scene_id="t3_018",
        description="Glare from oncoming headlights: weak pedestrian hypothesis.",
        primary_detection=SyntheticDetection("person", 0.32, (260.0, 160.0, 320.0, 360.0)),
        expected_action="discard",
    ),
    Task3Scene(
        scene_id="t3_019",
        description="Tunnel interior: moderate confidence vehicle detection in dim lighting.",
        primary_detection=SyntheticDetection("car", 0.38, (220.0, 220.0, 520.0, 360.0)),
        expected_action="request_rescan",
    ),
    Task3Scene(
        scene_id="t3_020",
        description="Wet road spray: uncertain motorcycle shape amid reflections.",
        primary_detection=SyntheticDetection("motorcycle", 0.44, (300.0, 240.0, 450.0, 350.0)),
        expected_action="request_rescan",
    ),
    Task3Scene(
        scene_id="t3_021",
        description="Evening fog near station exit: moderate confidence person outline.",
        primary_detection=SyntheticDetection("person", 0.48, (390.0, 180.0, 450.0, 380.0)),
        expected_action="request_rescan",
    ),
    Task3Scene(
        scene_id="t3_022",
        description="Sunlit boulevard: strong car detection despite small glare patch.",
        primary_detection=SyntheticDetection("car", 0.52, (180.0, 210.0, 560.0, 360.0)),
        expected_action="log_and_continue",
    ),
    Task3Scene(
        scene_id="t3_023",
        description="Airport service lane: bus contour clear under floodlights.",
        primary_detection=SyntheticDetection("bus", 0.61, (90.0, 170.0, 690.0, 360.0)),
        expected_action="log_and_continue",
    ),
    Task3Scene(
        scene_id="t3_024",
        description="Clear afternoon near tunnel mouth: stable truck detection in frame center.",
        primary_detection=SyntheticDetection("truck", 0.75, (110.0, 180.0, 610.0, 360.0)),
        expected_action="log_and_continue",
    ),
)

for _scene in TASK3_SCENES:
    if not (0.0 <= _scene.primary_detection.confidence <= 1.0):
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
# Episode bundles — one coherent 3-step episode per theme (Task1 → Task2 → Task3)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EpisodeBundle:
    """Three scenes (one per task) sharing a location/theme for multi-step RL episodes."""

    bundle_id: str
    name: str
    task1: Task1Scene
    task2: Task2Scene
    task3: Task3Scene


EPISODE_BUNDLES: tuple[EpisodeBundle, ...] = (
    EpisodeBundle(
        bundle_id="bnd_urban",
        name="Urban Street",
        task1=Task1Scene(
            scene_id="t1_bnd_urban",
            description=(
                "Urban street: one pedestrian standing on the sidewalk near a storefront, "
                "clear daylight, minimal occlusion."
            ),
            detection=SyntheticDetection("person", 0.91, (140.0, 90.0, 480.0, 440.0)),
            expected_label="person",
        ),
        task2=Task2Scene(
            scene_id="t2_bnd_urban",
            description=(
                "Same urban corridor: crosswalk with a pedestrian, a sedan in the lane, "
                "and a cyclist in the bike lane."
            ),
            detections=(
                SyntheticDetection("person", 0.92, (210.0, 110.0, 360.0, 450.0)),
                SyntheticDetection("car", 0.80, (400.0, 220.0, 680.0, 380.0)),
                SyntheticDetection("bicycle", 0.74, (120.0, 200.0, 340.0, 400.0)),
            ),
            expected_priority=("person", "car", "bicycle"),
        ),
        task3=Task3Scene(
            scene_id="t3_bnd_urban",
            description=(
                "Urban doorway: a faint ambiguous shape visible in a dark recessed doorway "
                "next to the sidewalk; could be debris or a person."
            ),
            primary_detection=SyntheticDetection("person", 0.31, (280.0, 160.0, 320.0, 360.0)),
            expected_action="discard",
            notes="Very low confidence — discard.",
        ),
    ),
    EpisodeBundle(
        bundle_id="bnd_warehouse",
        name="Warehouse",
        task1=Task1Scene(
            scene_id="t1_bnd_warehouse",
            description=(
                "Warehouse floor: one forklift truck centered in the aisle, pallets stacked "
                "along the racks."
            ),
            detection=SyntheticDetection("truck", 0.88, (90.0, 120.0, 620.0, 400.0)),
            expected_label="truck",
        ),
        task2=Task2Scene(
            scene_id="t2_bnd_warehouse",
            description=(
                "Loading bay: worker near a box truck, pallet load on the floor, "
                "and a safety cone marking the lane."
            ),
            detections=(
                SyntheticDetection("person", 0.88, (300.0, 130.0, 380.0, 410.0)),
                SyntheticDetection("truck", 0.87, (80.0, 150.0, 520.0, 390.0)),
                SyntheticDetection("suitcase", 0.75, (420.0, 300.0, 500.0, 360.0)),
                SyntheticDetection("traffic cone", 0.62, (240.0, 360.0, 270.0, 430.0)),
            ),
            expected_priority=("person", "truck", "suitcase", "traffic cone"),
        ),
        task3=Task3Scene(
            scene_id="t3_bnd_warehouse",
            description=(
                "Dusty warehouse corridor: hazy air and uneven lighting; an uncertain blob "
                "at the far end could be a person or equipment."
            ),
            primary_detection=SyntheticDetection("person", 0.42, (310.0, 140.0, 350.0, 340.0)),
            expected_action="request_rescan",
            notes="Moderate uncertainty — request rescan.",
        ),
    ),
    EpisodeBundle(
        bundle_id="bnd_parking",
        name="Parking Lot",
        task1=Task1Scene(
            scene_id="t1_bnd_parking",
            description=(
                "Open parking lot: one sedan facing the camera, lines and wheel stops visible."
            ),
            detection=SyntheticDetection("car", 0.92, (180.0, 200.0, 600.0, 380.0)),
            expected_label="car",
        ),
        task2=Task2Scene(
            scene_id="t2_bnd_parking",
            description=(
                "Same lot: car backing out, cyclist crossing behind, pedestrian walking, "
                "and a traffic cone near the curb."
            ),
            detections=(
                SyntheticDetection("car", 0.92, (150.0, 210.0, 560.0, 370.0)),
                SyntheticDetection("person", 0.89, (420.0, 180.0, 480.0, 380.0)),
                SyntheticDetection("bicycle", 0.85, (300.0, 230.0, 400.0, 360.0)),
                SyntheticDetection("traffic cone", 0.70, (520.0, 320.0, 550.0, 380.0)),
            ),
            expected_priority=("car", "person", "bicycle", "traffic cone"),
        ),
        task3=Task3Scene(
            scene_id="t3_bnd_parking",
            description=(
                "Clear afternoon in the lot: a vehicle contour is stable and well-framed despite "
                "minor glare on the hood; detection is reliable."
            ),
            primary_detection=SyntheticDetection("car", 0.65, (200.0, 220.0, 540.0, 360.0)),
            expected_action="log_and_continue",
            notes="0.65 — well above 0.50, clearly log_and_continue.",
        ),
    ),
    EpisodeBundle(
        bundle_id="bnd_school",
        name="School Zone",
        task1=Task1Scene(
            scene_id="t1_bnd_school",
            description=(
                "School drop-off lane: a yellow school bus stopped at the curb, signage visible."
            ),
            detection=SyntheticDetection("bus", 0.87, (60.0, 110.0, 660.0, 380.0)),
            expected_label="bus",
        ),
        task2=Task2Scene(
            scene_id="t2_bnd_school",
            description=(
                "School front: bus unloading, child with bicycle, crossing guard, and a dog "
                "on leash near the crosswalk."
            ),
            detections=(
                SyntheticDetection("bus", 0.90, (40.0, 120.0, 640.0, 360.0)),
                SyntheticDetection("person", 0.88, (360.0, 160.0, 420.0, 380.0)),
                SyntheticDetection("bicycle", 0.82, (280.0, 240.0, 380.0, 360.0)),
                SyntheticDetection("dog", 0.75, (480.0, 300.0, 540.0, 370.0)),
            ),
            expected_priority=("bus", "person", "bicycle", "dog"),
        ),
        task3=Task3Scene(
            scene_id="t3_bnd_school",
            description=(
                "Frosted glass entry: a blurred silhouette behind the door — person or "
                "reflection from the hall."
            ),
            primary_detection=SyntheticDetection("person", 0.38, (330.0, 150.0, 390.0, 380.0)),
            expected_action="request_rescan",
            notes="0.35–0.50 band.",
        ),
    ),
    EpisodeBundle(
        bundle_id="bnd_airport",
        name="Airport",
        task1=Task1Scene(
            scene_id="t1_bnd_airport",
            description=(
                "Airport ramp: one commercial jet parked, tail and wing visible under daylight."
            ),
            detection=SyntheticDetection("airplane", 0.95, (120.0, 80.0, 680.0, 300.0)),
            expected_label="airplane",
        ),
        task2=Task2Scene(
            scene_id="t2_bnd_airport",
            description=(
                "Service road near the terminal: airplane tail in background, fuel truck, "
                "ground crew person, and a traffic cone by the lane."
            ),
            detections=(
                SyntheticDetection("airplane", 0.94, (40.0, 60.0, 720.0, 280.0)),
                SyntheticDetection("person", 0.91, (380.0, 200.0, 440.0, 380.0)),
                SyntheticDetection("truck", 0.88, (180.0, 210.0, 520.0, 360.0)),
                SyntheticDetection("traffic cone", 0.70, (560.0, 320.0, 590.0, 380.0)),
            ),
            expected_priority=("airplane", "person", "truck", "traffic cone"),
        ),
        task3=Task3Scene(
            scene_id="t3_bnd_airport",
            description=(
                "Dense pre-dawn fog on the taxiway: sensor returns an extremely weak signal "
                "near the runway edge; shape completely indistinct, identity unknown."
            ),
            primary_detection=SyntheticDetection("airplane", 0.19, (400.0, 120.0, 520.0, 260.0)),
            expected_action="discard",
            notes="0.19 — well below 0.35, clearly discard.",
        ),
    ),
    EpisodeBundle(
        bundle_id="bnd_hospital",
        name="Hospital Entrance",
        task1=Task1Scene(
            scene_id="t1_bnd_hospital",
            description=(
                "Hospital ambulance bay: one person in scrubs walking toward the entrance."
            ),
            detection=SyntheticDetection("person", 0.89, (260.0, 140.0, 380.0, 420.0)),
            expected_label="person",
        ),
        task2=Task2Scene(
            scene_id="t2_bnd_hospital",
            description=(
                "Drop-off loop: ambulance-style truck with flashers, sedan, staff member, "
                "and a traffic cone channeling traffic."
            ),
            detections=(
                SyntheticDetection("truck", 0.92, (50.0, 160.0, 480.0, 380.0)),
                SyntheticDetection("person", 0.90, (520.0, 170.0, 580.0, 380.0)),
                SyntheticDetection("car", 0.87, (300.0, 220.0, 620.0, 360.0)),
                SyntheticDetection("traffic cone", 0.68, (200.0, 340.0, 230.0, 400.0)),
            ),
            expected_priority=("truck", "person", "car", "traffic cone"),
        ),
        task3=Task3Scene(
            scene_id="t3_bnd_hospital",
            description=(
                "Bright sun glare on the glass canopy: a washed-out figure near the doors; "
                "confidence is marginal."
            ),
            primary_detection=SyntheticDetection("person", 0.51, (290.0, 160.0, 350.0, 380.0)),
            expected_action="log_and_continue",
            notes="≥ 0.50.",
        ),
    ),
    EpisodeBundle(
        bundle_id="bnd_construction",
        name="Construction Site",
        task1=Task1Scene(
            scene_id="t1_bnd_construction",
            description=(
                "Active construction: one dump truck in the foreground, dirt and barriers."
            ),
            detection=SyntheticDetection("truck", 0.86, (100.0, 120.0, 640.0, 400.0)),
            expected_label="truck",
        ),
        task2=Task2Scene(
            scene_id="t2_bnd_construction",
            description=(
                "Same site: hard-hat worker, articulated truck, stop sign at the lane merge, "
                "and cones marking the trench."
            ),
            detections=(
                SyntheticDetection("person", 0.88, (280.0, 130.0, 360.0, 400.0)),
                SyntheticDetection("truck", 0.86, (40.0, 170.0, 420.0, 380.0)),
                SyntheticDetection("stop sign", 0.75, (600.0, 80.0, 650.0, 150.0)),
                SyntheticDetection("traffic cone", 0.72, (380.0, 350.0, 410.0, 420.0)),
            ),
            expected_priority=("person", "truck", "stop sign", "traffic cone"),
        ),
        task3=Task3Scene(
            scene_id="t3_bnd_construction",
            description=(
                "Dust cloud from a loader: a vague shape behind the plume — worker or stacked "
                "material."
            ),
            primary_detection=SyntheticDetection("person", 0.44, (320.0, 180.0, 370.0, 360.0)),
            expected_action="request_rescan",
            notes="0.35–0.50 band.",
        ),
    ),
    EpisodeBundle(
        bundle_id="bnd_night",
        name="Night Street",
        task1=Task1Scene(
            scene_id="t1_bnd_night",
            description=(
                "Night street: one bicycle with reflectors crossing a pool of streetlight."
            ),
            detection=SyntheticDetection("bicycle", 0.78, (180.0, 220.0, 420.0, 400.0)),
            expected_label="bicycle",
        ),
        task2=Task2Scene(
            scene_id="t2_bnd_night",
            description=(
                "Same block: cyclist in the intersection, pedestrian on the curb, "
                "and an approaching car with headlights."
            ),
            detections=(
                SyntheticDetection("bicycle", 0.85, (200.0, 210.0, 380.0, 390.0)),
                SyntheticDetection("person", 0.82, (440.0, 190.0, 500.0, 380.0)),
                SyntheticDetection("car", 0.78, (80.0, 200.0, 320.0, 360.0)),
            ),
            expected_priority=("bicycle", "person", "car"),
        ),
        task3=Task3Scene(
            scene_id="t3_bnd_night",
            description=(
                "Deep shadow under an awning: sensor returns a very faint flicker — "
                "likely noise or a stationary object, confidence is negligible."
            ),
            primary_detection=SyntheticDetection("bicycle", 0.21, (350.0, 240.0, 410.0, 360.0)),
            expected_action="discard",
            notes="0.21 — well below 0.35, clearly discard.",
        ),
    ),
)

for _bundle in EPISODE_BUNDLES:
    _c2 = _expected_triage_order(_bundle.task2.detections)
    if _c2 != _bundle.task2.expected_priority:
        raise RuntimeError(
            f"Episode bundle {_bundle.bundle_id} Task2 expected_priority mismatch: "
            f"{_c2} != {_bundle.task2.expected_priority}"
        )
    _d = _bundle.task3.primary_detection.confidence
    if not (0.0 <= _d <= 1.0):
        raise RuntimeError(
            f"Episode bundle {_bundle.bundle_id} Task3 confidence out of range: {_d}"
        )
    _exp3 = expected_low_confidence_action(_d)
    if _exp3 != _bundle.task3.expected_action:
        raise RuntimeError(
            f"Episode bundle {_bundle.bundle_id} Task3 expected_action mismatch: "
            f"{_exp3} != {_bundle.task3.expected_action}"
        )


def episode_bundle_by_id(bundle_id: str) -> EpisodeBundle:
    for b in EPISODE_BUNDLES:
        if b.bundle_id == bundle_id:
            return b
    raise KeyError(f"Unknown episode bundle_id: {bundle_id}")


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
