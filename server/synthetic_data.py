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
        description="Office lobby: modern couch centered in the lounge, clear lighting.",
        detection=SyntheticDetection("couch", 0.92, (120.0, 190.0, 620.0, 430.0)),
        expected_label="couch",
    ),
    Task1Scene(
        scene_id="t1_002",
        description="Parking lot: one sedan (car) parked in the designated stall.",
        detection=SyntheticDetection("car", 0.88, (200.0, 210.0, 600.0, 380.0)),
        expected_label="car",
    ),
    Task1Scene(
        scene_id="t1_003",
        description="Warehouse aisle: a forklift (truck) centered under high-bay lights.",
        detection=SyntheticDetection("truck", 0.91, (90.0, 120.0, 620.0, 400.0)),
        expected_label="truck",
    ),
    Task1Scene(
        scene_id="t1_004",
        description="Night street: a rider on a motorcycle side profile, clear headlamp.",
        detection=SyntheticDetection("motorcycle", 0.87, (200.0, 220.0, 450.0, 410.0)),
        expected_label="motorcycle",
    ),
    Task1Scene(
        scene_id="t1_005",
        description="Forest trail: a medium-sized dog running head, clear silhouette.",
        detection=SyntheticDetection("dog", 0.89, (300.0, 250.0, 450.0, 400.0)),
        expected_label="dog",
    ),
    Task1Scene(
        scene_id="t1_006",
        description="Construction road: a large dump truck in the foreground, tracks in dirt.",
        detection=SyntheticDetection("truck", 0.86, (100.0, 120.0, 640.0, 400.0)),
        expected_label="truck",
    ),
    Task1Scene(
        scene_id="t1_007",
        description="Airport gate: one commercial jet (airplane) parked, engine visible.",
        detection=SyntheticDetection("airplane", 0.90, (120.0, 80.0, 680.0, 300.0)),
        expected_label="airplane",
    ),
    Task1Scene(
        scene_id="t1_008",
        description="School loading lane: a yellow school bus stopped at the curb.",
        detection=SyntheticDetection("bus", 0.88, (50.0, 80.0, 650.0, 420.0)),
        expected_label="bus",
    ),
    Task1Scene(
        scene_id="t1_009",
        description="School sidewalk: a student wearing a backpack centered in frame.",
        detection=SyntheticDetection("backpack", 0.90, (260.0, 200.0, 420.0, 420.0)),
        expected_label="backpack",
    ),
    Task1Scene(
        scene_id="t1_010",
        description="Rainy street: open umbrella held by a pedestrian at the crosswalk.",
        detection=SyntheticDetection("umbrella", 0.93, (200.0, 100.0, 450.0, 420.0)),
        expected_label="umbrella",
    ),
    Task1Scene(
        scene_id="t1_011",
        description="Hospital entrance: one staff member (person) walking near the sliding doors.",
        detection=SyntheticDetection("person", 0.89, (260.0, 140.0, 380.0, 420.0)),
        expected_label="person",
    ),
    Task1Scene(
        scene_id="t1_012",
        description="Forest clearing: a hiker carrying a backpack looking toward the trail.",
        detection=SyntheticDetection("backpack", 0.88, (300.0, 250.0, 450.0, 400.0)),
        expected_label="backpack",
    ),
    Task1Scene(
        scene_id="t1_013",
        description="Mall plaza: shopper carrying a handbag outside the department store.",
        detection=SyntheticDetection("handbag", 0.91, (150.0, 220.0, 300.0, 380.0)),
        expected_label="handbag",
    ),
    Task1Scene(
        scene_id="t1_014",
        description="Airport terminal: traveler (person) with a suitcase near the check-in desk.",
        detection=SyntheticDetection("person", 0.87, (380.0, 200.0, 440.0, 380.0)),
        expected_label="person",
    ),
    Task1Scene(
        scene_id="t1_015",
        description="Rainy curb: a large umbrella leaning against a stone bench.",
        detection=SyntheticDetection("umbrella", 0.84, (330.0, 230.0, 430.0, 410.0)),
        expected_label="umbrella",
    ),
    Task1Scene(
        scene_id="t1_016",
        description="Office desk: a modern laptop open on a clean wooden surface.",
        detection=SyntheticDetection("laptop", 0.95, (200.0, 250.0, 400.0, 380.0)),
        expected_label="laptop",
    ),
    Task1Scene(
        scene_id="t1_017",
        description="Office reception: a guest (person) waiting in the lounge area.",
        detection=SyntheticDetection("person", 0.82, (150.0, 150.0, 250.0, 400.0)),
        expected_label="person",
    ),
    Task1Scene(
        scene_id="t1_018",
        description="Hospital hallway: a hospital chair (bench) placed near the nurse's station.",
        detection=SyntheticDetection("bench", 0.76, (240.0, 180.0, 460.0, 420.0)),
        expected_label="bench",
    ),
    Task1Scene(
        scene_id="t1_019",
        description="Office lounge: a comfortable couch under a designer lamp.",
        detection=SyntheticDetection("couch", 0.89, (100.0, 250.0, 600.0, 420.0)),
        expected_label="couch",
    ),
    Task1Scene(
        scene_id="t1_020",
        description="Conference room: one potted plant in the corner near the window.",
        detection=SyntheticDetection("potted plant", 0.97, (350.0, 280.0, 500.0, 450.0)),
        expected_label="potted plant",
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
        description="Urban intersection: a car, a motorcycle, and a fire hydrant near the curb.",
        detections=(
            SyntheticDetection("car", 0.92, (100.0, 200.0, 500.0, 400.0)),
            SyntheticDetection("motorcycle", 0.88, (150.0, 250.0, 300.0, 400.0)),
            SyntheticDetection("fire hydrant", 0.65, (550.0, 360.0, 590.0, 430.0)),
        ),
        expected_priority=("car", "motorcycle", "fire hydrant"),
    ),
    Task2Scene(
        scene_id="t2_002",
        description="Warehouse dock: a worker (person), a forklift (truck), and a portable scanner (handbag).",
        detections=(
            SyntheticDetection("truck", 0.87, (100.0, 150.0, 500.0, 400.0)),
            SyntheticDetection("person", 0.93, (420.0, 180.0, 500.0, 380.0)),
            SyntheticDetection("handbag", 0.77, (320.0, 250.0, 350.0, 300.0)),
        ),
        expected_priority=("person", "truck", "handbag"),
    ),
    Task2Scene(
        scene_id="t2_003",
        description="Bike lane: cyclist (person), parked car, suitcase left on sidewalk.",
        detections=(
            SyntheticDetection("person", 0.82, (150.0, 200.0, 400.0, 420.0)),
            SyntheticDetection("car", 0.79, (450.0, 230.0, 700.0, 360.0)),
            SyntheticDetection("suitcase", 0.70, (520.0, 300.0, 600.0, 380.0)),
        ),
        expected_priority=("person", "car", "suitcase"),
    ),
    Task2Scene(
        scene_id="t2_004",
        description="Construction: loader (truck), foreman (person), and a stop sign.",
        detections=(
            SyntheticDetection("person", 0.88, (300.0, 120.0, 380.0, 400.0)),
            SyntheticDetection("truck", 0.72, (50.0, 180.0, 280.0, 380.0)),
            SyntheticDetection("stop sign", 0.58, (620.0, 80.0, 660.0, 130.0)),
        ),
        expected_priority=("person", "truck", "stop sign"),
    ),
    Task2Scene(
        scene_id="t2_005",
        description="Campus: shuttle bus, student, and a bench.",
        detections=(
            SyntheticDetection("person", 0.72, (100.0, 90.0, 220.0, 420.0)),
            SyntheticDetection("bus", 0.89, (380.0, 240.0, 560.0, 400.0)),
            SyntheticDetection("bench", 0.65, (250.0, 50.0, 340.0, 140.0)),
        ),
        expected_priority=("bus", "person", "bench"),
    ),
    Task2Scene(
        scene_id="t2_006",
        description="Office lobby: security guard (person), visitor, and a potted plant.",
        detections=(
            SyntheticDetection("person", 0.65, (120.0, 220.0, 420.0, 360.0)),
            SyntheticDetection("person", 0.89, (40.0, 150.0, 640.0, 360.0)),
            SyntheticDetection("potted plant", 0.77, (560.0, 180.0, 620.0, 360.0)),
        ),
        expected_priority=("person", "person", "potted plant"),
    ),
    Task2Scene(
        scene_id="t2_007",
        description="Highway: motorcycle, car, and a bird in the sky.",
        detections=(
            SyntheticDetection("car", 0.78, (140.0, 220.0, 520.0, 360.0)),
            SyntheticDetection("motorcycle", 0.78, (260.0, 140.0, 340.0, 340.0)),
            SyntheticDetection("bird", 0.62, (380.0, 260.0, 430.0, 330.0)),
        ),
        expected_priority=("car", "motorcycle", "bird"),
    ),
    Task2Scene(
        scene_id="t2_008",
        description="Night street: car, pedestrian (person), and a fire hydrant.",
        detections=(
            SyntheticDetection("person", 0.79, (220.0, 120.0, 320.0, 420.0)),
            SyntheticDetection("fire hydrant", 0.80, (620.0, 40.0, 660.0, 120.0)),
            SyntheticDetection("car", 0.76, (40.0, 150.0, 640.0, 360.0)),
        ),
        expected_priority=("fire hydrant", "person", "car"),
    ),
    Task2Scene(
        scene_id="t2_009",
        description="Airport: traveler (person), suitcase, and a shuttle bus.",
        detections=(
            SyntheticDetection("person", 0.88, (150.0, 120.0, 230.0, 420.0)),
            SyntheticDetection("bus", 0.84, (260.0, 80.0, 740.0, 360.0)),
            SyntheticDetection("suitcase", 0.71, (320.0, 260.0, 380.0, 360.0)),
        ),
        expected_priority=("person", "bus", "suitcase"),
    ),
    Task2Scene(
        scene_id="t2_010",
        description="Shopping mall: shopper, chair, suitcase, and a potted plant.",
        detections=(
            SyntheticDetection("person", 0.92, (300.0, 140.0, 380.0, 420.0)),
            SyntheticDetection("suitcase", 0.86, (420.0, 160.0, 500.0, 430.0)),
            SyntheticDetection("chair", 0.80, (120.0, 220.0, 420.0, 380.0)),
            SyntheticDetection("potted plant", 0.60, (80.0, 260.0, 140.0, 360.0)),
        ),
        expected_priority=("person", "suitcase", "chair", "potted plant"),
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
        description="Thick brush: a low-lying heavy shape moving behind leaves; potential bear.",
        primary_detection=SyntheticDetection("bear", 0.28, (400.0, 350.0, 460.0, 410.0)),
        expected_action="discard",
        notes="0.28 < 0.35 discard.",
    ),
    Task3Scene(
        scene_id="t3_002",
        description="Taxiway fog: extremely weak signal near a hanger wing; identity unknown.",
        primary_detection=SyntheticDetection("airplane", 0.19, (400.0, 120.0, 520.0, 260.0)),
        expected_action="discard",
        notes="0.19 < 0.35 discard.",
    ),
    Task3Scene(
        scene_id="t3_003",
        description="Heavy rain: a blurry contour by the dock; truck or stacked pallets.",
        primary_detection=SyntheticDetection("truck", 0.38, (50.0, 120.0, 420.0, 380.0)),
        expected_action="request_rescan",
        notes="0.38 in 0.35-0.50 band.",
    ),
    Task3Scene(
        scene_id="t3_004",
        description="School fence: a low shape partially visible behind the chain-link; potential person.",
        primary_detection=SyntheticDetection("person", 0.38, (480.0, 320.0, 520.0, 440.0)),
        expected_action="request_rescan",
        notes="0.38 in 0.35-0.50 band.",
    ),
    Task3Scene(
        scene_id="t3_005",
        description="Night IR: warm blob on the roadway; motorcycle or debris.",
        primary_detection=SyntheticDetection("motorcycle", 0.42, (180.0, 220.0, 420.0, 400.0)),
        expected_action="request_rescan",
        notes="0.42 in 0.35-0.50 band.",
    ),
    Task3Scene(
        scene_id="t3_006",
        description="Dust cloud: a vague moving shape — worker (person) or shifted pile.",
        primary_detection=SyntheticDetection("person", 0.44, (320.0, 180.0, 370.0, 360.0)),
        expected_action="request_rescan",
        notes="0.44 in 0.35-0.50 band.",
    ),
    Task3Scene(
        scene_id="t3_007",
        description="Glass canopy glare: a blurred figure (person) near the sliding doors.",
        primary_detection=SyntheticDetection("person", 0.51, (290.0, 160.0, 350.0, 380.0)),
        expected_action="log_and_continue",
        notes="0.51 >= 0.50.",
    ),
    Task3Scene(
        scene_id="t3_008",
        description="Office lobby: a large soft shape against frosted glass; likely a chair.",
        primary_detection=SyntheticDetection("chair", 0.54, (280.0, 220.0, 450.0, 380.0)),
        expected_action="log_and_continue",
        notes="0.54 >= 0.50.",
    ),
    Task3Scene(
        scene_id="t3_009",
        description="Night rain: faint object near the curb house; likely a person.",
        primary_detection=SyntheticDetection("person", 0.21, (420.0, 260.0, 470.0, 360.0)),
        expected_action="discard",
        notes="0.21 < 0.35 discard.",
    ),
    Task3Scene(
        scene_id="t3_010",
        description="Dense fog: possible bicycle silhouette near the station gate.",
        primary_detection=SyntheticDetection("bicycle", 0.15, (330.0, 260.0, 410.0, 340.0)),
        expected_action="discard",
        notes="0.15 < 0.35 discard.",
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
            description="Urban street: one pedestrian (person) standing on the sidewalk near a storefront.",
            detection=SyntheticDetection("person", 0.91, (140.0, 90.0, 480.0, 440.0)),
            expected_label="person",
        ),
        task2=Task2Scene(
            scene_id="t2_bnd_urban",
            description="Urban intersection: a car, a bicycle, and a pedestrian waiting at the curb.",
            detections=(
                SyntheticDetection("car", 0.92, (100.0, 200.0, 500.0, 400.0)),
                SyntheticDetection("bicycle", 0.88, (150.0, 250.0, 300.0, 400.0)),
                SyntheticDetection("person", 0.85, (400.0, 150.0, 450.0, 420.0)),
            ),
            expected_priority=("car", "bicycle", "person"),
        ),
        task3=Task3Scene(
            scene_id="t3_bnd_urban",
            description="Urban doorway: a faint ambiguous shape in a dark recess; could be debris or a person.",
            primary_detection=SyntheticDetection("person", 0.24, (280.0, 160.0, 320.0, 360.0)),
            expected_action="discard",
            notes="0.24 < 0.35 threshold.",
        ),
    ),
    EpisodeBundle(
        bundle_id="bnd_warehouse",
        name="Warehouse",
        task1=Task1Scene(
            scene_id="t1_bnd_warehouse",
            description="Warehouse aisle: a forklift (truck) maneuvered by a worker.",
            detection=SyntheticDetection("truck", 0.88, (90.0, 120.0, 620.0, 400.0)),
            expected_label="truck",
        ),
        task2=Task2Scene(
            scene_id="t2_bnd_warehouse",
            description="Loading bay: a worker near a crate (suitcase), and a handheld pallet jack (truck).",
            detections=(
                SyntheticDetection("person", 0.88, (300.0, 130.0, 380.0, 410.0)),
                SyntheticDetection("truck", 0.87, (80.0, 150.0, 520.0, 390.0)),
                SyntheticDetection("suitcase", 0.75, (420.0, 300.0, 500.0, 360.0)),
            ),
            expected_priority=("person", "truck", "suitcase"),
        ),
        task3=Task3Scene(
            scene_id="t3_bnd_warehouse",
            description="Dusty corridor: an uncertain blob at the far end near stacked boxes.",
            primary_detection=SyntheticDetection("handbag", 0.42, (310.0, 140.0, 350.0, 340.0)),
            expected_action="request_rescan",
            notes="0.42 is in 0.35-0.50 band.",
        ),
    ),
    EpisodeBundle(
        bundle_id="bnd_parking",
        name="Parking Lot",
        task1=Task1Scene(
            scene_id="t1_bnd_parking",
            description="Parking lot: one car parked diagonally across two stalls.",
            detection=SyntheticDetection("car", 0.88, (120.0, 150.0, 580.0, 400.0)),
            expected_label="car",
        ),
        task2=Task2Scene(
            scene_id="t2_bnd_parking",
            description="Parking lane: car entering, pedestrian walking, and a parking meter.",
            detections=(
                SyntheticDetection("car", 0.90, (200.0, 200.0, 450.0, 420.0)),
                SyntheticDetection("person", 0.88, (500.0, 150.0, 550.0, 430.0)),
                SyntheticDetection("parking meter", 0.76, (150.0, 280.0, 200.0, 350.0)),
            ),
            expected_priority=("car", "person", "parking meter"),
        ),
        task3=Task3Scene(
            scene_id="t3_bnd_parking",
            description="Behind concrete pillar: a movement — potentially a stray dog.",
            primary_detection=SyntheticDetection("dog", 0.65, (450.0, 350.0, 520.0, 420.0)),
            expected_action="log_and_continue",
            notes="0.65 is >= 0.50.",
        ),
    ),
    EpisodeBundle(
        bundle_id="bnd_school",
        name="School Zone",
        task1=Task1Scene(
            scene_id="t1_bnd_school",
            description="School zone: a school bus parked in the loading lane.",
            detection=SyntheticDetection("bus", 0.96, (50.0, 80.0, 650.0, 420.0)),
            expected_label="bus",
        ),
        task2=Task2Scene(
            scene_id="t2_bnd_school",
            description="Sidewalk: student with a backpack, a crossing guard, and a lost dog.",
            detections=(
                SyntheticDetection("person", 0.94, (300.0, 120.0, 380.0, 440.0)),
                SyntheticDetection("backpack", 0.82, (150.0, 280.0, 320.0, 420.0)),
                SyntheticDetection("dog", 0.72, (100.0, 350.0, 180.0, 420.0)),
            ),
            expected_priority=("person", "backpack", "dog"),
        ),
        task3=Task3Scene(
            scene_id="t3_bnd_school",
            description="Near the heavy playground equipment: a low shape behind the fence — potentially a person.",
            primary_detection=SyntheticDetection("person", 0.38, (480.0, 320.0, 520.0, 440.0)),
            expected_action="request_rescan",
            notes="0.38 in 0.35-0.50 band.",
        ),
    ),
    EpisodeBundle(
        bundle_id="bnd_airport",
        name="Airport",
        task1=Task1Scene(
            scene_id="t1_bnd_airport",
            description="Airport ramp: one commercial jet (airplane) parked at the gate.",
            detection=SyntheticDetection("airplane", 0.95, (120.0, 80.0, 680.0, 300.0)),
            expected_label="airplane",
        ),
        task2=Task2Scene(
            scene_id="t2_bnd_airport",
            description="Terminal road: a shuttle bus, a traveler (person), and a luggage suitcase.",
            detections=(
                SyntheticDetection("bus", 0.91, (40.0, 60.0, 720.0, 280.0)),
                SyntheticDetection("person", 0.88, (380.0, 200.0, 440.0, 380.0)),
                SyntheticDetection("suitcase", 0.82, (180.0, 210.0, 520.0, 360.0)),
            ),
            expected_priority=("bus", "person", "suitcase"),
        ),
        task3=Task3Scene(
            scene_id="t3_bnd_airport",
            description="Taxiway fog: extremely weak signal near the hangar wing; identity unknown.",
            primary_detection=SyntheticDetection("airplane", 0.19, (400.0, 120.0, 520.0, 260.0)),
            expected_action="discard",
            notes="0.19 < 0.35.",
        ),
    ),
    EpisodeBundle(
        bundle_id="bnd_hospital",
        name="Hospital Entrance",
        task1=Task1Scene(
            scene_id="t1_bnd_hospital",
            description="Ambulance bay: an ambulance (truck) with rear doors open.",
            detection=SyntheticDetection("truck", 0.92, (100.0, 120.0, 620.0, 410.0)),
            expected_label="truck",
        ),
        task2=Task2Scene(
            scene_id="t2_bnd_hospital",
            description="Drop-off: a staff member (person), a wheelchair (bench class) on the ramp, and a car.",
            detections=(
                SyntheticDetection("person", 0.90, (480.0, 150.0, 540.0, 420.0)),
                SyntheticDetection("car", 0.88, (100.0, 220.0, 450.0, 380.0)),
                SyntheticDetection("bench", 0.74, (550.0, 340.0, 650.0, 420.0)),
            ),
            expected_priority=("person", "car", "bench"),
        ),
        task3=Task3Scene(
            scene_id="t3_bnd_hospital",
            description="Glass canopy glare: a blurred figure (person) near the sliding doors.",
            primary_detection=SyntheticDetection("person", 0.51, (290.0, 160.0, 350.0, 380.0)),
            expected_action="log_and_continue",
            notes="0.51 >= 0.50.",
        ),
    ),
    EpisodeBundle(
        bundle_id="bnd_construction",
        name="Construction Site",
        task1=Task1Scene(
            scene_id="t1_bnd_construction",
            description="Active site: a large dump truck in the foreground, tracks in dirt.",
            detection=SyntheticDetection("truck", 0.86, (100.0, 120.0, 640.0, 400.0)),
            expected_label="truck",
        ),
        task2=Task2Scene(
            scene_id="t2_bnd_construction",
            description="Excavation area: a foreman (person), a loader (truck), and a stop sign.",
            detections=(
                SyntheticDetection("person", 0.88, (280.0, 130.0, 360.0, 400.0)),
                SyntheticDetection("truck", 0.86, (40.0, 170.0, 420.0, 380.0)),
                SyntheticDetection("stop sign", 0.75, (600.0, 80.0, 650.0, 150.0)),
            ),
            expected_priority=("person", "truck", "stop sign"),
        ),
        task3=Task3Scene(
            scene_id="t3_bnd_construction",
            description="Dust plume: a vague moving shape — worker (person) or shifted pile.",
            primary_detection=SyntheticDetection("person", 0.44, (320.0, 180.0, 370.0, 360.0)),
            expected_action="request_rescan",
            notes="0.44 in 0.35-0.50 band.",
        ),
    ),
    EpisodeBundle(
        bundle_id="bnd_night",
        name="Night Street",
        task1=Task1Scene(
            scene_id="t1_bnd_night",
            description="Night street: a rider on a motorcycle under high-pressure sodium lights.",
            detection=SyntheticDetection("motorcycle", 0.82, (200.0, 220.0, 450.0, 410.0)),
            expected_label="motorcycle",
        ),
        task2=Task2Scene(
            scene_id="t2_bnd_night",
            description="Street corner: a car with headlights, a pedestrian (person), and a fire hydrant.",
            detections=(
                SyntheticDetection("car", 0.87, (100.0, 200.0, 450.0, 380.0)),
                SyntheticDetection("person", 0.84, (480.0, 180.0, 520.0, 430.0)),
                SyntheticDetection("fire hydrant", 0.62, (550.0, 360.0, 590.0, 430.0)),
            ),
            expected_priority=("car", "person", "fire hydrant"),
        ),
        task3=Task3Scene(
            scene_id="t3_bnd_night",
            description="Shadowy alley: sensor returns a faint infrared flicker; stationary dog or debris.",
            primary_detection=SyntheticDetection("dog", 0.21, (350.0, 240.0, 410.0, 360.0)),
            expected_action="discard",
            notes="0.21 < 0.35 discard.",
        ),
    ),
    EpisodeBundle(
        bundle_id="bnd_forest",
        name="Forest Trail",
        task1=Task1Scene(
            scene_id="t1_bnd_forest",
            description="Woodland path: a hiker with a backpack, clear morning light.",
            detection=SyntheticDetection("backpack", 0.88, (300.0, 250.0, 450.0, 400.0)),
            expected_label="backpack",
        ),
        task2=Task2Scene(
            scene_id="t2_bnd_forest",
            description="Trail junction: a dog on a leash, a bird on a branch, and a hiker far ahead.",
            detections=(
                SyntheticDetection("person", 0.89, (200.0, 150.0, 280.0, 420.0)),
                SyntheticDetection("dog", 0.82, (250.0, 320.0, 350.0, 410.0)),
                SyntheticDetection("bird", 0.75, (500.0, 50.0, 540.0, 100.0)),
            ),
            expected_priority=("person", "dog", "bird"),
        ),
        task3=Task3Scene(
            scene_id="t3_bnd_forest",
            description="Thick brush: a low-lying heavy shape moving behind the leaves; potentially a bear.",
            primary_detection=SyntheticDetection("bear", 0.28, (400.0, 350.0, 460.0, 410.0)),
            expected_action="discard",
            notes="0.28 < 0.35 discard.",
        ),
    ),
    EpisodeBundle(
        bundle_id="bnd_mall",
        name="Shopping Mall",
        task1=Task1Scene(
            scene_id="t1_bnd_mall",
            description="Mall escalator entrance: shopper carrying a handbag.",
            detection=SyntheticDetection("handbag", 0.92, (150.0, 220.0, 300.0, 380.0)),
            expected_label="handbag",
        ),
        task2=Task2Scene(
            scene_id="t2_bnd_mall",
            description="Food court: traveler with suitcase, person sitting, and an empty chair.",
            detections=(
                SyntheticDetection("person", 0.91, (300.0, 100.0, 380.0, 450.0)),
                SyntheticDetection("suitcase", 0.84, (400.0, 350.0, 500.0, 450.0)),
                SyntheticDetection("chair", 0.72, (200.0, 300.0, 280.0, 380.0)),
            ),
            expected_priority=("person", "suitcase", "chair"),
        ),
        task3=Task3Scene(
            scene_id="t3_bnd_mall",
            description="Restroom corridor: an unattended umbrella near the rest area; details are blurry.",
            primary_detection=SyntheticDetection("umbrella", 0.46, (380.0, 250.0, 440.0, 380.0)),
            expected_action="request_rescan",
            notes="0.46 in 0.35-0.50 band.",
        ),
    ),
    EpisodeBundle(
        bundle_id="bnd_office",
        name="Office Lobby",
        task1=Task1Scene(
            scene_id="t1_bnd_office",
            description="Tech office reception: a modern laptop open on the front desk.",
            detection=SyntheticDetection("laptop", 0.96, (200.0, 250.0, 400.0, 380.0)),
            expected_label="laptop",
        ),
        task2=Task2Scene(
            scene_id="t2_bnd_office",
            description="Lounge area: employee on a couch, a guest sitting, and a potted plant.",
            detections=(
                SyntheticDetection("person", 0.93, (150.0, 150.0, 250.0, 400.0)),
                SyntheticDetection("couch", 0.89, (100.0, 250.0, 600.0, 420.0)),
                SyntheticDetection("potted plant", 0.77, (350.0, 280.0, 500.0, 450.0)),
            ),
            expected_priority=("person", "couch", "potted plant"),
        ),
        task3=Task3Scene(
            scene_id="t3_bnd_office",
            description="Conference room window: a large soft shape against the frosted glass — likely a chair.",
            primary_detection=SyntheticDetection("chair", 0.54, (280.0, 220.0, 450.0, 380.0)),
            expected_action="log_and_continue",
            notes="0.54 >= 0.50.",
        ),
    ),
    EpisodeBundle(
        bundle_id="bnd_rainy",
        name="Rainy Street",
        task1=Task1Scene(
            scene_id="t1_bnd_rainy",
            description="Rainy curb: a person holding a large umbrella.",
            detection=SyntheticDetection("umbrella", 0.94, (200.0, 100.0, 450.0, 420.0)),
            expected_label="umbrella",
        ),
        task2=Task2Scene(
            scene_id="t2_bnd_rainy",
            description="Downtown curb: a car splashing, a bus approaching, and a pedestrian with umbrella.",
            detections=(
                SyntheticDetection("car", 0.88, (100.0, 220.0, 500.0, 400.0)),
                SyntheticDetection("bus", 0.88, (50.0, 50.0, 650.0, 380.0)),
                SyntheticDetection("umbrella", 0.86, (400.0, 150.0, 520.0, 280.0)),
            ),
            expected_priority=("bus", "car", "umbrella"),
        ),
        task3=Task3Scene(
            scene_id="t3_bnd_rainy",
            description="Heavy rain: a blurry contour by the dock; could be a truck or stacked pallets.",
            primary_detection=SyntheticDetection("truck", 0.38, (50.0, 120.0, 420.0, 380.0)),
            expected_action="request_rescan",
            notes="0.38 in 0.35-0.50 band.",
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
