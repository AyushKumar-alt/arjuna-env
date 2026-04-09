import re

with open('server/synthetic_data.py', 'r') as f:
    text = f.read()

blizzard_and_glare_bundles = """    EpisodeBundle(
        bundle_id="bnd_blizzard",
        name="Blizzard Whiteout",
        task1=Task1Scene(
            scene_id="t1_bnd_blizzard",
            description="Winter road: a snowplow clear through driving snow.",
            detection=SyntheticDetection("truck", 0.94, (100.0, 100.0, 600.0, 400.0)),
            expected_label="truck",
        ),
        task2=Task2Scene(
            scene_id="t2_bnd_blizzard",
            description="Frozen intersection: a pedestrian bracing against the wind, a snow-covered car, and a stop sign.",
            detections=(
                SyntheticDetection("person", 0.89, (350.0, 150.0, 450.0, 400.0)),
                SyntheticDetection("car", 0.82, (100.0, 200.0, 500.0, 400.0)),
                SyntheticDetection("stop sign", 0.70, (600.0, 100.0, 650.0, 150.0)),
            ),
            expected_priority=("person", "car", "stop sign"),
        ),
        task3=Task3Scene(
            scene_id="t3_bnd_blizzard",
            description="Complete whiteout: an extremely faint blob barely registers against the snow; potential bus.",
            primary_detection=SyntheticDetection("bus", 0.22, (200.0, 150.0, 550.0, 400.0)),
            expected_action="discard",
            notes="0.22 < 0.35 discard due to blizzard.",
        ),
    ),
    EpisodeBundle(
        bundle_id="bnd_glare",
        name="Sensor Glare",
        task1=Task1Scene(
            scene_id="t1_bnd_glare",
            description="Highway sunset: a motorcycle traveling fast in the clear left lane.",
            detection=SyntheticDetection("motorcycle", 0.95, (250.0, 200.0, 450.0, 420.0)),
            expected_label="motorcycle",
        ),
        task2=Task2Scene(
            scene_id="t2_bnd_glare",
            description="Glare zone: an ambulance with sirens, a pedestrian, and a traffic light blinded by sun.",
            detections=(
                SyntheticDetection("ambulance", 0.91, (120.0, 150.0, 450.0, 420.0)),
                SyntheticDetection("person", 0.87, (500.0, 180.0, 560.0, 400.0)),
                SyntheticDetection("traffic light", 0.65, (450.0, 50.0, 490.0, 120.0)),
            ),
            expected_priority=("ambulance", "person", "traffic light"),
        ),
        task3=Task3Scene(
            scene_id="t3_bnd_glare",
            description="Adversarial glare: camera lens flare creates an ambiguous phantom shape on the road.",
            primary_detection=SyntheticDetection("car", 0.46, (380.0, 250.0, 440.0, 380.0)),
            expected_action="request_rescan",
            notes="0.46 in 0.35-0.50 band.",
        ),
    ),
)
"""

text = re.sub(r'    \),\n\)\n', blizzard_and_glare_bundles, text, count=1)

with open('server/synthetic_data.py', 'w') as f:
    f.write(text)

print("Added Blizzard and Glare bundles completely safely.")
