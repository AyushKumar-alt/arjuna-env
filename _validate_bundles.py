from server.synthetic_data import EPISODE_BUNDLES, expected_low_confidence_action

for b in EPISODE_BUNDLES:
    c = b.task3.primary_detection.confidence
    exp = expected_low_confidence_action(c)
    status = "OK" if exp == b.task3.expected_action else "MISMATCH"
    print(f"{b.bundle_id:25} conf={c:.2f}  expected={b.task3.expected_action:17}  computed={exp:17}  {status}")

print("Done.")
