import csv
from server.synthetic_data import EPISODE_BUNDLES

def export_ood_dataset():
    output_file = "arjuna_ood_v1.csv"
    headers = [
        "bundle_id", 
        "bundle_name", 
        "task1_label", 
        "task1_description",
        "task2_objects", 
        "task2_description",
        "task3_confidence", 
        "task3_description",
        "task3_expected_action"
    ]

    with open(output_file, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)

        for b in EPISODE_BUNDLES:
            writer.writerow([
                b.bundle_id,
                b.name,
                b.task1.expected_label,
                b.task1.description,
                ", ".join([d.label for d in b.task2.detections]),
                b.task2.description,
                b.task3.primary_detection.confidence,
                b.task3.description,
                b.task3.expected_action
            ])
    
    print(f"Successfully exported {len(EPISODE_BUNDLES)} scenarios to {output_file}")

if __name__ == "__main__":
    export_ood_dataset()
