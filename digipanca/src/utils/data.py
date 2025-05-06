import os
import json

def get_patients_in_processed_folder(data_dir):

    metadata_path = os.path.join(data_dir, "metadata.json")
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    patient_ids = set([])
    for k, v in metadata.items():
        patient_ids.add(v["patient_id"])

    return list(patient_ids)
