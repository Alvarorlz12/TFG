import os
import argparse
import shutil
import json

from src.data.split_data import create_train_test_split
from src.data.preprocessing import process_patient
from src.utils.config import load_config

def clear_directory(directory):
    """Clear the contents of a directory."""
    if os.path.exists(directory):
        print(f"🧹 Cleaning {directory}...")
        shutil.rmtree(directory)
    os.makedirs(directory, exist_ok=True)
    print(f"✅ {directory} is clean.")

def generate_split(config_path=None):
    """
    Generate the train-test split.
    
    Parameters
    ----------
    config_path : str, optional
        Path to the configuration file. If not specified, the default
        configuration file is used.
    """
    config = load_config(config_path) if config_path else load_config()
    RAW_DATA_DIR = config["data"]["raw_dir"]
    TEST_SIZE = config["data"]["test_split"]
    VAL_SIZE = config["data"]["val_split"]
    SHUFFLE = config["data"]["shuffle"]
    FILENAME = os.path.basename(config["data"]["split_file"])

    sample_dirs = [
        os.path.join(RAW_DATA_DIR, d) for d in os.listdir(RAW_DATA_DIR)
    ]
    
    print("🔄 Creating data splits...")
    create_train_test_split(
        sample_dirs=sample_dirs,
        test_size=TEST_SIZE,
        val_size=VAL_SIZE,
        random_state=42,
        shuffle=SHUFFLE,
        filename=FILENAME
    )
    print("✅ Data splits created.")

def preprocess_data(config_path='configs/data/preprocess.yaml'):
    """
    Preprocess the data.
    
    Parameters
    ----------
    config_path : str, optional
        Path to the configuration file. If not specified, the default
        configuration file is used.
    """
    config = load_config(config_path)
    RAW_DATA_DIR = config["data"]["raw_dir"]
    PROCESSED_DATA_DIR = config["data"]["processed_dir"]
    SPLIT_FILE = config["data"]["split_path"]
    SUBVOLUME_SIZE = config["data"]["subvolume_size"]
    SUBVOLUME_STRIDE = config["data"]["subvolume_stride"]
    TARGET_ORIENTATION = tuple(config["data"]["target_orientation"])
    H_MAX = config["data"]["roi"].get("h_max", 512)
    W_MAX = config["data"]["roi"].get("w_max", 512)
    H_MIN = config["data"]["roi"].get("h_min", 0)
    W_MIN = config["data"]["roi"].get("w_min", 0)

    # Clear the processed data directory
    clear_directory(PROCESSED_DATA_DIR)

    # Cargar JSON de splits
    with open(SPLIT_FILE, "r") as f:
        split_dict = json.load(f)

    for split_type, patients in split_dict.items():
        if len(patients) == 0:
            continue
        output_dir = os.path.join(PROCESSED_DATA_DIR, split_type)
        os.makedirs(output_dir, exist_ok=True)

        print(f"🔄 Preprocessing {split_type} data... {len(patients)} patients found.")
        for patient_id in patients:
            patient_dir = os.path.join(RAW_DATA_DIR, patient_id)
            num_subvolumes = process_patient(
                patient_dir=patient_dir,
                output_dir=output_dir,
                subvolume_size=SUBVOLUME_SIZE,
                subvolume_stride=SUBVOLUME_STRIDE,
                target_orientation=TARGET_ORIENTATION,
                h_min=H_MIN,
                h_max=H_MAX,
                w_min=W_MIN,
                w_max=W_MAX
            )
            print(f"✅ {patient_id}: {num_subvolumes} sub-volumes saved.")
        print(f"✅ {split_type} completed\n")


def main():
    parser = argparse.ArgumentParser(description="Data splitting and preprocessing script.")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["split", "preprocess"],
        required=True,
        help="Mode of operation: split or preprocess."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to the configuration file (optional)."
    )

    args = parser.parse_args()

    if args.mode == "split":
        generate_split() if args.config is None else generate_split(args.config)
    elif args.mode == "preprocess":
        preprocess_data() if args.config is None else preprocess_data(args.config)

if __name__ == "__main__":
    main()