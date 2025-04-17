import os
import argparse
import shutil
import json

from src.data.split_data import create_train_test_split, create_kfold_split
from src.data.preprocessing import process_patient_3d, process_patient_2d
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
    SHUFFLE = config["data"]["shuffle"]
    FILENAME = os.path.basename(config["data"]["split_file"])
    IS_CV = config["data"].get("is_cv", False)

    sample_dirs = [
        os.path.join(RAW_DATA_DIR, d) for d in os.listdir(RAW_DATA_DIR)
    ]
    
    print("🔄 Creating data splits...")
    if IS_CV:
        N_SPLITS = config["data"].get("n_splits", 5)
        create_kfold_split(
            sample_dirs=sample_dirs,
            n_splits=N_SPLITS,
            random_state=42,
            shuffle=SHUFFLE,
            filename=FILENAME
        )
    else:
        TEST_SIZE = config["data"]["test_split"]
        VAL_SIZE = config["data"]["val_split"]
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
    IS_2D = config["data"].get("is_2d", False)
    # Common parameters
    RAW_DATA_DIR = config["data"]["raw_dir"]
    PROCESSED_DATA_DIR = config["data"]["processed_dir"]
    TARGET_ORIENTATION = tuple(config["data"]["target_orientation"])
    H_MAX = config["data"]["roi"].get("h_max", 512)
    W_MAX = config["data"]["roi"].get("w_max", 512)
    H_MIN = config["data"]["roi"].get("h_min", 0)
    W_MIN = config["data"]["roi"].get("w_min", 0)
    if not IS_2D:   # 3D parameters
        SUBVOLUME_SIZE = config["data"]["subvolume_size"]
        SUBVOLUME_STRIDE = config["data"]["subvolume_stride"]

    # Clear the processed data directory
    clear_directory(PROCESSED_DATA_DIR)

    # Load all patient from the raw data directory
    sample_dirs = [
        os.path.join(RAW_DATA_DIR, d) for d in os.listdir(RAW_DATA_DIR)
    ]

    output_folder = os.path.basename(RAW_DATA_DIR)
    output_dir = os.path.join(PROCESSED_DATA_DIR, output_folder)
    os.makedirs(output_dir, exist_ok=True)
    metadata = {}
    
    print(f"🔄 Preprocessing {output_folder} data... {len(sample_dirs)} patients found.")
    for patient_id in sample_dirs:

        patient_dir = os.path.normpath(os.path.join(RAW_DATA_DIR, patient_id))

        if IS_2D:
            # Process 2D data
            num_slices, patient_metadata = process_patient_2d(
                patient_dir=patient_dir,
                output_dir=output_dir,
                target_orientation=TARGET_ORIENTATION,
                h_min=H_MIN,
                h_max=H_MAX,
                w_min=W_MIN,
                w_max=W_MAX
            )
            # Update metadata with patient information
            metadata.update(patient_metadata)
            print(f"✅ {patient_id}: {num_slices} slices saved.")
        else:
            # Process 3D data
            num_subvolumes, patient_metadata = process_patient_3d(
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
            # Update metadata with patient information
            metadata.update(patient_metadata)
            print(f"✅ {patient_id}: {num_subvolumes} sub-volumes saved.")

    # Save metadata for the split
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"✅ {output_folder} completed\n")


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