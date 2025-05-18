import glob
import os
import argparse
import shutil
import json
import monai

from sklearn.model_selection import train_test_split

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
    TARGET_SPACING = config["data"].get("target_spacing", None)
    if TARGET_SPACING is not None:
        TARGET_SPACING = tuple(TARGET_SPACING)
    TARGET_ORIENTATION = tuple(config["data"]["target_orientation"])
    H_MAX = config["data"]["roi"].get("h_max", 512)
    W_MAX = config["data"]["roi"].get("w_max", 512)
    H_MIN = config["data"]["roi"].get("h_min", 0)
    W_MIN = config["data"]["roi"].get("w_min", 0)
    if not IS_2D:   # 3D parameters
        SUBVOLUME_SIZE = config["data"]["subvolume_size"]
        SUBVOLUME_STRIDE = config["data"]["subvolume_stride"]


    # Load all patient from the raw data directory
    sample_dirs = [
        os.path.join(RAW_DATA_DIR, d) for d in os.listdir(RAW_DATA_DIR)
    ]

    output_folder = os.path.basename(RAW_DATA_DIR)
    output_dir = os.path.join(PROCESSED_DATA_DIR, output_folder)
    # Clear the processed data directory
    clear_directory(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    metadata = {}
    
    print(f"🔄 Preprocessing {output_folder} data... {len(sample_dirs)} patients found.")
    for i, patient_id in enumerate(sample_dirs):

        patient_dir = os.path.normpath(os.path.join(RAW_DATA_DIR, patient_id))

        if IS_2D:
            # Process 2D data
            num_slices, patient_metadata = process_patient_2d(
                patient_dir=patient_dir,
                output_dir=output_dir,
                target_spacing=TARGET_SPACING,
                target_orientation=TARGET_ORIENTATION,
                h_min=H_MIN,
                h_max=H_MAX,
                w_min=W_MIN,
                w_max=W_MAX
            )
            # Update metadata with patient information
            metadata.update(patient_metadata)
            print(f"✅ ({i+1}/{len(sample_dirs)}) {os.path.basename(patient_id)}: {num_slices} slices saved.")
        else:
            # Process 3D data
            num_subvolumes, patient_metadata = process_patient_3d(
                patient_dir=patient_dir,
                output_dir=output_dir,
                target_spacing=TARGET_SPACING,
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
            print(f"✅ ({i+1}/{len(sample_dirs)}) {os.path.basename(patient_id)}: {num_subvolumes} sub-volumes saved.")

    # Save metadata for the split
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"✅ {output_folder} completed\n")

def generate_datalist(data_dir, output, val_split=0.2):
    """
    Generate a datalist for the dataset.
    
    Parameters
    ----------
    data_dir : str
        Path to the data directory. It must contain the following 
        subdirectories: labelsTr, imagesTr, labelsTs, imagesTs.
    output : str
        Path to the output file.
    """
    # Check if the data directory exists
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory {data_dir} does not exist.")
    
    def produce_sample_dict(line):
        return {"label": line, "image": line.replace("label", "image")}

    monai.utils.set_determinism(seed=123)

    # Create the datalist
    datalist = []
    test_list = []

    samples = sorted(glob.glob(os.path.join(data_dir, "labelsTr", "*"), recursive=True))
    samples = [_item.replace(os.path.join(data_dir, "labelsTr"), "labelsTr") for _item in samples]
    for sample in samples:
        datalist.append(produce_sample_dict(sample))

    test_samples = sorted(glob.glob(os.path.join(data_dir, "labelsTs", "*"), recursive=True))
    test_samples = [_item.replace(os.path.join(data_dir, "labelsTs"), "labelsTs") for _item in test_samples]
    for sample in test_samples:
        test_list.append(produce_sample_dict(sample))

    train_list, val_list = train_test_split(
        datalist,
        test_size=val_split,
        random_state=42,
        shuffle=False
    )

    # Create the final datalist
    datalist = {
        "training": train_list,
        "validation": val_list,
        "test": test_list
    }

    # Save the datalist to a JSON file
    with open(output, "w") as f:
        json.dump(datalist, f, ensure_ascii=True, indent=4)


def main():
    parser = argparse.ArgumentParser(description="Data splitting and preprocessing script.")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["split", "preprocess", "datalist"],
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
    elif args.mode == "datalist":
        generate_datalist(
            data_dir=os.path.join("data", "prepared"),
            output=os.path.join("data/splits", "datalist.json")
        )

if __name__ == "__main__":
    main()